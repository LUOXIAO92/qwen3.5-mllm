from pathlib import Path

import numpy as np
from transformers import ProcessorMixin, Qwen3VLProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    _get_feat_extract_output_lengths,
)

from .audio_video_utils import get_audio_token_timestamps, merge_audio_video_order
from .media_io import load_audio_video


def _waveform_lengths(audio):
    """返回每段波形的采样点数，批次的写法与 WhisperFeatureExtractor 的判断一致。"""
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        return [audio.shape[-1]] * audio.shape[0]
    if isinstance(audio, (list, tuple)) and len(audio) > 0 and not np.isscalar(audio[0]):
        return [np.asarray(a = wave).shape[-1] for wave in audio]
    return [np.asarray(a = audio).shape[-1]]


class _Qwen3_5AudioVideoProcessor(Qwen3VLProcessor):
    """只扩展联合音视频的占位序列，复用原有图像和视频预处理。"""

    def replace_video_token(self, video_inputs, video_idx, **kwargs):
        grid = video_inputs["video_grid_thw"][video_idx]
        frame_length = int(grid[1] * grid[2]) // (self.video_processor.merge_size ** 2)

        # 第一步：取得两路未舍入的时间，单位均为秒。
        audio_times = self.audio_timestamps[video_idx]
        # 解码时间已经以秒为单位，除数取 1，复用原函数的帧组平均计算。
        video_times = self._calculate_timestamps(
            indices = list(self.video_timestamps[video_idx]),
            video_fps = 1.0,
            merge_size = self.video_processor.temporal_patch_size,
        )

        # 第二步：复用 Omni 的归并循环，按时间从早到晚排列。
        order = merge_audio_video_order(
            audio_times = audio_times,
            video_times = video_times,
        )

        # 第三步：按 Qwen3.5 格式写入时间文本，相邻同标签共用。
        timestamps = {"video": video_times, "audio": audio_times}
        time_labels = [
            f"<{timestamps[modality][index]:.1f} seconds>"
            for modality, index in order
        ]

        parts = [self.tokenizer.audio_bos_token]
        previous_label = None
        for (modality, index), label in zip(order, time_labels):
            if label != previous_label:
                parts.append(label)
                previous_label = label
            if modality == "video":
                parts.append(
                    self.vision_start_token
                    + self.video_token * frame_length
                    + self.vision_end_token,
                )
            else:
                parts.append(self.tokenizer.audio_token)
        parts.append(self.tokenizer.audio_eos_token)
        return "".join(parts)


class Qwen3_5MLLMProcessor(ProcessorMixin):
    """保留 Qwen3.5 的文本、图像和视频处理，并加入音频处理。"""

    def __init__(
        self,
        image_processor,
        tokenizer,
        video_processor,
        feature_extractor,
        chat_template = None,
        audio_n_window = 50,
    ):
        super().__init__(
            image_processor = image_processor,
            tokenizer = tokenizer,
            video_processor = video_processor,
            feature_extractor = feature_extractor,
            chat_template = chat_template,
        )
        self.audio_n_window = audio_n_window
        self.audio_token = tokenizer.audio_token
        self.audio_start_token = tokenizer.audio_bos_token
        self.audio_end_token = tokenizer.audio_eos_token

    @property
    def vision_processor(self):
        return Qwen3VLProcessor(
            image_processor = self.image_processor,
            tokenizer = self.tokenizer,
            video_processor = self.video_processor,
            chat_template = self.chat_template,
        )

    def __call__(
        self,
        text = None,
        images = None,
        videos = None,
        audio = None,
        sampling_rate = None,
        return_tensors = "pt",
        use_audio_in_video = False,
        **kwargs,
    ):
        if text is None and images is None and videos is None and audio is None:
            raise ValueError("请提供 text、images、videos 或 audio。")
        decoded_media = None
        if use_audio_in_video:
            if audio is not None:
                raise ValueError("联合排列只使用视频文件自带的音轨，请勿另传 audio。")
            if kwargs.get("video_metadata") is not None:
                raise ValueError("联合输入的帧信息从视频文件读取，请勿另传 video_metadata。")
            video_paths = [videos] if isinstance(videos, (str, Path)) else videos
            if not isinstance(video_paths, (list, tuple)) or not video_paths or not all(
                isinstance(path, (str, Path)) for path in video_paths
            ):
                raise ValueError("联合排列需要带音轨的视频文件路径，或每条输入一个文件的路径列表。")
            sampling_rate = self.feature_extractor.sampling_rate
            do_sample_frames = kwargs.pop(
                "do_sample_frames", self.video_processor.do_sample_frames,
            )
            decoded_media = [
                load_audio_video(
                    video_path = path,
                    video_processor = self.video_processor,
                    sampling_rate = sampling_rate,
                    fps = kwargs.get("fps"),
                    num_frames = kwargs.get("num_frames"),
                    do_sample_frames = do_sample_frames,
                )
                for path in video_paths
            ]
            videos = [item["video"] for item in decoded_media]
            audio = [item["audio"] for item in decoded_media]
            kwargs["video_metadata"] = [item["metadata"] for item in decoded_media]
            kwargs["do_sample_frames"] = False
            if text is None:
                text = ["<|vision_start|><|video_pad|><|vision_end|>"] * len(video_paths)

        audio_inputs = None
        if audio is not None:
            if sampling_rate is None:
                raise ValueError("传入 audio 时，请用 sampling_rate 指明波形的采样率。")
            audio_inputs = self.feature_extractor(
                raw_speech = audio,
                sampling_rate = sampling_rate,
                padding = "longest",
                truncation = False,
                return_attention_mask = True,
                return_tensors = return_tensors,
            )
            # WhisperFeatureExtractor 把采样点级的 mask 每隔 hop_length 取一个值，得到「采样点数 ÷ hop_length」
            # 向上取整的帧数，只在批次补齐后的总长度不能整除时去掉最后一列，因此只修正了批内最长的一条。
            # 同一段音频单独处理得到向下取整的帧数，成批处理时可能多 1 帧、多 1 个音频 token。
            # 这里按每段音频自己的采样点数向下取整，重写有效帧的 mask，单独处理与成批处理结果一致。
            hop_length = self.feature_extractor.hop_length
            feature_lens = np.asarray(a = [length // hop_length for length in _waveform_lengths(audio = audio)])
            feature_mask = audio_inputs["attention_mask"]
            feature_mask[:] = 0
            for index, length in enumerate(feature_lens):
                feature_mask[index, :length] = 1
            audio_lengths = _get_feat_extract_output_lengths(
                input_lengths = feature_lens,
                n_window = self.audio_n_window,
            ).tolist()
            if text is None:
                text = [""] * len(audio_lengths)
            elif isinstance(text, str):
                text = [text]
            if len(text) != len(audio_lengths):
                raise ValueError("当前每条文本输入对应一段音频。")
            audio_marker = (
                self.audio_start_token + self.audio_token + self.audio_end_token
            )
            expanded_text = []
            for prompt, audio_length in zip(text, audio_lengths):
                if use_audio_in_video:
                    if prompt.count("<|video_pad|>") != 1 or prompt.count(self.audio_token) > 1:
                        raise ValueError("每条联合输入需要一个视频标记和至多一个音频标记。")
                    prompt = prompt.replace(audio_marker, "")
                    expanded_text.append(prompt.replace(self.audio_token, ""))
                    continue
                if self.audio_token not in prompt:
                    prompt = audio_marker + prompt
                if prompt.count(self.audio_token) != 1:
                    raise ValueError("每条文本请保留一个 <|audio_pad|> 音频标记。")
                expanded_text.append(
                    prompt.replace(self.audio_token, self.audio_token * audio_length),
                )
            text = expanded_text

        inputs = BatchFeature(data = {})
        vision_processor = self.vision_processor
        if use_audio_in_video:
            kwargs["return_metadata"] = True
            vision_processor = _Qwen3_5AudioVideoProcessor(
                image_processor = self.image_processor,
                tokenizer = self.tokenizer,
                video_processor = self.video_processor,
                chat_template = self.chat_template,
            )
            vision_processor.audio_timestamps = []
            for length, item in zip(feature_lens, decoded_media):
                audio_times = get_audio_token_timestamps(
                    feature_length = length,
                    n_window = self.audio_n_window,
                    hop_length = self.feature_extractor.hop_length,
                    sampling_rate = self.feature_extractor.sampling_rate,
                )
                audio_times = [
                    timestamp + item["audio_start_seconds"] for timestamp in audio_times
                ]
                vision_processor.audio_timestamps.append(audio_times)
            vision_processor.video_timestamps = [
                item["video_timestamps"] for item in decoded_media
            ]
        if text is not None or images is not None or videos is not None:
            inputs = vision_processor(
                text = text,
                images = images,
                videos = videos,
                return_tensors = return_tensors,
                **kwargs,
            )

        if audio_inputs is not None:
            inputs["input_features"] = audio_inputs["input_features"]
            inputs["feature_attention_mask"] = audio_inputs["attention_mask"]
            audio_token_id = self.tokenizer.convert_tokens_to_ids(tokens = self.audio_token)
            token_types = []
            for ids, types, length in zip(
                inputs["input_ids"], inputs["mm_token_type_ids"], audio_lengths,
            ):
                audio_mask = np.asarray(a = ids) == audio_token_id
                if int(audio_mask.sum()) != length:
                    raise ValueError("分词后音频标记不完整，请保留整段音频标记。")
                row_types = np.asarray(a = types).copy()
                row_types[audio_mask] = 3
                token_types.append(row_types.tolist())
            inputs["mm_token_type_ids"] = token_types

        if use_audio_in_video:
            inputs.pop("video_metadata")

        if audio_inputs is not None:
            inputs = BatchFeature(data = inputs.data, tensor_type = return_tensors)

        return inputs

    @property
    def model_input_names(self):
        return self.vision_processor.model_input_names + [
            "input_features",
            "feature_attention_mask",
        ]
