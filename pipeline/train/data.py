"""训练清单读取、目标文本选择与批次构造。

一条样本是一段完整的对话：用户消息为音频加固定指令，助手回答为目标文本。整段对话用阶段 1 模型自带的
chat template 渲染，再交给 processor 做特征提取和音频标记展开，得到最终的 input_ids。标签由
mm_assistant_mask（https://github.com/LUOXIAO92/MultimodalAssistantMask）在最终 input_ids 上按 token
匹配助手回合生成：助手回合开头和空的思考块不计损失，目标文本和结尾的 <|im_end|> 计算损失，其余位置填 -100。
transformers 的 return_assistant_tokens_mask 按展开音频标记之前的字符位置计算，展开后会错位或全为 0，
所以不用它。模型内部再把标签左移一位，用第 t 个位置的输出预测第 t + 1 个 token。

训练目标为空字符串的样本（空目标样本，即无人声片段）照常渲染：助手回合只有空的思考块和 <|im_end|>，
按同样的规则只有 <|im_end|> 计算损失，模型学着听到无人声的音频时直接结束输出。

训练时数据集按 sampling.SampleKey 取样本，给出背景声混音器时对语音样本实时混音（mixing.py），
空目标样本不混音；按整数下标取样本时（dev 评估、标签核对）不混音。
"""

import json
import random
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
from mm_assistant_mask import AssistantMaskSpec, build_assistant_labels

# 训练时数据集的下标：清单行号、轮次、在本轮样本序列中的位置（见 sampling.py）。
SampleKey = namedtuple(typename = "SampleKey", field_names = ("index", "epoch", "position"))
# 随机数流的编号：每轮的抽样次数和混音各用一条，互不影响。
STREAM_OCCURRENCES = 1
STREAM_MIXING = 2
IGNORE_INDEX = -100
SAMPLING_RATE = 16000
INSTRUCTION = "Transcribe the audio."
TARGETS = ("script", "original")
# 助手回合在 token 序列中的边界。add_generation_prompt = True 生成的提示以「助手回合开头 + 空的思考块」结尾，
# 这两段属于提示，不计损失；从目标文本开始到 <|im_end|> 为止计算损失，<|im_end|> 之后的换行不计。
ASSISTANT_MASK_SPEC = AssistantMaskSpec(
    assistant_header = "<|im_start|>assistant\n",
    assistant_end = "<|im_end|>",
    generation_stop = "<|im_end|>",
    excluded_assistant_prefixes = ("<think>\n\n</think>\n\n",),
)


def build_messages(audio_path):
    """按 CONVENTIONS 的对话格式构造用户消息：一段音频加一句固定指令，不给语言提示。"""
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": str(audio_path)},
                {"type": "text", "text": INSTRUCTION},
            ],
        },
    ]


def build_prompt(processor, audio_path):
    """展开 chat template，结尾是助手回合开头和空的思考块 <think>\\n\\n</think>\\n\\n。"""
    return processor.apply_chat_template(
        conversation = build_messages(audio_path = audio_path),
        tokenize = False,
        add_generation_prompt = True,
        enable_thinking = False,
    )


def build_training_text(processor, audio_path, target):
    """渲染带助手回答的完整对话。前半段与 build_prompt 的结果逐字相同，后面接目标文本、<|im_end|> 和换行。"""
    conversation = build_messages(audio_path = audio_path) + [
        {"role": "assistant", "content": [{"type": "text", "text": target}]},
    ]
    return processor.apply_chat_template(
        conversation = conversation,
        tokenize = False,
        add_generation_prompt = False,
        enable_thinking = False,
    )


def to_simplified(text):
    """把中文繁体转成简体，与朗读稿使用同一个词级转换（pipeline.tn.core，OpenCC），保证两种训练目标的字形一致。"""
    from pipeline.tn.core import to_simplified as tn_to_simplified

    return tn_to_simplified(text = text)


def target_text(row, target):
    """取一条样本的训练目标：script 用朗读稿，original 用原文（中文转为简体）。"""
    if target == "script":
        text = row["script"]
    elif target == "original":
        text = row["original"]
        if row["lang"] == "zh":
            text = to_simplified(text = text)
    else:
        raise ValueError(f"未知的训练目标：{target}，可选 {TARGETS}")
    return text.strip()


def read_manifest(
    path,
    audio_root = None,
    langs = None,
    max_duration_s = None,
):
    """读取训练清单（JSONL）。

    相对的 audio_path 以 audio_root 为基准解析，未给出时以清单所在目录为基准。
    langs 和 max_duration_s 用来筛选样本。
    """
    path = Path(path)
    root = Path(audio_root) if audio_root is not None else path.parent
    rows = []
    with open(file = path, mode = "r", encoding = "utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(s = line)
            if langs is not None and row["lang"] not in langs:
                continue
            if max_duration_s is not None and row["duration_s"] > max_duration_s:
                continue
            audio_path = Path(row["audio_path"])
            if not audio_path.is_absolute():
                audio_path = root / audio_path
            row["audio_path"] = str(audio_path)
            rows.append(row)
    return rows


def load_audio(path, start_s = None, stop_s = None):
    """读取音频，统一为 16 kHz 单声道 float32 波形。给出 start_s、stop_s 时只解码这一段（秒）。"""
    from torchcodec.decoders import AudioDecoder

    decoder = AudioDecoder(
        source = str(path),
        sample_rate = SAMPLING_RATE,
        num_channels = 1,
    )
    if start_s is None and stop_s is None:
        samples = decoder.get_all_samples()
    else:
        samples = decoder.get_samples_played_in_range(
            start_seconds = start_s or 0.0,
            stop_seconds = stop_s,
        )
    return samples.data.squeeze(dim = 0).numpy()


class TranscriptionDataset(torch.utils.data.Dataset):
    """每条样本给出波形和目标文本；特征提取和分词放在批次构造中统一完成。

    下标可以是整数（不混音），也可以是 sampling.SampleKey(index, epoch, position)：给出 mixer 时，
    按 (seed, epoch, position) 生成随机数，由 mixer 决定这条语音样本是否混音以及怎样混音。
    """

    def __init__(
        self,
        rows,
        target,
        mixer = None,
        seed = 0,
    ):
        self.rows = rows
        self.target = target
        self.targets = [target_text(row = row, target = target) for row in rows]
        self.empty = [text == "" for text in self.targets]
        self.mixer = mixer
        self.seed = seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        keyed = isinstance(key, SampleKey)
        index = key.index if keyed else int(key)
        row = self.rows[index]
        try:
            if row.get("offset_s") is not None:
                # 虚拟切分的片段：只解码整段文件中的这一段
                waveform = load_audio(
                    path = row["audio_path"],
                    start_s = row["offset_s"],
                    stop_s = row["offset_s"] + row["duration_s"],
                )
            else:
                waveform = load_audio(path = row["audio_path"])
        except Exception as error:
            # 几百万个文件里个别损坏（例如转换时机器意外关机）：记下路径，这条样本交给 TrainCollator 丢掉。
            # 同一个文件每次都读不出，续训与不中断的运行仍然一致。
            print(f"[bad-audio] {row['audio_path']}: {type(error).__name__}: {str(error)[:200]}", file = sys.stderr, flush = True)
            return None
        mix = None
        if self.mixer is not None and keyed and not self.empty[index]:
            rng = np.random.default_rng(seed = [self.seed, key.epoch, STREAM_MIXING, key.position])
            waveform, mix = self.mixer(waveform = waveform, rng = rng)
        return {
            "sample_id": row["sample_id"],
            "lang": row["lang"],
            "audio_path": row["audio_path"],
            "duration_s": float(row["duration_s"]),
            "waveform": waveform,
            "target": self.targets[index],
            "empty": self.empty[index],
            "mix": mix,
        }


def batch_metadata(batch, audio_token_id, n_window, n_window_infer):
    """在 CPU 上预先算出前向要用的位置与索引，训练时不必在 GPU 上逐条样本计算（任务 T59，--precompute-metadata）。

    原实现在前向中逐条样本计算这些量，每条都要把 GPU 上的数取回 CPU，CPU 因此反复等 GPU，批越大越慢。
    这里调用模型代码中同样的函数（音频塔的分块、窗口边界），或按同样的规则计算（位置编号、计损失的位置），
    结果与原实现逐位相同：

    - position_ids：语言模型的位置编号 (3, 批大小, 长度)。输入中只有文字和音频，三路位置相同，都是有效 token
      在本条中的序号，填充位置为 0（与 get_rope_index 对音频、文字输入的结果相同）。
    - audio_mask：input_ids 中音频标记的位置。
    - audio_padded_feature、audio_chunk_lengths、audio_valid_indices、audio_cu_seqlens、audio_max_seqlen：
      音频塔把梅尔特征切成 100 帧的块、卷积后按窗口做注意力，这些是 transformers 的 Qwen3OmniMoeAudioEncoder
      本来在前向开头算出的分块与窗口信息；它的 forward 接受预先算好的同名参数。
    - loss_index、loss_targets、loss_rows：计损失的位置在 (批大小 × 长度) 中的平展下标（第 t 个位置预测
      第 t + 1 个 token）、目标 token 和所在的样本行号。
    """
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        chunk_and_pad_features,
        get_audio_cu_seqlens,
        get_valid_indices,
    )

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    positions = (attention_mask.long().cumsum(dim = -1) - 1).clamp(min = 0) * attention_mask.long()
    feature_mask = batch["feature_attention_mask"].bool()
    feature_lens = feature_mask.sum(dim = -1)
    packed = batch["input_features"].transpose(dim0 = 1, dim1 = 2)[feature_mask].transpose(dim0 = 0, dim1 = 1)
    padded_feature, chunk_lengths = chunk_and_pad_features(
        input_features = packed,
        feature_lens = feature_lens,
        n_window = n_window,
    )
    cu_seqlens = get_audio_cu_seqlens(
        chunk_lengths = chunk_lengths,
        feature_lens = feature_lens,
        n_window_infer = n_window_infer,
        n_window = n_window,
    )
    labels = batch["labels"]
    length = labels.shape[1]
    shifted = labels[:, 1:]
    rows, columns = (shifted != IGNORE_INDEX).nonzero(as_tuple = True)
    return {
        "position_ids": positions[None].expand(size = (3, -1, -1)).contiguous(),
        "audio_mask": input_ids == audio_token_id,
        "audio_padded_feature": padded_feature,
        "audio_chunk_lengths": chunk_lengths,
        "audio_valid_indices": get_valid_indices(chunk_lengths = chunk_lengths, n_window = n_window),
        "audio_cu_seqlens": cu_seqlens,
        "audio_max_seqlen": int((cu_seqlens[1:] - cu_seqlens[:-1]).max()),
        "loss_index": rows * length + columns,
        "loss_targets": shifted[rows, columns],
        "loss_rows": rows,
    }


class TrainCollator:
    """把若干条样本拼成一个批次。

    每条样本渲染成带助手回答的完整对话，整批交给 processor 做特征提取、音频标记展开和分词（与评测推理
    相同的调用），再在最终的 input_ids 上用 mm_assistant_mask 生成标签。给出 audio_windows（音频塔的
    n_window、n_window_infer）时，再用 batch_metadata 预先算出前向要用的位置与索引。
    """

    def __init__(self, processor, audio_windows = None):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.audio_windows = audio_windows

    def __call__(self, items):
        # 读不出音频的样本（TranscriptionDataset 返回 None）不进批次，这一批少几条照常训练
        items = [item for item in items if item is not None]
        if not items:
            raise RuntimeError("这一批的音频全部读不出，检查日志中的 [bad-audio] 路径")
        texts = [
            build_training_text(
                processor = self.processor,
                audio_path = item["audio_path"],
                target = item["target"],
            )
            for item in items
        ]
        inputs = self.processor(
            text = texts,
            audio = [item["waveform"] for item in items],
            sampling_rate = SAMPLING_RATE,
            padding = True,
            return_tensors = "pt",
        )
        labels = build_assistant_labels(
            batch = inputs,
            tokenizer = self.tokenizer,
            spec = ASSISTANT_MASK_SPEC,
        )
        batch = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "mm_token_type_ids": inputs["mm_token_type_ids"],
            "labels": labels,
            "input_features": inputs["input_features"],
            "feature_attention_mask": inputs["feature_attention_mask"],
            "sample_ids": [item["sample_id"] for item in items],
            "langs": [item["lang"] for item in items],
            "audio_seconds": sum(item["duration_s"] for item in items),
            # 本批的空目标样本数和混音样本数，训练日志按它们统计实际比例。
            "n_empty": sum(bool(item.get("empty")) for item in items),
            "empty_rows": torch.tensor(data = [bool(item.get("empty")) for item in items], dtype = torch.bool),
            "n_mixed": sum(item.get("mix") is not None for item in items),
        }
        if self.audio_windows is not None:
            n_window, n_window_infer = self.audio_windows
            batch.update(batch_metadata(
                batch = batch,
                audio_token_id = self.tokenizer.convert_tokens_to_ids(tokens = self.processor.audio_token),
                n_window = n_window,
                n_window_infer = n_window_infer,
            ))
        return batch


def length_grouped_batches(
    durations,
    batch_size,
    seed,
    epoch,
    group_batches = 64,
    max_batch_seconds = None,
):
    """按时长分组成批，返回这一轮的批次列表（每个批次是样本下标列表）。

    每轮先按 (seed, epoch) 打乱全部样本，每 group_batches 个批次的样本按时长排序后依次装批，
    最后再打乱批次顺序。同一批内时长相近，填充少；批次之间仍是随机顺序。
    一批最多 batch_size 条；给出 max_batch_seconds 时，还要求「条数 × 批内最长时长」（填充后的
    音频总秒数，大致正比于显存）不超过它，长音频的批次因此条数更少。
    结果只由参数决定，断点续训时可以重新生成同一轮的批次再跳过已训练的部分。
    """
    generator = random.Random(x = f"{seed}-{epoch}")
    indices = list(range(len(durations)))
    generator.shuffle(x = indices)
    group_size = batch_size * group_batches
    batches = []
    for start in range(0, len(indices), group_size):
        group = sorted(indices[start:start + group_size], key = lambda index: durations[index])
        batch = []
        for index in group:
            # 组内按时长升序，新加入的样本就是批内最长的。
            too_long = (
                max_batch_seconds is not None
                and (len(batch) + 1) * durations[index] > max_batch_seconds
            )
            if batch and (len(batch) == batch_size or too_long):
                batches.append(batch)
                batch = []
            batch.append(index)
        if batch:
            batches.append(batch)
    generator.shuffle(x = batches)
    return batches


def summarize_rows(rows):
    """按语言统计样本数和时长，打印清单概况用。"""
    summary = {}
    for row in rows:
        entry = summary.setdefault(row["lang"], {"samples": 0, "hours": 0.0})
        entry["samples"] += 1
        entry["hours"] += row["duration_s"] / 3600
    return {
        lang: {"samples": entry["samples"], "hours": round(number = entry["hours"], ndigits = 3)}
        for lang, entry in sorted(summary.items())
    }
