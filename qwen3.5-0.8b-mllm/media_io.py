import json
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
from transformers.video_utils import VideoMetadata



def _check_audio_timeline(video_path, stream_index):
    """检查音轨逐帧时间连续，避免解码器把时间空洞直接拼接掉。"""
    try:
        result = subprocess.run(
            args = [
                "ffprobe", "-v", "error", "-select_streams", str(stream_index),
                "-show_frames", "-show_streams", "-show_entries",
                "frame=pts,nb_samples:stream=time_base,sample_rate",
                "-of", "json", "-i", str(video_path.resolve()),
            ],
            capture_output = True,
            text = True,
            check = True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("读取联合音视频需要 FFmpeg 提供的 ffprobe。") from error
    info = json.loads(s = result.stdout)
    stream = info["streams"][0]
    time_base = Fraction(numerator = stream["time_base"])
    sample_rate = int(stream["sample_rate"])
    tolerance = time_base + Fraction(numerator = 1, denominator = sample_rate)
    first_start = None
    sample_count = 0
    for frame in info["frames"]:
        if "pts" not in frame or "nb_samples" not in frame:
            raise ValueError("音轨缺少时间信息，无法进行音视频联合排列。")
        start = int(frame["pts"]) * time_base
        if first_start is None:
            first_start = start
        expected_start = first_start + Fraction(
            numerator = sample_count, denominator = sample_rate,
        )
        if abs(start - expected_start) > tolerance:
            raise ValueError("当前联合输入要求连续音轨；此文件的音频时间戳存在间断。")
        sample_count += int(frame["nb_samples"])


def load_audio_video(
    video_path,
    video_processor,
    sampling_rate,
    fps = None,
    num_frames = None,
    do_sample_frames = True,
):
    """从同一视频文件读取画面、音轨和时间，保留两路起始时间差。"""
    from torchcodec.decoders import AudioDecoder, VideoDecoder

    video_path = Path(video_path).expanduser()
    if not video_path.is_file():
        raise ValueError(f"找不到视频文件：{video_path}")
    try:
        video_decoder = VideoDecoder(source = video_path, dimension_order = "NHWC")
        audio_decoder = AudioDecoder(
            source = video_path,
            sample_rate = sampling_rate,
            num_channels = 1,
        )
    except (RuntimeError, ValueError) as error:
        raise ValueError(
            f"联合输入需要同一文件内可解码的视频轨道和音频轨道：{video_path}",
        ) from error

    _check_audio_timeline(video_path = video_path, stream_index = audio_decoder.stream_index)

    metadata = VideoMetadata(
        total_num_frames = video_decoder.metadata.num_frames,
        fps = video_decoder.metadata.average_fps,
        width = video_decoder.metadata.width,
        height = video_decoder.metadata.height,
        duration = video_decoder.metadata.duration_seconds,
        video_backend = "torchcodec",
    )
    if do_sample_frames:
        frame_indices = video_processor.sample_frames(
            metadata = metadata,
            fps = fps,
            num_frames = num_frames,
        )
    else:
        frame_indices = np.arange(stop = metadata.total_num_frames)
    frames = video_decoder.get_frames_at(indices = frame_indices.tolist())
    samples = audio_decoder.get_samples_played_in_range(
        start_seconds = audio_decoder.metadata.begin_stream_seconds,
    )
    if frames.data.shape[0] == 0 or samples.data.shape[-1] == 0:
        raise ValueError("联合输入的视频轨道和音频轨道都需要包含有效数据。")

    time_origin = min(float(frames.pts_seconds[0]), samples.pts_seconds)
    metadata.frames_indices = frame_indices.tolist()
    return {
        "video": frames.data.contiguous(),
        "audio": samples.data.squeeze(dim = 0).contiguous().numpy(),
        "metadata": metadata,
        "video_timestamps": (frames.pts_seconds - time_origin).tolist(),
        "audio_start_seconds": samples.pts_seconds - time_origin,
    }
