"""测试用的小音频文件与清单：在临时目录中写出 16 kHz 单声道 FLAC。"""

import json
from pathlib import Path

import numpy as np
import torch

from pipeline.train.data import SAMPLING_RATE


def write_flac(path, waveform):
    from torchcodec.encoders import AudioEncoder

    samples = torch.from_numpy(np.asarray(a = waveform, dtype = np.float32))[None]
    AudioEncoder(samples = samples, sample_rate = SAMPLING_RATE).to_file(dest = str(path))


def tone(seconds, frequency = 220.0, amplitude = 0.1):
    """模拟语音的正弦波。"""
    times = np.arange(int(seconds * SAMPLING_RATE)) / SAMPLING_RATE
    return (amplitude * np.sin(2 * np.pi * frequency * times)).astype(np.float32)


def noise(seconds, seed, amplitude = 0.05):
    rng = np.random.default_rng(seed = seed)
    return (amplitude * rng.standard_normal(size = int(seconds * SAMPLING_RATE))).astype(np.float32)


def write_background(directory, clips):
    """clips 是 [(类别, 秒数), ...]；写出 audio/ 下的背景声和 background.jsonl，返回列表路径。"""
    directory = Path(directory)
    (directory / "audio").mkdir(parents = True, exist_ok = True)
    manifest = directory / "background.jsonl"
    with open(file = manifest, mode = "w", encoding = "utf-8") as handle:
        for number, (category, seconds) in enumerate(clips):
            relative = f"audio/bg-{number}.flac"
            write_flac(path = directory / relative, waveform = noise(seconds = seconds, seed = number))
            handle.write(json.dumps(obj = {
                "audio_path": relative,
                "duration_s": seconds,
                "category": category,
            }) + "\n")
    return manifest


def speech_rows(directory, durations, lang = "en", empty = ()):
    """写出若干段正弦波作为「语音」，返回清单行；empty 中的下标是空目标样本（lang 为 zxx）。"""
    directory = Path(directory)
    (directory / "speech").mkdir(parents = True, exist_ok = True)
    rows = []
    for number, seconds in enumerate(durations):
        is_empty = number in empty
        path = directory / "speech" / f"s-{number}.flac"
        write_flac(path = path, waveform = tone(seconds = seconds, frequency = 200.0 + 10 * number))
        rows.append({
            "sample_id": f"s-{number}",
            "audio_path": str(path),
            "duration_s": seconds,
            "lang": "zxx" if is_empty else lang,
            "original": "" if is_empty else f"sentence {number}",
            "script": "" if is_empty else f"sentence {number}",
        })
    return rows
