"""波斯语 ASR 交叉核对封装：whisper-large-v3-turbo 的波斯语微调版。

默认模型 ``MohammadGholizadeh/whisper-large-v3-turbo-persian``：许可 MIT，不是 gated，在约 534 小时
波斯语（Filimo、YouTube、Common Voice 17）上全量微调。训练数据不含 FLEURS，所以在 FLEURS 上的
成绩可以直接作为独立核对。（另一个候选 ``givia/whisper-large-v3-turbo-persian`` 许可 CC-BY-4.0，
但训练数据含 FLEURS，用 FLEURS 评测会偏乐观，因此未选用。）

用法（在 envs/asr-nemo-fa 环境中运行，GPU 命令需通过 flock 排队）::

    flock <工作目录>/gpu.lock python -m pipeline.asr.whisper_fa \\
        --input audio.jsonl --output results.jsonl --batch-size 16 \\
        [--model MohammadGholizadeh/whisper-large-v3-turbo-persian] [--audio-root <目录>]

输入输出格式与 pipeline.asr.nemotron_farsi 相同：输入每行 ``sample_id``、``audio_path``、``lang``；
输出每行 ``sample_id``、``lang``、``asr_model``、``hypothesis``，顺序与输入一致。
相对 ``audio_path`` 相对 ``--audio-root`` 解析，未给时相对输入文件所在目录。

支持的语言代码：只有 ``fa``（解码时强制 ``language = "fa"``、``task = "transcribe"``，不带时间戳，贪心解码）。
超过 30 秒的音频按 30 秒窗口分块（``chunk_length_s = 30``）。

输出习惯：带标点（「،」「.」「؟」），保留 ZWNJ，口语化拼写（如「میشه」），数字多为阿拉伯数字。
不做文本规范化，只去掉首尾空白。

环境：与 pipeline.asr.nemotron_farsi 共用 envs/asr-nemo-fa（transformers 5.22.1、torch 2.13+cu130）。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

DEFAULT_MODEL = "MohammadGholizadeh/whisper-large-v3-turbo-persian"
TARGET_SR = 16000
SUPPORTED_LANGS = {"fa"}


def load_audio(path: Path) -> np.ndarray:
    """读取音频，转单声道，必要时重采样到 16 kHz，返回 float32 数组。"""
    audio, sr = sf.read(file = str(path), dtype = "float32", always_2d = True)
    audio = audio.mean(axis = 1)
    if sr != TARGET_SR:
        import torchaudio

        audio = torchaudio.functional.resample(
            waveform = torch.from_numpy(audio),
            orig_freq = sr,
            new_freq = TARGET_SR,
        ).numpy()
    return audio


def main() -> None:
    parser = argparse.ArgumentParser(description = "波斯语 ASR：whisper-large-v3-turbo 波斯语微调版")
    parser.add_argument("--input", required = True, type = Path)
    parser.add_argument("--output", required = True, type = Path)
    parser.add_argument("--batch-size", type = int, default = 16)
    parser.add_argument("--model", default = DEFAULT_MODEL, help = "模型 ID 或本地目录")
    parser.add_argument("--audio-root", type = Path, default = None)
    args = parser.parse_args()

    from transformers import pipeline

    rows = []
    with open(args.input, encoding = "utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    for row in rows:
        if row["lang"] not in SUPPORTED_LANGS:
            raise SystemExit(f"不支持的语言 {row['lang']}（sample_id = {row['sample_id']}），本模型只支持 fa")

    audio_root = args.audio_root if args.audio_root is not None else args.input.resolve().parent
    audio_root = Path(audio_root)

    asr = pipeline(
        task = "automatic-speech-recognition",
        model = args.model,
        dtype = torch.bfloat16,
        device = "cuda",
    )

    audios = []
    for row in rows:
        path = Path(row["audio_path"])
        audios.append(load_audio(path = path if path.is_absolute() else audio_root / path))
    total_audio_s = sum(len(a) for a in audios) / TARGET_SR

    # 按时长排序后分批，结果按原顺序写回
    order = sorted(range(len(rows)), key = lambda i: len(audios[i]))
    hypotheses = [""] * len(rows)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.time()
    outputs = asr(
        [{"raw": audios[i], "sampling_rate": TARGET_SR} for i in order],
        batch_size = args.batch_size,
        chunk_length_s = 30,
        generate_kwargs = {"language": "fa", "task": "transcribe", "num_beams": 1, "do_sample": False},
    )
    for i, out in zip(order, outputs):
        hypotheses[i] = out["text"].strip()
    torch.cuda.synchronize()
    wall_s = time.time() - start

    args.output.parent.mkdir(parents = True, exist_ok = True)
    with open(args.output, "w", encoding = "utf-8") as f:
        for row, hyp in zip(rows, hypotheses):
            record = {
                "sample_id": row["sample_id"],
                "lang": row["lang"],
                "asr_model": args.model,
                "hypothesis": hyp,
            }
            f.write(json.dumps(obj = record, ensure_ascii = False) + "\n")

    stats = {
        "n": len(rows),
        "audio_s": round(total_audio_s, 2),
        "wall_s": round(wall_s, 2),
        "rtf_x": round(total_audio_s / wall_s, 2),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "batch_size": args.batch_size,
    }
    print(json.dumps(obj = stats), file = sys.stderr)


if __name__ == "__main__":
    main()
