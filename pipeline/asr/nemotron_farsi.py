"""波斯语 ASR 封装：nemotron-asr-streaming-farsi（mehdi-hf/nemotron-asr-streaming-farsi）。

模型是 NVIDIA nemotron-3.5-asr-streaming-0.6b 的波斯语微调版（FastConformer 编码器 + RNN-T 解码器，
约 0.6B 参数），许可 OpenMDW-1.1。它原本为流式识别设计，本封装按整段音频离线推理，并选用延迟最大、
准确率最高的配置：前瞻 13 个令牌（约 1.12 秒，``set_num_lookahead_tokens(13)``），贪心解码。

用法（在 envs/asr-nemo-fa 环境中运行，GPU 命令需通过 flock 排队）::

    flock <工作目录>/gpu.lock python -m pipeline.asr.nemotron_farsi \\
        --input audio.jsonl --output results.jsonl --batch-size 16 \\
        [--model mehdi-hf/nemotron-asr-streaming-farsi] [--audio-root <目录>]

输入 JSONL 每行至少有 ``sample_id``、``audio_path``、``lang``。``audio_path`` 为相对路径时，
相对 ``--audio-root`` 解析；未给 ``--audio-root`` 时相对输入文件所在目录。
输出 JSONL 每行：``sample_id``、``lang``、``asr_model``、``hypothesis``，顺序与输入一致。
``match``、``compare_detail`` 由后续核对流水线填写，这里不写。吞吐统计以一行 JSON 写到标准错误。

支持的语言代码：只有 ``fa``。其他语言的行会直接报错。

推理路径：本封装使用 Hugging Face Transformers（>= 5.18）的 ``AutoModelForRNNT``。模型卡称其权重与
NeMo 的 ``.nemo`` 文件逐位相同，FLEURS 测试集上 WER 8.81 与 NeMo 的 8.77 一致，所以环境不必安装 NeMo。
语言提示固定为 ``fa-IR``（模型微调时只用了这一个提示）。NeMo 3.0 的 ``transcribe()`` 会对每条话语
随机选提示，不适合直接使用；Transformers 路径没有这个问题。

输出习惯：只有波斯文字母，无标点；保留 ZWNJ（零宽不连字，U+200C）拼写，如「می‌شه」「کتاب‌ها」；
数字写成读出来的词（「یازده و سی و پنج」而不是「11:35」）；没有拉丁字母。

文本规范化：模型自身输出已是规范化文本，封装不再做任何处理，只去掉首尾空白。

环境：Python 3.14（基于 conda 环境 py3.14 的 venv，--system-site-packages），torch 2.13+cu130，
transformers 5.22.1（venv 内安装，覆盖系统的 5.16.1），soundfile。
权重许可：OpenMDW-1.1。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

DEFAULT_MODEL = "mehdi-hf/nemotron-asr-streaming-farsi"
TARGET_SR = 16000
LOOKAHEAD_TOKENS = 13  # 前瞻约 1.12 秒，准确率最高
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


def read_rows(input_path: Path) -> list:
    rows = []
    with open(input_path, encoding = "utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description = "波斯语 ASR：nemotron-asr-streaming-farsi")
    parser.add_argument("--input", required = True, type = Path)
    parser.add_argument("--output", required = True, type = Path)
    parser.add_argument("--batch-size", type = int, default = 16)
    parser.add_argument("--model", default = DEFAULT_MODEL, help = "模型 ID 或本地目录")
    parser.add_argument("--audio-root", type = Path, default = None)
    parser.add_argument("--dtype", choices = ["float32", "bfloat16"], default = "float32")
    args = parser.parse_args()

    from transformers import AutoModelForRNNT, AutoProcessor

    rows = read_rows(input_path = args.input)
    for row in rows:
        if row["lang"] not in SUPPORTED_LANGS:
            raise SystemExit(f"不支持的语言 {row['lang']}（sample_id = {row['sample_id']}），本模型只支持 fa")

    audio_root = args.audio_root if args.audio_root is not None else args.input.resolve().parent
    audio_root = Path(audio_root)

    processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path = args.model)
    model = AutoModelForRNNT.from_pretrained(
        pretrained_model_name_or_path = args.model,
        dtype = getattr(torch, args.dtype),
        device_map = "cuda",
    ).eval()
    processor.set_num_lookahead_tokens(LOOKAHEAD_TOKENS)

    audios = []
    for row in rows:
        path = Path(row["audio_path"])
        audios.append(load_audio(path = path if path.is_absolute() else audio_root / path))
    total_audio_s = sum(len(a) for a in audios) / TARGET_SR

    # 按时长排序后分批，减少补零；结果按原顺序写回
    order = sorted(range(len(rows)), key = lambda i: len(audios[i]))
    hypotheses = [""] * len(rows)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.time()
    for b in range(0, len(order), args.batch_size):
        idx = order[b : b + args.batch_size]
        inputs = processor(
            [audios[i] for i in idx],
            sampling_rate = TARGET_SR,
            language = "fa-IR",
        )
        inputs = inputs.to(model.device, dtype = model.dtype)
        with torch.inference_mode():
            output = model.generate(**inputs, return_dict_in_generate = True)
        texts = processor.decode(output.sequences, skip_special_tokens = True)
        for i, text in zip(idx, texts):
            hypotheses[i] = text.strip()
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
        "dtype": args.dtype,
    }
    print(json.dumps(obj = stats), file = sys.stderr)


if __name__ == "__main__":
    main()
