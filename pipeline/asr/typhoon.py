"""Typhoon ASR 泰语识别封装（实验 v1，任务 T08b）。

支持的语言代码：th。

模型：typhoon-ai/typhoon-asr-qwen-1.7b-ctx（Apache-2.0），由 Qwen3-ASR-1.7B-hf 经 LoRA 微调后合并，约 2.07B 参数。
模型卡规定的提示格式：system 消息只写 ``Thai``（也可以在前面放热词表，本封装不使用热词），
用户消息放音频。输出是泰语转写文本。

环境：``envs/asr-hf``（transformers 5.13 以上、torch、soundfile、soxr）。
显存和吞吐见 issue #16 的验收评论。

用法::

    python -m pipeline.asr.typhoon --input audio.jsonl --output results.jsonl --batch-size 16 \
        [--model typhoon-ai/typhoon-asr-qwen-1.7b-ctx 或本地目录] [--audio-root DIR]

输入每行至少含 sample_id、audio_path、lang；audio_path 为相对路径时相对 --audio-root，
未给 --audio-root 时相对输入文件所在目录。输出每行含 sample_id、lang、asr_model、hypothesis。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile
import soxr
import torch

SAMPLE_RATE = 16000
SUPPORTED_LANGS = {"th"}
ASR_MODEL = "typhoon-asr-qwen-1.7b-ctx"


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(file = path, mode = "r", encoding = "utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_audio(path: str) -> np.ndarray:
    """读取音频，转单声道、16 kHz、float32。"""
    wav, sr = soundfile.read(file = path, dtype = "float32", always_2d = True)
    wav = wav.mean(axis = 1)
    if sr != SAMPLE_RATE:
        wav = soxr.resample(wav, sr, SAMPLE_RATE)
    return np.ascontiguousarray(wav, dtype = np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description = "Typhoon ASR 泰语识别")
    parser.add_argument("--input", required = True)
    parser.add_argument("--output", required = True)
    parser.add_argument("--batch-size", type = int, default = 16)
    parser.add_argument("--model", default = "typhoon-ai/typhoon-asr-qwen-1.7b-ctx", help = "模型 ID 或本地目录")
    parser.add_argument("--audio-root", default = None)
    parser.add_argument("--max-new-tokens", type = int, default = 440)
    args = parser.parse_args()

    from transformers import AutoModelForMultimodalLM, AutoProcessor

    rows = read_jsonl(path = args.input)
    root = args.audio_root or os.path.dirname(os.path.abspath(args.input))
    for row in rows:
        if row["lang"] not in SUPPORTED_LANGS:
            raise ValueError(f"Typhoon ASR 只支持 th，收到 {row['lang']}（{row['sample_id']}）")

    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        dtype = torch.bfloat16,
        device_map = "cuda",
    ).eval()

    torch.cuda.reset_peak_memory_stats()
    audios = [load_audio(path = os.path.join(root, row["audio_path"])) for row in rows]
    total_seconds = sum(len(a) for a in audios) / SAMPLE_RATE
    order = sorted(range(len(rows)), key = lambda i: len(audios[i]))
    hyps = [""] * len(rows)

    torch.cuda.synchronize()
    start = time.time()
    for b in range(0, len(order), args.batch_size):
        idx = order[b : b + args.batch_size]
        conversations = [
            [
                {"role": "system", "content": [{"type": "text", "text": "Thai"}]},
                {"role": "user", "content": [{"type": "audio", "audio": audios[i]}]},
            ]
            for i in idx
        ]
        inputs = processor.apply_chat_template(
            conversations,
            tokenize = True,
            add_generation_prompt = True,
            return_dict = True,
            return_tensors = "pt",
            padding = True,
        ).to(model.device, dtype = torch.bfloat16)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens = args.max_new_tokens, do_sample = False)
        out = out[:, inputs["input_ids"].shape[1] :]
        texts = processor.decode(out, return_format = "transcription_only")
        for i, text in zip(idx, texts):
            hyps[i] = text.strip()
    torch.cuda.synchronize()
    wall = time.time() - start

    with open(file = args.output, mode = "w", encoding = "utf-8") as f:
        for row, hyp in zip(rows, hyps):
            out_row = {
                "sample_id": row["sample_id"],
                "lang": row["lang"],
                "asr_model": ASR_MODEL,
                "hypothesis": hyp,
            }
            f.write(json.dumps(out_row, ensure_ascii = False) + "\n")

    peak = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"{ASR_MODEL}: {len(rows)} 条，音频 {total_seconds:.1f} s，墙钟 {wall:.1f} s，"
        f"速度 {total_seconds / wall:.1f} 音频秒/墙钟秒，峰值显存 {peak:.2f} GiB",
        file = sys.stderr,
    )


if __name__ == "__main__":
    main()
