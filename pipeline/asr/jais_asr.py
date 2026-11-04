"""Jais-ASR 阿拉伯语识别封装（实验 v1，任务 T08b）。

支持的语言代码：ar。

模型：IFM/Jais-ASR（Apache-2.0），基于 Jais-3-4B 的语音识别模型，bf16 权重约 11.4 GB，用 ``trust_remote_code`` 加载。
可选项：

- ``--diacritics``：输出阿拉伯语元音符号，默认关闭（朗读稿没有元音符号）。
- ``--code-switch``：保留英、法、西、意语片段的原文字母，默认关闭。
- ``--language``：目标语言或方言代码，默认 ``ar``（现代标准语和全部方言）；设为 ``msa`` 只允许现代标准语。

模型对没有语音的音频返回 ``[IFM-NoTranscript]``，后处理把它标为 refused，本封装此时写出空字符串。

环境：``envs/asr-hf``（transformers 5.16.1、torch、accelerate、soundfile、soxr）。
显存和吞吐见 issue #16 的验收评论。

用法::

    python -m pipeline.asr.jais_asr --input audio.jsonl --output results.jsonl --batch-size 8 \
        [--model IFM/Jais-ASR 或本地目录] [--diacritics] [--audio-root DIR]

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
SUPPORTED_LANGS = {"ar"}


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
    parser = argparse.ArgumentParser(description = "Jais-ASR 阿拉伯语识别")
    parser.add_argument("--input", required = True)
    parser.add_argument("--output", required = True)
    parser.add_argument("--batch-size", type = int, default = 8)
    parser.add_argument("--model", default = "IFM/Jais-ASR", help = "模型 ID 或本地目录")
    parser.add_argument("--audio-root", default = None)
    parser.add_argument("--diacritics", action = "store_true", help = "输出元音符号")
    parser.add_argument("--code-switch", action = "store_true", help = "保留外语片段的原文字母")
    parser.add_argument("--language", default = "ar", help = "目标语言或方言代码")
    parser.add_argument("--max-new-tokens", type = int, default = 512)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoProcessor

    rows = read_jsonl(path = args.input)
    root = args.audio_root or os.path.dirname(os.path.abspath(args.input))
    for row in rows:
        if row["lang"] not in SUPPORTED_LANGS:
            raise ValueError(f"Jais-ASR 只支持 ar，收到 {row['lang']}（{row['sample_id']}）")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code = True)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code = True,
        device_map = "cuda",
        dtype = torch.bfloat16,
    ).eval()
    asr_model = "Jais-ASR" + ("-diacritics" if args.diacritics else "")

    template_kwargs = {}
    if args.diacritics:
        template_kwargs["diacritics"] = True
    if args.code_switch:
        template_kwargs["code_switch"] = True
    prompt = processor.tokenizer.apply_chat_template(
        [{"role": "user", "content": [{"type": "audio"}]}],
        tokenize = False,
        add_generation_prompt = True,
        **template_kwargs,
    )

    torch.cuda.reset_peak_memory_stats()
    audios = [load_audio(path = os.path.join(root, row["audio_path"])) for row in rows]
    total_seconds = sum(len(a) for a in audios) / SAMPLE_RATE
    order = sorted(range(len(rows)), key = lambda i: len(audios[i]))
    hyps = [""] * len(rows)
    n_refused = 0
    n_truncated = 0

    torch.cuda.synchronize()
    start = time.time()
    for b in range(0, len(order), args.batch_size):
        idx = order[b : b + args.batch_size]
        inputs = processor(
            text = [prompt] * len(idx),
            audio = [audios[i] for i in idx],
            sampling_rate = [SAMPLE_RATE] * len(idx),
            padding = True,
            return_tensors = "pt",
        ).to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample = False,
                max_new_tokens = args.max_new_tokens,
                language = args.language,
            )
        results = processor.post_process_transcription(
            output,
            input_lengths = inputs["input_ids"].shape[1],
        )
        for i, result in zip(idx, results):
            hyps[i] = result.text.strip()
            n_refused += int(bool(result.refused))
            n_truncated += int(bool(getattr(result, "truncated", False)))
    torch.cuda.synchronize()
    wall = time.time() - start

    with open(file = args.output, mode = "w", encoding = "utf-8") as f:
        for row, hyp in zip(rows, hyps):
            out_row = {
                "sample_id": row["sample_id"],
                "lang": row["lang"],
                "asr_model": asr_model,
                "hypothesis": hyp,
            }
            f.write(json.dumps(out_row, ensure_ascii = False) + "\n")

    peak = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"{asr_model}: {len(rows)} 条，音频 {total_seconds:.1f} s，墙钟 {wall:.1f} s，"
        f"速度 {total_seconds / wall:.1f} 音频秒/墙钟秒，峰值显存 {peak:.2f} GiB，"
        f"refused {n_refused}，truncated {n_truncated}",
        file = sys.stderr,
    )


if __name__ == "__main__":
    main()
