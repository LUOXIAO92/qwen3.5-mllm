"""Haas-ja-v1.0 日语识别封装（实验 v1，任务 T08b）。

支持的语言代码：ja。

模型：third-intelligence/Haas-ja-v1.0（Apache-2.0），Parakeet-TDT 结构（FastConformer 编码器加 TDT 解码器），
预测网络用 Kimi Delta Attention，约 616M 参数，NeMo 2.7 格式。输出风格（汉字、假名、数字写法）由训练数据决定，
没有可调选项，也没有自带的标点规范化开关。

环境：``envs/asr-nemo``（Python 3.12，nemo-toolkit[asr]==2.7.3，flash-linear-attention==0.5.2，
torch==2.14.0 与 triton==3.8.0，与官方 uv.lock 一致；triton 3.9 会让 fla 报错）。
Haas-ASR 官方推理包 ``haas_asr`` 来自 https://github.com/third-intelligence/Haas-ASR ，克隆后把其 ``src`` 加入
PYTHONPATH 即可导入。
模型以 bf16 在 GPU 上运行。超过 30 秒的音频用 ``haas_asr.transcribe_longform`` 按静音处切分。
显存和吞吐见 issue #16 的验收评论。

用法::

    python -m pipeline.asr.haas_ja --input audio.jsonl --output results.jsonl --batch-size 32 \
        [--model third-intelligence/Haas-ja-v1.0 或 .nemo 文件] [--audio-root DIR]

输入每行至少含 sample_id、audio_path、lang；audio_path 为相对路径时相对 --audio-root，
未给 --audio-root 时相对输入文件所在目录。输出每行含 sample_id、lang、asr_model、hypothesis。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

SUPPORTED_LANGS = {"ja"}
ASR_MODEL = "Haas-ja-v1.0"
SHORT_LIMIT_SECONDS = 30.0


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(file = path, mode = "r", encoding = "utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description = "Haas-ja-v1.0 日语识别")
    parser.add_argument("--input", required = True)
    parser.add_argument("--output", required = True)
    parser.add_argument("--batch-size", type = int, default = 32)
    parser.add_argument("--model", default = "third-intelligence/Haas-ja-v1.0", help = "模型 ID 或 .nemo 文件")
    parser.add_argument("--audio-root", default = None)
    args = parser.parse_args()

    import haas_asr

    rows = read_jsonl(path = args.input)
    root = args.audio_root or os.path.dirname(os.path.abspath(args.input))
    for row in rows:
        if row["lang"] not in SUPPORTED_LANGS:
            raise ValueError(f"Haas-ja 只支持 ja，收到 {row['lang']}（{row['sample_id']}）")

    model = haas_asr.load_model(args.model)

    torch.cuda.reset_peak_memory_stats()
    audios = [haas_asr.load_audio(os.path.join(root, row["audio_path"])) for row in rows]
    total_seconds = sum(len(a) for a in audios) / haas_asr.SAMPLE_RATE
    hyps = [""] * len(rows)
    short = [i for i, a in enumerate(audios) if len(a) / haas_asr.SAMPLE_RATE <= SHORT_LIMIT_SECONDS]
    long_ = [i for i in range(len(rows)) if i not in set(short)]

    torch.cuda.synchronize()
    start = time.time()
    if short:
        texts = haas_asr.transcribe(model, [audios[i] for i in short], batch_size = args.batch_size)
        for i, text in zip(short, texts):
            hyps[i] = text.strip()
    for i in long_:
        hyps[i] = haas_asr.transcribe_longform(model, audios[i], batch_size = args.batch_size).strip()
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
