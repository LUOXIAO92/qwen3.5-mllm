"""GigaAM-v3 俄语识别封装（实验 v1，任务 T08b）。

支持的语言代码：ru。

模型：ai-sage/GigaAM-v3（MIT 许可），Conformer 编码器，约 240M 参数。仓库的不同 revision 对应不同变体：

- ``ctc``、``rnnt``：无标点、无大小写、数字按读法写出，贴近声学。默认使用 ``rnnt``。
- ``e2e_ctc``、``e2e_rnnt``：带标点和大小写，数字写成阿拉伯数字，可作为选项。

环境：``envs/asr-gigaam``（torch、torchaudio、transformers==4.57.1，加 hydra-core、omegaconf、sentencepiece、
soundfile、soxr；transformers 5 在初始化时会因 meta 设备报错，需要用 4.57.1）。
模型代码用 ``trust_remote_code`` 加载，读音频由本模块完成，所以不需要 ffmpeg 和 pyannote。
显存和吞吐见 issue #16 的验收评论。

用法::

    python -m pipeline.asr.gigaam --input audio.jsonl --output results.jsonl --batch-size 16 \
        [--model ai-sage/GigaAM-v3 或本地目录] [--variant rnnt] [--audio-root DIR]

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
SUPPORTED_LANGS = {"ru"}
VARIANTS = ["ctc", "rnnt", "e2e_ctc", "e2e_rnnt"]


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
    parser = argparse.ArgumentParser(description = "GigaAM-v3 俄语识别")
    parser.add_argument("--input", required = True)
    parser.add_argument("--output", required = True)
    parser.add_argument("--batch-size", type = int, default = 16)
    parser.add_argument("--model", default = "ai-sage/GigaAM-v3", help = "模型 ID 或本地目录")
    parser.add_argument(
        "--variant",
        default = "rnnt",
        choices = VARIANTS,
        help = "模型变体，对应仓库的 revision",
    )
    parser.add_argument("--audio-root", default = None)
    args = parser.parse_args()

    import transformers.dynamic_module_utils as dynamic_module_utils
    from transformers import AutoModel

    # 模型代码只在长音频切分函数里才导入 pyannote（并且需要接受 gated 模型的条款），短音频识别用不到。
    # transformers 会按文本扫描依赖并报错，这里跳过这项检查。
    dynamic_module_utils.check_imports = lambda filename: []

    rows = read_jsonl(path = args.input)
    root = args.audio_root or os.path.dirname(os.path.abspath(args.input))
    for row in rows:
        if row["lang"] not in SUPPORTED_LANGS:
            raise ValueError(f"GigaAM 只支持 ru，收到 {row['lang']}（{row['sample_id']}）")

    kwargs = {"trust_remote_code": True}
    if not os.path.isdir(args.model):
        kwargs["revision"] = args.variant
    model = AutoModel.from_pretrained(args.model, **kwargs).cuda().eval()
    asr = model.model
    asr_model = f"GigaAM-v3-{args.variant}"

    torch.cuda.reset_peak_memory_stats()
    audios = [load_audio(path = os.path.join(root, row["audio_path"])) for row in rows]
    total_seconds = sum(len(a) for a in audios) / SAMPLE_RATE
    order = sorted(range(len(rows)), key = lambda i: len(audios[i]))
    hyps = [""] * len(rows)

    torch.cuda.synchronize()
    start = time.time()
    with torch.inference_mode():
        for b in range(0, len(order), args.batch_size):
            idx = order[b : b + args.batch_size]
            lengths = torch.tensor([len(audios[i]) for i in idx], device = "cuda")
            batch = torch.zeros(len(idx), int(lengths.max()), device = "cuda")
            for k, i in enumerate(idx):
                batch[k, : len(audios[i])] = torch.from_numpy(audios[i])
            encoded, encoded_len = asr.forward(batch, lengths)
            texts = asr.decoding.decode(asr.head, encoded, encoded_len)
            for i, text in zip(idx, texts):
                hyps[i] = text.strip()
    torch.cuda.synchronize()
    wall = time.time() - start

    with open(file = args.output, mode = "w", encoding = "utf-8") as f:
        for row, hyp in zip(rows, hyps):
            out = {
                "sample_id": row["sample_id"],
                "lang": row["lang"],
                "asr_model": asr_model,
                "hypothesis": hyp,
            }
            f.write(json.dumps(out, ensure_ascii = False) + "\n")

    peak = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"{asr_model}: {len(rows)} 条，音频 {total_seconds:.1f} s，墙钟 {wall:.1f} s，"
        f"速度 {total_seconds / wall:.1f} 音频秒/墙钟秒，峰值显存 {peak:.2f} GiB",
        file = sys.stderr,
    )


if __name__ == "__main__":
    main()
