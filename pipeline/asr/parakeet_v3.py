"""Parakeet-TDT-0.6B-v3 封装：作为 en de fr es it pt ru 的第二个核对用 ASR。

模型：nvidia/parakeet-tdt-0.6b-v3（CC BY 4.0），FastConformer 编码器加 TDT 解码器（RNN-T 类结构，约 600M 参数），
覆盖 25 种欧洲语言，其中属于项目语言码的有：en de fr es it pt ru。其余项目语言不在支持范围内，
这些样本的 hypothesis 为空并带 error 字段。
模型自己判断语言，没有指定语言的入口，所以本封装不使用输入中的 lang 做转写，只用它检查是否在支持范围内。
不对输出做任何文字规范化；模型输出自带标点和大小写，数字写法由模型决定，需要在比对环节统一。

环境：envs/asr-nemo（Python 3.12，nemo-toolkit[asr] 2.7.3，torch 2.14.0+cu130），与 Haas-ja 共用，
不需要额外源码克隆。权重来自 Hugging Face 仓库 nvidia/parakeet-tdt-0.6b-v3，建立方法与实测数字见 issue #33。
模型以 bf16 在 GPU 上运行；音频统一读成 16 kHz 单声道 float32，按时长排序成批。
NeMo 的 TDT 解码每遇到一种新的批形状就保留一份显存（实测每批约 0.6–0.9 GB，torch 的已分配量不变、保留量持续增长，
关闭 use_cuda_graph_decoder 也一样；在批之间 empty_cache 会导致非法显存访问）。按时长排序后几乎每批形状都不同，
一次处理两千条时几十批之后显存涨满 32 GB。所以每次调用的条数要有限：流水线配置中 parakeet_v3 的 chunk_size 设为 256。

用法::

    python -m pipeline.asr.parakeet_v3 --input audio.jsonl --output results.jsonl --batch-size 32 \
        [--model nvidia/parakeet-tdt-0.6b-v3 或 .nemo 文件] [--audio-root DIR]

输入每行至少有 sample_id、audio_path、lang；audio_path 是相对路径时相对 --audio-root，
没有给 --audio-root 时相对输入文件所在目录。输出每行含 sample_id、lang、asr_model、hypothesis；
失败的样本 hypothesis 为空字符串并多一个 error 字段。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
ASR_MODEL_NAME = "parakeet-tdt-0.6b-v3"
SAMPLE_RATE = 16000
SUPPORTED_LANGS = {"en", "de", "fr", "es", "it", "pt", "ru"}


def load_audio(path: Path) -> np.ndarray:
    """读取音频，转为 16 kHz 单声道 float32。"""
    data, sr = sf.read(file = str(path), dtype = "float32", always_2d = True)
    data = data.mean(axis = 1)
    if sr != SAMPLE_RATE:
        import librosa  # NeMo 的依赖，环境中已有

        data = librosa.resample(y = data, orig_sr = sr, target_sr = SAMPLE_RATE)
    return data


def read_jsonl(path: Path) -> list[dict]:
    with open(file = path, mode = "r", encoding = "utf-8") as f:
        return [json.loads(s = line) for line in f if line.strip()]


def hyp_text(item) -> str:
    """NeMo 可能返回字符串或 Hypothesis 对象，统一取出文本。"""
    return (item if isinstance(item, str) else item.text).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description = "Parakeet-TDT-0.6B-v3 批量转写")
    parser.add_argument("--input", required = True, help = "含 sample_id、audio_path、lang 的 JSONL")
    parser.add_argument("--output", required = True, help = "识别结果 JSONL")
    parser.add_argument("--batch-size", type = int, default = 32)
    parser.add_argument("--model", default = DEFAULT_MODEL, help = "模型 ID 或 .nemo 文件")
    parser.add_argument("--audio-root", default = None, help = "相对 audio_path 的根目录，默认为输入文件所在目录")
    args = parser.parse_args()

    import nemo.collections.asr as nemo_asr

    input_path = Path(args.input)
    audio_root = Path(args.audio_root) if args.audio_root else input_path.resolve().parent
    rows = read_jsonl(path = input_path)

    audios: dict[int, np.ndarray] = {}
    errors: dict[int, str] = {}
    for i, row in enumerate(rows):
        if row["lang"] not in SUPPORTED_LANGS:
            errors[i] = f"unsupported_lang: {row['lang']}"
            continue
        p = Path(row["audio_path"])
        try:
            audios[i] = load_audio(path = p if p.is_absolute() else audio_root / p)
        except Exception as e:  # 读取失败只标记该条
            errors[i] = f"load_failed: {e}"

    model = nemo_asr.models.ASRModel.from_pretrained(model_name = args.model)
    model = model.to(dtype = torch.bfloat16).to(device = "cuda").eval()

    order = sorted(audios, key = lambda i: len(audios[i]))
    hyps: dict[int, str] = {}
    audio_seconds = 0.0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    for start in range(0, len(order), args.batch_size):
        chunk = order[start : start + args.batch_size]
        try:
            with torch.inference_mode():
                out = model.transcribe(
                    audio = [audios[i] for i in chunk],
                    batch_size = len(chunk),
                    verbose = False,
                )
            if isinstance(out, tuple):  # 旧版 NeMo 返回 (best, all)
                out = out[0]
            for i, item in zip(chunk, out):
                hyps[i] = hyp_text(item = item)
        except Exception as e:
            for i in chunk:
                errors[i] = f"generate_failed: {e}"
        audio_seconds += sum(len(audios[i]) for i in chunk) / SAMPLE_RATE
    torch.cuda.synchronize()
    wall = time.time() - t0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents = True, exist_ok = True)
    with open(file = output_path, mode = "w", encoding = "utf-8") as f:
        for i, row in enumerate(rows):
            rec = {
                "sample_id": row["sample_id"],
                "lang": row["lang"],
                "asr_model": ASR_MODEL_NAME,
                "hypothesis": hyps.get(i, ""),
            }
            if i in errors:
                rec["error"] = errors[i]
            f.write(json.dumps(obj = rec, ensure_ascii = False) + "\n")

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(
        f"样本 {len(rows)} 条，失败 {len(errors)} 条，音频 {audio_seconds:.1f} 秒，"
        f"墙钟 {wall:.1f} 秒，速度 {audio_seconds / max(wall, 1e-9):.1f} 音频秒/墙钟秒，"
        f"显存峰值 {peak_gb:.2f} GB",
        file = sys.stderr,
    )


if __name__ == "__main__":
    main()
