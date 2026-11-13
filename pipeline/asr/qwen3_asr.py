"""Qwen3-ASR-1.7B 封装：按指定语言转写，输出纯文本。

模型：Qwen/Qwen3-ASR-1.7B-hf（transformers 原生的 qwen3_asr 实现），许可 Apache-2.0。
支持的语言代码（项目语言码）：zh en ja ko de fr ru pt es it id ms th vi fil ar tr fa（全部 18 种）。
环境：envs/asr-qwen3（基于 conda 环境 py3.14 的虚拟环境，transformers 5.16.1，不需要安装 qwen-asr 包）。
显存、吞吐：见 issue #15 的验收评论。

用法：
    python -m pipeline.asr.qwen3_asr --input audio.jsonl --output results.jsonl --batch-size 16 \
        [--model Qwen/Qwen3-ASR-1.7B-hf] [--audio-root 音频根目录]

输入每行至少有 sample_id、audio_path、lang。audio_path 是相对路径时，相对 --audio-root 解析；
没有给 --audio-root 时，相对输入文件所在目录。有 offset_s 的行只识别从 offset_s 秒开始、长 duration_s 秒的一段
（pipeline/realspeech/materialize.py 的虚拟切分：片段不另存文件，读取时在内存中截取）。
输出每行：sample_id、lang、asr_model、hypothesis（只保留转写文本）。转写失败的样本
hypothesis 为空字符串，并多一个 error 字段。

语言与文本处理：
- 目标语言通过强制前缀 "language <语言名><asr_text>" 传给模型，模型只做转写，不会翻译，
  也不会输出语言标签。
- 模型没有关闭标点和反向文本规范化（ITN）的开关，输出的数字写法、标点由模型决定，
  需要在比对环节统一。
- 解码用贪心搜索，max_new_tokens 默认 512；处理器会去掉连续重复的字符或片段。
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B-hf"
ASR_MODEL_NAME = "qwen3-asr-1.7b"
SAMPLE_RATE = 16000


def load_audio(path: Path, offset_s: float | None = None, duration_s: float | None = None) -> np.ndarray:
    """读取音频，转为 16 kHz 单声道 float32。给出 offset_s 时只读从这里开始、长 duration_s 秒的一段（虚拟切分的片段）。"""
    if offset_s is None:
        data, sr = sf.read(file = str(path), dtype = "float32", always_2d = True)
    else:
        sr = sf.info(file = str(path)).samplerate
        data, sr = sf.read(
            file = str(path),
            start = round(offset_s * sr),
            frames = round(duration_s * sr),
            dtype = "float32",
            always_2d = True,
        )
    data = data.mean(axis = 1)
    if sr != SAMPLE_RATE:
        wave = torch.from_numpy(data).unsqueeze(0)
        wave = torchaudio.functional.resample(waveform = wave, orig_freq = sr, new_freq = SAMPLE_RATE)
        data = wave.squeeze(0).numpy()
    return data


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding = "utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def resolve_audio_path(audio_path: str, audio_root: Path) -> Path:
    p = Path(audio_path)
    return p if p.is_absolute() else audio_root / p


def duration_batches(order: list, seconds: list, max_items: int, max_seconds: float) -> list:
    """把按时长升序排好的下标切成批：每批不超过 max_items 条；max_seconds 大于 0 时，
    「条数 × 批内最长时长」（补零后的总时长）也不超过它。短音频一批能放更多条，长音频一批放得少，显存峰值相近。"""
    batches, current = [], []
    for i in order:
        if current and (len(current) >= max_items or (max_seconds > 0 and (len(current) + 1) * seconds[i] > max_seconds)):
            batches.append(current)
            current = []
        current.append(i)
    if current:
        batches.append(current)
    return batches


def main() -> None:
    parser = argparse.ArgumentParser(description = "Qwen3-ASR-1.7B 批量转写")
    parser.add_argument("--input", required = True, help = "含 sample_id、audio_path、lang 的 JSONL")
    parser.add_argument("--output", required = True, help = "识别结果 JSONL")
    parser.add_argument("--batch-size", type = int, default = 16)
    parser.add_argument("--model", default = DEFAULT_MODEL, help = "模型 ID 或本地目录")
    parser.add_argument("--audio-root", default = None, help = "相对 audio_path 的根目录，默认为输入文件所在目录")
    parser.add_argument("--max-new-tokens", type = int, default = 512)
    parser.add_argument("--max-batch-seconds", type = float, default = 0.0,
                        help = "大于 0 时按补零后的总时长组批，--batch-size 为每批条数上限")
    args = parser.parse_args()

    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    input_path = Path(args.input)
    audio_root = Path(args.audio_root) if args.audio_root else input_path.resolve().parent
    rows = read_jsonl(path = input_path)

    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        args.model,
        dtype = torch.bfloat16,
        device_map = "cuda",
    ).eval()

    audios: dict[int, np.ndarray] = {}
    errors: dict[int, str] = {}
    for i, row in enumerate(rows):
        try:
            audios[i] = load_audio(
                path = resolve_audio_path(audio_path = row["audio_path"], audio_root = audio_root),
                offset_s = row.get("offset_s"),
                duration_s = row.get("duration_s"),
            )
        except Exception as e:  # 读取失败只标记该条
            errors[i] = f"load_failed: {e}"
    # 按语言分组、组内按时长排序，批内长度接近，补零少；输出仍按输入顺序写出
    order = sorted(audios, key = lambda i: (rows[i]["lang"], len(audios[i])))

    hyps: dict[int, str] = {}
    audio_seconds = 0.0
    torch.cuda.synchronize()
    t0 = time.time()
    for lang in sorted({rows[i]["lang"] for i in order}):
        idx = [i for i in order if rows[i]["lang"] == lang]
        seconds = {i: len(audios[i]) / SAMPLE_RATE for i in idx}
        for chunk in duration_batches(order = idx, seconds = seconds, max_items = args.batch_size, max_seconds = args.max_batch_seconds):
            try:
                inputs = processor.apply_transcription_request(
                    audio = [audios[i] for i in chunk],
                    language = lang,
                ).to(model.device, dtype = torch.bfloat16)
                with torch.inference_mode():
                    out = model.generate(
                        **inputs,
                        max_new_tokens = args.max_new_tokens,
                        do_sample = False,
                    )
                new_tokens = out[:, inputs["input_ids"].shape[1] :]
                texts = processor.decode(new_tokens, return_format = "transcription_only")
                for i, text in zip(chunk, texts):
                    hyps[i] = text.strip()
            except Exception as e:
                for i in chunk:
                    errors[i] = f"generate_failed: {e}"
            audio_seconds += sum(len(audios[i]) for i in chunk) / SAMPLE_RATE
    torch.cuda.synchronize()
    wall = time.time() - t0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents = True, exist_ok = True)
    with open(output_path, "w", encoding = "utf-8") as f:
        for i, row in enumerate(rows):
            rec = {
                "sample_id": row["sample_id"],
                "lang": row["lang"],
                "asr_model": ASR_MODEL_NAME,
                "hypothesis": hyps.get(i, ""),
            }
            if i in errors:
                rec["error"] = errors[i]
            f.write(json.dumps(rec, ensure_ascii = False) + "\n")

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(
        f"样本 {len(rows)} 条，失败 {len(errors)} 条，音频 {audio_seconds:.1f} 秒，"
        f"墙钟 {wall:.1f} 秒，速度 {audio_seconds / max(wall, 1e-9):.1f} 音频秒/墙钟秒，"
        f"显存峰值 {peak_gb:.2f} GB"
    )


if __name__ == "__main__":
    main()
