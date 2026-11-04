"""Whisper-large-v3 封装：按指定语言转写，输出纯文本。

模型：openai/whisper-large-v3，许可 Apache-2.0。
支持的语言代码（项目语言码）：zh en ja ko de fr ru pt es it id ms th vi fil ar tr fa（全部 18 种）。
菲律宾语 fil 在 Whisper 中的语言标记是 tl（塔加洛语），封装内部自动转换。
环境：envs/asr-qwen3（与 Qwen3-ASR 共用，transformers 5.16.1）。
显存、吞吐：见 issue #15 的验收评论。

用法：
    python -m pipeline.asr.whisper --input audio.jsonl --output results.jsonl --batch-size 16 \
        [--model openai/whisper-large-v3] [--audio-root 音频根目录]

输入每行至少有 sample_id、audio_path、lang。audio_path 是相对路径时，相对 --audio-root 解析；
没有给 --audio-root 时，相对输入文件所在目录。
输出每行：sample_id、lang、asr_model、hypothesis（只保留转写文本）。转写失败的样本
hypothesis 为空字符串，并多一个 error 字段。

语言与文本处理：
- 解码时强制语言标记并指定 task = transcribe，模型只转写，不翻译。
- Whisper 没有关闭标点或数字写法的开关，输出就是模型原文，不套用 Whisper 自带的 EnglishTextNormalizer。
- 不超过 30 秒的音频按批量直接解码；超过 30 秒的音频走 transformers 的长音频顺序解码（带时间戳 token，
  最终文本里不含时间戳）。
- 贪心解码，不使用温度回退，避免结果不可复现。
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

DEFAULT_MODEL = "openai/whisper-large-v3"
ASR_MODEL_NAME = "whisper-large-v3"
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 30 * SAMPLE_RATE

# 项目语言码到 Whisper 语言标记；不在表中的直接使用原代码
WHISPER_LANG = {"fil": "tl"}


def load_audio(path: Path) -> np.ndarray:
    """读取音频，转为 16 kHz 单声道 float32。"""
    data, sr = sf.read(file = str(path), dtype = "float32", always_2d = True)
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


def main() -> None:
    parser = argparse.ArgumentParser(description = "Whisper-large-v3 批量转写")
    parser.add_argument("--input", required = True, help = "含 sample_id、audio_path、lang 的 JSONL")
    parser.add_argument("--output", required = True, help = "识别结果 JSONL")
    parser.add_argument("--batch-size", type = int, default = 16)
    parser.add_argument("--model", default = DEFAULT_MODEL, help = "模型 ID 或本地目录")
    parser.add_argument("--audio-root", default = None, help = "相对 audio_path 的根目录，默认为输入文件所在目录")
    parser.add_argument("--max-new-tokens", type = int, default = 440)
    args = parser.parse_args()

    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    input_path = Path(args.input)
    audio_root = Path(args.audio_root) if args.audio_root else input_path.resolve().parent
    rows = read_jsonl(path = input_path)

    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model,
        dtype = torch.float16,
        device_map = "cuda",
    ).eval()

    audios: dict[int, np.ndarray] = {}
    errors: dict[int, str] = {}
    for i, row in enumerate(rows):
        try:
            audios[i] = load_audio(path = resolve_audio_path(audio_path = row["audio_path"], audio_root = audio_root))
        except Exception as e:  # 读取失败只标记该条
            errors[i] = f"load_failed: {e}"
    order = sorted(audios, key = lambda i: (rows[i]["lang"], len(audios[i])))

    def transcribe(chunk: list[int], language: str, long_form: bool) -> list[str]:
        if long_form:
            feats = processor(
                [audios[i] for i in chunk],
                sampling_rate = SAMPLE_RATE,
                return_tensors = "pt",
                truncation = False,
                padding = "longest",
                return_attention_mask = True,
            )
        else:
            feats = processor(
                [audios[i] for i in chunk],
                sampling_rate = SAMPLE_RATE,
                return_tensors = "pt",
            )
        feats = feats.to(model.device, dtype = torch.float16)
        kwargs = dict(
            language = language,
            task = "transcribe",
            do_sample = False,
            num_beams = 1,
            max_new_tokens = args.max_new_tokens,
        )
        if long_form:
            kwargs["return_timestamps"] = True
            kwargs["condition_on_prev_tokens"] = False
        with torch.inference_mode():
            out = model.generate(**feats, **kwargs)
        if isinstance(out, dict) or hasattr(out, "sequences"):
            out = out["sequences"]
        return [t.strip() for t in processor.batch_decode(out, skip_special_tokens = True)]

    hyps: dict[int, str] = {}
    audio_seconds = 0.0
    torch.cuda.synchronize()
    t0 = time.time()
    for lang in sorted({rows[i]["lang"] for i in order}):
        language = WHISPER_LANG.get(lang, lang)
        short = [i for i in order if rows[i]["lang"] == lang and len(audios[i]) <= CHUNK_SAMPLES]
        long = [i for i in order if rows[i]["lang"] == lang and len(audios[i]) > CHUNK_SAMPLES]
        for group, long_form in ((short, False), (long, True)):
            for start in range(0, len(group), args.batch_size):
                chunk = group[start : start + args.batch_size]
                try:
                    for i, text in zip(chunk, transcribe(chunk = chunk, language = language, long_form = long_form)):
                        hyps[i] = text
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
