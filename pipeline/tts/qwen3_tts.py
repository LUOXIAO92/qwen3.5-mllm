"""Qwen3-TTS 封装：内置音色（CustomVoice）与声音克隆（Base）。

模型：Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice 与 Qwen/Qwen3-TTS-12Hz-1.7B-Base，许可 Apache-2.0。
支持的语言代码：zh en ja ko de fr ru pt es it（其余语言不支持，任务会被标记为 failed）。
环境：qwen-tts 0.1.1（上游仓库安装）+ 12.8 及以上构建的 PyTorch，虚拟环境 envs/tts-qwen3。
显存：两个模型各约 4.5 GB 权重，批量推理的峰值见 issue #11 的实测。
吞吐：见 issue #11 的验收评论（含批量大小、实时倍率、显存峰值）。

用法：
    python -m pipeline.tts.qwen3_tts --input jobs.jsonl --output-dir out --batch-size 8

每行任务的字段：
    sample_id, lang, text, mode
    mode = "builtin"：另给 builtin_name（音色名，见 BUILTIN_SPEAKERS）
    mode = "clone"  ：另给 ref_audio（参考音频路径）和 ref_text（参考音频的文本）
输出：<output-dir>/<sample_id>.flac（16 kHz 单声道）与 <output-dir>/results.jsonl，
结果行保留输入字段，并增加 audio_path、duration_s、status（ok 或 failed）、error。

文本处理：模型内部只把文本套进对话模板后分词，不做数字、符号的规范化，因此朗读稿原样合成。
长文本按句切分后分别合成，再用 0.25 秒静音拼接。
"""

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from pipeline.text.langs import CHARS_PER_SECOND

CODEC_TOKENS_PER_SECOND = 12.5
CUSTOM_VOICE_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
BASE_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
OUTPUT_SR = 16000
SENTENCE_GAP_S = 0.25
# 单个句子超过该字符数时，在逗号等处继续切分
MAX_SEGMENT_CHARS = 120

LANG_NAMES = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}

# 9 个内置音色及其母语（来自上游模型卡）
BUILTIN_SPEAKERS = {
    "Vivian": "zh",
    "Serena": "zh",
    "Uncle_Fu": "zh",
    "Dylan": "zh（北京话）",
    "Eric": "zh（四川话）",
    "Ryan": "en",
    "Aiden": "en",
    "Ono_Anna": "ja",
    "Sohee": "ko",
}

# 句末标点：中日文后面不一定有空格，用后行断言切分
_SENT_END = re.compile(r"(?<=[。！？!?…])\s*|(?<=[.])\s+")
_ABBREVIATION = re.compile(r"\b(Dr|Mr|Mrs|Ms|Prof|St|Sr|Jr|vs|etc|Nr|Hr|Fr|Sra|Sr|Sig)\.$")
_CLAUSE_END =re.compile(r"(?<=[，、；;,：:])\s*")


def split_text(text: str, max_chars: int = MAX_SEGMENT_CHARS) -> list:
    """按句切分；过长的句子再按逗号等切分。只会去掉片段之间的空白，不改动其他字符。"""
    # 先按句末标点切开，再把以常见缩写结尾的片段并入下一句（如 "Dr. Smith"）
    sentences = []
    for piece in _SENT_END.split(text.strip()):
        piece = piece.strip()
        if not piece:
            continue
        if sentences and _ABBREVIATION.search(sentences[-1]):
            sentences[-1] += " " + piece
        else:
            sentences.append(piece)
    segments = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            segments.append(sentence)
            continue
        buffer = ""
        for clause in _CLAUSE_END.split(sentence):
            if not clause:
                continue
            if buffer and len(buffer) + len(clause) > max_chars:
                segments.append(buffer.strip())
                buffer = ""
            buffer += clause + " "
        if buffer.strip():
            segments.append(buffer.strip())
    return segments


def load_models(modes: set, device: str) -> dict:
    """按任务里出现的模式加载模型，避免多占显存。"""
    from qwen_tts import Qwen3TTSModel

    models = {}
    for mode, model_id in (("builtin", CUSTOM_VOICE_ID), ("clone", BASE_ID)):
        if mode in modes:
            models[mode] = Qwen3TTSModel.from_pretrained(
                model_id,
                device_map = device,
                dtype = torch.bfloat16,
                attn_implementation = "sdpa",
            )
    return models


def to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(wav, dtype = np.float32)).reshape(1, -1)
    if sr != OUTPUT_SR:
        tensor = torchaudio.functional.resample(waveform = tensor, orig_freq = sr, new_freq = OUTPUT_SR)
    return tensor.squeeze(0).numpy()


def max_new_tokens_for(batch: list) -> int:
    """按文本长度给生成步数设上限，防止个别片段收不住尾、整批生成到默认的 2048 步。

    编解码器每秒约 12.5 个 token；上限取「该语言平均语速下时长的 2 倍再加 3 秒」，按批内最长的片段计算。
    """
    longest = max(
        len(seg["text"]) / CHARS_PER_SECOND.get(seg["lang"], 10.0)
        for seg in batch
    )
    return int(math.ceil(CODEC_TOKENS_PER_SECOND * (2.0 * longest + 3.0)))


def synthesize_batch(models: dict, batch: list, clone_prompts: dict):
    """合成一批同模式的片段，返回（波形列表，模型原生采样率）。"""
    mode = batch[0]["mode"]
    texts = [seg["text"] for seg in batch]
    languages = [LANG_NAMES[seg["lang"]] for seg in batch]
    max_new_tokens = max_new_tokens_for(batch = batch)
    if mode == "builtin":
        return models["builtin"].generate_custom_voice(
            text = texts,
            language = languages,
            speaker = [seg["builtin_name"] for seg in batch],
            max_new_tokens = max_new_tokens,
        )
    prompts = []
    for seg in batch:
        key = (seg["ref_audio"], seg["ref_text"])
        if key not in clone_prompts:
            # 同一个参考音色只提取一次特征
            clone_prompts[key] = models["clone"].create_voice_clone_prompt(
                ref_audio = seg["ref_audio"],
                ref_text = seg["ref_text"],
            )[0]
        prompts.append(clone_prompts[key])
    return models["clone"].generate_voice_clone(
        text = texts,
        language = languages,
        voice_clone_prompt = prompts,
        max_new_tokens = max_new_tokens,
    )


def validate_job(job: dict) -> str:
    """返回错误说明；任务合法时返回空字符串。"""
    if job.get("lang") not in LANG_NAMES:
        return f"unsupported lang: {job.get('lang')}"
    if not job.get("text", "").strip():
        return "empty text"
    if job.get("mode") == "builtin":
        if job.get("builtin_name") not in BUILTIN_SPEAKERS:
            return f"unknown builtin_name: {job.get('builtin_name')}"
    elif job.get("mode") == "clone":
        if not job.get("ref_audio") or not job.get("ref_text"):
            return "clone mode needs ref_audio and ref_text"
        if not Path(job["ref_audio"]).exists():
            return f"ref_audio not found: {job['ref_audio']}"
    else:
        return f"unknown mode: {job.get('mode')}"
    return ""


def failed_row(job: dict, error: str) -> dict:
    return {**job, "audio_path": None, "duration_s": None, "status": "failed", "error": error}


def main():
    parser = argparse.ArgumentParser(description = "Qwen3-TTS 批量合成")
    parser.add_argument("--input", required = True, help = "合成任务 JSONL")
    parser.add_argument("--output-dir", required = True, help = "FLAC 与 results.jsonl 的输出目录")
    parser.add_argument("--batch-size", type = int, default = 8, help = "每批的句子片段数")
    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument("--device", default = "cuda:0")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents = True, exist_ok = True)
    with open(args.input, encoding = "utf-8") as f:
        jobs = [json.loads(line) for line in f if line.strip()]

    results = {}
    segments = []
    for job_index, job in enumerate(jobs):
        error = validate_job(job = job)
        if error:
            results[job_index] = failed_row(job = job, error = error)
            continue
        for part_index, text in enumerate(split_text(text = job["text"])):
            segments.append({**job, "text": text, "job_index": job_index, "part_index": part_index})

    models = load_models(modes = {seg["mode"] for seg in segments}, device = args.device)
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    clone_prompts = {}
    waves = {}
    start = time.time()
    # 按模式分组，组内按文本长度排序，减少批内补齐
    for mode in ("builtin", "clone"):
        group = sorted((s for s in segments if s["mode"] == mode), key = lambda s: len(s["text"]))
        for i in range(0, len(group), args.batch_size):
            batch = group[i : i + args.batch_size]
            try:
                wavs, sr = synthesize_batch(models = models, batch = batch, clone_prompts = clone_prompts)
                for seg, wav in zip(batch, wavs):
                    waves[(seg["job_index"], seg["part_index"])] = to_16k(wav = wav, sr = sr)
            except Exception:
                # 整批失败时逐条重试，定位出错的片段
                for seg in batch:
                    try:
                        wavs, sr = synthesize_batch(models = models, batch = [seg], clone_prompts = clone_prompts)
                        waves[(seg["job_index"], seg["part_index"])] = to_16k(wav = wavs[0], sr = sr)
                    except Exception as exc:
                        results[seg["job_index"]] = failed_row(
                            job = jobs[seg["job_index"]],
                            error = f"{type(exc).__name__}: {exc}",
                        )
    elapsed = time.time() - start

    total_audio = 0.0
    gap = np.zeros(int(SENTENCE_GAP_S * OUTPUT_SR), dtype = np.float32)
    for job_index, job in enumerate(jobs):
        if job_index in results:
            continue
        pieces = []
        for key in sorted(k for k in waves if k[0] == job_index):
            if pieces:
                pieces.append(gap)
            pieces.append(waves[key])
        audio = np.concatenate(pieces)
        path = out_dir / f"{job['sample_id']}.flac"
        sf.write(file = str(path), data = audio, samplerate = OUTPUT_SR, format = "FLAC")
        duration = len(audio) / OUTPUT_SR
        total_audio += duration
        results[job_index] = {
            **job,
            "audio_path": str(path),
            "duration_s": round(duration, 3),
            "status": "ok",
            "error": None,
        }

    with open(out_dir / "results.jsonl", "w", encoding = "utf-8") as f:
        for job_index in range(len(jobs)):
            f.write(json.dumps(results[job_index], ensure_ascii = False) + "\n")
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(
        f"jobs={len(jobs)} segments={len(segments)} batch_size={args.batch_size} "
        f"audio={total_audio:.1f}s elapsed={elapsed:.1f}s rtf_x={total_audio / max(elapsed, 1e-9):.2f} "
        f"peak_vram={peak_gb:.2f}GB"
    )


if __name__ == "__main__":
    main()
