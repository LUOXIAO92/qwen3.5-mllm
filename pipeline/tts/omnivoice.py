"""OmniVoice（k2-fsa/OmniVoice）TTS 封装：读取合成任务，批量克隆合成，输出 16 kHz 单声道 FLAC。

用法（在 envs/tts-omnivoice 环境中运行，GPU 命令需通过 flock 排队）::

    flock <工作目录>/gpu.lock python -m pipeline.tts.omnivoice \\
        --input jobs.jsonl --output-dir out/ --batch-size 16

支持的语言代码（项目语言代码 -> OmniVoice 语言 ID）::

    zh en ja ko de fr ru pt es it id ms th vi fil tr fa 与 OmniVoice 的 ID 相同；
    ar -> arb（OmniVoice 中的 Standard Arabic，直接传 ar 会不被识别而退化为不指定语言）

输入 JSONL，每行一个对象：

    - ``sample_id``：必填，输出文件名为 ``<sample_id>.flac``
    - ``text``：必填，已规范化的朗读稿，原样送入模型
    - ``lang``：必填，上述语言代码
    - ``ref_audio``、``ref_text``：克隆模式必填（参考音频路径与其文本）。
      两者都缺省时退化为 OmniVoice 的自动音色（随机音色），结果中 ``mode`` 记为 ``auto``
    - ``seed``：可选，整数。同一批内取第一条的种子，批与批之间互不影响
    - 其余字段（``entry_id``、``voice_id``、``ref_lang`` 等）原样带入结果

输出：``<output-dir>/<sample_id>.flac`` 和 ``<output-dir>/results.jsonl``。
结果行包含输入的全部字段，另加 ``audio_path``、``duration_s``、``tts_model``、``mode``、
``status``（``ok`` 或 ``failed``）、``error``。吞吐统计以一行 JSON 写到标准错误。

文本规范化：OmniVoice 的 ``generate`` 默认 ``normalize_text = False``，本封装在
``synthesize_batch`` 中仍显式传入 ``normalize_text = False``，保证模型不再改动朗读稿。

环境：Python 3.11，torch 2.8.0+cu128（RTX 5090 / sm_120 可用），omnivoice 0.2.1
（上游仓库 ``uv pip install`` 安装，transformers >= 5.3）。
显存与吞吐（RTX 5090，fp16，num_step = 32，36 条冒烟句）：批量 16 时实时倍率约 10.8，显存峰值 3.4 GB；批量 36 时约 9.0，5.2 GB。
权重许可：CC-BY-NC（非商用，训练数据含 Emilia）；代码为 Apache-2.0。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from pipeline.text.langs import CHARS_PER_SECOND

MODEL_NAME = "k2-fsa/OmniVoice"
TARGET_SR = 16000

SUPPORTED_LANGS = {
    "zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es",
    "it", "id", "ms", "th", "vi", "fil", "ar", "tr", "fa",
}


# 项目语言代码到 OmniVoice 语言 ID 的映射，只有阿拉伯语不同
LANG_MAP = {"ar": "arb"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "OmniVoice 批量克隆合成")
    parser.add_argument("--input", required = True, help = "合成任务 JSONL")
    parser.add_argument("--output-dir", required = True, help = "FLAC 与 results.jsonl 的输出目录")
    parser.add_argument("--batch-size", type = int, default = 16, help = "每批条数")
    parser.add_argument("--num-step", type = int, default = 32, help = "扩散步数（上游默认 32）")
    parser.add_argument("--seed", type = int, default = 0, help = "任务行没有 seed 时使用")
    parser.add_argument("--model", default = MODEL_NAME, help = "模型名或本地路径")
    parser.add_argument("--device", default = "cuda:0")
    return parser.parse_args()


def load_jobs(path: str) -> list[dict]:
    jobs = []
    with open(path, encoding = "utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                jobs.append(json.loads(line))
    return jobs


def to_flac_16k(wave, sr: int, path: Path) -> float:
    """24 kHz 的模型输出转成 16 kHz 单声道 FLAC，返回时长（秒）。"""
    tensor = torch.as_tensor(wave, dtype = torch.float32).reshape(1, -1)
    tensor = torchaudio.functional.resample(waveform = tensor, orig_freq = sr, new_freq = TARGET_SR)
    audio = tensor.squeeze(0).clamp(-1.0, 1.0).numpy()
    sf.write(file = str(path), data = audio, samplerate = TARGET_SR, format = "FLAC", subtype = "PCM_16")
    return len(audio) / TARGET_SR


def synthesize_batch(model, batch: list[dict], prompts: dict, num_step: int, seed: int) -> list:
    """合成一批。克隆行用缓存的音色 prompt；没有参考音频的行走自动音色。"""
    texts = [job["text"] for job in batch]
    langs = [LANG_MAP.get(job["lang"], job["lang"]) for job in batch]
    clone_prompts = [prompts.get((job.get("ref_audio"), job.get("ref_text"))) for job in batch]
    torch.manual_seed(seed)
    # 一批内克隆行与自动音色行不能混用同一次调用，调用方已按模式分批
    kwargs = dict(
        text = texts,
        language = langs,
        num_step = num_step,
        # 关闭模型自带的文本规范化：朗读稿已由我们自己规范化，模型不得再改动
        normalize_text = False,
    )
    if all(p is not None for p in clone_prompts):
        kwargs["voice_clone_prompt"] = clone_prompts
    return model.generate(**kwargs)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents = True, exist_ok = True)
    jobs = load_jobs(path = args.input)

    from omnivoice import OmniVoice

    model = OmniVoice.from_pretrained(
        args.model,
        device_map = args.device,
        dtype = torch.float16,
    )
    sr = model.sampling_rate
    torch.cuda.reset_peak_memory_stats()

    # 每个不同的（参考音频，参考文本）只编码一次
    prompts: dict = {}
    results: dict[str, dict] = {}
    todo = []
    for job in jobs:
        result = dict(job)
        result.update(tts_model = "omnivoice", audio_path = None, duration_s = None, status = "failed", error = None)
        results[job["sample_id"]] = result
        ref_audio, ref_text = job.get("ref_audio"), job.get("ref_text")
        if job["lang"] not in SUPPORTED_LANGS:
            result["error"] = f"不支持的语言代码：{job['lang']}"
            continue
        if bool(ref_audio) != bool(ref_text):
            result["error"] = "ref_audio 与 ref_text 必须同时给出"
            continue
        result["mode"] = "clone" if ref_audio else "auto"
        if ref_audio and (ref_audio, ref_text) not in prompts:
            try:
                prompts[(ref_audio, ref_text)] = model.create_voice_clone_prompt(
                    ref_audio = ref_audio,
                    ref_text = ref_text,
                )
            except Exception as exc:
                result["error"] = f"参考音频处理失败：{exc}"
                continue
        todo.append(job)

    # 按模式分组、再按估计的朗读时长排序（不同语言的字符数不可直接比较），减少批内填充
    todo.sort(key = lambda j: (bool(j.get("ref_audio")), len(j["text"]) / CHARS_PER_SECOND.get(j["lang"], 10.0)))
    batches = [todo[i : i + args.batch_size] for i in range(0, len(todo), args.batch_size)]

    audio_seconds = 0.0
    start = time.time()
    for batch in batches:
        # 模式不同的行不放进同一批
        groups = [[j for j in batch if bool(j.get("ref_audio")) == m] for m in (True, False)]
        for group in groups:
            if not group:
                continue
            seed = group[0].get("seed", args.seed)
            try:
                waves = synthesize_batch(model = model, batch = group, prompts = prompts, num_step = args.num_step, seed = seed)
            except Exception as exc:
                for job in group:
                    results[job["sample_id"]]["error"] = f"合成失败：{exc}"
                continue
            for job, wave in zip(group, waves):
                path = out_dir / f"{job['sample_id']}.flac"
                duration = to_flac_16k(wave = wave, sr = sr, path = path)
                result = results[job["sample_id"]]
                result.update(audio_path = str(path), duration_s = round(duration, 3), status = "ok")
                audio_seconds += duration
    torch.cuda.synchronize()
    elapsed = time.time() - start

    with open(out_dir / "results.jsonl", "w", encoding = "utf-8") as f:
        for job in jobs:
            f.write(json.dumps(results[job["sample_id"]], ensure_ascii = False) + "\n")

    stats = dict(
        n_jobs = len(jobs),
        n_ok = sum(r["status"] == "ok" for r in results.values()),
        batch_size = args.batch_size,
        num_step = args.num_step,
        audio_seconds = round(audio_seconds, 2),
        gen_seconds = round(elapsed, 2),
        realtime_factor = round(audio_seconds / elapsed, 2) if elapsed > 0 else None,
        peak_vram_gb = round(torch.cuda.max_memory_allocated() / 2**30, 2),
    )
    print(json.dumps(stats, ensure_ascii = False), file = sys.stderr)


if __name__ == "__main__":
    main()
