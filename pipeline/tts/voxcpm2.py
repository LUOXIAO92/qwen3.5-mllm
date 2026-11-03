"""VoxCPM2（openbmb/VoxCPM2）TTS 封装：读取合成任务，克隆合成，输出 16 kHz 单声道 FLAC。

用法（在 envs/tts-voxcpm2 环境中运行，GPU 命令需通过 flock 排队）::

    flock <工作目录>/gpu.lock python -m pipeline.tts.voxcpm2 \\
        --input jobs.jsonl --output-dir out/ --batch-size 8

支持的语言代码（VoxCPM2 官方 30 种语言中与本项目重合的 17 种，无需语言标签，无需映射）::

    zh en ja ko de fr ru pt es it id ms th vi fil ar tr

不支持 ``fa``（波斯语）：这类任务行不会送入模型，结果中 ``status = "skipped"``。
``fil`` 对应上游的 Tagalog，模型直接按文本识别语言，不需要传语言参数。

输入 JSONL，每行一个对象：

    - ``sample_id``：必填，输出文件名为 ``<sample_id>.flac``
    - ``text``：必填，已规范化的朗读稿，原样送入模型
    - ``lang``：必填，上述语言代码
    - ``ref_audio``：必填，参考音频路径（克隆音色）
    - ``ref_text``：可选，参考音频的文本。给出时用上游的 Ultimate Cloning（提示音频续写加
      参考音频，音色最像）；缺省时用 Controllable Cloning（只用参考音频提取音色）
    - ``seed``：可选，整数，每条独立设置随机种子
    - 其余字段（``entry_id``、``voice_id``、``ref_lang`` 等）原样带入结果

输出：``<output-dir>/<sample_id>.flac`` 和 ``<output-dir>/results.jsonl``。
结果行包含输入的全部字段，另加 ``audio_path``、``duration_s``、``tts_model``、``mode``
（固定为 ``clone``）、``status``（``ok`` / ``failed`` / ``skipped``）、``error``。
吞吐统计以一行 JSON 写到标准错误。

文本规范化：上游 ``VoxCPM.generate`` 的 ``normalize`` 默认 ``False``，且上游在该入口里会把
换行和连续空白压成一个空格。本封装绕开 ``VoxCPM.generate``，直接调用
``tts_model.generate_with_prompt_cache``（见 ``synthesize``），文本一个字符都不改动，
tokenizer 之外没有任何改写。注意一个模型行为：上游把文本开头的小括号内容
``(...)`` 当作音色描述（Voice Design）。以 ``(`` 或 ``（`` 开头的朗读稿会被模型当作
指令而不是朗读内容，这类任务行标记为 ``failed``，不送入模型。

批量：上游 PyTorch 实现一次只合成一条，``--batch-size`` 只控制每批写盘与日志的粒度；
同一参考音频的编码结果在整个任务内复用。Nano-vLLM 需要 flash-attn（要在 sm_120 上用
nvcc 从源码编译，且与 torch 2.11 配套），本版本未采用。

环境：Python 3.12，torch 2.11.0+cu128（RTX 5090 / sm_120 可用），voxcpm 上游 main（提交 cce58a5b5，PyPI 的 2.0.3 没有 seed 参数，需从仓库安装）。
显存与吞吐：见 GitHub issue #10 的实测记录。
权重与代码许可：Apache-2.0。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

MODEL_NAME = "openbmb/VoxCPM2"
TARGET_SR = 16000

SUPPORTED_LANGS = {
    "zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es",
    "it", "id", "ms", "th", "vi", "fil", "ar", "tr",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "VoxCPM2 克隆合成")
    parser.add_argument("--input", required = True, help = "合成任务 JSONL")
    parser.add_argument("--output-dir", required = True, help = "FLAC 与 results.jsonl 的输出目录")
    parser.add_argument("--batch-size", type = int, default = 8, help = "每批条数（只影响写盘与日志粒度）")
    parser.add_argument("--cfg-value", type = float, default = 2.0, help = "引导强度（上游默认 2.0）")
    parser.add_argument("--timesteps", type = int, default = 10, help = "扩散步数（上游默认 10）")
    parser.add_argument("--seed", type = int, default = 0, help = "任务行没有 seed 时使用")
    parser.add_argument("--model", default = MODEL_NAME, help = "模型名或本地路径")
    parser.add_argument("--device", default = "cuda", help = "上游只在 device == \"cuda\" 时启用 torch.compile")
    parser.add_argument("--no-optimize", action = "store_true", help = "关闭 torch.compile")
    return parser.parse_args()


def load_model(model_name: str, device: str, optimize: bool):
    """加载 VoxCPM2，不加载降噪器（参考音频不做降噪）。"""
    from voxcpm import VoxCPM

    return VoxCPM.from_pretrained(
        hf_model_id = model_name,
        load_denoiser = False,
        optimize = optimize,
        device = device,
    )


def get_prompt_cache(model, caches: dict, ref_audio: str, ref_text: str | None):
    """同一参考音频（和参考文本）只编码一次。"""
    key = (ref_audio, ref_text)
    if key not in caches:
        if ref_text:
            # Ultimate Cloning：参考音频同时作提示音频和参考音频
            caches[key] = model.tts_model.build_prompt_cache(
                prompt_text = ref_text,
                prompt_wav_path = ref_audio,
                reference_wav_path = ref_audio,
            )
        else:
            caches[key] = model.tts_model.build_prompt_cache(reference_wav_path = ref_audio)
    return caches[key]


def synthesize(model, text: str, prompt_cache: dict, seed: int, cfg_value: float, timesteps: int):
    """合成一条，返回 16 kHz 单声道的 float32 numpy 数组。

    文本规范化关闭的位置：这里直接调用 tts_model.generate_with_prompt_cache，
    不经过 VoxCPM.generate（其 normalize 参数和换行/空白压缩都在 generate 里）。
    text 原样进入 tokenizer。
    """
    wav, _, _ = model.tts_model.generate_with_prompt_cache(
        target_text = text,
        prompt_cache = prompt_cache,
        inference_timesteps = timesteps,
        cfg_value = cfg_value,
        retry_badcase = True,
        seed = seed,
    )
    wav = wav.detach().float().cpu()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    wav = wav.mean(dim = 0, keepdim = True)
    wav = torchaudio.functional.resample(
        waveform = wav,
        orig_freq = model.tts_model.sample_rate,
        new_freq = TARGET_SR,
    )
    return wav.squeeze(0).numpy()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents = True, exist_ok = True)

    jobs = [json.loads(line) for line in Path(args.input).read_text(encoding = "utf-8").splitlines() if line.strip()]

    results = []
    todo = []
    for job in jobs:
        result = dict(job)
        result.update(tts_model = "voxcpm2", mode = "clone", audio_path = None, duration_s = None, status = "failed", error = None)
        text = job.get("text") or ""
        if job.get("lang") not in SUPPORTED_LANGS:
            result.update(status = "skipped", error = f"VoxCPM2 不支持语言 {job.get('lang')}")
        elif not job.get("ref_audio"):
            result["error"] = "缺少 ref_audio（本封装只支持克隆模式）"
        elif not text.strip():
            result["error"] = "text 为空"
        elif text.lstrip().startswith(("(", "（")):
            result["error"] = "文本以小括号开头，会被模型当作音色描述，未合成"
        else:
            todo.append(result)
        results.append(result)

    model = load_model(model_name = args.model, device = args.device, optimize = not args.no_optimize)
    caches: dict = {}
    torch.cuda.reset_peak_memory_stats()

    total_audio = 0.0
    start = time.time()
    for begin in range(0, len(todo), args.batch_size):
        for result in todo[begin:begin + args.batch_size]:
            try:
                cache = get_prompt_cache(
                    model = model,
                    caches = caches,
                    ref_audio = result["ref_audio"],
                    ref_text = result.get("ref_text"),
                )
                audio = synthesize(
                    model = model,
                    text = result["text"],
                    prompt_cache = cache,
                    seed = int(result.get("seed", args.seed)),
                    cfg_value = args.cfg_value,
                    timesteps = args.timesteps,
                )
                path = out_dir / f"{result['sample_id']}.flac"
                sf.write(file = str(path), data = audio, samplerate = TARGET_SR, format = "FLAC")
                duration = len(audio) / TARGET_SR
                total_audio += duration
                result.update(audio_path = str(path), duration_s = round(duration, 3), status = "ok")
            except Exception as exc:  # 单条失败不影响其余任务
                result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[voxcpm2] {min(begin + args.batch_size, len(todo))}/{len(todo)}", file = sys.stderr)

    elapsed = time.time() - start
    with open(out_dir / "results.jsonl", "w", encoding = "utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii = False) + "\n")

    stats = dict(
        n_jobs = len(jobs),
        n_ok = sum(r["status"] == "ok" for r in results),
        n_skipped = sum(r["status"] == "skipped" for r in results),
        n_failed = sum(r["status"] == "failed" for r in results),
        batch_size = args.batch_size,
        audio_s = round(total_audio, 2),
        elapsed_s = round(elapsed, 2),
        realtime_x = round(total_audio / elapsed, 2) if elapsed else None,
        peak_vram_gb = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
    )
    print(json.dumps(stats, ensure_ascii = False), file = sys.stderr)


if __name__ == "__main__":
    main()
