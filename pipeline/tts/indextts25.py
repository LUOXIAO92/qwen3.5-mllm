"""IndexTTS-2.5 封装（克隆模式）。

用途：读取合成任务 JSONL，用参考音频克隆音色，逐条合成语音，
写出 16 kHz 单声道 FLAC 和结果 JSONL。v1 只用于阿拉伯语，封装支持上游的 5 种语言。

支持的语言代码（上游 ``lang`` 参数，不区分大小写）：
    zh、en、ja、es、ar，直接按本项目的语言代码传给上游，无需映射。
    其他语言本封装不接受。

环境：``envs/tts-indextts25``（Python 3.11，torch 2.8.* + cu128，上游 uv.lock 原样安装；
cu128 构建带有 sm_120 内核，RTX 5090 无需换版本）。上游仓库
（https://github.com/index-tts/index-tts）需要克隆到本仓库之外，并在环境中安装。
权重用 ``hf download IndexTeam/IndexTTS-2.5 --local-dir <模型目录>`` 下载，
首次运行时上游会把辅助模型（w2v-bert-2.0、campplus、bigvgan）下载到 ``<模型目录>/hf_cache``。

用法（GPU 命令需通过 flock 排队）::

    python -m pipeline.tts.indextts25 \\
        --input jobs.jsonl --output-dir out --batch-size 4 --model-dir <模型目录>

任务 JSONL 每行字段：``sample_id``、``lang``、``text``、``ref_audio``（参考音频路径，必需），
可选 ``seed``。结果 JSONL（``<output-dir>/results.jsonl``）字段：``sample_id``、``lang``、
``text``、``tts_model``、``mode``（固定 ``clone``）、``ref_audio``、``seed``、``audio_path``、
``duration_s``、``status``（``ok`` 或 ``error``）、``error``。

批量说明：上游 ``infer`` 逐条自回归生成（内部 ``autoregressive_batch_size = 1``），
没有批量接口。``--batch-size`` 控制一次取多少条任务做分组：同一个参考音频的任务排在一起，
上游会缓存参考音频的特征，换参考音频才重新计算。模型只加载一次。

文本规范化：上游 ``infer`` 的 ``text_normalization`` 默认为 True，本封装在 ``synthesize`` 中
固定传 ``text_normalization = False``（见该处标注）。上游的字符清洗（``clean_pattern`` 替换特殊符号）、
zh/en/ja 转小写、es 转大写、发音标注解析、日语假名转换不受该参数控制，仍会执行；
阿拉伯语只经过字符清洗，文字本身不改动。

输出采样率：模型原生 22.05 kHz，封装用 torchaudio 重采样到 16 kHz，保存为单声道 FLAC。

许可：bilibili 模型许可，输出只能用于改进 IndexTTS 自身或非商用模型；本项目为纯开源非商用用途，
合成数据不对外发布。

显存和吞吐（RTX 5090，bf16，逐条生成，含首条预热，其间可能有其他进程共用显卡）：
10 句短句（每句约 3-4 秒）共 35.1 s 音频，合成 33.3 s，实时倍率约 1.05 倍，显存峰值 5.29 GB。
运行结束时会在 stderr 打印同样的统计（实时倍率 = 音频时长 / 合成耗时）。
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

SUPPORTED_LANGS = ("zh", "en", "ja", "es", "ar")
OUTPUT_SR = 16000


def load_jobs(path: Path) -> list:
    """读取任务 JSONL。"""
    jobs = []
    with open(file = path, mode = "r", encoding = "utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                jobs.append(json.loads(line))
    return jobs


def load_model(model_dir: Path, use_bf16: bool):
    """加载 IndexTTS-2.5，只加载一次。"""
    from indextts.infer_v2_5 import IndexTTS2

    return IndexTTS2(
        cfg_path = str(model_dir / "config.yaml"),
        model_dir = str(model_dir),
        use_bf16 = use_bf16,
        use_cuda_kernel = False,
        use_qwen_emo = False,
    )


def set_seed(seed: int) -> None:
    """固定随机种子，使同一任务可以复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def synthesize(tts, job: dict) -> np.ndarray:
    """合成一条，返回 16 kHz 单声道 float32 波形。"""
    if job["lang"].lower() not in SUPPORTED_LANGS:
        raise ValueError(f"不支持的语言：{job['lang']}")
    result = tts.infer(
        spk_audio_prompt = job["ref_audio"],
        text = job["text"],
        output_path = None,
        lang = job["lang"].lower(),
        # 关闭上游的文本规范化：朗读稿已由本项目自行规范化
        text_normalization = False,
        verbose = False,
    )
    if result is None:
        raise RuntimeError("上游没有返回音频")
    sr, wav = result  # int16，形状 (采样点, 声道)
    wav = torch.from_numpy(wav.astype(np.float32) / 32768.0).T.mean(dim = 0, keepdim = True)
    wav = torchaudio.functional.resample(waveform = wav, orig_freq = sr, new_freq = OUTPUT_SR)
    return wav.squeeze(0).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description = "IndexTTS-2.5 克隆合成")
    parser.add_argument("--input", type = Path, required = True, help = "合成任务 JSONL")
    parser.add_argument("--output-dir", type = Path, required = True, help = "输出目录")
    parser.add_argument("--batch-size", type = int, default = 4, help = "每组任务数（组内按参考音频排序）")
    parser.add_argument("--model-dir", type = Path, required = True, help = "IndexTTS-2.5 权重目录")
    parser.add_argument("--fp32", action = "store_true", help = "改用 fp32（默认 bf16）")
    args = parser.parse_args()

    args.output_dir.mkdir(parents = True, exist_ok = True)
    jobs = load_jobs(path = args.input)
    tts = load_model(model_dir = args.model_dir, use_bf16 = not args.fp32)
    torch.cuda.reset_peak_memory_stats()

    total_wall = 0.0
    total_audio = 0.0
    n_ok = 0
    results_path = args.output_dir / "results.jsonl"
    with open(file = results_path, mode = "w", encoding = "utf-8") as out:
        for start in range(0, len(jobs), args.batch_size):
            group = jobs[start:start + args.batch_size]
            # 同一个参考音频的任务挨在一起，利用上游的参考特征缓存
            group.sort(key = lambda j: j["ref_audio"])
            for job in group:
                seed = job.get("seed", 0)
                row = {
                    "sample_id": job["sample_id"],
                    "lang": job["lang"],
                    "text": job["text"],
                    "tts_model": "indextts-2.5",
                    "mode": "clone",
                    "ref_audio": job["ref_audio"],
                    "seed": seed,
                    "audio_path": None,
                    "duration_s": None,
                    "status": "ok",
                    "error": None,
                }
                try:
                    set_seed(seed = seed)
                    t0 = time.perf_counter()
                    wav = synthesize(tts = tts, job = job)
                    torch.cuda.synchronize()
                    wall = time.perf_counter() - t0
                    audio_path = args.output_dir / f"{job['sample_id']}.flac"
                    sf.write(file = str(audio_path), data = wav, samplerate = OUTPUT_SR, format = "FLAC")
                    row["audio_path"] = str(audio_path)
                    row["duration_s"] = round(len(wav) / OUTPUT_SR, 3)
                    total_wall += wall
                    total_audio += len(wav) / OUTPUT_SR
                    n_ok += 1
                except Exception as e:  # 单条失败不影响其余任务
                    row["status"] = "error"
                    row["error"] = repr(e)
                out.write(json.dumps(row, ensure_ascii = False) + "\n")
                out.flush()

    peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
    speed = total_audio / total_wall if total_wall > 0 else float("nan")
    print(
        f"成功 {n_ok}/{len(jobs)} 条，音频 {total_audio:.1f} s，合成 {total_wall:.1f} s，"
        f"实时倍率 {speed:.2f}x，显存峰值 {peak_gb:.2f} GB",
        file = sys.stderr,
    )


if __name__ == "__main__":
    main()
