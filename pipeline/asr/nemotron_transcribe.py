"""Nemotron-3.5-Transcribe（NVFP4）封装：按指定语言逐字转写，输出带标点和大小写的文本。

模型：nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4（OpenMDW-1.1），Transformer ASR 编码器加 Mamba-Transformer
混合 MoE 语言模型，总参数 34B，每个 token 激活 4.3B。解码器为 NVFP4，Mamba 层、音频编码器和多 token 预测（MTP）头为 FP8，
只能在 Blackwell GPU（例如 RTX 5090）上运行。
支持的语言代码（项目语言码）：zh en ja ko de fr ru pt es it id th vi fil ar tr（16 种）。ms、fa 不在模型支持范围内，
这些样本的 hypothesis 为空并带 error 字段。

环境：envs/asr-nemotron（Python 3.12，vLLM 0.32.0 加 NeMo Speech main 分支的 vLLM 插件 nemo_speechlm），
用 vLLM 离线批量推理。建立方法和权重修订号见 issue #37。FlashInfer 第一次运行时编译量化 kernel（本机约 15 分钟），
需要 CUDA 13 工具包（在 PATH 上的 nvcc，或 CUDA_HOME，默认找 /usr/local/cuda）；编译缓存的位置可以用
VLLM_CACHE_ROOT、FLASHINFER_WORKSPACE_BASE 指定。

显存与吞吐（RTX 5090 32 GB，FLEURS dev 16 种语言各 50 句，共 9616 秒）：权重占 20.6 GiB，vLLM 按
--gpu-memory-utilization（默认 0.85）预先占好显存，其余 4.6 GiB 用作 KV 缓存，运行时整卡占用约 31 GB，
不能与其他 GPU 任务同时运行。引擎启动（读权重、编译缓存命中、自动调优）约 1–2 分钟。--batch-size 32 时转写
约 130–180 音频秒/墙钟秒；批量 64 反而降到约 40。vLLM 默认会把 FlashInfer 的自动调优结果存成文件、下次启动直接
读取，实测读取旧结果后只有约 34 音频秒/墙钟秒，所以本封装每次运行用新的临时目录重新调优。多 token 预测（MTP）
投机解码在这里不提速，默认关闭（--spec-tokens 0）。
同一输入重复运行，输出并不完全相同：任意两次运行之间约三分之一的句子有差别（多为个别词，少数只差标点），
9 次运行的宏平均错误率在 5.86%–6.27% 之间。

提示词：系统提示留空，用户消息为下面的指令加一段音频，关闭思考模式，贪心解码。默认写法 written 取自说明卡评测 FLEURS
时用的指令，只把 "as spoken" 换成指定语言，要求逐字（保留口头语和重复）、带标点和大小写、数字用标准书写形式；
中文另外指定简体字。另一种写法 verbatim 是说明卡推荐的通用逐字转写指令，不提数字和书写系统：

    written:  Audio transcription. Language: <语言>, in its standard writing system. Wording: verbatim, including fillers
              and repetitions. Formatting: punctuation and capitalization; numbers in their standard written form.
              Output: transcript only.
    verbatim: Audio transcription. Language: <语言>. Wording: verbatim, including fillers and repetitions.
              Formatting: punctuation and capitalization. Output: transcript only.

两种写法在 FLEURS 上的错误率相同（同样设置下宏平均都是 5.86%），区别在数字：原文写成阿拉伯数字的 129 句里，written 有 51 句
输出阿拉伯数字，verbatim 只有 6 句，其余都拼成词（偶尔拼错语言，例如德语句子里出现中文数字）。训练目标沿用数据集
原文的书面写法，所以默认用 written。

不对输出做任何文字规范化，也不套用说明卡里去除循环重复的后处理；只去掉可能出现的空思考块并去掉首尾空白。
模型在无人声的短片段上会编造内容（3 秒的静音、噪声片段大多输出一句话），下游需要另行过滤。

用法::

    python -m pipeline.asr.nemotron_transcribe --input audio.jsonl --output results.jsonl --batch-size 32 \
        [--model nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4 或本地目录] [--audio-root DIR] [--prompt-style written]

输入每行至少有 sample_id、audio_path、lang；audio_path 是相对路径时相对 --audio-root，没有给 --audio-root 时
相对输入文件所在目录。输出每行含 sample_id、lang、asr_model、hypothesis；失败的样本 hypothesis 为空字符串并多一个
error 字段。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4"
ASR_MODEL_NAME = "nemotron-3.5-transcribe-nvfp4"
SAMPLE_RATE = 16000
# 编码器每个音频 token 约对应 80 毫秒，用于估计一段音频占多少上下文
SECONDS_PER_AUDIO_TOKEN = 0.08
# 提示词与对话模板本身的 token 数上限（实际约 60）
PROMPT_TOKEN_BUDGET = 128

# 项目语言码 → 提示词里的语言名（说明卡语言表的写法）
LANG_NAMES = {
    "zh": "Mandarin Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
    "id": "Indonesian",
    "th": "Thai",
    "vi": "Vietnamese",
    "fil": "Filipino (Tagalog)",
    "ar": "Arabic",
    "tr": "Turkish",
}
SUPPORTED_LANGS = set(LANG_NAMES)
# 书写系统：中文指定简体字，其余语言用各自的标准书写系统
SCRIPTS = {"zh": "in simplified Chinese characters"}
DEFAULT_SCRIPT = "in its standard writing system"

PROMPT_TEMPLATES = {
    "written": (
        "Audio transcription. Language: {language}, {script}. "
        "Wording: verbatim, including fillers and repetitions. "
        "Formatting: punctuation and capitalization; numbers in their standard written form. "
        "Output: transcript only."
    ),
    "verbatim": (
        "Audio transcription. Language: {language}. "
        "Wording: verbatim, including fillers and repetitions. "
        "Formatting: punctuation and capitalization. "
        "Output: transcript only."
    ),
}

THINK_RE = re.compile(pattern = r"<think>.*?</think>", flags = re.DOTALL)


def build_instruction(lang: str, style: str) -> str:
    return PROMPT_TEMPLATES[style].format(
        language = LANG_NAMES[lang],
        script = SCRIPTS.get(lang, DEFAULT_SCRIPT),
    )


def load_audio(path: Path) -> np.ndarray:
    """读取音频，转为 16 kHz 单声道 float32。"""
    data, sr = sf.read(file = str(path), dtype = "float32", always_2d = True)
    data = data.mean(axis = 1)
    if sr != SAMPLE_RATE:
        import librosa  # NeMo 的依赖，环境中已有

        data = librosa.resample(y = data, orig_sr = sr, target_sr = SAMPLE_RATE)
    return np.ascontiguousarray(a = data, dtype = np.float32)


def read_jsonl(path: Path) -> list[dict]:
    with open(file = path, mode = "r", encoding = "utf-8") as f:
        return [json.loads(s = line) for line in f if line.strip()]


class GpuMemoryMonitor:
    """在后台线程里定时读取整张 GPU 的已用显存，记录峰值。

    vLLM 的推理在单独的引擎进程里进行，本进程的 torch 显存统计看不到，所以用 NVML 读整卡的用量。
    """

    def __init__(self, interval: float = 0.2) -> None:
        import pynvml

        pynvml.nvmlInit()
        self._nvml = pynvml
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(index = 0)
        self._interval = interval
        self._stop = threading.Event()
        self.peak_bytes = 0
        self._thread = threading.Thread(target = self._run, daemon = True)

    def _run(self) -> None:
        while not self._stop.is_set():
            used = self._nvml.nvmlDeviceGetMemoryInfo(handle = self._handle).used
            self.peak_bytes = max(self.peak_bytes, used)
            time.sleep(self._interval)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()


def transcribe(args: argparse.Namespace, langs: list[str], audios: list[np.ndarray], max_audio_seconds: float) -> dict:
    """启动 vLLM 引擎，按各条的语言转写一批音频。返回转写文本、达到长度上限的条数、加载与转写用时和显存峰值。"""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = args.model, trust_remote_code = True)

    def render(lang: str) -> str:
        """按模型自带的对话模板展开提示；<|audio|> 是音频的占位符，必须放在指令末尾。"""
        instruction = build_instruction(lang = lang, style = args.prompt_style)
        messages = [{"role": "user", "content": instruction + " <|audio|>"}]
        return tokenizer.apply_chat_template(
            conversation = messages,
            add_generation_prompt = True,
            tokenize = False,
            enable_thinking = False,
        )

    llm_kwargs = dict(
        model = args.model,
        trust_remote_code = True,
        dtype = "bfloat16",
        kv_cache_dtype = "fp8",  # 导出的检查点没有声明 KV 缓存格式，说明卡要求显式指定 fp8
        gpu_memory_utilization = args.gpu_memory_utilization,
        max_model_len = args.max_model_len,
        max_num_seqs = args.batch_size,
        max_num_batched_tokens = 4096,
        enable_chunked_prefill = True,
        enable_prefix_caching = False,
        mm_processor_cache_gb = 0,
        async_scheduling = False,
        limit_mm_per_prompt = {"audio": {"count": 1, "length": int(max_audio_seconds * SAMPLE_RATE)}},
        hf_overrides = {"ctc_timestamps": {"enabled": False}},  # 不需要词级时间戳，不加载对齐头
    )
    if args.spec_tokens > 0:
        # 混合（Mamba）主干上的投机解码要用 ReplaySSM，并关闭 Mamba 前缀缓存
        llm_kwargs["speculative_config"] = {"method": "mtp", "num_speculative_tokens": args.spec_tokens}
        llm_kwargs.update(use_replayssm = True, mamba_backend = "flashinfer", mamba_cache_mode = "none")

    t_load = time.time()
    llm = LLM(**llm_kwargs)
    load_seconds = time.time() - t_load
    sampling = SamplingParams(temperature = 0.0, max_tokens = args.max_new_tokens)

    # 每条请求带各自语言的提示词，全部一次交给 vLLM，由它按 --batch-size 连续成批调度
    prompts = {lang: render(lang = lang) for lang in set(langs)}
    reqs = [
        {"prompt": prompts[lang], "multi_modal_data": {"audio": (audio, SAMPLE_RATE)}}
        for lang, audio in zip(langs, audios)
    ]
    monitor = GpuMemoryMonitor()
    monitor.start()
    t0 = time.time()
    try:
        outputs = llm.generate(prompts = reqs, sampling_params = sampling, use_tqdm = False)
    finally:
        wall = time.time() - t0
        monitor.stop()
    completions = [out.outputs[0] for out in outputs]
    return {
        "texts": [THINK_RE.sub(repl = "", string = c.text).strip() for c in completions],
        "n_truncated": sum(c.finish_reason == "length" for c in completions),
        "load_seconds": load_seconds,
        "wall": wall,
        "peak_gb": monitor.peak_bytes / 1024**3,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description = "Nemotron-3.5-Transcribe（NVFP4）批量转写")
    parser.add_argument("--input", required = True, help = "含 sample_id、audio_path、lang 的 JSONL")
    parser.add_argument("--output", required = True, help = "识别结果 JSONL")
    parser.add_argument("--batch-size", type = int, default = 32, help = "vLLM 同时解码的最大条数（max_num_seqs）")
    parser.add_argument("--model", default = DEFAULT_MODEL, help = "模型 ID 或本地目录")
    parser.add_argument("--audio-root", default = None, help = "相对 audio_path 的根目录，默认为输入文件所在目录")
    parser.add_argument("--prompt-style", default = "written", choices = sorted(PROMPT_TEMPLATES), help = "提示词写法")
    parser.add_argument("--max-new-tokens", type = int, default = 512)
    parser.add_argument("--max-model-len", type = int, default = 2048, help = "单条请求的上下文长度上限")
    parser.add_argument("--gpu-memory-utilization", type = float, default = 0.85)
    parser.add_argument(
        "--spec-tokens",
        type = int,
        default = 0,
        help = "MTP 投机解码每步提议的 token 数，0 为关闭（默认，实测不提速）",
    )
    args = parser.parse_args()

    # 说明卡要求的运行设置，必须在导入 vLLM 之前设好
    os.environ.setdefault("VLLM_PLUGINS", "nemo_speechlm")
    os.environ.setdefault("VLLM_DISABLED_KERNELS", "FlashInferFP8ScaledMMLinearKernel")
    # FlashInfer 编译 kernel 时调用 ninja，它装在虚拟环境的 bin 里；直接用环境的 python 运行（没有激活环境）时也要找得到
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    # vLLM 默认把 FlashInfer 的自动调优结果存成文件，下次启动直接读取；实测读取旧结果后转写慢约 4 倍，
    # 所以每次运行用一个新的临时目录，启动时重新调优（多用约 30 秒），进程结束后删除
    autotune_dir = None
    if "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR" not in os.environ:
        autotune_dir = tempfile.TemporaryDirectory(prefix = "nemotron-autotune-")
        os.environ["VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"] = autotune_dir.name

    input_path = Path(args.input)
    audio_root = Path(args.audio_root) if args.audio_root else input_path.resolve().parent
    rows = read_jsonl(path = input_path)
    # 音频 token、提示词和输出都要放进 max_model_len，超出的音频直接标记失败，避免整批请求被 vLLM 拒绝
    max_audio_seconds = (args.max_model_len - args.max_new_tokens - PROMPT_TOKEN_BUDGET) * SECONDS_PER_AUDIO_TOKEN

    audios: dict[int, np.ndarray] = {}
    errors: dict[int, str] = {}
    for i, row in enumerate(rows):
        if row["lang"] not in SUPPORTED_LANGS:
            errors[i] = f"unsupported_lang: {row['lang']}"
            continue
        p = Path(row["audio_path"])
        try:
            audio = load_audio(path = p if p.is_absolute() else audio_root / p)
        except Exception as e:  # 读取失败只标记该条
            errors[i] = f"load_failed: {e}"
            continue
        if len(audio) / SAMPLE_RATE > max_audio_seconds:
            errors[i] = f"too_long: {len(audio) / SAMPLE_RATE:.1f}s > {max_audio_seconds:.1f}s"
            continue
        audios[i] = audio

    # 请求按语言、时长排序，输出仍按输入顺序写出；没有可转写的音频时不启动引擎
    order = sorted(audios, key = lambda i: (rows[i]["lang"], len(audios[i])))
    audio_seconds = sum(len(audios[i]) for i in order) / SAMPLE_RATE
    hyps: dict[int, str] = {}
    stats = {"n_truncated": 0, "load_seconds": 0.0, "wall": 0.0, "peak_gb": 0.0}
    if order:
        try:
            result = transcribe(
                args = args,
                langs = [rows[i]["lang"] for i in order],
                audios = [audios[i] for i in order],
                max_audio_seconds = max_audio_seconds,
            )
            hyps = dict(zip(order, result.pop("texts")))
            stats = result
        except Exception as e:
            for i in order:
                errors[i] = f"generate_failed: {e}"

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

    print(
        f"样本 {len(rows)} 条，失败 {len(errors)} 条，达到 max_new_tokens 的 {stats['n_truncated']} 条，"
        f"音频 {audio_seconds:.1f} 秒，加载 {stats['load_seconds']:.1f} 秒，转写墙钟 {stats['wall']:.1f} 秒，"
        f"速度 {audio_seconds / max(stats['wall'], 1e-9):.1f} 音频秒/墙钟秒，整卡显存峰值 {stats['peak_gb']:.2f} GB",
        file = sys.stderr,
    )
    if autotune_dir is not None:
        autotune_dir.cleanup()


if __name__ == "__main__":
    main()
