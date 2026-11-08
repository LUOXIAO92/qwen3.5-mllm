"""Fun-ASR-MLT-Nano-2512 封装：按指定语言转写，输出模型原始文本。

模型：FunAudioLLM/Fun-ASR-MLT-Nano-2512（0.8B，音频编码器 + Qwen3-0.6B 解码器，许可 Apache-2.0）。
支持的语言代码（项目语言码）：zh en ja ko pt id ms th vi fil ar（共 11 种）。
其余项目语言（de fr ru es it tr fa）模型不支持，遇到时该条失败并写 error。
环境：envs/asr-funasr-mlt（基于 conda 环境 py3.14 的虚拟环境，额外安装 funasr 1.4.24，
transformers 随之升到 5.22.2）。推理代码 model.py 来自上游仓库 FunAudioLLM/Fun-ASR，
通过 --funasr-src 指向它的克隆目录（建立方法见 issue #34）。
显存、吞吐：见 issue #34 的验收评论。

用法：
    python -m pipeline.asr.fun_asr_mlt --input audio.jsonl --output results.jsonl --batch-size 1 \
        --funasr-src Fun-ASR 克隆目录 [--model FunAudioLLM/Fun-ASR-MLT-Nano-2512] [--audio-root 音频根目录]

输入每行至少有 sample_id、audio_path、lang。audio_path 是相对路径时，相对 --audio-root 解析；
没有给 --audio-root 时，相对输入文件所在目录。
输出每行：sample_id、lang、asr_model、hypothesis（模型原始文本，不做任何规范化）。转写失败的样本
hypothesis 为空字符串，并多一个 error 字段。

语言与文本处理：
- 语言通过提示词「语音转写成<语言名>」指定，语言名用中文（中文、英文、日文、韩文、葡萄牙语、
  印尼语、马来语、泰语、越南语、菲律宾语、阿拉伯语）。
- itn = False 时提示词末尾加「，不进行文本规整」，模型输出不做反向文本规范化，数字多保持口语词形。
  本封装固定 itn = False；输出取 funasr 返回的 text 字段，只做该模型自带的空白合并。
- 解码用贪心（do_sample = False）。上游 inference 每次只解码批内第 1 条，所以实际批大小为 1，
  --batch-size 只为与其他封装保持同一命令行，传入的值被忽略。
- 解码器用 bf16，编码器保持 fp32。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

DEFAULT_MODEL = "FunAudioLLM/Fun-ASR-MLT-Nano-2512"
ASR_MODEL_NAME = "fun-asr-mlt-nano-2512"

# 项目语言码 -> 提示词中的语言名
LANG_NAMES = {
    "zh": "中文",
    "en": "英文",
    "ja": "日文",
    "ko": "韩文",
    "pt": "葡萄牙语",
    "id": "印尼语",
    "ms": "马来语",
    "th": "泰语",
    "vi": "越南语",
    "fil": "菲律宾语",
    "ar": "阿拉伯语",
}


def read_jsonl(path: Path) -> list[dict]:
    with open(file = path, mode = "r", encoding = "utf-8") as f:
        return [json.loads(s = line) for line in f if line.strip()]


def resolve_audio_path(audio_path: str, audio_root: Path) -> Path:
    p = Path(audio_path)
    return p if p.is_absolute() else audio_root / p


def main() -> None:
    parser = argparse.ArgumentParser(description = "Fun-ASR-MLT-Nano-2512 批量转写")
    parser.add_argument("--input", required = True, help = "含 sample_id、audio_path、lang 的 JSONL")
    parser.add_argument("--output", required = True, help = "识别结果 JSONL")
    parser.add_argument("--batch-size", type = int, default = 1, help = "仅为接口一致，实际批大小为 1")
    parser.add_argument("--model", default = DEFAULT_MODEL, help = "模型 ID 或本地目录")
    parser.add_argument("--audio-root", default = None, help = "相对 audio_path 的根目录，默认为输入文件所在目录")
    parser.add_argument("--funasr-src", required = True, help = "上游 Fun-ASR 仓库克隆目录（含 model.py）")
    parser.add_argument("--max-new-tokens", type = int, default = 512)
    args = parser.parse_args()

    src = Path(args.funasr_src).resolve()
    sys.path.insert(0, str(src))  # model.py 会 import 同目录的 ctc、tools
    from funasr import AutoModel

    input_path = Path(args.input)
    audio_root = Path(args.audio_root) if args.audio_root else input_path.resolve().parent
    rows = read_jsonl(path = input_path)

    model_kwargs = dict(
        model = args.model,
        trust_remote_code = True,
        remote_code = str(src / "model.py"),
        device = "cuda:0",
        disable_update = True,
    )
    if not Path(args.model).exists():
        model_kwargs["hub"] = "hf"
    model = AutoModel(**model_kwargs)

    hyps: dict[int, str] = {}
    errors: dict[int, str] = {}
    audio_seconds = 0.0
    torch.cuda.synchronize()
    t0 = time.time()
    for i, row in enumerate(rows):
        lang = row["lang"]
        if lang not in LANG_NAMES:
            errors[i] = f"unsupported_lang: {lang}"
            continue
        path = resolve_audio_path(audio_path = row["audio_path"], audio_root = audio_root)
        try:
            res = model.generate(
                input = [str(path)],
                cache = {},
                batch_size = 1,
                language = LANG_NAMES[lang],
                itn = False,
                llm_dtype = "bf16",
                max_length = args.max_new_tokens,
                llm_kwargs = {"do_sample": False},
                disable_pbar = True,
            )
            hyps[i] = res[0]["text"].strip()
            info = sf.info(str(path))
            audio_seconds += info.frames / info.samplerate
        except Exception as e:  # 失败只标记该条
            errors[i] = f"generate_failed: {e}"
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
        f"显存峰值 {peak_gb:.2f} GB"
    )


if __name__ == "__main__":
    main()
