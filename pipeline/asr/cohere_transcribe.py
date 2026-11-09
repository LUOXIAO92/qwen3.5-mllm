"""Cohere Transcribe 封装：按指定语言转写，带标点，输出模型原始文本。

模型（都是 2B 参数，Conformer 编码器 + 轻量 Transformer 解码器的 AED 结构，许可 Apache-2.0；
Hugging Face 上是 gated 仓库，需要先在模型页同意条件才能下载）：
- 通用版 CohereLabs/cohere-transcribe-03-2026（默认）。支持的项目语言码：zh en ja ko de fr pt es it vi ar（11 种）。
- 阿拉伯语版 CohereLabs/cohere-transcribe-arabic-07-2026，用 --model 指定。模型卡把通用版列为基础模型，
  针对阿拉伯语（含方言、阿英混说）优化；结构、分词器、特征提取设置与通用版相同，只支持 ar、en。仓库里的
  normalizer.json 是英式→美式拼写对照表（与 Whisper 评测用的英文规范化表同类），只供评测时规范化参考，
  transformers 的 cohere_asr 处理器不读取它，对输出没有影响。
模型不支持的语言（例如 ru id ms th fil tr fa，阿拉伯语版另有 zh ja 等）遇到时该条失败并写 error；
可用语言取自模型配置的 supported_languages。asr_model 字段取 --model 的最后一段。
环境：envs/asr-cohere（基于 conda 环境 py3.14 的虚拟环境，transformers 5.16.1 自带 cohere_asr 实现，
不需要 trust_remote_code；额外安装 soundfile 0.14.0 和特征提取需要的 librosa 1.0.0，建立方法见 issue #38）。
显存、吞吐：RTX 5090 上 bf16、批量 32，FLEURS dev 11 种语言各 50 句（550 句，6426 秒音频），
显存峰值 5.5 GB（torch 统计的已分配峰值），343 音频秒/墙钟秒（不含模型加载）。阿拉伯语版结构相同，
同一批阿拉伯语音频上速度与显存和通用版一致。详见 issue #38。

用法：
    python -m pipeline.asr.cohere_transcribe --input audio.jsonl --output results.jsonl --batch-size 32 \
        [--model CohereLabs/cohere-transcribe-03-2026] [--audio-root 音频根目录]

输入每行至少有 sample_id、audio_path、lang。audio_path 是相对路径时，相对 --audio-root 解析；
没有给 --audio-root 时，相对输入文件所在目录。
输出每行：sample_id、lang、asr_model、hypothesis（模型原始文本，不做任何规范化）。转写失败的样本
hypothesis 为空字符串，并多一个 error 字段。

语言与文本处理：
- 模型不做语言识别，必须指定语言。处理器把语言码写进解码器的提示词（<|zh|> 这类标记），
  同时写入 <|pnc|>（输出标点和大小写）和 <|noitn|>。<|noitn|> 是处理器固定写入的，没有开关；
  实测数字仍常写成阿拉伯数字（例如「1000 多年」），中文输出常用半角逗号，写法需要在比对环节统一。
- 输出只去掉特殊标记和首尾空白，不做其他改动。
- 超过 30 秒的音频由特征提取器在能量最低处切成不超过 35 秒的片段，分别解码后按顺序拼回
  （zh、ja 直接相连，其余语言用空格连接）。
- 解码用贪心搜索，max_new_tokens 默认 512。模型卡说明它对非人声也倾向于转写，
  静音、噪声片段可能得到非空输出，需要在核对环节处理。阿拉伯语版按 ar 转写噪声、静音时，
  常输出 @@@ضوضاء（噪声）、@@@فراغ（空白）这类标记，本封装原样写出。
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

DEFAULT_MODEL = "CohereLabs/cohere-transcribe-03-2026"
SAMPLE_RATE = 16000

# 项目语言码，与模型语言码相同；实际可用的语言取它与模型配置 supported_languages 的交集
PROJECT_LANGS = {
    "zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es",
    "it", "id", "ms", "th", "vi", "fil", "ar", "tr", "fa",
}


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
    with open(file = path, mode = "r", encoding = "utf-8") as f:
        return [json.loads(s = line) for line in f if line.strip()]


def resolve_audio_path(audio_path: str, audio_root: Path) -> Path:
    p = Path(audio_path)
    return p if p.is_absolute() else audio_root / p


def main() -> None:
    parser = argparse.ArgumentParser(description = "Cohere Transcribe 批量转写")
    parser.add_argument("--input", required = True, help = "含 sample_id、audio_path、lang 的 JSONL")
    parser.add_argument("--output", required = True, help = "识别结果 JSONL")
    parser.add_argument("--batch-size", type = int, default = 32)
    parser.add_argument("--model", default = DEFAULT_MODEL, help = "模型 ID 或本地目录")
    parser.add_argument("--audio-root", default = None, help = "相对 audio_path 的根目录，默认为输入文件所在目录")
    parser.add_argument("--max-new-tokens", type = int, default = 512)
    args = parser.parse_args()

    from transformers import AutoProcessor, CohereAsrForConditionalGeneration

    input_path = Path(args.input)
    audio_root = Path(args.audio_root) if args.audio_root else input_path.resolve().parent
    rows = read_jsonl(path = input_path)

    processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path = args.model)
    model = CohereAsrForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path = args.model,
        dtype = torch.bfloat16,
        device_map = "cuda",
    ).eval()
    # 通用版支持 14 种语言，阿拉伯语版只支持 en、ar，都写在模型配置的 supported_languages 中
    supported_langs = PROJECT_LANGS & set(model.config.supported_languages)
    # asr_model 取模型 ID 或目录的最后一段，例如 cohere-transcribe-arabic-07-2026
    asr_model_name = Path(args.model.rstrip("/")).name

    audios: dict[int, np.ndarray] = {}
    errors: dict[int, str] = {}
    for i, row in enumerate(rows):
        if row["lang"] not in supported_langs:
            errors[i] = f"unsupported_lang: {row['lang']}"
            continue
        try:
            audios[i] = load_audio(path = resolve_audio_path(audio_path = row["audio_path"], audio_root = audio_root))
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
        for start in range(0, len(idx), args.batch_size):
            chunk = idx[start : start + args.batch_size]
            try:
                inputs = processor(
                    audio = [audios[i] for i in chunk],
                    language = lang,
                    punctuation = True,
                    sampling_rate = SAMPLE_RATE,
                    return_tensors = "pt",
                )
                # 长音频切片后，一条输入对应多个片段；audio_chunk_index 记录片段属于哪一条
                audio_chunk_index = inputs.pop("audio_chunk_index")
                inputs = inputs.to(device = model.device, dtype = model.dtype)
                with torch.inference_mode():
                    out = model.generate(
                        **inputs,
                        max_new_tokens = args.max_new_tokens,
                        do_sample = False,
                    )
                texts = processor.decode(
                    token_ids = out,
                    skip_special_tokens = True,
                    audio_chunk_index = audio_chunk_index,
                    language = lang,
                )
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
    with open(file = output_path, mode = "w", encoding = "utf-8") as f:
        for i, row in enumerate(rows):
            rec = {
                "sample_id": row["sample_id"],
                "lang": row["lang"],
                "asr_model": asr_model_name,
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
