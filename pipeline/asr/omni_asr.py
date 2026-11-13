"""Omnilingual ASR（omniASR-LLM-7B）封装：按指定语言转写，输出模型原样的文本。

模型：facebook/omniASR-LLM-7B（Meta Omnilingual ASR，许可 Apache-2.0）。结构为 wav2vec2 音频编码器
加 LLaMA 式 Transformer 解码器，词表是字符级。推理用官方 omnilingual-asr 包（基于 fairseq2）
的 ASRInferencePipeline，解码用该管线的默认设置（beam search，nbest = 1）。
支持的语言代码（项目语言码 → 模型语言码）：zh→cmn_Hans en→eng_Latn ja→jpn_Jpan ko→kor_Hang
de→deu_Latn fr→fra_Latn ru→rus_Cyrl pt→por_Latn es→spa_Latn it→ita_Latn id→ind_Latn ms→zsm_Latn
th→tha_Thai vi→vie_Latn fil→fil_Latn ar→arb_Arab tr→tur_Latn fa→fas_Arab（全部 18 种）。
环境：envs/asr-omni（独立环境，Python 3.12，torch 2.8.0+cu128，fairseq2 0.6+cu128，omnilingual-asr 0.2.0，
建立方法见 issue #35）。fairseq2 需要系统的 libsndfile。
显存、吞吐：RTX 5090、bf16、批次 16，FLEURS dev 480 句（共 5972 秒音频）显存峰值 25.7 GB
（torch 统计的已分配峰值），速度 23 音频秒/墙钟秒（不含模型加载）。详见 issue #35。

用法：
    python -m pipeline.asr.omni_asr --input audio.jsonl --output results.jsonl --batch-size 16 \
        [--model facebook/omniASR-LLM-7B] [--audio-root 音频根目录]

--model 是 Hugging Face 仓库 ID 或本地目录，目录中要有一个 .pt 权重和一个 .model 分词器文件。
模型结构取自 omnilingual-asr 自带的 fairseq2 资产卡，卡名默认由权重文件名得到（omniASR-LLM-7B.pt →
omniASR_LLM_7B），也可以用 --card 指定。权重和分词器都从这个目录读取，fairseq2 不再另外下载。

输入每行至少有 sample_id、audio_path、lang。audio_path 是相对路径时，相对 --audio-root 解析；
没有给 --audio-root 时，相对输入文件所在目录。有 offset_s 的行只识别从 offset_s 秒开始、长 duration_s 秒的一段
（pipeline/realspeech/materialize.py 的虚拟切分：片段不另存文件，读取时在内存中截取）。
输出每行：sample_id、lang、asr_model、hypothesis（只保留转写文本）。转写失败的样本
hypothesis 为空字符串，并多一个 error 字段。

语言与文本处理：
- 目标语言通过语言码（例如 tur_Latn）传给模型。
- 模型没有标点、大小写、数字写法的开关，本封装也不对输出做任何规范化，统一写法留给比对环节。
  这个版本（v1）的字符词表没有大写字母，输出全部小写，不输出句读标点（只有词内的撇号、连字符
  和数字里的点号、逗号），数字有时写成阿拉伯数字；日语长音符「ー」和波斯语的零宽不连字（U+200C）
  也不在词表中，前者输出为空格，后者直接省略。
- 官方管线只接受 40 秒以内的音频，更长的样本直接标记为失败（error 为 too_long）。
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

DEFAULT_MODEL = "facebook/omniASR-LLM-7B"
SAMPLE_RATE = 16000
MAX_AUDIO_SEC = 40.0

# 项目语言码 → omniASR 语言码（ISO 639-3 加文字代码）
LANG_CODES = {
    "zh": "cmn_Hans",
    "en": "eng_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "ru": "rus_Cyrl",
    "pt": "por_Latn",
    "es": "spa_Latn",
    "it": "ita_Latn",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "th": "tha_Thai",
    "vi": "vie_Latn",
    "fil": "fil_Latn",
    "ar": "arb_Arab",
    "tr": "tur_Latn",
    "fa": "fas_Arab",
}


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
    with open(file = path, mode = "r", encoding = "utf-8") as f:
        return [json.loads(s = line) for line in f if line.strip()]


def resolve_audio_path(audio_path: str, audio_root: Path) -> Path:
    p = Path(audio_path)
    return p if p.is_absolute() else audio_root / p


def find_model_files(model: str) -> tuple[Path, Path]:
    """返回 (.pt 权重, .model 分词器)。model 不是本地目录时，按 HF 仓库 ID 下载到默认缓存。"""
    model_dir = Path(model)
    if not model_dir.is_dir():
        from huggingface_hub import snapshot_download

        model_dir = Path(snapshot_download(repo_id = model))
    checkpoints = sorted(model_dir.glob("*.pt"))
    tokenizers = sorted(model_dir.glob("*.model"))
    if len(checkpoints) != 1 or len(tokenizers) != 1:
        raise FileNotFoundError(f"{model_dir} 中应恰好有一个 .pt 权重和一个 .model 分词器")
    return checkpoints[0], tokenizers[0]


def build_pipeline(checkpoint: Path, tokenizer_path: Path, card_name: str):
    """用本地权重和分词器替换资产卡中的下载地址，构建官方推理管线。"""
    from fairseq2.assets import AssetCard, get_asset_store
    from fairseq2.data.tokenizers.hub import load_tokenizer
    from fairseq2.data.tokenizers.ref import resolve_tokenizer_reference
    from fairseq2.models.hub import load_model
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    store = get_asset_store()
    base_card = store.retrieve_card(name = card_name)
    # 新卡继承原卡的模型结构，只把权重地址换成本地文件。
    # 用 absolute 不用 resolve：HF 缓存里的文件是指向无扩展名 blob 的链接，fairseq2 靠 .pt 扩展名识别权重格式
    model_card = AssetCard(
        name = card_name,
        metadata = {"checkpoint": str(checkpoint.absolute())},
        base = base_card,
    )
    tokenizer_base = resolve_tokenizer_reference(asset_store = store, card = base_card)
    tokenizer_card = AssetCard(
        name = tokenizer_base.name,
        metadata = {"tokenizer": str(tokenizer_path.absolute())},
        base = tokenizer_base,
    )
    device = torch.device("cuda")
    model = load_model(
        card = model_card,
        device = device,
        dtype = torch.bfloat16,
        mmap = True,
    )
    tokenizer = load_tokenizer(card = tokenizer_card)
    return ASRInferencePipeline(
        model_card = None,
        model = model,
        tokenizer = tokenizer,
        device = device,
        dtype = torch.bfloat16,
    )


def duration_batches(order: list, seconds: dict, max_items: int, max_seconds: float) -> list:
    """把按时长升序排好的下标切成批：每批不超过 max_items 条；max_seconds 大于 0 时，「条数 × 批内最长时长」
    （补零后的总时长）也不超过它。真人语音切分后的片段多在 15 秒以内，固定 16 条一批用不满显存（16 条是按 FLEURS
    最长 40 秒的句子定的），按时长组批时短片段一批能放更多条，显存峰值相近。与 qwen3_asr.py、typhoon.py 中的同名函数相同。"""
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
    parser = argparse.ArgumentParser(description = "omniASR-LLM 批量转写")
    parser.add_argument("--input", required = True, help = "含 sample_id、audio_path、lang 的 JSONL")
    parser.add_argument("--output", required = True, help = "识别结果 JSONL")
    parser.add_argument("--batch-size", type = int, default = 16)
    parser.add_argument("--max-batch-seconds", type = float, default = 0.0,
                        help = "大于 0 时按补零后的总时长组批，--batch-size 为每批条数上限")
    parser.add_argument("--model", default = DEFAULT_MODEL, help = "HF 仓库 ID 或本地目录（含 .pt 与 .model）")
    parser.add_argument("--card", default = None, help = "fairseq2 资产卡名，默认由权重文件名推出")
    parser.add_argument("--audio-root", default = None, help = "相对 audio_path 的根目录，默认为输入文件所在目录")
    args = parser.parse_args()

    # fairseq2 发现 CONDA_PREFIX 时只在 conda 环境里找 libsndfile；本环境是 uv 虚拟环境，改用系统库
    os.environ.pop("CONDA_PREFIX", None)

    checkpoint, tokenizer_path = find_model_files(model = args.model)
    card_name = args.card or checkpoint.stem.replace("-", "_")
    asr_model_name = checkpoint.stem.lower()

    input_path = Path(args.input)
    audio_root = Path(args.audio_root) if args.audio_root else input_path.resolve().parent
    rows = read_jsonl(path = input_path)

    pipe = build_pipeline(
        checkpoint = checkpoint,
        tokenizer_path = tokenizer_path,
        card_name = card_name,
    )

    audios: dict[int, np.ndarray] = {}
    errors: dict[int, str] = {}
    for i, row in enumerate(rows):
        if row["lang"] not in LANG_CODES:
            errors[i] = f"unsupported_lang: {row['lang']}"
            continue
        try:
            audio = load_audio(
                path = resolve_audio_path(audio_path = row["audio_path"], audio_root = audio_root),
                offset_s = row.get("offset_s"),
                duration_s = row.get("duration_s"),
            )
        except Exception as e:  # 读取失败只标记该条
            errors[i] = f"load_failed: {e}"
            continue
        if len(audio) / SAMPLE_RATE > MAX_AUDIO_SEC:
            errors[i] = f"too_long: {len(audio) / SAMPLE_RATE:.1f}s > {MAX_AUDIO_SEC:.0f}s"
            continue
        audios[i] = audio
    # 按语言分组、组内按时长排序，批内长度接近，补零少；输出仍按输入顺序写出
    order = sorted(audios, key = lambda i: (rows[i]["lang"], len(audios[i])))

    def transcribe(chunk: list[int]) -> list[str]:
        return pipe.transcribe(
            inp = [{"waveform": audios[i], "sample_rate": SAMPLE_RATE} for i in chunk],
            lang = [LANG_CODES[rows[i]["lang"]] for i in chunk],
            batch_size = len(chunk),
        )

    hyps: dict[int, str] = {}
    audio_seconds = 0.0
    torch.cuda.synchronize()
    t0 = time.time()
    seconds = {i: len(audios[i]) / SAMPLE_RATE for i in audios}
    for lang in sorted({rows[i]["lang"] for i in order}):
        idx = [i for i in order if rows[i]["lang"] == lang]
        for chunk in duration_batches(order = idx, seconds = seconds, max_items = args.batch_size, max_seconds = args.max_batch_seconds):
            try:
                texts = transcribe(chunk = chunk)
                for i, text in zip(chunk, texts):
                    hyps[i] = text.strip()
            except Exception:
                # 整批失败（例如显存不足）时逐条重试，只把仍然失败的样本标记出来
                torch.cuda.empty_cache()
                for i in chunk:
                    try:
                        hyps[i] = transcribe(chunk = [i])[0].strip()
                    except Exception as e:
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
