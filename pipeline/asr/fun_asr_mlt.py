"""Fun-ASR-MLT-Nano-2512 封装：按指定语言成批转写，输出模型原始文本。

模型：FunAudioLLM/Fun-ASR-MLT-Nano-2512（0.8B，音频编码器 + Qwen3-0.6B 解码器，许可 Apache-2.0）。
支持的语言代码（项目语言码）：zh en ja ko pt id ms th vi fil ar（共 11 种）。
其余项目语言（de fr ru es it tr fa）模型不支持，遇到时该条失败并写 error。
环境：envs/asr-funasr-mlt（基于 conda 环境 py3.14 的虚拟环境，额外安装 funasr 1.4.24，
transformers 随之升到 5.22.2）。模型结构和权重加载用上游仓库 FunAudioLLM/Fun-ASR 的 model.py，
通过 --funasr-src 指向它的克隆目录（建立方法见 pipeline/ENVIRONMENTS.md）。
显存、吞吐：FLEURS dev zh ja ko id 共 1,251 句（约 4.1 小时）上，--batch-size 256 --max-batch-seconds 960
时显存峰值约 4.9 GB，约 330–350 音频秒/墙钟秒；逐条解码约 19 音频秒/墙钟秒（issue #40）。

用法：
    python -m pipeline.asr.fun_asr_mlt --input audio.jsonl --output results.jsonl --batch-size 256 \
        --max-batch-seconds 960 --funasr-src Fun-ASR 克隆目录 \
        [--model FunAudioLLM/Fun-ASR-MLT-Nano-2512] [--audio-root 音频根目录]

输入每行至少有 sample_id、audio_path、lang。audio_path 是相对路径时，相对 --audio-root 解析；
没有给 --audio-root 时，相对输入文件所在目录。
输出每行：sample_id、lang、asr_model、hypothesis（模型原始文本，不做任何规范化）。转写失败的样本
hypothesis 为空字符串，并多一个 error 字段。

语言与文本处理：
- 语言通过提示词「语音转写成<语言名>」指定，语言名用中文（中文、英文、日文、韩文、葡萄牙语、
  印尼语、马来语、泰语、越南语、菲律宾语、阿拉伯语）。
- itn = False 时提示词末尾加「，不进行文本规整」，模型输出不做反向文本规范化，数字多保持口语词形。
  本封装固定 itn = False；输出文本与上游 inference 的 text 字段相同：把「/sil」换成空格、合并空白。
- 解码用贪心（do_sample = False），max_new_tokens 默认 512。

批量解码：
上游 model.py 的 inference 每次只解码一条（批大小大于 1 时直接报错）。本封装照常用 funasr 的 AutoModel
加载同一份模型，只把推理步骤改成成批计算，每一步与上游 inference 相同：
1. 读音频、算 fbank 特征用 funasr 的 load_audio_text_image_video 和 extract_fbank，每条按自己的长度单独计算。
   上游的 fbank 带 dither（每帧加一点随机噪声），本封装按输入顺序逐条计算，特征与逐条封装逐位相同。
2. 编码器和 adaptor 成批计算（fp32），补零的帧由长度掩码排除。
3. 每条的输入是「系统提示 + 用户提示词 + 音频向量 + 助手回合开头」的词向量，按批内最长的一条在左侧补齐，
   补齐的位置注意力掩码为 0；解码器用 bf16，在 bf16 autocast 下成批 generate。
上游 inference 每条还会跑 CTC 解码和强制对齐来算时间戳，它们不影响 text，本封装不计算。
成批计算的矩阵乘法与逐条计算的舍入不同，个别句子的贪心解码会走到另一个词。上面的 FLEURS dev 1,251 句中，
95.5% 与逐条解码逐字相同，各语言字/词错误率相差不超过 0.05 个百分点（issue #40）。

组批：按语言分组、组内按时长升序，每批不超过 --batch-size 条；--max-batch-seconds 大于 0 时，
「条数 × 批内最长时长」（补零后的总时长）也不超过它，短音频一批放得多，长音频一批放得少，显存峰值相近。
"""

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

DEFAULT_MODEL = "FunAudioLLM/Fun-ASR-MLT-Nano-2512"
ASR_MODEL_NAME = "fun-asr-mlt-nano-2512"
SYSTEM_PROMPT = "You are a helpful assistant."
LOAD_WORKERS = 8  # 读音频的线程数

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


def duration_batches(order: list, seconds: dict, max_items: int, max_seconds: float) -> list:
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


def prompt_ids(asr, tokenizer, lang: str) -> tuple:
    """音频前、后两段文字的 token。与上游 data_load_speech 拼出的对话相同：
    <|im_start|>system\\n{系统提示}<|im_end|>\\n<|im_start|>user\\n{提示词}[音频]<|im_end|>\\n<|im_start|>assistant\\n"""
    prompt = asr.get_prompt(hotwords = [], language = LANG_NAMES[lang], itn = False)
    prefix = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}"
    suffix = "<|im_end|>\n<|im_start|>assistant\n"
    return tokenizer.encode(text = prefix), tokenizer.encode(text = suffix)


@torch.inference_mode()
def transcribe_batch(asr, tokenizer, feats: list, prefix: list, suffix: list, max_new_tokens: int) -> list:
    """一批 fbank 特征（每条形状 [帧数, 560]）→ 转写文本列表。"""
    device = next(asr.parameters()).device
    llm = asr.llm
    frames = [f.shape[0] for f in feats]
    lengths = torch.tensor(data = frames, dtype = torch.int32, device = device)
    speech = pad_sequence(sequences = feats, batch_first = True, padding_value = 0.0).to(device)
    encoder_out, encoder_out_lens = asr.audio_encoder(xs_pad = speech, ilens = lengths)
    adaptor_out, _ = asr.audio_adaptor(x = encoder_out, ilens = encoder_out_lens)

    embed = llm.get_input_embeddings()
    prefix_emb = embed(input = torch.tensor(data = prefix, dtype = torch.long, device = device))
    suffix_emb = embed(input = torch.tensor(data = suffix, dtype = torch.long, device = device))
    totals = [len(prefix) + n + len(suffix) for n in frames]
    width = max(totals)
    inputs_embeds = torch.zeros(
        len(feats),
        width,
        prefix_emb.shape[-1],
        dtype = prefix_emb.dtype,
        device = device,
    )
    attention_mask = torch.zeros(
        len(feats),
        width,
        dtype = torch.long,
        device = device,
    )
    for b, (n, total) in enumerate(zip(frames, totals)):
        row = torch.cat(tensors = [prefix_emb, adaptor_out[b, :n].to(prefix_emb.dtype), suffix_emb], dim = 0)
        inputs_embeds[b, width - total :] = row
        attention_mask[b, width - total :] = 1

    with torch.autocast(device_type = device.type, dtype = torch.bfloat16):
        generated = llm.generate(
            inputs_embeds = inputs_embeds,
            attention_mask = attention_mask,
            max_new_tokens = max_new_tokens,
            pad_token_id = llm.config.pad_token_id or llm.config.eos_token_id,
            do_sample = False,
        )
    responses = tokenizer.batch_decode(sequences = generated, skip_special_tokens = True)
    return [re.sub(pattern = r"\s+", repl = " ", string = r.replace("/sil", " ")).strip() for r in responses]


def main() -> None:
    parser = argparse.ArgumentParser(description = "Fun-ASR-MLT-Nano-2512 批量转写")
    parser.add_argument("--input", required = True, help = "含 sample_id、audio_path、lang 的 JSONL")
    parser.add_argument("--output", required = True, help = "识别结果 JSONL")
    parser.add_argument("--batch-size", type = int, default = 1, help = "每批条数上限")
    parser.add_argument("--model", default = DEFAULT_MODEL, help = "模型 ID 或本地目录")
    parser.add_argument("--audio-root", default = None, help = "相对 audio_path 的根目录，默认为输入文件所在目录")
    parser.add_argument("--funasr-src", required = True, help = "上游 Fun-ASR 仓库克隆目录（含 model.py）")
    parser.add_argument("--max-new-tokens", type = int, default = 512)
    parser.add_argument(
        "--max-batch-seconds",
        type = float,
        default = 0.0,
        help = "大于 0 时按补零后的总时长组批，--batch-size 为每批条数上限",
    )
    args = parser.parse_args()

    src = Path(args.funasr_src).resolve()
    sys.path.insert(0, str(src))  # model.py 会 import 同目录的 ctc、tools
    from funasr import AutoModel
    from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video

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
    auto = AutoModel(**model_kwargs)
    asr = auto.model
    tokenizer = auto.kwargs["tokenizer"]
    frontend = auto.kwargs["frontend"]

    errors: dict[int, str] = {}
    paths: dict[int, Path] = {}
    for i, row in enumerate(rows):
        lang = row["lang"]
        if lang not in LANG_NAMES:
            errors[i] = f"unsupported_lang: {lang}"
            continue
        paths[i] = resolve_audio_path(audio_path = row["audio_path"], audio_root = audio_root)

    feats: dict[int, torch.Tensor] = {}
    seconds: dict[int, float] = {}
    torch.cuda.synchronize()
    t0 = time.time()
    # 读音频（解码、重采样）在线程池中并行。fbank 带 dither：每帧加一点随机噪声，噪声取自 torch 的全局随机数，
    # 所以在主线程按输入顺序逐条计算。模型加载后随机数状态固定（funasr 设种子 0），于是每条的特征与逐条封装
    # 按同样顺序处理时逐位相同，同一输入重复运行结果也相同。
    with ThreadPoolExecutor(max_workers = LOAD_WORKERS) as pool:
        loaded = {
            i: pool.submit(load_audio_text_image_video, data_or_path_or_list = str(path), fs = frontend.fs)
            for i, path in paths.items()
        }
        for i in paths:
            try:
                data = loaded.pop(i).result()
                speech, _ = extract_fbank(
                    data = data,
                    data_type = "sound",
                    frontend = frontend,
                    is_final = True,
                )
                feats[i] = speech[0]
                seconds[i] = data.shape[-1] / frontend.fs
            except Exception as e:  # 读取失败只标记该条
                errors[i] = f"load_failed: {e}"

    # 按语言分组、组内按时长排序，批内长度接近，补零少；输出仍按输入顺序写出
    hyps: dict[int, str] = {}
    audio_seconds = 0.0
    for lang in sorted({rows[i]["lang"] for i in feats}):
        prefix, suffix = prompt_ids(asr = asr, tokenizer = tokenizer, lang = lang)
        idx = sorted((i for i in feats if rows[i]["lang"] == lang), key = lambda i: seconds[i])
        batches = duration_batches(
            order = idx,
            seconds = seconds,
            max_items = args.batch_size,
            max_seconds = args.max_batch_seconds,
        )
        for chunk in batches:
            try:
                texts = transcribe_batch(
                    asr = asr,
                    tokenizer = tokenizer,
                    feats = [feats[i] for i in chunk],
                    prefix = prefix,
                    suffix = suffix,
                    max_new_tokens = args.max_new_tokens,
                )
                for i, text in zip(chunk, texts):
                    hyps[i] = text
                audio_seconds += sum(seconds[i] for i in chunk)
            except Exception as e:  # 一批失败只标记这一批
                for i in chunk:
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
