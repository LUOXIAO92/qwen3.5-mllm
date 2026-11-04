"""评测推理脚本（实验 v1，任务 T15）：批量转写评测清单并打分。

用法（在仓库根目录运行，推理需要 GPU，请经 GPU 锁排队）：

    flock <gpu.lock> python -m pipeline.eval.run_eval \
        --model-dir <完整模型目录> \
        --projector <projector.safetensors> \
        --manifest-dir <含 <lang>/test.jsonl 的目录> \
        --langs zh en ja \
        --limit 20 \
        --batch-size 8 \
        --output-dir <输出目录>

输入
====

- 评测清单 `<manifest-dir>/<lang>/test.jsonl`：每行含 sample_id、audio_path、duration_s、raw_transcription。
  audio_path 相对 `--audio-root`（默认等于 `--manifest-dir`），参考文本取 raw_transcription。
- `--model-dir`：带 remote code 的完整模型目录，用 `trust_remote_code = True` 加载，bf16 推理。
  目录里可以是阶段 1 的模型（projector 未训练），也可以是已经合入 projector 的模型。
- `--projector`（可选）：一个 safetensors 文件，只含完整模型 state_dict 中以 `model.audio_projector.`
  开头的键，键名与完整模型一致。加载前检查文件里的键恰好等于模型的全部 projector 参数，
  不缺也不多，形状一致。不给时直接使用 `--model-dir` 里的 projector。

输入格式
========

与训练一致（pipeline/CONVENTIONS.md「训练与评测的对话格式」）：用户消息依次是一段音频和
`Transcribe the audio.`，用模型自带 chat template 展开，`add_generation_prompt = True`、
`enable_thinking = False`，助手回合以空思考块开头。批量生成用左填充、贪心解码，
`max_new_tokens` 按该批最长音频的时长估算，防止未训练的模型无限输出。

输出（写入 `--output-dir`）
===========================

- `hypotheses.jsonl`：逐条结果，含 sample_id、lang、reference、hypothesis、duration_s、n_new_tokens，
  以及打分器给出的 basic、numeric 两种逐条分数。
- `summary.json` 与 `summary.md`：按语言的结果表，基础分和数词统一分都列出。
- `perf.json`：每个批量的生成耗时、音频总时长、实时倍率、生成 token 数和峰值显存。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

from pipeline.eval.scoring import LANGS, METRIC_BY_LANG, score_corpus

PROMPT_TEXT = "Transcribe the audio."
PROJECTOR_PREFIX = "model.audio_projector."
_THINK_BLOCK = re.compile(pattern = r"<think>.*?</think>", flags = re.DOTALL)


def build_messages(audio_path: str) -> list:
    """按 CONVENTIONS 构造一条对话：音频 + 固定指令。"""
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": PROMPT_TEXT},
            ],
        },
    ]


def clean_output(text: str) -> str:
    """去掉生成中残留的思考块、孤立的思考标记和特殊标记。"""
    text = _THINK_BLOCK.sub(repl = "", string = text)
    text = text.replace("<think>", "").replace("</think>", "")
    text = re.sub(pattern = r"<\|[^|>]*\|>", repl = "", string = text)
    return text.strip()


def read_manifest(manifest_dir: Path, lang: str, limit: int | None) -> list:
    """读取一种语言的 test 清单，取前 limit 条。"""
    path = manifest_dir / lang / "test.jsonl"
    rows = []
    with open(file = path, mode = "r", encoding = "utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(s = line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_projector(model, projector_path: Path) -> None:
    """加载 projector 检查点：键必须恰好覆盖模型的全部 projector 参数。"""
    from safetensors.torch import load_file

    state = load_file(filename = str(projector_path), device = "cpu")
    expected = {
        key: value
        for key, value in model.state_dict().items()
        if key.startswith(PROJECTOR_PREFIX)
    }
    missing = sorted(set(expected) - set(state))
    extra = sorted(set(state) - set(expected))
    if missing or extra:
        raise ValueError(
            f"projector 检查点的键与模型不一致。缺少：{missing}；多余：{extra}",
        )
    for key, value in expected.items():
        if tuple(state[key].shape) != tuple(value.shape):
            raise ValueError(
                f"{key} 形状不一致：检查点 {tuple(state[key].shape)}，模型 {tuple(value.shape)}",
            )
    result = model.load_state_dict(state_dict = state, strict = False)
    # strict = False 只是因为文件只含 projector；此处确认没有意外的多余键
    if result.unexpected_keys:
        raise ValueError(f"出现多余的键：{result.unexpected_keys}")


def load_model(model_dir: Path, projector_path: Path | None):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path = str(model_dir),
        trust_remote_code = True,
    )
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        pretrained_model_name_or_path = str(model_dir),
        trust_remote_code = True,
        dtype = torch.bfloat16,
    )
    if projector_path is not None:
        load_projector(model = model, projector_path = projector_path)
    model.to(device = "cuda")
    model.eval()
    return model, processor


def build_inputs(processor, audio_paths: list, waveforms: list):
    """展开 chat template 并处理音频，返回 (展开前的文本列表, 模型输入)。"""
    prompts = [
        processor.apply_chat_template(
            conversation = build_messages(audio_path = path),
            tokenize = False,
            add_generation_prompt = True,
            enable_thinking = False,
        )
        for path in audio_paths
    ]
    inputs = processor(
        text = prompts,
        audio = waveforms,
        sampling_rate = 16000,
        padding = True,
        return_tensors = "pt",
    )
    return prompts, inputs


def transcribe_batch(model, processor, rows: list, audio_root: Path, max_new_tokens_cfg: dict) -> tuple:
    """转写一个批量，返回 (hypotheses, n_new_tokens 列表, 生成耗时秒, 峰值显存字节)。"""
    import soundfile as sf
    import torch

    audio_paths = [str(audio_root / row["audio_path"]) for row in rows]
    waveforms = []
    for path in audio_paths:
        wave, sample_rate = sf.read(file = path, dtype = "float32")
        if sample_rate != 16000:
            raise ValueError(f"{path} 采样率为 {sample_rate}，需要 16000")
        if wave.ndim > 1:
            wave = wave.mean(axis = 1)
        waveforms.append(wave)
    longest = max(len(wave) for wave in waveforms) / 16000
    max_new_tokens = int(
        math.ceil(longest * max_new_tokens_cfg["tokens_per_second"]) + max_new_tokens_cfg["extra"],
    )

    _, inputs = build_inputs(processor = processor, audio_paths = audio_paths, waveforms = waveforms)
    inputs = inputs.to(device = "cuda")
    inputs["input_features"] = inputs["input_features"].to(dtype = torch.bfloat16)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            do_sample = False,
            use_cache = True,
            pad_token_id = processor.tokenizer.pad_token_id,
            # 阶段 1 的 generation_config 只在 <|endoftext|> 停止；助手回合以 <|im_end|> 结束，两者都作为停止标记
            eos_token_id = processor.tokenizer.convert_tokens_to_ids(tokens = ["<|im_end|>", "<|endoftext|>"]),
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated()

    new_ids = generated[:, inputs["input_ids"].shape[1]:]
    pad_id = processor.tokenizer.pad_token_id
    n_new = [int((row != pad_id).sum()) for row in new_ids]
    texts = processor.batch_decode(sequences = new_ids, skip_special_tokens = True)
    return [clean_output(text = t) for t in texts], n_new, elapsed, peak, max_new_tokens


def format_markdown(summary: dict) -> str:
    """按语言的结果表（错误率为百分数）。"""

    def pct(v):
        return "n/a" if v is None else f"{v * 100:.2f}"

    lines = [
        "| 语言 | 指标 | 条数 | 参考长度 | 基础分 % | 数词统一分 % | S | D | I |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lang, v in summary.items():
        basic, numeric = v["basic"], v["numeric"]
        lines.append(
            f"| {lang} | {basic['metric']} | {basic['n']} | {numeric.get('ref_len', '')} "
            f"| {pct(basic['rate'])} | {pct(numeric['rate'])} "
            f"| {numeric.get('sub', '')} | {numeric.get('del', '')} | {numeric.get('ins', '')} |",
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: list | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "批量转写评测清单并按语言打分")
    parser.add_argument("--model-dir", required = True, type = Path, help = "完整模型目录（remote code）")
    parser.add_argument("--projector", default = None, type = Path, help = "projector 检查点（safetensors，可选）")
    parser.add_argument("--manifest-dir", required = True, type = Path, help = "含 <lang>/test.jsonl 的目录")
    parser.add_argument("--audio-root", default = None, type = Path, help = "audio_path 的根目录，默认等于 --manifest-dir")
    parser.add_argument("--langs", nargs = "+", default = LANGS, help = "要评测的语言代码，默认全部 18 种")
    parser.add_argument("--limit", default = None, type = int, help = "每种语言最多评测多少条，默认全部")
    parser.add_argument("--batch-size", default = 8, type = int, help = "批量大小")
    parser.add_argument("--tokens-per-second", default = 12.0, type = float, help = "max_new_tokens 估算：每秒音频的 token 上限")
    parser.add_argument("--extra-tokens", default = 32, type = int, help = "max_new_tokens 估算：额外余量")
    parser.add_argument("--output-dir", required = True, type = Path, help = "输出目录")
    parser.add_argument("--show-inputs", default = 0, type = int, help = "每种语言打印前几条样本展开前的输入文本")
    return parser.parse_args(argv)


def main(argv: list | None = None) -> int:
    args = parse_args(argv = argv)
    for lang in args.langs:
        if lang not in METRIC_BY_LANG:
            raise SystemExit(f"不支持的语言代码：{lang}")
    audio_root = args.audio_root if args.audio_root is not None else args.manifest_dir
    args.output_dir.mkdir(parents = True, exist_ok = True)

    model, processor = load_model(model_dir = args.model_dir, projector_path = args.projector)
    token_cfg = {"tokens_per_second": args.tokens_per_second, "extra": args.extra_tokens}

    results = []
    perf = []
    for lang in args.langs:
        rows = read_manifest(manifest_dir = args.manifest_dir, lang = lang, limit = args.limit)
        # 按时长排序，同一批的音频长度接近，填充少
        rows = sorted(rows, key = lambda r: r["duration_s"])
        if args.show_inputs:
            prompts = [
                processor.apply_chat_template(
                    conversation = build_messages(audio_path = str(audio_root / r["audio_path"])),
                    tokenize = False,
                    add_generation_prompt = True,
                    enable_thinking = False,
                )
                for r in rows[: args.show_inputs]
            ]
            for r, prompt in zip(rows, prompts):
                print(f"[{lang}] {r['sample_id']} 展开前的输入文本：\n{prompt}\n", flush = True)
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start: start + args.batch_size]
            hyps, n_new, elapsed, peak, max_new = transcribe_batch(
                model = model,
                processor = processor,
                rows = batch,
                audio_root = audio_root,
                max_new_tokens_cfg = token_cfg,
            )
            audio_seconds = sum(r["duration_s"] for r in batch)
            perf.append(
                {
                    "lang": lang,
                    "batch_size": len(batch),
                    "audio_seconds": round(audio_seconds, 2),
                    "gen_seconds": round(elapsed, 3),
                    "realtime_factor": round(audio_seconds / elapsed, 2),
                    "new_tokens": sum(n_new),
                    "tokens_per_second": round(sum(n_new) / elapsed, 1),
                    "max_new_tokens": max_new,
                    "peak_mem_gib": round(peak / 2 ** 30, 2),
                },
            )
            for row, hyp, n in zip(batch, hyps, n_new):
                results.append(
                    {
                        "sample_id": row["sample_id"],
                        "lang": lang,
                        "reference": row["raw_transcription"],
                        "hypothesis": hyp,
                        "duration_s": row["duration_s"],
                        "n_new_tokens": n,
                    },
                )
            print(f"[{lang}] {start + len(batch)}/{len(rows)} 条，{elapsed:.1f}s，峰值显存 {peak / 2 ** 30:.2f} GiB", flush = True)

    per_sample, summary = score_corpus(rows = results)
    with open(file = args.output_dir / "hypotheses.jsonl", mode = "w", encoding = "utf-8") as f:
        for row, scored in zip(results, per_sample):
            record = dict(row)
            record["basic"] = scored["basic"]
            record["numeric"] = scored["numeric"]
            f.write(json.dumps(obj = record, ensure_ascii = False) + "\n")
    (args.output_dir / "summary.json").write_text(
        data = json.dumps(obj = summary, ensure_ascii = False, indent = 2),
        encoding = "utf-8",
    )
    table = format_markdown(summary = summary)
    (args.output_dir / "summary.md").write_text(data = table, encoding = "utf-8")
    (args.output_dir / "perf.json").write_text(
        data = json.dumps(obj = perf, ensure_ascii = False, indent = 2),
        encoding = "utf-8",
    )
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
