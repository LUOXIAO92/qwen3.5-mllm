"""音频评测的 vLLM 引擎（任务 #71）：在宿主机上用融合模型的音频塔与 projector 算出每条请求的音频向量，交给容器中的 vLLM。

    python -m pipeline.eval.audio_embeds --model-dir <融合模型目录> [--projector <projector.safetensors>] \\
        --items <任务目录>/audio_items.jsonl --out-dir <任务目录>

run_eval.py 与 instruct_probe.py 加 --engine vllm 时由它们启动（单独的进程，算完退出、显存全部释放后再启动 vLLM），一般
不直接运行。

做法与融合模型的 forward 相同（qwen3.5-2b-mllm/modeling_qwen3_5_mllm.py）：提示文字与波形一起交给 processor（按音频长度
展开 <|audio_pad|>，算 log-mel 特征），get_audio_features 经音频塔与 projector 得到每个音频 token 的向量（bfloat16）。
transformers 版把这些向量 masked_scatter 到 <|audio_pad|> 的位置；vLLM 版把它们原样交给 vLLM（混合输入，见
pipeline/eval/vllm_generate.py 开头），其余位置由 vLLM 按 token id 查词嵌入。语言模型的词嵌入在这里用不到，所以只载入
融合模型与 projector，不载入 --language-model。分批的方式由调用方给出（items 的 batch），与 transformers 版的批一致，
音频塔看到的输入（批内补齐的特征）与 transformers 版完全相同。

items（JSONL，每行一条请求）：index、batch（批编号）、prompt（展开前的提示文字）、audio_paths（音频文件的绝对路径，
时间段任务有几段）、gap_s 与 segment_bounds（几段音频首尾相接时的静音秒数与各段的起止时间，与 instruct_probe 相同）。

输出（--out-dir）：embeds-<序号>.safetensors（每条一个键 r<index>，形状为（音频 token 数, 隐藏维数）的 bfloat16，每个文件
约 1 GB）与 audio_items.out.jsonl（每行 index、prompt_token_ids（不含补齐）、file、key、audio_tokens、samples（波形的
采样点数））。最后写 audio_items.out.jsonl，有它表示全部完成。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SHARD_BYTES = 1 << 30
OUTPUT = "audio_items.out.jsonl"


def read_items(path: Path) -> list:
    with open(file = path, encoding = "utf-8") as handle:
        return [json.loads(s = line) for line in handle if line.strip()]


def item_waveform(item: dict):
    """一条请求的波形：一段音频直接读入；几段时中间插入 gap_s 秒的静音首尾相接，并核对各段的起止时间。"""
    from pipeline.eval.instruct_probe import SAMPLING_RATE, concatenate
    from pipeline.eval.run_eval import load_waveform

    waves = [load_waveform(path = path) for path in item["audio_paths"]]
    if len(waves) == 1:
        return waves[0]
    wave, bounds = concatenate(waveforms = waves, gap = int(round(item["gap_s"] * SAMPLING_RATE)))
    if item.get("segment_bounds") is not None and [list(bound) for bound in bounds] != item["segment_bounds"]:
        raise ValueError(f"第 {item['index']} 条读入的音频长度与记录的各段时间不一致")
    return wave


def encode_batch(model, processor, items: list) -> list:
    """一批请求的 (prompt_token_ids, 音频向量, 采样点数)，与 transformers 版的 build_inputs + forward 中的计算相同。"""
    import torch

    from pipeline.eval.run_eval import SAMPLING_RATE

    waveforms = [item_waveform(item = item) for item in items]
    inputs = processor(
        text = [item["prompt"] for item in items],
        audio = waveforms,
        sampling_rate = SAMPLING_RATE,
        padding = True,
        return_tensors = "pt",
    )
    device = next(model.parameters()).device
    audio_id = model.config.audio_token_id
    with torch.inference_mode():
        features = model.model.get_audio_features(
            input_features = inputs["input_features"].to(device = device, dtype = torch.bfloat16),
            feature_attention_mask = inputs["feature_attention_mask"].to(device = device),
        )
        features = features.to(dtype = model.get_input_embeddings().weight.dtype).cpu()
    out = []
    start = 0
    for row in range(len(items)):
        ids = inputs["input_ids"][row][inputs["attention_mask"][row].bool()].tolist()
        count = sum(1 for token in ids if token == audio_id)
        out.append((ids, features[start: start + count].clone(), len(waveforms[row])))
        start += count
    if start != features.shape[0]:
        raise RuntimeError(f"音频向量 {features.shape[0]} 行，提示中的音频占位符共 {start} 个")
    return out


def encode(model_dir: Path, projector: Path | None, items: list, out_dir: Path, log = print) -> list:
    """按 batch 分批算出全部请求的音频向量，写 safetensors 文件，返回 audio_items.out.jsonl 的各行（按 index 排列）。"""
    import torch
    from safetensors.torch import save_file

    from pipeline.eval.run_eval import load_model

    start = time.perf_counter()
    model, processor = load_model(model_dir = model_dir, projector_path = projector)
    log(f"载入融合模型 {time.perf_counter() - start:.0f} 秒")
    batches = {}
    for item in items:
        batches.setdefault(item["batch"], []).append(item)
    rows = []
    pending = {}
    pending_bytes = 0
    shard = 0

    def flush():
        nonlocal pending, pending_bytes, shard
        if pending:
            save_file(tensors = pending, filename = str(Path(out_dir) / f"embeds-{shard:03d}.safetensors"), metadata = {"format": "pt"})
            shard += 1
        pending = {}
        pending_bytes = 0

    tokens = 0
    for number, batch in enumerate(sorted(batches)):
        group = batches[batch]
        for item, (ids, features, samples) in zip(group, encode_batch(model = model, processor = processor, items = group)):
            key = f"r{item['index']}"
            pending[key] = features
            pending_bytes += features.numel() * features.element_size()
            rows.append({
                "index": item["index"],
                "prompt_token_ids": ids,
                "file": f"embeds-{shard:03d}.safetensors",
                "key": key,
                "audio_tokens": int(features.shape[0]),
                "samples": int(samples),
            })
            tokens += int(features.shape[0])
        if pending_bytes >= SHARD_BYTES:
            flush()
        if (number + 1) % 50 == 0:
            log(f"{number + 1} / {len(batches)} 批，{len(rows)} 条，{tokens} 个音频 token，{time.perf_counter() - start:.0f} 秒")
    flush()
    del model
    torch.cuda.empty_cache()
    rows.sort(key = lambda row: row["index"])
    lines = [json.dumps(obj = row) for row in rows]
    temporary = Path(out_dir) / (OUTPUT + ".tmp")
    temporary.write_text(data = "\n".join(lines) + "\n", encoding = "utf-8")
    temporary.replace(target = Path(out_dir) / OUTPUT)
    log(f"音频向量：{len(rows)} 条，{tokens} 个音频 token，{shard} 个文件，{time.perf_counter() - start:.0f} 秒")
    return rows


def run_encoder(model_dir: Path, projector: Path | None, items: list, job_dir: Path, log) -> dict:
    """在单独的进程中运行 encode（算完退出，显存全部释放），返回 {index: audio_items.out.jsonl 的一行}。"""
    import subprocess

    job_dir = Path(job_dir)
    lines = [json.dumps(obj = item, ensure_ascii = False) for item in items]
    (job_dir / "audio_items.jsonl").write_text(data = "\n".join(lines) + "\n", encoding = "utf-8")
    command = [
        sys.executable, "-m", "pipeline.eval.audio_embeds",
        "--model-dir", str(model_dir),
        "--items", str(job_dir / "audio_items.jsonl"),
        "--out-dir", str(job_dir),
    ]
    if projector is not None:
        command += ["--projector", str(projector)]
    log(f"算音频向量：{len(items)} 条")
    code = subprocess.run(args = command, cwd = REPO, stdin = subprocess.DEVNULL, check = False).returncode
    if code != 0 or not (job_dir / OUTPUT).exists():
        raise RuntimeError(f"算音频向量的进程退出码 {code}，任务目录 {job_dir} 保留")
    return {row["index"]: row for row in read_items(path = job_dir / OUTPUT)}


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required = True, type = Path, help = "融合模型目录")
    parser.add_argument("--projector", default = None, type = Path, help = "projector 检查点（safetensors，可选）")
    parser.add_argument("--items", required = True, type = Path, help = "请求列表（JSONL）")
    parser.add_argument("--out-dir", required = True, type = Path, help = "输出目录（任务目录）")
    args = parser.parse_args(args = argv)

    def log(message):
        print(f"{time.strftime('%H:%M:%S')} {message}", flush = True)

    encode(
        model_dir = args.model_dir,
        projector = args.projector,
        items = read_items(path = args.items),
        out_dir = args.out_dir,
        log = log,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
