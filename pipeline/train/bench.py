"""测量不同批量下训练一步的显存与吞吐。

用法：

    python -m pipeline.train.bench \\
        --model-dir <阶段 1 模型目录> --manifest <清单.jsonl> --batch-sizes 8,16,32,64 --output <结果.json>

对每个批量，按训练时的方式（按时长分组成批、数据读取进程、前向、反传、优化器更新）先跑
--warmup 步再计时 --steps 步，记录每秒样本数、每秒处理的音频秒数和峰值显存。热身与计时用同一组
读取进程，读取进程的启动和 Triton 内核的自动调优都落在热身中。另用清单中最长的样本组成一批跑一步，
得到这个批量下显存的上限。

WSL 下显存用满后会悄悄借用内存、速度骤降而不报错，所以这里把 PyTorch 可用的显存限制在
--memory-fraction（默认 0.95）以内，超出时记为 out of memory。

显存与速度选项（梯度检查点、块流式加载、torch.compile 等，见 optimize.py）与 train.py 相同。
这里只测模型本身，批次固定、不混音；与真实训练一致的端到端比较（加权抽样、混音、读取进程）直接用 train.py
跑若干步，见 pipeline/train/__init__.py「显存与吞吐」。
"""

import argparse
import itertools
import json
import os
import time

import torch

from pipeline.train.data import TrainCollator, TranscriptionDataset, length_grouped_batches, read_manifest
from pipeline.train.model import load_model, move_batch, projector_parameters, target_loss
from pipeline.train.optimize import add_arguments, apply, options_from_args

# 由 main 设置：apply 的结果，前向时提供激活值卸载的上下文。
OPTIMIZED = None


def train_step(model, optimizer, batch, feature_dtype):
    """训练一步，返回计入序列的 token 数（含音频标记）。"""
    inputs = move_batch(batch = batch, device = "cuda", feature_dtype = feature_dtype)
    with OPTIMIZED.forward_context():
        loss_sum, n_tokens, _ = target_loss(model = model, inputs = inputs)
    (loss_sum / n_tokens).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none = True)
    return int(inputs["attention_mask"].sum())


def measure(
    model,
    optimizer,
    loader,
    feature_dtype,
    warmup,
):
    """先跑 warmup 步，再把余下的批次计时，返回 (样本数, 音频秒数, token 数, 秒数, 步数)。"""
    batches = iter(loader)
    for batch in itertools.islice(batches, warmup):
        train_step(
            model = model,
            optimizer = optimizer,
            batch = batch,
            feature_dtype = feature_dtype,
        )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples, audio, tokens, steps = 0, 0.0, 0, 0
    start = time.perf_counter()
    for batch in batches:
        tokens += train_step(
            model = model,
            optimizer = optimizer,
            batch = batch,
            feature_dtype = feature_dtype,
        )
        samples += len(batch["sample_ids"])
        audio += batch["audio_seconds"]
        steps += 1
    torch.cuda.synchronize()
    return samples, audio, tokens, time.perf_counter() - start, steps


def main(argv = None):
    parser = argparse.ArgumentParser(description = "训练显存与吞吐测量")
    parser.add_argument("--model-dir", required = True)
    parser.add_argument("--manifest", required = True)
    parser.add_argument("--audio-root", default = None)
    parser.add_argument("--target", default = "script", choices = ("script", "original"))
    parser.add_argument("--max-duration-s", type = float, default = None)
    parser.add_argument("--batch-sizes", default = "8,16,32,64")
    parser.add_argument("--warmup", type = int, default = 4)
    parser.add_argument("--steps", type = int, default = 20)
    parser.add_argument("--num-workers", type = int, default = 8)
    parser.add_argument("--output", default = None)
    parser.add_argument("--deterministic", action = "store_true", help = "使用确定性算法，测量其开销")
    add_arguments(parser = parser)
    args = parser.parse_args(args = argv)
    if args.memory_fraction is None:
        args.memory_fraction = 0.95
    os.environ.setdefault(key = "TOKENIZERS_PARALLELISM", value = "false")
    if args.deterministic:
        os.environ.setdefault(key = "CUBLAS_WORKSPACE_CONFIG", value = ":4096:8")
        torch.use_deterministic_algorithms(mode = True)
    global OPTIMIZED

    rows = read_manifest(path = args.manifest, audio_root = args.audio_root, max_duration_s = args.max_duration_s)
    model, processor = load_model(model_dir = args.model_dir)
    OPTIMIZED = apply(model = model, options = options_from_args(args = args))
    feature_dtype = model.model.audio_tower.dtype
    optimizer = torch.optim.AdamW(params = projector_parameters(model = model), lr = 1e-5)
    dataset = TranscriptionDataset(rows = rows, target = args.target)
    audio_config = model.config.audio_config
    collator = TrainCollator(
        processor = processor,
        audio_windows = (audio_config.n_window, audio_config.n_window_infer) if args.precompute_metadata else None,
    )
    durations = [row["duration_s"] for row in rows]
    longest = sorted(range(len(rows)), key = lambda index: durations[index], reverse = True)

    results = []
    for batch_size in [int(value) for value in args.batch_sizes.split(sep = ",")]:
        batches = length_grouped_batches(
            durations = durations,
            batch_size = batch_size,
            seed = 0,
            epoch = 0,
        )
        batches = [batch for batch in batches if len(batch) == batch_size][:args.warmup + args.steps]
        loader = torch.utils.data.DataLoader(
            dataset = dataset,
            batch_sampler = batches,
            collate_fn = collator,
            num_workers = args.num_workers,
            pin_memory = True,
            prefetch_factor = 4,
        )
        try:
            samples, audio, tokens, elapsed, steps = measure(
                model = model,
                optimizer = optimizer,
                loader = loader,
                feature_dtype = feature_dtype,
                warmup = args.warmup,
            )
            typical_peak = torch.cuda.max_memory_allocated()
            typical_reserved = torch.cuda.max_memory_reserved()
            torch.cuda.reset_peak_memory_stats()
            worst = collator(items = [dataset[index] for index in longest[:batch_size]])
            try:
                train_step(
                    model = model,
                    optimizer = optimizer,
                    batch = worst,
                    feature_dtype = feature_dtype,
                )
                worst_peak = round(number = torch.cuda.max_memory_allocated() / 2 ** 30, ndigits = 2)
            except torch.OutOfMemoryError:
                worst_peak = "out of memory"
                optimizer.zero_grad(set_to_none = True)
                torch.cuda.empty_cache()
            result = {
                "batch_size": batch_size,
                "steps": steps,
                "mean_audio_s": round(number = audio / samples, ndigits = 2),
                "samples_per_s": round(number = samples / elapsed, ndigits = 1),
                "audio_s_per_s": round(number = audio / elapsed, ndigits = 1),
                "tokens_per_s": round(number = tokens / elapsed),
                "step_s": round(number = elapsed / steps, ndigits = 3),
                "peak_mem_gb": round(number = typical_peak / 2 ** 30, ndigits = 2),
                "peak_reserved_gb": round(number = typical_reserved / 2 ** 30, ndigits = 2),
                "longest_audio_s": durations[longest[batch_size - 1]],
                "longest_batch_tokens": int(worst["input_ids"].numel()),
                "longest_batch_peak_mem_gb": worst_peak,
            }
        except torch.OutOfMemoryError:
            result = {"batch_size": batch_size, "error": "out of memory"}
            optimizer.zero_grad(set_to_none = True)
            torch.cuda.empty_cache()
        print(json.dumps(obj = result), flush = True)
        results.append(result)

    if args.output:
        with open(file = args.output, mode = "w", encoding = "utf-8") as handle:
            handle.write(json.dumps(obj = results, ensure_ascii = False, indent = 2) + "\n")


if __name__ == "__main__":
    main()
