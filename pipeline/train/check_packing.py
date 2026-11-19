"""在真实模型和真实训练批次上核对拼接批次（--pack-sequences，任务 #66）。

用法：在 train.py 的命令行前加本脚本的选项，批次按 train.py 的训练计划取出（同样的清单、分片、混音和种子）：

    python -m pipeline.train.check_packing --steps 0,1,2 --leak-steps 0 [--fp32-reference] \\
        --model-dir <模型目录> --train-manifest <清单> --target original --output-dir <报告目录> \\
        --batch-size 16 --max-duration-s 30 --max-batch-seconds 360 [--audio-shards ...] [--noise-manifest ...] \\
        [--train-language-model --language-model-lr 1e-5] [--checkpoint-layers N]

报告写到 <输出目录>/check_packing.json，并打印摘要。核对三项：

1. 损失与梯度：同一批先按填充的批次（sdpa，原实现）、再按拼接的批次（flash_attention_2 加样本边界）各做一次前向
   反传，比较每个计损失 token 的损失、projector 的梯度，以及 --train-language-model 时语言模型全部参数的梯度。
   --fp32-reference 时另用 float32 模型算填充的批次作参考，分别给出两种批次与参考的差距：拼接若只是换了计算顺序，
   它与参考的差距应与填充的批次相当。
2. 串扰：拼接的批次中只改动一条样本（把它的音频倒放，时长不变；或把它的目标文本 token 换成别的 token，长度
   不变），其余样本在语言模型最后一层的隐状态应当逐位不变。另把线性注意力的短卷积和递推换回 transformers 的
   PyTorch 参考实现再做一次，作对照：参考实现不认样本边界，改动会传到后面的样本。
3. 内核：用 torch.profiler 记下拼接批次一次前向反传中的内核，确认用的是 flash-attn 的变长注意力、causal-conv1d
   和 fla 的分段内核，没有 transformers 参考实现的 aten::conv1d。
"""

import argparse
import copy
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from pipeline.train.data import IGNORE_INDEX, TrainCollator, TranscriptionDataset, pack_batch
from pipeline.train.mixing import BackgroundMixer, parse_category_weights
from pipeline.train.model import (
    LANGUAGE_MODEL_PREFIXES,
    PACKED_ATTENTION,
    PROJECTOR_PREFIX,
    check_packing_kernels,
    hidden_states_precomputed,
    load_model,
    move_batch,
    token_losses,
)
from pipeline.train.optimize import apply as apply_optimizations
from pipeline.train.optimize import options_from_args
from pipeline.train.shards import attach_shards, parse_dirs, read_header
from pipeline.train.train import make_batch_stream, parse_args, training_rows, training_weights


def parse_own(argv):
    parser = argparse.ArgumentParser(description = "核对拼接批次（其余参数与 train.py 相同）", add_help = False)
    parser.add_argument("--steps", default = "0,1,2", help = "核对损失与梯度的训练步，逗号分隔")
    parser.add_argument("--leak-steps", default = "0", help = "做串扰核对的训练步，逗号分隔")
    parser.add_argument("--fp32-reference", action = "store_true", help = "另用 float32 模型算填充批次的参考值")
    return parser.parse_known_args(args = argv)


def build_items(args, steps):
    """按 train.py 的训练计划取出各步的样本（读出音频并按计划混音），返回 {步: 样本列表}。"""
    rows = training_rows(args = args)
    shards = None
    if args.audio_shards:
        shards, _ = attach_shards(
            rows = rows,
            header = read_header(index_path = args.audio_shards),
            dirs = parse_dirs(spec = args.audio_shard_dirs),
            fallback = args.audio_shard_fallback,
        )
    mixer = None
    if args.noise_manifest:
        mixer = BackgroundMixer(
            manifest = args.noise_manifest,
            prob = args.noise_prob,
            snr_min = args.noise_snr_min,
            snr_max = args.noise_snr_max,
            category_weights = parse_category_weights(specs = args.noise_category_weight),
            audio_root = args.noise_audio_root,
        )
    dataset = TranscriptionDataset(
        rows = rows,
        target = args.target,
        mixer = mixer,
        seed = args.seed,
        storage = Path(args.rows_cache_dir) / "rows.jsonl.bin" if args.rows_cache_dir else None,
        max_chars_per_second = args.max_target_chars_per_second or None,
        shards = shards,
    )
    weights = training_weights(args = args, rows = rows, empty = dataset.empty)
    plan = make_batch_stream(args = args, rows = rows, weights = weights).plan
    occurrences, batches = plan.position_batches(epoch = 0)
    items = {}
    for step in steps:
        keys = plan.keys(occurrences = occurrences, epoch = 0, batch = batches[step * args.grad_accum])
        items[step] = [item for item in (dataset[key] for key in keys) if item is not None]
    return items


def trainable_names(model, train_language_model):
    return [
        name for name, _ in model.named_parameters()
        if name.startswith(PROJECTOR_PREFIX) or (train_language_model and name.startswith(LANGUAGE_MODEL_PREFIXES))
    ]


def set_packed(model, packed):
    """在同一个模型上切换全注意力层的实现：填充的批次用 sdpa（原实现），拼接的批次用 flash_attention_2。"""
    model.set_attn_implementation(attn_implementation = PACKED_ATTENTION if packed else "sdpa")


def losses_and_grads(model, batch, names, feature_dtype):
    """一次前向反传（按计损失 token 数取平均，与训练相同），返回 (每个 token 的损失, 所在样本, {参数名: CPU 上的梯度})。"""
    model.zero_grad(set_to_none = True)
    inputs = move_batch(batch = batch, device = "cuda", feature_dtype = feature_dtype)
    losses, _, rows = token_losses(model = model, inputs = inputs)
    (losses.sum() / losses.numel()).backward()
    parameters = dict(model.named_parameters())
    grads = {name: parameters[name].grad.detach().to(device = "cpu", copy = True) for name in names}
    model.zero_grad(set_to_none = True)
    return losses.detach().float().cpu(), rows.cpu(), grads


def summarize_grads(reference, result):
    """梯度的相对误差 ||差|| / ||参考||：projector 与语言模型各自把全部参数接成一个向量计算，另给出误差最大的参数。"""
    norms = {}
    for name, want in reference.items():
        want = want.float()
        norms[name] = (float((want - result[name].float()).norm()), float(want.norm()))

    def combined(names):
        difference = sum(norms[name][0] ** 2 for name in names) ** 0.5
        return difference / max(sum(norms[name][1] ** 2 for name in names) ** 0.5, 1e-30)

    errors = {name: difference / max(norm, 1e-30) for name, (difference, norm) in norms.items()}
    projector = [name for name in norms if name.startswith(PROJECTOR_PREFIX)]
    language = [name for name in norms if not name.startswith(PROJECTOR_PREFIX)]
    summary = {
        "projector_relative": combined(names = projector),
        "projector_max_param_relative": max(errors[name] for name in projector),
    }
    if language:
        summary["language_model_relative"] = combined(names = language)
        worst = sorted(language, key = lambda name: errors[name], reverse = True)[:5]
        summary["language_model_worst_params"] = {
            name: {"relative": errors[name], "reference_norm": norms[name][1]} for name in worst
        }
    return summary


def loss_summary(reference, result):
    difference = (reference - result).abs()
    return {
        "tokens": int(reference.numel()),
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "mean_loss_reference": float(reference.mean()),
        "mean_loss_result": float(result.mean()),
    }


def check_losses_and_grads(model, items, collator, names, feature_dtype):
    """各步在填充与拼接两种批次上的损失与梯度。返回 {步: {"padded": ..., "packed": ...}}（损失、梯度在 CPU 上）。"""
    results = {}
    for step, step_items in items.items():
        padded = collator(items = step_items)
        packed = pack_batch(batch = copy.copy(padded))
        entry = {}
        for label, batch, use_packed in (("padded", padded, False), ("packed", packed, True)):
            set_packed(model = model, packed = use_packed)
            entry[label] = losses_and_grads(
                model = model,
                batch = batch,
                names = names,
                feature_dtype = feature_dtype,
            )
        mask = padded["attention_mask"].bool()
        entry["shape"] = {
            "samples": len(step_items),
            "padded_tokens": int(mask.numel()),
            "real_tokens": int(mask.sum()),
            "padding_fraction": round(number = 1 - float(mask.sum()) / mask.numel(), ndigits = 4),
            "loss_tokens": int(padded["loss_targets"].numel()),
        }
        results[step] = entry
    return results


@contextmanager
def reference_linear_attention():
    """把线性注意力的短卷积和递推临时换回 transformers 的 PyTorch 参考实现（不认样本边界），作串扰核对的对照。"""
    from transformers.models.qwen3_5 import modeling_qwen3_5

    saved = {name: getattr(modeling_qwen3_5, name) for name in ("causal_conv1d_fn", "torch_chunk_gated_delta_rule")}
    try:
        for name, function in saved.items():
            setattr(modeling_qwen3_5, name, function.__wrapped__)
        yield
    finally:
        for name, function in saved.items():
            setattr(modeling_qwen3_5, name, function)


def packed_hidden(model, batch, feature_dtype):
    inputs = move_batch(batch = batch, device = "cuda", feature_dtype = feature_dtype)
    with torch.no_grad():
        return hidden_states_precomputed(model = model, inputs = inputs)[0].float().cpu()


def per_sample_difference(first, second, starts):
    return [float((first[start:end] - second[start:end]).abs().max()) for start, end in zip(starts[:-1], starts[1:])]


def check_leakage(model, items, collator, feature_dtype):
    """拼接的批次中改动第 j 条样本（中间一条），比较各条样本的隐状态。"""
    set_packed(model = model, packed = True)
    original = pack_batch(batch = collator(items = items))
    starts = original["cu_seq_lens_q"].tolist()
    target = len(items) // 2
    # 改音频：第 j 条倒放，时长和 token 数不变。
    flipped_items = [dict(item) for item in items]
    flipped_items[target]["waveform"] = np.flip(flipped_items[target]["waveform"]).copy()
    flipped = pack_batch(batch = collator(items = flipped_items))
    # 改文字：第 j 条计损失的 token（目标文本与 <|im_end|>）在输入中换成别的 token，长度不变。
    retexted = copy.copy(original)
    input_ids = original["input_ids"].clone()
    positions = original["loss_index"][original["loss_rows"] == target] + 1
    input_ids[0, positions] = (input_ids[0, positions] + 1000) % 200000
    retexted["input_ids"] = input_ids
    base = packed_hidden(model = model, batch = original, feature_dtype = feature_dtype)
    report = {"changed_sample": target, "samples": len(items)}
    for label, batch in (("audio_reversed", flipped), ("text_tokens_replaced", retexted)):
        differences = per_sample_difference(
            first = base,
            second = packed_hidden(model = model, batch = batch, feature_dtype = feature_dtype),
            starts = starts,
        )
        report[label] = {
            "changed_sample_max_abs": differences[target],
            "other_samples_max_abs": max(value for index, value in enumerate(differences) if index != target),
            "later_samples_max_abs": max(differences[target + 1:], default = 0.0),
        }
    # 对照：参考实现不认样本边界，改动会传到后面的样本。
    with reference_linear_attention():
        fallback_base = packed_hidden(model = model, batch = original, feature_dtype = feature_dtype)
        differences = per_sample_difference(
            first = fallback_base,
            second = packed_hidden(model = model, batch = flipped, feature_dtype = feature_dtype),
            starts = starts,
        )
    report["control_reference_linear_attention_audio_reversed"] = {
        "earlier_samples_max_abs": max(differences[:target], default = 0.0),
        "later_samples_max_abs": max(differences[target + 1:], default = 0.0),
    }
    return report


def kernel_report(model, items, collator, feature_dtype):
    """profiler 记下拼接批次一次前向反传的内核与算子，按关键字计数。"""
    from torch.profiler import ProfilerActivity, profile

    set_packed(model = model, packed = True)
    batch = pack_batch(batch = collator(items = items))
    model.zero_grad(set_to_none = True)
    inputs = move_batch(batch = batch, device = "cuda", feature_dtype = feature_dtype)
    with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        losses, _, _ = token_losses(model = model, inputs = inputs)
        losses.mean().backward()
        torch.cuda.synchronize()
    model.zero_grad(set_to_none = True)
    counts = {}
    for event in profiler.key_averages():
        counts[event.key] = counts.get(event.key, 0) + event.count

    def matching(word):
        # 模板展开的内核名很长，只留前 100 个字符。
        return {
            key[:100]: count for key, count in counts.items()
            if word in key.lower() and not key.startswith("aten::")
        }

    return {
        "flash_attn": matching(word = "flash"),
        "causal_conv1d": matching(word = "causal_conv1d"),
        "fla_chunk": matching(word = "chunk"),
        "aten_conv1d_ops": counts.get("aten::conv1d", 0),
        "aten_scaled_dot_product_attention_ops": counts.get("aten::scaled_dot_product_attention", 0),
    }


def main(argv = None):
    own, rest = parse_own(argv = sys.argv[1:] if argv is None else argv)
    args = parse_args(argv = rest)
    steps = [int(step) for step in own.steps.split(sep = ",") if step]
    leak_steps = [int(step) for step in own.leak_steps.split(sep = ",") if step]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents = True, exist_ok = True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    start = time.perf_counter()
    items = build_items(args = args, steps = sorted(set(steps + leak_steps)))
    print(f"取出 {len(items)} 步的样本，用时 {time.perf_counter() - start:.0f} 秒", flush = True)

    model, processor = load_model(
        model_dir = args.model_dir,
        attn_implementation = PACKED_ATTENTION,
        train_language_model = args.train_language_model,
    )
    report = {"kernels_in_use": check_packing_kernels(model = model), "config": vars(args) | vars(own)}
    feature_dtype = model.model.audio_tower.dtype
    apply_optimizations(
        model = model,
        options = options_from_args(args = args),
        train_language_model = args.train_language_model,
    )
    audio_config = model.config.audio_config
    collator = TrainCollator(processor = processor, audio_windows = (audio_config.n_window, audio_config.n_window_infer))
    names = trainable_names(model = model, train_language_model = args.train_language_model)

    path = output_dir / "check_packing.json"

    def write_report():
        path.write_text(data = json.dumps(obj = report, ensure_ascii = False, indent = 1, default = str) + "\n", encoding = "utf-8")

    report["kernels"] = kernel_report(model = model, items = items[leak_steps[0]], collator = collator, feature_dtype = feature_dtype)
    report["leakage"] = {
        step: check_leakage(model = model, items = items[step], collator = collator, feature_dtype = feature_dtype)
        for step in leak_steps
    }
    write_report()
    print(json.dumps(obj = {"kernels": report["kernels"], "leakage": report["leakage"]}, ensure_ascii = False, indent = 1), flush = True)
    print(f"内核与串扰核对完成，已用 {time.perf_counter() - start:.0f} 秒", flush = True)
    results = check_losses_and_grads(
        model = model,
        items = {step: items[step] for step in steps},
        collator = collator,
        names = names,
        feature_dtype = feature_dtype,
    )
    del model
    torch.cuda.empty_cache()

    references = {}
    if own.fp32_reference:
        model, _ = load_model(
            model_dir = args.model_dir,
            dtype = torch.float32,
            train_language_model = args.train_language_model,
        )
        # float32 的中间结果是 bfloat16 的两倍，全部层做梯度检查点；音频塔的变长注意力只支持半精度，改用逐窗口的 sdpa。
        options = options_from_args(args = args) | {
            "gradient_checkpointing": True,
            "checkpoint_layers": None,
            "audio_attention": "sdpa",
        }
        apply_optimizations(model = model, options = options, train_language_model = args.train_language_model)
        for step in steps:
            references[step] = losses_and_grads(
                model = model,
                batch = collator(items = items[step]),
                names = names,
                feature_dtype = torch.float32,
            )
        del model
        torch.cuda.empty_cache()

    report["steps"] = {}
    for step, entry in results.items():
        padded_losses, padded_rows, padded_grads = entry["padded"]
        packed_losses, packed_rows, packed_grads = entry["packed"]
        if not torch.equal(padded_rows, packed_rows):
            raise RuntimeError(f"第 {step} 步两种批次计损失的 token 不一一对应")
        summary = summarize_grads(reference = padded_grads, result = packed_grads)
        step_report = {
            "batch": entry["shape"],
            "packed_vs_padded": {"loss": loss_summary(reference = padded_losses, result = packed_losses), "grads": summary},
        }
        if step in references:
            reference_losses, _, reference_grads = references[step]
            for label, (losses, grads) in (("padded", (padded_losses, padded_grads)), ("packed", (packed_losses, packed_grads))):
                grads_summary = summarize_grads(reference = reference_grads, result = grads)
                step_report[f"{label}_vs_fp32"] = {
                    "loss": loss_summary(reference = reference_losses, result = losses),
                    "grads": grads_summary,
                }
        report["steps"][step] = step_report
        print(json.dumps(obj = {"step": step, **step_report}, ensure_ascii = False, indent = 1), flush = True)
    write_report()
    print(f"报告：{path}，共用 {time.perf_counter() - start:.0f} 秒", flush = True)


if __name__ == "__main__":
    main()
