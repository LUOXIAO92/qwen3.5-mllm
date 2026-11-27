"""回放差距（任务 #71）：训练后的语言模型在回放评测集上的每 token 负对数似然（NLL），减去参考模型的 NLL。

    flock <gpu.lock> python -m pipeline.eval.replay_gap --model-dir <融合模型目录> \\
        [--language-model <检查点>/language_model.safetensors] \\
        --replay-dir <回放目录> --output-dir <输出目录>

回放文本是参考模型 p_ref 自己采样的，在这些文本上 E[log p_ref(x) − log p_θ(x)] 就是每 token 的前向 KL(p_ref ‖ p_θ)
（arXiv 2605.26097 式 1）。所以「训练后模型的 NLL 减去参考模型的 NLL」是前向 KL 的估计：0 表示在这类文本上与参考模型
一样，越大忘得越多。

- 样本：回放目录中每组 --split（默认 eval，不参与任何训练）的全部保留样本，用保存的 token id，不重新分词。
- 计 NLL 的 token 与训练的损失相同（pipeline/train/replay.py）：free_raw 全部生成 token，cont_raw 不含文档开头，
  free_chat、cont_chat 只算助手轮（含结尾 <|im_end|>），free_chat_think 的助手轮含推理和 </think>。free_chat_think 另把
  助手轮分成推理（到 </think> 为止）和回答两部分。
- 参考模型的 NLL：teacher/lp.f32（参考模型用同一套 transformers 实现、bfloat16 前向算出的实际 token 对数概率）；没有
  teacher/ 时用采样时记录的 raw/lp.f32。不给 --language-model 时评测的就是参考模型本身，差距应接近 0，可以用来核对。
- 模型与精度：run_eval.load_model 加载（bfloat16），只用其中的语言模型做纯文本前向；样本按长度排序，凑成填充后
  不超过 --max-batch-tokens 个 token 的批次，右侧填充、不用注意力掩码（因果注意力和线性注意力都只看前文，右侧填充
  不影响前面位置）。词表投影只在计 NLL 的位置上分块计算，float32。

输出 summary.json 与 summary.md：每组的样本数、token 数、NLL、参考 NLL、差距（前向 KL 估计），按语言（续写两组用文档
语言 doc_lang，其余用生成文本的 lang）分开的同样几项；ALL 是各组差距的无权重平均，另给按 token 加权的平均。

vLLM 引擎（--engine vllm，任务 #71，见 pipeline/eval/vllm_host.py）：每条样本整条作为提示交给 vLLM，只生成 1 个 token，
prompt_logprobs = 0 返回提示中每个位置实际 token 的对数概率（bfloat16 的 logits 转成 float32 后对全词表做 log_softmax，
与上面的计算相同），取计 NLL 的位置。样本、计 NLL 的位置、参考 NLL（仍是 teacher/lp.f32）与汇总都不变；summary.json 另有
engine 与 vllm。两种引擎的 NLL 只差在内核的数值上（vLLM 的线性注意力与 FlashAttention 内核、批的组成），参考模型自己的
差距因此不严格为 0。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

GROUPS = ("free_raw", "free_chat", "cont_raw", "cont_chat", "free_chat_think")
PAD_ID = 248044
THINK_END_ID = 248069


def batches_by_tokens(lengths, max_batch_tokens):
    """按长度升序把样本下标分批，每批「条数 × 最长」不超过 max_batch_tokens（单条更长时自己一批）。"""
    order = sorted(range(len(lengths)), key = lambda index: lengths[index])
    batches, batch = [], []
    for index in order:
        if batch and (len(batch) + 1) * lengths[index] > max_batch_tokens:
            batches.append(batch)
            batch = []
        batch.append(index)
    if batch:
        batches.append(batch)
    return batches


def think_parts(sample):
    """free_chat_think 计 NLL 的位置分成推理（到 </think> 为止，含）和回答，返回与计 NLL 位置等长的布尔数组（真为推理）。"""
    import numpy as np

    targets = sample["ids"][sample["labels"] != -100]
    ends = np.nonzero(targets == THINK_END_ID)[0]
    reasoning = np.zeros(shape = len(targets), dtype = bool)
    if len(ends):
        reasoning[: ends[0] + 1] = True
    return reasoning


def score_batch(model, samples, chunk_size = 1024):
    """一批样本的逐 token NLL（float32，按样本依次排列，每条是计 NLL 位置的数组）。"""
    import torch

    device = next(model.parameters()).device
    longest = max(len(sample["ids"]) for sample in samples)
    input_ids = torch.full(size = (len(samples), longest), fill_value = PAD_ID, dtype = torch.long)
    rows, columns, targets = [], [], []
    for row, sample in enumerate(samples):
        ids = torch.from_numpy(sample["ids"])
        input_ids[row, : len(ids)] = ids
        positions = torch.from_numpy(sample["labels"] != -100).nonzero(as_tuple = True)[0]
        rows.append(torch.full(size = (len(positions),), fill_value = row, dtype = torch.long))
        columns.append(positions - 1)
        targets.append(ids[positions])
    rows, columns, targets = torch.cat(tensors = rows), torch.cat(tensors = columns), torch.cat(tensors = targets)
    with torch.inference_mode():
        hidden = model.model.language_model(input_ids = input_ids.to(device = device), use_cache = False).last_hidden_state
        flat = hidden.reshape(-1, hidden.shape[-1]).index_select(dim = 0, index = (rows * longest + columns).to(device = device))
        weight = model.lm_head.weight
        targets = targets.to(device = device)
        nll = torch.empty(size = (len(targets),), dtype = torch.float32, device = device)
        for start in range(0, len(targets), chunk_size):
            end = min(start + chunk_size, len(targets))
            logits = torch.nn.functional.linear(input = flat[start:end], weight = weight).float()
            nll[start:end] = torch.logsumexp(input = logits, dim = -1) - logits.gather(dim = -1, index = targets[start:end, None])[:, 0]
    nll = nll.cpu().numpy()
    out = []
    start = 0
    for sample in samples:
        out.append(nll[start: start + sample["n_loss"]])
        start += sample["n_loss"]
    return out


def add(total, key, tokens, nll, ref):
    entry = total.setdefault(key, {"samples": 0, "tokens": 0, "nll_sum": 0.0, "ref_sum": 0.0})
    entry["samples"] += 1
    entry["tokens"] += int(tokens)
    entry["nll_sum"] += float(nll)
    entry["ref_sum"] += float(ref)


def finish(entry):
    tokens = max(entry["tokens"], 1)
    nll = entry["nll_sum"] / tokens
    ref = entry["ref_sum"] / tokens
    return {"samples": entry["samples"], "tokens": entry["tokens"], "nll": round(number = nll, ndigits = 5), "ref_nll": round(number = ref, ndigits = 5), "gap": round(number = nll - ref, ndigits = 5)}


def evaluate_group(model, root, group, split, max_batch_tokens, limit = None):
    """一组的 NLL 与参考 NLL，返回 {整体与按语言、思考组的推理与回答}。"""
    from pipeline.train.replay import ReplayGroup

    reader = ReplayGroup(root = root, group = group, split = split)
    count = len(reader) if limit is None else min(limit, len(reader))
    lengths = [int(reader.totals[index]) for index in range(count)]
    scored = []
    for indices in batches_by_tokens(lengths = lengths, max_batch_tokens = max_batch_tokens):
        samples = [reader.sample(index = index, reference = True) for index in indices]
        scored += list(zip(samples, score_batch(model = model, samples = samples)))
    return aggregate_group(group = group, scored = scored)


def group_samples(root, group, split, limit = None):
    """一组要评测的样本（reader.sample，含参考对数概率）。"""
    from pipeline.train.replay import ReplayGroup

    reader = ReplayGroup(root = root, group = group, split = split)
    count = len(reader) if limit is None else min(limit, len(reader))
    return [reader.sample(index = index, reference = True) for index in range(count)]


def nll_from_prompt_logprobs(sample, values):
    """vLLM 的 prompt_logprobs（提示中每个位置实际 token 的对数概率，第一个位置为 None）在计 NLL 的位置上的负值（float32）。"""
    import numpy as np

    positions = np.nonzero(sample["labels"] != -100)[0]
    if len(positions) and positions[0] < 1:
        raise ValueError("第一个位置没有前文，不能计 NLL")
    return -np.asarray(a = [values[int(position)] for position in positions], dtype = np.float32)


def evaluate_vllm(args, log):
    """vLLM 引擎：全部组的样本一起交给 vLLM（一个容器），返回 ({组: 结果}, vLLM 的记录)。"""
    from pipeline.eval import vllm_host
    from pipeline.eval.think_check import STOP_IDS

    samples = {group: group_samples(root = args.replay_dir, group = group, split = args.split, limit = args.limit) for group in args.groups}
    requests = []
    owners = []
    for group in args.groups:
        for sample in samples[group]:
            owners.append((group, sample))
            requests.append({
                "index": len(requests),
                "seed": 0,
                "prompt_token_ids": [int(token) for token in sample["ids"]],
                "max_tokens": 1,
                "prompt_logprobs": 0,
            })
    settings = vllm_host.job_settings(
        args = args,
        max_model_len = max(len(request["prompt_token_ids"]) for request in requests) + 1,
        max_tokens = 1,
        temperature = 0.0,
        stop_token_ids = list(STOP_IDS),
    )
    log(f"{len(requests)} 条样本，{sum(len(request['prompt_token_ids']) for request in requests)} 个 token")
    results, record = vllm_host.run_job(
        args = args,
        model_dir = args.model_dir,
        language_model = args.language_model,
        requests = requests,
        settings = settings,
        kind = "replay",
        label = args.output_dir.parent.name,
        log = log,
    )
    scored = {group: [] for group in args.groups}
    for request, (group, sample) in zip(requests, owners):
        scored[group].append((sample, nll_from_prompt_logprobs(sample = sample, values = results[request["index"]]["prompt_logprobs"])))
    return {group: aggregate_group(group = group, scored = scored[group]) for group in args.groups}, record


def aggregate_group(group, scored):
    """[(样本, 逐 token NLL)] 汇总成一组的结果（整体、按语言、思考组的推理与回答）。"""
    totals = {}
    for sample, nll in scored:
        ref = -sample["ref_lp"]
        record = sample["record"]
        lang = record.get("doc_lang") or record.get("lang") or "unknown"
        add(
            total = totals,
            key = "ALL",
            tokens = len(nll),
            nll = nll.sum(),
            ref = ref.sum(),
        )
        add(
            total = totals,
            key = f"lang:{lang}",
            tokens = len(nll),
            nll = nll.sum(),
            ref = ref.sum(),
        )
        if group == "free_chat_think":
            reasoning = think_parts(sample = sample)
            for name, mask in (("reasoning", reasoning), ("answer", ~reasoning)):
                if mask.any():
                    add(
                        total = totals,
                        key = f"part:{name}",
                        tokens = mask.sum(),
                        nll = nll[mask].sum(),
                        ref = ref[mask].sum(),
                    )
    result = finish(entry = totals.pop("ALL"))
    result["per_lang"] = {key[5:]: finish(entry = value) for key, value in sorted(totals.items()) if key.startswith("lang:")}
    parts = {key[5:]: finish(entry = value) for key, value in totals.items() if key.startswith("part:")}
    if parts:
        result["parts"] = parts
    return result


def format_markdown(summary):
    lines = [
        f"模型：{summary['model']}",
        "",
        "| 组 | 样本 | token | NLL | 参考 NLL | 差距（前向 KL 估计） |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group, item in summary["groups"].items():
        lines.append(f"| {group} | {item['samples']} | {item['tokens']} | {item['nll']:.4f} | {item['ref_nll']:.4f} | {item['gap']:.4f} |")
        for part, value in item.get("parts", {}).items():
            lines.append(f"| {group}·{part} | {value['samples']} | {value['tokens']} | {value['nll']:.4f} | {value['ref_nll']:.4f} | {value['gap']:.4f} |")
    lines.append(f"| ALL（各组平均） |  |  |  |  | {summary['mean_gap']:.4f} |")
    lines.append(f"| ALL（按 token 加权） |  | {summary['tokens']} |  |  | {summary['token_weighted_gap']:.4f} |")
    return "\n".join(lines) + "\n"


def main(argv = None):
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model-dir",
        required = True,
        type = Path,
        help = "融合模型目录",
    )
    parser.add_argument(
        "--language-model",
        default = None,
        type = Path,
        help = "全量微调检查点中的 language_model.safetensors；不给时评测参考模型本身",
    )
    parser.add_argument(
        "--replay-dir",
        required = True,
        type = Path,
        help = "回放数据目录（replay-noloop）",
    )
    parser.add_argument(
        "--groups",
        nargs = "+",
        default = list(GROUPS),
        choices = GROUPS,
    )
    parser.add_argument("--split", default = "eval", choices = ("eval", "train"))
    parser.add_argument(
        "--max-batch-tokens",
        type = int,
        default = 32768,
        help = "每批填充后的 token 上限",
    )
    parser.add_argument(
        "--limit",
        type = int,
        default = None,
        help = "每组只用前若干条（快速核对用）",
    )
    parser.add_argument("--output-dir", required = True, type = Path)
    from pipeline.eval import vllm_host

    vllm_host.add_arguments(parser = parser)
    args = parser.parse_args(args = argv)
    vllm_host.check_arguments(
        parser = parser,
        args = args,
        default_job_dir = args.output_dir / "vllm-job",
        language_model = args.language_model,
    )

    def log(message):
        print(f"{time.strftime('%H:%M:%S')} {message}", flush = True)

    args.output_dir.mkdir(parents = True, exist_ok = True)
    groups = {}
    record = None
    start = time.perf_counter()
    if args.engine == "vllm":
        groups, record = evaluate_vllm(args = args, log = log)
    else:
        from pipeline.eval.run_eval import load_model

        model, _ = load_model(model_dir = args.model_dir, projector_path = None, language_model_path = args.language_model)
        for group in args.groups:
            groups[group] = evaluate_group(
                model = model,
                root = args.replay_dir,
                group = group,
                split = args.split,
                max_batch_tokens = args.max_batch_tokens,
                limit = args.limit,
            )
    for group, item in groups.items():
        print(f"{group}: {item['tokens']} token，NLL {item['nll']:.4f}，参考 {item['ref_nll']:.4f}，差距 {item['gap']:.4f}", flush = True)
    tokens = sum(item["tokens"] for item in groups.values())
    summary = {
        "model": str(args.language_model or args.model_dir),
        "split": args.split,
        "groups": groups,
        "mean_gap": round(number = sum(item["gap"] for item in groups.values()) / len(groups), ndigits = 5),
        "token_weighted_gap": round(number = sum(item["gap"] * item["tokens"] for item in groups.values()) / max(tokens, 1), ndigits = 5),
        "tokens": tokens,
        "seconds": round(number = time.perf_counter() - start, ndigits = 1),
        "engine": args.engine,
        **({"vllm": record} if record is not None else {}),
    }
    (args.output_dir / "summary.json").write_text(data = json.dumps(obj = summary, ensure_ascii = False, indent = 1) + "\n", encoding = "utf-8")
    (args.output_dir / "summary.md").write_text(data = format_markdown(summary = summary), encoding = "utf-8")
    print(json.dumps(obj = {key: summary[key] for key in ("mean_gap", "token_weighted_gap", "tokens", "seconds")}), flush = True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
