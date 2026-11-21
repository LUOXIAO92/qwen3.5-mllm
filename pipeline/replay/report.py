"""汇总回放文本：数量、退化与截断、长度分布、语言分布、吞吐和参考模型 NLL，写出 summary.json 与 summary.md。

    python -m pipeline.replay.report --replay-dir <回放目录> --out-dir <输出目录>

读取 finalize、langid、nll 的结果和 shard 二进制中的对数概率，只用 CPU。两种 NLL：

- 训练写法 NLL（`nll`）：nll 一步按训练的渲染与分词重新前向得到，是以后比较「当前模型 − 参考模型」的基线；
- 采样时 NLL（`gen_nll`）：生成时记录的采样 token 自身的对数概率取负，按保留的生成 token 平均，
  分词就是采样到的 token 序列。两者相差很小，说明重新分词与 vLLM 数值都没有引入明显偏差。
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from pipeline.replay.prompts import GROUPS, SPLITS
from pipeline.replay.store import ShardReader

LENGTH_BINS = [(0, 64), (64, 256), (256, 1024), (1024, 4096), (4096, 16384), (16384, 10**9)]


def read_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path, encoding = "utf-8") as handle:
        return [json.loads(line) for line in handle]


def percentiles(values):
    if not values:
        return {}
    array = np.asarray(values, dtype = np.float64)
    return {f"p{q}": round(float(np.percentile(array, q)), 4) for q in (10, 50, 90, 99)} | {"mean": round(float(array.mean()), 4), "max": round(float(array.max()), 4)}


def gen_nll(rows, raw_dir):
    """采样时 NLL：保留的生成 token 的 -logprob，按部分（角色）汇总，并按文档语言分开。"""
    reader = ShardReader(raw_dir = raw_dir)
    sums = defaultdict(float)
    counts = Counter()
    by_lang = defaultdict(lambda: [0.0, 0])
    for row in rows:
        gen_pieces = [piece for piece in row["pieces"] if "gen" in piece]
        for piece in gen_pieces:
            if not piece["keep"]:
                continue
            lp = np.asarray(reader.piece(number = row["shard"], piece = piece, name = "lp"))[: piece["keep"]]
            role = piece["gen"]
            sums[role] += float(-lp.sum())
            counts[role] += len(lp)
            if piece is gen_pieces[-1]:
                # 主体：最后一个生成片段（free_chat 为助手回答，其余为唯一的生成片段）
                sums["main"] += float(-lp.sum())
                counts["main"] += len(lp)
            if row.get("doc_lang") and role != "user":
                by_lang[row["doc_lang"]][0] += float(-lp.sum())
                by_lang[row["doc_lang"]][1] += len(lp)
    result = {role: round(sums[role] / counts[role], 4) for role in counts}
    if by_lang:
        result["by_doc_lang"] = {lang: round(value[0] / value[1], 4) for lang, value in sorted(by_lang.items())}
    return result


def diversity(rows, raw_dir, budget = 200_000, opening = 8):
    """样本之间的多样性：前 budget 个保留生成 token 中不同 4-gram 的比例；开头 opening 个 token 与其他样本相同的样本比例。

    不同 4-gram 的比例看最后一个生成片段（free_chat 为助手回答），开头看第一个生成片段（free_chat 为用户消息）；
    按样本顺序取，各组用同样的 token 预算，可以直接比较。
    """
    from pipeline.replay.degeneration import ngram_hashes

    reader = ShardReader(raw_dir = raw_dir)
    chunks = []
    total = 0
    openings = Counter()
    for row in rows:
        pieces = [p for p in row["pieces"] if "gen" in p]
        first = np.asarray(reader.piece(number = row["shard"], piece = pieces[0]))[: pieces[0]["keep"]]
        openings[tuple(first[:opening].tolist())] += 1
        ids = np.asarray(reader.piece(number = row["shard"], piece = pieces[-1]))[: pieces[-1]["keep"]]
        if total < budget:
            chunks.append(ngram_hashes(ids = ids[: budget - total], n = 4))
            total += min(len(ids), budget - total)
    hashes = np.concatenate(chunks) if chunks else np.zeros(0, dtype = np.uint64)
    shared = sum(count for count in openings.values() if count > 1)
    return {
        "distinct4": round(len(np.unique(hashes)) / max(len(hashes), 1), 4),
        "distinct4_tokens": int(total),
        "shared_opening": round(shared / max(len(rows), 1), 4),
    }


def retokenized_ratio(rows, nll_rows, group):
    """重新分词后的 token 数 / 采样时的 token 数（同一批样本的主体部分），整体与按语言给出。

    两者不同说明分词器对模型自己写出的文本给出了另一种切分；比值明显大于 1 时，训练写法 NLL 按 token 平均会偏高。
    """
    main = "assistant" if group in ("free_chat", "cont_chat") else "generated"
    sampled = {row["sample_id"]: [p for p in row["pieces"] if "gen" in p][-1]["keep"] for row in rows}
    language = {row["sample_id"]: row.get("doc_lang") or row.get("lang") for row in rows}
    totals = defaultdict(lambda: [0, 0])
    for row in nll_rows:
        if row["sample_id"] not in sampled:
            continue
        for key in ("all", language[row["sample_id"]]):
            totals[key][0] += row[f"{main}_tokens"]
            totals[key][1] += sampled[row["sample_id"]]
    return {key: round(value[0] / max(value[1], 1), 4) for key, value in sorted(totals.items())}


def nll_summary(rows, group):
    if not rows:
        return None
    parts = ["user", "assistant"] if group == "free_chat" else (["assistant"] if group == "cont_chat" else ["generated"])
    result = {"samples": len(rows)}
    for part in parts:
        tokens = sum(row[f"{part}_tokens"] for row in rows)
        result[part] = {"tokens": tokens, "nll": round(sum(row[f"{part}_nll_sum"] for row in rows) / max(tokens, 1), 4)}
    main = parts[-1]
    for key in ("doc_lang", "lang"):
        groups = defaultdict(lambda: [0.0, 0])
        for row in rows:
            if row.get(key):
                groups[row[key]][0] += row[f"{main}_nll_sum"]
                groups[row[key]][1] += row[f"{main}_tokens"]
        if groups and (key == "doc_lang" or group.startswith("free")):
            result[f"by_{key}"] = {lang: {"tokens": value[1], "nll": round(value[0] / max(value[1], 1), 4)} for lang, value in sorted(groups.items())}
    return result


def split_summary(replay_dir, group, split):
    split_dir = Path(replay_dir) / group / split
    rows = read_jsonl(split_dir / "samples.jsonl")
    if not rows:
        return None
    kept = [row for row in rows if row["keep"]]
    runs = read_jsonl(split_dir / "raw" / "runs.jsonl")
    lengths = [row["n_kept"] for row in kept]
    lost_truncation = sum(row["n_gen"] - row["n_kept"] for row in kept)
    lost_dropped = sum(row["n_gen"] for row in rows if not row["keep"])
    summary = {
        "samples": len(rows),
        "kept_samples": len(kept),
        "dropped": dict(Counter(row["drop_reason"] for row in rows if not row["keep"])),
        "gen_tokens": sum(row["n_gen"] for row in rows),
        "kept_tokens": sum(lengths),
        "truncated_samples": sum(row["truncated"] for row in kept),
        "tokens_lost_truncation": lost_truncation,
        "tokens_lost_dropped": lost_dropped,
        "finish": dict(Counter(row["finish_reason"] for row in kept)),
        "ended_samples": sum(row["ended"] for row in kept),
        "hit_length_cap": sum(row["finish_reason"] == "length" for row in kept),
        "length": percentiles(values = lengths),
        "length_bins": {f"{low}-{high if high < 10**9 else 'inf'}": sum(low <= n < high for n in lengths) for low, high in LENGTH_BINS},
        "gzip_ratio": percentiles(values = [row["gzip_ratio"] for row in kept]),
        "rep4": percentiles(values = [row["rep4"] for row in kept]),
        "window_repeat_over_0.5": sum(row["window_repeat"] > 0.5 for row in kept),
        "code_like": sum(row["code_like"] for row in kept),
        "gen_nll_over_4": sum((row.get("gen_nll") or 0) > 4 for row in kept),
        "lang_samples": dict(Counter(row.get("lang", "?") for row in kept).most_common()),
    }
    tokens_by_lang = Counter()
    for row in kept:
        tokens_by_lang[row.get("lang", "?")] += row["n_kept"]
    summary["lang_tokens"] = dict(tokens_by_lang.most_common())
    if group == "free_chat":
        summary["user_lang_samples"] = dict(Counter(row.get("user_lang", "?") for row in kept).most_common())
        summary["user_tokens"] = percentiles(values = [next(p["keep"] for p in row["pieces"] if p.get("gen") == "user") for row in kept])
    if group.startswith("cont"):
        summary["doc_lang_samples"] = dict(sorted(Counter(row["doc_lang"] for row in kept).items()))
        tokens_by_doc = Counter()
        for row in kept:
            tokens_by_doc[row["doc_lang"]] += row["n_kept"]
        summary["doc_lang_tokens"] = dict(sorted(tokens_by_doc.items()))
        same = sum(row.get("lang") == row["doc_lang"] for row in kept)
        summary["same_lang_as_doc"] = round(same / max(len(kept), 1), 4)
        summary["source_samples"] = dict(Counter(row["source"] for row in kept).most_common())
        summary["prefix_tokens"] = percentiles(values = [row["prefix_tokens"] for row in kept])
    if runs:
        seconds = sum(run["seconds"] for run in runs)
        tokens = sum(run["gen_tokens"] for run in runs)
        summary["throughput"] = {"runs": len(runs), "gen_tokens": tokens, "seconds": round(seconds, 1), "tokens_per_second": round(tokens / max(seconds, 1e-9), 1)}
    summary["gen_nll"] = gen_nll(rows = kept, raw_dir = split_dir / "raw")
    summary["diversity"] = diversity(rows = kept, raw_dir = split_dir / "raw")
    nll_rows = read_jsonl(split_dir / "nll.jsonl")
    summary["nll"] = nll_summary(rows = nll_rows, group = group)
    if nll_rows:
        summary["nll"]["retokenized_ratio"] = retokenized_ratio(rows = kept, nll_rows = nll_rows, group = group)
    return summary


def markdown(summary):
    lines = ["# 回放文本汇总", ""]
    lines.append("| 组 | 划分 | 样本 | 保留 | 保留 token | 截断 | 截断损失 token | 丢弃 | 长度中位数 | 达到上限 | gzip 压缩率中位数 | 训练写法 NLL | 采样时 NLL |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for key, item in summary.items():
        group, split = key.split("/")
        nll = item.get("nll") or {}
        main = "assistant" if group.endswith("chat") else "generated"
        nll_text = f"{nll[main]['nll']:.4f}" if main in nll else ""
        gen_text = f"{item['gen_nll'].get('main', float('nan')):.4f}"
        dropped = ", ".join(f"{k} {v}" for k, v in item["dropped"].items()) or "0"
        lines.append(
            f"| {group} | {split} | {item['samples']} | {item['kept_samples']} | {item['kept_tokens']:,} | {item['truncated_samples']} | "
            f"{item['tokens_lost_truncation']:,} | {dropped} | {item['length'].get('p50', 0):.0f} | {item['hit_length_cap']} | "
            f"{item['gzip_ratio'].get('p50', 0):.3f} | {nll_text} | {gen_text} |"
        )
    lines.append("")
    for key, item in summary.items():
        lines.append(f"## {key}")
        lines.append("")
        lines.append(f"- 长度分档（保留的生成 token）：{item['length_bins']}")
        lines.append(f"- 语言（样本数）：{item['lang_samples']}")
        lines.append(f"- 像代码的样本：{item['code_like']}；平均 NLL 超过 4 的样本：{item['gen_nll_over_4']}；多样性：{item['diversity']}")
        if "doc_lang_samples" in item:
            lines.append(f"- 续写语言与文档语言相同的比例：{item['same_lang_as_doc']}")
            lines.append(f"- 来源：{item['source_samples']}")
        if "user_lang_samples" in item:
            lines.append(f"- 用户消息语言：{item['user_lang_samples']}")
        if "throughput" in item:
            lines.append(f"- 吞吐：{item['throughput']['tokens_per_second']} token/s（{item['throughput']['runs']} 次运行）")
        nll = item.get("nll")
        if nll:
            for part in ("user", "assistant", "generated"):
                if part in nll:
                    lines.append(f"- 训练写法 NLL（{part}）：{nll[part]['nll']}，{nll[part]['tokens']:,} token")
            if "by_doc_lang" in nll:
                lines.append("- 按文档语言的 NLL：" + "，".join(f"{lang} {value['nll']}" for lang, value in nll["by_doc_lang"].items()))
            if "retokenized_ratio" in nll:
                ratios = nll["retokenized_ratio"]
                high = {lang: ratio for lang, ratio in ratios.items() if lang != "all" and ratio > 1.05}
                lines.append(f"- 重新分词 / 采样 token 数：{ratios['all']}；超过 1.05 的语言：{high or '无'}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay-dir", required = True)
    parser.add_argument("--out-dir", required = True, help = "summary.json 与 summary.md 的输出目录")
    args = parser.parse_args()
    summary = {}
    for split in SPLITS:
        for group in GROUPS:
            item = split_summary(replay_dir = args.replay_dir, group = group, split = split)
            if item is not None:
                summary[f"{group}/{split}"] = item
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents = True, exist_ok = True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii = False, indent = 2) + "\n", encoding = "utf-8")
    (out_dir / "summary.md").write_text(markdown(summary = summary), encoding = "utf-8")
    print(markdown(summary = summary))


if __name__ == "__main__":
    main()
