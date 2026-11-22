"""汇总回放文本：数量、退化与截断、长度分布、语言分布、吞吐和参考模型 NLL，写出 summary.json 与 summary.md。

    python -m pipeline.replay.report --replay-dir <回放目录> --out-dir <输出目录>

读取 finalize、langid、nll、teacher 的结果和 shard 二进制中的对数概率，只用 CPU。三种 NLL：

- 训练写法 NLL（`nll`，筛选的回放目录）：nll 一步按训练的渲染与分词重新前向得到，是以后比较「当前模型 − 参考模型」的基线；
- 参考 NLL（`teacher_nll`，不筛选的回放目录）：teacher 一步用保存的 token id 整条前向（与按 token id 训练时的输入相同），
  生成 token 的对数概率取负后按 token 平均，同样作为「当前模型 − 参考模型」的基线；同时给出 top-32 之外的尾部概率；
- 采样时 NLL（`gen_nll`）：生成时 vLLM 记录的采样 token 自身的对数概率取负，按保留的生成 token 平均。

不筛选的回放目录另外统计循环：`loop_samples` 是出现循环（按 finalize 的规则）的样本数，`tokens_after_loop_onset` 是
循环起点之后的生成 token 数（筛选模式会截掉的部分）。
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from pipeline.replay.prompts import ALL_GROUPS, CHAT_GROUPS, SELF_USER_GROUPS, SPLITS
from pipeline.replay.store import ShardReader
from pipeline.replay.teacher import Teacher

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
    main = "assistant" if group in ("free_chat", "cont_chat", "prompt_chat") else "generated"
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


def teacher_summary(rows, split_dir, group):
    """参考 NLL 与 top-32 之外的尾部概率，取自 teacher 的输出；按部分（角色）给出，主体部分再按语言分开。"""
    teacher_dir = Path(split_dir) / "teacher"
    done = set()
    progress = teacher_dir / "progress.jsonl"
    if not progress.exists():
        return None
    with open(progress, encoding = "utf-8") as handle:
        for line in handle:
            done.update(json.loads(line)["sample_ids"])
    rows = [row for row in rows if row["sample_id"] in done]
    if not rows:
        return None
    shards = {}
    sums = defaultdict(float)
    counts = Counter()
    by_lang = defaultdict(lambda: [0.0, 0])
    tails = []
    key = "doc_lang" if group.startswith("cont") else ("prompt_lang" if group == "prompt_chat" else "lang")
    for row in rows:
        if row["shard"] not in shards:
            shards[row["shard"]] = Teacher.stored(teacher_dir = teacher_dir, number = row["shard"])
        arrays = shards[row["shard"]]
        gen_pieces = [piece for piece in row["pieces"] if "gen" in piece]
        for piece in gen_pieces:
            span = slice(piece["offset"], piece["offset"] + piece["keep"])
            nll = float(-np.asarray(arrays["lp"][span], dtype = np.float64).sum())
            sums[piece["gen"]] += nll
            counts[piece["gen"]] += piece["keep"]
            if piece["gen"] != "user":
                by_lang[row.get(key) or "?"][0] += nll
                by_lang[row.get(key) or "?"][1] += piece["keep"]
            top = np.exp(np.asarray(arrays["top_lp"][span], dtype = np.float32)).sum(axis = 1)
            tails.append(np.clip(1.0 - top, 0.0, 1.0))
    tail = np.concatenate(tails) if tails else np.zeros(0)
    result = {"samples": len(rows)}
    for role in counts:
        result[role] = {"tokens": counts[role], "nll": round(sums[role] / max(counts[role], 1), 4)}
    result[f"by_{key}"] = {lang: {"tokens": value[1], "nll": round(value[0] / max(value[1], 1), 4)} for lang, value in sorted(by_lang.items())}
    result["top32_tail_mass"] = percentiles(values = tail.tolist()) if len(tail) < 2_000_000 else percentiles(values = tail[:: len(tail) // 1_000_000].tolist())
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
    noloop = any(row.get("selection") == "noloop" for row in rows)
    if noloop:
        # 丢弃循环：有循环的样本整条不用，保留的样本不截断
        dropped = [row for row in rows if row["drop_reason"] == "loop"]
        summary["noloop"] = True
        summary["status"] = dict(Counter(row.get("status") for row in kept))
        summary["dropped_loop_samples"] = len(dropped)
        summary["dropped_loop_share"] = round(len(dropped) / max(len(rows), 1), 4)
        summary["dropped_loop_tokens"] = sum(row["n_gen"] for row in dropped)
        summary["generated_tokens"] = summary["kept_tokens"] + summary["dropped_loop_tokens"]
        summary["dropped_token_share"] = round(summary["dropped_loop_tokens"] / max(summary["generated_tokens"], 1), 4)
        summary["dropped_loop_kind"] = dict(Counter(row["loop_onset"]["kind"] for row in dropped if row.get("loop_onset")))
        summary["dropped_loop_role"] = dict(Counter(row["loop_onset"]["role"] for row in dropped if row.get("loop_onset")))
        summary["dropped_loop_onset"] = percentiles(values = [row["loop_onset"]["position"] for row in dropped if row.get("loop_onset")])
        summary["imported_samples"] = sum(bool(row.get("origin")) for row in rows)
        summary["kept_hit_cap"] = sum(row["finish_reason"] == "length" for row in kept)
        unclosed = [row for row in rows if row["drop_reason"] == "think_unclosed"]
        if unclosed:
            summary["dropped_think_unclosed_samples"] = len(unclosed)
            summary["dropped_think_unclosed_tokens"] = sum(row["n_gen"] for row in unclosed)
            summary["generated_tokens"] += summary["dropped_think_unclosed_tokens"]
    if not noloop and any("pre_loop_tokens" in row for row in kept):
        # 不筛选：循环只是元数据
        summary["unfiltered"] = True
        summary["status"] = dict(Counter(row.get("status") for row in kept))
        summary["loop_samples"] = sum(row["loop_onset"] is not None for row in kept)
        summary["loop_share"] = round(summary["loop_samples"] / max(len(kept), 1), 4)
        summary["loop_kind"] = dict(Counter(row["loop_onset"]["kind"] for row in kept if row["loop_onset"]))
        summary["tokens_after_loop_onset"] = sum(row["n_gen"] - row["pre_loop_tokens"] for row in kept)
        summary["tokens_after_loop_onset_share"] = round(summary["tokens_after_loop_onset"] / max(summary["kept_tokens"], 1), 4)
        summary["cap_samples_with_loop"] = sum(row["finish_reason"] == "length" and row["loop_onset"] is not None for row in kept)
    tokens_by_lang = Counter()
    for row in kept:
        tokens_by_lang[row.get("lang", "?")] += row["n_kept"]
    summary["lang_tokens"] = dict(tokens_by_lang.most_common())
    if group == "free_chat_think":
        # 写出率按没有循环的全部样本计（没有写出 </think> 的样本在丢弃循环的目录中整条不用）
        answered = [row for row in rows if "think_closed" in row and row["drop_reason"] in (None, "think_unclosed")]
        closed = [row for row in answered if row["think_closed"]]
        summary["think"] = {
            "samples_with_assistant": len(answered),
            "closed": len(closed),
            "close_rate": round(len(closed) / max(len(answered), 1), 4),
            "not_closed": len(answered) - len(closed),
            "reasoning_tokens": percentiles(values = [row["reasoning_tokens"] for row in closed]),
            "answer_tokens": percentiles(values = [row["answer_tokens"] for row in closed]),
            "reasoning_tokens_not_closed": percentiles(values = [row["reasoning_tokens"] for row in answered if not row["think_closed"] and "reasoning_tokens" in row]),
            "no_assistant": sum("reasoning_tokens" not in row for row in answered),
            "tokens_closed": sum(row["n_gen"] for row in closed),
            "tokens_not_closed": sum(row["n_gen"] for row in answered if not row["think_closed"]),
            "samples_not_closed": sum(not row["think_closed"] for row in answered),
            "not_closed_hit_cap": sum(not row["think_closed"] and row["finish_reason"] == "length" for row in answered),
        }
    if group in SELF_USER_GROUPS:
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
    if group == "prompt_chat":
        summary["prompt_lang_samples"] = dict(sorted(Counter(row["prompt_lang"] for row in kept).items()))
        tokens_by_prompt = Counter()
        for row in kept:
            tokens_by_prompt[row["prompt_lang"]] += row["n_kept"]
        summary["prompt_lang_tokens"] = dict(sorted(tokens_by_prompt.items()))
        summary["same_lang_as_prompt"] = round(sum(row.get("lang") == row["prompt_lang"] for row in kept) / max(len(kept), 1), 4)
        summary["source_samples"] = dict(Counter(row["source"] for row in kept).most_common())
        summary["user_tokens"] = percentiles(values = [row["user_tokens"] for row in kept if "user_tokens" in row])
        parts = defaultdict(lambda: [0, 0])
        for row in kept:
            parts[row.get("part") or "instruct"][0] += 1
            parts[row.get("part") or "instruct"][1] += row["n_kept"]
        summary["part_samples_tokens"] = {part: {"samples": value[0], "tokens": value[1]} for part, value in sorted(parts.items())}
        summary["kind_samples"] = dict(Counter(row["kind"] for row in kept if row.get("kind")))
    if runs:
        seconds = sum(run["seconds"] for run in runs)
        # 中途暂停的样本在下次运行才写完，按每次实际解码的 token 计吞吐（旧记录没有这一项时用写完的样本的 token）
        tokens = sum(run.get("decoded_tokens", run["gen_tokens"]) for run in runs)
        summary["throughput"] = {"runs": len(runs), "gen_tokens": tokens, "seconds": round(seconds, 1), "tokens_per_second": round(tokens / max(seconds, 1e-9), 1)}
    summary["gen_nll"] = gen_nll(rows = kept, raw_dir = split_dir / "raw")
    summary["diversity"] = diversity(rows = kept, raw_dir = split_dir / "raw")
    teacher = teacher_summary(rows = kept, split_dir = split_dir, group = group)
    if teacher:
        summary["teacher_nll"] = teacher
    nll_rows = read_jsonl(split_dir / "nll.jsonl")
    summary["nll"] = nll_summary(rows = nll_rows, group = group)
    if nll_rows:
        summary["nll"]["retokenized_ratio"] = retokenized_ratio(rows = kept, nll_rows = nll_rows, group = group)
    return summary


def unfiltered_table(summary):
    """不筛选目录的总表；采样时 NLL 是生成时记录的采样 token 自身的对数概率取负后的平均（这个目录不做 top-32 一步）。"""
    lines = [
        "| 组 | 划分 | 样本 | 生成 token | 长度中位数 | 长度 p90 | 达到 16k 上限 | 有循环的样本 | 循环起点之后的 token | gzip 压缩率中位数 | 采样时 NLL |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, item in summary.items():
        group, split = key.split("/")
        lines.append(
            f"| {group} | {split} | {item['samples']:,} | {item['kept_tokens']:,} | {item['length'].get('p50', 0):.0f} | "
            f"{item['length'].get('p90', 0):.0f} | {item['hit_length_cap']:,}（{item['hit_length_cap'] / max(item['samples'], 1):.1%}） | "
            f"{item['loop_samples']:,}（{item['loop_share']:.1%}） | {item['tokens_after_loop_onset_share']:.1%} | "
            f"{item['gzip_ratio'].get('p50', 0):.3f} | {item['gen_nll'].get('main', float('nan')):.4f} |"
        )
    return lines


def noloop_table(summary):
    lines = [
        "| 组 | 划分 | 用掉的提示 | 保留样本 | 保留 token | 有循环而丢弃的样本 | 丢弃样本花掉的 token | 保留样本长度中位数 | 长度 p90 | 保留样本达到 16k 上限 | gzip 压缩率中位数 | 参考 NLL | 采样时 NLL | top-32 之外概率均值 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, item in summary.items():
        group, split = key.split("/")
        main = "assistant" if group in CHAT_GROUPS else "text"
        teacher = item.get("teacher_nll") or {}
        nll_text = f"{teacher[main]['nll']:.4f}" if main in teacher else ""
        tail_text = f"{teacher['top32_tail_mass']['mean']:.4f}" if teacher.get("top32_tail_mass") else ""
        lines.append(
            f"| {group} | {split} | {item['samples']:,} | {item['kept_samples']:,} | {item['kept_tokens']:,} | "
            f"{item['dropped_loop_samples']:,}（{item['dropped_loop_share']:.1%}） | {item['dropped_loop_tokens']:,}（{item['dropped_token_share']:.1%}） | "
            f"{item['length'].get('p50', 0):.0f} | {item['length'].get('p90', 0):.0f} | {item['kept_hit_cap']:,} | "
            f"{item['gzip_ratio'].get('p50', 0):.3f} | {nll_text} | {item['gen_nll'].get('main', float('nan')):.4f} | {tail_text} |"
        )
    return lines


def markdown(summary):
    lines = ["# 回放文本汇总", ""]
    if any(item.get("noloop") for item in summary.values()):
        lines = ["# 丢弃循环的回放文本汇总", ""]
        lines.extend(noloop_table(summary = summary))
        lines.append("")
        lines.append("保留的样本都没有循环，平均比全部生成的样本短（长样本更容易出现循环），长度分布偏向短文本。")
        lines.append("")
        for key, item in summary.items():
            lines.append(f"## {key}")
            lines.append("")
            lines.append(f"- 保留样本长度分档（生成 token）：{item['length_bins']}；长度分位数：{item['length']}")
            lines.append(f"- 保留样本的结束方式：{item['finish']}；状态：{item['status']}；由不筛选目录导入的样本（含丢弃的）：{item['imported_samples']}")
            lines.append(f"- 丢弃样本的循环种类：{item['dropped_loop_kind']}；出现循环的片段：{item['dropped_loop_role']}；循环起点位置：{item['dropped_loop_onset']}")
            lines.append(f"- 语言（保留样本数）：{item['lang_samples']}")
            lines.append(f"- 语言（保留 token 数）：{item['lang_tokens']}")
            lines.append(f"- 像代码的样本：{item['code_like']}；平均 NLL 超过 4 的样本：{item['gen_nll_over_4']}；多样性：{item['diversity']}")
            if "doc_lang_samples" in item:
                lines.append(f"- 续写语言与文档语言相同的比例：{item['same_lang_as_doc']}；文档语言（保留样本数）：{item['doc_lang_samples']}")
                lines.append(f"- 来源：{item['source_samples']}；文档开头 token 数：{item['prefix_tokens']}")
            if "user_lang_samples" in item:
                lines.append(f"- 用户消息语言：{item['user_lang_samples']}；用户消息 token 数：{item['user_tokens']}")
            if "think" in item:
                think = item["think"]
                lines.append(
                    f"- 思考模式：没有循环的样本 {think['samples_with_assistant']}（其中没有助手回答的 {think.get('no_assistant', 0)}），"
                    f"写出 </think> 的 {think['closed']}（{think['close_rate']:.1%}），"
                    f"没有写出的 {think['not_closed']}；思考 token 数 {think['reasoning_tokens']}；回答 token 数 {think['answer_tokens']}；"
                    f"没有写出 </think> 的样本的思考长度 {think['reasoning_tokens_not_closed']}"
                )
                lines.append(
                    f"- 生成 token：写出 </think> 的样本 {think['tokens_closed']:,}，没有写出的样本 {think['tokens_not_closed']:,}"
                    f"（{think['samples_not_closed']} 条，其中 {think['not_closed_hit_cap']} 条达到 16k 上限；丢弃循环的目录中这些样本整条不用）"
                )
            if "prompt_lang_samples" in item:
                lines.append(f"- 提问语言（保留样本数）：{item['prompt_lang_samples']}；回答与提问语言相同的比例：{item['same_lang_as_prompt']}")
                lines.append(f"- 提问语言（保留 token 数）：{item['prompt_lang_tokens']}")
                lines.append(f"- 来源：{item['source_samples']}；提问 token 数：{item['user_tokens']}")
                lines.append(f"- 各部分（instruct 为数据集提问，agent 为工具调用与编程上下文）：{item['part_samples_tokens']}；agent 种类：{item['kind_samples']}")
            if "throughput" in item:
                lines.append(f"- 本目录中生成的吞吐：{item['throughput']['tokens_per_second']} token/s（{item['throughput']['runs']} 次运行，{item['throughput']['seconds']:.0f} 秒）")
            teacher = item.get("teacher_nll")
            if teacher:
                for part in ("user", "assistant", "text"):
                    if part in teacher:
                        lines.append(f"- 参考 NLL（{part}）：{teacher[part]['nll']}，{teacher[part]['tokens']:,} token")
                by_key = "by_doc_lang" if "by_doc_lang" in teacher else "by_lang"
                lines.append(f"- 参考 NLL（主体，{by_key}）：" + "，".join(f"{lang} {value['nll']}" for lang, value in teacher[by_key].items()))
                lines.append(f"- top-32 之外的概率：{teacher['top32_tail_mass']}")
            lines.append("")
        return "\n".join(lines) + "\n"
    if any(item.get("unfiltered") for item in summary.values()):
        lines = ["# 不筛选的回放文本汇总", ""]
        lines.extend(unfiltered_table(summary = summary))
        lines.append("")
        lines.append(
            "「循环起点之后的 token」是有循环的样本在循环起点之后的生成 token 占全部生成 token 的比例。每个划分的样本是提示清单前若干条的"
            "全部样本；改为丢弃循环（replay-noloop）后不再往这里生成，没有达到 token 目标的划分保留到停下时已经写完的样本。"
        )
        lines.append("")
        for key, item in summary.items():
            lines.append(f"## {key}")
            lines.append("")
            lines.append(f"- 长度分档（生成 token）：{item['length_bins']}；长度分位数：{item['length']}")
            lines.append(f"- 结束方式：{item['finish']}；状态：{item['status']}；达到上限且有循环：{item['cap_samples_with_loop']}；循环种类：{item['loop_kind']}")
            lines.append(f"- 语言（样本数）：{item['lang_samples']}")
            lines.append(f"- 语言（token 数）：{item['lang_tokens']}")
            lines.append(f"- 像代码的样本：{item['code_like']}；平均 NLL 超过 4 的样本：{item['gen_nll_over_4']}；多样性：{item['diversity']}")
            if "doc_lang_samples" in item:
                lines.append(f"- 续写语言与文档语言相同的比例：{item['same_lang_as_doc']}；文档语言（样本数）：{item['doc_lang_samples']}")
                lines.append(f"- 来源：{item['source_samples']}；文档开头 token 数：{item['prefix_tokens']}")
            if "user_lang_samples" in item:
                lines.append(f"- 用户消息语言：{item['user_lang_samples']}；用户消息 token 数：{item['user_tokens']}")
            if "throughput" in item:
                lines.append(f"- 吞吐：{item['throughput']['tokens_per_second']} token/s（{item['throughput']['runs']} 次运行，{item['throughput']['seconds']:.0f} 秒）")
            teacher = item.get("teacher_nll")
            if teacher:
                for part in ("user", "assistant", "text"):
                    if part in teacher:
                        lines.append(f"- 参考 NLL（{part}）：{teacher[part]['nll']}，{teacher[part]['tokens']:,} token")
                by_key = "by_doc_lang" if "by_doc_lang" in teacher else "by_lang"
                lines.append(f"- 参考 NLL（主体，{by_key}）：" + "，".join(f"{lang} {value['nll']}" for lang, value in teacher[by_key].items()))
                lines.append(f"- top-32 之外的概率：{teacher['top32_tail_mass']}")
            lines.append("")
        return "\n".join(lines) + "\n"
    lines.append("| 组 | 划分 | 样本 | 保留 | 保留 token | 截断 | 截断损失 token | 丢弃 | 长度中位数 | 达到上限 | gzip 压缩率中位数 | 训练写法 NLL | 采样时 NLL |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for key, item in summary.items():
        group, split = key.split("/")
        nll = item.get("nll") or {}
        main = "assistant" if group in CHAT_GROUPS else "generated"
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
        for group in ALL_GROUPS:
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
