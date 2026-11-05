"""把网页与新闻来源的候选并入 part1，写出最终条目文件（T03b）。

    python -m pipeline.text.merge --part1-dir <part1> --cand-dir <候选目录> --exclusion <exclusion.jsonl> \
        --out-dir <final>

规则：

- part1 的条目原样保留（`entry_id` 不变，下游已按它引用）。
- 新候选先过 `check_text`（`web = True`）和 FLEURS 排除，再与 part1 以及彼此按规范化键去重，part1 优先。
- 各长度档的目标条数减去 part1 已有条数，就是该档要补的条数；各路来源（每个候选目录一路，HPLT 按体裁分路）
  轮流取样，体裁大致均衡。来源不足时不跨档硬凑，缺口写进 `summary.json` 的 `gaps`。
- 新条目的 `entry_id` 为 `<lang>-<来源短名>-<序号>`，短名与 part1 的不同，不会冲突。
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .build import SOURCE_INFO, interleave, read_candidates, to_entry, write_jsonl
from .dedup import Deduper
from .exclusion import is_excluded, load_blocklist, load_exclusion
from .filters import check_text
from .langs import LANGS
from .lengths import BINS
from .mtscore import CharLM, percentile
from .sources.web import entry_options

WEB_SOURCES = ["hplt_na", "hplt_op", "hplt_in", "fw2hq", "fwedu", "wikinews"]
DEFAULT_TARGETS = {"short": 12000, "medium": 18000, "long": 10000}


def merge_language(
    lang: str,
    *,
    part1_dir: Path,
    cand_dir: Path,
    sources: list[str],
    exclusion: dict[str, set[str]],
    targets: dict[str, int],
    seed: int,
    lm_filter: float = 0.0,
) -> tuple[list[dict], list[dict], dict]:
    """返回 (最终条目, 新增条目, 统计)。"""
    rng = random.Random(f"merge-{seed}-{lang}")
    with open(part1_dir / f"{lang}.jsonl", encoding = "utf-8") as handle:
        part1 = [json.loads(line) for line in handle]
    deduper = Deduper(lang = lang)
    for entry in part1:
        deduper.mark_seen(entry["original"])
    have = Counter(entry["length_bin"] for entry in part1)

    blocklist = load_blocklist()
    stats: Counter = Counter()
    by_source: dict[str, list[dict]] = {}
    for source in sources:
        path = cand_dir / source / f"{lang}.jsonl"
        if not path.exists():
            continue
        rows = []
        for row in read_candidates(path):
            stats[f"read_{source}"] += 1
            if row["ref"] in blocklist:
                continue
            if check_text(row["text"], lang = lang, options = entry_options(lang, SOURCE_INFO[source][0])) is not None:
                stats["drop_rules"] += 1
                continue
            if is_excluded(row["text"], lang = lang, exclusion = exclusion):
                stats["drop_fleurs"] += 1
                continue
            if not deduper.add(row["text"]):
                stats["drop_duplicate"] += 1
                continue
            row["_source"] = source
            rows.append(row)
        rng.shuffle(rows)
        by_source[source] = rows

    if lm_filter > 0:
        # 字符模型交叉熵：用 part1 的人写文本训练，丢弃候选里交叉熵最高的 `lm_filter` 比例（噪声、评论串、外语夹杂）
        trusted = [entry["original"] for entry in part1][:20000]
        model = CharLM()
        for text in trusted:
            model.train(text)
        scored = {id(row): model.cross_entropy(row["text"]) for rows in by_source.values() for row in rows}
        cutoff = percentile(list(scored.values()), 1.0 - lm_filter) if scored else 0.0
        for source, rows in by_source.items():
            kept = [row for row in rows if scored[id(row)] <= cutoff]
            stats["drop_lm"] += len(rows) - len(kept)
            by_source[source] = kept

    chosen: list[dict] = []
    gaps: dict[str, int] = {}
    for bin_name in BINS:
        need = max(0, targets[bin_name] - have[bin_name])
        groups = [[row for row in by_source[source] if row["length_bin"] == bin_name] for source in sorted(by_source)]
        taken = interleave(groups, limit = need)
        chosen.extend(taken)
        if len(taken) < need:
            gaps[bin_name] = need - len(taken)
    rng.shuffle(chosen)

    counters: Counter = Counter()
    new_entries = []
    for row in chosen:
        slug = SOURCE_INFO[row["_source"]][1]
        counters[slug] += 1
        new_entries.append(to_entry(row, source_dir = row["_source"], index = counters[slug]))
    return part1 + new_entries, new_entries, {"gaps": gaps, "filter_stats": dict(stats)}


def summarize(entries: list[dict]) -> dict:
    table: dict = defaultdict(lambda: defaultdict(int))
    licenses: dict = defaultdict(set)
    hours = 0.0
    for entry in entries:
        table[entry["source"]][entry["length_bin"]] += 1
        licenses[entry["source"]].add(entry["source_license"])
        hours += entry["est_duration_s"] / 3600
    bins = Counter(entry["length_bin"] for entry in entries)
    return {
        "total": len(entries),
        "est_hours": round(hours, 1),
        "by_bin": {name: bins[name] for name in BINS},
        "by_source_bin": {source: dict(counts) for source, counts in table.items()},
        "licenses": {source: sorted(values) for source, values in licenses.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part1-dir", required = True)
    parser.add_argument("--cand-dir", required = True)
    parser.add_argument("--exclusion", required = True)
    parser.add_argument("--out-dir", required = True)
    parser.add_argument("--lang", nargs = "+", default = ["all"])
    parser.add_argument("--sources", nargs = "+", default = WEB_SOURCES)
    parser.add_argument("--target-short", type = int, default = DEFAULT_TARGETS["short"])
    parser.add_argument("--target-medium", type = int, default = DEFAULT_TARGETS["medium"])
    parser.add_argument("--target-long", type = int, default = DEFAULT_TARGETS["long"])
    parser.add_argument("--lm-langs", nargs = "*", default = ["fil", "vi", "th"], help = "启用字符模型过滤的语言")
    parser.add_argument("--lm-drop", type = float, default = 0.10, help = "字符模型过滤丢弃的比例")
    parser.add_argument("--review-size", type = int, default = 30)
    parser.add_argument("--seed", type = int, default = 0)
    args = parser.parse_args()

    langs = LANGS if args.lang == ["all"] else args.lang
    exclusion = load_exclusion(args.exclusion)
    out_dir = Path(args.out_dir)
    (out_dir / "review").mkdir(parents = True, exist_ok = True)
    targets = {"short": args.target_short, "medium": args.target_medium, "long": args.target_long}

    summary: dict = {}
    for lang in langs:
        entries, new_entries, info = merge_language(
            lang,
            part1_dir = Path(args.part1_dir),
            cand_dir = Path(args.cand_dir),
            sources = args.sources,
            exclusion = exclusion,
            targets = targets,
            seed = args.seed,
            lm_filter = args.lm_drop if lang in args.lm_langs else 0.0,
        )
        write_jsonl(out_dir / f"{lang}.jsonl", entries)
        # 审阅样本从新增的网页条目里随机取：part1 的条目已经审阅过。
        review = random.Random(f"review-{args.seed}-{lang}").sample(new_entries, k = min(args.review_size, len(new_entries)))
        write_jsonl(out_dir / "review" / f"{lang}.jsonl", review)
        summary[lang] = {**summarize(entries), "new": len(new_entries), **info}
        print(
            f"[{lang}] 共 {summary[lang]['total']} 条（新增 {len(new_entries)}），{summary[lang]['by_bin']}，"
            f"约 {summary[lang]['est_hours']} 小时，缺口 {info['gaps'] or '无'}",
            flush = True,
        )
    with open(out_dir / "summary.json", "w", encoding = "utf-8") as handle:
        json.dump(summary, handle, ensure_ascii = False, indent = 2)


if __name__ == "__main__":
    main()
