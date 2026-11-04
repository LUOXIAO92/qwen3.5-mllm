"""朗读稿审阅报告：按语言统计三种状态，用 pipeline.compare 比对改写过的条目，并逐条列出原文与朗读稿。

    python -m pipeline.tn.review --scripts scripts.jsonl --output review.md

比对时把朗读稿当作 script、清理后的原文当作 hypothesis：数词统一后应一致。单位、货币和时刻的读法
（「twelve kilometers」对「12 km」、「十四点三十分」对「14:30」）是比对模块的已知局限，列为预期内的不一致。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..compare import compare
from .core import LANGS, clean
from .run import STATUSES, summarize

# tn_note 中出现这些类别时，比对不一致属于比对模块的已知局限
EXPECTED_KINDS = ("单位", "货币", "时刻")


def load(path: Path) -> list:
    with open(file = path, mode = "r", encoding = "utf-8") as handle:
        return [json.loads(s = line) for line in handle if line.strip()]


def check(row: dict) -> tuple:
    """返回 (是否一致, 差异说明, 是否属于预期内的不一致)。"""
    result = compare(script = row["script"], hypothesis = clean(text = row["original"], lang = row["lang"]), lang = row["lang"])
    expected = any(kind in row["tn_note"] for kind in EXPECTED_KINDS)
    return result.match, result.detail, expected


def cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render(rows: list) -> str:
    summary = summarize(rows = rows)
    lines = ["## 各语言的状态比例", "", "| 语言 | 条数 | unchanged | normalized | dropped |", "| --- | --- | --- | --- | --- |"]
    totals = {s: 0 for s in STATUSES}
    for lang, stats in summary.items():
        parts = [f"{stats[s]}（{stats['ratio'][s]:.0%}）" for s in STATUSES]
        lines.append(f"| {lang} | {stats['n']} | " + " | ".join(parts) + " |")
        for s in STATUSES:
            totals[s] += stats[s]
    n = sum(totals.values())
    lines.append(f"| 合计 | {n} | " + " | ".join(f"{totals[s]}（{totals[s] / n:.0%}）" for s in STATUSES) + " |")

    normalized = [row for row in rows if row["tn_status"] == "normalized"]
    results = [(row, *check(row = row)) for row in normalized]
    unexpected = [(row, detail) for row, match, detail, expected in results if not match and not expected]
    expected = [(row, detail) for row, match, detail, exp in results if not match and exp]
    lines += ["", "## normalized 条目与原文的比对", ""]
    lines.append(
        f"共 {len(normalized)} 条，数词统一后一致 {sum(1 for r in results if r[1])} 条；"
        f"不一致 {len(unexpected) + len(expected)} 条，其中单位、货币、时刻读法造成的（比对模块的已知局限）"
        f"{len(expected)} 条，其他 {len(unexpected)} 条。"
    )
    if unexpected or expected:
        lines += ["", "| entry_id | 类别 | 差异 |", "| --- | --- | --- |"]
        for row, detail in unexpected:
            lines.append(f"| {row['entry_id']} | 其他 | {cell(text = detail)} |")
        for row, detail in expected:
            lines.append(f"| {row['entry_id']} | 单位/货币/时刻 | {cell(text = detail)} |")

    lines += ["", "## 改写过的条目", "", "| entry_id | 原文 | 朗读稿 |", "| --- | --- | --- |"]
    for row in normalized:
        lines.append(f"| {row['entry_id']} | {cell(text = row['original'])} | {cell(text = row['script'])} |")

    lines += ["", "## dropped 条目的原因", "", "| entry_id | 原因 |", "| --- | --- |"]
    for row in rows:
        if row["tn_status"] == "dropped":
            lines.append(f"| {row['entry_id']} | {cell(text = row['tn_note'])} |")
    return "\n".join(lines) + "\n"


def main(argv = None) -> int:
    parser = argparse.ArgumentParser(description = "生成朗读稿审阅报告（Markdown）")
    parser.add_argument("--scripts", required = True, type = Path, help = "pipeline.tn.run 写出的 JSONL")
    parser.add_argument("--output", required = True, type = Path, help = "报告 Markdown")
    args = parser.parse_args(args = argv)
    rows = load(path = args.scripts)
    rows.sort(key = lambda row: (LANGS.index(row["lang"]), row["entry_id"]))
    args.output.parent.mkdir(parents = True, exist_ok = True)
    args.output.write_text(data = render(rows = rows), encoding = "utf-8")
    print(f"写出 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
