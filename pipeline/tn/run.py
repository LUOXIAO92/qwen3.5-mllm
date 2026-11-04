"""命令行入口：读取文本条目，补上 script、tn_status、tn_note 后写出。

    python -m pipeline.tn.run --input entries.jsonl --output scripts.jsonl

输入字段必须符合 pipeline/CONVENTIONS.md 的文本条目格式，不符合时列出问题并以退出码 2 结束，不写任何输出。
输出保留输入的全部字段和顺序；同时在输出旁写一份 <输出名>.summary.json，记录每种语言三种状态的条数和比例。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import LANGS, normalize_entry

REQUIRED = {
    "entry_id": str,
    "lang": str,
    "source": str,
    "source_license": str,
    "source_ref": str,
    "original": str,
    "n_chars": int,
    "length_bin": str,
    "est_duration_s": (int, float),
}
LENGTH_BINS = ("short", "medium", "long")
STATUSES = ("unchanged", "normalized", "dropped")
MAX_ERRORS = 20


class InputError(Exception):
    pass


def check_row(row, number: int) -> list:
    """检查一行条目，返回问题列表。"""
    if not isinstance(row, dict):
        return [f"第 {number} 行不是 JSON 对象"]
    errors = []
    for name, kind in REQUIRED.items():
        if name not in row:
            errors.append(f"第 {number} 行缺少字段 {name}")
        elif isinstance(row[name], bool) or not isinstance(row[name], kind):
            errors.append(f"第 {number} 行字段 {name} 的类型不对：{row[name]!r}")
    if errors:
        return errors
    if row["lang"] not in LANGS:
        errors.append(f"第 {number} 行的语言代码 {row['lang']!r} 不在 {' '.join(LANGS)} 中")
    if not row["entry_id"].startswith(row["lang"] + "-"):
        errors.append(f"第 {number} 行的 entry_id {row['entry_id']!r} 不是 <lang>-<source>-<序号> 格式")
    if row["length_bin"] not in LENGTH_BINS:
        errors.append(f"第 {number} 行的 length_bin {row['length_bin']!r} 不在 {' '.join(LENGTH_BINS)} 中")
    if not row["original"].strip():
        errors.append(f"第 {number} 行的 original 为空")
    return errors


def read_entries(path: Path) -> list:
    """读取并检查文本条目；有问题时抛出 InputError，列出前 MAX_ERRORS 个问题。"""
    if not path.exists():
        raise InputError(f"找不到输入文件 {path}")
    rows = []
    errors = []
    seen = set()
    with open(file = path, mode = "r", encoding = "utf-8") as handle:
        for number, line in enumerate(handle, start = 1):
            if not line.strip():
                continue
            try:
                row = json.loads(s = line)
            except json.JSONDecodeError as exc:
                errors.append(f"第 {number} 行不是合法的 JSON：{exc.msg}")
                continue
            problems = check_row(row = row, number = number)
            if not problems and row["entry_id"] in seen:
                problems.append(f"第 {number} 行的 entry_id {row['entry_id']!r} 重复")
            if problems:
                errors.extend(problems)
                continue
            seen.add(row["entry_id"])
            rows.append(row)
    if errors:
        shown = errors[:MAX_ERRORS]
        more = f"\n……共 {len(errors)} 个问题" if len(errors) > MAX_ERRORS else ""
        raise InputError(f"{path} 不符合文本条目格式（见 pipeline/CONVENTIONS.md）：\n" + "\n".join(shown) + more)
    if not rows:
        raise InputError(f"{path} 中没有条目")
    return rows


def summarize(rows: list) -> dict:
    """按语言统计三种状态的条数和比例。"""
    summary = {}
    for row in rows:
        stats = summary.setdefault(row["lang"], {"n": 0, **{s: 0 for s in STATUSES}})
        stats["n"] += 1
        stats[row["tn_status"]] += 1
    ordered = {}
    for lang in LANGS:
        if lang in summary:
            stats = summary[lang]
            stats["ratio"] = {s: round(stats[s] / stats["n"], 3) for s in STATUSES}
            ordered[lang] = stats
    return ordered


def main(argv = None) -> int:
    parser = argparse.ArgumentParser(description = "为文本条目生成朗读稿（script、tn_status、tn_note）")
    parser.add_argument("--input", required = True, type = Path, help = "文本条目 JSONL")
    parser.add_argument("--output", required = True, type = Path, help = "输出 JSONL")
    args = parser.parse_args(args = argv)

    try:
        rows = read_entries(path = args.input)
    except InputError as exc:
        print(exc, file = sys.stderr)
        return 2

    out_rows = []
    for row in rows:
        result = normalize_entry(original = row["original"], lang = row["lang"])
        out_rows.append({**row, "script": result.script, "tn_status": result.status, "tn_note": result.note})

    args.output.parent.mkdir(parents = True, exist_ok = True)
    tmp_path = args.output.with_name(name = args.output.name + ".tmp")
    with open(file = tmp_path, mode = "w", encoding = "utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(obj = row, ensure_ascii = False) + "\n")
    tmp_path.replace(target = args.output)

    summary = summarize(rows = out_rows)
    summary_path = args.output.with_suffix(suffix = ".summary.json")
    summary_path.write_text(
        data = json.dumps(obj = {"input": args.input.name, "langs": summary}, ensure_ascii = False, indent = 2),
        encoding = "utf-8",
    )
    print("lang  n  unchanged  normalized  dropped")
    for lang, stats in summary.items():
        print(f"{lang:4} {stats['n']:3} {stats['unchanged']:10} {stats['normalized']:11} {stats['dropped']:8}")
    print(f"写出 {len(out_rows)} 条：{args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
