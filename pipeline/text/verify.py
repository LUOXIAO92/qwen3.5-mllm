"""核对条目文件：字段齐全、entry_id 唯一、规范化键无重复、与 FLEURS 排除清单重合为 0、过滤规则全部通过。

    python -m pipeline.text.verify --entries-dir <data/T03/part1> --exclusion <exclusion.jsonl>

任何一项不满足时以非零状态退出。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .exclusion import is_excluded, load_exclusion
from .filters import check_text
from .sources.web import entry_options
from .langs import LANGS
from .lengths import length_bin
from .normalize import norm_key

REQUIRED = [
    "entry_id", "lang", "source", "source_license", "source_ref", "original",
    "n_chars", "length_bin", "est_duration_s",
]


def verify_file(path: Path, *, lang: str, exclusion: dict[str, set[str]]) -> dict:
    problems: dict[str, int] = {}

    def flag(name: str) -> None:
        problems[name] = problems.get(name, 0) + 1

    seen_ids: set[str] = set()
    seen_keys: set[str] = set()
    total = 0
    with open(path, encoding = "utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            total += 1
            if any(field not in row for field in REQUIRED):
                flag("missing_field")
            if row["entry_id"] in seen_ids:
                flag("duplicate_id")
            seen_ids.add(row["entry_id"])
            key = norm_key(row["original"], lang = lang)
            if key in seen_keys:
                flag("duplicate_text")
            seen_keys.add(key)
            if is_excluded(row["original"], lang = lang, exclusion = exclusion):
                flag("fleurs_overlap")
            options = entry_options(lang, row["source"])
            if check_text(row["original"], lang = lang, options = options) is not None:
                flag("filter_fail")
            if row["n_chars"] != len(row["original"]) or row["lang"] != lang:
                flag("field_mismatch")
            if length_bin(row["est_duration_s"]) != row["length_bin"] and abs(row["est_duration_s"] - 5.0) > 0.1 and abs(row["est_duration_s"] - 15.0) > 0.1:
                flag("bin_mismatch")
    return {"total": total, "problems": problems}


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entries-dir", required = True)
    parser.add_argument("--exclusion", required = True)
    args = parser.parse_args()

    exclusion = load_exclusion(args.exclusion)
    failed = False
    for lang in LANGS:
        path = Path(args.entries_dir) / f"{lang}.jsonl"
        if not path.exists():
            print(f"[{lang}] 缺少文件")
            failed = True
            continue
        result = verify_file(path, lang = lang, exclusion = exclusion)
        print(f"[{lang}] {result['total']} 条，问题：{result['problems'] or '无'}")
        failed = failed or bool(result["problems"])
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
