"""从条目文件按语言抽样，生成小规模端到端测试用的条目（实验 v1，任务 T17）。

用法：

    python -m pipeline.text.sample --entries-dir <条目目录> --output <entries.jsonl> --per-lang 20 --seed 20261105

条目目录下每种语言一个 `<lang>.jsonl`（如 T03 的 part1）。每种语言抽 `--per-lang` 条，其中约
`--digit-share` 比例含数字（含阿拉伯-印度数字、波斯数字、泰文数字），用来检验朗读稿改写；含数字与不含数字
两组内部按长度档轮流抽取，保证短、中、长都有。条目字段原样保留，只读不改。同时在输出旁写一份
`<输出名>.summary.json`，记录每种语言的条数、含数字条数和长度档分布。
"""

import argparse
import json
import random
import re
from pathlib import Path

from .langs import LANGS

DIGITS = re.compile(pattern = r"[0-9٠-٩۰-۹๐-๙]")
BINS = ["short", "medium", "long"]


def has_digit(text):
    return DIGITS.search(string = text) is not None


def pick_balanced(rows, count, rng):
    """按长度档轮流抽取 count 条。"""
    pools = {name: [row for row in rows if row["length_bin"] == name] for name in BINS}
    for pool in pools.values():
        rng.shuffle(x = pool)
    picked = []
    while len(picked) < count and any(pools.values()):
        for name in BINS:
            if pools[name] and len(picked) < count:
                picked.append(pools[name].pop())
    return picked


def sample_lang(rows, per_lang, digit_share, rng):
    with_digit = [row for row in rows if has_digit(text = row["original"])]
    without_digit = [row for row in rows if not has_digit(text = row["original"])]
    n_digit = min(round(per_lang * digit_share), len(with_digit))
    picked = pick_balanced(rows = with_digit, count = n_digit, rng = rng)
    picked += pick_balanced(rows = without_digit, count = per_lang - len(picked), rng = rng)
    return sorted(picked, key = lambda row: row["entry_id"])


def read_jsonl(path):
    with open(file = path, mode = "r", encoding = "utf-8") as handle:
        return [json.loads(s = line) for line in handle if line.strip()]


def main(argv = None):
    parser = argparse.ArgumentParser(description = "按语言抽样，生成端到端测试用的条目")
    parser.add_argument("--entries-dir", required = True, type = Path, help = "每种语言一个 <lang>.jsonl 的目录")
    parser.add_argument("--output", required = True, type = Path, help = "输出 JSONL")
    parser.add_argument("--per-lang", default = 20, type = int, help = "每种语言的条数")
    parser.add_argument("--digit-share", default = 1 / 3, type = float, help = "含数字条目的目标比例")
    parser.add_argument("--seed", default = 20261105, type = int, help = "随机种子")
    parser.add_argument("--langs", nargs = "+", default = LANGS, help = "语言代码，默认全部 18 种")
    args = parser.parse_args(args = argv)

    picked_all = []
    summary = {}
    for lang in args.langs:
        rows = read_jsonl(path = args.entries_dir / f"{lang}.jsonl")
        rng = random.Random(f"{args.seed}-{lang}")
        picked = sample_lang(
            rows = rows,
            per_lang = args.per_lang,
            digit_share = args.digit_share,
            rng = rng,
        )
        picked_all.extend(picked)
        summary[lang] = {
            "n": len(picked),
            "with_digit": sum(has_digit(text = row["original"]) for row in picked),
            "bins": {name: sum(row["length_bin"] == name for row in picked) for name in BINS},
        }
        print(lang, summary[lang])

    args.output.parent.mkdir(parents = True, exist_ok = True)
    with open(file = args.output, mode = "w", encoding = "utf-8") as handle:
        for row in picked_all:
            handle.write(json.dumps(obj = row, ensure_ascii = False) + "\n")
    summary_path = args.output.with_suffix(suffix = ".summary.json")
    summary_path.write_text(
        data = json.dumps(obj = {"seed": args.seed, "per_lang": args.per_lang, "langs": summary}, ensure_ascii = False, indent = 2),
        encoding = "utf-8",
    )
    print(f"写出 {len(picked_all)} 条：{args.output}")


if __name__ == "__main__":
    main()
