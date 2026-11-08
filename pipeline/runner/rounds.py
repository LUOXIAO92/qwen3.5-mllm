"""为合成抽取文本条目（实验 v1，任务 T13）。

两个子命令：

- test：每种语言随机抽固定条数，作为留出音色测试集的条目。
- train：按每种语言的目标通过时长和预计通过率，抽一轮训练用的条目；短、中、长三档按 30%、45%、25% 抽。

两者都排除 tn_status 为 dropped 的条目、--exclude 文件中出现过的 entry_id，以及音色库参考片段用到的条目
（ref_entry_id），随机种子由轮次名和语言决定，同样的输入重跑得到同样的结果。

用法（在仓库根目录运行，python 指 <工作目录>/envs/runner/bin/python）::

    python -m pipeline.runner.rounds test \\
        --scripts-dir $DATASET/text/qwen3.5-asr/scripts-v1 \\
        --voices <工作目录>/data/T07/voices.jsonl \\
        --per-lang 60 \\
        --output $DATASET/text/qwen3.5-asr/rounds/test-all.jsonl

    python -m pipeline.runner.rounds train \\
        --name r1 \\
        --hours 5 \\
        --config pipeline/runner/config_v1.yaml \\
        --scripts-dir $DATASET/text/qwen3.5-asr/scripts-v1 \\
        --voices <工作目录>/data/T07/voices.jsonl \\
        --exclude $DATASET/text/qwen3.5-asr/rounds/test-all.jsonl \\
        --out-dir $DATASET/text/qwen3.5-asr/rounds

train 为每种语言写出 <out-dir>/<轮次名>-<语言>.jsonl，并打印每种语言的预计通过率、抽取条数和可用条数。
"""

import argparse
import json
import math
import random
from pathlib import Path

from pipeline.text.langs import LANGS

from .config import load_config

# 试验（#21）中各「语言 × TTS 模型」组合的核对通过率，用来估计每种语言要抽多少条。
# 没有试验数据的组合按 DEFAULT_PASS_RATE 估计。
PILOT_PASS_RATES = {
    "zh": {"qwen3_tts": 0.32, "omnivoice": 0.24},
    "en": {"supertonic3": 0.83, "omnivoice": 0.60, "qwen3_tts": 0.59},
    "ja": {"qwen3_tts": 0.33, "supertonic3": 0.29, "omnivoice": 0.16},
    "ko": {"supertonic3": 0.70, "qwen3_tts": 0.50, "omnivoice": 0.31},
    "de": {"qwen3_tts": 0.59, "supertonic3": 0.57, "omnivoice": 0.50},
    "fr": {"qwen3_tts": 0.43, "supertonic3": 0.40, "omnivoice": 0.38},
    "ru": {"supertonic3": 0.65, "voxcpm2": 0.31, "omnivoice": 0.21},
    "pt": {"supertonic3": 0.67, "qwen3_tts": 0.59, "voxcpm2": 0.32, "omnivoice": 0.33},
    "es": {"qwen3_tts": 0.84, "omnivoice": 0.48, "supertonic3": 0.41, "voxcpm2": 0.36},
    "it": {"qwen3_tts": 0.72, "supertonic3": 0.50, "omnivoice": 0.21},
    "id": {"supertonic3": 0.78, "voxcpm2": 0.41, "omnivoice": 0.38},
    "ms": {"voxcpm2": 0.18, "omnivoice": 0.14},
    "th": {"voxcpm2": 0.38, "omnivoice": 0.19},
    "vi": {"vieneu": 0.44, "omnivoice": 0.40, "voxcpm2": 0.39, "supertonic3": 0.30},
    "fil": {"voxcpm2": 0.35, "omnivoice": 0.24},
    "ar": {"indextts25": 0.73, "supertonic3": 0.45, "omnivoice": 0.33},
    "tr": {"supertonic3": 0.89, "voxcpm2": 0.50, "omnivoice": 0.33},
    "fa": {"omnivoice": 0.11},
}
DEFAULT_PASS_RATE = 0.6
# 试验中通过样本的平均时长（秒），用来把目标小时数换算成条数
MEAN_PASSED_SECONDS = 7.1
# 通过率极低的语言（如 fa）按估计会要很多条，每轮最多抽这么多
MAX_PER_LANG = 10000
LENGTH_SHARES = (("short", 0.30), ("medium", 0.45), ("long", 0.25))


def read_jsonl(path: Path) -> list:
    with open(file = path, mode = "r", encoding = "utf-8") as handle:
        return [json.loads(s = line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    with open(file = path, mode = "w", encoding = "utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(obj = row, ensure_ascii = False) + "\n")


def used_entry_ids(exclude: list, voices: Path) -> set:
    """已经用过、不能再抽的条目：--exclude 文件中的 entry_id 和音色参考片段的 ref_entry_id。"""
    used = set()
    for path in exclude:
        used |= {row["entry_id"] for row in read_jsonl(path = path)}
    used |= {row["ref_entry_id"] for row in read_jsonl(path = voices) if row.get("ref_entry_id")}
    return used


def candidates(scripts_dir: Path, lang: str, used: set) -> list:
    rows = read_jsonl(path = scripts_dir / f"{lang}.jsonl")
    return [row for row in rows if row["tn_status"] != "dropped" and row["entry_id"] not in used]


def expected_pass_rate(weights: dict, lang: str) -> float:
    """按配置里 TTS 模型的抽取权重，对试验通过率加权平均。"""
    rates = PILOT_PASS_RATES.get(lang, {})
    total = sum(weights.values())
    return sum(weight * rates.get(model, DEFAULT_PASS_RATE) for model, weight in weights.items()) / total


def run_test(args: argparse.Namespace) -> None:
    used = used_entry_ids(exclude = args.exclude, voices = args.voices)
    out = []
    for lang in args.langs:
        rows = candidates(scripts_dir = args.scripts_dir, lang = lang, used = used)
        random.Random(f"{args.name}-{lang}").shuffle(rows)
        out += rows[: args.per_lang]
    write_jsonl(path = args.output, rows = out)
    print(json.dumps(obj = {"entries": len(out), "output": str(args.output)}, ensure_ascii = False))


def run_train(args: argparse.Namespace) -> None:
    # 只读取各语言的 TTS 权重，用不到工作目录
    cfg = load_config(path = args.config, work_dir = Path("."))
    used = used_entry_ids(exclude = args.exclude, voices = args.voices)
    plan = {}
    for lang in args.langs:
        rate = expected_pass_rate(weights = cfg["langs"][lang]["tts"], lang = lang)
        need = min(MAX_PER_LANG, math.ceil(args.hours * 3600 / MEAN_PASSED_SECONDS / max(rate, 0.05)))
        rows = candidates(scripts_dir = args.scripts_dir, lang = lang, used = used)
        rng = random.Random(f"{args.name}-{lang}")
        take = []
        for length_bin, share in LENGTH_SHARES:
            pool = [row for row in rows if row["length_bin"] == length_bin]
            rng.shuffle(pool)
            take += pool[: math.ceil(need * share)]
        take = take[:need]
        write_jsonl(path = args.out_dir / f"{args.name}-{lang}.jsonl", rows = take)
        plan[lang] = {"expected_pass": round(rate, 3), "entries": len(take), "available": len(rows)}
        print(lang, json.dumps(obj = plan[lang]))
    print("total entries", sum(item["entries"] for item in plan.values()))


def main() -> None:
    parser = argparse.ArgumentParser(description = "为合成抽取文本条目")
    sub = parser.add_subparsers(dest = "command", required = True)
    for name in ("test", "train"):
        p = sub.add_parser(name)
        p.add_argument("--scripts-dir", required = True, type = Path, help = "含 <语言>.jsonl 朗读稿的目录")
        p.add_argument("--voices", required = True, type = Path, help = "音色库 voices.jsonl")
        p.add_argument("--exclude", nargs = "*", type = Path, default = [], help = "这些文件中的 entry_id 不再抽取")
        p.add_argument("--langs", nargs = "+", default = LANGS)
    test = sub.choices["test"]
    test.add_argument("--name", default = "test", help = "轮次名，决定随机种子")
    test.add_argument("--per-lang", type = int, default = 60)
    test.add_argument("--output", required = True, type = Path)
    train = sub.choices["train"]
    train.add_argument("--name", required = True, help = "轮次名，决定随机种子和输出文件名")
    train.add_argument("--hours", required = True, type = float, help = "每种语言的目标通过时长（小时）")
    train.add_argument("--config", required = True, type = Path, help = "合成配置，读取各语言 TTS 模型的权重")
    train.add_argument("--out-dir", required = True, type = Path)
    args = parser.parse_args()
    if args.command == "test":
        run_test(args = args)
    else:
        run_train(args = args)


if __name__ == "__main__":
    main()
