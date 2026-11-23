"""IFEval 打分（任务 #71 的文本基准）：用官方的 evaluation_lib 对生成结果做严格与宽松两种判定。

    python -m pipeline.eval.ifeval.score --input <IFEval 目录>/ifeval_input_data.jsonl \\
        --responses <ifeval.responses.jsonl> --output <ifeval.json>

只用 CPU，需要 langdetect 与 nltk（≥ 3.9.1，带 punkt_tab 数据，缺少时报错退出，不自动下载）。生成由
pipeline.eval.text_bench ifeval 完成，responses.jsonl 每行含 key、prompt、response，还可以带生成的统计（n_new_tokens、
capped）。只给部分提问的回答时（冒烟测试的 --ifeval-limit）只对这些提问打分。

指标与官方相同（evaluation_lib.print_report）：prompt 级为全部指令都照做的提问比例，instruction 级为照做的指令比例；
strict 直接判定回答，loose 另外试去掉第一行、最后一行、两者以及去掉 `*` 的几种写法，任一种照做即算照做。
language:response_language 与英文大小写两类指令用 langdetect 判断语言；langdetect 默认带随机性，这里固定
DetectorFactory.seed = 0，同一份回答每次判定相同（官方代码没有固定种子）。

输出 --output（JSON）：n、prompt_strict、inst_strict、prompt_loose、inst_loose，tiers 中按指令大类（tier0）与具体指令
（tier1）的照做比例，generation 中回答长度与写到上限的比例；同目录 <名称>.items.jsonl 为逐条的判定（key 与每条指令
strict / loose 是否照做），不含提问与回答原文。
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

CHECKER = "google-research/google-research instruction_following_eval @ e6890f85757dd84e27ca6df2dd30651dafad28e0"


def check_dependencies() -> None:
    """langdetect、nltk 与 punkt_tab 数据都在，否则报错退出（不下载）。"""
    try:
        import langdetect  # noqa: F401
        import nltk
    except ImportError as error:
        raise SystemExit(f"IFEval 打分需要 langdetect 与 nltk：{error}") from error
    try:
        nltk.data.find(resource_name = "tokenizers/punkt_tab")
    except LookupError as error:
        raise SystemExit("IFEval 打分需要 nltk 的 punkt_tab 数据（nltk.download('punkt_tab')），这里不自动下载") from error


def rates(outputs: list) -> dict:
    """官方 print_report 中的各项比例：prompt 级、instruction 级，以及按大类（tier0）和具体指令（tier1）。"""
    prompt_correct = sum(all(item.follow_instruction_list) for item in outputs)
    instruction_total = sum(len(item.instruction_id_list) for item in outputs)
    instruction_correct = sum(sum(item.follow_instruction_list) for item in outputs)
    tier0 = collections.defaultdict(lambda: [0, 0])
    tier1 = collections.defaultdict(lambda: [0, 0])
    for item in outputs:
        for instruction_id, followed in zip(item.instruction_id_list, item.follow_instruction_list):
            for table, key in ((tier0, instruction_id.split(":")[0]), (tier1, instruction_id)):
                table[key][0] += int(followed)
                table[key][1] += 1
    return {
        "prompt": prompt_correct / len(outputs),
        "instruction": instruction_correct / instruction_total,
        "tier0": {key: round(number = value[0] / value[1], ndigits = 4) for key, value in sorted(tier0.items())},
        "tier1": {key: round(number = value[0] / value[1], ndigits = 4) for key, value in sorted(tier1.items())},
    }


def score(input_path: Path, responses_path: Path) -> tuple:
    """返回 (汇总, 逐条判定列表)。"""
    from langdetect import DetectorFactory

    from pipeline.eval.ifeval import evaluation_lib

    DetectorFactory.seed = 0
    inputs = evaluation_lib.read_prompt_list(input_jsonl_filename = str(input_path))
    with open(file = responses_path, encoding = "utf-8") as handle:
        responses = [json.loads(s = line) for line in handle if line.strip()]
    by_prompt = {row["prompt"]: row["response"] for row in responses}
    key_of = {row["prompt"]: row["key"] for row in responses}
    known = {item.prompt for item in inputs}
    unknown = [row["key"] for row in responses if row["prompt"] not in known]
    if unknown:
        raise ValueError(f"回答中的提问不在 IFEval 输入中：key {unknown[:5]}")
    inputs = [item for item in inputs if item.prompt in by_prompt]
    strict = [evaluation_lib.test_instruction_following_strict(inp = item, prompt_to_response = by_prompt) for item in inputs]
    loose = [evaluation_lib.test_instruction_following_loose(inp = item, prompt_to_response = by_prompt) for item in inputs]
    strict_rates = rates(outputs = strict)
    loose_rates = rates(outputs = loose)
    tokens = [row["n_new_tokens"] for row in responses if "n_new_tokens" in row]
    summary = {
        "bench": "ifeval",
        "n": len(inputs),
        "prompt_strict": round(number = strict_rates["prompt"], ndigits = 4),
        "inst_strict": round(number = strict_rates["instruction"], ndigits = 4),
        "prompt_loose": round(number = loose_rates["prompt"], ndigits = 4),
        "inst_loose": round(number = loose_rates["instruction"], ndigits = 4),
        "tiers": {
            "strict": {"tier0": strict_rates["tier0"], "tier1": strict_rates["tier1"]},
            "loose": {"tier0": loose_rates["tier0"], "tier1": loose_rates["tier1"]},
        },
        "generation": {
            "mean_new_tokens": round(number = sum(tokens) / len(tokens), ndigits = 1) if tokens else None,
            "capped": round(number = sum(bool(row.get("capped")) for row in responses) / len(responses), ndigits = 4),
            "empty": round(number = sum(not row["response"].strip() for row in responses) / len(responses), ndigits = 4),
        },
        "checker": CHECKER,
        "langdetect_seed": 0,
    }
    items = [
        {
            "key": key_of[item_strict.prompt],
            "instruction_id_list": item_strict.instruction_id_list,
            "strict": item_strict.follow_instruction_list,
            "loose": item_loose.follow_instruction_list,
        }
        for item_strict, item_loose in zip(strict, loose)
    ]
    return summary, items


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required = True, type = Path, help = "IFEval 的 ifeval_input_data.jsonl")
    parser.add_argument("--responses", required = True, type = Path, help = "text_bench ifeval 写出的 responses.jsonl")
    parser.add_argument("--output", required = True, type = Path, help = "打分结果 JSON")
    args = parser.parse_args(args = argv)
    check_dependencies()
    summary, items = score(input_path = args.input, responses_path = args.responses)
    generation = args.responses.with_name(args.responses.name.replace(".responses.jsonl", ".generation.json"))
    if generation != args.responses and generation.exists():
        summary["generation"].update(json.loads(s = generation.read_text(encoding = "utf-8")))
    args.output.parent.mkdir(parents = True, exist_ok = True)
    with open(file = args.output.with_name(args.output.stem + ".items.jsonl"), mode = "w", encoding = "utf-8") as handle:
        for item in items:
            handle.write(json.dumps(obj = item, ensure_ascii = False) + "\n")
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(data = json.dumps(obj = summary, ensure_ascii = False, indent = 1) + "\n", encoding = "utf-8")
    temporary.replace(target = args.output)
    print(
        f"IFEval {summary['n']} 条：prompt strict {summary['prompt_strict']:.4f}，inst strict {summary['inst_strict']:.4f}，"
        f"prompt loose {summary['prompt_loose']:.4f}，inst loose {summary['inst_loose']:.4f}",
        flush = True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
