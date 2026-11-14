"""用主 ASR 的识别结果替换数据集的转写（重新识别模式，在 envs/runner 中运行）。

数据集的转写本身由模型生成、质量不可靠时（例如 Emilia 的原始转写），不再拿它作比对依据：主 ASR 跑完之后，
把每个任务的 text 和 original 换成主 ASR 的识别结果，原转写存到 source_transcript 字段；随后的 compare 里主 ASR
与自身一致，交叉 ASR 与主 ASR 的结果比对，两个不同模型都认同的样本才通过。训练目标就是主 ASR 的识别结果。

用法::

    python -m pipeline.realspeech.adopt_asr --out-dir <输出目录>

主 ASR 没有结果或结果为空的任务保留原样，会在比对中被判为不一致。重复运行时已替换过的任务不再改动。

--mark 只给全部任务标上 retranscribe（在时长检查之前运行，见 run.sh）：数据集的原转写不作依据，时长检查不再按
原转写的「秒/字」比例筛选，只检查时长不为零，所有音频都交给主 ASR 识别。WorldSpeech 的原转写与片段常对不齐，
按它筛选曾在识别之前丢掉 1,014 条（ko 374、ru 302、tr 189、th 108 等），其中多数从未经过 ASR（用户 2026-11-14：
重要的是时长分布和正确性，不设比例上下限）。

    python -m pipeline.realspeech.adopt_asr --out-dir <输出目录> --mark
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description = "用主 ASR 的识别结果替换数据集的转写")
    parser.add_argument("--out-dir", required = True, type = Path)
    parser.add_argument("--mark", action = "store_true", help = "只给全部任务标上 retranscribe")
    args = parser.parse_args()

    if args.mark:
        tasks_path = args.out_dir / "tasks.jsonl"
        with open(file = tasks_path, mode = "r", encoding = "utf-8") as handle:
            tasks = [json.loads(s = line) for line in handle if line.strip()]
        for task in tasks:
            task["retranscribe"] = True
        write_tasks(path = tasks_path, tasks = tasks)
        print(json.dumps(obj = {"tasks": len(tasks), "marked": len(tasks)}, ensure_ascii = False))
        return

    hypotheses = {}
    with open(file = args.out_dir / "asr_primary.jsonl", mode = "r", encoding = "utf-8") as handle:
        for line in handle:
            row = json.loads(s = line)
            if (row.get("hypothesis") or "").strip():
                hypotheses[row["sample_id"]] = " ".join(row["hypothesis"].split())
    tasks_path = args.out_dir / "tasks.jsonl"
    with open(file = tasks_path, mode = "r", encoding = "utf-8") as handle:
        tasks = [json.loads(s = line) for line in handle if line.strip()]
    replaced = 0
    for task in tasks:
        if "source_transcript" in task or task["sample_id"] not in hypotheses:
            continue
        task["source_transcript"] = task["original"]
        task["text"] = task["original"] = hypotheses[task["sample_id"]]
        task["target_from"] = "transcript"
        replaced += 1
    write_tasks(path = tasks_path, tasks = tasks)
    print(json.dumps(obj = {"tasks": len(tasks), "replaced": replaced}, ensure_ascii = False))


def write_tasks(path: Path, tasks: list) -> None:
    tmp = path.with_suffix(".jsonl.tmp")
    with open(file = tmp, mode = "w", encoding = "utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(obj = task, ensure_ascii = False) + "\n")
    tmp.replace(path)


if __name__ == "__main__":
    main()
