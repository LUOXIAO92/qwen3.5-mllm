"""用主 ASR 的识别结果替换数据集的转写（重新识别模式，在 envs/runner 中运行）。

数据集的转写本身由模型生成、质量不可靠时（例如 Emilia 的原始转写），不再拿它作比对依据：主 ASR 跑完之后，
把每个任务的 text 和 original 换成主 ASR 的识别结果，原转写存到 source_transcript 字段；随后的 compare 里主 ASR
与自身一致，交叉 ASR 与主 ASR 的结果比对，两个不同模型都认同的样本才通过。训练目标就是主 ASR 的识别结果。

用法::

    python -m pipeline.realspeech.adopt_asr --out-dir <输出目录>

主 ASR 没有结果或结果为空的任务保留原样，会在比对中被判为不一致。重复运行时已替换过的任务不再改动。
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description = "用主 ASR 的识别结果替换数据集的转写")
    parser.add_argument("--out-dir", required = True, type = Path)
    args = parser.parse_args()

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
    tmp = tasks_path.with_suffix(".jsonl.tmp")
    with open(file = tmp, mode = "w", encoding = "utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(obj = task, ensure_ascii = False) + "\n")
    tmp.replace(tasks_path)
    print(json.dumps(obj = {"tasks": len(tasks), "replaced": replaced}, ensure_ascii = False))


if __name__ == "__main__":
    main()
