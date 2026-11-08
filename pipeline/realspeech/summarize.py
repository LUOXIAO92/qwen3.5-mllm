"""汇总若干个导入—核对输出目录的结果，打印 Markdown 表格。

每个输出目录读取 stats.json（判定条数、通过条数）、synth_results.jsonl（全部样本的时长）和 manifest.jsonl
（通过样本的时长），列出：样本数、通过数、通过率（按条数）、全部时长、通过时长、时长通过率。

用法::

    python -m pipeline.realspeech.summarize <目录 1> [<目录 2> ...]
"""

import json
import sys
from pathlib import Path


def read_jsonl(path: Path) -> list:
    with open(file = path, mode = "r", encoding = "utf-8") as handle:
        return [json.loads(s = line) for line in handle if line.strip()]


def summarize(directory: Path):
    if not (directory / "manifest.jsonl").exists() or not (directory / "stats.json").exists():
        return None
    stats = json.loads(s = (directory / "stats.json").read_text(encoding = "utf-8"))
    total_s = sum(row["duration_s"] or 0 for row in read_jsonl(path = directory / "synth_results.jsonl"))
    pass_s = sum(row["duration_s"] or 0 for row in read_jsonl(path = directory / "manifest.jsonl"))
    return {
        "name": directory.name,
        "samples": stats["total"],
        "pass": stats["pass"],
        "rate": stats["pass"] / stats["total"] if stats["total"] else 0.0,
        "total_h": total_s / 3600,
        "pass_h": pass_s / 3600,
    }


def main() -> None:
    rows = [row for row in (summarize(directory = Path(arg)) for arg in sys.argv[1:]) if row]
    print("| 目录 | 样本 | 通过 | 通过率 | 全部时长 | 通过时长 | 时长通过率 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        share = row["pass_h"] / row["total_h"] if row["total_h"] else 0.0
        print(
            f"| {row['name']} | {row['samples']} | {row['pass']} | {row['rate']:.1%} | "
            f"{row['total_h']:.2f} h | {row['pass_h']:.2f} h | {share:.1%} |"
        )
    print(f"| 合计 | {sum(r['samples'] for r in rows)} | {sum(r['pass'] for r in rows)} | | "
          f"{sum(r['total_h'] for r in rows):.2f} h | {sum(r['pass_h'] for r in rows):.2f} h | |")


if __name__ == "__main__":
    main()
