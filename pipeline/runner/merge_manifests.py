"""把多个输出目录的训练清单合并成一份，audio_path 改为绝对路径（实验 v1，任务 T14）。

每个输出目录的 manifest.jsonl 中 audio_path 相对该目录；合并后的清单放在哪里都能直接找到音频。

用法::

    python -m pipeline.runner.merge_manifests <输出 JSONL> <输出目录 1> [<输出目录 2> ...]
"""

import json
import sys
from pathlib import Path


def main() -> None:
    out = Path(sys.argv[1])
    rows = []
    for directory in map(Path, sys.argv[2:]):
        with open(file = directory / "manifest.jsonl", mode = "r", encoding = "utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(s = line)
                row["audio_path"] = str((directory / row["audio_path"]).resolve())
                rows.append(row)
    out.parent.mkdir(parents = True, exist_ok = True)
    with open(file = out, mode = "w", encoding = "utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(obj = row, ensure_ascii = False) + "\n")
    print(len(rows), out)


if __name__ == "__main__":
    main()
