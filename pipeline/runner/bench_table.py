"""把 bench.py 写出的 bench.jsonl 汇总成「批次—吞吐—峰值显存」的 markdown 表。

用法::

    python -m pipeline.runner.bench_table <bench 目录>   # 目录下每个子目录有一个 bench.jsonl
"""

import json
import sys
from pathlib import Path


def main() -> None:
    base = Path(sys.argv[1])
    print("| 模型/组 | 进程 | 批次 | 条数 | 用时 s | 启动 s | 样本/秒 | 音频秒/秒 | 峰值显存 GB | GPU 利用率均值 % | 备注 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for path in sorted(base.glob("*/bench.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding = "utf-8").splitlines()]
        # 启动时间有冷热缓存的波动，同一组里取最小值（热缓存）来折算吞吐
        floor = min((row["startup_s"] for row in rows if row["ok"]), default = 0.0)
        for row in rows:
            if row["ok"]:
                effective = max(row["wall_s"] - floor, 1e-6)
                row["samples_per_s"] = round(row["n_ok"] / effective, 2)
                if row.get("audio_s_per_s") is not None:
                    row["audio_s_per_s"] = round(row["audio_s"] / effective, 2)
                row["startup_s"] = floor
            if not row["ok"]:
                note = "失败：" + str(row.get("error", ""))[:60]
                print(f"| {path.parent.name} | {row['processes']} | {row['batch_size']} | {row['n']} | {row['wall_s']} | | | | {row['peak_delta_mib'] / 1024:.1f} | | {note} |")
                continue
            flag = "接近 32 GB，不选" if row["peak_mib"] >= 31000 else ""
            audio = row["audio_s_per_s"] if row.get("audio_s_per_s") is not None else ""
            print(
                f"| {path.parent.name} | {row['processes']} | {row['batch_size']} | {row['n_ok']}/{row['n']} | {row['wall_s']} | "
                f"{row['startup_s']} | {row['samples_per_s']} | {audio} | {row['peak_delta_mib'] / 1024:.1f} | {row['util_mean']} | {flag} |"
            )


if __name__ == "__main__":
    main()
