"""把多个输出目录的训练清单合并成一份，audio_path 改为绝对路径（实验 v1，任务 T14；真人语音批次 #32）。

每个输出目录的 manifest.jsonl 中 audio_path 相对该目录；合并后的清单放在哪里都能直接找到音频。

给出 ``--cap-hours`` 时，每种语言累计时长（duration_s 之和）达到上限后不再收录该语言的条目；
``--uncapped`` 列出的语言不受上限，``--lang-cap zh=900`` 给单个语言另设上限（实验 v2b：中文 900 小时、其余 600 小时）。
同一个 sample_id 只收录第一次出现的条目（例如同一批合成音频由两次核对各写出一份清单）。目录按命令行顺序处理，排在前面的目录优先占用上限，
因此人工核对过的数据（例如 Common Voice）应写在前面。同一目录内的条目先用 ``--seed`` 打乱再收录，
超出上限时得到的是随机子集，避免只保留原始顺序靠前的说话人或章节。

用法::

    python -m pipeline.runner.merge_manifests <输出 JSONL> <输出目录 1> [<输出目录 2> ...] \\
        [--cap-hours 600 --uncapped en --lang-cap zh=900 --seed 0]
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def read_manifest(directory: Path) -> list[dict]:
    # 目录只解析一次；逐条 resolve() 在几十万条的清单上要做几十万次文件系统查询
    base = directory.resolve()
    rows = []
    with open(file = directory / "manifest.jsonl", mode = "r", encoding = "utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(s = line)
            row["audio_path"] = str(base / row["audio_path"])
            rows.append(row)
    return rows


def merge(
    directories: list[Path],
    cap_hours: float | None = None,
    uncapped: set[str] = frozenset(),
    seed: int = 0,
    lang_caps: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    """返回 (合并后的条目, 每个目录的收录统计)。

    不给上限时保持各目录内原有顺序；给上限时目录内先打乱，再按语言累计时长截断。lang_caps 为 {语言: 小时}，
    优先于 cap_hours。
    """
    caps = {lang: hours * 3600 for lang, hours in (lang_caps or {}).items()}
    cap_seconds = None if cap_hours is None else cap_hours * 3600
    used = defaultdict(float)
    seen = set()
    merged = []
    report = []
    for directory in directories:
        rows = read_manifest(directory = directory)
        if cap_seconds is not None or caps:
            random.Random(f"{seed}:{directory.name}").shuffle(rows)
        kept = 0
        duplicates = 0
        kept_seconds = 0.0
        total_seconds = 0.0
        for row in rows:
            seconds = float(row.get("duration_s") or 0.0)
            total_seconds += seconds
            if row["sample_id"] in seen:
                duplicates += 1
                continue
            lang = row.get("lang")
            limit = caps.get(lang, None if lang in uncapped else cap_seconds)
            if limit is not None and used[lang] + seconds > limit:
                continue
            seen.add(row["sample_id"])
            used[lang] += seconds
            merged.append(row)
            kept += 1
            kept_seconds += seconds
        report.append({
            "directory": str(directory),
            "rows": len(rows),
            "kept": kept,
            "hours": round(total_seconds / 3600, 2),
            "kept_hours": round(kept_seconds / 3600, 2),
            "duplicates": duplicates,
        })
    return merged, report


def main() -> None:
    parser = argparse.ArgumentParser(description = "合并多个输出目录的 manifest.jsonl")
    parser.add_argument("out", type = Path)
    parser.add_argument("directories", type = Path, nargs = "+")
    parser.add_argument("--cap-hours", type = float, default = None, help = "每种语言的时长上限（小时）")
    parser.add_argument("--uncapped", nargs = "*", default = [], help = "不受上限的语言代码")
    parser.add_argument("--lang-cap", nargs = "*", default = [], help = "单个语言的上限，如 zh=900，优先于 --cap-hours")
    parser.add_argument("--seed", type = int, default = 0)
    args = parser.parse_args()
    lang_caps = {item.split("=")[0]: float(item.split("=")[1]) for item in args.lang_cap}

    rows, report = merge(
        directories = args.directories,
        cap_hours = args.cap_hours,
        uncapped = set(args.uncapped),
        seed = args.seed,
        lang_caps = lang_caps,
    )
    args.out.parent.mkdir(parents = True, exist_ok = True)
    with open(file = args.out, mode = "w", encoding = "utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(obj = row, ensure_ascii = False) + "\n")

    hours = defaultdict(float)
    for row in rows:
        hours[row.get("lang")] += float(row.get("duration_s") or 0.0) / 3600
    for item in report:
        note = f"，重复 {item['duplicates']} 条" if item["duplicates"] else ""
        print(f"{item['directory']}: 收录 {item['kept']}/{item['rows']} 条，{item['kept_hours']}/{item['hours']} 小时{note}")
    print("按语言：", json.dumps({lang: round(value, 1) for lang, value in sorted(hours.items())}, ensure_ascii = False))
    print(len(rows), args.out)


if __name__ == "__main__":
    main()
