"""把 T02 的 FLEURS dev 清单转成训练清单格式，作为开发用清单；另取一个多语言小子集做过拟合测试。

用法：

    python -m pipeline.train.fleurs_dev \\
        --fleurs-dir <T02 产出目录，含 <lang>/dev.jsonl> \\
        --output <dev 训练清单.jsonl> \\
        --overfit-output <过拟合子集.jsonl> --overfit-size 32 --overfit-max-duration-s 15

FLEURS 是真人录音，朗读稿和原文都取 raw_transcription；TTS 相关字段（tts_model、voice_id、
asr_model）留空，holdout 与 cross_lingual 为 false。audio_path 写成绝对路径（不展开符号链接）。
"""

import argparse
import json
import random
from pathlib import Path

LANGS = ("zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it", "id", "ms", "th", "vi", "fil", "ar", "tr", "fa")


def convert_row(row, fleurs_dir):
    return {
        "sample_id": row["sample_id"],
        "audio_path": str((fleurs_dir / row["audio_path"]).absolute()),
        "duration_s": row["duration_s"],
        "lang": row["lang"],
        "original": row["raw_transcription"],
        "script": row["raw_transcription"],
        "source": row["source"],
        "source_license": row["source_license"],
        "tts_model": None,
        "voice_id": None,
        "holdout": False,
        "cross_lingual": False,
        "asr_model": None,
        "speaker_id": row.get("speaker_id"),
        "fleurs_id": row.get("fleurs_id"),
    }


def pick_overfit(rows, size, max_duration_s, seed):
    """按语言轮流抽样，直到凑够 size 条，每条不超过 max_duration_s 秒。"""
    generator = random.Random(x = seed)
    pools = {}
    for row in rows:
        if row["duration_s"] <= max_duration_s and row["original"].strip():
            pools.setdefault(row["lang"], []).append(row)
    for lang in pools:
        generator.shuffle(x = pools[lang])
    order = [lang for lang in LANGS if lang in pools]
    selected = []
    while len(selected) < size and any(pools[lang] for lang in order):
        for lang in order:
            if pools[lang] and len(selected) < size:
                selected.append(pools[lang].pop())
    return selected


def write_jsonl(path, rows):
    path.parent.mkdir(parents = True, exist_ok = True)
    with open(file = path, mode = "w", encoding = "utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(obj = row, ensure_ascii = False) + "\n")


def main(argv = None):
    parser = argparse.ArgumentParser(description = "FLEURS dev 清单转训练清单格式")
    parser.add_argument("--fleurs-dir", required = True, type = Path)
    parser.add_argument("--output", required = True, type = Path)
    parser.add_argument("--overfit-output", default = None, type = Path)
    parser.add_argument("--overfit-size", type = int, default = 32)
    parser.add_argument("--overfit-max-duration-s", type = float, default = 15.0)
    parser.add_argument("--seed", type = int, default = 0)
    args = parser.parse_args(args = argv)

    rows = []
    for lang in LANGS:
        path = args.fleurs_dir / lang / "dev.jsonl"
        with open(file = path, mode = "r", encoding = "utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(convert_row(row = json.loads(s = line), fleurs_dir = args.fleurs_dir))
    write_jsonl(path = args.output, rows = rows)
    print(f"dev 清单：{len(rows)} 条 → {args.output}")

    if args.overfit_output is not None:
        subset = pick_overfit(
            rows = rows,
            size = args.overfit_size,
            max_duration_s = args.overfit_max_duration_s,
            seed = args.seed,
        )
        write_jsonl(path = args.overfit_output, rows = subset)
        langs = sorted({row["lang"] for row in subset})
        seconds = sum(row["duration_s"] for row in subset)
        print(f"过拟合子集：{len(subset)} 条，{len(langs)} 种语言，共 {seconds:.0f} 秒 → {args.overfit_output}")


if __name__ == "__main__":
    main()
