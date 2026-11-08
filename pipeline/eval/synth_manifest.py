"""把合成测试集的清单转成评测清单的格式（实验 v1，任务 T16）。

合成—核对流水线在输出目录写出 manifest.jsonl（核对通过的样本，audio_path 相对该目录）。评测脚本
pipeline.eval.run_eval 读取 <清单目录>/<语言>/test.jsonl，参考文本取 raw_transcription。这里按语言拆开，
audio_path 改为绝对路径，参考文本取原文（original），其余字段（tts_model、voice_id、mode、cross_lingual 等）
原样保留，供分组统计使用。

用法::

    python -m pipeline.eval.synth_manifest <合成输出目录> <评测清单目录>
"""

import json
import sys
from pathlib import Path


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    by_lang = {}
    with open(file = src / "manifest.jsonl", mode = "r", encoding = "utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(s = line)
            row["audio_path"] = str((src / row["audio_path"]).resolve())
            row["raw_transcription"] = row["original"]
            by_lang.setdefault(row["lang"], []).append(row)
    for lang, rows in by_lang.items():
        (dst / lang).mkdir(parents = True, exist_ok = True)
        with open(file = dst / lang / "test.jsonl", mode = "w", encoding = "utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(obj = row, ensure_ascii = False) + "\n")
    print(json.dumps(obj = {lang: len(rows) for lang, rows in sorted(by_lang.items())}))


if __name__ == "__main__":
    main()
