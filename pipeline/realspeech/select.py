"""从真人语音数据集选出候选条目：清理转写、检查语言、去掉 FLEURS 句子（在 envs/t03 中运行）。

用法（在仓库根目录运行）::

    python -m pipeline.realspeech.select \\
        --source emilia \\
        --lang de \\
        --input $DATASET/audio/amphion/Emilia-Dataset/Emilia/DE/*.tar \\
        --out-dir <输出目录> \\
        --glotlid <GlotLID model_v3.bin> \\
        --exclusion <FLEURS exclusion.jsonl> \\
        --target transcript

--source 取 sources.py 中 SOURCES 的名字；--input 对 makesense_emilia 是数据根目录（后面可再跟存放其余音频的同结构目录），对其余数据集是文件列表。
--target transcript 表示比对通过后用数据集的转写作训练目标；asr 表示改用主 ASR 的识别结果（转写没有标点或大小写时）。
--human-verified 标记转写已经过人工核验（如 Common Voice 的已验证录音）：之后跳过 ASR 核对，只做时长检查。
--max-hours 按数据集给出的时长累计，选够这么多小时就停（没有时长的条目不计入）；--max-per-transcript 限制同一句话
最多选几条录音（Common Voice 中同一句常由许多人朗读）。
写出 <输出目录>/selected.jsonl 和 select_stats.json（各丢弃原因的条数）。
"""

import argparse
import inspect
import json
import re
from collections import Counter
from pathlib import Path

from pipeline.text.exclusion import is_excluded, load_exclusion
from pipeline.text.langid import LangIdentifier
from pipeline.text.langs import GLOTLID_LABELS

from .sources import SOURCES

# 中文、日语的字与字之间不该有空格（WorldSpeech 的中文转写在词之间插了空格）
_CJK = r"　-〿぀-ヿ㐀-䶿一-鿿＀-￯"
_CJK_SPACE = re.compile(rf"(?<=[{_CJK}])\s+(?=[{_CJK}])")
_SOFT_HYPHEN = "­"


def clean_transcript(text: str, lang: str) -> str:
    """去掉软连字符，合并连续空白；中文、日语再去掉字与字之间的空格。"""
    text = " ".join(text.replace(_SOFT_HYPHEN, "").split())
    if lang in ("zh", "ja"):
        text = _CJK_SPACE.sub("", text)
    return text


def safe_id(text: str) -> str:
    return re.sub(pattern = r"[^A-Za-z0-9_.-]", repl = "_", string = text)


def main() -> None:
    parser = argparse.ArgumentParser(description = "选出真人语音的候选条目")
    parser.add_argument("--source", required = True, choices = sorted(SOURCES))
    parser.add_argument("--lang", required = True)
    parser.add_argument("--input", required = True, nargs = "+")
    parser.add_argument("--out-dir", required = True, type = Path)
    parser.add_argument("--glotlid", required = True, help = "GlotLID model_v3.bin")
    parser.add_argument("--exclusion", required = True, help = "FLEURS 排除清单 exclusion.jsonl")
    parser.add_argument("--target", choices = ["transcript", "asr"], default = "transcript")
    parser.add_argument("--min-lid-prob", type = float, default = 0.5)
    parser.add_argument("--extra-labels", nargs = "*", default = [], help = "也接受的 GlotLID 标签，例如马来语接受 ind_Latn")
    parser.add_argument("--min-duration", type = float, default = 1.0)
    parser.add_argument("--max-duration", type = float, default = 30.0)
    parser.add_argument("--max-cer", type = float, default = 1.0, help = "worldspeech：按数据集自带的 cer 粗筛")
    parser.add_argument("--max-wmer", type = float, default = 100.0, help = "mgb2：按片段的 WMER（%%）粗筛")
    parser.add_argument("--limit", type = int, default = None, help = "最多选出多少条（小规模试验用）")
    parser.add_argument("--max-hours", type = float, default = None, help = "按数据集给出的时长累计，选够这么多小时就停")
    parser.add_argument("--max-per-transcript", type = int, default = 0, help = "同一句转写最多选几条录音，0 为不限")
    parser.add_argument("--human-verified", action = "store_true", help = "转写已经人工核验，之后跳过 ASR 核对")
    args = parser.parse_args()

    reader = SOURCES[args.source]
    if args.source == "makesense_emilia":
        rows = reader(root = args.input[0], lang = args.lang, audio_roots = tuple(args.input[1:]))
    else:
        options = {"max_cer": args.max_cer, "max_wmer": args.max_wmer}
        accepted = inspect.signature(reader).parameters
        rows = reader(paths = args.input, lang = args.lang, **{key: value for key, value in options.items() if key in accepted})

    identifier = LangIdentifier(model_path = args.glotlid)
    exclusion = load_exclusion(path = args.exclusion)
    labels = set(GLOTLID_LABELS[args.lang]) | set(args.extra_labels)
    reasons = Counter()
    seen = set()
    per_transcript = Counter()
    args.out_dir.mkdir(parents = True, exist_ok = True)
    selected = 0
    seconds = 0.0
    with open(file = args.out_dir / "selected.jsonl", mode = "w", encoding = "utf-8") as handle:
        for row in rows:
            text = clean_transcript(text = row["transcript"], lang = args.lang)
            duration = row.get("duration_s")
            if not text:
                reasons["empty_transcript"] += 1
                continue
            if duration is not None and not args.min_duration <= duration <= args.max_duration:
                reasons["duration"] += 1
                continue
            label, probability = identifier.predict(text = text)
            if label not in labels or probability < args.min_lid_prob:
                reasons[f"lid:{label or 'none'}"] += 1
                continue
            if is_excluded(text = text, lang = args.lang, exclusion = exclusion):
                reasons["fleurs"] += 1
                continue
            if args.max_per_transcript and per_transcript[text] >= args.max_per_transcript:
                reasons["repeated_transcript"] += 1
                continue
            sample_id = f"real-{args.source}-{args.lang}-{safe_id(text = row['uid'])}"
            if sample_id in seen:
                reasons["duplicate_id"] += 1
                continue
            seen.add(sample_id)
            per_transcript[text] += 1
            row = {**row, "sample_id": sample_id, "transcript": text, "target_from": args.target, "lid": [label, round(probability, 3)]}
            if args.human_verified:
                row["human_verified"] = True
            handle.write(json.dumps(obj = row, ensure_ascii = False) + "\n")
            selected += 1
            seconds += duration or 0.0
            if args.limit and selected >= args.limit:
                break
            if args.max_hours and seconds >= args.max_hours * 3600:
                break
    stats = {"selected": selected, "hours_with_duration": round(seconds / 3600, 2), "dropped": dict(reasons.most_common())}
    (args.out_dir / "select_stats.json").write_text(json.dumps(obj = stats, ensure_ascii = False, indent = 2), encoding = "utf-8")
    print(json.dumps(obj = stats, ensure_ascii = False))


if __name__ == "__main__":
    main()
