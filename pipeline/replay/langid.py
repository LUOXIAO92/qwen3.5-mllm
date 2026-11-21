"""给 finalize 写出的样本补上语言：GlotLID v3 判断生成文本的语言，另记主要文字。

    python -m pipeline.replay.langid --replay-dir <回放目录> --langid-model <GlotLID model_v3.bin>

在 `envs/t03` 中运行（需要 fasttext）。判断的文本：无模板的两组为 `generated_text`，带模板的两组为
`assistant_text`，free_chat 的 `user_text` 另判一次。字段：`lang`（项目的语言代码；GlotLID 给出其他语言时为
`other`，文本为空时为 `none`）、`lang_label`、`lang_prob`、`script`（字母中占比最高的 Unicode 文字），
free_chat 另有 `user_lang`、`user_lang_label`。结果直接写回 `samples.jsonl`。
"""

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path

from pipeline.text.langid import LangIdentifier
from pipeline.text.langs import GLOTLID_LABELS

GROUPS = ("free_raw", "free_chat", "cont_raw", "cont_chat")
SPLITS = ("eval", "train")
LABEL_TO_LANG = {label: lang for lang, labels in GLOTLID_LABELS.items() for label in labels}
MAX_CHARS = 20000


def main_script(text):
    counts = Counter()
    for char in text[:MAX_CHARS]:
        if char.isalpha():
            name = unicodedata.name(char, "")
            counts[name.split(" ")[0] if name else "UNKNOWN"] += 1
    return counts.most_common(1)[0][0] if counts else "NONE"


def detect(identifier, text):
    text = (text or "").strip()
    if not text:
        return "none", "", 0.0
    label, prob = identifier.predict(text[:MAX_CHARS])
    return LABEL_TO_LANG.get(label, "other"), label, round(prob, 4)


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay-dir", required = True)
    parser.add_argument("--langid-model", required = True)
    parser.add_argument("--groups", nargs = "+", default = list(GROUPS))
    parser.add_argument("--splits", nargs = "+", default = list(SPLITS))
    args = parser.parse_args()
    identifier = LangIdentifier(model_path = args.langid_model)
    for group in args.groups:
        for split in args.splits:
            path = Path(args.replay_dir) / group / split / "samples.jsonl"
            if not path.exists():
                continue
            with open(path, encoding = "utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            for row in rows:
                text = row.get("generated_text") if group in ("free_raw", "cont_raw") else row.get("assistant_text")
                row["lang"], row["lang_label"], row["lang_prob"] = detect(identifier = identifier, text = text)
                row["script"] = main_script(text = text or "")
                if group == "free_chat":
                    row["user_lang"], row["user_lang_label"], _ = detect(identifier = identifier, text = row.get("user_text"))
            tmp = path.with_suffix(".jsonl.tmp")
            with open(tmp, "w", encoding = "utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii = False) + "\n")
            tmp.replace(path)
            counts = Counter(row["lang"] for row in rows if row["keep"])
            print(f"{group}/{split}: {dict(counts.most_common())}", flush = True)


if __name__ == "__main__":
    main()
