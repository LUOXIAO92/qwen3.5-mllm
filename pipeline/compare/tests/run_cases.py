"""逐语言统计样例结果：python -m pipeline.compare.tests.run_cases [-v]

-v 时列出所有未通过的样例。
"""

from __future__ import annotations

import sys

from pipeline.compare import LANGS, compare
from pipeline.compare.tests.cases import NEGATIVE, POSITIVE


def evaluate() -> tuple:
    rows = []
    failures = []
    for lang in LANGS:
        pos_ok = 0
        for script, hyp, original, note in POSITIVE[lang]:
            r = compare(
                script = script,
                hypothesis = hyp,
                lang = lang,
                original = original,
            )
            if r.match:
                pos_ok += 1
            else:
                failures.append((lang, "正例", note, script, hyp, original, r.detail))
        neg_ok = 0
        for script, hyp, original, note in NEGATIVE[lang]:
            r = compare(
                script = script,
                hypothesis = hyp,
                lang = lang,
                original = original,
            )
            if not r.match:
                neg_ok += 1
            else:
                failures.append((lang, "反例", note, script, hyp, original, "被误收"))
        rows.append((lang, len(POSITIVE[lang]), pos_ok, len(NEGATIVE[lang]), neg_ok))
    return rows, failures


def main(argv: list) -> int:
    rows, failures = evaluate()
    print("| 语言 | 正例 | 正例通过 | 反例 | 反例拒绝（误收） |")
    print("| --- | --- | --- | --- | --- |")
    for lang, n_pos, pos_ok, n_neg, neg_ok in rows:
        print(f"| {lang} | {n_pos} | {pos_ok} | {n_neg} | {neg_ok}（{n_neg - neg_ok}） |")
    false_accepts = sum(n_neg - neg_ok for _, _, _, n_neg, neg_ok in rows)
    print(f"\n正例 {sum(r[1] for r in rows)} 组，反例 {sum(r[3] for r in rows)} 组，误收 {false_accepts} 组。")
    if "-v" in argv:
        for row in failures:
            print(" | ".join(str(x) for x in row))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(argv = sys.argv[1:]))
