"""逐语言统计样例结果：python -m pipeline.compare.tests.run_cases [-v]

先列出原有样例（cases.py）的逐语言结果，再列出比对增强样例（cases_ext.py）的逐组结果。
-v 时列出所有未通过的样例。
"""

from __future__ import annotations

import sys

from pipeline.compare import LANGS, compare, ja_reading
from pipeline.compare.tests.cases import NEGATIVE, POSITIVE
from pipeline.compare.tests.cases_ext import EXT_NEGATIVE, EXT_POSITIVE


def _check(rows: list, expect: bool, label: str, failures: list) -> int:
    ok = 0
    for lang, script, hyp, original, note in rows:
        r = compare(
            script = script,
            hypothesis = hyp,
            lang = lang,
            original = original,
        )
        if r.match == expect:
            ok += 1
        else:
            failures.append((lang, label, note, script, hyp, original, r.detail or "被误收"))
    return ok


def evaluate() -> tuple:
    rows = []
    failures = []
    for lang in LANGS:
        pos = [(lang, *case) for case in POSITIVE[lang]]
        neg = [(lang, *case) for case in NEGATIVE[lang]]
        pos_ok = _check(
            rows = pos,
            expect = True,
            label = "正例",
            failures = failures,
        )
        neg_ok = _check(
            rows = neg,
            expect = False,
            label = "反例",
            failures = failures,
        )
        rows.append((lang, len(pos), pos_ok, len(neg), neg_ok))
    return rows, failures


def evaluate_ext() -> tuple:
    rows = []
    failures = []
    reading = ja_reading.available()
    for group in EXT_POSITIVE:
        if group == "ja_reading" and not reading:
            continue
        pos, neg = EXT_POSITIVE[group], EXT_NEGATIVE[group]
        pos_ok = _check(
            rows = pos,
            expect = True,
            label = "正例",
            failures = failures,
        )
        neg_ok = _check(
            rows = neg,
            expect = False,
            label = "反例",
            failures = failures,
        )
        rows.append((group, len(pos), pos_ok, len(neg), neg_ok))
    return rows, failures


def _table(rows: list, head: str) -> None:
    print(f"| {head} | 正例 | 正例通过 | 反例 | 反例拒绝（误收） |")
    print("| --- | --- | --- | --- | --- |")
    for name, n_pos, pos_ok, n_neg, neg_ok in rows:
        print(f"| {name} | {n_pos} | {pos_ok} | {n_neg} | {neg_ok}（{n_neg - neg_ok}） |")
    false_accepts = sum(n_neg - neg_ok for _, _, _, n_neg, neg_ok in rows)
    print(f"\n正例 {sum(r[1] for r in rows)} 组，反例 {sum(r[3] for r in rows)} 组，误收 {false_accepts} 组。\n")


def main(argv: list) -> int:
    rows, failures = evaluate()
    _table(rows = rows, head = "语言")
    ext_rows, ext_failures = evaluate_ext()
    _table(rows = ext_rows, head = "增强项")
    if not ja_reading.available():
        print("没有安装 fugashi 和 unidic-lite，跳过 ja_reading 组。\n")
    failures += ext_failures
    if "-v" in argv:
        for row in failures:
            print(" | ".join(str(x) for x in row))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(argv = sys.argv[1:]))
