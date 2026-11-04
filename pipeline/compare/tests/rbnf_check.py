"""开发用的词表回归检查：用 CLDR RBNF 生成各种读法，检查解析结果的数值。

需要 unicode-rbnf（MIT 许可，数据为 Unicode 许可），只在开发环境中安装，不是运行依赖：

    python -m pipeline.compare.tests.rbnf_check [显示条数] [语言代码 ...]

对每种语言、每个规则集（基数各格各性、序数、年份读法），把一批数值的读法交给
解析器，要求得到一个数值记号且值相同；年份读法允许分段（nineteen eighty-four）。
菲律宾语的 RBNF 读法使用旧式拼写（pû’t、带重音符号），不参与检查；日语、中文 RBNF 的
年份读法省略了〇（1903 → 一九三），也不检查。其余未通过的条目是 RBNF 自身的错误，
例如俄语 третьому、восемьдесятое，法语 cent-et-unième，意大利语 unmilasimo。
"""

from __future__ import annotations

import sys

from pipeline.compare.langs import LANGS, get_spec
from pipeline.compare.parse import Num, Text, analyze

# 序数由前缀文字加基数构成的语言：第、ที่、thứ、ke、번째/째 等
PREFIX_ORDINAL = {"zh", "ja", "th", "vi", "id", "ms", "ko", "ar"}
SKIP = {"fil"}
# RBNF 输出中的软连字符（德语）和零宽空格（泰语）
_RBNF_INVISIBLE = dict.fromkeys([0x00AD, 0x200B], None)

NUMBERS = sorted(set(
    list(range(0, 121))
    + list(range(121, 1000, 7))
    + [1000, 1001, 1010, 1100, 1984, 2000, 2001, 2020, 2024, 2026, 2100]
    + list(range(1900, 2031, 3))
    + [10000, 12345, 20000, 99999, 100000, 123456, 999999, 1000000, 2000000, 3500000, 10 ** 9]
))


def check(lang: str, limit: int = 15) -> tuple:
    from unicode_rbnf import FormatPurpose, RbnfEngine

    spec = get_spec(lang = lang)
    engine = RbnfEngine.for_language(lang)
    total = 0
    bad = []
    for n in NUMBERS:
        for purpose in (FormatPurpose.CARDINAL, FormatPurpose.ORDINAL, FormatPurpose.YEAR):
            if lang in ("ja", "zh") and purpose == FormatPurpose.YEAR:
                continue
            try:
                result = engine.format_number(n, purpose = purpose)
            except Exception:
                continue
            for ruleset, text in result.text_by_ruleset.items():
                if "financial" in ruleset or "days" in ruleset:
                    continue
                text = text.translate(_RBNF_INVISIBLE)
                total += 1
                items = analyze(text = text, spec = spec)[0]
                nums = [it for it in items if isinstance(it, Num)]
                others = [it for it in items if not isinstance(it, Num)]
                ok = False
                if len(nums) == 1 and nums[0].value == n:
                    ok = not others or (
                        lang in PREFIX_ORDINAL and purpose == FormatPurpose.ORDINAL
                        and all(isinstance(it, Text) for it in others)
                    )
                elif purpose == FormatPurpose.YEAR and nums and not others:
                    ok = "".join(x.digits for x in nums) == str(n)
                if ok and purpose == FormatPurpose.ORDINAL and lang not in PREFIX_ORDINAL:
                    ok = "ord" in nums[0].kinds
                if not ok:
                    parsed = [(type(it).__name__, getattr(it, "digits", getattr(it, "text", ""))) for it in items]
                    bad.append((n, ruleset, text, parsed))
    for row in bad[:limit]:
        print(lang, *row)
    return total, len(bad)


def main(argv: list) -> None:
    limit = 15
    if argv and argv[0].isdigit():
        limit, argv = int(argv[0]), argv[1:]
    langs = argv or [lang for lang in LANGS if lang not in SKIP]
    for lang in langs:
        total, nbad = check(lang = lang, limit = limit)
        print(f"{lang}: {total} 个读法，{nbad} 个未通过")


if __name__ == "__main__":
    main(argv = sys.argv[1:])
