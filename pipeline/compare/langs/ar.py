"""阿拉伯语数词（标准阿拉伯语）：阴阳性、主格与宾属格、双数、序数。

文本已去掉元音符号和 tatweel，并把 أ إ آ → ا、ى → ي、ة → ه 统一（词表同样处理）。
数词前的附着成分（و、ب、ل、ك、ف、ال）拆成文字；数内的 و 作连接词。
"""

from ..numerals import (
    TEEN_SUFFIX, LangSpec, add, big, big_lex, conn, ordinal, place, teen10, zero,
)

UNITS = {
    1: ["واحد", "واحدة", "أحد", "إحدى"],
    2: ["اثنان", "اثنين", "اثنتان", "اثنتين", "اثنا", "اثني", "اثنتا", "اثنتي", "ثنتان", "ثنتين"],
    3: ["ثلاثة", "ثلاث"],
    4: ["أربعة", "أربع"],
    5: ["خمسة", "خمس"],
    6: ["ستة", "ست"],
    7: ["سبعة", "سبع"],
    8: ["ثمانية", "ثماني", "ثمان"],
    9: ["تسعة", "تسع"],
}
TENS = {
    20: ["عشرون", "عشرين"], 30: ["ثلاثون", "ثلاثين"], 40: ["أربعون", "أربعين"],
    50: ["خمسون", "خمسين"], 60: ["ستون", "ستين"], 70: ["سبعون", "سبعين"],
    80: ["ثمانون", "ثمانين"], 90: ["تسعون", "تسعين"],
}
HUNDRED_PREFIX = {
    3: ["ثلاث"], 4: ["أربع"], 5: ["خمس"], 6: ["ست"], 7: ["سبع"], 8: ["ثمان", "ثماني"], 9: ["تسع"],
}
ORDINALS = {
    1: ["أول", "أولى", "حادي", "حادية"],
    2: ["ثاني", "ثانية", "ثان"],
    3: ["ثالث", "ثالثة"],
    4: ["رابع", "رابعة"],
    5: ["خامس", "خامسة"],
    6: ["سادس", "سادسة"],
    7: ["سابع", "سابعة"],
    8: ["ثامن", "ثامنة"],
    9: ["تاسع", "تاسعة"],
}
# 附着在词首的成分：连词 و/ف，介词 ب/ل/ك，冠词 ال（ل + ال 写作 لل）
_PROCLITICS = [
    "وبال", "وال", "بال", "فال", "كال", "لل", "ولل", "ال",
    "وب", "ول", "فب", "فل", "وك", "و", "ف", "ب", "ل", "ك",
]


def _affixes(word: str, spec: LangSpec, ext: bool = False):
    for p in _PROCLITICS:
        if word.startswith(p) and len(word) - len(p) >= 2:
            # 以 و 开头的附着成分在数内作连接词（خمسة وعشرون）
            yield p, word[len(p):], "", p.startswith("و")


def build() -> LangSpec:
    spec = LangSpec(
        code = "ar",
        unit_before_tens = True,
        ordinal_unit_head = True,
        split_affixes = _affixes,
    )
    spec.add(words = "صفر", morphs = zero())
    for v, words in UNITS.items():
        spec.add(words = words, morphs = add(value = v))
    # عشرة / عشر：单独是「十」，跟在个位数后面构成十一至十九
    spec.add(words = ["عشرة", "عشر"], morphs = TEEN_SUFFIX)
    spec.add(words = ["عشرة", "عشر"], morphs = teen10())
    for v, words in TENS.items():
        spec.add(words = words, morphs = add(value = v))
    spec.add(words = ["مائة", "مئة"], morphs = place(exp = 2))
    spec.add(
        words = ["مائتان", "مائتين", "مئتان", "مئتين", "مائتا", "مائتي", "مئتا", "مئتي"],
        morphs = add(value = 200),
    )
    for v, prefixes in HUNDRED_PREFIX.items():
        spec.add(words = [p + h for p in prefixes for h in ("مائة", "مئة")], morphs = add(value = v * 100))
    spec.add(words = "ألف", morphs = big(value = 1000))
    spec.add(words = ["ألفان", "ألفين", "ألفي"], morphs = big_lex(coef = 2, value = 1000))
    spec.add(words = ["آلاف", "ألوف"], morphs = big(value = 1000, implicit_one = False))
    spec.add(words = "مليون", morphs = big(value = 10 ** 6))
    spec.add(words = ["مليونان", "مليونين", "مليوني"], morphs = big_lex(coef = 2, value = 10 ** 6))
    spec.add(words = "ملايين", morphs = big(value = 10 ** 6, implicit_one = False))
    spec.add(words = ["مليار", "بليون"], morphs = big(value = 10 ** 9))
    spec.add(words = ["ملياران", "مليارين", "ملياري"], morphs = big_lex(coef = 2, value = 10 ** 9))
    spec.add(words = "مليارات", morphs = big(value = 10 ** 9, implicit_one = False))
    spec.add(words = "و", morphs = conn())
    for v, words in ORDINALS.items():
        spec.add(words = words, morphs = ordinal(morph = add(value = v)))
    spec.add(words = ["عاشر", "عاشرة"], morphs = ordinal(morph = teen10()))
    spec.add_sep(words = ["فاصلة", "فاصل"])
    spec.add_pct(words = ["بالمائة", "بالمئة", "في المائة", "في المئة", "بالمية"])
    spec.add_minus(words = "سالب")
    return spec
