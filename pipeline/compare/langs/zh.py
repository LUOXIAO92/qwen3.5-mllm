"""中文数词（文本已转为简体）。两/二、幺/一、〇/零、大写数字在比较时合并。"""

from ..numerals import LangSpec, add, big, gap, place, with_fold, zero

UNITS = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九"}
FINANCIAL = {1: "壹", 2: "贰", 3: "叁", 4: "肆", 5: "伍", 6: "陆", 7: "柒", 8: "捌", 9: "玖"}
PLACES = [("十", "拾", 1), ("百", "佰", 2), ("千", "仟", 3)]


def build() -> LangSpec:
    spec = LangSpec(
        code = "zh",
        scan = True,
        big_compound = True,
        digits_after_big = True,
        pct_symbol_prefix = True,
    )
    # 零：单独读作 0，在数内表示空位（一百零五）
    spec.add(words = "零", morphs = zero())
    spec.add(words = "零", morphs = gap())
    spec.add(words = "〇", morphs = with_fold(morph = zero(), fold = "零"))
    spec.add(words = "〇", morphs = with_fold(morph = gap(), fold = "零"))
    for v, w in UNITS.items():
        spec.add(words = w, morphs = add(value = v))
    for v, w in FINANCIAL.items():
        spec.add(words = w, morphs = with_fold(morph = add(value = v), fold = UNITS[v]))
    spec.add(words = "两", morphs = with_fold(morph = add(value = 2), fold = "二"))
    spec.add(words = "幺", morphs = with_fold(morph = add(value = 1), fold = "一"))
    spec.add(words = "廿", morphs = add(value = 20))
    spec.add(words = "卅", morphs = add(value = 30))
    for w, financial, exp in PLACES:
        spec.add(words = w, morphs = place(exp = exp))
        spec.add(words = financial, morphs = with_fold(morph = place(exp = exp), fold = w))
    spec.add(words = "万", morphs = big(value = 10 ** 4))
    spec.add(words = "亿", morphs = big(value = 10 ** 8))
    spec.add_sep(words = "点")
    spec.add_pct(words = "百分之", position = "pre")
    spec.add_minus(words = "负")
    return spec
