"""日语数词（汉字数字；文本中的平假名已统一为片假名）。"""

from ..numerals import LangSpec, add, big, place, with_fold, zero

UNITS = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九"}
PLACES = {"十": 1, "百": 2, "千": 3}
BIGS = {"万": 10 ** 4, "億": 10 ** 8, "兆": 10 ** 12}


def build() -> LangSpec:
    spec = LangSpec(code = "ja", scan = True, digits_after_big = True)
    spec.add(words = "〇", morphs = zero())
    spec.add(words = "零", morphs = with_fold(morph = zero(), fold = "〇"))
    for v, w in UNITS.items():
        spec.add(words = w, morphs = add(value = v))
    for w, exp in PLACES.items():
        spec.add(words = w, morphs = place(exp = exp))
    for w, v in BIGS.items():
        spec.add(words = w, morphs = big(value = v))
    spec.add_sep(words = "点")
    spec.add_pct(words = "パーセント")
    spec.add_minus(words = "マイナス")
    return spec
