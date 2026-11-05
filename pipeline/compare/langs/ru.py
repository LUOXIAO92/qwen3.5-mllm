"""俄语数词：基数、序数、集合数词的全部格、性、数形式（ё 已统一为 е）。

小数读法「три целых пять десятых」在 post 中改写为 3、小数点、5。
"""

from ..numerals import (
    ABBR_FOLD, CARD_ONLY, CARD_OR_ORD, ORD_ONLY, LangSpec, add, big, big_lex, ordinal, teen10, with_fold, zero,
)
from ..parse import Mark, Num, Text

CARDINALS = {
    1: "один одна одно одни одного одной одному одним одном одну одною одними одних",
    2: "два две двух двум двумя",
    3: "три трех трем тремя",
    4: "четыре четырех четырем четырьмя",
    5: "пять пяти пятью",
    6: "шесть шести шестью",
    7: "семь семи семью",
    8: "восемь восьми восемью восьмью",
    9: "девять девяти девятью",
    40: "сорок сорока",
    50: "пятьдесят пятидесяти пятьюдесятью",
    60: "шестьдесят шестидесяти шестьюдесятью",
    70: "семьдесят семидесяти семьюдесятью",
    80: "восемьдесят восьмидесяти восемьюдесятью восьмьюдесятью",
    90: "девяносто девяноста",
    100: "сто ста",
    200: "двести двухсот двумстам двумястами двухстах",
    300: "триста трехсот тремстам тремястами трехстах",
    400: "четыреста четырехсот четыремстам четырьмястами четырехстах",
    500: "пятьсот пятисот пятистам пятьюстами пятистах",
    600: "шестьсот шестисот шестистам шестьюстами шестистах",
    700: "семьсот семисот семистам семьюстами семистах",
    800: "восемьсот восьмисот восьмистам восемьюстами восьмьюстами восьмистах",
    900: "девятьсот девятисот девятистам девятьюстами девятистах",
}
# 以 -ать 结尾、按「-ь / -и / -ью」变格的数词
SOFT = {
    10: "десять", 11: "одиннадцать", 12: "двенадцать", 13: "тринадцать", 14: "четырнадцать",
    15: "пятнадцать", 16: "шестнадцать", 17: "семнадцать", 18: "восемнадцать",
    19: "девятнадцать", 20: "двадцать", 30: "тридцать",
}
# 集合数词：двое、трое 单列，其余按 -о / -ых / -ым / -ыми 变格
COLLECTIVE = {
    2: "двое двоих двоим двоими",
    3: "трое троих троим троими",
}
COLLECTIVE_STEMS = {
    4: "четвер", 5: "пятер", 6: "шестер", 7: "семер", 8: "восьмер", 9: "девятер", 10: "десятер",
}
NOUN_ENDINGS = ["", "а", "у", "ом", "е", "ы", "ов", "ам", "ами", "ах"]
BIGS = {10 ** 6: "миллион", 10 ** 9: "миллиард", 10 ** 12: "триллион"}
THOUSAND = "тысяча тысячи тысяче тысячу тысячей тысячею тысячью тысяч тысячам тысячами тысячах"

ORD_STEMS = {
    1: "перв", 2: "втор", 4: "четверт", 5: "пят", 6: "шест", 7: "седьм", 8: "восьм", 9: "девят",
    10: "десят", 11: "одиннадцат", 12: "двенадцат", 13: "тринадцат", 14: "четырнадцат",
    15: "пятнадцат", 16: "шестнадцат", 17: "семнадцат", 18: "восемнадцат", 19: "девятнадцат",
    20: "двадцат", 30: "тридцат", 40: "сороков", 50: "пятидесят", 60: "шестидесят",
    70: "семидесят", 80: "восьмидесят", 90: "девяност",
    100: "сот", 200: "двухсот", 300: "трехсот", 400: "четырехсот", 500: "пятисот",
    600: "шестисот", 700: "семисот", 800: "восьмисот", 900: "девятисот",
}
# двухтысячный（第两千）这类把系数写进词里的序数
THOUSAND_ORD_PREFIX = {
    2: "двух", 3: "трех", 4: "четырех", 5: "пяти", 6: "шести", 7: "семи", 8: "восьми", 9: "девяти",
    10: "десяти", 20: "двадцати", 100: "сто",
}
HARD = ["ый", "ой", "ого", "ому", "ым", "ом", "ая", "ую", "ою", "ое", "ые", "ых", "ыми"]
SOFT_TRET = ["ий", "ья", "ье", "ьи", "ьего", "ьему", "ьим", "ьем", "ьей", "ью", "ьих", "ьими"]


def _hard(stem: str) -> list:
    return [stem + e for e in HARD]


def build() -> LangSpec:
    spec = LangSpec(
        code = "ru",
        dec_comma = True,
        space_groups = True,
        digits_after_big = True,
        post = _post,
    )
    spec.add(words = "ноль нуль ноля нуля нолю нулю нолем нулем ноле нуле".split(), morphs = zero())
    for v, forms in CARDINALS.items():
        spec.add(words = forms.split(), morphs = add(value = v))
    for v, word in SOFT.items():
        forms = [word, word[:-1] + "и", word[:-1] + "ью"]
        spec.add(words = forms, morphs = teen10() if v == 10 else add(value = v))
    for v, forms in COLLECTIVE.items():
        spec.add(words = forms.split(), morphs = add(value = v))
    for v, stem in COLLECTIVE_STEMS.items():
        forms = [stem + e for e in ("о", "ых", "ым", "ыми")]
        spec.add(words = forms, morphs = teen10() if v == 10 else add(value = v))
    spec.add(words = THOUSAND.split(), morphs = big(value = 1000))
    for v, stem in BIGS.items():
        spec.add(words = [stem + e for e in NOUN_ENDINGS], morphs = big(value = v))

    for v, stem in ORD_STEMS.items():
        m = teen10() if v == 10 else add(value = v)
        spec.add(words = _hard(stem = stem), morphs = ordinal(morph = m))
    spec.add(words = ["трет" + e for e in SOFT_TRET], morphs = ordinal(morph = add(value = 3)))
    spec.add(words = _hard(stem = "нулев"), morphs = ordinal(morph = zero()))
    spec.add(words = _hard(stem = "тысячн"), morphs = ordinal(morph = big(value = 1000)))
    for coef, prefix in THOUSAND_ORD_PREFIX.items():
        spec.add(
            words = _hard(stem = prefix + "тысячн"),
            morphs = ordinal(morph = big_lex(coef = coef, value = 1000)),
        )
    for v, stem in BIGS.items():
        spec.add(words = _hard(stem = stem + "н"), morphs = ordinal(morph = big(value = v)))

    spec.add_sep(words = ["запятая", "запятой"])
    spec.add_pct(words = ["процент" + e for e in NOUN_ENDINGS])
    spec.add_minus(words = "минус")
    # 3-х、5-ти 既可能是基数（трех、пяти）也可能是序数；其余后缀表示序数
    spec.add_ordinal_suffix(pattern = r"-(?:ти|ми|х|и)(?![^\W\d_])", kinds = CARD_OR_ORD)
    ord_endings = "ый|ий|ой|ая|ое|ые|ого|ему|ому|ым|им|ом|ем|ую|ых|их|ыми|ими|ей|ья|ье|ьи|го|му|й|я|е|ю|м"
    spec.add_ordinal_suffix(pattern = r"-(?:" + ord_endings + r")(?![^\W\d_])", kinds = ORD_ONLY)
    # 数量缩写：12 тыс.、5 млн、3 млрд
    for word, value in (("тыс", 1000), ("млн", 10 ** 6), ("млрд", 10 ** 9)):
        spec.add(words = word, morphs = with_fold(morph = big(value = value, implicit_one = False), fold = ABBR_FOLD + word))
    return spec


# ---- 小数读法 ----

_WHOLE = {"целых", "целая", "целой", "целую", "целыми", "целым"}
_DENOM = {10: 1, 100: 2, 1000: 3, 10 ** 4: 4, 10 ** 5: 5, 10 ** 6: 6}


def _frac_from(items: list, m: int):
    """识别「пять десятых」「двадцать пять сотых」「сто двадцать пять тысячных」。"""
    a = items[m] if m < len(items) else None
    b = items[m + 1] if m + 1 < len(items) else None
    if isinstance(a, Num) and a.origin == "word" and "ord" in a.kinds:
        # 整个分数被组合成一个序数：пять тысячных → 5000 的序数形式
        for denom, width in sorted(_DENOM.items(), reverse = True):
            if a.value % denom == 0 and 0 < a.value // denom < denom:
                return a.value // denom, width, a.surface, a.display, m + 1
        if a.value in _DENOM:
            return 1, _DENOM[a.value], a.surface, a.display, m + 1
        return None
    if (
        isinstance(a, Num) and a.origin == "word" and "card" in a.kinds
        and isinstance(b, Num) and b.origin == "word" and "ord" in b.kinds and b.value in _DENOM
    ):
        width = _DENOM[b.value]
        if len(str(a.value)) <= width:
            return a.value, width, a.surface + b.surface, a.display + " " + b.display, m + 2
    return None


def _post(items: list) -> list:
    out = []
    k = 0
    while k < len(items):
        it = items[k]
        nxt = items[k + 1] if k + 1 < len(items) else None
        if (
            isinstance(it, Num) and it.origin == "word" and not it.frac and "card" in it.kinds
            and isinstance(nxt, Text) and nxt.text in _WHOLE
        ):
            m = k + 2
            conj = ""
            if m < len(items) and isinstance(items[m], Text) and items[m].text == "и":
                conj = "и"
                m += 1
            frac = _frac_from(items = items, m = m)
            if frac is not None:
                value, width, surface, display, end = frac
                out.append(it)
                out.append(Mark(kind = "sep", origin = "word", surface = nxt.text + conj))
                out.append(Num(
                    value = value,
                    digits = str(value).zfill(width),
                    origin = "word",
                    kinds = CARD_ONLY,
                    frac = True,
                    surface = surface,
                    display = display,
                ))
                k = end
                continue
        out.append(it)
        k += 1
    return out
