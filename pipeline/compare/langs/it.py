"""意大利语数词。复合词（duemilaventisei、millenovecentottantaquattro）按部件切分。"""

from ..numerals import (
    ORD_ONLY, LangSpec, add, big, ordinal, place, teen10, zero,
)

UNITS = {
    1: ["uno", "una", "un"], 2: ["due"], 3: ["tre", "tré"], 4: ["quattro"], 5: ["cinque"],
    6: ["sei"], 7: ["sette"], 8: ["otto"], 9: ["nove"],
}
TEENS = {
    11: "undici", 12: "dodici", 13: "tredici", 14: "quattordici", 15: "quindici",
    16: "sedici", 17: "diciassette", 18: "diciotto", 19: "diciannove",
}
TENS = {
    20: "venti", 30: "trenta", 40: "quaranta", 50: "cinquanta",
    60: "sessanta", 70: "settanta", 80: "ottanta", 90: "novanta",
}
ENDINGS = ["o", "a", "i", "e"]
ORD_UNITS = {
    1: "prim", 2: "second", 3: "terz", 4: "quart", 5: "quint",
    6: "sest", 7: "settim", 8: "ottav", 9: "non",
}
# 复合序数里个位的词干：ventunesimo、ventiduesimo、ventitreesimo
ORD_UNIT_STEMS = {1: "un", 2: "du", 3: "tre", 4: "quattr", 5: "cinqu", 6: "sei", 7: "sett", 8: "ott", 9: "nov"}


def _esimo(stem: str) -> list:
    return [stem + "esim" + e for e in ENDINGS]


def build() -> LangSpec:
    spec = LangSpec(code = "it", compound = True, dec_comma = True)
    spec.add(words = "zero", morphs = zero())
    for v, words in UNITS.items():
        spec.add(words = words, morphs = add(value = v))
    spec.add(words = "dieci", morphs = teen10())
    for v, w in TEENS.items():
        spec.add(words = w, morphs = add(value = v))
    for v, w in TENS.items():
        spec.add(words = w, morphs = add(value = v))
        # 元音省略的形式只出现在复合词中：ventuno、trentotto
        spec.add(words = w[:-1], morphs = add(value = v), bound = True)
    spec.add(words = "cento", morphs = place(exp = 2))
    spec.add(words = "cent", morphs = place(exp = 2), bound = True)
    spec.add(words = "mille", morphs = big(value = 1000))
    spec.add(words = "mila", morphs = big(value = 1000, implicit_one = False))
    spec.add(words = ["milione", "milioni"], morphs = big(value = 10 ** 6))
    spec.add(words = ["miliardo", "miliardi"], morphs = big(value = 10 ** 9))

    for v, stem in ORD_UNITS.items():
        spec.add(words = [stem + e for e in ENDINGS], morphs = ordinal(morph = add(value = v)))
    spec.add(words = ["decim" + e for e in ENDINGS], morphs = ordinal(morph = teen10()))
    for v, stem in ORD_UNIT_STEMS.items():
        spec.add(words = _esimo(stem = stem), morphs = ordinal(morph = add(value = v, after = "any")))
    for v, w in TEENS.items():
        spec.add(words = _esimo(stem = w[:-1]), morphs = ordinal(morph = add(value = v)))
    for v, w in TENS.items():
        spec.add(words = _esimo(stem = w[:-1]), morphs = ordinal(morph = add(value = v)))
    spec.add(words = _esimo(stem = "cent"), morphs = ordinal(morph = place(exp = 2)))
    spec.add(words = _esimo(stem = "mill"), morphs = ordinal(morph = big(value = 1000)))
    spec.add(words = _esimo(stem = "milion"), morphs = ordinal(morph = big(value = 10 ** 6)))
    spec.add(words = _esimo(stem = "miliard"), morphs = ordinal(morph = big(value = 10 ** 9)))

    spec.add_sep(words = "virgola")
    spec.add_pct(words = ["per cento", "percento"])
    spec.add_minus(words = "meno")
    spec.add_ordinal_suffix(pattern = r"(?:°|o|a)(?![^\W\d_])", kinds = ORD_ONLY)
    return spec
