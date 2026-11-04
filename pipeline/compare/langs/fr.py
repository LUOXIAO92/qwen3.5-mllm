"""法语数词。70–99 的多词读法（soixante-dix、quatre-vingt-dix-sept）整体登记。"""

from ..numerals import (
    ORD_ONLY, LangSpec, add, big, conn, ordinal, place, teen10, zero,
)

UNITS = {
    1: ["un", "une"], 2: ["deux"], 3: ["trois"], 4: ["quatre"], 5: ["cinq"],
    6: ["six"], 7: ["sept"], 8: ["huit"], 9: ["neuf"],
}
TEENS = {11: "onze", 12: "douze", 13: "treize", 14: "quatorze", 15: "quinze", 16: "seize"}
TENS = {
    20: "vingt", 30: "trente", 40: "quarante", 50: "cinquante", 60: "soixante",
    70: "septante", 80: "huitante", 90: "nonante",
}
BIGS = [(["mille", "mil"], 1000), (["million", "millions"], 10 ** 6), (["milliard", "milliards"], 10 ** 9)]
LAST_DIGITS = [(7, "sept"), (8, "huit"), (9, "neuf")]

# 序数：去掉词尾 e，cinq → cinqu，neuf → neuv，再加 ième
_ORD_STEM = {"cinq": "cinqu", "neuf": "neuv"}


def _ord_word(word: str) -> list:
    if word == "un":
        stem = "un"
    else:
        stem = _ORD_STEM.get(word, word[:-1] if word.endswith("e") else word)
    return [stem + "ième", stem + "ièmes"]


def _multiword() -> dict:
    """17–19 与 70–99 的多词读法 → (值, 占用的位)。"""
    out = {}
    for u, w in LAST_DIGITS:
        out["dix " + w] = (10 + u, {1, 0})
    out["soixante dix"] = (70, {1, 0})
    out["soixante et onze"] = (71, {1, 0})
    for v, w in TEENS.items():
        if v != 11:
            out["soixante " + w] = (60 + v, {1, 0})
    for u, w in LAST_DIGITS:
        out["soixante dix " + w] = (70 + u, {1, 0})
    out["quatre vingts"] = (80, {1, 0})
    out["quatre vingt"] = (80, {1})
    out["quatre vingt dix"] = (90, {1, 0})
    for v, w in TEENS.items():
        out["quatre vingt " + w] = (80 + v, {1, 0})
    for u, w in LAST_DIGITS:
        out["quatre vingt dix " + w] = (90 + u, {1, 0})
    return out


def build() -> LangSpec:
    spec = LangSpec(code = "fr", dec_comma = True, space_groups = True)
    spec.add(words = "zéro", morphs = zero())
    for v, words in UNITS.items():
        spec.add(words = words, morphs = add(value = v))
    spec.add(words = "dix", morphs = teen10())
    for v, w in TEENS.items():
        spec.add(words = w, morphs = add(value = v))
    for v, w in TENS.items():
        spec.add(words = w, morphs = add(value = v))
    # dix-neuf cent quatre-vingt-quatre：年份读法中 cent 的系数可以是两位数
    spec.add(words = ["cent", "cents"], morphs = place(exp = 2, coef_max = 99))
    for words, v in BIGS:
        spec.add(words = words, morphs = big(value = v))
    spec.add(words = "et", morphs = conn(after = {"tens"}, before = {"unit"}))
    for text, (v, pl) in _multiword().items():
        spec.add(words = text, morphs = add(value = v, places = pl))
        head, last = text.rsplit(" ", 1)
        last_ords = ["vingtième", "vingtièmes"] if last == "vingts" else _ord_word(word = last)
        spec.add(words = [head + " " + o for o in last_ords], morphs = ordinal(morph = add(value = v, places = {1, 0})))

    spec.add(words = ["premier", "première", "premiers", "premières"], morphs = ordinal(morph = add(value = 1)))
    spec.add(words = ["second", "seconde", "seconds", "secondes"], morphs = ordinal(morph = add(value = 2)))
    # unième 只出现在 vingt et unième 这类复合序数中
    spec.add(words = _ord_word(word = "un"), morphs = ordinal(morph = add(value = 1, after = "any")))
    for v, words in UNITS.items():
        if v > 1:
            spec.add(words = _ord_word(word = words[0]), morphs = ordinal(morph = add(value = v)))
    spec.add(words = _ord_word(word = "dix"), morphs = ordinal(morph = teen10()))
    for v, w in TEENS.items():
        spec.add(words = _ord_word(word = w), morphs = ordinal(morph = add(value = v)))
    for v, w in TENS.items():
        spec.add(words = _ord_word(word = w), morphs = ordinal(morph = add(value = v)))
    spec.add(words = _ord_word(word = "cent"), morphs = ordinal(morph = place(exp = 2)))
    spec.add(words = _ord_word(word = "mille"), morphs = ordinal(morph = big(value = 1000)))
    spec.add(words = _ord_word(word = "million"), morphs = ordinal(morph = big(value = 10 ** 6)))
    spec.add(words = _ord_word(word = "milliard"), morphs = ordinal(morph = big(value = 10 ** 9)))

    spec.add_sep(words = "virgule")
    spec.add_pct(words = "pour cent")
    spec.add_minus(words = "moins")
    spec.add_ordinal_suffix(
        pattern = r"(?:ères|èmes|emes|ers|res|ère|ème|eme|er|re|es|e)(?![^\W\d_])",
        kinds = ORD_ONLY,
    )
    return spec
