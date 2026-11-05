"""德语数词。复合词（zweitausendsechsundzwanzig）按词表部件切分；序数带全部词尾。"""

from ..numerals import (
    ABBR_FOLD, CARD_OR_ORD, LangSpec, add, big, conn, ordinal, place, teen10, with_fold, zero,
)

UNITS = {
    1: ["eins", "ein", "eine", "einen", "einem", "einer", "eines"],
    2: ["zwei", "zwo"],
    3: ["drei"],
    4: ["vier"],
    5: ["fünf"],
    6: ["sechs"],
    7: ["sieben"],
    8: ["acht"],
    9: ["neun"],
}
TEENS = {
    11: "elf", 12: "zwölf", 13: "dreizehn", 14: "vierzehn", 15: "fünfzehn",
    16: "sechzehn", 17: "siebzehn", 18: "achtzehn", 19: "neunzehn",
}
TENS = {
    20: "zwanzig", 30: "dreißig", 40: "vierzig", 50: "fünfzig",
    60: "sechzig", 70: "siebzig", 80: "achtzig", 90: "neunzig",
}
BIGS = [
    (["tausend"], 1000),
    (["million", "millionen"], 10 ** 6),
    (["milliarde", "milliarden"], 10 ** 9),
    (["billion", "billionen"], 10 ** 12),
]
# 序数词干（不含词尾）
ORD_STEMS = {
    1: ["erst"], 2: ["zweit"], 3: ["dritt"], 4: ["viert"], 5: ["fünft"], 6: ["sechst"],
    7: ["siebt", "siebent"], 8: ["acht"], 9: ["neunt"],
}
ENDINGS = ["e", "en", "er", "em", "es"]


def _ord_forms(stems: list) -> list:
    return [s + e for s in stems for e in ENDINGS]


def build() -> LangSpec:
    spec = LangSpec(
        code = "de",
        compound = True,
        dec_comma = True,
        unit_before_tens = True,
        split_words_need_scale = True,
        big_compound = True,
    )
    spec.add(words = "null", morphs = zero())
    for v, words in UNITS.items():
        spec.add(words = words, morphs = add(value = v))
    spec.add(words = "zehn", morphs = teen10())
    for v, w in TEENS.items():
        spec.add(words = w, morphs = add(value = v))
    for v, w in TENS.items():
        spec.add(words = w, morphs = add(value = v))
    spec.add(words = "hundert", morphs = place(exp = 2, coef_max = 99))
    for words, v in BIGS:
        spec.add(words = words, morphs = big(value = v))
    # und：个位接十位（einundzwanzig），或位值词之后接下去（hundertundeins）
    spec.add(words = "und", morphs = conn(after = {"unit"}, before = {"tens"}))
    spec.add(words = "und", morphs = conn(after = {"place", "big"}, before = {"unit", "teen", "tens"}))

    for v, stems in ORD_STEMS.items():
        spec.add(words = _ord_forms(stems = stems), morphs = ordinal(morph = add(value = v)))
    spec.add(words = _ord_forms(stems = ["zehnt"]), morphs = ordinal(morph = teen10()))
    for v, w in TEENS.items():
        spec.add(words = _ord_forms(stems = [w + "t"]), morphs = ordinal(morph = add(value = v)))
    for v, w in TENS.items():
        spec.add(words = _ord_forms(stems = [w + "st"]), morphs = ordinal(morph = add(value = v)))
    spec.add(words = _ord_forms(stems = ["hundertst"]), morphs = ordinal(morph = place(exp = 2, coef_max = 99)))
    spec.add(words = _ord_forms(stems = ["tausendst"]), morphs = ordinal(morph = big(value = 1000)))
    spec.add(words = _ord_forms(stems = ["millionst", "millionenst"]), morphs = ordinal(morph = big(value = 10 ** 6)))
    spec.add(
        words = _ord_forms(stems = ["milliardst", "milliardest", "milliardenst"]),
        morphs = ordinal(morph = big(value = 10 ** 9)),
    )

    spec.add_sep(words = "komma")
    spec.add_pct(words = "prozent")
    spec.add_minus(words = "minus")
    # 「am 3. Mai」：数字后的句点可能是序数标记，也可能是句号
    spec.add_ordinal_suffix(pattern = r"\.(?= +[^\W\d_])", kinds = CARD_OR_ORD)
    # 数量缩写：5 Mio.、3 Mrd.、12 Tsd.
    for word, value in (("mio", 10 ** 6), ("mrd", 10 ** 9), ("tsd", 1000)):
        spec.add(words = word, morphs = with_fold(morph = big(value = value, implicit_one = False), fold = ABBR_FOLD + word))
    return spec
