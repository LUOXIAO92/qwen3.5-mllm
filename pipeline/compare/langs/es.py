"""西班牙语数词，含阴阳性和序数的全部词尾。"""

from ..numerals import (
    ORD_ONLY, LangSpec, add, big, conn, ordinal, teen10, zero,
)

UNITS = {
    1: ["uno", "un", "una"], 2: ["dos"], 3: ["tres"], 4: ["cuatro"], 5: ["cinco"],
    6: ["seis"], 7: ["siete"], 8: ["ocho"], 9: ["nueve"],
}
TEENS = {
    11: "once", 12: "doce", 13: "trece", 14: "catorce", 15: "quince",
    16: "dieciséis", 17: "diecisiete", 18: "dieciocho", 19: "diecinueve",
}
VEINTI = {
    21: ["veintiuno", "veintiún", "veintiuna"], 22: ["veintidós"], 23: ["veintitrés"],
    24: ["veinticuatro"], 25: ["veinticinco"], 26: ["veintiséis"], 27: ["veintisiete"],
    28: ["veintiocho"], 29: ["veintinueve"],
}
TENS = {
    20: "veinte", 30: "treinta", 40: "cuarenta", 50: "cincuenta",
    60: "sesenta", 70: "setenta", 80: "ochenta", 90: "noventa",
}
HUNDREDS = {
    200: "doscient", 300: "trescient", 400: "cuatrocient", 500: "quinient",
    600: "seiscient", 700: "setecient", 800: "ochocient", 900: "novecient",
}
GENDER = ["o", "a", "os", "as"]

ORD_UNITS = {
    1: ["primer"], 2: ["segund"], 3: ["tercer"], 4: ["cuart"], 5: ["quint"], 6: ["sext"],
    7: ["séptim", "sétim"], 8: ["octav"], 9: ["noven", "non"],
}
ORD_TEENS = {
    11: ["undécim", "decimoprimer"], 12: ["duodécim", "decimosegund"], 13: ["decimotercer"],
    14: ["decimocuart"], 15: ["decimoquint"], 16: ["decimosext"], 17: ["decimoséptim"],
    18: ["decimoctav"], 19: ["decimonoven"],
}
ORD_TENS = {
    10: ["décim"], 20: ["vigésim"], 30: ["trigésim"], 40: ["cuadragésim"], 50: ["quincuagésim"],
    60: ["sexagésim"], 70: ["septuagésim", "setuagésim"], 80: ["octogésim"], 90: ["nonagésim"],
}
ORD_HUNDREDS = {
    100: ["centésim"], 200: ["ducentésim"], 300: ["tricentésim", "trecentésim"],
    400: ["cuadringentésim"], 500: ["quingentésim"], 600: ["sexcentésim"],
    700: ["septingentésim"], 800: ["octingentésim", "octingésim"], 900: ["noningentésim"],
}


def _gendered(stems: list) -> list:
    return [s + e for s in stems for e in GENDER]


def build() -> LangSpec:
    spec = LangSpec(
        code = "es",
        dec_comma = True,
        ordinal_all_parts = True,
        big_compound = True,
    )
    spec.add(words = "cero", morphs = zero())
    for v, words in UNITS.items():
        spec.add(words = words, morphs = add(value = v))
    spec.add(words = "diez", morphs = teen10())
    for v, w in TEENS.items():
        spec.add(words = w, morphs = add(value = v))
    for v, words in VEINTI.items():
        spec.add(words = words, morphs = add(value = v))
    for v, w in TENS.items():
        spec.add(words = w, morphs = add(value = v))
    spec.add(words = ["cien", "ciento"], morphs = add(value = 100))
    for v, stem in HUNDREDS.items():
        spec.add(words = [stem + "os", stem + "as"], morphs = add(value = v))
    spec.add(words = "mil", morphs = big(value = 1000))
    spec.add(words = ["millón", "millones"], morphs = big(value = 10 ** 6))
    spec.add(words = ["billón", "billones"], morphs = big(value = 10 ** 12))
    spec.add(words = "y", morphs = conn(after = {"tens"}, before = {"unit"}))

    # 词尾脱落的形式：primer、tercer、decimoprimer、decimotercer
    spec.add(words = "primer", morphs = ordinal(morph = add(value = 1)))
    spec.add(words = "tercer", morphs = ordinal(morph = add(value = 3)))
    spec.add(words = "decimoprimer", morphs = ordinal(morph = add(value = 11)))
    spec.add(words = "decimotercer", morphs = ordinal(morph = add(value = 13)))
    for v, stems in ORD_UNITS.items():
        spec.add(words = _gendered(stems = stems), morphs = ordinal(morph = add(value = v)))
    for v, stems in ORD_TEENS.items():
        spec.add(words = _gendered(stems = stems), morphs = ordinal(morph = add(value = v)))
    for v, stems in ORD_TENS.items():
        # 序数十位只占十位，后面可以再接序数个位（décimo tercero、vigésimo primero）
        spec.add(words = _gendered(stems = stems), morphs = ordinal(morph = add(value = v, places = {1})))
    for v, stems in ORD_HUNDREDS.items():
        spec.add(words = _gendered(stems = stems), morphs = ordinal(morph = add(value = v)))
    spec.add(words = _gendered(stems = ["milésim"]), morphs = ordinal(morph = big(value = 1000)))
    spec.add(words = _gendered(stems = ["millonésim"]), morphs = ordinal(morph = big(value = 10 ** 6)))

    spec.add_sep(words = ["coma", "punto"])
    spec.add_pct(words = "por ciento")
    spec.add_minus(words = "menos")
    spec.add_ordinal_suffix(
        pattern = r"\.?(?:os|as|er|ra|ro|do|da|to|ta|vo|va|mo|ma|no|na|o|a|°)(?![^\W\d_])",
        kinds = ORD_ONLY,
    )
    return spec
