"""葡萄牙语数词（巴西与欧洲拼写），含阴阳性和序数的全部词尾。"""

from ..numerals import (
    ORD_ONLY, LangSpec, add, big, conn, ordinal, teen10, zero,
)

UNITS = {
    1: ["um", "uma"], 2: ["dois", "duas"], 3: ["três"], 4: ["quatro"], 5: ["cinco"],
    6: ["seis"], 7: ["sete"], 8: ["oito"], 9: ["nove"],
}
TEENS = {
    11: ["onze"], 12: ["doze"], 13: ["treze"], 14: ["catorze", "quatorze"], 15: ["quinze"],
    16: ["dezesseis", "dezasseis"], 17: ["dezessete", "dezassete"], 18: ["dezoito"],
    19: ["dezenove", "dezanove"],
}
TENS = {
    20: ["vinte"], 30: ["trinta"], 40: ["quarenta"], 50: ["cinquenta", "cinqüenta"],
    60: ["sessenta"], 70: ["setenta"], 80: ["oitenta"], 90: ["noventa"],
}
HUNDREDS = {
    200: "duzent", 300: "trezent", 400: "quatrocent", 500: "quinhent",
    600: "seiscent", 700: "setecent", 800: "oitocent", 900: "novecent",
}
GENDER = ["o", "a", "os", "as"]
ORD_UNITS = {
    1: ["primeir"], 2: ["segund"], 3: ["terceir"], 4: ["quart"], 5: ["quint"], 6: ["sext"],
    7: ["sétim"], 8: ["oitav"], 9: ["non"],
}
ORD_TENS = {
    10: ["décim"], 20: ["vigésim"], 30: ["trigésim"], 40: ["quadragésim"],
    50: ["quinquagésim", "qüinquagésim"], 60: ["sexagésim"], 70: ["septuagésim", "setuagésim"],
    80: ["octogésim"], 90: ["nonagésim"],
}
ORD_HUNDREDS = {
    100: ["centésim"], 200: ["ducentésim"], 300: ["trecentésim", "tricentésim"],
    400: ["quadringentésim"], 500: ["quingentésim"], 600: ["sexcentésim", "seiscentésim"],
    700: ["septingentésim", "setingentésim"], 800: ["octingentésim"],
    900: ["nongentésim", "noningentésim"],
}


def _gendered(stems: list) -> list:
    return [s + e for s in stems for e in GENDER]


def build() -> LangSpec:
    spec = LangSpec(code = "pt", dec_comma = True, ordinal_all_parts = True)
    spec.add(words = "zero", morphs = zero())
    for v, words in UNITS.items():
        spec.add(words = words, morphs = add(value = v))
    spec.add(words = "dez", morphs = teen10())
    for v, words in TEENS.items():
        spec.add(words = words, morphs = add(value = v))
    for v, words in TENS.items():
        spec.add(words = words, morphs = add(value = v))
    spec.add(words = ["cem", "cento"], morphs = add(value = 100))
    for v, stem in HUNDREDS.items():
        spec.add(words = [stem + "os", stem + "as"], morphs = add(value = v))
    spec.add(words = "mil", morphs = big(value = 1000))
    spec.add(words = ["milhão", "milhões"], morphs = big(value = 10 ** 6))
    spec.add(words = ["bilhão", "bilhões"], morphs = big(value = 10 ** 9))
    spec.add(words = ["bilião", "biliões", "trilhão", "trilhões"], morphs = big(value = 10 ** 12))
    spec.add(words = "e", morphs = conn())

    for v, stems in ORD_UNITS.items():
        spec.add(words = _gendered(stems = stems), morphs = ordinal(morph = add(value = v)))
    for v, stems in ORD_TENS.items():
        spec.add(words = _gendered(stems = stems), morphs = ordinal(morph = add(value = v, places = {1})))
    for v, stems in ORD_HUNDREDS.items():
        spec.add(words = _gendered(stems = stems), morphs = ordinal(morph = add(value = v)))
    spec.add(words = _gendered(stems = ["milésim"]), morphs = ordinal(morph = big(value = 1000)))
    spec.add(words = _gendered(stems = ["milionésim"]), morphs = ordinal(morph = big(value = 10 ** 6)))
    spec.add(words = _gendered(stems = ["bilionésim"]), morphs = ordinal(morph = big(value = 10 ** 9)))

    spec.add_sep(words = "vírgula")
    spec.add_pct(words = "por cento")
    spec.add_minus(words = "menos")
    spec.add_ordinal_suffix(pattern = r"\.?(?:os|as|o|a|°)(?![^\W\d_])", kinds = ORD_ONLY)
    return spec
