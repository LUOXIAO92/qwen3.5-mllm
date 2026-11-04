"""土耳其语数词。数词后的格词尾（beşte、dörde）拆成独立文字，与「5'te」「4'e」对齐。"""

from ..numerals import (
    CARD_OR_ORD, ORD_ONLY, LangSpec, add, big, ordinal, place, zero,
)

CARDINALS = {
    1: "bir", 2: "iki", 3: "üç", 4: "dört", 5: "beş", 6: "altı", 7: "yedi", 8: "sekiz", 9: "dokuz",
    10: "on", 20: "yirmi", 30: "otuz", 40: "kırk", 50: "elli", 60: "altmış", 70: "yetmiş",
    80: "seksen", 90: "doksan",
}
BIGS = {1000: "bin", 10 ** 6: "milyon", 10 ** 9: "milyar", 10 ** 12: "trilyon"}
_VOWELS = "aeıioöuü"
_HARMONY = {"a": "ı", "ı": "ı", "e": "i", "i": "i", "o": "u", "u": "u", "ö": "ü", "ü": "ü"}
# 元音开头的词尾前，词末清辅音浊化：dört → dörd-
_SOFTEN = {"dört": "dörd"}
# 数词后常见的词尾（格、复数、领属、系词、-lik、分配数词）
SUFFIXES = {
    "a", "e", "ya", "ye", "ı", "i", "u", "ü", "yı", "yi", "yu", "yü",
    "da", "de", "ta", "te", "dan", "den", "tan", "ten",
    "ın", "in", "un", "ün", "nın", "nin", "nun", "nün",
    "la", "le", "yla", "yle", "lar", "ler", "ları", "leri", "lara", "lere", "larda", "lerde",
    "lardan", "lerden", "ların", "lerin",
    "sı", "si", "su", "sü", "dır", "dir", "dur", "dür", "tır", "tir", "tur", "tür",
    "lık", "lik", "luk", "lük", "lı", "li", "lu", "lü", "er", "ar", "şer", "şar",
}


def _ordinal_form(word: str) -> str:
    """按元音和谐加序数后缀：bir → birinci，dört → dördüncü。"""
    last = [c for c in word if c in _VOWELS][-1]
    v = _HARMONY[last]
    if word[-1] in _VOWELS:
        return word + "nc" + v
    stem = _SOFTEN.get(word, word)
    return stem + v + "nc" + v


def _affixes(word: str, spec: LangSpec):
    for k in range(2, len(word)):
        stem, suffix = word[:k], word[k:]
        if suffix in SUFFIXES and stem in spec.pieces:
            yield "", stem, suffix, False


def build() -> LangSpec:
    spec = LangSpec(
        code = "tr",
        dec_comma = True,
        pct_symbol_prefix = True,
        split_affixes = _affixes,
    )
    spec.add(words = "sıfır", morphs = zero())
    for v, w in CARDINALS.items():
        spec.add(words = w, morphs = add(value = v))
        spec.add(words = _ordinal_form(word = w), morphs = ordinal(morph = add(value = v)))
    spec.add(words = "dörd", morphs = add(value = 4))
    spec.add(words = "yüz", morphs = place(exp = 2))
    spec.add(words = _ordinal_form(word = "yüz"), morphs = ordinal(morph = place(exp = 2)))
    for v, w in BIGS.items():
        spec.add(words = w, morphs = big(value = v))
        spec.add(words = _ordinal_form(word = w), morphs = ordinal(morph = big(value = v)))
    spec.add_sep(words = "virgül")
    spec.add_pct(words = "yüzde", position = "pre")
    spec.add_minus(words = "eksi")
    spec.add_ordinal_suffix(pattern = r"\.(?= +[^\W\d_])", kinds = CARD_OR_ORD)
    spec.add_ordinal_suffix(pattern = r"(?:inci|ıncı|uncu|üncü|nci|ncı|ncu|ncü)(?![^\W\d_])", kinds = ORD_ONLY)
    return spec
