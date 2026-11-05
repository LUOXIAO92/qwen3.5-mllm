"""土耳其语数词。数词后的词尾（beşte、dörde、yedisini、sekizdeki、dokuzuncusu）拆成独立文字，
与「5'te」「4'e」「7'sini」「8'deki」「9'uncusu」对齐。"""

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
_V4 = ("ı", "i", "u", "ü")


def _suffix_chains() -> frozenset:
    """数词后常见的词尾串：[复数][领属][格][-ki][系词]，以及 -lik、-li、分配数词 -er 加格。

    不检查元音和谐；词尾只是从数词上拆下来的文字，比较时两边的词尾仍要逐字相同。
    """
    poss = [""] + list(_V4) + ["s" + v for v in _V4] + ["ları", "leri"]
    bare_cases = (
        ["", "a", "e", "ya", "ye", "da", "de", "ta", "te", "dan", "den", "tan", "ten", "la", "le", "yla", "yle"]
        + list(_V4) + ["y" + v for v in _V4] + [v + "n" for v in _V4] + ["n" + v + "n" for v in _V4]
    )
    poss_cases = ["", "na", "ne", "nda", "nde", "ndan", "nden", "yla", "yle"] + ["n" + v for v in _V4] + ["n" + v + "n" for v in _V4]
    copula = [""] + [c + v + t for c in ("d", "t", "yd") for v in _V4 for t in ("r", "")] + ["ken", "yken"]
    chains = set()
    for plural in ("", "lar", "ler"):
        for p in poss:
            for case in bare_cases if p == "" else poss_cases:
                ki = ("", "ki") if case.endswith(("da", "de", "ta", "te", "n")) and case else ("",)
                for k in ki:
                    for cop in copula:
                        chains.add(plural + p + case + k + cop)
    for head in ("lık", "lik", "luk", "lük", "lı", "li", "lu", "lü", "ar", "er", "şar", "şer"):
        chains.add(head)
        for case in ("a", "e", "da", "de", "ta", "te", "dan", "den", "tan", "ten", "la", "le", "dır", "dir", "tır", "tir"):
            chains.add(head + case)
    for soft in ("lığ", "liğ", "luğ", "lüğ"):
        for tail in ("ı", "i", "u", "ü", "a", "e", "ın", "in", "un", "ün", "ını", "ini", "ına", "ine", "ında", "inde"):
            chains.add(soft + tail)
    chains.discard("")
    return frozenset(c for c in chains if len(c) <= 10)


# 数词后常见的词尾（格、复数、领属、系词、-lik、分配数词），评测打分和比对都按这些切分
SUFFIXES = {
    "a", "e", "ya", "ye", "ı", "i", "u", "ü", "yı", "yi", "yu", "yü",
    "da", "de", "ta", "te", "dan", "den", "tan", "ten",
    "ın", "in", "un", "ün", "nın", "nin", "nun", "nün",
    "la", "le", "yla", "yle", "lar", "ler", "ları", "leri", "lara", "lere", "larda", "lerde",
    "lardan", "lerden", "ların", "lerin",
    "sı", "si", "su", "sü", "dır", "dir", "dur", "dür", "tır", "tir", "tur", "tür",
    "lık", "lik", "luk", "lük", "lı", "li", "lu", "lü", "er", "ar", "şer", "şar",
}
# 比对时另外切分的词尾串（数词、单位词后），评测打分不切，免得 birliği、altında 这类词被拆开
SUFFIX_CHAINS = _suffix_chains() | SUFFIXES
_ORDINAL_ENDINGS = "inci|ıncı|uncu|üncü|nci|ncı|ncu|ncü"


def _ordinal_form(word: str) -> str:
    """按元音和谐加序数后缀：bir → birinci，dört → dördüncü。"""
    last = [c for c in word if c in _VOWELS][-1]
    v = _HARMONY[last]
    if word[-1] in _VOWELS:
        return word + "nc" + v
    stem = _SOFTEN.get(word, word)
    return stem + v + "nc" + v


def _affixes(word: str, spec: LangSpec, ext: bool = False):
    suffixes = SUFFIX_CHAINS if ext else SUFFIXES
    for k in range(2, len(word)):
        stem, suffix = word[:k], word[k:]
        if suffix in suffixes and stem in spec.pieces:
            yield "", stem, suffix, False


def build() -> LangSpec:
    spec = LangSpec(
        code = "tr",
        dec_comma = True,
        pct_symbol_prefix = True,
        big_compound = True,
        digits_after_big = True,
        split_affixes = _affixes,
        suffixes = SUFFIX_CHAINS,
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
    spec.add_ordinal_suffix(pattern = r"(?:" + _ORDINAL_ENDINGS + r")(?![^\W\d_])", kinds = ORD_ONLY)
    # 比对时：数字与序数后缀之间可以有空格（3 üncü、50 nci），序数后缀后面还可以有词尾（99'uncusu、3'üncüde），
    # 序数后缀算进数，其余词尾成为文字
    spec.add_ordinal_suffix(pattern = r" (?:" + _ORDINAL_ENDINGS + r")(?![^\W\d_])", kinds = ORD_ONLY, ext_only = True)
    chain = "|".join(sorted(SUFFIX_CHAINS, key = len, reverse = True))
    spec.add_ordinal_suffix(
        pattern = r" ?(?:" + _ORDINAL_ENDINGS + r")(?=(?:" + chain + r")(?![^\W\d_]))",
        kinds = ORD_ONLY,
        ext_only = True,
    )
    return spec
