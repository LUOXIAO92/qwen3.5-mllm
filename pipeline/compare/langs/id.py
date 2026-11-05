"""印尼语数词。se- 前缀（sepuluh、seratus、seribu）按部件切分；ke- 序数前缀拆成文字，与「ke-2」对齐。"""

from ..numerals import (
    TEEN_SUFFIX, LangSpec, add, big, ordinal, place, with_fold, zero,
)

UNITS_ID = ["satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan"]
BIGS_ID = [("ribu", 1000), ("juta", 10 ** 6), ("miliar", 10 ** 9), ("triliun", 10 ** 12)]


def fill_malay_family(spec: LangSpec, units: list, zeros: list, bigs: list) -> None:
    """印尼语与马来语共用的构词：belas、puluh、ratus、ribu、se- 前缀。"""
    spec.add(words = zeros, morphs = zero())
    for v, w in enumerate(units, start = 1):
        spec.add(words = w, morphs = add(value = v))
    # se-（一）只出现在 sepuluh、sebelas、seratus、seribu 这类词内，后面必须接位值词或 belas
    se = add(value = 1, needs = frozenset({"place", "big", "teen_suffix"}))
    spec.add(words = "se", morphs = se, bound = True)
    spec.add(words = "belas", morphs = TEEN_SUFFIX)
    spec.add(words = "puluh", morphs = place(exp = 1, implicit_one = False))
    spec.add(words = "ratus", morphs = place(exp = 2, implicit_one = False))
    for w, v in bigs:
        spec.add(words = w, morphs = big(value = v, implicit_one = False))
    # pertama（第一）只单独使用，「dua ratus pertama」是「前 200」，不是第 201
    spec.add(words = "pertama", morphs = ordinal(morph = add(value = 1, alone = True)))


# 年代后缀 -an 只接在这些词后：tujuh puluhan（70 年代）、sepuluhan、seribu sembilan ratus sembilan puluhan
_DECADE_STEMS = ("puluh", "sepuluh", "belas", "ratus", "ribu")


def malay_affixes(word: str, spec: LangSpec, ext: bool = False):
    """ke- 序数前缀、-nya 后缀与年代后缀 -an：kedua、ketiganya、puluhan（与 1970-an 对应）。"""
    suffix = ""
    stem = word
    if stem.endswith("nya") and len(stem) > 5:
        stem, suffix = stem[:-3], "nya"
        yield "", stem, suffix, False
    if stem.startswith("ke") and len(stem) > 4:
        yield "ke", stem[2:], suffix, False
    if word.endswith("an") and word[:-2].endswith(_DECADE_STEMS):
        yield "", word[:-2], "an", False


def build() -> LangSpec:
    spec = LangSpec(
        code = "id",
        compound = True,
        dec_comma = True,
        big_compound = True,
        one_ten_closed = True,
        split_affixes = malay_affixes,
    )
    fill_malay_family(
        spec = spec,
        units = UNITS_ID,
        zeros = ["nol", "kosong"],
        bigs = BIGS_ID,
    )
    # milyar 是 miliar 的旧拼法
    milyar = with_fold(morph = big(value = 10 ** 9, implicit_one = False), fold = "miliar")
    spec.add(words = "milyar", morphs = milyar)
    spec.add_sep(words = "koma")
    spec.add_pct(words = "persen")
    spec.add_minus(words = ["minus", "negatif"])
    return spec
