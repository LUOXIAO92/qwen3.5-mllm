"""印尼语数词。se- 前缀（sepuluh、seratus、seribu）按部件切分；ke- 序数前缀拆成文字，与「ke-2」对齐。"""

from ..numerals import (
    TEEN_SUFFIX, LangSpec, add, big, ordinal, place, zero,
)

UNITS_ID = ["satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan"]
BIGS_ID = [("ribu", 1000), ("juta", 10 ** 6), ("miliar", 10 ** 9), ("milyar", 10 ** 9), ("triliun", 10 ** 12)]


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
    spec.add(words = "pertama", morphs = ordinal(morph = add(value = 1)))


def malay_affixes(word: str, spec: LangSpec):
    """ke- 序数前缀与 -nya 后缀：kedua、ketiganya。"""
    suffix = ""
    stem = word
    if stem.endswith("nya") and len(stem) > 5:
        stem, suffix = stem[:-3], "nya"
        yield "", stem, suffix, False
    if stem.startswith("ke") and len(stem) > 4:
        yield "ke", stem[2:], suffix, False


def build() -> LangSpec:
    spec = LangSpec(
        code = "id",
        compound = True,
        dec_comma = True,
        split_affixes = malay_affixes,
    )
    fill_malay_family(
        spec = spec,
        units = UNITS_ID,
        zeros = ["nol", "kosong"],
        bigs = BIGS_ID,
    )
    spec.add_sep(words = "koma")
    spec.add_pct(words = "persen")
    spec.add_minus(words = ["minus", "negatif"])
    return spec
