"""马来语数词（马来西亚习惯：小数点为「.」，千分位为「,」）。"""

from ..numerals import LangSpec
from .id import fill_malay_family, malay_affixes

UNITS_MS = ["satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "lapan", "sembilan"]
BIGS_MS = [("ribu", 1000), ("juta", 10 ** 6), ("bilion", 10 ** 9), ("trilion", 10 ** 12)]


def build() -> LangSpec:
    spec = LangSpec(
        code = "ms",
        compound = True,
        big_compound = True,
        one_ten_closed = True,
        split_affixes = malay_affixes,
    )
    fill_malay_family(
        spec = spec,
        units = UNITS_MS,
        zeros = ["kosong", "sifar"],
        bigs = BIGS_MS,
    )
    spec.add_sep(words = ["perpuluhan", "titik"])
    spec.add_pct(words = ["peratus", "persen"])
    spec.add_minus(words = ["negatif", "minus"])
    return spec
