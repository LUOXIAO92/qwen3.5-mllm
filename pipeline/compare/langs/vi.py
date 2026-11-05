"""越南语数词。

读法变体在比较时合并：mốt/một、tư/bốn（mươi 之后）、lăm/nhăm/năm（十位之后）、lẻ/linh、
ngàn/nghìn、tỉ/tỷ、bẩy/bảy。năm 紧跟在十位之后时不作「五」（hai mươi năm 是「二十年」），
十位后的五读作 lăm。

「hai trăm năm」可以是 205，也可以是 200 加「năm」（年），读音相同，比对时两种切分都尝试（_splits）。
"""

from ..extend import word_num
from ..numerals import (
    LangSpec, add, big, gap, place, with_fold, zero,
)

_BEFORE_YEAR = {"trăm", "nghìn", "ngàn", "triệu", "tỷ", "tỉ"}


def _splits(num) -> list:
    words = num.display.split(" ")
    if len(words) < 2 or words[-1] != "năm" or words[-2] not in _BEFORE_YEAR:
        return []
    head = word_num(
        value = num.value - 5,
        surface = num.surface[:-len("năm")],
        display = " ".join(words[:-1]),
        kinds = num.kinds,
    )
    year = word_num(
        value = 5,
        surface = "năm",
        display = "năm",
        kinds = num.kinds,
    )
    return [[head, year]]


def build() -> LangSpec:
    spec = LangSpec(code = "vi", dec_comma = True, splitter = _splits)
    spec.add(words = "không", morphs = zero())
    # không trăm：百位为零（hai nghìn không trăm hai mươi sáu）
    spec.add(words = "không trăm", morphs = add(value = 0, places = {2}))
    spec.add(words = "linh", morphs = gap(before = {"unit"}))
    spec.add(words = "lẻ", morphs = with_fold(morph = gap(before = {"unit"}), fold = "linh"))
    spec.add(words = ["một", "nhất"], morphs = add(value = 1))
    spec.add(words = "mốt", morphs = with_fold(morph = add(value = 1), fold = "một", after = "place_tens"))
    spec.add(words = ["hai", "nhì"], morphs = add(value = 2))
    spec.add(words = "ba", morphs = add(value = 3))
    spec.add(words = "bốn", morphs = add(value = 4))
    spec.add(words = "tư", morphs = with_fold(morph = add(value = 4), fold = "bốn", after = "place_tens"))
    spec.add(words = "tư", morphs = add(value = 4))
    spec.add(words = "năm", morphs = add(value = 5, not_after_tens = True))
    spec.add(words = ["lăm", "nhăm"], morphs = with_fold(morph = add(value = 5), fold = "năm", after = "tens"))
    spec.add(words = "sáu", morphs = add(value = 6))
    spec.add(words = "bảy", morphs = add(value = 7))
    spec.add(words = "bẩy", morphs = with_fold(morph = add(value = 7), fold = "bảy"))
    spec.add(words = "tám", morphs = add(value = 8))
    spec.add(words = "chín", morphs = add(value = 9))
    spec.add(words = "mười", morphs = add(value = 10))
    spec.add(words = "mươi", morphs = place(exp = 1, implicit_one = False))
    spec.add(words = "trăm", morphs = place(exp = 2))
    spec.add(words = "nghìn", morphs = big(value = 1000))
    spec.add(words = "ngàn", morphs = with_fold(morph = big(value = 1000), fold = "nghìn"))
    spec.add(words = "vạn", morphs = big(value = 10 ** 4))
    spec.add(words = "triệu", morphs = big(value = 10 ** 6))
    spec.add(words = "tỷ", morphs = big(value = 10 ** 9))
    spec.add(words = "tỉ", morphs = with_fold(morph = big(value = 10 ** 9), fold = "tỷ"))
    spec.add_sep(words = ["phẩy", "chấm"])
    spec.add_pct(words = "phần trăm")
    spec.add_minus(words = "âm")
    return spec
