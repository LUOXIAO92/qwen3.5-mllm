"""英语数词。"""

from ..numerals import (
    ORD_ONLY, LangSpec, add, big, conn, ordinal, place, teen10, zero,
)

UNITS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
TEENS = ["eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
UNIT_ORDS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
TEEN_ORDS = [
    "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth",
    "sixteenth", "seventeenth", "eighteenth", "nineteenth",
]
TENS_ORDS = ["twentieth", "thirtieth", "fortieth", "fiftieth", "sixtieth", "seventieth", "eightieth", "ninetieth"]
BIGS = [("thousand", 1000), ("million", 10 ** 6), ("billion", 10 ** 9), ("trillion", 10 ** 12)]


def fill(spec: LangSpec) -> None:
    """登记英语基数词和序数词（菲律宾语也借用）。"""
    # 年份里的 oh（nineteen oh three）也是零
    spec.add(words = ["zero", "nought", "naught", "oh"], morphs = zero())
    for v, w in enumerate(UNITS, start = 1):
        spec.add(words = w, morphs = add(value = v))
    spec.add(words = "ten", morphs = teen10())
    for v, w in enumerate(TEENS, start = 11):
        spec.add(words = w, morphs = add(value = v))
    for v, w in zip(range(20, 100, 10), TENS):
        spec.add(words = w, morphs = add(value = v))
    # nineteen hundred：hundred 的系数可以是两位数
    spec.add(words = "hundred", morphs = place(exp = 2, coef_max = 99))
    for w, v in BIGS:
        spec.add(words = w, morphs = big(value = v))
    for v, w in enumerate(UNIT_ORDS, start = 1):
        spec.add(words = w, morphs = ordinal(morph = add(value = v)))
    spec.add(words = "tenth", morphs = ordinal(morph = teen10()))
    for v, w in enumerate(TEEN_ORDS, start = 11):
        spec.add(words = w, morphs = ordinal(morph = add(value = v)))
    for v, w in zip(range(20, 100, 10), TENS_ORDS):
        spec.add(words = w, morphs = ordinal(morph = add(value = v)))
    spec.add(words = "hundredth", morphs = ordinal(morph = place(exp = 2, coef_max = 99)))
    for w, v in BIGS:
        spec.add(words = w + "th", morphs = ordinal(morph = big(value = v)))


def _affixes(word: str, spec: LangSpec):
    """年代复数：nineties → ninety + s，与「90s」对齐。"""
    if word.endswith("ies") and word[:-3] + "y" in TENS:
        yield "", word[:-3] + "y", "s", False


def build() -> LangSpec:
    spec = LangSpec(code = "en", split_affixes = _affixes)
    fill(spec = spec)
    # a hundred、a thousand：a 只在后面紧跟位值词时作「一」
    spec.add(words = "a", morphs = add(value = 1, needs = frozenset({"place", "big"})))
    spec.add(words = "and", morphs = conn(after = {"place", "big"}, before = {"unit", "teen", "tens"}))
    spec.add_sep(words = "point")
    spec.add_pct(words = ["percent", "per cent"])
    spec.add_minus(words = ["minus", "negative"])
    spec.add_ordinal_suffix(pattern = r"(?:st|nd|rd|th)(?![^\W\d_])", kinds = ORD_ONLY)
    return spec
