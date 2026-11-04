"""菲律宾语数词：他加禄语数词（含连接成分 -ng、't、na、at），以及常混用的英语数词和西班牙语来源数词。"""

from ..numerals import (
    TEEN_PREFIX, LangSpec, add, big, big_lex, conn, ordinal, place, teen10,
)
from . import en

UNITS = {1: "isa", 2: "dalawa", 3: "tatlo", 4: "apat", 5: "lima", 6: "anim", 7: "pito", 8: "walo", 9: "siyam"}
TENS = {
    20: "dalawampu", 30: "tatlumpu", 40: "apatnapu", 50: "limampu",
    60: "animnapu", 70: "pitumpu", 80: "walumpu", 90: "siyamnapu",
}
SPANISH = {
    1: ["uno"], 2: ["dos"], 3: ["tres"], 4: ["kuwatro", "kwatro"], 5: ["singko"], 6: ["sais"],
    7: ["siyete"], 8: ["otso"], 9: ["nuwebe"], 11: ["onse"], 12: ["dose"], 13: ["trese"],
    14: ["katorse"], 15: ["kinse"], 20: ["beynte", "bente"], 30: ["trenta", "treynta"],
}
_VOWELS = "aeiou"
# 序数前缀 ika-/pang-，以及只在序数中出现的词干 lawa（二）、tlo（三）
_ORD_PREFIXES = ["ika", "panga", "pang"]
_ORD_STEMS = {"lawa": "dalawa", "tlo": "tatlo"}


def _linked(word: str) -> list:
    """以元音结尾的数词可带连接成分 -ng（dalawang daan、isang libo）。"""
    return [word, word + "ng"] if word[-1] in _VOWELS else [word]


def _affixes(word: str, spec: LangSpec):
    for p in _ORD_PREFIXES:
        if word.startswith(p) and len(word) > len(p) + 2:
            stem = word[len(p):]
            if stem.endswith("ng") and stem[:-2] in _ORD_STEMS:
                stem = _ORD_STEMS[stem[:-2]] + "ng"
            yield p, _ORD_STEMS.get(stem, stem), "", False


def build() -> LangSpec:
    spec = LangSpec(code = "fil", compound = True, split_affixes = _affixes)
    en.fill(spec = spec)
    for v, w in UNITS.items():
        spec.add(words = _linked(word = w), morphs = add(value = v))
    spec.add(words = ["sampu", "sampung"], morphs = teen10())
    # labing-isa、labindalawa：十一至十九
    spec.add(words = ["labing", "labin", "labim"], morphs = TEEN_PREFIX)
    for v, w in TENS.items():
        spec.add(words = _linked(word = w), morphs = add(value = v))
        # dalawampu't isa：'t 是连接词 at 的缩约
        spec.add(words = w + "t", morphs = (add(value = v), conn(after = {"tens"}, before = {"unit"})))
    spec.add(words = ["daan", "raan", "daang", "raang"], morphs = place(exp = 2, implicit_one = False))
    spec.add(words = ["sandaan", "sandaang", "isandaan"], morphs = add(value = 100))
    spec.add(words = ["libo", "libong"], morphs = big(value = 1000, implicit_one = False))
    spec.add(words = ["sanlibo", "sanlibong"], morphs = big_lex(coef = 1, value = 1000))
    spec.add(words = ["milyon", "milyong"], morphs = big(value = 10 ** 6))
    spec.add(words = ["bilyon", "bilyong"], morphs = big(value = 10 ** 9))
    spec.add(
        words = "at",
        morphs = conn(
            after = {"place", "big", "tens", "hund"},
            before = {"unit", "teen", "tens", "hund", "teen_prefix"},
        ),
    )
    # na 在辅音结尾的数词和位值词之间作连接成分（apat na raan）
    spec.add(
        words = "na",
        morphs = conn(after = {"unit", "teen", "tens"}, before = {"place", "big"}, linker = True),
    )
    for v, words in SPANISH.items():
        spec.add(words = words, morphs = add(value = v))
    spec.add(words = "diyes", morphs = teen10())
    spec.add(words = "una", morphs = ordinal(morph = add(value = 1)))
    spec.add_sep(words = "punto")
    spec.add_pct(words = ["porsiyento", "porsyento", "bahagdan", "percent"])
    spec.add_minus(words = "negatibo")
    return spec
