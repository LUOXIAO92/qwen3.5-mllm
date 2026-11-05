"""韩语数词：汉字词（일、이、삼……）和固有词（하나、둘、세……），按最长匹配扫描。

两套数词词形不同，两边都写成读法时不会互相合并；유월、시월只在「월」前识别。
同一组内不混用两套数词，所以「십세」是「십」（10）加量词「세」（岁），「이십세기」是 20 加「세기」，
只有汉字词的百位以上后面可以接固有词（백스물셋）。

汉字词的个位与常见量词、助词同音：「육십일」可以是 61，也可以是 60 加「일」（日），
「십이」可以是 12，也可以是 10 加助词「이」，「이천구백칠십」可以是 2970，也可以是「이」（这）加 1970。
读音完全相同，比对时几种切分都尝试（_splits）。固有词不作「만」「억」的系数，「두만」是「두」加「만」。
"""

from ..extend import reparse, word_num
from ..numerals import LangSpec, add, big, place, zero

SINO = {1: "일", 2: "이", 3: "삼", 4: "사", 5: "오", 6: "육", 7: "칠", 8: "팔", 9: "구"}
SINO_PLACES = {"십": 1, "백": 2, "천": 3}
SINO_BIGS = {"만": 10 ** 4, "억": 10 ** 8, "조": 10 ** 12}
NATIVE = {
    1: ["하나", "한", "첫"], 2: ["둘", "두"], 3: ["셋", "세", "석", "서"], 4: ["넷", "네", "넉", "너"],
    5: ["다섯"], 6: ["여섯"], 7: ["일곱"], 8: ["여덟"], 9: ["아홉"],
}
NATIVE_TENS = {
    10: ["열"], 20: ["스물", "스무"], 30: ["서른"], 40: ["마흔"], 50: ["쉰"],
    60: ["예순"], 70: ["일흔"], 80: ["여든"], 90: ["아흔"],
}
_SINO_UNIT_VALUES = {w: v for v, w in SINO.items()}
_SINO_PLACE_CHARS = set(SINO_PLACES) | set(SINO_BIGS)


# 一个数最多给出几种切分
_MAX_SPLITS = 6


def _splits(num, spec) -> list:
    """汉字词数的其他切分，每处切分前后读音不变。

    末尾的个位单独成数（육십일 = 육십 + 일），位值词前的个位单独成数（이천구백칠십 = 이 + 천구백칠십，
    만이천구백 = 만 + 이 + 천구백），以及这些切分同时进行（이십일 = 이 + 십 + 일）。
    """
    s = num.surface
    if len(s) < 2:
        return []
    choices = []
    if s[-1] in _SINO_UNIT_VALUES and s[-2] in _SINO_PLACE_CHARS:
        choices.append({len(s) - 1})
    for p in range(len(s) - 1):
        if s[p] in _SINO_UNIT_VALUES and s[p + 1] in _SINO_PLACE_CHARS:
            choices.append({p + 1} if p == 0 else {p, p + 1})
    if len(choices) > 1:
        choices.append(set().union(*choices))
    out = []
    for cuts in choices[:_MAX_SPLITS]:
        bounds = [0] + sorted(cuts) + [len(s)]
        pieces = [s[a:b] for a, b in zip(bounds, bounds[1:])]
        values = [reparse(text = piece, spec = spec) for piece in pieces]
        if any(v is None for v in values):
            continue
        out.append([
            word_num(
                value = value,
                surface = piece,
                display = piece,
                kinds = num.kinds,
            )
            for piece, value in zip(pieces, values)
        ])
    return out


def build() -> LangSpec:
    spec = LangSpec(code = "ko", scan = True, digits_after_big = True)

    def splitter(num) -> list:
        return _splits(num = num, spec = spec)

    spec.splitter = splitter
    spec.add(words = ["영", "공"], morphs = zero())
    for v, w in SINO.items():
        spec.add(words = w, morphs = add(value = v, system = "sino"))
    for w, exp in SINO_PLACES.items():
        spec.add(words = w, morphs = place(exp = exp, system = "sino"))
    for w, v in SINO_BIGS.items():
        spec.add(words = w, morphs = big(value = v))
    for v, words in NATIVE.items():
        spec.add(words = words, morphs = add(value = v, system = "native"))
    for v, words in NATIVE_TENS.items():
        spec.add(words = words, morphs = add(value = v, system = "native"))
    # 六月、十月的特殊读法
    spec.add(words = "유", morphs = add(value = 6, system = "sino"))
    spec.lookahead["유"] = "월"
    spec.add(words = "시", morphs = add(value = 10, system = "sino"))
    spec.lookahead["시"] = "월"
    spec.add_sep(words = "점")
    spec.add_pct(words = ["퍼센트", "프로"])
    spec.add_minus(words = "마이너스")
    return spec
