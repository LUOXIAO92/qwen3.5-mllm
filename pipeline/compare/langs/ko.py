"""韩语数词：汉字词（일、이、삼……）和固有词（하나、둘、세……），按最长匹配扫描。

两套数词词形不同，两边都写成读法时不会互相合并；유월、시월只在「월」前识别。
"""

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


def build() -> LangSpec:
    spec = LangSpec(code = "ko", scan = True, digits_after_big = True)
    spec.add(words = ["영", "공"], morphs = zero())
    for v, w in SINO.items():
        spec.add(words = w, morphs = add(value = v))
    for w, exp in SINO_PLACES.items():
        spec.add(words = w, morphs = place(exp = exp))
    for w, v in SINO_BIGS.items():
        spec.add(words = w, morphs = big(value = v))
    for v, words in NATIVE.items():
        spec.add(words = words, morphs = add(value = v))
    for v, words in NATIVE_TENS.items():
        spec.add(words = words, morphs = add(value = v))
    # 六月、十月的特殊读法
    spec.add(words = "유", morphs = add(value = 6))
    spec.lookahead["유"] = "월"
    spec.add(words = "시", morphs = add(value = 10))
    spec.lookahead["시"] = "월"
    spec.add_sep(words = "점")
    spec.add_pct(words = ["퍼센트", "프로"])
    spec.add_minus(words = "마이너스")
    return spec
