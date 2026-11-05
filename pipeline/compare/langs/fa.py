"""波斯语数词。文本已把 ي/ك 等统一为波斯字形，零宽不连字改为空格。

序数后缀 -م、-مین 直接登记；「سی ام」这类被空格隔开的 ام 作为序数标记。
"""

from ..numerals import (
    ORD_ONLY, LangSpec, add, big, conn, ordinal, place, teen10, zero,
)

CARDINALS = {
    1: ["یک"], 2: ["دو"], 3: ["سه"], 4: ["چهار"], 5: ["پنج"], 6: ["شش", "شیش"], 7: ["هفت"],
    8: ["هشت"], 9: ["نه"],
    11: ["یازده"], 12: ["دوازده"], 13: ["سیزده"], 14: ["چهارده"], 15: ["پانزده", "پونزده"],
    16: ["شانزده"], 17: ["هفده", "هیفده"], 18: ["هجده", "هیجده", "هژده"], 19: ["نوزده"],
    20: ["بیست"], 30: ["سی"], 40: ["چهل"], 50: ["پنجاه"], 60: ["شصت"], 70: ["هفتاد"],
    80: ["هشتاد"], 90: ["نود"],
    200: ["دویست"], 300: ["سیصد"], 400: ["چهارصد"], 500: ["پانصد"], 600: ["ششصد"],
    700: ["هفتصد"], 800: ["هشتصد"], 900: ["نهصد"],
}
BIGS = {1000: ["هزار"], 10 ** 6: ["میلیون"], 10 ** 9: ["میلیارد", "بیلیون"], 10 ** 12: ["تریلیون"]}
# 序数词干：سه → سو（سوم），以元音 ی 结尾的 سی 加 ام
_ORD_STEM = {"سه": "سو"}


def _ord_forms(word: str) -> list:
    if word.endswith("ی"):
        return [word + "ام", word + "امین"]
    stem = _ORD_STEM.get(word, word)
    return [stem + "م", stem + "مین"]


def _affixes(word: str, spec: LangSpec, ext: bool = False):
    """数内的连接词 و 有时与后一个词连写：بیست（零宽不连字）ویک → بیست ویک。"""
    if word.startswith("و") and len(word) >= 3:
        yield "و", word[1:], "", True


def build() -> LangSpec:
    spec = LangSpec(code = "fa", split_affixes = _affixes)
    spec.add(words = "صفر", morphs = zero())
    for v, words in CARDINALS.items():
        spec.add(words = words, morphs = add(value = v))
        for w in words:
            spec.add(words = _ord_forms(word = w), morphs = ordinal(morph = add(value = v)))
    spec.add(words = "ده", morphs = teen10())
    spec.add(words = _ord_forms(word = "ده"), morphs = ordinal(morph = teen10()))
    spec.add(words = "صد", morphs = place(exp = 2))
    spec.add(words = _ord_forms(word = "صد"), morphs = ordinal(morph = place(exp = 2)))
    for v, words in BIGS.items():
        spec.add(words = words, morphs = big(value = v))
        for w in words:
            spec.add(words = _ord_forms(word = w), morphs = ordinal(morph = big(value = v)))
    spec.add(words = ["یکم", "یکمین", "اول", "اولین", "نخست", "نخستین"], morphs = ordinal(morph = add(value = 1)))
    spec.add(words = "و", morphs = conn())
    spec.add_ord_mark(words = ["ام", "امین"])
    spec.add_sep(words = "ممیز")
    spec.add_pct(words = "درصد")
    spec.add_minus(words = "منفی")
    spec.add_ordinal_suffix(pattern = r" ?(?:امین|مین|ام|م)(?![^\W\d_])", kinds = ORD_ONLY)
    return spec
