"""日语按读音比较：用形态分析把汉字、假名写法不同的词换成片假名读音。

朗读稿「言って」「嬉しい」「稀」与识别结果「いって」「うれしい」「稀れ」写法不同、读音相同。
compare 先按写法比较，不一致时把两边都换成读音再比较一次，任一次一致即判一致。

含汉字的词取 UniDic 的假名读音（kana），只由假名写成的词直接用原来的假名，以免形态分析把写错的假名
（「なた」）当成别的词形（「なった」）。之后在词内把长音的几种写法统一成「ー」（コウ、コオ、コー；ケイ、ケー），
并合并发音相同的假名：ヴァ/バ、ヂ/ジ、ヅ/ズ。所以同音词（交響曲、公共局）和 UniDic 登记的送假名
变体（嬉い、引渡し）也判为读音相同。

数词和紧跟在数词后的量词、汉字词保留原来的写法，仍由数词解析处理，因为「4月1日」这类阿拉伯数字后的
量词读音由上下文决定（UniDic 给「1日」的「日」读作「カ」），并且两边的切分可能不同（「千九百五十年代」
切成「年」「代」，「1950年代」切成「年代」）。不是数词的词即使以数字开头（一緒、一番）也换成读音，
这样「一緒」与「いっしょ」一致；「一日」对「1日」这类差别由按写法的比较处理。词典中没有的词
（多为片假名外来语、拉丁字母）保留原样。

依赖 fugashi（MIT）和 unidic-lite（MIT，词典为 UniDic，BSD 许可）。两者是可选依赖：没有安装时
available() 返回 False，compare 只按写法比较，并在差异说明中注明，第一次调用时发出警告。
"""

from __future__ import annotations

import unicodedata
import warnings
from functools import lru_cache

_COUNTER_POS = {"助数詞", "助数詞可能"}
_KANA_FOLD = [
    ("ヴァ", "バ"), ("ヴィ", "ビ"), ("ヴェ", "ベ"), ("ヴォ", "ボ"), ("ヴ", "ブ"),
    ("ヂ", "ジ"), ("ヅ", "ズ"),
]
_HIRAGANA_TO_KATAKANA = {c: c + 0x60 for c in range(0x3041, 0x3097)}

# 片假名所在的段，用于把长音的写法统一成「ー」
_ROWS = {
    "a": "アカガサザタダナハバパマヤラワァャ",
    "i": "イキギシジチヂニヒビピミリィ",
    "u": "ウクグスズツヅヌフブプムユルゥュ",
    "e": "エケゲセゼテデネヘベペメレェ",
    "o": "オコゴソゾトドノホボポモヨロヲォョ",
}
_VOWEL_OF = {ch: v for v, chars in _ROWS.items() for ch in chars}
_LONG = {"a": "ア", "i": "イ", "u": "ウ", "e": "エイ", "o": "オウ"}

MISSING_NOTE = "日语未按读音比较：没有安装 fugashi 和 unidic-lite"
_warned = False


@lru_cache(maxsize = 1)
def _tagger():
    try:
        import fugashi
        import unidic_lite  # noqa: F401  只检查词典是否安装，fugashi 会自动找到它
    except ImportError:
        return None
    try:
        return fugashi.Tagger()
    except RuntimeError:
        return None


def available() -> bool:
    """形态分析器是否可用；不可用时第一次调用发出警告。"""
    global _warned
    ok = _tagger() is not None
    if not ok and not _warned:
        warnings.warn(
            message = "pipeline.compare：" + MISSING_NOTE + "，日语只按写法比较。安装：pip install fugashi unidic-lite",
            category = RuntimeWarning,
            stacklevel = 3,
        )
        _warned = True
    return ok


def _feature(word, name: str) -> str:
    value = getattr(word.feature, name, None)
    return value if isinstance(value, str) and value not in ("", "*") else ""


def _is_kanji(text: str) -> bool:
    return bool(text) and all("一" <= ch <= "鿿" or ch in "々〆ヶ" for ch in text)


def _is_kana(text: str) -> bool:
    return bool(text) and all("ぁ" <= ch <= "ヿ" for ch in text)


def _is_numeral(word) -> bool:
    return _feature(word = word, name = "pos2") == "数詞" or any("0" <= ch <= "9" for ch in word.surface)


def _long_vowels(text: str) -> str:
    """长音的写法统一成「ー」：同段元音（カア）、オ段加ウ（コウ）、エ段加イ（ケイ）。"""
    out = []
    for ch in text:
        vowel = _VOWEL_OF.get(out[-1]) if out else None
        if vowel is not None and ch in _LONG[vowel]:
            out.append("ー")
            continue
        out.append(ch)
    return "".join(out).replace("ーー", "ー")


def to_reading(text: str) -> str:
    """把文本换成片假名读音；数词和数词后的量词保留原写法。需要 available() 为 True。"""
    tagger = _tagger()
    text = unicodedata.normalize("NFKC", text)
    out = []
    after_numeral = False
    for word in tagger(text):
        surface = word.surface
        if _is_numeral(word = word):
            out.append(surface)
            after_numeral = True
            continue
        counter = (
            _feature(word = word, name = "pos1") == "接尾辞"
            or _feature(word = word, name = "pos2") in _COUNTER_POS
            or _feature(word = word, name = "pos3") in _COUNTER_POS
        )
        if after_numeral and (counter or _is_kanji(text = surface)):
            # 数词后连续的量词和汉字词（年、代、年間、回）保留写法
            out.append(surface)
            continue
        after_numeral = False
        kana = _feature(word = word, name = "kana")
        if _is_kana(text = surface) or word.is_unk or not kana:
            kana = surface.translate(_HIRAGANA_TO_KATAKANA)
        # 长音只在词内统一：「先生」センセイ 与「せんせー」一致，「て・いる」与「て・える」不一致
        out.append(_long_vowels(text = kana))
    reading = "".join(out)
    for old, new in _KANA_FOLD:
        reading = reading.replace(old, new)
    return reading.replace("ーー", "ー")
