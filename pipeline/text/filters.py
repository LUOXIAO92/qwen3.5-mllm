"""可朗读性过滤：判断一句文本能不能原样交给 TTS 朗读。

`check_text` 返回丢弃原因（字符串），通过则返回 None。规则只依赖文本本身
和语言代码，不依赖来源，所以网页类来源可以直接复用。语言检测在 `langid.py`。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from .rules import check_language_rules
from .langs import (
    CASED_LANGS,
    DEFAULT_MIN_SCRIPT_RATIO,
    EXPECTED_SCRIPTS,
    MIN_SCRIPT_RATIO,
    NO_SPACE_LANGS,
)

TERMINALS = ".!?…。！？؟۔｡‼⁇⁈"
CLOSERS = "\"'”’»」』）)]】》〉*"


@dataclass(frozen = True)
class FilterOptions:
    """过滤参数。`require_terminal` 要求以句末标点结尾（泰语等无句末标点的语言关闭）。"""

    require_terminal: bool = True
    min_words: int = 3  # 有空格分词的语言
    min_chars: int = 6  # 不分词的语言（中文、日语、泰语）
    max_chars: int = 600
    max_digit_ratio: float = 0.12
    max_symbols: int = 2
    max_bracket_groups: int = 1  # 括号组超过这个数就丢弃（维基段落里括号注释堆叠）
    max_bracket_ratio: float = 0.25  # 括号内容占全句比例上限


def default_options(lang: str) -> FilterOptions:
    """按语言给出默认参数：泰语没有句末标点，不要求结尾标点，最短字符数也更大。"""
    if lang == "th":
        return FilterOptions(require_terminal = False, min_chars = 12)
    if lang == "ja":
        return FilterOptions(min_chars = 8)
    return FilterOptions()


_URL = re.compile(
    r"(https?://|ftp://|www\.|\b[a-z0-9-]+\.(com|org|net|edu|gov|io|info|biz|html?|php)\b)",
    re.IGNORECASE,
)
_EMAIL = re.compile(r"\S+@\S+|\bexample\.com\b|email@", re.IGNORECASE)
_CODE = re.compile(
    r"[{}]|</?[a-zA-Z][^>]{0,40}>|=>|==|&&|\|\||::|\\[a-zA-Z]{2,}|\\n|"
    r"\b(def|class|import|return|elif|printf|println|console\.log|function|SELECT|INSERT)\b\s*[\w(<\"'{]|"
    r"\b[A-Za-z_]\w*\.[A-Za-z_]\w*\(|\b[a-z]+_[a-z_]+\b|__\w+__|&[a-z]+;|\$[^$ ]+\$|\bhref\b|<!--"
)
_MARKDOWN = re.compile(r"^\s*([#>*\-–—•·▪■●◆▶►◦]+|\d{1,2}[.)、]|\(\d{1,2}\)|[①-⑩])\s|\*\*|__|\|.*\||\[\[|\]\]|\{\{|`")
_BRACKET_NOTE = re.compile(r"\[[^\]]{0,40}\]")
_EMPTY_BRACKET = re.compile(r"“\s*”|\"\"|«\s*»|‘\s*’|[(（\[【「『][\s,;:，、；：\-–—·.…/]*[)）\]】」』]")
_HOLE = re.compile(r"[:：]\s*[,，;；、)）]|[(（]\s*[,，;；:：]|[,，;；]\s*[)）]")
_REPEAT_CHAR = re.compile(r"([^\W\d_])\1{5,}|([!?.,;:~=*_-])\2{3,}")
_REPEAT_WORD = re.compile(r"\b(\w+)(\s+\1\b){3,}", re.IGNORECASE)
_BAD_CHARS = re.compile(r"[ɐ-ʯ←-⇿⌀-⏿■-➿⬀-⯿-�]|ⓘ")
_DIGIT_RUN = re.compile(r"\d{9,}")
_GAMBLING_ZH = re.compile(
    "博彩|赌场|賭場|赌博|娱乐城|娛樂城|百家乐|百家樂|时时彩|時時彩|六合彩|注册送|註冊送|棋牌|彩票|下注|太阳城|太陽城|代理加盟"
)
_PAIRS = [("(", ")"), ("（", "）"), ("[", "]"), ("「", "」"), ("『", "』"), ("“", "”"), ("«", "»")]


@lru_cache(maxsize = 4096)
def _script_of(ch: str) -> str:
    try:
        return unicodedata.name(ch).split()[0]
    except ValueError:
        return "UNKNOWN"


def script_ratio(text: str, *, lang: str) -> float:
    """期望文字的字母占全部字母的比例。"""
    expected = EXPECTED_SCRIPTS[lang]
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    hits = sum(1 for ch in letters if _script_of(ch) in expected)
    return hits / len(letters)


def _has_terminal(text: str) -> bool:
    stripped = text.rstrip(CLOSERS + " ")
    return bool(stripped) and stripped[-1] in TERMINALS


def _unbalanced(text: str) -> bool:
    for left, right in _PAIRS:
        if text.count(left) != text.count(right):
            return True
    return text.count('"') % 2 == 1


def check_text(text: str, *, lang: str, options: FilterOptions | None = None) -> str | None:
    """逐条规则检查，返回第一个命中的丢弃原因；全部通过返回 None。"""
    options = options or default_options(lang)
    if not text:
        return "empty"
    n_chars = len(text)
    if n_chars > options.max_chars:
        return "too_long"
    if lang in NO_SPACE_LANGS:
        if n_chars < options.min_chars:
            return "too_short"
    elif len(text.split()) < options.min_words:
        return "too_short"

    if _URL.search(text):
        return "url"
    if _EMAIL.search(text):
        return "email"
    if _MARKDOWN.search(text):
        return "markup"
    if _CODE.search(text):
        return "code"
    if _BRACKET_NOTE.search(text):
        return "bracket_note"
    if _EMPTY_BRACKET.search(text) or _HOLE.search(text):
        return "template_hole"
    if _BAD_CHARS.search(text):
        return "bad_chars"
    if _REPEAT_CHAR.search(text) or _REPEAT_WORD.search(text):
        return "repetition"
    if _DIGIT_RUN.search(text):
        return "long_number"
    if lang == "zh" and _GAMBLING_ZH.search(text):
        return "gambling"

    if text.endswith((":", "：", ";", "；", ",", "，", "、")):
        return "list_intro"
    if options.require_terminal and not _has_terminal(text):
        return "no_terminal"
    if text.count(";") + text.count("；") >= 3 or text.count("、") >= 6 or "•" in text:
        return "list_like"
    if lang not in NO_SPACE_LANGS:
        words = text.split()
        separators = sum(text.count(mark) for mark in ",;،،")
        if len(words) >= 6 and separators / len(words) >= 0.3:
            return "list_like"

    digits = sum(ch.isdigit() for ch in text)
    if digits / n_chars > options.max_digit_ratio:
        return "digits"
    symbols = sum(unicodedata.category(ch)[0] == "S" for ch in text)
    if symbols > options.max_symbols:
        return "symbols"
    punct = sum(unicodedata.category(ch)[0] == "P" for ch in text)
    if punct / n_chars > 0.3:
        return "punctuation"
    if _unbalanced(text):
        return "unbalanced"

    reason = check_language_rules(
        text,
        lang = lang,
        max_bracket_groups = options.max_bracket_groups,
        max_bracket_ratio = options.max_bracket_ratio,
    )
    if reason:
        return reason

    letters = [ch for ch in text if ch.isalpha()]
    if len(letters) < 2:
        return "no_letters"
    if len(letters) >= 5 and all(ch.isupper() for ch in letters):
        return "all_caps"
    if lang in CASED_LANGS:
        first = next((ch for ch in text if ch.isalnum()), "")
        if first.isalpha() and not first.isupper():
            return "lowercase_start"

    if script_ratio(text, lang = lang) < MIN_SCRIPT_RATIO.get(lang, DEFAULT_MIN_SCRIPT_RATIO):
        return "script"
    if lang == "ja" and not any("぀" <= ch <= "ヿ" for ch in text):
        return "no_kana"
    return None
