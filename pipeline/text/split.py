"""按规则切分句子，用于排除比对和把维基段落拆成整句。

规则切分不追求完美：缩写和首字母（`J. R. R.`、`Dr.`）不切，其余在句末标点
之后切开。泰语没有句末标点，整段当作一句。
"""

from __future__ import annotations

import re

# 句末标点后可以跟的右引号、右括号。
_CLOSERS = "\"'”’»」』）)]】》〉"

# 东亚句末标点：后面不需要空格就切。
_CJK_END = re.compile(r"([。！？!?…]+[" + re.escape(_CLOSERS) + r"]*)")
# 西文、阿拉伯文句末标点：后面要有空白才切。
_LATIN_END = re.compile(r"([.!?…؟۔]+[" + re.escape(_CLOSERS) + r"]*)(\s+)")

# 以点结尾但不是句末的常见缩写（小写比较）。
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "sra", "srta", "st", "vs", "etc", "no", "nº", "jr", "inc", "ltd",
    "co", "corp", "fig", "al", "ed", "eds", "vol", "pp", "ca", "cf", "approx", "dept", "est", "gen", "col",
    "lt", "sgt", "capt", "gov", "sen", "rep", "rev", "hon", "mt", "ft", "bzw", "usw", "ggf", "dr.", "nr",
    "mme", "mlle", "mm", "cap", "av", "ex", "tel", "ul", "tt", "ss", "bpk", "ibu", "hj", "dll", "dsb", "yg",
    "г", "гг", "ул", "им", "см", "рис", "тыс", "млн", "млрд", "т", "д", "п", "др", "проф", "акад",
}

_CJK_LANGS = {"zh", "ja", "ko"}


def split_sentences(text: str, *, lang: str) -> list[str]:
    """把文本切成句子列表，不丢字符（只丢句间空白）。"""
    text = text.strip()
    if not text:
        return []
    if lang == "th":
        return [text]
    if lang in {"zh", "ja"}:
        return _split_cjk(text)
    return _split_latin(text)


def _split_cjk(text: str) -> list[str]:
    parts = _CJK_END.split(text)
    sentences = []
    for index in range(0, len(parts) - 1, 2):
        sentence = (parts[index] + parts[index + 1]).strip()
        if sentence:
            sentences.append(sentence)
    tail = parts[-1].strip() if len(parts) % 2 == 1 else ""
    if tail:
        sentences.append(tail)
    return sentences


def _is_abbreviation(before: str) -> bool:
    """句点前的最后一个词是缩写或单个字母（首字母）时不切。"""
    words = before.split()
    if not words:
        return False
    last = words[-1].strip("\"'“‘«(（[").rstrip(".").lower()
    if len(last) == 1 and last.isalpha():
        return True
    return last in _ABBREVIATIONS


def _split_latin(text: str) -> list[str]:
    sentences = []
    start = 0
    for match in _LATIN_END.finditer(text):
        end = match.end(1)
        mark = match.group(1)
        if mark.startswith(".") and len(mark) == 1 and _is_abbreviation(text[start:match.start(1)]):
            continue
        following = text[match.end():match.end() + 1]
        if following and following.islower():
            continue  # 句点后接小写字母，按句内处理
        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences
