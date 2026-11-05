"""英语、德语、法语、西班牙语、葡萄牙语、意大利语、俄语共用的辅助规则：缩写展开、罗马数字、字母缩写、
空格分隔的千分位。

缩写只收录读法固定的写法（etc.、z. B.、т. е.），以及后面紧跟大写词时读法固定的称谓（Dr. Smith）。
展开后原文中作为句号的点要保留：「…, etc.」在句末时读作「…, et cetera.」。
"""

from __future__ import annotations

import re

from ..rules import Rule, alternation

ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
ROMAN_VALID = re.compile(pattern = r"M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})")


def roman_value(text: str) -> int | None:
    """合法罗马数字的值，不合法时返回 None。"""
    if not text or not ROMAN_VALID.fullmatch(string = text):
        return None
    total = 0
    for i, ch in enumerate(text):
        v = ROMAN_VALUES[ch]
        if i + 1 < len(text) and ROMAN_VALUES[text[i + 1]] > v:
            total -= v
        else:
            total += v
    return total


def sentence_end_after(text: str, end: int, starters: set) -> bool:
    """缩写的点同时是句号：位于文本末尾，或后面是大写开头的常见句首词。"""
    rest = text[end:]
    if not rest.strip():
        return True
    m = re.match(pattern = r"\s+([A-ZÀ-ÝА-ЯЁ][^\W\d_]*)", string = rest)
    return bool(m) and m.group(1) in starters


def next_is_capitalized(text: str, end: int) -> bool:
    """后面隔一个空格是大写开头的词（人名、地名）。"""
    return re.match(pattern = r"\s[A-ZÀ-ÝА-ЯЁ][^\W\d_]", string = text[end:end + 3]) is not None


def abbreviation_rules(table: dict, starters: set) -> list:
    """把缩写表变成规则。

    table 的值是 (读法, 条件)：条件为 "any" 时总是展开；"name" 只在后面紧跟大写词时展开（称谓）；
    "number" 只在后面紧跟数字时展开（No. 5）。条件不满足时规则不适用，原文中的缩写由 detect 检出后整条丢弃。
    缩写以点结尾并且同时是句号时，读法后补上句号。
    """
    words = sorted(table, key = len, reverse = True)
    pattern = re.compile(pattern = rf"(?<![\w.])(?:{alternation(words = words)})(?![^\W\d_])")

    def handle(m, ctx):
        surface = m.group()
        spoken, when = table[surface]
        text = ctx.text
        if when == "name" and not next_is_capitalized(text = text, end = m.end()):
            return None
        if when == "number" and not re.match(pattern = r"\s?\d", string = text[m.end():m.end() + 2]):
            return None
        if surface.endswith(".") and when == "any" and sentence_end_after(text = text, end = m.end(), starters = starters):
            spoken += "."
        ctx.kind(name = "缩写", surface = surface)
        return spoken

    return [Rule(pattern = pattern, handle = handle)]


# 全由大写字母加点组成的缩写（U.S.、D.C.、J.R.R.），读作逐个字母
INITIALISM = re.compile(pattern = r"(?<![\w.])(?:[A-Z]\.){2,}(?![^\W\d_])")


def initialism_rule(starters: set) -> Rule:
    """U.S. → U S：字母之间用空格隔开，去掉点；位于句末时保留句号。"""

    def handle(m, ctx):
        letters = m.group().replace(".", "")
        out = " ".join(letters)
        if sentence_end_after(text = ctx.text, end = m.end(), starters = starters):
            out += "."
        ctx.kind(name = "字母缩写", surface = m.group())
        return out

    return Rule(pattern = INITIALISM, handle = handle)


def space_thousands_pattern() -> str:
    """以空格分千位的数（10 000、299 792 458）：首组 1–3 位，其余每组 3 位。"""
    return r"(?<![\d.,])\d{1,3}(?: \d{3})+(?![\d]|[.,]\d)"


def unit_follows(text: str, end: int, units = ()) -> bool:
    """数后面紧跟单位、百分号或货币符号（这些由单位、百分数、货币规则处理）。"""
    from ..rules import UNKNOWN_UNITS

    words = set(UNKNOWN_UNITS) | set(units)
    alt = "|".join(re.escape(pattern = w) for w in sorted(words, key = len, reverse = True))
    return re.match(pattern = rf"\s?(?:(?:{alt})(?![^\W\d_])|[%‰€$£¥])", string = text[end:]) is not None


def words_before(text: str, start: int, n: int = 3) -> list:
    """前面最多 n 个词（小写，最近的在前），遇到句末标点停止。"""
    head = text[:start]
    cut = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
    if cut >= 0:
        head = head[cut + 2:]
    words = re.findall(pattern = r"[^\W\d_]+(?:['’][^\W\d_]+)?", string = head)
    return [w.lower() for w in reversed(words[-n:])]
