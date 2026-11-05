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


def number_word_guard(words, conjunctions = ()) -> Rule:
    """数字后紧跟数词（2020 one、1730 seis、2030 y una）：读出来会连成另一个数，丢弃。"""
    from ..rules import Skip

    alt = "|".join(re.escape(pattern = w) for w in sorted(set(words), key = len, reverse = True))
    conj = "|".join(re.escape(pattern = w) for w in conjunctions)
    lead = rf"(?:(?:{conj})\s+)?" if conj else ""
    pattern = re.compile(
        pattern = rf"(?<![\d.,])\d+(?:[.,]\d+)*(?=\s+{lead}(?:{alt})(?![^\W\d_]))",
        flags = re.IGNORECASE,
    )
    return Rule(
        pattern = pattern,
        handle = lambda m, ctx: Skip(reason = f"数字后紧跟数词，读出来会连成另一个数「{ctx.text[m.start():m.end() + 12].strip()}」"),
    )


def number_words(card, ordinal = None) -> set:
    """一种语言的数词（0–20、整十及 1–10 的序数），用于 number_word_guard。

    不收百、千、百万（2 mil pessoas、5 Millionen 中的量级词是正常写法），也不收数内的连接词（y、e、et、and）。
    """
    values = list(range(0, 21)) + [30, 40, 50, 60, 70, 80, 90]
    words = set()
    for n in values:
        words.update(w.lower() for w in re.split(pattern = r"[\s-]+", string = card(n = n)) if w)
        if ordinal and 1 <= n <= 10:
            words.update(w.lower() for w in re.split(pattern = r"[\s-]+", string = ordinal(n = n)) if w)
    return words - {"y", "e", "et", "and", "und", "и", "ed", "o", "u"}


def dotted_decimal_guard() -> Rule:
    """法语、俄语不用点作小数点或千分位：3.8、iOS 4.0 多为版本号或外来写法，读法不固定，丢弃。"""
    from ..rules import Skip

    return Rule(
        pattern = re.compile(pattern = r"(?<![\d.,])\d+(?:\.\d+)+(?![\d])"),
        handle = lambda m, ctx: Skip(reason = f"数字中的点读法不固定（版本号或外来写法）「{m.group()}」"),
    )


# 各语言的数量级词（小写，含缩写），用于 currency_scale_guard
SCALE_WORDS = {
    "thousand", "thousands", "million", "millions", "billion", "billions", "trillion", "trillions", "k", "m", "mn",
    "bn", "b", "tn", "crore", "lakh", "mil", "millón", "millon", "millones", "billón", "billones", "milhão",
    "milhões", "bilhão", "bilhões", "trilhão", "trilhões", "mila", "mille", "milione", "milioni", "miliardo",
    "miliardi", "milliard", "milliards", "md", "mds", "tausend", "millionen", "milliarde", "milliarden", "mio",
    "mrd", "tsd",
}


def currency_scale_guard(symbols, scales) -> Rule:
    """货币符号加数后面的数量级词没有按登记的写法出现（$550 Millones、$5 Billion、$5k、€ 5 Mrd.），丢弃。

    登记的写法（$550 millones、$5 million）由货币规则读作「数 + 数量级词 + 货币」；大小写不同、未登记或
    直接连写的数量级词如果放过，货币规则只读到数，会得到「quinientos cincuenta dólares Millones」这样语序错误的读法。
    """
    from ..rules import Skip

    syms = "|".join(re.escape(pattern = s) for s in sorted(symbols, key = len, reverse = True))
    pattern = re.compile(pattern = rf"(?<![A-Za-z])(?:{syms})\s?\d[\d.,]*(?P<sp>\s?)(?P<word>[^\W\d_]+)")

    def handle(m, ctx):
        word = m.group("word")
        if not m.group("sp") or (word.lower() in SCALE_WORDS and word not in scales):
            return Skip(reason = f"货币金额后的数量级词写法未登记「{m.group()}」")
        return None

    return Rule(pattern = pattern, handle = handle)


def range_guard() -> Rule:
    """数的范围（1500–1600、10 000 – 20 000、5–30-jährige、Nr. 5–7），与公共保护规则相同，但排在各语言的早期规则之前，
    避免空格千分位、编号缩写、复合词等规则只改写范围的一端。"""
    from ..rules import DASHES, Skip

    num = r"(?:\d{1,3}(?: \d{3})+|\d+(?:[.,]\d+)*)"
    dash = re.escape(pattern = DASHES)
    return Rule(
        pattern = re.compile(pattern = rf"(?<![\d.,\w]){num}\s*[{dash}]\s*{num}(?![\d])"),
        handle = lambda m, ctx: Skip(reason = f"范围或编号「{m.group()}」"),
    )


def fraction_guard() -> Rule:
    """日期写法（03.11.2026）和分数、斜线（1/2-Liter、1/2 h），同样排在早期规则之前。"""
    from ..rules import Skip

    return Rule(
        pattern = re.compile(pattern = r"\d{1,4}[./-]\d{1,2}[./-]\d{1,4}|\d+\s*/\s*\d+"),
        handle = lambda m, ctx: Skip(reason = f"日期、分数或斜线写法「{m.group()}」"),
    )


def score_guard() -> Rule:
    """比分、比例与后面的词用连字符相连（4:2-Sieg、17:10-Favorit）：先于复合词规则丢弃，避免只改写冒号后的数。"""
    from ..rules import Skip

    return Rule(
        pattern = re.compile(pattern = r"(?<![\d.,:])\d+\s?[:：]\s?\d+(?=[-‐‑][^\W\d_])"),
        handle = lambda m, ctx: Skip(reason = f"比分或比例与后面的词相连「{m.group()}」"),
    )


def hms_guard() -> Rule:
    """时:分:秒（2:12:00、1:23:45 Uhr）多为比赛用时，读作时刻或时长都不可靠，丢弃。"""
    from ..rules import Skip

    return Rule(
        pattern = re.compile(pattern = r"(?<![\d.,:])\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?(?![\d:])"),
        handle = lambda m, ctx: Skip(reason = f"时:分:秒写法的读法不固定「{m.group()}」"),
    )


def words_before(text: str, start: int, n: int = 3) -> list:
    """前面最多 n 个词（小写，最近的在前），遇到句末标点停止。"""
    head = text[:start]
    cut = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
    if cut >= 0:
        head = head[cut + 2:]
    words = re.findall(pattern = r"[^\W\d_]+(?:['’][^\W\d_]+)?", string = head)
    return [w.lower() for w in reversed(words[-n:])]
