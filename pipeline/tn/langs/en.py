"""英语：基数、序数（21st）、年份（nineteen sixty）、年代（1990s）、「Month D, YYYY」日期、时刻、单位和货币。

1000–2099 之间不带千分位的四位数可能是年份也可能是数量（「in 1500 cases」），只有前面是 in、since、
from 这类词、后面是标点、大写词或虚词时才按年份读，其余丢弃。「13 October」这类日在前的日期要读成
「the thirteenth of October」，会增加原文没有的词，暂不处理。
"""

from __future__ import annotations

import re

from ..numwords import n2w
from ..rules import Rule, Skip, alternation
from .base import Lang, word_after, word_before

MONTHS = (
    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
    "November", "December",
)
YEAR_BEFORE = {
    "in", "since", "from", "through", "until", "till", "during", "circa", "early", "late", "mid", "year", "by",
    "of", "to", "and", "or", "between", "before", "after", "around", "as",
}
FUNCTION_AFTER = {
    "the", "a", "an", "and", "or", "to", "through", "until", "till", "when", "it", "he", "she", "they", "we",
    "i", "was", "were", "is", "are", "had", "has", "have", "but", "as", "at", "on", "in", "by", "with", "for",
    "of", "after", "before", "during", "while", "which", "that", "this", "his", "her", "its", "their", "there",
}
ORDINAL_SUFFIX = {1: "st", 2: "nd", 3: "rd"}


def ordinal_suffix(n: int) -> str:
    if 11 <= n % 100 <= 13:
        return "th"
    return ORDINAL_SUFFIX.get(n % 10, "th")


def pluralize(words: str) -> str:
    """年份读法的最后一个词变复数：nineteen ninety → nineteen nineties。"""
    head, _, last = words.rpartition(" ")
    if last.endswith("y"):
        last = last[:-1] + "ies"
    else:
        last += "s"
    return (head + " " if head else "") + last


class En(Lang):
    code = "en"
    point = "point"
    percent = "percent"
    units = {
        "km/h": ("kilometer per hour", "kilometers per hour"), "mph": ("mile per hour", "miles per hour"),
        "km²": ("square kilometer", "square kilometers"), "m²": ("square meter", "square meters"),
        "km": ("kilometer", "kilometers"), "cm": ("centimeter", "centimeters"), "mm": ("millimeter", "millimeters"),
        "kg": ("kilogram", "kilograms"), "m": ("meter", "meters"), "ft": ("foot", "feet"), "mi": ("mile", "miles"),
        "lb": ("pound", "pounds"), "lbs": ("pound", "pounds"),
        "°C": ("degree Celsius", "degrees Celsius"), "℃": ("degree Celsius", "degrees Celsius"),
        "°F": ("degree Fahrenheit", "degrees Fahrenheit"), "℉": ("degree Fahrenheit", "degrees Fahrenheit"),
    }
    currencies = {"$": ("dollar", "dollars"), "€": ("euro", "euros"), "£": ("pound", "pounds")}
    scales = ("thousand", "million", "billion", "trillion")

    def card(self, n: int) -> str:
        return n2w(n = n, lang = "en")

    def attributive(self, m, ctx) -> bool:
        """前面是 a/an 时金额、单位作定语：a $70 million impact → a seventy million dollar impact。"""
        return word_before(text = ctx.text, start = m.start()) in ("a", "an")

    def unit_rules(self) -> list:
        rules = super().unit_rules()
        for rule in rules:
            rule.handle = self.singular_after_article(handle = rule.handle, table = self.units)
        return rules

    def currency_rules(self) -> list:
        rules = super().currency_rules()
        for rule in rules:
            rule.handle = self.singular_after_article(handle = rule.handle, table = self.currencies)
        return rules

    def singular_after_article(self, handle, table: dict):
        def wrapped(m, ctx):
            out = handle(m = m, ctx = ctx)
            if isinstance(out, str) and self.attributive(m = m, ctx = ctx):
                for singular, plural in table.values():
                    if out.endswith(" " + plural):
                        return out[:-len(plural)] + singular
            return out

        return wrapped

    def digits(self, digits: str) -> str:
        return " ".join(n2w(n = int(d), lang = "en") for d in digits)

    def year(self, n: int) -> str:
        return n2w(n = n, lang = "en", to = "year")

    def extra_rules(self) -> list:
        months = alternation(words = MONTHS)
        return [
            Rule(
                pattern = re.compile(pattern = r"\b(?:at|by|from|until|till|before|after)\s\d{1,2}\.\d{2}(?!\d)"),
                handle = lambda m, ctx: Skip(reason = f"时刻写法（7.30）「{m.group()}」"),
            ),
            Rule(
                pattern = re.compile(
                    pattern = (
                        rf"(?P<month>{months})\s(?P<day>\d{{1,2}})(?P<suf>st|nd|rd|th)?"
                        rf"(?:(?P<sep>,?\s)(?P<year>\d{{4}}))?(?!\d|[.,]\d)"
                    ),
                ),
                handle = self.month_day,
            ),
            Rule(
                pattern = re.compile(pattern = rf"(?<![\d.,])\d{{1,2}}(?:st|nd|rd|th)?\s(?:of\s)?(?:{months})\b"),
                handle = lambda m, ctx: Skip(reason = f"日在前的日期需要加「the … of」「{m.group()}」"),
            ),
            Rule(pattern = re.compile(pattern = r"(?<![\d.,])(?P<num>\d+)(?P<suf>st|nd|rd|th)\b"), handle = self.ordinal),
            Rule(pattern = re.compile(pattern = r"(?<![\d.,])(?P<num>1\d{2}0|20\d0)s\b"), handle = self.decade),
            Rule(pattern = re.compile(pattern = r"(?<![\d.,])(?P<h>\d{1,2}):(?P<mm>\d{2})(?![\d:])"), handle = self.time),
            Rule(pattern = re.compile(pattern = rf"(?P<num>{self.num})-(?P<word>[a-z]+)"), handle = self.compound),
            Rule(pattern = re.compile(pattern = r"(?<![\d.,])(?P<num>\d{4})(?![\d]|[.,]\d)"), handle = self.four_digits),
        ]

    def month_day(self, m, ctx):
        day = int(m.group("day"))
        if not 1 <= day <= 31 or (m.group("suf") and m.group("suf") != ordinal_suffix(n = day)):
            return Skip(reason = f"日期写法无法确定「{m.group()}」")
        out = m.group("month") + " " + ctx.num(spoken = n2w(n = day, lang = "en", to = "ordinal"), written = f"{day}th")
        if m.group("year"):
            year = int(m.group("year"))
            out += m.group("sep") + ctx.num(spoken = self.year(n = year), written = m.group("year"))
        ctx.kind(name = "日期", surface = m.group())
        return out

    def ordinal(self, m, ctx):
        n = int(m.group("num"))
        if m.group("suf") != ordinal_suffix(n = n) or (len(m.group("num")) > 1 and m.group("num").startswith("0")):
            return Skip(reason = f"序数写法无法确定「{m.group()}」")
        ctx.kind(name = "序数", surface = m.group())
        return ctx.num(spoken = n2w(n = n, lang = "en", to = "ordinal"), written = m.group())

    def decade(self, m, ctx):
        n = int(m.group("num"))
        ctx.kind(name = "年代", surface = m.group())
        return ctx.num(spoken = pluralize(words = self.year(n = n)), written = m.group())

    def time(self, m, ctx):
        h, mm = int(m.group("h")), int(m.group("mm"))
        if h > 24 or mm > 59:
            return Skip(reason = f"时刻写法无法确定「{m.group()}」")
        if mm == 0 and not 1 <= h <= 12:
            return Skip(reason = f"整点的 24 小时制读法不固定「{m.group()}」")
        ctx.kind(name = "时刻", surface = m.group())
        hour = ctx.num(spoken = self.card(n = h), written = m.group("h"))
        if mm == 0:
            return hour + " o'clock"
        minute = self.card(n = mm) if mm >= 10 else "oh " + self.card(n = mm)
        return hour + " " + ctx.num(spoken = minute, written = m.group("mm"))

    def compound(self, m, ctx):
        """24-year-old → twenty-four-year-old。"""
        n = self.parse(written = m.group("num"))
        if n is None or not n.is_int or 1000 <= n.value <= 2099:
            return None
        ctx.kind(name = "基数", surface = m.group())
        return self.say(n = n, ctx = ctx) + "-" + m.group("word")

    def four_digits(self, m, ctx):
        n = int(m.group("num"))
        if not 1000 <= n <= 2099:
            return None
        text = ctx.text
        before = word_before(text = text, start = m.start())
        after = word_after(text = text, end = m.end())
        rest = text[m.end():].lstrip()
        after_ok = (
            not rest
            or not (rest[0].isalnum())
            or after[:1].isupper()
            or after.lower() in FUNCTION_AFTER
        )
        if before in YEAR_BEFORE and after_ok:
            ctx.kind(name = "年份", surface = m.group())
            return ctx.num(spoken = self.year(n = n), written = m.group())
        return Skip(reason = f"四位数是年份还是数量需要上下文「{m.group()}」")


LANG = En
