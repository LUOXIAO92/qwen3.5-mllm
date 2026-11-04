"""波斯语：数词不随名词变化，各部分用「و」连接（1894 → هزار و هشتصد و نود و چهار）；年份按数值读。

日期中的日读序数（12 مرداد → دوازدهم مرداد）。数字与文字直接相连（3ام）或用连字符相连（14-ساله）时丢弃。
千分位为「٬」或「,」，小数点为「٫」或「.」；「۲/۵」这类斜线写法由公共规则丢弃。
"""

from __future__ import annotations

import re

from ..numwords import fa_cardinal, fa_digits
from ..rules import Rule, Skip, alternation
from .base import Lang

ZWNJ = chr(0x200C)
ARABIC_LETTERS = f"{chr(0x0620)}-{chr(0x064A)}{chr(0x066E)}-{chr(0x06D3)}{chr(0x06FA)}-{chr(0x06FF)}"
MONTHS = (
    "فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور", "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند",
    "ژانویه", "فوریه", "مارس", "آوریل", "مه", "ژوئن", "ژوئیه", "اوت", "سپتامبر", "اکتبر", "نوامبر", "دسامبر",
)


def fa_ordinal(n: int) -> str:
    """序数：12 → دوازدهم，3 → سوم，23 → بیست و سوم，30 → سی‌ام。"""
    words = fa_cardinal(n = n)
    if words.endswith("سه"):
        return words[:-2] + "سوم"
    if words.endswith("سی"):
        return words + ZWNJ + "ام"
    return words + "م"


class Fa(Lang):
    code = "fa"
    point = "ممیز"
    percent = "درصد"
    units = {"km": ("کیلومتر", "کیلومتر"), "kg": ("کیلوگرم", "کیلوگرم"), "m": ("متر", "متر")}
    currencies = {"$": ("دلار", "دلار"), "€": ("یورو", "یورو")}

    def card(self, n: int) -> str:
        return fa_cardinal(n = n)

    def digits(self, digits: str) -> str:
        return fa_digits(digits = digits)

    def extra_rules(self) -> list:
        months = alternation(words = MONTHS)
        return [
            Rule(
                pattern = re.compile(pattern = rf"(?<![\d.,٫٬])(?P<num>\d{{1,2}})\s(?P<month>{months})(?![{ARABIC_LETTERS}])"),
                handle = self.date,
            ),
            Rule(
                pattern = re.compile(pattern = rf"[{ARABIC_LETTERS}]\d+|\d+[{ARABIC_LETTERS}]"),
                handle = lambda m, ctx: Skip(reason = f"数字与文字直接相连「{m.group()}」"),
            ),
        ]

    def date(self, m, ctx):
        n = int(m.group("num"))
        if not 1 <= n <= 31 or m.group("num").startswith("0"):
            return Skip(reason = f"日期写法无法确定「{m.group()}」")
        ctx.kind(name = "日期", surface = m.group())
        ctx.num(spoken = fa_ordinal(n = n), written = m.group("num"))
        return fa_ordinal(n = n) + " " + m.group("month")


LANG = Fa
