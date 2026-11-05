"""印尼语：数词不变形，基数、年份、日期都按数值读；千分位为「.」，小数点为「,」。

- 「ke-21」读 kedua puluh satu；「1970-an」「70-an」读 seribu sembilan ratus tujuh puluhan、tujuh puluhan。
- 时刻：「pukul 17.30」「pukul 2:30」读 pukul tujuh belas tiga puluh、pukul dua tiga puluh，整点只读小时；
  没有 pukul、jam 时，后面要有 pagi、siang、sore、malam 或 WIB 等时区。分钟为 01–09 的写法丢弃。
- 单独的「°」读 derajat；「-10 derajat」读 minus sepuluh derajat。
- 不带千分位的 5–6 位数（10000 orang）后接文字时按数值读。

马来语（ms.py）沿用这里的规则，只换数词和单位的词。
"""

from __future__ import annotations

import re

from ..numwords import idms_cardinal, idms_digits
from ..rules import Rule, Skip
from .base import Lang
from .extras import long_plain_rule, negative_rule

TIME_WORDS = ("pagi", "siang", "sore", "petang", "malam", "WIB", "WITA", "WIT")


class Id(Lang):
    code = "id"
    point = "koma"
    percent = "persen"
    minus = "minus"
    hour_words = ("pukul", "jam", "Pukul", "Jam")
    units = {
        "km/h": ("kilometer per jam", "kilometer per jam"), "km²": ("kilometer persegi", "kilometer persegi"),
        "m²": ("meter persegi", "meter persegi"), "km": ("kilometer", "kilometer"), "cm": ("sentimeter", "sentimeter"),
        "mm": ("milimeter", "milimeter"), "kg": ("kilogram", "kilogram"), "m": ("meter", "meter"),
        "g": ("gram", "gram"), "°C": ("derajat Celsius", "derajat Celsius"), "℃": ("derajat Celsius", "derajat Celsius"),
        "°F": ("derajat Fahrenheit", "derajat Fahrenheit"), "°": ("derajat", "derajat"),
    }
    currencies = {"$": ("dolar", "dolar"), "€": ("euro", "euro"), "Rp": ("rupiah", "rupiah"), "RM": ("ringgit", "ringgit")}
    scales = ("ribu", "juta", "miliar", "triliun")

    def card(self, n: int) -> str:
        return idms_cardinal(n = n, lang = self.code)

    def digits(self, digits: str) -> str:
        return idms_digits(digits = digits, lang = self.code)

    def rules(self) -> list:
        """负数先于公共保护规则运行（公共规则把 -10 当作负号或连字符丢弃）。"""
        if self.minus is None:
            return super().rules()
        return [negative_rule(lang = self, minus = self.minus, words = ("derajat",))] + super().rules()

    def extra_rules(self) -> list:
        hours = "|".join(self.hour_words)
        times = "|".join(TIME_WORDS)
        return [
            Rule(
                pattern = re.compile(
                    pattern = rf"(?:(?P<word>\b(?:{hours}))\s)?(?<![\d.,:])(?P<h>\d{{1,2}})[.:](?P<m>\d{{2}})(?!\d|[.,:]\d)(?=(?P<after>\s(?:{times})\b)?)",
                ),
                handle = self.time,
            ),
            Rule(pattern = re.compile(pattern = r"(?<![\w-])ke-(?P<num>\d{1,3})(?!\d|[.,]\d)"), handle = self.ordinal),
            Rule(pattern = re.compile(pattern = r"(?<![\w.,-])(?P<num>\d{2,4})-an(?![\w-])"), handle = self.decade),
            long_plain_rule(lang = self),
        ]

    def time(self, m, ctx):
        h, mm = int(m.group("h")), int(m.group("m"))
        if not (m.group("word") or m.group("after")):
            return None
        if not (0 <= h <= 24 and 0 <= mm <= 59):
            return Skip(reason = f"时刻写法无法确定「{m.group()}」")
        if 0 < mm < 10:
            return Skip(reason = f"分钟以 0 开头的时刻暂不处理「{m.group()}」")
        spoken = ctx.num(spoken = self.card(n = h), written = m.group("h"))
        if mm:
            spoken += " " + ctx.num(spoken = self.card(n = mm), written = m.group("m"))
        ctx.kind(name = "时刻", surface = m.group())
        return (m.group("word") + " " if m.group("word") else "") + spoken

    def ordinal(self, m, ctx):
        n = int(m.group("num"))
        if m.group("num").startswith("0"):
            return Skip(reason = f"序数写法无法确定「{m.group()}」")
        ctx.kind(name = "序数", surface = m.group())
        return ctx.num(spoken = "ke" + self.card(n = n), written = m.group())

    def decade(self, m, ctx):
        n = int(m.group("num"))
        if m.group("num").startswith("0") or n % 10:
            return Skip(reason = f"年代写法无法确定「{m.group()}」")
        ctx.kind(name = "年代", surface = m.group())
        return ctx.num(spoken = self.card(n = n), written = m.group("num")) + "an"


LANG = Id
