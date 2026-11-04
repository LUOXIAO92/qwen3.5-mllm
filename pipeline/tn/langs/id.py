"""印尼语：数词不变形，基数、年份、日期都按数值读；「ke-21」读 kedua puluh satu；千分位为「.」，小数点为「,」。

马来语（ms.py）沿用这里的规则，只换数词和单位的词。「pukul 09.30」这类时刻和「2000-an」由公共规则丢弃。
"""

from __future__ import annotations

import re

from ..numwords import idms_cardinal, idms_digits
from ..rules import Rule, Skip
from .base import Lang


class Id(Lang):
    code = "id"
    point = "koma"
    percent = "persen"
    units = {
        "km/h": ("kilometer per jam", "kilometer per jam"), "km²": ("kilometer persegi", "kilometer persegi"),
        "m²": ("meter persegi", "meter persegi"), "km": ("kilometer", "kilometer"), "cm": ("sentimeter", "sentimeter"),
        "mm": ("milimeter", "milimeter"), "kg": ("kilogram", "kilogram"), "m": ("meter", "meter"),
        "g": ("gram", "gram"), "°C": ("derajat Celsius", "derajat Celsius"), "℃": ("derajat Celsius", "derajat Celsius"),
        "°F": ("derajat Fahrenheit", "derajat Fahrenheit"),
    }
    currencies = {"$": ("dolar", "dolar"), "€": ("euro", "euro"), "Rp": ("rupiah", "rupiah")}
    scales = ("ribu", "juta", "miliar", "triliun")

    def card(self, n: int) -> str:
        return idms_cardinal(n = n, lang = self.code)

    def digits(self, digits: str) -> str:
        return idms_digits(digits = digits, lang = self.code)

    def extra_rules(self) -> list:
        return [
            Rule(pattern = re.compile(pattern = r"(?<![\w-])ke-(?P<num>\d{1,3})(?!\d|[.,]\d)"), handle = self.ordinal),
        ]

    def ordinal(self, m, ctx):
        n = int(m.group("num"))
        if m.group("num").startswith("0"):
            return Skip(reason = f"序数写法无法确定「{m.group()}」")
        ctx.kind(name = "序数", surface = m.group())
        return ctx.num(spoken = "ke" + self.card(n = n), written = m.group())


LANG = Id
