"""土耳其语：数词不变形，分开写（iki bin on）；「4. sınıf」读 dördüncü sınıf；「%25」读 yüzde yirmi beş。

数字后接撇号后缀（5'te、2.30'a）需要元音和谐和辅音交替，由公共规则丢弃。千分位为「.」，小数点为「,」。
"""

from __future__ import annotations

import re

from ..numwords import tr_cardinal, tr_digits, tr_ordinal
from ..rules import Rule, Skip
from .base import Lang


class Tr(Lang):
    code = "tr"
    point = "virgül"
    percent = "yüzde"
    percent_prefix = True
    units = {
        "km²": ("kilometrekare", "kilometrekare"), "m²": ("metrekare", "metrekare"), "km": ("kilometre", "kilometre"),
        "cm": ("santimetre", "santimetre"), "mm": ("milimetre", "milimetre"), "kg": ("kilogram", "kilogram"),
        "m": ("metre", "metre"), "g": ("gram", "gram"), "°C": ("santigrat derece", "santigrat derece"),
        "℃": ("santigrat derece", "santigrat derece"),
    }
    currencies = {
        "$": ("dolar", "dolar"), "£": ("sterlin", "sterlin"), "₺": ("Türk lirası", "Türk lirası"),
        "TL": ("Türk lirası", "Türk lirası"),
    }
    scales = ("bin", "milyon", "milyar")

    def card(self, n: int) -> str:
        return tr_cardinal(n = n)

    def digits(self, digits: str) -> str:
        return tr_digits(digits = digits)

    def extra_rules(self) -> list:
        return [
            Rule(
                pattern = re.compile(pattern = r"(?<![\d.,])(?P<num>\d{1,4})\.(?=\s[a-zçğıöşü])"),
                handle = self.ordinal,
            ),
            Rule(
                pattern = re.compile(pattern = r"(?<![\d.,])\d{1,4}\.(?=\s[A-ZÇĞİÖŞÜ])"),
                handle = lambda m, ctx: Skip(reason = f"数字后的句点可能表示序数（2. Dünya Savaşı）「{m.group()}」"),
            ),
        ]

    def ordinal(self, m, ctx):
        n = int(m.group("num"))
        ctx.kind(name = "序数", surface = m.group())
        return ctx.num(spoken = tr_ordinal(n = n), written = m.group())


LANG = Tr
