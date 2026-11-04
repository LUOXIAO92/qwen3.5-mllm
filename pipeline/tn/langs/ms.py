"""马来语（马来西亚）：规则同印尼语，数词 8 为 lapan、10^9 为 bilion，小数点读 perpuluhan；
千分位与小数点和英语相同（1,000.5）。年份按完整数值读（seribu sembilan ratus enam puluh tujuh）。
"""

from __future__ import annotations

import re

from ..rules import Rule, Skip
from .id import Id


class Ms(Id):
    code = "ms"
    point = "perpuluhan"
    percent = "peratus"
    units = {
        "km/h": ("kilometer sejam", "kilometer sejam"), "km²": ("kilometer persegi", "kilometer persegi"),
        "m²": ("meter persegi", "meter persegi"), "km": ("kilometer", "kilometer"), "cm": ("sentimeter", "sentimeter"),
        "mm": ("milimeter", "milimeter"), "kg": ("kilogram", "kilogram"), "m": ("meter", "meter"),
        "g": ("gram", "gram"), "°C": ("darjah Celsius", "darjah Celsius"), "℃": ("darjah Celsius", "darjah Celsius"),
        "°F": ("darjah Fahrenheit", "darjah Fahrenheit"),
    }
    currencies = {"$": ("dolar", "dolar"), "€": ("euro", "euro"), "RM": ("ringgit", "ringgit")}
    scales = ("ribu", "juta", "bilion", "trilion")

    def extra_rules(self) -> list:
        """「pukul 7.30」是时刻，按小数读会出错，丢弃。"""
        return [
            Rule(
                pattern = re.compile(pattern = r"\b(?:pukul|jam|Pukul|Jam)\s\d{1,2}\.\d{2}(?!\d)"),
                handle = lambda m, ctx: Skip(reason = f"时刻写法（7.30）「{m.group()}」"),
            ),
        ] + super().extra_rules()


LANG = Ms
