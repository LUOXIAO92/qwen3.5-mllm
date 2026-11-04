"""越南语：数词不变形，年份、日期按数值读（năm 2026 → năm hai nghìn không trăm hai mươi sáu）；
「thứ 1」「thứ 4」读 thứ nhất、thứ tư，「tháng 4」读 tháng tư。千分位为「.」，小数点为「,」。
单位的越南语写法不统一（ki-lô-mét、kilômét、cây số），暂不处理。
"""

from __future__ import annotations

import re

from ..numwords import vi_cardinal, vi_digits
from ..rules import Rule, Skip
from .base import Lang


class Vi(Lang):
    code = "vi"
    point = "phẩy"
    percent = "phần trăm"
    currencies = {"$": ("đô la", "đô la"), "€": ("euro", "euro"), "₫": ("đồng", "đồng")}
    scales = ("nghìn", "ngàn", "triệu", "tỷ", "tỉ")

    def card(self, n: int) -> str:
        return vi_cardinal(n = n)

    def digits(self, digits: str) -> str:
        return vi_digits(digits = digits)

    def extra_rules(self) -> list:
        return [
            Rule(
                pattern = re.compile(pattern = r"(?P<word>\b[Tt]hứ|\b[Tt]háng)\s(?P<num>\d{1,2})(?!\d|[.,]\d)"),
                handle = self.ordinal_or_month,
            ),
        ]

    def ordinal_or_month(self, m, ctx):
        word, n = m.group("word"), int(m.group("num"))
        if m.group("num").startswith("0"):
            return Skip(reason = f"写法无法确定「{m.group()}」")
        is_month = word.lower() == "tháng"
        if is_month and not 1 <= n <= 12:
            return Skip(reason = f"月份写法无法确定「{m.group()}」")
        if n == 4:
            spoken = "tư"
        elif n == 1 and not is_month:
            spoken = "nhất"
        else:
            spoken = self.card(n = n)
        ctx.kind(name = "月份" if is_month else "序数", surface = m.group())
        ctx.num(spoken = word + " " + spoken, written = m.group())
        return word + " " + spoken


LANG = Vi
