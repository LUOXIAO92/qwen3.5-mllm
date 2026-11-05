"""越南语：数词不变形，年份、日期按数值读（năm 2026 → năm hai nghìn không trăm hai mươi sáu）；
「thứ 1」「thứ 4」读 thứ nhất、thứ tư，「tháng 4」读 tháng tư。千分位为「.」，小数点为「,」。

- 时刻：「lúc 6:30 tối」「14:00」读 sáu giờ ba mươi phút、mười bốn giờ；要求前面有 lúc、vào、khoảng、từ、đến
  等词，或后面有 sáng、trưa、chiều、tối、đêm。
- 单位：km、m、cm、mm、kg、g、km/h 按音节读（ki lô mét、mét、xen ti mét、mi li mét、ki lô gam、gam、
  ki lô mét trên giờ），°C 读 độ C；「-10°C」读 âm mười độ C。
- 不带千分位的 5–6 位数（10000 người）后接文字时按数值读；「4.9%」这类以句点作小数点的写法丢弃。
"""

from __future__ import annotations

import re

from ..numwords import vi_cardinal, vi_digits
from ..rules import Rule, Skip
from .base import Lang
from .extras import long_plain_rule, negative_rule

TIME_BEFORE = ("lúc", "vào", "khoảng", "từ", "đến", "tới", "trước", "sau", "Lúc", "Vào", "Khoảng", "Từ", "Đến")
TIME_AFTER = ("sáng", "trưa", "chiều", "tối", "đêm")


class Vi(Lang):
    code = "vi"
    point = "phẩy"
    percent = "phần trăm"
    units = {
        "km/h": ("ki lô mét trên giờ", "ki lô mét trên giờ"), "km": ("ki lô mét", "ki lô mét"),
        "cm": ("xen ti mét", "xen ti mét"), "mm": ("mi li mét", "mi li mét"), "kg": ("ki lô gam", "ki lô gam"),
        "m": ("mét", "mét"), "g": ("gam", "gam"), "°C": ("độ C", "độ C"), "℃": ("độ C", "độ C"),
    }
    currencies = {"$": ("đô la", "đô la"), "€": ("euro", "euro"), "₫": ("đồng", "đồng")}
    scales = ("nghìn", "ngàn", "triệu", "tỷ", "tỉ")

    def card(self, n: int) -> str:
        return vi_cardinal(n = n)

    def digits(self, digits: str) -> str:
        return vi_digits(digits = digits)

    def rules(self) -> list:
        """负温度先于公共保护规则运行（公共规则把 -10 当作负号或连字符丢弃）。"""
        return [negative_rule(lang = self, minus = "âm", words = ("độ",))] + super().rules()

    def extra_rules(self) -> list:
        before = "|".join(TIME_BEFORE)
        after = "|".join(TIME_AFTER)
        return [
            Rule(
                pattern = re.compile(pattern = r"(?P<word>\b[Tt]hứ|\b[Tt]háng)\s(?P<num>\d{1,2})(?!\d|[.,]\d)"),
                handle = self.ordinal_or_month,
            ),
            Rule(
                pattern = re.compile(
                    pattern = rf"(?:(?P<word>\b(?:{before}))\s)?(?<![\d.,:])(?P<h>\d{{1,2}}):(?P<m>\d{{2}})(?!\d|[.,:]\d)(?=(?P<after>\s(?:{after})\b)?)",
                ),
                handle = self.time,
            ),
            long_plain_rule(lang = self),
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

    def time(self, m, ctx):
        h, mm = int(m.group("h")), int(m.group("m"))
        if not (m.group("word") or m.group("after")):
            return Skip(reason = f"冒号写法无法确定是否为时刻「{m.group()}」")
        if not (0 <= h <= 24 and 0 <= mm <= 59):
            return Skip(reason = f"时刻写法无法确定「{m.group()}」")
        spoken = ctx.num(spoken = self.card(n = h), written = m.group("h")) + " giờ"
        if mm:
            spoken += " " + ctx.num(spoken = self.card(n = mm), written = m.group("m")) + " phút"
        ctx.kind(name = "时刻", surface = m.group())
        return (m.group("word") + " " if m.group("word") else "") + spoken


LANG = Vi
