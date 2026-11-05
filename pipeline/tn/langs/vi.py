"""越南语：数词不变形，年份、日期按数值读（năm 2026 → năm hai nghìn không trăm hai mươi sáu）；
「thứ 1」「thứ 4」读 thứ nhất、thứ tư，「tháng 4」读 tháng tư。千分位为「.」，小数点为「,」。

- 时刻：「lúc 6:30 tối」「14:00」读 sáu giờ ba mươi phút、mười bốn giờ；要求前面有 lúc、vào、khoảng、từ、đến
  等词，或后面有 sáng、trưa、chiều、tối、đêm。
- 单位：km、m、cm、mm、kg、g、km/h 按音节读（ki lô mét、mét、xen ti mét、mi li mét、ki lô gam、gam、
  ki lô mét trên giờ），°C 读 độ C；「-10°C」读 âm mười độ C。
- 不带千分位的 5–6 位数（10000 người）后接文字时按数值读；「4.9%」这类以句点作小数点的写法丢弃。
- 句号后缺少空格（rồi.Tạm）时补上空格。
- 斜线日期：ngày、hôm、sáng、chiều 等之后的「24/9」「11/03/2020」读 hai mươi tư tháng chín（năm …），
  tháng 之后的「1/2020」读 tháng một năm hai nghìn không trăm hai mươi；「6h」「14h30」读 sáu giờ、
  mười bốn giờ ba mươi phút。ha、ml、mg 读 héc ta、mi li lít、mi li gam。
"""

from __future__ import annotations

import re

from ..numwords import vi_cardinal, vi_digits
from ..rules import Rule, Skip
from .base import Lang
from .extras import long_plain_rule, missing_space_rule, negative_rule

LOWER = "a-zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
UPPER = LOWER.upper()
TIME_BEFORE = ("lúc", "vào", "khoảng", "từ", "đến", "tới", "trước", "sau", "Lúc", "Vào", "Khoảng", "Từ", "Đến")
TIME_AFTER = ("sáng", "trưa", "chiều", "tối", "đêm")
# 斜线日期前面的词：日/月（/年）
DAY_BEFORE = ("ngày", "hôm", "sáng", "trưa", "chiều", "tối", "đêm", "rạng sáng", "Ngày", "Hôm", "Sáng", "Trưa", "Chiều",
              "Tối", "Đêm", "Rạng sáng")


class Vi(Lang):
    code = "vi"
    point = "phẩy"
    # 小数部分为 1–2 位时按整数读（3,14 → ba phẩy mười bốn）
    frac_as_number = True
    percent = "phần trăm"
    units = {
        "km/h": ("ki lô mét trên giờ", "ki lô mét trên giờ"), "km": ("ki lô mét", "ki lô mét"),
        "cm": ("xen ti mét", "xen ti mét"), "mm": ("mi li mét", "mi li mét"), "kg": ("ki lô gam", "ki lô gam"),
        "m": ("mét", "mét"), "g": ("gam", "gam"), "°C": ("độ C", "độ C"), "℃": ("độ C", "độ C"),
        "ha": ("héc ta", "héc ta"), "ml": ("mi li lít", "mi li lít"), "mg": ("mi li gam", "mi li gam"),
    }
    currencies = {"$": ("đô la", "đô la"), "€": ("euro", "euro"), "₫": ("đồng", "đồng")}
    scales = ("nghìn", "ngàn", "triệu", "tỷ", "tỉ")

    def card(self, n: int) -> str:
        return vi_cardinal(n = n)

    def digits(self, digits: str) -> str:
        return vi_digits(digits = digits)

    def rules(self) -> list:
        """负温度、斜线日期先于公共保护规则运行（公共规则把 -10、24/9 丢弃）。"""
        days = "|".join(DAY_BEFORE)
        return [
            negative_rule(lang = self, minus = "âm", words = ("độ",)),
            Rule(
                pattern = re.compile(pattern = rf"(?<!\w)(?P<word>(?:{days})\s)(?P<d>\d{{1,2}})/(?P<m>\d{{1,2}})(?:/(?P<y>\d{{4}}))?(?![\d/])"),
                handle = self.slash_date,
            ),
            Rule(
                pattern = re.compile(pattern = r"(?<!\w)(?P<word>[Tt]háng\s)(?P<m>\d{1,2})/(?P<y>\d{4})(?![\d/])"),
                handle = self.slash_month,
            ),
        ] + super().rules()

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
            Rule(
                pattern = re.compile(pattern = r"(?<![\w.,:])(?P<h>\d{1,2})h(?P<m>\d{2})?(?![\w])"),
                handle = self.h_time,
            ),
            missing_space_rule(left = LOWER, right = rf"[{UPPER}][{LOWER}]+"),
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

    def _month(self, written: str) -> str | None:
        n = int(written)
        if not 1 <= n <= 12:
            return None
        return "tư" if n == 4 else self.card(n = n)

    def slash_date(self, m, ctx):
        day, month = int(m.group("d")), self._month(written = m.group("m"))
        if not 1 <= day <= 31 or month is None:
            return Skip(reason = f"日期写法无法确定「{m.group().strip()}」")
        out = m.group("word") + ctx.num(spoken = self.card(n = day), written = m.group("d"))
        out += " tháng " + ctx.num(spoken = month, written = m.group("m"))
        if m.group("y"):
            out += " năm " + ctx.num(spoken = self.card(n = int(m.group("y"))), written = m.group("y"))
        ctx.kind(name = "日期", surface = m.group().strip())
        return out

    def slash_month(self, m, ctx):
        month = self._month(written = m.group("m"))
        if month is None:
            return Skip(reason = f"日期写法无法确定「{m.group().strip()}」")
        ctx.kind(name = "日期", surface = m.group().strip())
        return (m.group("word") + ctx.num(spoken = month, written = m.group("m")) + " năm "
                + ctx.num(spoken = self.card(n = int(m.group("y"))), written = m.group("y")))

    def h_time(self, m, ctx):
        h, mm = int(m.group("h")), int(m.group("m") or 0)
        if not (0 <= h <= 24 and 0 <= mm <= 59):
            return Skip(reason = f"时刻写法无法确定「{m.group()}」")
        spoken = ctx.num(spoken = self.card(n = h), written = m.group("h")) + " giờ"
        if mm:
            spoken += " " + ctx.num(spoken = self.card(n = mm), written = m.group("m")) + " phút"
        ctx.kind(name = "时刻", surface = m.group())
        return spoken

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
