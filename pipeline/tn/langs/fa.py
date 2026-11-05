"""波斯语：数词不随名词变化，各部分用「و」连接（1894 → هزار و هشتصد و نود و چهار）；年份按数值读。

千分位为「٬」或「,」，小数点为「٫」或「.」；波斯数字（۰–۹）与阿拉伯-印度数字（٠–٩）按数值读，「۲/۵」这类
斜线写法由公共规则丢弃。

- 日期中的日读序数（12 مرداد → دوازدهم مرداد）。
- 数字直接接序数后缀：「15ام」「20امین」「3م」读 پانزدهم、بیستمین、سوم。
- 数字直接接 ساله、سالگی（14ساله、90سالگی，也包括用连字符相连的 14-ساله）读 چهارده‌ساله、نود‌سالگی（数词与后缀之间
  用零宽不连字）。其余与文字直接相连的数字丢弃。
- 百分号在数字前（٪۱۰，波斯语文本常见的录入顺序）或后都读 ده درصد。
- 时刻「ساعت 2:30」「ساعت 13:00」读 ساعت دو و سی دقیقه、ساعت سیزده；要求前面有 ساعت。
- 不带千分位的 5–6 位数（۱۰۰۰۰ راه）后接文字时按数值读；负数只改写温度（-5 درجه → منفی پنج درجه）。
- 词中用于拉长字形的 tatweel（فـقط）删去；单独的 هـ 等缩写仍丢弃。
"""

from __future__ import annotations

import re

from ..numwords import fa_cardinal, fa_digits
from ..rules import DASHES, Rule, Skip, alternation
from .base import Lang
from .extras import long_plain_rule, negative_rule

ZWNJ = chr(0x200C)
TATWEEL = chr(0x0640)
ARABIC_LETTERS = f"{chr(0x0620)}-{chr(0x063F)}{chr(0x0641)}-{chr(0x064A)}{chr(0x066E)}-{chr(0x06D3)}{chr(0x06FA)}-{chr(0x06FF)}"
MONTHS = (
    "فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور", "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند",
    "ژانویه", "فوریه", "مارس", "آوریل", "مه", "ژوئن", "ژوئیه", "اوت", "سپتامبر", "اکتبر", "نوامبر", "دسامبر",
)
# 数字后直接相连、读作「数词 + 后缀」的词（14ساله → چهارده‌ساله）
AGE_SUFFIXES = ("سالگی", "ساله")


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

    def rules(self) -> list:
        """负温度、与文字用连字符相连的年龄先于公共保护规则运行。"""
        dash = re.escape(pattern = DASHES)
        ages = alternation(words = AGE_SUFFIXES)
        return [
            negative_rule(lang = self, minus = "منفی", words = ("درجه",)),
            Rule(
                pattern = re.compile(pattern = rf"(?<![\d.,٫٬\w])(?P<num>\d{{1,3}})[{dash}]?(?P<suf>{ages})(?![{ARABIC_LETTERS}])"),
                handle = self.age,
            ),
        ] + super().rules()

    def extra_rules(self) -> list:
        months = alternation(words = MONTHS)
        return [
            Rule(
                pattern = re.compile(pattern = rf"(?<![\d.,٫٬])(?P<num>\d{{1,2}})\s(?P<month>{months})(?![{ARABIC_LETTERS}])"),
                handle = self.date,
            ),
            Rule(
                pattern = re.compile(pattern = rf"(?<![\d.,٫٬\w])(?P<num>\d{{1,3}})(?P<suf>امین|مین|ام|م)(?![{ARABIC_LETTERS}])"),
                handle = self.ordinal,
            ),
            Rule(
                pattern = re.compile(pattern = rf"(?<![\w.,٫٬])[%٪]\s?(?P<num>{self.num})"),
                handle = self.percent_before,
            ),
            Rule(
                pattern = re.compile(pattern = r"(?P<word>ساعت\s)(?P<h>\d{1,2}):(?P<m>\d{2})(?!\d|[.,:٫]\d)"),
                handle = self.time,
            ),
            Rule(
                pattern = re.compile(pattern = rf"[{ARABIC_LETTERS}]+{TATWEEL}+[{ARABIC_LETTERS}]+"),
                handle = self.tatweel,
            ),
            long_plain_rule(lang = self),
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

    def ordinal(self, m, ctx):
        n = int(m.group("num"))
        if n == 0 or m.group("num").startswith("0"):
            return Skip(reason = f"序数写法无法确定「{m.group()}」")
        spoken = fa_ordinal(n = n)
        if m.group("suf") in ("امین", "مین"):
            spoken += "ین"
        ctx.kind(name = "序数", surface = m.group())
        return ctx.num(spoken = spoken, written = m.group("num"))

    def age(self, m, ctx):
        n = int(m.group("num"))
        if n == 0 or m.group("num").startswith("0"):
            return Skip(reason = f"写法无法确定「{m.group()}」")
        ctx.kind(name = "基数", surface = m.group())
        return ctx.num(spoken = self.card(n = n), written = m.group("num")) + ZWNJ + m.group("suf")

    def percent_before(self, m, ctx):
        n = self.parse(written = m.group("num"))
        if n is None:
            return Skip(reason = f"数字写法无法确定「{m.group()}」")
        spoken = self.say(n = n, ctx = ctx)
        if isinstance(spoken, Skip):
            return spoken
        ctx.kind(name = "百分数", surface = m.group())
        return spoken + " " + self.percent

    def time(self, m, ctx):
        h, mm = int(m.group("h")), int(m.group("m"))
        if not (0 <= h <= 24 and 0 <= mm <= 59):
            return Skip(reason = f"时刻写法无法确定「{m.group()}」")
        spoken = ctx.num(spoken = self.card(n = h), written = m.group("h"))
        if mm:
            spoken += " و " + ctx.num(spoken = self.card(n = mm), written = m.group("m")) + " دقیقه"
        ctx.kind(name = "时刻", surface = m.group())
        return m.group("word") + spoken

    def tatweel(self, m, ctx):
        ctx.kind(name = "删去拉长符", surface = m.group())
        return m.group().replace(TATWEEL, "")


LANG = Fa
