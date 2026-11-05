"""波斯语：数词不随名词变化，各部分用「و」连接（1894 → هزار و هشتصد و نود و چهار）；年份按数值读。

千分位为「٬」或「,」，小数点为「٫」或「.」；波斯数字（۰–۹）与阿拉伯-印度数字（٠–٩）按数值读，「۲/۵」这类
斜线写法由公共规则丢弃。

- 日期中的日读序数（12 مرداد → دوازدهم مرداد）。
- 数字直接接序数后缀：「15ام」「20امین」「3م」读 پانزدهم、بیستمین、سوم。
- 数字直接接 ساله、سالگی（14ساله、90سالگی，也包括用连字符相连的 14-ساله）读 چهارده‌ساله、نود‌سالگی（数词与后缀之间
  用零宽不连字）。数字与后面的词漏了空格（۴هزار、۳ماه）、或 سال、بر、از 等与后面的数字漏了空格（سال۱۹۶۵）时补上
  空格；其余与文字直接相连的数字丢弃。
- 百分号在数字前（٪۱۰，波斯语文本常见的录入顺序）或后都读 ده درصد。
- 时刻「ساعت 2:30」「ساعت 13:00」读 ساعت دو و سی دقیقه、ساعت سیزده；要求前面有 ساعت。
- 不带千分位的 5–6 位数（۱۰۰۰۰ راه）后接文字时按数值读；负数只改写温度（-5 درجه → منفی پنج درجه）。
- 词中用于拉长字形的 tatweel（فـقط）删去；单独的 هـ 等缩写仍丢弃。句号后缺少空格（داد.به）时补上空格。
"""

from __future__ import annotations

import re

from ..numwords import fa_cardinal, fa_digits
from ..rules import DASHES, Rule, Skip, alternation
from .base import Lang
from .extras import adjacent_numeral_rule, common_rules, long_plain_rule, missing_space_rule, negative_rule

ZWNJ = chr(0x200C)
TATWEEL = chr(0x0640)
ARABIC_LETTERS = f"{chr(0x0620)}-{chr(0x063F)}{chr(0x0641)}-{chr(0x064A)}{chr(0x066E)}-{chr(0x06D3)}{chr(0x06FA)}-{chr(0x06FF)}"
# 含 tatweel 的整个词（جميــع、سـعادتـمـند）：词首前不是字母，tatweel 后面还有字母（不含 هـ、بـ 这类缩写和介词）
TATWEEL_WORD = (
    rf"(?<![\w\u0640\u064B-\u0652])(?=[{ARABIC_LETTERS}\u0640\u064B-\u0652]*\u0640+[{ARABIC_LETTERS}])"
    rf"[{ARABIC_LETTERS}\u0640\u064B-\u0652]+"
)
# 写成文字的数词：数字后紧跟这些词时读法有歧义（۲۰۵۰ سه برابر → دو هزار و پنجاه سه）
ADJACENT_NUMERALS = ("یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه", "ده", "صد", "بیست", "سی", "چهل", "پنجاه")
MONTHS = (
    "فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور", "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند",
    "ژانویه", "فوریه", "مارس", "آوریل", "مه", "ژوئن", "ژوئیه", "اوت", "سپتامبر", "اکتبر", "نوامبر", "دسامبر",
)
# 后面紧跟数字（漏了空格）时可以补上空格的词
GLUED_BEFORE = "سال|بر|از|در|تا|با|به|حدود|ساعت|ماه|روز"
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
        return common_rules(lang = self, list_numbers = False, standalone_citation = True) + [
            adjacent_numeral_rule(lang = self, words = ADJACENT_NUMERALS),
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
                pattern = re.compile(pattern = TATWEEL_WORD),
                handle = self.tatweel,
            ),
            long_plain_rule(lang = self),
            missing_space_rule(left = ARABIC_LETTERS, right = rf"[{ARABIC_LETTERS}]{{2,}}"),
            # 数字与后面的词之间漏了空格（۴هزار、۳ماه、۱۲طبقه）：读出数字并补空格
            Rule(
                pattern = re.compile(pattern = rf"(?<![\d.,٫٬\w])(?P<num>{self.num})(?P<word>[{ARABIC_LETTERS}]{{2,}})"),
                handle = self.glued_after,
            ),
            # 常见的介词、名词与后面的数字之间漏了空格（سال۱۹۶۵、بر2）
            Rule(
                pattern = re.compile(pattern = rf"(?<![\w])(?P<word>{GLUED_BEFORE})(?P<num>\d+(?:[٫٬.,]\d+)*)(?![\d\w])"),
                handle = self.glued_before,
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

    def _say_written(self, ctx, written: str):
        n = self.parse(written = written)
        if n is None:
            return Skip(reason = f"数字写法无法确定「{written}」")
        return self.say(n = n, ctx = ctx)

    def glued_after(self, m, ctx):
        spoken = self._say_written(ctx = ctx, written = m.group("num"))
        if isinstance(spoken, Skip):
            return spoken
        ctx.kind(name = "数字后补空格", surface = m.group())
        return spoken + " " + m.group("word")

    def glued_before(self, m, ctx):
        spoken = self._say_written(ctx = ctx, written = m.group("num"))
        if isinstance(spoken, Skip):
            return spoken
        ctx.kind(name = "数字前补空格", surface = m.group())
        return m.group("word") + " " + spoken

    def tatweel(self, m, ctx):
        ctx.kind(name = "删去拉长符", surface = m.group())
        return m.group().replace(TATWEEL, "")


LANG = Fa
