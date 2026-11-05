"""印尼语：数词不变形，基数、年份、日期都按数值读；千分位为「.」，小数点为「,」。

- 「ke-21」读 kedua puluh satu；「1970-an」「70-an」「1980an」读 seribu sembilan ratus tujuh puluhan、tujuh puluhan 等。
- 马来语「16hb Julai」读 enam belas haribulan Julai。
- 时刻：「pukul 17.30」「pukul 2:30」读 pukul tujuh belas tiga puluh、pukul dua tiga puluh，整点只读小时；
  没有 pukul、jam 时，后面要有 pagi、siang、sore、malam 或 WIB 等时区。分钟为 01–09 的写法丢弃。
- 单独的「°」读 derajat；「-10 derajat」读 minus sepuluh derajat。
- 不带千分位的 5–6 位数（10000 orang）后接文字时按数值读。
- hubungi、telepon、nomor 后的数字可能逐位读，丢弃。
- 「Rp 893 M」「Rp 5 jt」读 delapan ratus sembilan puluh tiga miliar rupiah、lima juta rupiah（M = miliar、jt = juta、
  rb = ribu、T = triliun）；金额和价格结尾的「,-」「,00」不读；后面没有金额的「Rp」、不带分隔符的「Jam 1005」丢弃。
- 「RM10 Juta」「$5 juta」读 sepuluh juta ringgit、lima juta dolar：数量级词（大小写不限）读在货币名前。

马来语（ms.py）沿用这里的规则，只换数词和单位的词。
"""

from __future__ import annotations

import re

from ..numwords import idms_cardinal, idms_digits
from ..rules import Rule, Skip, alternation
from .base import ANY_SCALES, Lang, word_after
from .extras import adjacent_numeral_rule, common_rules, long_plain_rule, negative_rule

# 金额中的数量级词及其缩写（Rp 893 M = 893 miliar rupiah）
MONEY_SCALES = {
    "rb": "ribu", "ribu": "ribu", "jt": "juta", "juta": "juta", "M": "miliar", "miliar": "miliar", "milyar": "miliar",
    "T": "triliun", "triliun": "triliun",
}
MONEY_SCALES.update({k.capitalize(): v for k, v in list(MONEY_SCALES.items()) if len(k) > 1})
# 写成文字的数词（不含 ribu、juta 等数量级词）：数字后紧跟这些词时读法有歧义
ADJACENT_NUMERALS = (
    "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "lapan", "sembilan", "sepuluh", "sebelas",
    "seratus", "puluh", "belas", "ratus",
)
# 货币符号后可以出现的数量级词（两种语言的写法都收，读时按原文的词、改成小写）
SCALE_WORDS = "ribu|juta|bilion|billion|miliar|milyar|trilion|triliun"
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
        """共用规则、数字后紧跟数词、负数先于公共保护规则运行（公共规则把 -10 当作负号或连字符丢弃）。"""
        first = common_rules(lang = self) + [adjacent_numeral_rule(lang = self, words = ADJACENT_NUMERALS)]
        if self.minus is not None:
            first.append(negative_rule(lang = self, minus = self.minus, words = ("derajat",)))
        return first + super().rules()

    def extra_rules(self) -> list:
        hours = "|".join(self.hour_words)
        times = "|".join(TIME_WORDS)
        scales = alternation(words = MONEY_SCALES)
        return [
            # 「Jam 1005」「pukul 0930」：不带分隔符的时刻写法不确定
            Rule(
                pattern = re.compile(pattern = rf"\b(?:{hours})\s\d{{3,4}}(?!\d|[.,:]\d)"),
                handle = lambda m, ctx: Skip(reason = f"不带分隔符的时刻写法不确定「{m.group()}」"),
            ),
            # 「Rp 893 M」「Rp5 jt」「Rp 1.000.000,-」：金额缩写和结尾的 ,- ,00
            Rule(
                pattern = re.compile(
                    pattern = rf"(?<![A-Za-z])Rp\.?\s?(?P<num>{self.num_bare})(?:\s?(?P<scale>{scales})(?![\w]))?(?P<tail>,-|,00(?!\d))?",
                ),
                handle = self.rupiah,
            ),
            # 「RM10 Juta」「$5 juta」「RM1.5 bilion」：货币符号后的数量级词（大小写不限）读在货币名前
            Rule(
                pattern = re.compile(
                    pattern = rf"(?<![A-Za-z])(?P<cur>RM|US\$|\$|€)\s?(?P<num>{self.num_bare})\s?(?P<scale>{SCALE_WORDS})(?![\w])",
                    flags = re.IGNORECASE,
                ),
                handle = self.currency_scale,
            ),
            Rule(
                pattern = re.compile(pattern = r"(?<![A-Za-z])Rp\.?(?![\w])"),
                handle = lambda m, ctx: Skip(reason = f"货币符号后没有金额「{m.group()}」"),
            ),
            # 「1.000.000,-」：价格结尾的 ,- 不读
            Rule(pattern = re.compile(pattern = rf"(?P<num>{self.num}),-"), handle = self.price_dash),
            Rule(
                pattern = re.compile(
                    pattern = rf"(?:(?P<word>\b(?:{hours}))\s)?(?<![\d.,:])(?P<h>\d{{1,2}})[.:](?P<m>\d{{2}})(?!\d|[.,:]\d)(?=(?P<after>\s(?:{times})\b)?)",
                ),
                handle = self.time,
            ),
            Rule(
                pattern = re.compile(pattern = r"\b(?:[Hh]ubungi|[Tt]elepon|[Tt]elefon|[Mm]enelepon|[Nn]omor|[Nn]ombor|[Tt]elp\.?)\s(?P<num>\d+)"),
                handle = lambda m, ctx: Skip(reason = f"电话号码、编号的读法（逐位或按数值）不确定「{m.group()}」"),
            ),
            Rule(pattern = re.compile(pattern = r"(?<![\w-])ke-(?P<num>\d{1,3})(?!\d|[.,]\d)"), handle = self.ordinal),
            Rule(pattern = re.compile(pattern = r"(?<![\w.,-])(?P<num>\d{2,4})-?an(?![\w-])"), handle = self.decade),
            Rule(pattern = re.compile(pattern = r"(?<![\w.,])(?P<num>\d{1,2})\s?hb(?![\w])"), handle = self.haribulan),
            long_plain_rule(lang = self),
        ]

    def rupiah(self, m, ctx):
        n = self.parse(written = m.group("num"))
        if n is None:
            return Skip(reason = f"金额写法无法确定「{m.group()}」")
        scale = MONEY_SCALES.get(m.group("scale") or "", "")
        # 「Rp5.000,00」：小数部分全为 0 的是分位，不读
        if n.frac is not None and set(n.frac) == {"0"} and not scale:
            n = self.parse(written = n.int_digits)
        if not n.is_int and not scale:
            return Skip(reason = f"金额的小数部分读法不确定「{m.group()}」")
        after = word_after(text = ctx.text, end = m.end()).lower()
        if not scale and after in {k.lower() for k in MONEY_SCALES} | ANY_SCALES:
            return Skip(reason = f"金额后的数量级词未登记「{m.group()} {after}」")
        spoken = self.say(n = n, ctx = ctx)
        if isinstance(spoken, Skip):
            return spoken
        ctx.kind(name = "货币", surface = m.group())
        # 「Rp180 juta rupiah」：原文已写出 rupiah 时不再加
        if after == "rupiah":
            return self.join(spoken, scale)
        return self.join(spoken, scale, "rupiah")

    def currency_scale(self, m, ctx):
        n = self.parse(written = m.group("num"))
        if n is None:
            return Skip(reason = f"金额写法无法确定「{m.group()}」")
        cur = m.group("cur").upper()
        names = {"RM": "ringgit", "US$": "dolar Amerika", "$": "dolar", "€": "euro"}
        spoken = self.say(n = n, ctx = ctx)
        if isinstance(spoken, Skip):
            return spoken
        name = names[cur]
        # 原文已写出货币名（RM10 juta ringgit）时不再加
        if word_after(text = ctx.text, end = m.end()).lower() == name.split()[0]:
            name = ""
        ctx.kind(name = "货币", surface = m.group())
        return self.join(spoken, m.group("scale").lower(), name)

    def price_dash(self, m, ctx):
        n = self.parse(written = m.group("num"))
        if n is None or not n.is_int:
            return Skip(reason = f"金额写法无法确定「{m.group()}」")
        ctx.kind(name = "基数", surface = m.group())
        return ctx.num(spoken = self.card(n = n.value), written = m.group("num"))

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

    def haribulan(self, m, ctx):
        """马来语「16hb Julai」：hb 是 haribulan 的缩写。"""
        n = int(m.group("num"))
        if self.code != "ms" or not 1 <= n <= 31 or m.group("num").startswith("0"):
            return Skip(reason = f"日期写法无法确定「{m.group()}」")
        ctx.kind(name = "日期", surface = m.group())
        return ctx.num(spoken = self.card(n = n), written = m.group("num")) + " haribulan"

    def decade(self, m, ctx):
        n = int(m.group("num"))
        if m.group("num").startswith("0") or n % 10:
            return Skip(reason = f"年代写法无法确定「{m.group()}」")
        ctx.kind(name = "年代", surface = m.group())
        return ctx.num(spoken = self.card(n = n), written = m.group("num")) + "an"


LANG = Id
