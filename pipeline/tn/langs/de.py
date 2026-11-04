"""德语：基数、年份（neunzehnhundertvierundzwanzig）、小数、百分数、单位、货币和「14:30 Uhr」。

德语数词只有「1」随名词变格（ein、eine、einen、einem），值为 1 或末两位为 01 的数丢弃；序数和日期
（3. Mai、am 3.）要变格，也丢弃。1100–1999 不带千分位的四位数，后面是大写名词（1500 Menschen）或
前面是表示数量的词（mit、über、rund 等）时可能是数量，丢弃；其余按年份读。2000 以上的年份与基数读法相同。
"""

from __future__ import annotations

import re

from ..numwords import n2w
from ..rules import Rule, Skip, alternation
from .base import Lang, word_after, word_before

MONTHS = (
    "Januar", "Jänner", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober",
    "November", "Dezember",
)
QUANTITY_BEFORE = {
    "mit", "über", "unter", "rund", "etwa", "circa", "fast", "knapp", "als", "insgesamt", "auf", "zu", "für",
    "gegen", "nur", "je", "pro", "mehr", "weniger", "die", "der", "den", "dem", "des", "das", "ein", "eine",
    "einer", "einem", "einen", "diese", "dieser", "alle", "allen", "sämtliche",
}


def needs_inflection(n: int) -> bool:
    """1 和末两位为 01 的数在名词前要变格（ein Mann、hundertein Jahre）。"""
    return n == 1 or n % 100 == 1


class De(Lang):
    code = "de"
    point = "Komma"
    percent = "Prozent"
    units = {
        "km/h": ("Kilometer pro Stunde", "Kilometer pro Stunde"), "km²": ("Quadratkilometer", "Quadratkilometer"),
        "m²": ("Quadratmeter", "Quadratmeter"), "km": ("Kilometer", "Kilometer"), "cm": ("Zentimeter", "Zentimeter"),
        "mm": ("Millimeter", "Millimeter"), "kg": ("Kilogramm", "Kilogramm"), "m": ("Meter", "Meter"),
        "g": ("Gramm", "Gramm"), "°C": ("Grad Celsius", "Grad Celsius"), "℃": ("Grad Celsius", "Grad Celsius"),
        "°F": ("Grad Fahrenheit", "Grad Fahrenheit"), "℉": ("Grad Fahrenheit", "Grad Fahrenheit"),
    }
    currencies = {"€": ("Euro", "Euro"), "$": ("Dollar", "Dollar"), "£": ("Pfund", "Pfund")}

    def card(self, n: int) -> str:
        return n2w(n = n, lang = "de")

    def digits(self, digits: str) -> str:
        return " ".join(n2w(n = int(d), lang = "de") for d in digits)

    def unit_number(self, m, ctx, n):
        if n.is_int and needs_inflection(n = n.value):
            return Skip(reason = f"「1」要随格变化「{m.group()}」")
        return self.say(n = n, ctx = ctx)

    currency_number = unit_number

    def extra_rules(self) -> list:
        months = alternation(words = MONTHS)
        return [
            Rule(
                pattern = re.compile(pattern = rf"(?<![\d.,])\d{{1,2}}\.\s?(?:{months})\b"),
                handle = lambda m, ctx: Skip(reason = f"日期中的序数要变格「{m.group()}」"),
            ),
            Rule(pattern = re.compile(pattern = r"(?<![\d.,])(?P<num>\d{1,4})\.(?=\s*\S)"), handle = self.dotted),
            Rule(
                pattern = re.compile(pattern = r"(?<![\d.,])(?P<h>\d{1,2}):(?P<mm>\d{2})\s?Uhr\b"),
                handle = self.time,
            ),
            Rule(pattern = re.compile(pattern = r"(?<![\d.,])(?P<num>1[1-9]\d{2})(?![\d]|[.,]\d)"), handle = self.year),
        ]

    def dotted(self, m, ctx):
        """数字后有句点且后面还有文字：四位数年份视为句末，其余可能是序数（3. Liga），丢弃。"""
        rest = ctx.text[m.end():].lstrip()
        if len(m.group("num")) == 4 and rest[:1].isupper():
            return None
        return Skip(reason = f"数字后的句点可能表示序数「{m.group()}」")

    def time(self, m, ctx):
        h, mm = int(m.group("h")), int(m.group("mm"))
        if h > 24 or mm > 59:
            return Skip(reason = f"时刻写法无法确定「{m.group()}」")
        ctx.kind(name = "时刻", surface = m.group())
        hour = "ein" if h == 1 else self.card(n = h)
        out = ctx.num(spoken = hour, written = m.group("h")) + " Uhr"
        if mm:
            out += " " + ctx.num(spoken = self.card(n = mm), written = m.group("mm"))
        return out

    def year(self, m, ctx):
        before = word_before(text = ctx.text, start = m.start())
        after = word_after(text = ctx.text, end = m.end())
        if before in QUANTITY_BEFORE or (after[:1].isupper() and after not in MONTHS):
            return Skip(reason = f"四位数是年份还是数量需要上下文「{m.group()}」")
        ctx.kind(name = "年份", surface = m.group())
        return ctx.num(spoken = n2w(n = int(m.group("num")), lang = "de", to = "year"), written = m.group())

    def cardinal(self, m, ctx, n):
        if n.is_int and needs_inflection(n = n.value):
            return Skip(reason = f"「1」要随名词变格「{m.group()}」")
        if n.is_int and not n.grouped and 1100 <= n.value <= 1999:
            return Skip(reason = f"四位数是年份还是数量需要上下文「{m.group()}」")
        return self.say(n = n, ctx = ctx)


LANG = De
