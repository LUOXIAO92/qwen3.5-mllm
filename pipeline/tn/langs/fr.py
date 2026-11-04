"""法语：基数、年份、「1er octobre 2023」日期、序数（1er、2e）、小数（virgule）、百分数、单位、货币（€、$）。

以 1 结尾的数（un/une、vingt et un/une、quatre-vingt-un/une）只在确定为阳性时改写，onze、
soixante et onze、quatre-vingt-onze 不变。年份中的 cent、quatre-vingt 不加 s。英镑（la livre）为阴性，不收录。
"""

from __future__ import annotations

import re

from ..numwords import n2w
from ..rules import Rule, Skip
from .romance import Romance


class Fr(Romance):
    code = "fr"
    point = "virgule"
    frac_as_number = True
    percent = "pour cent"
    units = {
        "km/h": ("kilomètre par heure", "kilomètres par heure"), "km²": ("kilomètre carré", "kilomètres carrés"),
        "m²": ("mètre carré", "mètres carrés"), "km": ("kilomètre", "kilomètres"), "cm": ("centimètre", "centimètres"),
        "mm": ("millimètre", "millimètres"), "kg": ("kilogramme", "kilogrammes"), "m": ("mètre", "mètres"),
        "g": ("gramme", "grammes"), "°C": ("degré Celsius", "degrés Celsius"), "℃": ("degré Celsius", "degrés Celsius"),
        "°F": ("degré Fahrenheit", "degrés Fahrenheit"), "℉": ("degré Fahrenheit", "degrés Fahrenheit"),
    }
    currencies = {"€": ("euro", "euros"), "$": ("dollar", "dollars")}
    months = (
        "janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre",
        "décembre",
    )
    year_before = {
        "en", "de", "depuis", "dès", "vers", "jusqu'en", "avant", "après", "année", "an", "fin", "début",
        "printemps", "été", "automne", "hiver", "entre", "et", "à", "l'an", "l'année", "mi", "années",
    }
    masc_after = {
        "ans", "jours", "mois", "siècles", "degrés", "kilomètres", "mètres", "points", "hommes", "habitants",
        "euros", "dollars", "millions", "milliards", "pays", "membres", "joueurs", "sièges", "étudiants",
    }
    first_day = "premier"

    def card(self, n: int) -> str:
        return n2w(n = n, lang = "fr")

    def digits(self, digits: str) -> str:
        return " ".join(n2w(n = int(d), lang = "fr") for d in digits)

    def year_words(self, n: int) -> str:
        """年份中的 cent、quatre-vingt 不加 s（en mille neuf cent、les années mille neuf cent quatre-vingt）。"""
        return re.sub(pattern = r"\b(cent|vingt)s\b", repl = r"\1", string = self.card(n = n))

    def plural(self, n) -> bool:
        """法语从 2 起用复数：1,5 kilomètre。"""
        return n.value >= 2

    def gender_sensitive(self, n: int) -> bool:
        for group in (n % 1000, n // 1000 % 1000):
            if group != 1 and group % 10 == 1 and group % 100 not in (11, 71, 91):
                return True
        return n % 1000 == 1

    def ordinal_rules(self) -> list:
        return [
            Rule(
                pattern = re.compile(pattern = r"(?<![\d.,])(?P<num>\d{1,4})(?P<suf>er|re|ère|e|ème|è)(?![\w])"),
                handle = self.ordinal,
            ),
        ]

    def ordinal(self, m, ctx):
        n = int(m.group("num"))
        suf = m.group("suf")
        if n == 1 and suf == "er":
            spoken = "premier"
        elif n == 1 and suf in ("re", "ère"):
            spoken = "première"
        elif n >= 2 and suf in ("e", "ème", "è"):
            spoken = n2w(n = n, lang = "fr", to = "ordinal")
        else:
            return Skip(reason = f"序数写法无法确定「{m.group()}」")
        ctx.kind(name = "序数", surface = m.group())
        return ctx.num(spoken = spoken, written = m.group())


LANG = Fr
