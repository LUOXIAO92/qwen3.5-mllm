"""日语：数字写成汉字数字（1950 → 千九百五十，4万1千 → 四万一千），量词的读音交给 TTS 前端。"""

from __future__ import annotations

import re

from ..numwords import ja_cardinal, ja_digits
from ..rules import Rule, Skip
from .base import Lang


class Ja(Lang):
    code = "ja"
    point = "点"
    spaced = False
    percent = "パーセント"
    units = {
        "km²": ("平方キロメートル", "平方キロメートル"), "m²": ("平方メートル", "平方メートル"),
        "km": ("キロメートル", "キロメートル"), "cm": ("センチメートル", "センチメートル"),
        "mm": ("ミリメートル", "ミリメートル"), "kg": ("キログラム", "キログラム"), "m": ("メートル", "メートル"),
        "g": ("グラム", "グラム"), "°C": ("度", "度"), "℃": ("度", "度"),
    }
    currencies = {"$": ("ドル", "ドル"), "€": ("ユーロ", "ユーロ"), "£": ("ポンド", "ポンド"), "¥": ("円", "円")}
    scales = ("万", "億")

    def card(self, n: int) -> str:
        return ja_cardinal(n = n)

    def digits(self, digits: str) -> str:
        return ja_digits(digits = digits)

    def cardinal(self, m, ctx, n):
        long_out = self.long_number(m = m, ctx = ctx, n = n)
        return long_out if long_out is not None else self.say(n = n, ctx = ctx)

    def extra_rules(self) -> list:
        return [
            Rule(pattern = re.compile(pattern = r"(?P<h>\d{1,2})[:：](?P<mm>\d{2})(?!\d)"), handle = self.time),
        ]

    def time(self, m, ctx):
        h, mm = int(m.group("h")), int(m.group("mm"))
        if h > 24 or mm > 59:
            return Skip(reason = f"时刻写法无法确定「{m.group()}」")
        ctx.kind(name = "时刻", surface = m.group())
        hour = ctx.num(spoken = self.card(n = h), written = m.group("h"))
        if mm == 0:
            return hour + "時"
        return hour + "時" + ctx.num(spoken = self.card(n = mm), written = m.group("mm")) + "分"


LANG = Ja
