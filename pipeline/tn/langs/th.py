"""泰语：数词不变形，泰文数字与阿拉伯数字都按数值读（ศตวรรษที่ 19 → ศตวรรษที่สิบเก้า）；
佛历年份按原值读。数词与前后文字之间的空格保持原样。「14.30 น.」这类时刻丢弃。
"""

from __future__ import annotations

import re

from ..numwords import th_cardinal, th_digits
from ..rules import Rule, Skip
from .base import Lang


class Th(Lang):
    code = "th"
    point = "จุด"
    spaced = False
    percent = "เปอร์เซ็นต์"
    units = {
        "km/h": ("กิโลเมตรต่อชั่วโมง", "กิโลเมตรต่อชั่วโมง"), "km²": ("ตารางกิโลเมตร", "ตารางกิโลเมตร"),
        "m²": ("ตารางเมตร", "ตารางเมตร"), "km": ("กิโลเมตร", "กิโลเมตร"), "cm": ("เซนติเมตร", "เซนติเมตร"),
        "mm": ("มิลลิเมตร", "มิลลิเมตร"), "kg": ("กิโลกรัม", "กิโลกรัม"), "m": ("เมตร", "เมตร"), "g": ("กรัม", "กรัม"),
        "°C": ("องศาเซลเซียส", "องศาเซลเซียส"), "℃": ("องศาเซลเซียส", "องศาเซลเซียส"),
    }
    currencies = {"$": ("ดอลลาร์", "ดอลลาร์"), "€": ("ยูโร", "ยูโร"), "£": ("ปอนด์", "ปอนด์"), "฿": ("บาท", "บาท")}
    scales = ("ล้าน",)

    def card(self, n: int) -> str:
        return th_cardinal(n = n)

    def digits(self, digits: str) -> str:
        return th_digits(digits = digits)

    def extra_rules(self) -> list:
        return [
            Rule(
                pattern = re.compile(pattern = r"(?:เวลา\s?)?\d{1,2}[.:]\d{2}(?:\s?น\.)?"),
                handle = self.time_like,
            ),
        ]

    def time_like(self, m, ctx):
        text = m.group()
        if text.startswith("เวลา") or text.endswith("น.") or ":" in text:
            return Skip(reason = f"时刻暂不处理「{text}」")
        return None


LANG = Th
