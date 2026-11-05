"""泰语：数词不变形，泰文数字与阿拉伯数字都按数值读（ศตวรรษที่ 19 → ศตวรรษที่สิบเก้า）；
佛历年份按原值读。数词与前后文字之间的空格保持原样。

- 时刻：「14.30 น.」「เวลา 8.00」「14:30」读 สิบสี่นาฬิกาสามสิบนาที、แปดนาฬิกา（24 小时制的正式读法），
  「น.」是 นาฬิกา 的缩写，并入读法。没有 เวลา、น. 的「14.30」按小数读。
- 纪年：「พ.ศ. 2500」「ค.ศ.1993」读 พุทธศักราช、คริสต์ศักราช 加年份。
- 金额：「99.50 บาท」读 เก้าสิบเก้าบาทห้าสิบสตางค์。
- 「-10 องศา」读 ลบสิบองศา。
- 句末误加的句点（ได้. 、ครับ. ）：泰语不用句点断句，句点前的词至少三个字符、末三个字符中有元音符号、后面是
  空白或文末时删去句点；单个辅音加句点（จอมพล ป.）、末尾不含元音符号的缩写（รพ.、ดร.、กสพท.）和字母名（เอส.）
  仍按缩写丢弃。
"""

from __future__ import annotations

import re

from ..numwords import th_cardinal, th_digits
from ..rules import Rule, Skip
from .base import Lang
from .extras import negative_rule

THAI = f"{chr(0x0E01)}-{chr(0x0E4E)}"
# 元音符号与声调符号（不含辅音）
THAI_MARKS = set(chr(c) for c in list(range(0x0E30, 0x0E3B)) + list(range(0x0E40, 0x0E45)) + list(range(0x0E47, 0x0E4F)))
LETTER_NAMES = {
    "เอ", "บี", "ซี", "ดี", "อี", "เอฟ", "จี", "เอช", "ไอ", "เจ", "เค", "แอล", "เอ็ม", "เอ็น", "โอ", "พี", "คิว",
    "อาร์", "เอส", "ที", "ยู", "วี", "ดับเบิลยู", "เอ็กซ์", "วาย", "แซด",
}
ERAS = {"พ.ศ.": "พุทธศักราช", "ค.ศ.": "คริสต์ศักราช"}


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

    def rules(self) -> list:
        """负温度和纪年先于公共保护规则运行（公共规则把 -10、ค.ศ.1993 丢弃）。"""
        eras = "|".join(re.escape(pattern = e) for e in ERAS)
        era = Rule(
            pattern = re.compile(pattern = rf"(?P<era>{eras})(?P<space>\s?)(?P<num>\d{{3,4}})(?![\d.,])"),
            handle = self.era_year,
        )
        return [negative_rule(lang = self, minus = "ลบ", words = ("องศา",)), era] + super().rules()

    def extra_rules(self) -> list:
        return [
            Rule(
                pattern = re.compile(
                    pattern = r"(?P<word>เวลา\s?)?(?<![\d.,:])(?P<h>\d{1,2})[.:](?P<m>\d{2})(?!\d|[.,:]\d)(?P<abbr>\s?น\.)?",
                ),
                handle = self.time,
            ),
            Rule(
                pattern = re.compile(pattern = r"(?<![\d.,])(?P<baht>\d{1,3}(?:,\d{3})*|\d+)\.(?P<satang>\d{2})(?P<space>\s?)บาท"),
                handle = self.baht,
            ),
            Rule(
                pattern = re.compile(pattern = rf"(?P<word>[{THAI}]+)\.(?=\s|$)"),
                handle = self.sentence_period,
            ),
        ]

    def time(self, m, ctx):
        h, mm = int(m.group("h")), int(m.group("m"))
        if not (m.group("word") or m.group("abbr") or ":" in m.group()):
            return None
        if not (0 <= h <= 24 and 0 <= mm <= 59):
            return Skip(reason = f"时刻写法无法确定「{m.group().strip()}」")
        spoken = ctx.num(spoken = self.card(n = h), written = m.group("h")) + "นาฬิกา"
        if mm:
            spoken += ctx.num(spoken = self.card(n = mm), written = m.group("m")) + "นาที"
        ctx.kind(name = "时刻", surface = m.group().strip())
        return (m.group("word") or "") + spoken

    def era_year(self, m, ctx):
        ctx.kind(name = "年份", surface = m.group())
        year = ctx.num(spoken = self.card(n = int(m.group("num"))), written = m.group("num"))
        return ERAS[m.group("era")] + m.group("space") + year

    def baht(self, m, ctx):
        n = self.parse(written = m.group("baht"))
        if n is None:
            return Skip(reason = f"金额写法无法确定「{m.group()}」")
        baht = ctx.num(spoken = self.card(n = n.value), written = m.group("baht"))
        satang = int(m.group("satang"))
        out = baht + "บาท"
        if satang:
            out += ctx.num(spoken = self.card(n = satang), written = m.group("satang")) + "สตางค์"
        ctx.kind(name = "货币", surface = m.group())
        return out

    def sentence_period(self, m, ctx):
        word = m.group("word")
        start = m.start()
        if start > 0 and not ctx.text[start - 1].isspace():
            return None
        # 句点前最后一个音节（末三个字符）不含元音符号时可能是缩写（ดร.、นพ.、กสพท.）
        if len(word) < 3 or word in LETTER_NAMES or not (set(word[-3:]) & THAI_MARKS):
            return None
        ctx.kind(name = "句末句点", surface = word[-1] + ".")
        return word


LANG = Th
