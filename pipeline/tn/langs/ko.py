"""韩语：按后面的量词选汉字词数词（1944년 → 천구백사십사년）或固有词数词（3개 → 세개）。

固有词数词只用于 1–99；6월、10월读作유월、시월。后面既不是登记的量词也不是助词的数字，
读法无法确定，丢弃（例如「3 포인트」）。量词与数词之间的空格保持原样。
"""

from __future__ import annotations

import re

from ..numwords import ko_digits, ko_native_attr, ko_sino
from ..rules import Rule, Skip, alternation
from .base import Lang

# 用固有词数词的量词。번（次数/编号）、대（台/年代）、장（张/章）有两种读法，不收录。
NATIVE_COUNTERS = (
    "개", "명", "마리", "살", "권", "잔", "병", "가지", "채", "척", "켤레", "벌", "그루", "송이",
    "시간", "시", "달", "군데", "곳", "사람", "차례",
)
# 用汉字词数词的量词和单位
SINO_COUNTERS = (
    "년대", "년간", "년째", "년생", "년", "개월", "월", "일간", "일", "층", "분", "초", "원", "회", "호선", "호",
    "도", "량", "위", "등", "학년", "세기", "주년", "주", "점", "배", "퍼센트", "킬로미터", "미터", "센티미터",
    "킬로그램", "그램", "달러", "유로", "엔", "세", "번지", "인", "편", "화", "차", "단계", "기", "부",
)
# 数字后直接接助词时按数值读（汉字词）：「계수는 2이다」→「이이다」
PARTICLES = (
    "이다", "이며", "이고", "이라는", "입니다", "이", "가", "은", "는", "을", "를", "의", "와", "과", "에서", "에",
    "으로", "로", "도", "만", "까지", "부터", "보다",
)
SPECIAL_MONTHS = {6: "유월", 10: "시월"}


class Ko(Lang):
    code = "ko"
    point = "점"
    percent = "퍼센트"
    units = {
        "km²": ("제곱킬로미터", "제곱킬로미터"), "m²": ("제곱미터", "제곱미터"), "km": ("킬로미터", "킬로미터"),
        "cm": ("센티미터", "센티미터"), "mm": ("밀리미터", "밀리미터"), "kg": ("킬로그램", "킬로그램"),
        "m": ("미터", "미터"), "g": ("그램", "그램"), "°C": ("도", "도"), "℃": ("도", "도"),
    }
    currencies = {"$": ("달러", "달러"), "€": ("유로", "유로"), "£": ("파운드", "파운드"), "₩": ("원", "원")}
    scales = ("만", "억")

    def card(self, n: int) -> str:
        return ko_sino(n = n)

    def digits(self, digits: str) -> str:
        return ko_digits(digits = digits)

    def extra_rules(self) -> list:
        native = alternation(words = NATIVE_COUNTERS)
        sino = alternation(words = SINO_COUNTERS)
        particles = alternation(words = PARTICLES)
        return [
            Rule(
                pattern = re.compile(pattern = rf"(?P<num>{self.num})(?P<sp>\s?)(?P<counter>{sino})"),
                handle = self.sino_counter,
            ),
            Rule(
                pattern = re.compile(pattern = rf"(?P<num>{self.num})(?P<sp>\s?)(?P<counter>{native})"),
                handle = self.native_counter,
            ),
            Rule(
                pattern = re.compile(pattern = rf"(?P<num>{self.num})(?=(?:{particles})(?![가-힣]))"),
                handle = self.with_particle,
            ),
        ]

    def sino_counter(self, m, ctx):
        n = self.parse(written = m.group("num"))
        if n is None:
            return Skip(reason = f"数字写法无法确定「{m.group()}」")
        counter = m.group("counter")
        if counter == "월" and n.is_int and n.value in SPECIAL_MONTHS:
            ctx.num(spoken = SPECIAL_MONTHS[n.value], written = m.group())
            ctx.kind(name = "月份", surface = m.group())
            return SPECIAL_MONTHS[n.value]
        spoken = self.say(n = n, ctx = ctx)
        if isinstance(spoken, Skip):
            return spoken
        ctx.kind(name = "汉字词数词", surface = m.group())
        return spoken + m.group("sp") + counter

    def native_counter(self, m, ctx):
        n = self.parse(written = m.group("num"))
        if n is None or not n.is_int or not 1 <= n.value <= 99:
            return Skip(reason = f"固有词数词只处理 1–99 的整数「{m.group()}」")
        ctx.kind(name = "固有词数词", surface = m.group())
        spoken = ctx.num(spoken = ko_native_attr(n = n.value), written = n.written)
        return spoken + m.group("sp") + m.group("counter")

    def with_particle(self, m, ctx):
        n = self.parse(written = m.group("num"))
        if n is None:
            return Skip(reason = f"数字写法无法确定「{m.group()}」")
        ctx.kind(name = "汉字词数词", surface = m.group())
        return self.say(n = n, ctx = ctx)

    def cardinal(self, m, ctx, n):
        return Skip(reason = f"韩语数词读法取决于后面的量词「{m.group()}」")


LANG = Ko
