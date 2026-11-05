"""韩语：按量词选汉字词数词（1944년 → 천구백사십사년）或固有词数词（3개 → 세개），量词与数词之间的空格保持原样。

- 汉字词数词：年、月（6월、10월读유월、시월）、日、分、秒、层、回、号、位、度、岁（세）、世纪、周、
  韩元与外币、外来语单位（킬로미터、페이지、라운드）、体育计数（승、패、이닝、경기）等；「제」后面的数，
  「1억 5천만」这类带만、억、조、천、백的大数，「100여 종」中「여」前的数，括号里的四位年份，都用汉字词。
- 固有词数词：개、명、마리、살、시간、가지、곳、차례 等，只用于 1–99；1–12 点的「시」；「번째」（1 读
  「첫」）；10 以内的「배」。整百、整千（100명、1,000개）和小数（6.5개）用汉字词，其余三位以上的数丢弃。
- 量词后面紧跟的必须是空格、标点、助词或词尾（개의、명씩、시경）。量词与后面的字连成别的词时
  （개국、시즌、차례 的「차」、명창）不按这个量词处理，避免「46개국」读成「마흔여섯개국」。
- 数字直接接助词（계수는 2이다）时用汉字词，但只在数字前面是句首、标点，或以助词、词尾结尾的词时改写；
  「그림 1에서」「블랙 옵스 7에는」这类名称后的编号可能按外语读，丢弃。
- 其余读法取决于量词而量词不明确的数（3 포인트、2번、10대、3권）丢弃。时刻（14:30）的 24 小时读法不固定，丢弃。
"""

from __future__ import annotations

import re

from ..numwords import ko_digits, ko_native_attr, ko_sino
from ..rules import Rule, Skip, alternation
from . import cjk
from .base import Lang

# 用固有词数词的量词。번（次数/编号）、대（台/代）、장（张/章）、권（本/卷）、점（件/分）、편（篇/部）、
# 기（架/期）、종（种）、건（件）、골（球）随用法换读法，不收录。
NATIVE_COUNTERS = (
    "개", "명", "마리", "살", "잔", "병", "가지", "채", "척", "켤레", "벌", "그루", "송이", "시간", "군데", "곳",
    "사람", "차례", "칸", "마디", "끼", "자루", "그릇", "조각", "바퀴", "걸음", "달",
)
# 用汉字词数词的量词和单位
SINO_COUNTERS = (
    "년대", "년간", "년째", "년생", "년도", "년", "개월", "개국", "개소", "개년", "개항", "개조", "개사", "월", "일간",
    "일", "층", "분기", "분", "초", "원", "회", "호선", "호기", "호실", "호점", "호", "도", "량", "위", "등급", "등", "급", "학년",
    "학기", "세기", "세대", "세", "주년", "주", "번지", "인분", "인조", "인승", "인칭", "인자", "인", "일자", "화", "차전", "차",
    "단계", "단", "부작", "부", "군단", "군", "사단", "여단", "연대", "대대", "중대", "소대", "함대", "대손", "대조",
    "대째", "대륙", "대강", "표", "집", "촌", "축", "심", "평", "품", "교시", "절", "동", "계", "승",
    "패", "무", "타점", "이닝", "실점", "득점", "경기", "연승", "연패", "연속", "강", "루타", "루수", "루", "순위",
    "라운드", "시즌", "퍼센트포인트", "퍼센트", "킬로미터", "센티미터", "밀리미터", "나노미터", "미터", "킬로그램",
    "밀리그램", "그램", "킬로톤", "메가톤", "톤", "리터", "밀리리터", "헥타르", "에이커", "마일", "피트", "인치",
    "야드", "파운드", "비트", "바이트", "킬로바이트", "메가바이트", "기가바이트", "페이지", "광년", "볼트", "와트",
    "킬로와트", "메가와트", "칼로리", "킬로칼로리", "마력", "포인트", "기압", "헤르츠", "달러", "유로", "엔", "위안",
    "프랑", "루블", "페소",
)
# 量词后面可以紧跟的字：助词、词尾的第一个字
ENDING_START = "의이가은는을를에와과도만까부보처씩쯤경께째간뿐밖나마로으였인일입라란랑여때다면전후"
# 汉字词量词后面另外可以紧跟的字（3월말、5회초、3년차、1층짜리、6부대、1년정도、15년형）
SINO_EXTRA_START = "말초중차짜생대정형급"
# 数字直接接助词时按汉字词读
PARTICLES = (
    "이다", "이며", "이고", "이라는", "이라고", "라는", "라고", "입니다", "이", "가", "은", "는", "을", "를", "의",
    "와", "과", "에서", "에게", "에", "으로", "로", "도", "만", "까지", "부터", "보다", "처럼",
)
# 数字前面的词以这些字结尾（助词、词尾）或就是这些词时，数字是数值本身，不是名称后的编号
VALUE_BEFORE_ENDINGS = "는은가이을를에와과고면서기다며도의로"
VALUE_BEFORE_WORDS = ("약", "총", "각", "단", "및", "또는", "평균", "최대", "최소", "무려", "불과", "겨우", "단지", "즉")
SPECIAL_MONTHS = {6: "유월", 10: "시월"}
BIG_UNITS = {"만": 10 ** 4, "억": 10 ** 8, "조": 10 ** 12}
SMALL_UNITS = {"천": 1000, "백": 100}


def boundary(extra: str = "") -> str:
    """量词后面的边界：不是韩文字母，或者是助词、词尾（以及 extra 中的字）的开头。"""
    return rf"(?=[^가-힣]|$|[{ENDING_START}{extra}])"


class Ko(Lang):
    code = "ko"
    point = "점"
    percent = "퍼센트"
    units = {
        "km²": ("제곱킬로미터", "제곱킬로미터"), "m²": ("제곱미터", "제곱미터"), "cm²": ("제곱센티미터", "제곱센티미터"),
        "m³": ("세제곱미터", "세제곱미터"), "km": ("킬로미터", "킬로미터"), "cm": ("센티미터", "센티미터"),
        "mm": ("밀리미터", "밀리미터"), "nm": ("나노미터", "나노미터"), "μm": ("마이크로미터", "마이크로미터"),
        "µm": ("마이크로미터", "마이크로미터"), "kg": ("킬로그램", "킬로그램"), "mg": ("밀리그램", "밀리그램"),
        "m": ("미터", "미터"), "g": ("그램", "그램"), "t": ("톤", "톤"), "ml": ("밀리리터", "밀리리터"),
        "mL": ("밀리리터", "밀리리터"), "L": ("리터", "리터"), "ha": ("헥타르", "헥타르"), "kW": ("킬로와트", "킬로와트"),
        "MW": ("메가와트", "메가와트"), "W": ("와트", "와트"), "kV": ("킬로볼트", "킬로볼트"), "V": ("볼트", "볼트"),
        "Hz": ("헤르츠", "헤르츠"), "kHz": ("킬로헤르츠", "킬로헤르츠"), "MHz": ("메가헤르츠", "메가헤르츠"),
        "GHz": ("기가헤르츠", "기가헤르츠"), "mi": ("마일", "마일"), "ft": ("피트", "피트"), "yd": ("야드", "야드"),
        "lb": ("파운드", "파운드"), "kcal": ("킬로칼로리", "킬로칼로리"), "KB": ("킬로바이트", "킬로바이트"),
        "kB": ("킬로바이트", "킬로바이트"), "MB": ("메가바이트", "메가바이트"), "GB": ("기가바이트", "기가바이트"),
        "TB": ("테라바이트", "테라바이트"), "ms": ("밀리초", "밀리초"), "°C": ("도", "도"), "℃": ("도", "도"),
        "°": ("도", "도"),
    }
    currencies = {"$": ("달러", "달러"), "€": ("유로", "유로"), "£": ("파운드", "파운드"), "₩": ("원", "원")}
    scales = ("만", "억")

    def card(self, n: int) -> str:
        return ko_sino(n = n)

    def digits(self, digits: str) -> str:
        return ko_digits(digits = digits)

    def percent_rules(self) -> list:
        return cjk.fullwidth_percent(rules = super().percent_rules())

    def currency_rules(self) -> list:
        return cjk.guard_currency(rules = super().currency_rules(), names = ("달러", "원", "유로", "파운드", "엔"))

    def extra_rules(self) -> list:
        num = self.num
        native = alternation(words = NATIVE_COUNTERS)
        sino = alternation(words = SINO_COUNTERS)
        particles = alternation(words = PARTICLES)
        unit = r"(?:[천백]?[만억조]|[천백])"
        # 「100여 종」「10여명」：여 后面是空格、标点或量词；「27여신」（여신，女神）不算
        approx_after = alternation(words = NATIVE_COUNTERS + SINO_COUNTERS + ("만", "억", "천", "대", "건", "권", "점", "편"))
        return cjk.guard_rules(num = num) + [
            Rule(
                pattern = re.compile(pattern = r"\d{1,2}[:：]\d{2}"),
                handle = lambda m, ctx: Skip(reason = f"时刻的 24 小时制读法不固定「{m.group()}」"),
            ),
            Rule(pattern = re.compile(pattern = r"(?<=[(（])\d{4}(?=[)）])"), handle = self.bracket_year),
            Rule(pattern = re.compile(pattern = rf"(?<=제)(?P<sp>\s?)(?P<num>{num})"), handle = self.ordinal_je),
            Rule(
                pattern = re.compile(pattern = rf"{num}\s?{unit}(?:\s?{num}\s?{unit})*"),
                handle = self.big_number,
            ),
            Rule(
                pattern = re.compile(pattern = rf"(?P<num>{num})(?=여(?:[^가-힣]|$|(?:{approx_after})))"),
                handle = self.approx,
            ),
            Rule(pattern = re.compile(pattern = rf"(?P<num>{num})(?P<sp>\s?)(?=번째)"), handle = self.ordinal_native),
            Rule(
                pattern = re.compile(pattern = rf"(?P<num>{num})(?P<sp>\s?)(?P<counter>{sino}){boundary(extra = SINO_EXTRA_START)}"),
                handle = self.sino_counter,
            ),
            Rule(
                # 「1남 2녀」：남、녀 后面只接空格、标点或助词，「25 남자」不算
                pattern = re.compile(pattern = rf"(?P<num>{num})(?P<sp>\s?)(?P<counter>남|녀){boundary()}"),
                handle = self.sino_counter,
            ),
            Rule(
                pattern = re.compile(pattern = rf"(?P<num>{num})(?P<sp>\s?)(?P<counter>{native}|시|배){boundary()}"),
                handle = self.native_counter,
            ),
            Rule(
                pattern = re.compile(pattern = rf"(?P<num>{num})(?=(?:{particles})(?:는|도|의|만)?(?![가-힣]))"),
                handle = self.with_particle,
            ),
        ]

    # ------------------------------------------------------------------
    # 汉字词数词
    # ------------------------------------------------------------------

    def sino(self, n, ctx, kind: str, surface: str):
        out = self.say(n = n, ctx = ctx)
        if not isinstance(out, Skip):
            ctx.kind(name = kind, surface = surface)
        return out

    def bracket_year(self, m, ctx):
        """「선조 25년(1592)」：括号里的四位数是年份，用汉字词读。"""
        n = self.parse(written = m.group())
        if n is None or not 1000 <= n.value <= 2099:
            return Skip(reason = f"括号里的四位数读法不确定「{m.group()}」")
        return self.sino(n = n, ctx = ctx, kind = "年份", surface = m.group())

    def ordinal_je(self, m, ctx):
        """「제56회」「제2공화국」：제 后面的序数用汉字词。"""
        n = self.parse(written = m.group("num"))
        if n is None or not n.is_int:
            return Skip(reason = f"数字写法无法确定「제{m.group()}」")
        out = self.sino(n = n, ctx = ctx, kind = "序数", surface = "제" + m.group())
        return out if isinstance(out, Skip) else m.group("sp") + out

    def big_number(self, m, ctx):
        """「1억 5천만」「4만」「2천」：带만、억、조、천、백的数，按韩语的四位分组算出数值，用汉字词读。"""
        parts = re.findall(pattern = rf"({self.num})\s?([천백]?)([만억조]?)", string = m.group())
        total = 0
        small = 0
        for written, small_unit, big_unit in parts:
            n = self.parse(written = written)
            if n is None:
                return Skip(reason = f"数字写法无法确定「{m.group()}」")
            if not n.is_int:
                if len(parts) > 1 or small_unit:
                    return Skip(reason = f"带小数的大数暂不处理「{m.group()}」")
                spoken = self.say(n = n, ctx = ctx)
                ctx.kind(name = "汉字词数词", surface = m.group())
                return spoken + big_unit
            value = n.value * SMALL_UNITS.get(small_unit, 1)
            if not big_unit:
                small += value
                continue
            small += value
            total += small * BIG_UNITS[big_unit]
            small = 0
        total += small
        ctx.kind(name = "汉字词数词", surface = m.group())
        # pipeline.compare 读不懂「1천 6백만」这类天、百后面再接数字的写法，核对读法与算出的数值
        return ctx.num(spoken = ko_sino(n = total), written = str(total))

    def approx(self, m, ctx):
        """「100여 종」「10여명」：여 前面的数用汉字词。"""
        n = self.parse(written = m.group("num"))
        if n is None or not n.is_int:
            return Skip(reason = f"数字写法无法确定「{m.group()}여」")
        return self.sino(n = n, ctx = ctx, kind = "汉字词数词", surface = m.group() + "여")

    def sino_counter(self, m, ctx):
        n = self.parse(written = m.group("num"))
        if n is None:
            return Skip(reason = f"数字写法无法确定「{m.group()}」")
        counter = m.group("counter")
        if counter == "월" and n.is_int and n.value in SPECIAL_MONTHS:
            ctx.num(spoken = SPECIAL_MONTHS[n.value], written = m.group())
            ctx.kind(name = "月份", surface = m.group())
            return SPECIAL_MONTHS[n.value]
        out = self.sino(n = n, ctx = ctx, kind = "汉字词数词", surface = m.group())
        return out if isinstance(out, Skip) else out + m.group("sp") + counter

    # ------------------------------------------------------------------
    # 固有词数词
    # ------------------------------------------------------------------

    def native_or_sino(self, n, ctx, counter: str, surface: str, limit: int = 99):
        """固有词量词前的数：1–limit 用固有词，整百、整千和小数用汉字词，其余丢弃。"""
        if n.is_int and 1 <= n.value <= limit:
            ctx.kind(name = "固有词数词", surface = surface)
            return ctx.num(spoken = ko_native_attr(n = n.value), written = n.written)
        if (not n.is_int) or (n.value >= 100 and n.value % 100 == 0):
            return self.sino(n = n, ctx = ctx, kind = "汉字词数词", surface = surface)
        return Skip(reason = f"「{counter}」前的数用固有词还是汉字词不确定「{surface}」")

    def ordinal_native(self, m, ctx):
        """「1번째」→「첫번째」，「21번째」→「스물한번째」。"""
        n = self.parse(written = m.group("num"))
        if n is None or not n.is_int or n.value == 0:
            return Skip(reason = f"数字写法无法确定「{m.group()}번째」")
        if n.value == 1:
            ctx.kind(name = "固有词数词", surface = m.group() + "번째")
            return ctx.num(spoken = "첫", written = n.written) + m.group("sp")
        out = self.native_or_sino(n = n, ctx = ctx, counter = "번째", surface = m.group() + "번째")
        return out if isinstance(out, Skip) else out + m.group("sp")

    def native_counter(self, m, ctx):
        n = self.parse(written = m.group("num"))
        if n is None:
            return Skip(reason = f"数字写法无法确定「{m.group()}」")
        counter = m.group("counter")
        surface = m.group()
        if counter == "시":
            # 「오후 6시」：1–12 点用固有词；0 点和 13–24 点的读法不固定
            if not (n.is_int and 1 <= n.value <= 12):
                return Skip(reason = f"「시」前的数只处理 1–12「{surface}」")
            out = self.native_or_sino(n = n, ctx = ctx, counter = counter, surface = surface)
        elif counter == "배":
            # 「두 배」「열 배」；「1.5배」「백 배」用汉字词
            out = self.native_or_sino(n = n, ctx = ctx, counter = counter, surface = surface, limit = 10)
        else:
            out = self.native_or_sino(n = n, ctx = ctx, counter = counter, surface = surface)
        return out if isinstance(out, Skip) else out + m.group("sp") + counter

    # ------------------------------------------------------------------
    # 助词前的数
    # ------------------------------------------------------------------

    def value_position(self, text: str, start: int) -> bool:
        """数字是否处在数值的位置：句首、标点后，或前一个词以助词、词尾结尾。"""
        before = text[:start]
        if not before or not (before[-1].isspace() or before[-1].isalnum()):
            return True
        if not before[-1].isspace():
            return False
        words = before.split()
        if not words:
            return True
        word = words[-1]
        return word in VALUE_BEFORE_WORDS or word[-1] in VALUE_BEFORE_ENDINGS or not word[-1].isalnum()

    def with_particle(self, m, ctx):
        n = self.parse(written = m.group("num"))
        if n is None:
            return Skip(reason = f"数字写法无法确定「{m.group()}」")
        if not self.value_position(text = ctx.text, start = m.start()):
            return Skip(reason = f"名称后的数字可能按外语读或是编号「{m.group()}」")
        if n.is_int and not n.grouped and n.value >= 1000:
            return Skip(reason = f"助词前的四位数可能是型号「{m.group()}」")
        return self.sino(n = n, ctx = ctx, kind = "汉字词数词", surface = m.group())

    def cardinal(self, m, ctx, n):
        return Skip(reason = f"韩语数词读法取决于后面的量词「{m.group()}」")


LANG = Ko
