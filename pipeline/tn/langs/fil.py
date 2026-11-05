"""菲律宾语：数量、年份、序数和百分数用他加禄语数词，「alas 6」这类整点用西班牙语来源的数词。

依据：FLEURS 菲律宾语（fil_ph）开发集和测试集中原文含数字的 212 条录音，用 Qwen3-ASR 与 Whisper 转写后
对照原文（T04c）。朗读者把年份读作 isang libo siyam na raan at animnapu't anim（1966）、dalawang libo at lima
（2005），数量读作 labing-isang taong gulang（11）、sampung ahente（10）、apat na raan at siyam（409）、
sampung libong taon（10,000），序数读作 ikalabingwalong siglo（ika-18），百分数读作 walong porsiyento，没有出现
英语或西班牙语来源的读法。整点「alas 6」中的 alas 来自西班牙语，后面只能接西班牙语数词（alas sais）。

- 基数：1–10 isa … sampu，11–19 labing-isa、labindalawa … labinsiyam，整十 dalawampu … siyamnapu，
  21 dalawampu't isa；百位 isang daan、apat na raan，千位 isang libo，百万 isang milyon；最后不足一百的部分前
  加 at（isang libo siyam na raan at animnapu't anim）。
- 连接成分：数词修饰后面的名词时加 -ng（以元音或 n 结尾，sampung ahente、isang daang taon）或 na（以其他辅音
  结尾，apat na libro）。原文已写出 na 时保留；后面是 sa、ng、at 等虚词、标点，或前面是 noong、taong、月份等
  年份语境时不加。
- 序数：「ika-18」读 ikalabingwalo，「ika-2」「ika-3」读 ikalawa、ikatlo，后接名词时同样加连接成分；ika-1 丢弃。
- 百分数：「70%」读 pitumpung porsiyento。
- 缩写「atbp.」读 at iba pa（FLEURS 朗读者同样这样读）。
- 行首的经节号（6 Nang magkagayo'y…）和列表序号（3. Una）删去，句中的丢弃；他加禄语编号用词之后（Ruta Bilang 66、
  Kabanata 3）不加连接成分；前面是英语词、英语月份或其他名称（my number 1、Windows 11、Volume 1、January 15、
  Defenders 2），或后面是 AM、PM 时，读法无法确定，丢弃。
- 丢弃：小数、时刻（2:30）、AD/BC 纪年、0、序数 100 以上。
"""

from __future__ import annotations

import re

from ..rules import Number, Rule, Skip
from .base import Lang
from .extras import common_rules

UNITS = ["", "isa", "dalawa", "tatlo", "apat", "lima", "anim", "pito", "walo", "siyam"]
TEENS = {
    10: "sampu", 11: "labing-isa", 12: "labindalawa", 13: "labintatlo", 14: "labing-apat", 15: "labinlima",
    16: "labing-anim", 17: "labimpito", 18: "labingwalo", 19: "labinsiyam",
}
TENS = ["", "", "dalawampu", "tatlumpu", "apatnapu", "limampu", "animnapu", "pitumpu", "walumpu", "siyamnapu"]
SPANISH_HOURS = {
    1: "una", 2: "dos", 3: "tres", 4: "kuwatro", 5: "singko", 6: "sais", 7: "siyete", 8: "otso", 9: "nuwebe",
    10: "diyes", 11: "onse", 12: "dose",
}
MONTHS = (
    "Enero", "Pebrero", "Marso", "Abril", "Mayo", "Hunyo", "Hulyo", "Agosto", "Setyembre", "Oktubre", "Nobyembre",
    "Disyembre", "Septiyembre", "Setiyembre", "Febrero",
)
# 前面是这些词时，数字是年份，读法后不加连接成分
YEAR_BEFORE = {"noong", "Noong", "taong", "Taong", "dekada", "dekadang", "siglo", "siglong"} | set(MONTHS)
# 后面是这些词时，数词后不加连接成分
FUNCTION_WORDS = {
    "sa", "ng", "nang", "at", "ay", "o", "na", "ang", "si", "ni", "kay", "para", "mula", "hanggang", "noong", "pa",
    "lang", "lamang", "din", "rin", "ba", "man", "naman", "po", "ho", "kung", "kapag", "nga", "daw", "raw", "pala",
    "yata", "ko", "mo", "niya", "namin", "natin", "nila", "ninyo", "siya", "ako", "ikaw", "kami", "tayo", "sila",
    "kayo", "ito", "iyan", "iyon", "dito", "diyan", "doon", "upang", "dahil", "kaya", "pero", "ngunit", "subalit",
    "hindi", "ayon", "bago", "habang", "gaya", "tulad", "kaysa", "nga", "pa'y", "na'y", "ba't", "lahat",
}
# 行首经节号后面常见的词（6 Nang magkagayo'y…、2 Nguni't…）：这时行首的数字是节号，删去
VERSE_NEXT = FUNCTION_WORDS | {"nguni't", "sapagka't", "datapuwa't", "kaya't", "at", "ang", "sa", "nang", "ngunit"}
# 前面是这些英语词或编号用词时，数字的读法（英语 one、他加禄语 isa）无法确定，丢弃
ENGLISH_CONTEXT = {
    "number", "no", "top", "the", "my", "version", "windows", "iphone", "ios", "android", "season", "episode",
    "chapter", "volume", "vol", "issue", "part", "level", "grade", "phase", "round", "page", "step", "rank", "class",
    "room", "gen", "edition", "series", "model", "type", "category", "formula", "apollo", "boeing", "airbus", "galaxy",
    "pixel", "xbox", "playstation", "ps", "office", "covid", "web", "day", "week", "game", "cup", "world", "item",
    "unit", "lot", "block", "phase", "track", "act", "scene", "verse", "psalm", "psalms", "route", "highway",
    "january", "february", "march", "april", "june", "july", "august", "september", "october", "november",
    "december",
}
# 他加禄语的编号用词：后面的数字按他加禄语读，不加连接成分（Ruta Bilang 66、Kabanata 3）
TAGALOG_LABELS = {
    "Bilang", "Kabanata", "Baitang", "Gradong", "Grado", "Pahina", "Taludtod", "Seksiyon", "Seksyon", "Artikulo",
    "Kalye", "Distrito", "Purok", "Sona", "Silid", "Kuwarto", "Antas", "Yugto", "Bahagi", "Aklat", "Ruta", "Batas",
    "Konstitusyon", "Konstitusyong",
}
VOWELS = "aeiou"


def fil_below_100(n: int) -> str:
    if n < 10:
        return UNITS[n]
    if n < 20:
        return TEENS[n]
    t, u = divmod(n, 10)
    return TENS[t] + ("'t " + UNITS[u] if u else "")


def linked(words: str) -> str:
    """加连接成分：以元音结尾加 -ng，以 n 结尾加 -g，以其他辅音结尾后接 na。"""
    last = words[-1]
    if last in VOWELS:
        return words + "ng"
    if last == "n":
        return words + "g"
    return words + " na"


def _hundreds(h: int) -> str:
    multiplier = linked(words = UNITS[h])
    return multiplier + (" raan" if multiplier.endswith(" na") else " daan")


def fil_cardinal(n: int) -> str:
    """他加禄语基数：1966 → isang libo siyam na raan at animnapu't anim，409 → apat na raan at siyam。"""
    if not 1 <= n < 10 ** 9:
        raise ValueError("菲律宾语基数只支持 1 到 999999999")
    parts = []
    for value, word in ((10 ** 6, "milyon"), (1000, "libo")):
        q, n = divmod(n, value)
        if q:
            parts.append(linked(words = fil_cardinal(n = q)) + " " + word)
    h, rest = divmod(n, 100)
    if h:
        parts.append(_hundreds(h = h))
    if rest:
        tail = fil_below_100(n = rest)
        if parts:
            return " ".join(parts) + " at " + tail
        return tail
    return " ".join(parts)


def fil_ordinal(n: int) -> str:
    """序数：2 → ikalawa，3 → ikatlo，18 → ikalabingwalo，21 → ikadalawampu't isa。只支持 2–99。"""
    if not 2 <= n <= 99:
        raise ValueError("菲律宾语序数只支持 2–99")
    if n == 2:
        return "ikalawa"
    if n == 3:
        return "ikatlo"
    return "ika" + fil_below_100(n = n)


class Fil(Lang):
    code = "fil"
    percent = "porsiyento"

    def card(self, n: int) -> str:
        return fil_cardinal(n = n)

    def rules(self) -> list:
        """序数「ika-18」先于公共保护规则运行（公共规则把字母、连字符、数字当作型号丢弃）。"""
        ordinal = Rule(
            pattern = re.compile(pattern = r"(?<![\w-])(?P<ika>[Ii]ka)-(?P<num>\d{1,3}(?:,\d{3})*)(?!\d|[.,]\d)"),
            handle = self.ordinal,
        )
        verse = Rule(pattern = re.compile(pattern = r"^\d{1,3}\s(?=(?P<next>[A-Z][\w'’]*))"), handle = self.verse_number)
        return common_rules(lang = self) + [verse, ordinal] + super().rules()

    def verse_number(self, m, ctx):
        """行首的经节号（6 Nang magkagayo'y…）删去；后面是其他大写词时（5 Paraan…）按数量读。"""
        if m.group("next").lower() not in VERSE_NEXT:
            return None
        ctx.kind(name = "删去行首节号", surface = m.group().strip())
        return ""

    def extra_rules(self) -> list:
        return [
            Rule(
                pattern = re.compile(pattern = r"\b(?P<alas>[Aa]las)(?P<sep>[\s-])(?P<h>\d{1,2})(?P<min>:\d{2})?(?!\d|[.,:]\d)"),
                handle = self.spanish_hour,
            ),
            Rule(
                pattern = re.compile(pattern = r"(?<![\w.])atbp\.(?!\w)"),
                handle = self.atbp,
            ),
            Rule(
                pattern = re.compile(pattern = r"(?<![\w.,])(?!0)\d{5,6}(?![\d.,])(?=\s[^\W\d_])"),
                handle = self.long_plain,
            ),
            Rule(
                pattern = re.compile(pattern = rf"(?P<num>{self.num})\s?(?:AD|BC|A\.D\.|B\.C\.|CE|BCE)(?!\w)"),
                handle = lambda m, ctx: Skip(reason = f"纪元标记的读法不确定「{m.group()}」"),
            ),
        ]

    def long_plain(self, m, ctx):
        """不带千分位的 5–6 位数后接文字（10000 yen）：按数量读。"""
        n = Number(written = m.group(), value = int(m.group()), frac = None, int_digits = m.group(), grouped = False)
        out = self.cardinal(m = m, ctx = ctx, n = n)
        if isinstance(out, str):
            ctx.kind(name = "基数", surface = m.group())
        return out

    def atbp(self, m, ctx):
        ctx.kind(name = "缩写", surface = m.group())
        # 缩写的句点同时是句号时保留句号
        rest = ctx.text[m.end():]
        if not rest.strip() or re.match(pattern = r"\s+[A-Z\"“]", string = rest):
            return "at iba pa."
        return "at iba pa"

    @staticmethod
    def prev_word(ctx, start: int) -> tuple:
        """前一个词，以及它是否在句首（文本开头或句号、问号、叹号之后）。"""
        before = ctx.text[:start].rstrip()
        m = re.search(pattern = r"([^\W\d_][\w'’]*)$", string = before)
        if not m:
            return "", False
        head = before[:m.start()].rstrip()
        return m.group(1), (not head or head[-1] in ".!?")

    def needs_linker(self, ctx, start: int, end: int) -> bool:
        """数词后面是名词等实词时加连接成分；前面是大写开头的名称、编号用词（Volume 1、Ruta Bilang 66）时不加。"""
        word, at_start = self.prev_word(ctx = ctx, start = start)
        if word in TAGALOG_LABELS:
            return False
        rest = ctx.text[end:]
        m = re.match(pattern = r" ([^\W\d_][\w'’-]*)", string = rest)
        if not m:
            return False
        word = m.group(1)
        return word.lower() not in FUNCTION_WORDS

    def year_context(self, ctx, start: int) -> bool:
        before = ctx.text[:start].rstrip()
        m = re.search(pattern = r"([^\W\d_]+)$", string = before)
        if not m:
            return False
        word = m.group(1)
        if word in YEAR_BEFORE:
            return True
        # 「ika-18 ng Disyembre ng 2022」「Pebrero, taong 1960」：月份后的 ng
        return word == "ng" and re.search(pattern = rf"\b(?:{'|'.join(MONTHS)})\s+ng$", string = before) is not None

    def cardinal(self, m, ctx, n):
        if not n.is_int:
            return Skip(reason = f"菲律宾语小数的读法（punto、tuldok）不确定「{m.group()}」")
        if n.value == 0:
            return Skip(reason = f"菲律宾语零的读法（sero、wala）不确定「{m.group()}」")
        word, at_start = self.prev_word(ctx = ctx, start = m.start())
        if word.lower() in ENGLISH_CONTEXT:
            return Skip(reason = f"英语语境或编号中的数字，读法不确定「{word} {m.group()}」")
        # 名称后的编号（Defenders 2、Apollo 11）多为英语读法
        if word[:1].isupper() and not at_start and word.lower() not in FUNCTION_WORDS and word not in TAGALOG_LABELS \
                and not self.year_context(ctx = ctx, start = m.start()):
            return Skip(reason = f"名称后的编号，读法不确定「{word} {m.group()}」")
        rest = ctx.text[m.end():]
        if re.match(pattern = r"\s?(?:[AaPp]\.?[Mm]\.?)(?![\w])", string = rest):
            return Skip(reason = f"英语时刻写法，读法不确定「{m.group()}{rest[:4]}」")
        # 句中的经节号、序号（…magnanakaw. 5 Kayong lahat…）
        before = ctx.text[:m.start()].rstrip()
        if before and before[-1] in ".!?" and re.match(pattern = r"\s[A-Z]", string = rest):
            return Skip(reason = f"句中的经节号或序号「{m.group()}」")
        spoken = ctx.num(spoken = self.card(n = n.value), written = m.group())
        if self.year_context(ctx = ctx, start = m.start()):
            ctx.kind(name = "日期" if word in MONTHS and n.value <= 31 else "年份", surface = m.group())
            return spoken
        if self.needs_linker(ctx = ctx, start = m.start(), end = m.end()):
            return linked(words = spoken)
        return spoken

    def percent_number(self, m, ctx, n):
        if not n.is_int or n.value == 0:
            return Skip(reason = f"菲律宾语小数、零的读法不确定「{m.group()}」")
        return linked(words = ctx.num(spoken = self.card(n = n.value), written = n.written))

    def ordinal(self, m, ctx):
        written = m.group("num").replace(",", "")
        n = int(written)
        if not 2 <= n <= 99 or written.startswith("0"):
            return Skip(reason = f"菲律宾语序数暂只处理 2–99「{m.group()}」")
        spoken = fil_ordinal(n = n)
        if m.group("ika")[0] == "I":
            spoken = spoken[0].upper() + spoken[1:]
        ctx.kind(name = "序数", surface = m.group())
        ctx.num(spoken = spoken, written = m.group())
        if self.needs_linker(ctx = ctx, start = m.start(), end = m.end()):
            return linked(words = spoken)
        return spoken

    def spanish_hour(self, m, ctx):
        h = int(m.group("h"))
        if m.group("min") and m.group("min") != ":00":
            return Skip(reason = f"时刻的分钟读法（y medya、trenta）不确定「{m.group()}」")
        if h not in SPANISH_HOURS or m.group("h").startswith("0"):
            return Skip(reason = f"时刻写法无法确定「{m.group()}」")
        ctx.kind(name = "时刻", surface = m.group())
        hour = ctx.num(spoken = SPANISH_HOURS[h], written = m.group("h"))
        return m.group("alas") + m.group("sep") + hour


LANG = Fil
