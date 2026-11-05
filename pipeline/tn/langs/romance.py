"""法语、西班牙语、葡萄牙语、意大利语的公共规则。

这几种语言的部分数词随名词的阴阳性变化（法语 un/une，西语 uno/una 与 doscientos/doscientas，葡语
um/uma、dois/duas 与 duzentos/duzentas，意语 uno/una）。这类数按后面名词的性选择词形：名词在各语言的
词表中（años 阳性、personas 阴性），或带有可靠的阴性词尾（-ciones、-ções、-zioni、-tions）时改写；
年份、日期中的日、单位、货币按阳性读；名词后的编号（Fórmula 1、capítulo 21、Rai 1）按阳性基数读；
其余丢弃。数字后紧跟大写词（多为专名，例如「3 Arts Entertainment」）时也丢弃。

罗马数字只在三种位置改写：君主、教皇名之后（Felipe II、Louis XIV、Pio XII），世纪词前后（siglo XX、
XX secolo），以及名词前作序数（II Guerra Mundial → Segunda Guerra Mundial）。法语带序数后缀的写法
（XVIIe、Ier）读作序数。西语、葡语的君主和世纪 I–X 读序数、XI 以上读基数，意语一律读序数，法语君主
读基数（Louis quatorze）。
"""

from __future__ import annotations

import re

from ..rules import Number, Rule, Skip, alternation
from .base import Lang, word_after, word_before
from .euro import abbreviation_rules, initialism_rule, roman_value, space_thousands_pattern, unit_follows

# 两个字母的 CD、DC、MC 等多为缩写，不算罗马数字（与 detect 一致）
ROMAN = r"(?<![\w'’-])(?P<roman>[IVXLCDM]+)(?![\w'’-])"


def roman_ok(token: str) -> int | None:
    value = roman_value(text = token)
    if value is None or (len(token) == 2 and not set(token) <= set("IVXL")):
        return None
    return value


class Romance(Lang):
    months = ()
    # 年份前常见的词（en、desde、nel 等）
    year_before = set()
    # 后面紧跟 3 位数也读作年份的词（en 884、Em 917），后面不能是名词
    year_before_short = set()
    # 已知的阳性复数名词（兼容旧写法，并入 nouns）
    masc_after = set()
    # 名词的性：词形（小写）→ "m" 或 "f"；单数、复数都可以登记
    nouns = {}
    # 可靠的阴性词尾（-ciones、-tions），以及可靠的阳性词尾
    fem_suffixes = ()
    masc_suffixes = ()
    # 名词后读编号的词（capítulo 21、Fórmula 1），数按阳性基数读
    id_nouns = set()
    # 日期中的「1 日」读法（premier、primero、primeiro、primo）
    first_day = ""
    # 日期中日与月之间、月与年之间的连接（西语、葡语为 de）
    date_link = r"\s"
    # 缩写表（见 euro.abbreviation_rules）、编号缩写（n.º 5 → número cinco）、句首词
    abbreviations = {}
    number_abbreviations = {}
    starters = set()
    # 罗马数字
    regnal_names = set()
    regnal_female = set()
    century_words = set()
    # 君主序号、世纪按序数读的上限（西语、葡语 10，意语一律序数）；法语君主读基数，设为 0
    roman_ordinal_max = 10
    # 量级词：后面的名词决定性（2 mil pessoas → duas mil pessoas）
    thousand_word = None

    # ------------------------------------------------------------------
    # 读法
    # ------------------------------------------------------------------

    def gender_sensitive(self, n: int) -> bool:
        raise NotImplementedError

    def masc(self, n: int) -> str:
        """名词前的阳性形式。"""
        return self.card(n = n)

    def fem(self, n: int) -> str | None:
        """名词前的阴性形式，无法确定时返回 None。"""
        return None

    def ordinal_words(self, n: int, gender: str = "m") -> str | None:
        """序数（阴性把每个词末的 o 改为 a）；num2words 输出有误的数返回 None。"""
        return None

    def plural(self, n) -> bool:
        return not (n.is_int and n.value == 1)

    def say(self, n, ctx, card = None):
        """小数部分为两位、读作数时是多个词（sesenta y uno）的，改为逐位读（比对模块只认单词的小数读法）。"""
        if not n.is_int and self.frac_as_number and len(n.frac) <= 2 and not n.frac.startswith("0"):
            if re.search(pattern = r"\s", string = self.card(n = int(n.frac))):
                card = card or self.card
                spoken = self.join(card(n = n.value), self.point, self.digits(digits = n.frac))
                return ctx.num(spoken = spoken, written = n.written)
        return super().say(n = n, ctx = ctx, card = card)

    def gender_of(self, word: str) -> str | None:
        """名词的性：词表优先，其次是可靠的词尾。"""
        w = word.lower()
        if w in self.nouns:
            return self.nouns[w]
        if w in {x.lower() for x in self.masc_after}:
            return "m"
        if self.fem_suffixes and w.endswith(self.fem_suffixes):
            return "f"
        if self.masc_suffixes and w.endswith(self.masc_suffixes):
            return "m"
        return None

    def unit_number(self, m, ctx, n):
        """单位和货币都是阳性名词；小数的整数部分按普通基数读（西语 uno coma cinco）。"""
        return self.say(n = n, ctx = ctx, card = self.masc if n.is_int else None)

    currency_number = unit_number

    # ------------------------------------------------------------------
    # 规则
    # ------------------------------------------------------------------

    def rules(self) -> list:
        early = abbreviation_rules(table = self.abbreviations, starters = self.starters) if self.abbreviations else []
        if self.number_abbreviations:
            early.append(Rule(
                pattern = re.compile(
                    pattern = rf"(?<![\w.])(?P<abbr>{alternation(words = self.number_abbreviations)})\s?(?P<num>\d+)(?![\d]|[.,]\d)",
                ),
                handle = self.numbered,
            ))
        early += self.early_rules()
        early += [
            # 数字后的撇号表示分钟或英尺（il 40' del primo tempo），读法取决于上下文
            Rule(
                pattern = re.compile(pattern = r"(?<![\d.,])\d+['’](?![\w])"),
                handle = lambda m, ctx: Skip(reason = f"数字后的撇号（分钟、英尺）「{m.group()}」"),
            ),
            initialism_rule(starters = self.starters),
            Rule(pattern = re.compile(pattern = space_thousands_pattern()), handle = self.space_thousands),
            Rule(pattern = re.compile(pattern = ROMAN), handle = self.roman),
        ]
        return early + super().rules()

    def early_rules(self) -> list:
        return []

    def extra_rules(self) -> list:
        months = alternation(words = self.months)
        link = self.date_link
        return [
            Rule(
                pattern = re.compile(
                    pattern = (
                        rf"(?<![\d.,])(?P<day>\d{{1,2}})(?P<er>er|º|°)?(?P<l1>{link})(?P<month>{months})\b"
                        rf"(?:(?P<l2>{link})(?P<year>\d{{3,4}})(?!\d|[.,]\d))?"
                    ),
                    flags = re.IGNORECASE,
                ),
                handle = self.date,
            ),
        ] + self.ordinal_rules() + [
            Rule(
                pattern = re.compile(pattern = r"(?<![\d.,])(?P<num>1\d{3}|20\d{2})(?![\d]|[.,]\d)"),
                handle = self.year,
            ),
            Rule(
                pattern = re.compile(pattern = r"(?<![\d.,])(?P<num>[1-9]\d{2})(?![\d]|[.,]\d)"),
                handle = self.short_year,
            ),
        ]

    def ordinal_rules(self) -> list:
        """3º、36ª、1.º 这类带序数标记的写法（法语另有规则）。"""
        return [
            Rule(
                pattern = re.compile(pattern = r"(?<![\d.,])(?P<num>\d{1,3})\.?(?P<suf>[ºª])(?![\w])"),
                handle = self.ordinal_mark,
            ),
        ]

    # ------------------------------------------------------------------
    # 缩写、空格千分位
    # ------------------------------------------------------------------

    def numbered(self, m, ctx):
        if m.group("num").startswith("0"):
            return Skip(reason = f"编号写法无法确定「{m.group()}」")
        ctx.kind(name = "缩写", surface = m.group("abbr"))
        n = int(m.group("num"))
        return self.number_abbreviations[m.group("abbr")] + " " + ctx.num(spoken = self.card(n = n), written = m.group("num"))

    def space_thousands(self, m, ctx):
        """4 561、299 792 458：以空格分千位的数按一个数读（后面是单位时丢弃）。"""
        if unit_follows(text = ctx.text, end = m.end(), units = self.units):
            return Skip(reason = f"空格分隔的数字后接单位「{m.group()}」")
        digits = m.group().replace(" ", "")
        n = Number(
            written = digits,
            value = int(digits),
            frac = None,
            int_digits = digits,
            grouped = True,
        )
        out = self.cardinal(m = m, ctx = ctx, n = n)
        if isinstance(out, str):
            ctx.kind(name = "空格分千位", surface = m.group())
        return out

    # ------------------------------------------------------------------
    # 罗马数字
    # ------------------------------------------------------------------

    def roman_reading(self, value: int, gender: str, ordinal_max: int) -> str | None:
        if value <= ordinal_max:
            return self.ordinal_words(n = value, gender = gender)
        return self.card(n = value)

    def roman(self, m, ctx):
        token = m.group("roman")
        value = roman_ok(token = token)
        if value is None:
            return None
        text = ctx.text
        single = len(token) == 1
        before = text[:m.start()]
        prev_m = re.search(pattern = r"([^\W\d_]+)\.?\s$", string = before)
        prev = prev_m.group(1) if prev_m else ""
        after = word_after(text = text, end = m.end())
        # 单个字母后接点和大写的姓：名字的首字母（Francisco I. Madero）
        initial = re.match(pattern = r"\.\s([A-ZÀ-Ý][^\W\d_]+)", string = text[m.end():])
        if single and initial and initial.group(1) not in self.starters:
            return None
        # 君主、教皇（Felipe II、Pio XII）
        if prev in self.regnal_names or prev in self.regnal_female:
            gender = "f" if prev in self.regnal_female else "m"
            spoken = self.roman_reading(value = value, gender = gender, ordinal_max = self.roman_ordinal_max)
            if spoken is None:
                return Skip(reason = f"罗马数字的读法无法确定「{m.group()}」")
            ctx.kind(name = "罗马数字", surface = f"{prev} {token}")
            return ctx.num(spoken = spoken, written = str(value))
        # 世纪（siglo XX、XX secolo、siglos XIX y XX）
        chain = re.search(pattern = r"([^\W\d_]+)\.?\s(?:[IVXLCDM]+\s(?:y|e|ed|et)\s)?$", string = before)
        century = (chain and chain.group(1).lower() in self.century_words) or after.lower() in self.century_words
        if century and not text[m.end():m.end() + 1] in "-–":
            spoken = self.roman_reading(value = value, gender = "m", ordinal_max = self.century_ordinal_max())
            if spoken is None:
                return Skip(reason = f"罗马数字的读法无法确定「{m.group()}」")
            ctx.kind(name = "罗马数字", surface = m.group())
            return ctx.num(spoken = spoken, written = str(value))
        # 单个字母（I、V、X）只在君主名后、世纪词前后改写，其余保持原样（意语冠词 I、法语字母）
        if single:
            return None
        # 名词后的编号（capítulo III、tome II）
        if prev.lower() in self.roman_id_nouns():
            spoken = self.roman_after_noun(value = value, noun = prev)
            if spoken is None:
                return Skip(reason = f"罗马数字的读法取决于上下文「{prev} {token}」")
            ctx.kind(name = "罗马数字", surface = f"{prev} {token}")
            return ctx.num(spoken = spoken, written = str(value))
        # 名词前作序数（II Guerra Mundial → Segunda Guerra Mundial，su XIII edición，XII duque de …）
        gender = self.gender_of(word = after) if after else None
        determiner = prev.lower() in self.determiners()
        if gender and (self.roman_before_noun_ok(noun = after) or determiner or after.lower() in self.title_nouns()):
            spoken = self.ordinal_words(n = value, gender = gender)
            if spoken is not None:
                spoken = self.apocope_ordinal(words = spoken, noun = after)
                ctx.kind(name = "罗马数字", surface = f"{token} {after}")
                return ctx.num(spoken = spoken, written = str(value))
        # 世纪词出现在前面几个词内（siglo XVIII y principios del XIX、siglo XVII, XVIII, XIX）
        recent = re.findall(pattern = r"[^\W\d_]+", string = before[-50:])
        if self.century_words and any(w.lower() in self.century_words for w in recent[-5:]) and not text[m.end():m.end() + 1] in "-–":
            spoken = self.roman_reading(value = value, gender = "m", ordinal_max = self.century_ordinal_max())
            if spoken is not None:
                ctx.kind(name = "罗马数字", surface = m.group())
                return ctx.num(spoken = spoken, written = str(value))
        return Skip(reason = f"罗马数字的读法取决于上下文「{m.group()}」")

    def determiners(self) -> set:
        """罗马数字前的限定词：后面的名词前的罗马数字读序数（su XIII edición）。"""
        return set()

    def title_nouns(self) -> set:
        """贵族称号：前面的罗马数字读序数（XII duque del Infantado）。"""
        return set()

    def century_ordinal_max(self) -> int:
        return self.roman_ordinal_max

    def roman_id_nouns(self) -> set:
        return set()

    def roman_after_noun(self, value: int, noun: str) -> str | None:
        return self.card(n = value)

    def roman_before_noun_ok(self, noun: str) -> bool:
        """名词前的罗马数字只在名词大写（西语、葡语的专名：Guerra Mundial、Reich）时改写。"""
        return noun[:1].isupper()

    def apocope_ordinal(self, words: str, noun: str) -> str:
        return words

    # ------------------------------------------------------------------
    # 序数、日期、年份
    # ------------------------------------------------------------------

    def ordinal(self, n: int) -> str:
        return self.card(n = n)

    def ordinal_mark(self, m, ctx):
        n = int(m.group("num"))
        if n == 0 or m.group("num").startswith("0"):
            return Skip(reason = f"序数写法无法确定「{m.group()}」")
        gender = "f" if m.group("suf") == "ª" else "m"
        words = self.ordinal_words(n = n, gender = gender)
        if words is None:
            return Skip(reason = f"序数写法无法确定「{m.group()}」")
        after = word_after(text = ctx.text, end = m.end())
        if gender == "m" and words.endswith(("primero", "tercero")) and after:
            noun_gender = self.gender_of(word = after)
            if noun_gender != "m" or after.lower().endswith("s"):
                return Skip(reason = f"名词前 primero、tercero 要缩略为 primer、tercer「{m.group()}」")
            words = words[:-1]
        ctx.kind(name = "序数", surface = m.group())
        return ctx.num(spoken = words, written = m.group())

    def year_words(self, n: int) -> str:
        return self.card(n = n)

    def date(self, m, ctx):
        day = int(m.group("day"))
        if not 1 <= day <= 31 or (m.group("er") and day != 1):
            return Skip(reason = f"日期写法无法确定「{m.group()}」")
        spoken = self.first_day if day == 1 else self.card(n = day)
        out = ctx.num(spoken = spoken, written = m.group("day")) + m.group("l1") + m.group("month")
        if m.group("year"):
            out += m.group("l2") + ctx.num(spoken = self.year_words(n = int(m.group("year"))), written = m.group("year"))
        ctx.kind(name = "日期", surface = m.group())
        return out

    def is_year_context(self, m, ctx) -> bool:
        text = ctx.text
        before = word_before(text = text, start = m.start())
        if before in self.year_before or before in {mo.lower() for mo in self.months}:
            return True
        # (1673)、(Toronto, 1961)
        lead = text[:m.start()].rstrip()
        rest = text[m.end():].lstrip()
        return bool(lead) and lead[-1] in "(," and rest[:1] == ")"

    def year(self, m, ctx):
        if not self.is_year_context(m = m, ctx = ctx) or unit_follows(text = ctx.text, end = m.end(), units = self.units):
            return None
        after = word_after(text = ctx.text, end = m.end())
        if self.gender_of(word = after) is not None and after[:1].islower():
            # 后面是名词：数量（en 1500 casas）
            return None
        ctx.kind(name = "年份", surface = m.group())
        return ctx.num(spoken = self.year_words(n = int(m.group("num"))), written = m.group())

    def short_year(self, m, ctx):
        """en 884、Em 917、nel 410：三位数紧跟在 en、em、nel 之后，后面不是词（标点、句末或介词）时是年份。"""
        if word_before(text = ctx.text, start = m.start()) not in self.year_before_short:
            return None
        if unit_follows(text = ctx.text, end = m.end(), units = self.units):
            return None
        rest = ctx.text[m.end():]
        nxt = re.match(pattern = r"\s+([^\W\d_]+)(\.)?", string = rest)
        if nxt and not nxt.group(2) and nxt.group(1) not in self.function_words() | self.year_before_short:
            return None
        ctx.kind(name = "年份", surface = m.group())
        return ctx.num(spoken = self.year_words(n = int(m.group("num"))), written = m.group())

    # ------------------------------------------------------------------
    # 普通数字
    # ------------------------------------------------------------------

    def noun_after(self, ctx, end: int) -> tuple:
        """数后面的名词：跳过 mil（2 mil pessoas）和 de（millones de personas 不需要），返回 (名词, 是否隔着 mil)。"""
        after = word_after(text = ctx.text, end = end)
        if self.thousand_word and after == self.thousand_word:
            rest = ctx.text[end:]
            m = re.match(pattern = rf"\s*{self.thousand_word}\s+([^\W\d_]+)", string = rest)
            return (m.group(1) if m else "", True)
        return (after, False)

    def cardinal(self, m, ctx, n):
        after = word_after(text = ctx.text, end = m.end())
        if after[:1].isupper() and after.lower() not in {w.lower() for w in self.months}:
            return Skip(reason = f"数字后接专名，读法无法确定「{m.group()} {after}」")
        if not (n.is_int and self.gender_sensitive(n = n.value)):
            return self.say(n = n, ctx = ctx)
        noun, via_thousand = self.noun_after(ctx = ctx, end = m.end())
        gender = self.gender_of(word = noun) if noun else None
        if gender == "m":
            card = self.card if via_thousand else self.masc
            return self.say(n = n, ctx = ctx, card = card)
        if gender == "f":
            fem = self.fem(n = n.value)
            if fem is None:
                return Skip(reason = f"阴性数词的读法不固定「{m.group()} {noun}」")
            ctx.kind(name = "阴性基数", surface = f"{m.group()} {noun}")
            return ctx.num(spoken = fem, written = n.written)
        # 1000–2099 的数后面不是名词（标点、句末、连词、介词）：年份，按阳性读
        if n.is_int and not n.grouped and 1000 <= n.value <= 2099 and (not after or after in self.function_words()):
            ctx.kind(name = "年份", surface = m.group())
            return ctx.num(spoken = self.year_words(n = n.value), written = n.written)
        # 名词后的编号（capítulo 21、Fórmula 1）：后面不是名词时按阳性基数读
        prev = word_before(text = ctx.text, start = m.start())
        if prev in self.id_nouns:
            ctx.kind(name = "编号", surface = f"{prev} {m.group()}")
            return self.say(n = n, ctx = ctx)
        return Skip(reason = f"数词的阴阳性取决于名词「{m.group()}」")

    def function_words(self) -> set:
        """可以紧跟在编号后面的虚词（capítulo 1 y 2、Fórmula 1 en 2010）。"""
        return set()
