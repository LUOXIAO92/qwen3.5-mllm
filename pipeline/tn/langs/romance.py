"""法语、西班牙语、葡萄牙语、意大利语的公共规则。

这几种语言的部分数词随名词的阴阳性变化（法语 un/une，西语 uno/una 与 doscientos/doscientas，葡语
um/uma、dois/duas 与 duzentos/duzentas，意语 uno/una）。这类数只在确定是阳性的位置改写：年份、日期
中的日、单位、货币，以及后面是登记的阳性名词（años、anos、anni、ans 等）；其余丢弃。数字后紧跟大写词
（多为专名，例如「3 Arts Entertainment」）时也丢弃。
"""

from __future__ import annotations

import re

from ..rules import Rule, Skip, alternation
from .base import Lang, word_after, word_before


class Romance(Lang):
    months = ()
    # 年份前常见的词（en、desde、nel 等）
    year_before = set()
    # 年代：les années 1980、la década de 1920、negli anni 1990 中的年份
    # 已知的阳性复数名词，后面是这些词时变性数词用阳性
    masc_after = set()
    # 日期中的「1 日」读法（premier、primero、primeiro、primo）
    first_day = ""
    # 日期中日与月之间、月与年之间的连接（西语、葡语为 de）
    date_link = r"\s"

    def gender_sensitive(self, n: int) -> bool:
        raise NotImplementedError

    def masc(self, n: int) -> str:
        """名词前的阳性形式。"""
        return self.card(n = n)

    def plural(self, n) -> bool:
        return not (n.is_int and n.value == 1)

    def unit_number(self, m, ctx, n):
        """单位和货币都是阳性名词；小数的整数部分按普通基数读（西语 uno coma cinco）。"""
        return self.say(n = n, ctx = ctx, card = self.masc if n.is_int else None)

    currency_number = unit_number

    def extra_rules(self) -> list:
        months = alternation(words = self.months)
        link = self.date_link
        return [
            Rule(
                pattern = re.compile(
                    pattern = (
                        rf"(?<![\d.,])(?P<day>\d{{1,2}})(?P<er>er)?(?P<l1>{link})(?P<month>{months})\b"
                        rf"(?:(?P<l2>{link})(?P<year>\d{{4}})(?!\d|[.,]\d))?"
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
        ]

    def ordinal_rules(self) -> list:
        """3º、36ª、1.º 这类带序数标记的写法（法语另有规则）。"""
        return [
            Rule(
                pattern = re.compile(pattern = r"(?<![\d.,])(?P<num>\d{1,3})\.?(?P<suf>[ºª])(?![\w])"),
                handle = self.ordinal_mark,
            ),
        ]

    def ordinal(self, n: int) -> str:
        return self.card(n = n)

    def ordinal_mark(self, m, ctx):
        n = int(m.group("num"))
        if n == 0 or m.group("num").startswith("0"):
            return Skip(reason = f"序数写法无法确定「{m.group()}」")
        words = self.ordinal(n = n)
        if m.group("suf") == "ª":
            # 阴性：每个词词尾的 o 改为 a（vigésimo primeiro → vigésima primeira）
            words = re.sub(pattern = r"o\b", repl = "a", string = words)
        elif words.endswith(("primero", "tercero")) and word_after(text = ctx.text, end = m.end()):
            return Skip(reason = f"名词前 primero、tercero 要缩略为 primer、tercer「{m.group()}」")
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

    def year(self, m, ctx):
        before = word_before(text = ctx.text, start = m.start())
        if before not in self.year_before:
            return None
        ctx.kind(name = "年份", surface = m.group())
        return ctx.num(spoken = self.year_words(n = int(m.group("num"))), written = m.group())

    def cardinal(self, m, ctx, n):
        after = word_after(text = ctx.text, end = m.end())
        if after[:1].isupper() and after.lower() not in {w.lower() for w in self.months}:
            return Skip(reason = f"数字后接专名，读法无法确定「{m.group()} {after}」")
        if n.is_int and self.gender_sensitive(n = n.value):
            if after.lower() in self.masc_after:
                return self.say(n = n, ctx = ctx, card = self.masc)
            return Skip(reason = f"数词的阴阳性取决于名词「{m.group()}」")
        return self.say(n = n, ctx = ctx)
