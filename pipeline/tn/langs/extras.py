"""几种语言共用的附加规则：负温度、不带千分位的 5–6 位数。

印尼语、越南语、泰语、土耳其语、波斯语使用。负温度规则要放在公共保护规则之前（公共规则把「-10」当作
负号或连字符丢弃），由各语言的 rules() 放在最前面。
"""

from __future__ import annotations

import re

from ..rules import DASHES, Rule, Skip, alternation


def negative_rule(lang, minus: str, words: tuple) -> Rule:
    """「-10 derajat」「-10°C」：负号紧贴数字，后面是温度单位或 words 中的词，读作 minus 加数值（加单位）。

    lang 是该语言的 Lang 对象；°C、℃ 等写在 lang.units 中的单位一并读出。
    """
    dash = re.escape(pattern = DASHES)
    symbols = [u for u in ("°C", "℃", "°F", "°") if u in lang.units]
    after = alternation(words = words + ("°",))
    tail = rf"(?=\s?(?:{after}))"
    if symbols:
        tail = rf"(?:(?P<unit>\s?(?:{alternation(words = symbols)}))|{tail})"

    def handle(m, ctx):
        n = lang.parse(written = m.group("num"))
        if n is None:
            return Skip(reason = f"数字写法无法确定「{m.group()}」")
        spoken = lang.say(n = n, ctx = ctx)
        if isinstance(spoken, Skip):
            return spoken
        spoken = lang.join(minus, spoken)
        ctx.checks[-1] = (spoken, "-" + m.group("num"))
        if m.groupdict().get("unit"):
            spoken = lang.join(spoken, lang.units[m.group("unit").strip()][1])
            ctx.kind(name = "单位", surface = m.group())
        ctx.kind(name = "负数", surface = m.group())
        return spoken

    return Rule(
        pattern = re.compile(pattern = rf"(?<![\w.,{dash}])[-−](?P<num>{lang.num}){tail}"),
        handle = handle,
    )


def long_plain_rule(lang, lookahead: str = r"\s[^\W\d_]") -> Rule:
    """不带千分位的 5–6 位数，后面紧接文字（10000 người、20000 نسخه）：按数值读。

    公共规则把 4 位以上不带千分位的数当作编号丢弃；后面跟着文字时多为数量，这里先处理。
    """

    def handle(m, ctx):
        ctx.kind(name = "基数", surface = m.group())
        return ctx.num(spoken = lang.card(n = int(m.group("num"))), written = m.group("num"))

    return Rule(
        pattern = re.compile(pattern = rf"(?<![\w.,٫٬])(?P<num>(?![0٠۰])\d{{5,6}})(?![\d.,٫٬])(?={lookahead})"),
        handle = handle,
    )


def missing_space_rule(left: str, right: str, min_left: int = 2) -> Rule:
    """句号后缺少空格（rồi.Tạm、açar.Eğer、داد.به）：两边都是完整的词时补上空格。

    left、right 是句号前后允许的字符类（正则字符集内容）；拉丁文字要求句号后是大写字母开头的词，
    避免改动 db.o、google.com 这类写法。
    """

    def handle(m, ctx):
        ctx.kind(name = "句号后补空格", surface = m.group())
        return m.group("a") + ". " + m.group("b")

    return Rule(
        pattern = re.compile(pattern = rf"(?<![\w.])(?P<a>[{left}]{{{min_left},}})\.(?P<b>{right})"),
        handle = handle,
    )
