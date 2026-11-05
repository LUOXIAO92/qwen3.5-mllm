"""T04c 八种语言（id、ms、th、vi、fil、ar、tr、fa）共用的附加规则。

负温度、5–6 位数、句号后补空格由各语言按需使用；common_rules() 中的规则（重复的数字、非标准标点、尺寸、
列表序号、引用标号）由每种语言的 rules() 放在最前面，先于公共保护规则运行。
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


# ---------------------------------------------------------------------------
# 各语言最先运行的共用规则（common_rules）
# ---------------------------------------------------------------------------

LATIN_UPPER = "A-ZÀ-ÖØ-ÞĀ-ſƀ-ɏḀ-ỿ"


def _lower_first(text: str, lang: str) -> str:
    """把括号里的数词改成句中的写法：Delapan → delapan，DELAPAN → delapan；土耳其语 İ、I 分别对应 i、ı。"""
    def low(s: str) -> str:
        if lang == "tr":
            s = s.replace("İ", "i").replace("I", "ı")
        return s.lower()

    if text.isupper():
        return low(s = text)
    if text[:1].isupper():
        return low(s = text[:1]) + text[1:]
    return text


def duplicate_number_rules(lang) -> list:
    """数字与括号里的同一个数重复写出时只读一次。

    - 「8 (Delapan)」「3 (tiga) tahun」：括号里是这个数的拼写，读括号里的词，删去数字和括号；
    - 「tiga (3) tahun」「شصت (۶۰)」：括号里的数字与前面的数词相同，删去括号。
    比较用 pipeline.compare，数值相同才算重复。
    """
    from ...compare import compare

    def spelled_after(m, ctx):
        words = m.group("words").strip()
        if not any(ch.isalpha() for ch in words):
            return None
        if not compare(script = words, hypothesis = m.group("num"), lang = lang.code).match:
            return None
        ctx.kind(name = "删去重复的数字", surface = m.group())
        return ctx.num(spoken = _lower_first(text = words, lang = lang.code), written = m.group("num"))

    def digits_after(m, ctx):
        written = m.group("num")
        before = ctx.text[:m.start()].rstrip()
        tokens = before.split()
        candidates = [" ".join(tokens[-k:]) for k in range(1, min(6, len(tokens)) + 1)]
        n = lang.parse(written = written)
        if n is not None and n.is_int:
            try:
                reading = lang.card(n = n.value)
            except (ValueError, NotImplementedError, KeyError):
                reading = ""
            # 泰语等不分词的语言：数词直接接在前面的字后面（ระยะเวลาสาม (3) ปี）
            if reading and before.endswith(reading):
                candidates.append(reading)
        for words in candidates:
            words = words.strip("«»\"“”'‘’,،")
            if not words or any(ch.isdigit() for ch in words):
                continue
            if compare(script = words, hypothesis = written, lang = lang.code).match:
                ctx.kind(name = "删去重复的数字", surface = m.group().strip())
                return ""
        return None

    num = r"\d+(?:[.,٫٬]\d+)*"
    return [
        Rule(
            pattern = re.compile(pattern = rf"(?<![\d.,٫٬])(?P<num>{num})\s?\((?P<words>[^()\d]{{1,60}})\)"),
            handle = spelled_after,
        ),
        Rule(pattern = re.compile(pattern = rf"\s?\((?P<num>{num})\)"), handle = digits_after),
    ]


def odd_punctuation_rule() -> Rule:
    """数字之间夹着 ! ? ; 这类非标准写法（28!10）：读法无法确定，丢弃。"""
    return Rule(
        pattern = re.compile(pattern = r"\d+\s?[!?;؟؛]\s?\d+"),
        handle = lambda m, ctx: Skip(reason = f"数字之间的标点写法不标准「{m.group()}」"),
    )


def dimension_rule(lang, times: str | None) -> Rule:
    """尺寸「140 x 130 x 60 mm」：times 为乘号的读法（越南语 nhân）；为 None 时丢弃。"""
    num = lang.num
    units = alternation(words = lang.units) if lang.units else None
    unit = rf"(?:\s?(?P<unit>{units})(?![\w/²³]))?" if units else r"(?P<unit>)"
    pattern = rf"(?P<chain>{num}(?:\s?[x×]\s?{num})+){unit}"

    def handle(m, ctx):
        if times is None:
            return Skip(reason = f"尺寸写法的读法未登记「{m.group()}」")
        parts = re.split(pattern = r"\s?[x×]\s?", string = m.group("chain"))
        spoken = []
        for part in parts:
            n = lang.parse(written = part)
            if n is None:
                return Skip(reason = f"数字写法无法确定「{m.group()}」")
            out = lang.say(n = n, ctx = ctx)
            if isinstance(out, Skip):
                return out
            spoken.append(out)
        text = f" {times} ".join(spoken)
        if m.group("unit"):
            text = lang.join(text, lang.units[m.group("unit")][1])
        ctx.kind(name = "尺寸", surface = m.group())
        return text

    return Rule(pattern = re.compile(pattern = rf"(?<![\w.,])(?:{pattern})(?![\w])"), handle = handle)


def list_number_rule(upper: str = LATIN_UPPER) -> Rule:
    """列表序号「3.Mungkin」「2. Pertama」：在文本开头删去；在句中（句号、冒号之后）丢弃整条。"""

    def handle(m, ctx):
        if m.start() == 0:
            ctx.kind(name = "删去行首序号", surface = m.group().strip())
            return ""
        return Skip(reason = f"句中的列表序号「{m.group().strip()}」")

    return Rule(
        pattern = re.compile(pattern = rf"(?:^|(?<=[.!?:;]\s))\d{{1,2}}\.\s?(?=[{upper}])"),
        handle = handle,
    )


def citation_rule(standalone: bool) -> Rule:
    """文献引用、脚注标号：紧贴在词后的「(2)」「[3]」丢弃；standalone 为 True 时单独的「(2)」也丢弃。"""
    glued = r"(?<=[^\W\d_])[(\[]\d{1,3}[)\]]"
    pattern = rf"{glued}|(?<!\S)[(\[]\d{{1,3}}[)\]]" if standalone else glued
    return Rule(
        pattern = re.compile(pattern = pattern),
        handle = lambda m, ctx: Skip(reason = f"文献引用或脚注标号「{m.group()}」"),
    )


def common_rules(lang, times: str | None = None, list_numbers: bool = True, standalone_citation: bool = False) -> list:
    """各语言最先运行的共用规则：重复的数字、非标准标点、尺寸、列表序号、引用标号。"""
    rules = duplicate_number_rules(lang = lang) + [odd_punctuation_rule(), dimension_rule(lang = lang, times = times)]
    if list_numbers:
        rules.append(list_number_rule())
    rules.append(citation_rule(standalone = standalone_citation))
    return rules


def adjacent_numeral_rule(lang, words: tuple) -> Rule:
    """数字后面紧跟写成文字的数词（30 một số loại、2030 dua puluh satu、۲۰۵۰ سه برابر）：读出后与后面的数词连成
    另一个数（ba mươi một = 31），读法有歧义，丢弃。words 是该语言个位、十位的数词，不含 năm（年）这类常用作名词的词。"""
    return Rule(
        pattern = re.compile(pattern = rf"(?P<num>{lang.num})(?=\s(?:{alternation(words = words)})(?![\w]))"),
        handle = lambda m, ctx: Skip(reason = f"数字后紧跟数词，读法与后面的数词连成另一个数「{m.group()}」"),
    )
