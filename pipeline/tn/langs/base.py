"""各语言规则的公共部分。

每种语言是 Lang 的一个子类，提供基数读法 card()、逐位读法 digits()、小数点、百分号、单位和货币的词，
需要特殊处理的写法（年份、日期、序数、时刻、量词等）放在 extra_rules() 里，最后由 cardinal() 决定一个
普通数字怎样读，或者因为没有把握而 Skip。规则的运行顺序见 Lang.rules()。
"""

from __future__ import annotations

import re

from ..rules import (
    LATINISH, Number, Rule, Skip, adjacency_rules, alternation, code_prefix_rule, colon_rule, guard_rules,
    next_word, number_pattern, parse_number, prev_word, unknown_unit_rule, unparsable_rule,
)

# 单位、货币符号后面不能紧跟拉丁、西里尔字母或斜线（km 不能匹配 kmh、km/h 中的 km）
END = rf"(?![{LATINISH}/²³])"
# 各语言的数量级词。货币符号后的数字后面跟着本语言没有登记的数量级词时（$45 millones），丢弃
ANY_SCALES = {
    "thousand", "million", "millions", "billion", "billions", "trillion", "Tausend", "Million", "Millionen",
    "Milliarde", "Milliarden", "mil", "millón", "millones", "milhão", "milhões", "bilhão", "bilhões", "mila",
    "milione", "milioni", "miliardo", "miliardi", "mille", "milliard", "milliards", "ribu", "juta", "miliar",
    "bilion", "triliun", "nghìn", "ngàn", "triệu", "tỷ", "tỉ", "bin", "milyon", "milyar", "هزار", "میلیون",
    "میلیارد",
}


class Lang:
    code = ""
    # 小数点的读法；None 表示小数暂不处理
    point = None
    # 小数部分为 1–2 位且不以 0 开头时按整数读（法语 virgule quatorze），否则逐位读
    frac_as_number = False
    # 数词与后面文字之间是否用空格隔开（中文、日语、泰语为 False）
    spaced = True
    # 百分号的读法；percent_prefix 表示原文写作「%25」（土耳其语），percent_word_first 表示读法放在数前（中文「百分之十」）
    percent = None
    percent_prefix = False
    percent_word_first = False
    # 单位：写法 → (单数读法, 复数读法)；货币：符号 → (单数读法, 复数读法)
    units = {}
    currencies = {}
    # 货币后面可以跟的数量级词（million、juta），读成「数 + 数量级词 + 货币」；
    # scale_links 给出数量级词与货币之间要加的词（西语 millones de dólares）
    scales = ()
    scale_links = {}

    def __init__(self):
        self.num = number_pattern(lang = self.code)
        # 允许紧跟在拉丁字母后的数字写法，用于 RM50、Rp50 000 这类货币前缀
        self.num_bare = number_pattern(lang = self.code, bare = True)

    # ------------------------------------------------------------------
    # 读法
    # ------------------------------------------------------------------

    def card(self, n: int) -> str:
        raise NotImplementedError

    def digits(self, digits: str) -> str:
        return " ".join(self.card(n = int(d)) for d in digits)

    def join(self, *parts: str) -> str:
        sep = " " if self.spaced else ""
        return sep.join(p for p in parts if p)

    def parse(self, written: str) -> Number | None:
        return parse_number(written = written, lang = self.code)

    def say(self, n: Number, ctx, card = None) -> str | Skip:
        """读一个数（整数或小数），并登记核对项。card 可以换成别的整数读法（例如阴性形式）。"""
        card = card or self.card
        if n.is_int:
            return ctx.num(spoken = card(n = n.value), written = n.written)
        if self.point is None:
            return Skip(reason = f"小数暂不处理「{n.written}」")
        if self.frac_as_number and len(n.frac) <= 2 and not n.frac.startswith("0"):
            frac = self.card(n = int(n.frac))
        else:
            frac = self.digits(digits = n.frac)
        return ctx.num(spoken = self.join(card(n = n.value), self.point, frac), written = n.written)

    def plural(self, n: Number) -> bool:
        return not (n.is_int and n.value == 1)

    def unit_word(self, n: Number, forms: tuple) -> str:
        return forms[1] if self.plural(n = n) else forms[0]

    # ------------------------------------------------------------------
    # 规则
    # ------------------------------------------------------------------

    def rules(self) -> list:
        return (
            guard_rules(lang = self.code)
            + self.extra_rules()
            + self.percent_rules()
            + self.unit_rules()
            + self.currency_rules()
            + [unknown_unit_rule(lang = self.code), code_prefix_rule(lang = self.code), colon_rule()]
            + adjacency_rules(lang = self.code)
            + [unparsable_rule(lang = self.code)]
            + [Rule(pattern = re.compile(pattern = self.num), handle = self.handle_cardinal)]
        )

    def extra_rules(self) -> list:
        return []

    def long_number(self, m, ctx, n):
        """中文、日语、韩语里 5 位以上不带千分位的数，后面不是文字（人、元、명 等）时多为编号，丢弃。"""
        if n.grouped or len(n.int_digits) < 5:
            return None
        rest = ctx.text[m.end():]
        if rest[:1].isalpha() and not rest[:1].isascii():
            return None
        return Skip(reason = f"长数字可能是编号「{m.group()}」")

    def handle_cardinal(self, m, ctx):
        n = self.parse(written = m.group())
        if n is None:
            return Skip(reason = f"数字写法无法确定「{m.group()}」")
        out = self.cardinal(m = m, ctx = ctx, n = n)
        if isinstance(out, str) and not ctx.kinds:
            ctx.kind(name = "基数" if n.is_int else "小数", surface = m.group())
        return out

    def cardinal(self, m, ctx, n: Number):
        """普通数字的读法。各语言按需要覆盖，没有把握时返回 Skip。"""
        return self.say(n = n, ctx = ctx)

    def percent_rules(self) -> list:
        if self.percent is None:
            return [Rule(
                pattern = re.compile(pattern = rf"{self.num}\s?[%٪]|[%٪]\s?{self.num}"),
                handle = lambda m, ctx: Skip(reason = f"百分数暂不处理「{m.group()}」"),
            )]
        if self.percent_prefix:
            pattern = rf"[%٪]\s?(?P<num>{self.num})"
        else:
            pattern = rf"(?P<num>{self.num})\s?[%٪]"

        def handle(m, ctx):
            n = self.parse(written = m.group("num"))
            if n is None:
                return Skip(reason = f"数字写法无法确定「{m.group()}」")
            out = self.percent_number(m = m, ctx = ctx, n = n)
            if isinstance(out, Skip):
                return out
            ctx.kind(name = "百分数", surface = m.group())
            if self.percent_prefix or self.percent_word_first:
                return self.join(self.percent, out)
            return self.join(out, self.percent)

        return [Rule(pattern = re.compile(pattern = pattern), handle = handle)]

    def percent_number(self, m, ctx, n: Number):
        return self.say(n = n, ctx = ctx)

    def unit_rules(self) -> list:
        if not self.units:
            return []
        pattern = rf"(?P<num>{self.num})\s?(?P<unit>{alternation(words = self.units)}){END}"

        def handle(m, ctx):
            n = self.parse(written = m.group("num"))
            if n is None:
                return Skip(reason = f"数字写法无法确定「{m.group()}」")
            out = self.unit_number(m = m, ctx = ctx, n = n)
            if isinstance(out, Skip):
                return out
            ctx.kind(name = "单位", surface = m.group())
            return self.join(out, self.unit_word(n = n, forms = self.units[m.group("unit")]))

        return [Rule(pattern = re.compile(pattern = pattern), handle = handle)]

    def unit_number(self, m, ctx, n: Number):
        """单位前的数。单位都是阳性名词，罗曼语族在这里改用阳性形式。"""
        return self.say(n = n, ctx = ctx)

    def currency_rules(self) -> list:
        if not self.currencies:
            return []
        symbols = alternation(words = self.currencies)
        scale = rf"(?:\s?(?P<scale>{alternation(words = self.scales)}){END})?" if self.scales else r"(?P<scale>)"
        prefix = rf"(?<![A-Za-z])(?P<cur>{symbols})\s?(?P<num>{self.num_bare}){scale}"
        suffix = rf"(?P<num>{self.num})\s?(?P<cur>{symbols}){END}"

        def handle(m, ctx):
            n = self.parse(written = m.group("num"))
            if n is None or not n.is_int:
                return Skip(reason = f"货币金额暂不处理「{m.group()}」")
            out = self.currency_number(m = m, ctx = ctx, n = n)
            if isinstance(out, Skip):
                return out
            groups = m.groupdict()
            scale_word = groups.get("scale")
            if "scale" in groups and not scale_word and word_after(text = ctx.text, end = m.end()) in ANY_SCALES:
                return Skip(reason = f"货币后的数量级词未登记「{m.group()} {word_after(text = ctx.text, end = m.end())}」")
            forms = self.currencies[m.group("cur")]
            # 「$45 millones de dólares」：符号和货币名重复，改写后会读两遍
            following = ctx.text[m.end():m.end() + 30].lower()
            if any(form.lower() in following for form in forms):
                return Skip(reason = f"货币符号与货币名重复「{m.group()}{ctx.text[m.end():m.end() + 15]}」")
            word = forms[1] if (scale_word or n.value != 1) else forms[0]
            ctx.kind(name = "货币", surface = m.group())
            return self.join(out, scale_word or "", self.scale_links.get(scale_word, ""), word)

        return [
            Rule(pattern = re.compile(pattern = prefix), handle = handle),
            Rule(pattern = re.compile(pattern = suffix), handle = handle),
        ]

    def currency_number(self, m, ctx, n: Number):
        return self.say(n = n, ctx = ctx)


def word_before(text: str, start: int) -> str:
    """前一个词（小写），没有时为空字符串。"""
    word = prev_word(text = text, start = start)
    return word.lower() if word else ""


def word_after(text: str, end: int) -> str:
    word = next_word(text = text, end = end)
    return word or ""
