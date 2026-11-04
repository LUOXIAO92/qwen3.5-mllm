"""俄语：数词的格由上下文决定，只处理两类能确定格的写法，其余丢弃。

1. 年份后接「год」的各种形式：格由 год 的词形给出（2008 года → две тысячи восьмого года，
   в 1950-х годах → в тысяча девятьсот пятидесятых годах）。「году」既可能是前置格（в 2025 году）
   也可能是与格（к 2025 году），按前面的介词区分。
2. 支配属格的介词或比较词之后、复数属格名词之前的基数：более 600 000 копий → более шестисот тысяч копий。
   末位为 1 的数（одного/одной）随名词变性，丢弃；后面是单数属格（до 40 этажа）时是序数，也丢弃。

单位、百分数、小数、日期（13 октября）的格同样取决于上下文，暂不处理。
"""

from __future__ import annotations

import re

from ..numwords import n2w
from ..rules import Rule, Skip, alternation
from .base import Lang, word_after, word_before

# год 的词形 → (num2words 的格, 是否复数)
GOD_FORMS = {
    "году": (None, False), "года": ("genitive", False), "год": ("nominative", False),
    "годом": ("instrumental", False), "годах": ("prepositional", True), "годов": ("genitive", True),
    "годы": ("nominative", True), "годам": ("dative", True), "годами": ("instrumental", True),
}
PREPOSITIONAL_BEFORE = {"в", "во", "о", "об", "при"}
DATIVE_BEFORE = {"к", "ко", "по"}
# 复数属格常见词尾（копий、человек 等零词尾的词单独列出），以及形容词复数属格 -ых、-их
GENITIVE_PLURAL_END = ("ов", "ев", "ёв", "ей", "ий", "ых", "их")
GENITIVE_PLURAL_WORDS = {"человек", "лет", "раз", "тысяч", "школ", "книг", "солдат", "штук", "минут", "недель"}
GENITIVE_BEFORE = {
    "более", "менее", "около", "свыше", "от", "до", "из", "без", "для", "после", "кроме", "среди", "у",
    "порядка", "больше", "меньше", "нескольких",
}


class Ru(Lang):
    code = "ru"

    def card(self, n: int) -> str:
        return n2w(n = n, lang = "ru")

    def extra_rules(self) -> list:
        forms = alternation(words = GOD_FORMS)
        return [
            Rule(
                pattern = re.compile(
                    pattern = rf"(?<![\d.,])(?P<num>\d{{3,4}})(?P<suf>-(?:х|е|м|ми|го|му|й))?\s(?P<god>{forms})(?![а-яё])",
                ),
                handle = self.year,
            ),
        ]

    def year(self, m, ctx):
        n = int(m.group("num"))
        case, plural = GOD_FORMS[m.group("god")]
        if case is None:
            before = word_before(text = ctx.text, start = m.start())
            if before in PREPOSITIONAL_BEFORE:
                case = "prepositional"
            elif before in DATIVE_BEFORE:
                case = "dative"
            else:
                return Skip(reason = f"「году」的格需要前面的介词判断「{m.group()}」")
        if plural != bool(m.group("suf") and m.group("suf") in ("-х", "-е", "-м", "-ми")) and m.group("suf"):
            return Skip(reason = f"年代写法与 год 的词形不一致「{m.group()}」")
        spoken = n2w(n = n, lang = "ru", to = "ordinal", case = case, plural = plural)
        ctx.num(spoken = spoken, written = m.group("num"))
        ctx.kind(name = "年代" if plural else "年份", surface = m.group())
        return spoken + " " + m.group("god")

    def cardinal(self, m, ctx, n):
        before = word_before(text = ctx.text, start = m.start())
        if before not in GENITIVE_BEFORE:
            return Skip(reason = f"俄语数词的格取决于上下文「{m.group()}」")
        if not n.is_int:
            return Skip(reason = f"小数的格变化暂不处理「{m.group()}」")
        if n.value % 10 == 1 and n.value % 100 != 11:
            return Skip(reason = f"末位为 1 的数随名词变性「{m.group()}」")
        after = word_after(text = ctx.text, end = m.end()).lower()
        if not (after.endswith(GENITIVE_PLURAL_END) or after in GENITIVE_PLURAL_WORDS):
            # 「до 40 этажа」后接单数属格，是序数（сорокового）
            return Skip(reason = f"后面的名词不像复数属格，可能是序数「{m.group()} {after}」")
        spoken = n2w(n = n.value, lang = "ru", case = "genitive")
        # одной тысячи пятисот → тысячи пятисот
        spoken = re.sub(pattern = r"^(?:одной (?=тысячи)|одного (?=миллиона|миллиарда))", repl = "", string = spoken)
        ctx.kind(name = "属格基数", surface = m.group())
        return ctx.num(spoken = spoken, written = n.written)


LANG = Ru
