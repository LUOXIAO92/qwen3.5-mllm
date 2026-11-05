"""土耳其语：数词不变形，分开写（iki bin on）；「4. sınıf」读 dördüncü sınıf；「%25」读 yüzde yirmi beş。

千分位为「.」，小数点为「,」。数字后接撇号后缀（2006'da、%65'i、4'ü、7'nci、1990'lı）时，把后缀接到读法的
最后一个词上（iki bin altıda、yüzde altmış beşi、dördü、yedinci、bin dokuz yüz doksanlı）。原文的后缀按读音
写成，因此先按读法最后一个词的元音和谐、清浊同化生成全部合法后缀，原文后缀在其中才改写，否则说明原文写错，
丢弃。序数后缀与数字之间多了空格（119 uncu madde）时同样合并；「17.-18. yüzyıllar」这类序数范围丢弃。dört 在元音开头的后缀前变为 dörd，-lık 在元音前变为 -lığ。

时刻：「saat 14:30」「06:30'da」「2.30'a」读 on dört otuz、altı otuzda、iki otuza，整点只读小时（3:00 → üç）；
冒号写法要求前面有 saat 或后面接后缀，句点写法要求前面有 saat 或后面接后缀；「1:58'lik」这类时长丢弃。
「2. Dünya Savaşı」「23. Tümen」「II. Dünya Savaşı」在登记的名词前读序数。「-40 derece」读 eksi kırk derece。
句号后缺少空格（açar.Eğer）时补上空格。
"""

from __future__ import annotations

import re
from functools import lru_cache

from ..numwords import tr_cardinal, tr_digits, tr_ordinal
from ..rules import DASHES, Rule, Skip
from .base import Lang
from .extras import common_rules, missing_space_rule, negative_rule

VOWELS = "aeıioöuü"
BACK = "aıou"
VOICELESS = "çfhkpsşt"
_I4 = {"a": "ı", "ı": "ı", "o": "u", "u": "u", "e": "i", "i": "i", "ö": "ü", "ü": "ü"}
_SUFFIX_CHARS = "a-zçğıöşü"

# 后缀模板：A = a/e，I = ı/i/u/ü，D = d/t；括号中的辅音只在元音后出现，(I) 只在辅音后出现
_DERIVATION = ["", "lIk", "lI", "(I)ncI"]
_PLURAL = ["", "lAr"]
_POSSESSIVE = ["", "(s)I", "(I)m", "(I)mIz"]
_CASES = ["", "(y)I", "(y)A", "DA", "DAn", "(n)In", "(y)lA", "DAki", "DIr", "(y)DI", "(y)mIş"]
# 第三人称领属后的格词尾带 n：sini、sine、sinde、sinden、sinin、sindeki
_CASES_AFTER_POSSESSIVE = ["", "nI", "nA", "nDA", "nDAn", "nIn", "(y)lA", "nDAki", "DIr", "(y)DI"]
_TOKEN = re.compile(pattern = r"\([a-zşI]\)|[AID]|[a-zçğıöşü]")
# 读序数的名词（数字加句点之后，按词首匹配，包括 Kolorduya、Tümeni 这类带词尾的形式）
ORDINAL_NOUNS = {
    "Dünya", "Tümen", "Tümeni", "Kolordu", "Kolordusu", "Ordu", "Ordusu", "Alay", "Alayı", "Tugay", "Tugayı",
    "Kafkas", "Lig", "Ligi", "Dönem", "Kuşak", "Filo", "Filosu",
}
_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}


def _last_vowel(text: str) -> str | None:
    for ch in reversed(text):
        if ch in VOWELS:
            return ch
    return None


def _realize(form: str, template: str) -> str:
    """按 form 的读音把一个后缀模板写成实际后缀。"""
    out = ""
    for token in _TOKEN.findall(string = template):
        cur = form + out
        last = cur[-1]
        vowel = _last_vowel(text = cur)
        if token.startswith("(") and token != "(I)":
            if last in VOWELS:
                out += token[1]
        elif token == "(I)":
            if last not in VOWELS:
                out += _I4[vowel]
        elif token == "A":
            out += "a" if vowel in BACK else "e"
        elif token == "I":
            out += _I4[vowel]
        elif token == "D":
            out += "t" if last in VOICELESS else "d"
        else:
            out += token
    return out


def _attach(form: str, suffix: str) -> str:
    """接上一个已写成的后缀，元音前 dört → dörd、-lık → -lığ。"""
    if suffix[:1] in VOWELS:
        if form.endswith("dört"):
            form = form[:-1] + "d"
        elif re.search(pattern = r"l[ıiuü]k$", string = form):
            form = form[:-1] + "ğ"
    return form + suffix


@lru_cache(maxsize = 4096)
def suffix_forms(word: str) -> dict:
    """一个数词（读法的最后一个词）后面所有合法的后缀写法 → 接上后缀后的读法。"""
    forms = {}
    for derivation in _DERIVATION:
        for plural in _PLURAL:
            for possessive in _POSSESSIVE:
                cases = _CASES_AFTER_POSSESSIVE if possessive == "(s)I" else _CASES
                for case in cases:
                    form = word
                    for template in (derivation, plural, possessive, case):
                        if template:
                            form = _attach(form = form, suffix = _realize(form = form, template = template))
                    # 原文写在撇号后的部分：dördü 写作 4'ü（dört 到 dörd 的变化不改变长度）
                    written = form[len(word):]
                    if written and written not in forms:
                        forms[written] = form
    # 分配数词：beşer、ikişer、dörder
    for template in ("(ş)Ar",):
        form = _attach(form = word, suffix = _realize(form = word, template = template))
        forms.setdefault(form[len(word):], form)
    return forms


def with_suffix(spoken: str, suffix: str) -> str | None:
    """把原文撇号后的后缀接到读法上；后缀与读法的元音和谐不符时返回 None。"""
    head, _, last = spoken.rpartition(" ")
    form = suffix_forms(word = last).get(suffix)
    if form is None:
        return None
    return (head + " " + form) if head else form


class Tr(Lang):
    code = "tr"
    point = "virgül"
    # 小数部分为 1–2 位时按整数读（3,14 → üç virgül on dört）
    frac_as_number = True
    percent = "yüzde"
    percent_prefix = True
    units = {
        "km²": ("kilometrekare", "kilometrekare"), "m²": ("metrekare", "metrekare"), "km": ("kilometre", "kilometre"),
        "cm": ("santimetre", "santimetre"), "mm": ("milimetre", "milimetre"), "kg": ("kilogram", "kilogram"),
        "m": ("metre", "metre"), "g": ("gram", "gram"), "°C": ("santigrat derece", "santigrat derece"),
        "℃": ("santigrat derece", "santigrat derece"),
    }
    currencies = {
        "$": ("dolar", "dolar"), "£": ("sterlin", "sterlin"), "₺": ("Türk lirası", "Türk lirası"),
        "TL": ("Türk lirası", "Türk lirası"),
    }
    scales = ("bin", "milyon", "milyar")

    def card(self, n: int) -> str:
        return tr_cardinal(n = n)

    def digits(self, digits: str) -> str:
        return tr_digits(digits = digits)

    def rules(self) -> list:
        """后缀、时刻、负温度先于公共保护规则运行（公共规则会把 2006'da、-40 丢弃）。"""
        return common_rules(lang = self, list_numbers = False) + self.pre_rules() + super().rules()

    def pre_rules(self) -> list:
        num = self.num
        dash = re.escape(pattern = DASHES)
        apostrophe = "['’]"
        suffix = rf"(?P<suf>[{_SUFFIX_CHARS}]+)(?![\w'’])"
        before = rf"(?<![\d{dash}./:,'’\w])"
        return [
            # 「17.-18. yüzyıllar」：序数范围的读法（ve、ile、连读）不确定，丢弃
            Rule(
                pattern = re.compile(pattern = rf"(?<![\d.,])\d{{1,4}}\.\s?[{dash}]\s?\d{{1,4}}\.(?!\d)"),
                handle = lambda m, ctx: Skip(reason = f"序数范围「{m.group()}」"),
            ),
            Rule(
                pattern = re.compile(pattern = rf"{before}(?P<saat>[Ss]aat\s)?(?P<h>\d{{1,2}})(?P<sep>[:.])(?P<m>\d{{2}})(?:{apostrophe}{suffix})?(?!\d|[.,:]\d)"),
                handle = self.time,
            ),
            Rule(
                pattern = re.compile(pattern = rf"(?<![\w.,])[%٪]\s?(?P<num>{num}){apostrophe}{suffix}"),
                handle = self.percent_suffix,
            ),
            Rule(
                pattern = re.compile(pattern = rf"{before}(?P<num>{num}){apostrophe}{suffix}"),
                handle = self.number_suffix,
            ),
            # 「119 uncu madde」：序数后缀与数字之间多了空格，后缀与读音和谐时合并（yüz on dokuzuncu）
            Rule(
                pattern = re.compile(pattern = rf"{before}(?P<num>{num})\s(?P<suf>[ıiuü]?nc[ıiuü](?:s[ıiuü]|[ıiuü]n|y[ae]|d[ae])?)(?![\w'’])"),
                handle = self.number_suffix,
            ),
            Rule(
                pattern = re.compile(pattern = rf"{num}{apostrophe}\s"),
                handle = lambda m, ctx: Skip(reason = f"数字后的撇号与后缀分开「{m.group().strip()}」"),
            ),
            negative_rule(lang = self, minus = "eksi", words = ("derece",)),
            Rule(
                pattern = re.compile(pattern = r"(?<![\w.])(?P<roman>[IVX]{1,4})\.\s(?=(?P<next>[A-ZÇĞİÖŞÜ][a-zçğıöşü]+))"),
                handle = self.roman_ordinal,
            ),
        ]

    def extra_rules(self) -> list:
        return [
            missing_space_rule(left = "a-zçğıöşü", right = "[A-ZÇĞİÖŞÜ][a-zçğıöşü]+"),
            Rule(
                pattern = re.compile(pattern = r"(?<![\d.,])(?P<num>\d{1,4})\.(?=\s[a-zçğıöşü])"),
                handle = self.ordinal,
            ),
            Rule(
                pattern = re.compile(pattern = r"(?<![\d.,])(?P<num>\d{1,4})\.(?=\s(?P<next>[A-ZÇĞİÖŞÜ][a-zçğıöşü]+))"),
                handle = self.ordinal_before_name,
            ),
        ]

    # ------------------------------------------------------------------

    def say_number(self, written: str, ctx):
        n = self.parse(written = written)
        if n is None:
            return Skip(reason = f"数字写法无法确定「{written}」")
        return self.say(n = n, ctx = ctx)

    def number_suffix(self, m, ctx):
        spoken = self.say_number(written = m.group("num"), ctx = ctx)
        if isinstance(spoken, Skip):
            return spoken
        suffix = m.group("suf")
        # 序数后缀：7'nci → yedinci，读法换成序数
        if re.fullmatch(pattern = r"[ıiuü]?nc[ıiuü]", string = suffix):
            n = self.parse(written = m.group("num"))
            if not n.is_int:
                return Skip(reason = f"序数写法无法确定「{m.group()}」")
            ordinal = tr_ordinal(n = n.value)
            ctx.checks[-1] = (ordinal, m.group("num"))
            if with_suffix(spoken = spoken, suffix = suffix) != ordinal:
                return Skip(reason = f"序数后缀与读音不符「{m.group()}」")
            ctx.kind(name = "序数", surface = m.group())
            return ordinal
        out = with_suffix(spoken = spoken, suffix = suffix)
        if out is None:
            return Skip(reason = f"撇号后缀与数词读音不和谐「{m.group()}」")
        ctx.kind(name = "带后缀的数", surface = m.group())
        return out

    def percent_suffix(self, m, ctx):
        spoken = self.say_number(written = m.group("num"), ctx = ctx)
        if isinstance(spoken, Skip):
            return spoken
        out = with_suffix(spoken = spoken, suffix = m.group("suf"))
        if out is None:
            return Skip(reason = f"撇号后缀与数词读音不和谐「{m.group()}」")
        ctx.kind(name = "百分数", surface = m.group())
        return self.percent + " " + out

    def time(self, m, ctx):
        h, mm = int(m.group("h")), int(m.group("m"))
        saat, suffix = m.group("saat"), m.group("suf")
        if not (0 <= h <= 24 and 0 <= mm <= 59):
            return None
        if m.group("sep") == "." and not (saat or suffix):
            return None
        if not (saat or suffix):
            return Skip(reason = f"冒号写法无法确定是否为时刻「{m.group()}」")
        if suffix and re.match(pattern = r"l[ıiuü]", string = suffix):
            return Skip(reason = f"时长写法「{m.group()}」")
        hour = ctx.num(spoken = self.card(n = h), written = m.group("h"))
        if mm == 0:
            spoken = hour
        elif mm < 10:
            spoken = hour + " sıfır " + ctx.num(spoken = self.card(n = mm), written = m.group("m")[1])
        else:
            spoken = hour + " " + ctx.num(spoken = self.card(n = mm), written = m.group("m"))
        if suffix:
            spoken = with_suffix(spoken = spoken, suffix = suffix)
            if spoken is None:
                return Skip(reason = f"撇号后缀与数词读音不和谐「{m.group()}」")
        ctx.kind(name = "时刻", surface = m.group())
        return (saat or "") + spoken

    def roman_ordinal(self, m, ctx):
        value = _ROMAN.get(m.group("roman"))
        if value is None or not m.group("next").startswith(tuple(ORDINAL_NOUNS)):
            return None
        ctx.kind(name = "序数", surface = m.group().strip())
        return ctx.num(spoken = tr_ordinal(n = value), written = f"{value}.") + " "

    def ordinal(self, m, ctx):
        n = int(m.group("num"))
        ctx.kind(name = "序数", surface = m.group())
        return ctx.num(spoken = tr_ordinal(n = n), written = m.group())

    def ordinal_before_name(self, m, ctx):
        if not m.group("next").startswith(tuple(ORDINAL_NOUNS)):
            return Skip(reason = f"数字后的句点可能表示序数（2. Dünya Savaşı）「{m.group()}」")
        return self.ordinal(m = m, ctx = ctx)


LANG = Tr
