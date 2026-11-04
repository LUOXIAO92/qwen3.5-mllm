"""改写引擎与各语言共用的规则。

一种语言的改写由一串规则组成，每条规则是一个正则和一个处理函数。引擎按顺序对每条规则查找匹配，
已被前面规则占用的位置不再处理。处理函数返回三种结果之一：

- 字符串：用它替换整个匹配；
- Skip(reason)：这一处没有把握，占住位置、保留原文，整条最终标 dropped，reason 写进 tn_note；
- None：规则不适用，留给后面的规则。

处理函数通过 Ctx.num(spoken, written) 写出数词，同时登记一对核对项：读法 spoken 与原文写法 written
随后用 pipeline.compare 比较，数值不一致的整条丢弃，用来拦住读法函数自身的错误。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, NamedTuple

# 拉丁（含越南语用到的扩展字母）、希腊、西里尔字母（与数字直接相连时多为型号、编号，例如 WIN350、B5500）
LATINISH = (
    f"A-Za-z{chr(0x00C0)}-{chr(0x024F)}{chr(0x1E00)}-{chr(0x1EFF)}"
    f"{chr(0x0370)}-{chr(0x03FF)}{chr(0x0400)}-{chr(0x04FF)}"
)
DASHES = "-‐‑‒–—―−~〜～"


class Skip(NamedTuple):
    reason: str


@dataclass
class Number:
    """一个阿拉伯数字写法。value 是整数部分，frac 是小数部分的数字串（没有小数时为 None）。"""

    written: str
    value: int
    frac: str | None
    int_digits: str
    grouped: bool

    @property
    def is_int(self) -> bool:
        return self.frac is None


@dataclass
class Ctx:
    lang: str
    text: str
    checks: list = field(default_factory = list)
    kinds: list = field(default_factory = list)

    def num(self, spoken: str, written: str) -> str:
        """登记一处数词读法，返回读法本身。"""
        self.checks.append((spoken, written))
        return spoken

    def kind(self, name: str, surface: str) -> None:
        self.kinds.append((name, surface))


@dataclass
class Rule:
    pattern: re.Pattern
    handle: Callable


# ---------------------------------------------------------------------------
# 数字写法
# ---------------------------------------------------------------------------

# 各语言的千分位与小数点。法语、俄语以空格分千位；阿拉伯语、波斯语另有 ٬（千分位）和 ٫（小数点）。
SEPARATORS = {
    "zh": (",", "."), "ja": (",", "."), "ko": (",", "."), "en": (",", "."), "th": (",", "."),
    "ms": (",", "."), "fil": (",", "."),
    "de": (".", ","), "id": (".", ","), "vi": (".", ","), "tr": (".", ","), "es": (".", ","),
    "pt": (".", ","), "it": (".", ","),
    "fr": (" ", ","), "ru": (" ", ","),
    "ar": (",٬", ".٫"), "fa": ("٬,", "٫."),
}
# 没有千分位时允许的最长位数；更长的数字多为编号、电话，丢弃
MAX_PLAIN_DIGITS = {"zh": 8, "ja": 8, "ko": 8}
DEFAULT_MAX_PLAIN = 4


def number_pattern(lang: str, bare: bool = False) -> str:
    """一个数字写法的正则：数字串，中间可以有该语言的千分位或小数点（分隔符后必须紧跟数字）。

    默认要求前面不是拉丁、西里尔字母，避免把 WIN350 中的 350 当成独立的数；bare 为 True 时不加这条限制。
    """
    thousands, decimal = SEPARATORS[lang]
    seps = re.escape(pattern = thousands + decimal)
    before = r"(?<!\d)" if bare else rf"(?<![\d{LATINISH}])"
    return rf"{before}\d+(?:[{seps}]\d+)*(?!\d)"


def ascii_digits(text: str) -> str:
    return "".join(str(unicodedata.digit(ch)) if ch.isdigit() else ch for ch in text)


def parse_number(written: str, lang: str) -> Number | None:
    """按该语言的分隔符习惯解析数字写法，写法有歧义或不合规时返回 None。"""
    thousands, decimal = SEPARATORS[lang]
    text = ascii_digits(text = written)
    frac = None
    dec_positions = [i for i, ch in enumerate(text) if ch in decimal]
    if len(dec_positions) > 1:
        return None
    if dec_positions:
        i = dec_positions[0]
        text, frac = text[:i], text[i + 1:]
        if not frac.isdigit():
            return None
    groups = re.split(pattern = f"[{re.escape(pattern = thousands)}]", string = text)
    if len(groups) > 1:
        if not (1 <= len(groups[0]) <= 3) or any(len(g) != 3 for g in groups[1:]) or groups[0].startswith("0"):
            return None
    digits = "".join(groups)
    if not digits.isdigit():
        return None
    if len(groups) == 1 and len(digits) > MAX_PLAIN_DIGITS.get(lang, DEFAULT_MAX_PLAIN):
        return None
    if len(digits) > 1 and digits.startswith("0"):
        return None
    return Number(
        written = written,
        value = int(digits),
        frac = frac,
        int_digits = digits,
        grouped = len(groups) > 1,
    )


# ---------------------------------------------------------------------------
# 上下文
# ---------------------------------------------------------------------------

_WORD = re.compile(pattern = r"[^\W\d_]+(?:['’-][^\W\d_]+)*")
_PREV_WORD = re.compile(pattern = r"([^\W\d_]+(?:['’][^\W\d_]+)*\.?)\s*$")


def prev_word(text: str, start: int) -> str | None:
    """匹配前面紧邻的一个词（中间只隔空白，词尾可以带缩写句点），没有时返回 None。"""
    m = _PREV_WORD.search(string = text[:start])
    return m.group(1) if m else None


def next_word(text: str, end: int) -> str | None:
    """匹配后面紧邻的一个词（中间只隔空白），没有时返回 None。"""
    rest = text[end:]
    m = _WORD.match(string = rest.lstrip())
    return m.group() if m else None


# ---------------------------------------------------------------------------
# 共用规则
# ---------------------------------------------------------------------------

def guard_rules(lang: str) -> list:
    """最先运行的保护规则：范围、日期和分数写法、负号、撇号后缀、空格分隔的数字、连续的数字。"""
    num = number_pattern(lang = lang)
    dash = re.escape(pattern = DASHES)
    rules = [
        Rule(
            pattern = re.compile(pattern = r"\d{1,4}[./-]\d{1,2}[./-]\d{1,4}"),
            handle = lambda m, ctx: Skip(reason = f"日期写法「{m.group()}」"),
        ),
        Rule(
            pattern = re.compile(pattern = rf"{num}\s*[{dash}]\s*{num}"),
            handle = lambda m, ctx: Skip(reason = f"范围或编号「{m.group()}」"),
        ),
        Rule(
            pattern = re.compile(pattern = rf"(?<![\w.,])[{dash}]{num}"),
            handle = lambda m, ctx: Skip(reason = f"负号或连字符「{m.group()}」"),
        ),
        Rule(
            pattern = re.compile(pattern = r"\d+\s*/\s*\d+"),
            handle = lambda m, ctx: Skip(reason = f"分数或斜线写法「{m.group()}」"),
        ),
        Rule(
            pattern = re.compile(pattern = rf"{num}['’]\w+"),
            handle = lambda m, ctx: Skip(reason = f"数字后接撇号后缀「{m.group()}」"),
        ),
        Rule(
            pattern = re.compile(pattern = rf"['’‘]{num}"),
            handle = lambda m, ctx: Skip(reason = f"撇号省略的年份「{m.group()}」"),
        ),
        Rule(
            # 字母、连字符、数字（X-1、C-47、COVID-19）多为型号；印尼语、马来语的 ke-21 由各语言规则处理
            pattern = re.compile(pattern = rf"(?<![\w-])(?!ke-)[{LATINISH}]+[{dash}]{num}"),
            handle = lambda m, ctx: Skip(reason = f"型号或编号「{m.group()}」"),
        ),
        Rule(
            # 句号、逗号后缺空格就接数字（angetreten.1998）
            pattern = re.compile(pattern = rf"(?<=[^\W\d_][.,]){num}"),
            handle = lambda m, ctx: Skip(reason = f"标点后缺少空格「{ctx.text[max(0, m.start() - 2):m.end()]}」"),
        ),
    ]
    if " " not in SEPARATORS[lang][0]:
        # 不以空格分千位的语言里出现「2 880」「34 400」：多半是外来写法的千分位，不能拆成两个数
        rules.append(Rule(
            pattern = re.compile(pattern = r"(?<![\d.,])\d{1,3}(?: \d{3})+(?!\d)"),
            handle = lambda m, ctx: Skip(reason = f"空格分隔的数字「{m.group()}」"),
        ))

    def consecutive(m, ctx):
        # 法语、俄语的「600 000」是一个数，交给后面的规则
        if parse_number(written = m.group(), lang = lang) is not None:
            return None
        return Skip(reason = f"连续的数字「{m.group()}」")

    # 两个数字之间只隔一个空格（52 3048、2006 300）：多为编号或两个量，读法无法确定
    rules.append(Rule(pattern = re.compile(pattern = rf"{num} {num}"), handle = consecutive))
    return rules


def colon_rule() -> Rule:
    """冒号与数字紧贴：时刻由各语言规则先处理，剩下的（脚注、比分、编号）丢弃。"""
    return Rule(
        pattern = re.compile(pattern = r"\d*[:：]\d+|\d+[:：]"),
        handle = lambda m, ctx: Skip(reason = f"冒号与数字相连「{m.group().strip()}」"),
    )


def adjacency_rules(lang: str) -> list:
    """数字与拉丁、西里尔字母直接相连（型号、编号），或与文字用连字符相连。"""
    num = number_pattern(lang = lang, bare = True)
    dash = re.escape(pattern = DASHES)
    return [
        Rule(
            pattern = re.compile(pattern = rf"[{LATINISH}]+{num}[{LATINISH}]*|{num}[{LATINISH}]+"),
            handle = lambda m, ctx: Skip(reason = f"数字与字母相连「{m.group()}」"),
        ),
        Rule(
            pattern = re.compile(pattern = rf"[^\W\d_]+[{dash}]{num}|{num}[{dash}][^\W\d_]+"),
            handle = lambda m, ctx: Skip(reason = f"数字与文字用连字符相连「{m.group()}」"),
        ),
    ]


# 未登记读法的单位缩写：数字后接这些写法时丢弃（已登记的单位由各语言的单位规则先处理）
UNKNOWN_UNITS = (
    "h", "min", "s", "ms", "Hz", "kHz", "MHz", "GHz", "kW", "MW", "GW", "kWh", "MWh", "W", "V", "kV", "mA",
    "ml", "dl", "cl", "l", "L", "t", "ha", "PS", "hp", "bar", "hPa", "Pa", "kn", "nm", "µm", "μm", "Mbit",
    "kbit", "MB", "GB", "TB", "KB", "kB", "Mbps", "yd", "oz", "lb", "lbs", "cal", "kcal", "kJ", "J", "N",
    "rpm", "dB", "ppm", "ppb", "mol", "K", "mi", "ft", "km", "m", "cm", "mm", "kg", "g", "mg", "µg", "km²",
    "m²", "km/h", "m/s", "°C", "°F", "℃", "℉", "°",
)


def unknown_unit_rule(lang: str) -> Rule:
    num = number_pattern(lang = lang)
    units = alternation(words = UNKNOWN_UNITS)
    return Rule(
        pattern = re.compile(pattern = rf"{num}\s?(?:{units})(?![{LATINISH}])"),
        handle = lambda m, ctx: Skip(reason = f"单位读法未登记「{m.group()}」"),
    )


def code_prefix_rule(lang: str) -> Rule:
    """前面是全大写缩写的数字多为编号或型号（PA 225、AL 35），读法不固定。"""
    num = number_pattern(lang = lang)
    return Rule(
        pattern = re.compile(pattern = rf"(?<![\w])[A-Z]{{2,}}\s{num}"),
        handle = lambda m, ctx: Skip(reason = f"大写缩写后的编号「{m.group()}」"),
    )


def unparsable_rule(lang: str) -> Rule:
    """无法按该语言习惯解析的数字写法（例如英语里的 1,5、德语里的 2.30、过长的数字串）。"""
    num = number_pattern(lang = lang)

    def handle(m, ctx):
        if parse_number(written = m.group(), lang = lang) is None:
            return Skip(reason = f"数字写法无法确定「{m.group()}」")
        return None

    return Rule(pattern = re.compile(pattern = num), handle = handle)


def alternation(words) -> str:
    """把一组写法拼成正则分支，长的在前。"""
    return "|".join(re.escape(pattern = w) for w in sorted(words, key = len, reverse = True))


# ---------------------------------------------------------------------------
# 引擎
# ---------------------------------------------------------------------------

class Rewrite(NamedTuple):
    """改写结果。

    replaced 是被读法替换的原文片段 (start, end)；skips 是 (原因, (start, end))，片段保留原文。
    两者都按原文的位置记录，用来判断原文中需要改写的片段是否都已处理。
    """

    text: str
    replaced: list
    skips: list
    checks: list
    kinds: list


# 数词与前后文字之间不加空格的语言
UNSPACED = ("zh", "ja", "th", "ko")


def _capitalize(out: str, text: str, start: int, lang: str) -> str:
    """句首的数字改成读法后首字母大写（1920 soll sie … → Neunzehnhundertzwanzig soll sie …）。"""
    before = text[:start].rstrip()
    if out[:1].islower() and (not before or before[-1] in ".!?"):
        first = "İ" if (lang == "tr" and out[0] == "i") else out[0].upper()
        return first + out[1:]
    return out


def _pad(out: str, text: str, start: int, end: int, lang: str) -> str:
    """读法与原文中紧贴的字母之间补空格（alımlarında%5 → alımlarında yüzde beş）。"""
    if lang in UNSPACED:
        return out
    if start > 0 and text[start - 1].isalpha() and out[:1].isalpha():
        out = " " + out
    if end < len(text) and text[end].isalpha() and out[-1:].isalpha():
        out = out + " "
    return out


def apply_rules(text: str, lang: str, rules: list) -> Rewrite:
    taken = []
    replacements = []
    skips = []
    checks = []
    kinds = []
    for rule in rules:
        for m in rule.pattern.finditer(string = text):
            start, end = m.span()
            if start == end or any(start < b and a < end for a, b in taken):
                continue
            ctx = Ctx(lang = lang, text = text)
            out = rule.handle(m = m, ctx = ctx)
            if out is None:
                continue
            taken.append((start, end))
            if isinstance(out, Skip):
                skips.append((out.reason, (start, end)))
                continue
            out = _capitalize(out = out, text = text, start = start, lang = lang)
            replacements.append((start, end, _pad(out = out, text = text, start = start, end = end, lang = lang)))
            checks.extend(ctx.checks)
            kinds.extend(ctx.kinds)
    pieces = []
    pos = 0
    for start, end, out in sorted(replacements):
        pieces.append(text[pos:start])
        pieces.append(out)
        pos = end
    pieces.append(text[pos:])
    # 读法替换后可能出现连续空格
    new_text = re.sub(pattern = r"[ ]{2,}", repl = " ", string = "".join(pieces))
    return Rewrite(
        text = new_text,
        replaced = [(start, end) for start, end, _ in replacements],
        skips = skips,
        checks = checks,
        kinds = kinds,
    )
