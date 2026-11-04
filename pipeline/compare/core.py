"""比较入口：compare 与 normalize_for_scoring。"""

from __future__ import annotations

import difflib
import re
import unicodedata
from typing import NamedTuple

from .langs import get_spec
from .normalize import normalize_text
from .parse import Mark, Num, Text, analyze

# 一个阿拉伯数字最多和几个连续的读法记号拼接比较（例如 nineteen eighty-four ↔ 1984）
MAX_CONCAT = 8
MAX_DIFFS = 5


class CompareResult(NamedTuple):
    match: bool
    detail: str


# ---------------------------------------------------------------------------
# 逐单元比较
# ---------------------------------------------------------------------------

def _units(items: list) -> list:
    """普通文字拆成单个字符（比较时忽略空白），记号保持为一个单元。"""
    out = []
    for it in items:
        if isinstance(it, Text):
            out.extend(it.text)
        else:
            out.append(it)
    return out


def _zeros_ok(digits: str) -> bool:
    """前导零只容许「09」这类两位写法（时刻、日期）；「00」「007」必须逐位对应读法。"""
    return digits[0] != "0" or len(digits) == 1 or (len(digits) == 2 and digits[1] != "0")


def _eq(a, b) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    if isinstance(a, Num) and isinstance(b, Num):
        if a.origin == "word" and b.origin == "word":
            # 两边都是读法：要求词形一致（只合并词表中登记的书写变体）
            return a.surface == b.surface
        if not (a.kinds & b.kinds) or a.frac != b.frac:
            return False
        if a.frac:
            return a.digits == b.digits
        return a.value == b.value and _zeros_ok(digits = a.digits) and _zeros_ok(digits = b.digits)
    if isinstance(a, Mark) and isinstance(b, Mark):
        if a.kind != b.kind:
            return False
        if a.origin == "word" and b.origin == "word":
            return a.surface == b.surface
        return True
    return False


def _concat_ok(d, seq) -> bool:
    """一个阿拉伯数字对应几个连续的读法记号：逐段读法拼起来等于该数字串。"""
    if not isinstance(d, Num) or d.origin != "digit" or "card" not in d.kinds:
        return False
    parts = []
    for w in seq:
        if not isinstance(w, Num) or w.origin != "word" or "card" not in w.kinds or w.frac != d.frac:
            return False
        parts.append(w.digits)
    return "".join(parts) == d.digits


def _match_units(a: list, b: list, lang: str) -> bool:
    """两串比较单元能否逐一对应（允许拼接对应和泰语重复符）。"""
    n, m = len(a), len(b)
    stack = [(0, 0)]
    seen = {(0, 0)}
    while stack:
        i, j = stack.pop()
        if i == n and j == m:
            return True
        if i >= n or j >= m:
            continue
        x, y = a[i], b[j]
        succ = []
        if _eq(a = x, b = y):
            succ.append((i + 1, j + 1))
        for k in range(2, min(MAX_CONCAT, m - j) + 1):
            if _concat_ok(d = x, seq = b[j:j + k]):
                succ.append((i + 1, j + k))
        for k in range(2, min(MAX_CONCAT, n - i) + 1):
            if _concat_ok(d = y, seq = a[i:i + k]):
                succ.append((i + k, j + 1))
        if lang == "th":
            forward = _mai_yamok(
                a = a,
                b = b,
                i = i,
                j = j,
            )
            backward = _mai_yamok(
                a = b,
                b = a,
                i = j,
                j = i,
            )
            succ.extend(forward)
            succ.extend((q, p) for p, q in backward)
        for state in succ:
            if state not in seen:
                seen.add(state)
                stack.append(state)
    return False


_TH_LEADING = set("เแโใไ")
_TH_FOLLOWING = set("ะาำๅ")


def _th_cuts_syllable(units: list, start: int, end: int) -> bool:
    """units[start:end] 的两端是否切在泰文音节内部（前置元音、上下标符号、后置元音）。"""
    first = units[start]
    if start > 0 and units[start - 1] in _TH_LEADING:
        return True
    if unicodedata.category(first) == "Mn" or first in _TH_FOLLOWING:
        return True
    if end < len(units) and isinstance(units[end], str):
        nxt = units[end]
        if unicodedata.category(nxt) == "Mn" or nxt in _TH_FOLLOWING:
            return True
    return units[end - 1] in _TH_LEADING


def _mai_yamok(a: list, b: list, i: int, j: int) -> list:
    """泰语重复符 ๆ：a 中的 ๆ 对应 b 中把前面完整的 2 个以上字符再写一遍。"""
    if a[i] != "ๆ":
        return []
    out = []
    for r in range(2, 13):
        if j - r < 0 or i - r < 0 or j + r > len(b):
            break
        prev = b[j - r:j]
        if not all(isinstance(c, str) for c in prev):
            break
        if b[j:j + r] != prev or a[i - r:i] != prev:
            continue
        if _th_cuts_syllable(units = b, start = j - r, end = j) or _th_cuts_syllable(units = b, start = j, end = j + r):
            continue
        out.append((i + 1, j + r))
    return out


# ---------------------------------------------------------------------------
# 差异说明
# ---------------------------------------------------------------------------

def _key(it) -> str:
    if isinstance(it, Text):
        return "T" + it.text
    if isinstance(it, Num):
        return f"N{it.digits}|{it.surface if it.origin == 'word' else ''}"
    return f"M{it.kind}|{it.surface if it.origin == 'word' else ''}"


def _show(items: list, joiner: str) -> str:
    """把一段比较单元写成便于阅读的文字（规范化后的形式）。"""
    parts = []
    glue_next = False
    for it in items:
        glue = glue_next
        glue_next = False
        if isinstance(it, Text):
            token = it.text
        elif isinstance(it, Num):
            token = it.display
            glue = glue or (it.frac and it.origin != "word")
        else:
            token = it.surface
            if it.origin == "sym":
                glue = it.kind != "minus"
                glue_next = it.kind != "pct"
        if glue and parts:
            parts[-1] += token
        else:
            parts.append(token)
    text = joiner.join(parts)
    return text if len(text) <= 40 else text[:37] + "…"


def _diff(s_items: list, h_items: list, lang: str, joiner: str) -> list:
    matcher = difflib.SequenceMatcher(
        isjunk = None,
        a = [_key(it = it) for it in s_items],
        b = [_key(it = it) for it in h_items],
        autojunk = False,
    )
    out = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        a, b = s_items[i1:i2], h_items[j1:j2]
        if _match_units(a = _units(items = a), b = _units(items = b), lang = lang):
            continue
        if tag == "replace":
            out.append(f"替换「{_show(items = a, joiner = joiner)}」→「{_show(items = b, joiner = joiner)}」")
        elif tag == "delete":
            out.append(f"识别缺少「{_show(items = a, joiner = joiner)}」")
        else:
            out.append(f"识别多出「{_show(items = b, joiner = joiner)}」")
        if len(out) >= MAX_DIFFS:
            out.append("……")
            break
    return out or ["不一致"]


# ---------------------------------------------------------------------------
# 与原文的数字核对
# ---------------------------------------------------------------------------

_ROMAN = re.compile(pattern = r"(?<![A-Za-z])[IVXLCDM]{2,}(?![A-Z])")
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
_ROMAN_TABLE = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
    (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]


def _roman_value(s: str) -> int | None:
    total = 0
    for k, ch in enumerate(s):
        v = _ROMAN_VALUES[ch]
        if k + 1 < len(s) and _ROMAN_VALUES[s[k + 1]] > v:
            total -= v
        else:
            total += v
    rebuilt = []
    rest = total
    for v, sym in _ROMAN_TABLE:
        while rest >= v:
            rebuilt.append(sym)
            rest -= v
    return total if "".join(rebuilt) == s else None


def _decimal(int_digits: str, frac_digits: str) -> str:
    frac = frac_digits.rstrip("0")
    head = str(int(int_digits)) if int_digits else "0"
    return head + ("." + frac if frac else "")


def _frac_run(items: list, k: int) -> tuple:
    """items[k] 是小数点时，返回其后的小数部分数字串和结束位置。"""
    parts = []
    m = k + 1
    while m < len(items) and isinstance(items[m], Num) and items[m].frac:
        parts.append(items[m].digits)
        m += 1
    return "".join(parts), m


def _is_sep(it) -> bool:
    return isinstance(it, Mark) and it.kind == "sep"


def _support(original: str, spec) -> set:
    """原文中出现的数值（阿拉伯数字、读法、罗马数字），以字符串集合表示。"""
    support = set()

    def add_digits(d: str) -> None:
        if d:
            support.add(d)
            support.add(str(int(d)))

    items = analyze(text = original, spec = spec)[0]
    for k, it in enumerate(items):
        if not isinstance(it, Num):
            continue
        add_digits(d = it.digits)
        support.add(str(it.value))
        add_digits(d = it.lead)
        if not it.frac and k + 1 < len(items) and _is_sep(it = items[k + 1]):
            frac, _ = _frac_run(items = items, k = k + 1)
            if frac:
                support.add(_decimal(int_digits = it.digits, frac_digits = frac))
        parts = []
        for nxt in items[k:k + 4]:
            if not (isinstance(nxt, Num) and nxt.origin == "word"):
                break
            parts.append(nxt.digits)
            if len(parts) >= 2:
                support.add("".join(parts))
    for d in re.findall(pattern = r"\d+", string = normalize_text(text = original, lang = spec.code)):
        add_digits(d = d)
    for r in _ROMAN.findall(string = unicodedata.normalize("NFKC", original)):
        v = _roman_value(s = r)
        if v:
            support.add(str(v))
    return support


def _unsupported(items: list, support: set) -> list:
    """识别结果中写成阿拉伯数字、但在原文里找不到的数值。"""
    missing = []
    k = 0
    while k < len(items):
        it = items[k]
        if isinstance(it, Num) and it.origin != "word" and not it.frac:
            if k + 1 < len(items) and _is_sep(it = items[k + 1]):
                frac, end = _frac_run(items = items, k = k + 1)
                if frac:
                    value = _decimal(int_digits = it.digits, frac_digits = frac)
                    if value not in support:
                        missing.append(value)
                    k = end
                    continue
            ok = (
                it.digits in support
                or str(it.value) in support
                or (it.lead and (it.lead in support or str(int(it.lead)) in support))
            )
            if not ok:
                missing.append(it.display)
        k += 1
    return missing


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------

def compare(script: str, hypothesis: str, lang: str, original: str | None = None) -> CompareResult:
    """比较朗读稿与识别结果。

    返回 CompareResult(match, detail)：match 为 True 表示一致；detail 为差异说明，
    一致时为空字符串。给出 original 时，还要求识别结果中的阿拉伯数字都能在原文中找到。
    """
    spec = get_spec(lang = lang)
    s_variants = analyze(text = script, spec = spec)
    h_variants = analyze(text = hypothesis, spec = spec)
    matched = None
    for h in h_variants:
        h_units = _units(items = h)
        for s in s_variants:
            if _match_units(a = _units(items = s), b = h_units, lang = spec.code):
                matched = h
                break
        if matched is not None:
            break
    joiner = "" if spec.scan else " "
    problems = []
    if matched is None:
        problems = _diff(
            s_items = s_variants[0],
            h_items = h_variants[0],
            lang = spec.code,
            joiner = joiner,
        )
    if original is not None:
        checked = matched if matched is not None else h_variants[0]
        missing = _unsupported(items = checked, support = _support(original = original, spec = spec))
        if missing:
            problems.append("原文中没有识别结果里的数字：" + "、".join(missing))
    return CompareResult(match = not problems, detail = "；".join(problems))


def normalize_for_scoring(text: str, lang: str) -> str:
    """评测打分用的规范化文本。

    数词（读法和阿拉伯数字）都写成数值，小数写成「3.5」，百分号写成「%」。
    中文、日语、韩语、泰语的结果不含空格，适合按字计算错误率；其余语言以单个空格分词。
    """
    spec = get_spec(lang = lang)
    items = analyze(text = text, spec = spec)[0]
    pieces = []
    glue_next = False
    for it in items:
        glue = glue_next
        glue_next = False
        if isinstance(it, Text):
            token = it.text
        elif isinstance(it, Num):
            token = it.digits
            glue = glue or it.frac
        elif it.kind == "sep":
            token = "."
            glue = True
            glue_next = True
        elif it.kind == "pct":
            token = "%"
        else:
            token = "-"
            glue_next = True
        if glue and pieces:
            pieces[-1] += token
        else:
            pieces.append(token)
    return ("" if spec.scan else " ").join(pieces)
