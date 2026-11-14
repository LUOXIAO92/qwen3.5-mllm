"""比较入口：compare 与 normalize_for_scoring。"""

from __future__ import annotations

import difflib
import re
import unicodedata
from typing import NamedTuple

from . import ja_reading, zh_convert
from .langs import get_spec
from .normalize import normalize_text
from .numerals import ABBR_FOLD
from .parse import Mark, Num, Text, analyze, roman_spans

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
            # 两边都是读法：要求词形一致（只合并词表中登记的书写变体）；
            # 数量缩写（Mio.、млн）与全称（Millionen、миллионов）按数值比较
            if ABBR_FOLD in a.surface or ABBR_FOLD in b.surface:
                return a.value == b.value and a.kinds & b.kinds and a.frac == b.frac
            return a.surface == b.surface
        if not (a.kinds & b.kinds) or a.frac != b.frac:
            return False
        if a.frac:
            return a.digits == b.digits
        return a.value == b.value and _zeros_ok(digits = a.digits) and _zeros_ok(digits = b.digits)
    if isinstance(a, Mark) and isinstance(b, Mark):
        if {a.kind, b.kind} == {"unit", "abbr"} and "sym" in (a.origin, b.origin):
            # 「M.」在数后是米，也可以是 monsieur：符号写法的编号集合里带着缩写的读法
            return bool(a.ids & b.ids)
        if a.kind != b.kind:
            return False
        if a.origin == "word" and b.origin == "word":
            return a.surface == b.surface
        if a.kind in ("unit", "abbr"):
            return bool(a.ids & b.ids)
        if a.kind == "time" and a.origin == "sym" and b.origin == "sym":
            # 「14:30 Uhr」与「14:30」：冒号、h 之外读出的词要相同
            return a.surface.split("|")[1:] == b.surface.split("|")[1:]
        return True
    return False


def _is_word_int(it) -> bool:
    return isinstance(it, Num) and it.origin == "word" and not it.frac


def _regroup_ok(x, seq) -> bool:
    """同一串数词被标点隔开（「yedi, yüz elli milyon」对「yedi yüz elli milyon」）：原词逐字相同。

    比较的是原来的词（display），不是合并变体后的词形，「hai mươi năm」不会与「hai mươi lăm」一致。
    """
    if not _is_word_int(it = x) or not all(_is_word_int(it = w) for w in seq):
        return False
    return "".join(w.display for w in seq).replace(" ", "") == x.display.replace(" ", "")


def _as_text(it) -> str:
    """单位、缩写的读法记号也可以按普通文字比较（「yen dari」对「yendari」，比较时忽略空白）。"""
    if isinstance(it, Mark) and it.origin == "word" and it.kind in ("unit", "abbr"):
        return it.surface
    return ""


def _text_run(units: list, start: int, text: str) -> bool:
    end = start + len(text)
    return end <= len(units) and all(isinstance(u, str) for u in units[start:end]) and "".join(units[start:end]) == text


def _alt_steps(x, a_len: int, b: list, i: int, j: int, lang: str) -> list:
    """x = a[i] 是带其他切分的数时，按每种切分与 b[j:] 的开头比较，返回可以到达的 (i, j)。"""
    if not isinstance(x, Num) or not x.alts:
        return []
    out = []
    for span, parts in x.alts:
        if i + span > a_len:
            continue
        for k in range(1, min(len(parts) + 1, len(b) - j) + 1):
            if _match_units(a = list(parts), b = b[j:j + k], lang = lang):
                out.append((i + span, j + k))
    return out


def _skippable(it) -> bool:
    """冒号写法的时刻记号（后面没有读出的词）可以不对应任何东西：「5:30」对「five thirty」；
    符号写法的分钟 00 也可以不读（14:00 对 vierzehn Uhr）。"""
    return isinstance(it, Mark) and ((it.kind == "time" and it.surface == ":") or it.kind == "time00")


def _conn_step(x, b: list, j: int, lang: str) -> int:
    """冒号写法的时刻对读法中小时与分钟之间的连接词（3:15 对 tres y quince）：返回对上的单元数，0 表示对不上。"""
    if not (isinstance(x, Mark) and x.kind == "time" and x.surface == ":") or j >= len(b):
        return 0
    conns = get_spec(lang = lang).ext.conn
    y = b[j]
    if isinstance(y, Mark) and y.origin == "word" and y.surface in conns:
        return 1
    for word in conns:
        if _text_run(units = b, start = j, text = word):
            return len(word)
    return 0


def _date_step(x, b: list, j: int) -> int:
    """斜线日期的记号对读法中的「tháng」「năm」：返回对上的单元数，0 表示对不上。
    「năm」同时是数词「五」，读法一边可能是值为 5 的读法数。"""
    if not (isinstance(x, Mark) and x.kind == "date") or j >= len(b):
        return 0
    y = b[j]
    if isinstance(y, Num) and y.origin == "word" and y.display in x.ids:
        return 1
    for word in x.ids:
        if _text_run(units = b, start = j, text = word.replace(" ", "")):
            return len(word.replace(" ", ""))
    return 0


def _zero_minutes(x, seq) -> bool:
    """符号写法的分钟 00 对读出的「zéro zéro」「oh oh」。"""
    if not (isinstance(x, Mark) and x.kind == "time00") or not seq:
        return False
    return all(isinstance(w, Num) and w.origin == "word" and w.value == 0 for w in seq) and len(seq) == 2


def _concat_ok(d, seq) -> bool:
    """一个阿拉伯数字对应几个连续的读法记号：逐段读法拼起来等于该数字串。"""
    if not isinstance(d, Num) or d.origin != "digit" or "card" not in d.kinds:
        return False
    parts = []
    for w in seq:
        if not isinstance(w, Num) or w.origin != "word" or "card" not in w.kinds or w.frac != d.frac or w.split:
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
            # 只剩符号写法的时刻记号（「14:00」去掉 00 后剩下的记号）
            if i < n and _skippable(it = a[i]) and (i + 1, j) not in seen:
                seen.add((i + 1, j))
                stack.append((i + 1, j))
            if j < m and _skippable(it = b[j]) and (i, j + 1) not in seen:
                seen.add((i, j + 1))
                stack.append((i, j + 1))
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
        if isinstance(x, Num) and isinstance(y, Num):
            for k in range(2, min(MAX_CONCAT, m - j) + 1):
                if _regroup_ok(x = x, seq = b[j:j + k]):
                    succ.append((i + 1, j + k))
            for k in range(2, min(MAX_CONCAT, n - i) + 1):
                if _regroup_ok(x = y, seq = a[i:i + k]):
                    succ.append((i + k, j + 1))
        if _skippable(it = x):
            succ.append((i + 1, j))
        if _skippable(it = y):
            succ.append((i, j + 1))
        if _zero_minutes(x = x, seq = b[j:j + 2]):
            succ.append((i + 1, j + 2))
        step = _conn_step(
            x = x,
            b = b,
            j = j,
            lang = lang,
        )
        if step:
            succ.append((i + 1, j + step))
        step = _conn_step(
            x = y,
            b = a,
            j = i,
            lang = lang,
        )
        if step:
            succ.append((i + step, j + 1))
        if _zero_minutes(x = y, seq = a[i:i + 2]):
            succ.append((i + 2, j + 1))
        step = _date_step(x = x, b = b, j = j)
        if step:
            succ.append((i + 1, j + step))
        step = _date_step(x = y, b = a, j = i)
        if step:
            succ.append((i + step, j + 1))
        succ.extend(_alt_steps(
            x = x,
            a_len = n,
            b = b,
            i = i,
            j = j,
            lang = lang,
        ))
        succ.extend((p, q) for q, p in _alt_steps(
            x = y,
            a_len = m,
            b = a,
            i = j,
            j = i,
            lang = lang,
        ))
        text = _as_text(it = x)
        if text and _text_run(units = b, start = j, text = text):
            succ.append((i + 1, j + len(text)))
        text = _as_text(it = y)
        if text and _text_run(units = a, start = i, text = text):
            succ.append((i + len(text), j + 1))
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
    if it.kind == "unit":
        return f"Munit|{it.origin}|{it.surface}"
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
            token = it.surface.split("|")[0] if it.kind == "time" else it.surface
            if it.origin == "sym":
                glue = it.kind != "minus"
                glue_next = it.kind not in ("pct", "unit")
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
    for _, _, v, _ in roman_spans(text = unicodedata.normalize("NFKC", original), lang = spec.code):
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

def _align(script: str, hypothesis: str, spec) -> tuple:
    """返回 (一致时识别结果的比较单元或 None, 朗读稿的候选, 识别结果的候选)。"""
    s_variants = analyze(text = script, spec = spec, ext = True)
    h_variants = analyze(text = hypothesis, spec = spec, ext = True)
    s_units = [_units(items = s) for s in s_variants]
    for h in h_variants:
        h_units = _units(items = h)
        for units in s_units:
            if _match_units(a = units, b = h_units, lang = spec.code):
                return h, s_variants, h_variants
    return None, s_variants, h_variants


def compare(script: str, hypothesis: str, lang: str, original: str | None = None) -> CompareResult:
    """比较朗读稿与识别结果。

    返回 CompareResult(match, detail)：match 为 True 表示一致；detail 为差异说明，
    一致时为空字符串。给出 original 时，还要求识别结果中的阿拉伯数字都能在原文中找到。
    日语先按写法比较，不一致时再按读音比较（需要 fugashi 和 unidic-lite，见 ja_reading.py）。
    中文用 OpenCC 词组级转换统一为简体，没有 OpenCC 时退回单字表，并在差异说明中注明（见 zh_convert.py）。
    """
    spec = get_spec(lang = lang)
    matched, s_variants, h_variants = _align(script = script, hypothesis = hypothesis, spec = spec)
    note = ""
    if matched is None and spec.code == "ja":
        if ja_reading.available():
            reading = _align(
                script = ja_reading.to_reading(text = script),
                hypothesis = ja_reading.to_reading(text = hypothesis),
                spec = spec,
            )
            matched = reading[0]
            if matched is None:
                # 差异说明按读音给出，只剩真正读音不同的地方
                s_variants, h_variants = reading[1], reading[2]
                note = "（按读音）"
        else:
            note = "（" + ja_reading.MISSING_NOTE + "）"
    elif matched is None and spec.code == "zh" and not zh_convert.available():
        note = "（" + zh_convert.MISSING_NOTE + "）"
    joiner = "" if spec.scan else " "
    problems = []
    if matched is None:
        problems = _diff(
            s_items = s_variants[0],
            h_items = h_variants[0],
            lang = spec.code,
            joiner = joiner,
        )
        if note:
            problems[-1] += note
    if original is not None:
        checked = matched if matched is not None else h_variants[0]
        missing = _unsupported(items = checked, support = _support(original = original, spec = spec))
        if missing:
            problems.append("原文中没有识别结果里的数字：" + "、".join(missing))
    return CompareResult(match = not problems, detail = "；".join(problems))


def char_error_rate(script: str, hypothesis: str, lang: str) -> float:
    """朗读稿与识别结果的字符错误率：两者先按 normalize_for_scoring 规范化（数词统一），去掉空格，
    再按字符计算编辑距离除以朗读稿的字符数。空格不计，词的切分不同（pampang 与 pang pang）不算错误。"""
    ref = normalize_for_scoring(text = script, lang = lang).replace(" ", "")
    hyp = normalize_for_scoring(text = hypothesis, lang = lang).replace(" ", "")
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, ref_char in enumerate(ref, start = 1):
        current = [i] + [0] * len(hyp)
        for j, hyp_char in enumerate(hyp, start = 1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref_char != hyp_char))
        previous = current
    return previous[-1] / len(ref)


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
