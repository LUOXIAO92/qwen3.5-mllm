"""把文本切分成比较单元：普通文字、数值记号和三种符号记号。

数值记号 Num 对所有语言相同：
  value   整数值（小数的整数部分和小数部分分开成两个 Num，中间是小数点记号）
  digits  用于逐位拼接和小数部分比较的数字串，阿拉伯数字保留原样（含前导零）
  origin  word（读法）、digit（阿拉伯数字）或 mixed（「5 миллионов」这类混写）
  kinds   基数 card、序数 ord；不带序数标记的阿拉伯数字两者都允许
  frac    是否位于小数点之后
  surface 读法的比较用词形（已合并已知书写变体、去掉空格）
  split   是否是比对时另行切分出来的数
  alts    比对时的其他切分（「육십일」= 육십 + 일，「五千米」= 五千 + 米），见 extend.py
符号记号 Mark：小数点 sep、百分号 pct、负号 minus，区分符号写法 sym 与读法 word。
比对时（ext = True）还有单位 unit（ids 为单位编号集合）和时刻 time 两种记号，见 units.py、extend.py。
"""

from __future__ import annotations

import itertools
import re
import unicodedata
from dataclasses import dataclass

from .normalize import normalize_text
from .units import CENTURY_WORDS, DOLLAR_PREFIXES, ROMAN_SUFFIXES, SQUARE_CHARS, SQUARE_MARKS, SYMBOL_CHARS
from .numerals import (
    CARD_ONLY, CARD_OR_ORD, CONN_CLITIC, ORD_ONLY, LangSpec, NumState, add, places_of,
)

MAX_VARIANTS = 8
MAX_EXT_VARIANTS = 16


@dataclass(frozen = True, slots = True)
class Text:
    text: str


@dataclass(frozen = True, slots = True)
class Num:
    value: int
    digits: str
    origin: str
    kinds: frozenset
    frac: bool
    surface: str
    display: str
    lead: str = ""
    # 比对时另行切分出来的数（韩语「육십일」切成「육십」「일」）；不参与逐段拼接（年份式读法）
    split: bool = False
    # 比对时这个数（连同后面 span - 1 个单元）的其他读法：((span, 比较单元序列), ...)，见 extend.py
    alts: tuple = ()
    # 比对时由罗马数字（Henry VIII、XIXe siècle）换来的数
    roman: bool = False


@dataclass(frozen = True, slots = True)
class Mark:
    kind: str
    origin: str
    surface: str
    ids: frozenset = frozenset()


class Tok:
    """切分阶段的记号。kind：word、char、digit、sep、pct、minus、bound、alt，
    比对时另有 usym（货币、度数符号）、colon（数字之间的冒号）和 slash（日期中数字之间的斜线）。"""

    __slots__ = ("kind", "text", "frac", "kinds", "alts", "roman", "brk")

    def __init__(self, kind, text = "", frac = False, kinds = CARD_ONLY, alts = None):
        self.kind = kind
        self.text = text
        self.frac = frac
        self.kinds = kinds
        self.alts = alts
        self.roman = False
        self.brk = False

    def __repr__(self):
        return f"Tok({self.kind}, {self.text!r})"


_BOUND = Tok(kind = "bound")
_APOSTROPHES = set("'’ʼ`´‘")
_WORD_HYPHENS = set("-‐‑")
_SPACE_GROUP = {chr(c) for c in (0x00A0, 0x202F, 0x2009, 0x2007)}  # 不换行空格、窄空格等千分位分隔符
_NUM_SEPS = set(".,٫٬")
_MULT_NEEDS = frozenset({"place", "big"})


def _is_digit(ch: str) -> bool:
    return "0" <= ch <= "9"


def _is_letter(ch: str) -> bool:
    """字母、组合符号，以及〇这类非 ASCII 的数字字符。"""
    cat = unicodedata.category(ch)
    return cat[0] in "LM" or cat in ("Nl", "No")


# ---------------------------------------------------------------------------
# 数字串
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 罗马数字（比对时）
# ---------------------------------------------------------------------------

ROMAN_MARK = "\ue010"
_ROMAN_RUN = re.compile(pattern = r"(?<![^\W\d_])(?P<r>[IVXL]+)(?P<suf>[^\W\d_]*)")
_ROMAN_VALUE = {"I": 1, "V": 5, "X": 10, "L": 50}
_ROMAN_TABLE = [(50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
_CYRILLIC_NEXT = re.compile(pattern = r"\.?\s?[\u0400-\u04ff]")
_INITIAL_NEXT = re.compile(pattern = r"\.?\s?(?P<c>[^\W\d_])(?![^\W\d_])")
_NUMERO = re.compile(pattern = r"(?<![^\W\d_])([nN])\.?\s?[º°]")


def roman_value(text: str) -> int | None:
    """规范写法的罗马数字（1–89，只用 I V X L）的值，不是时返回 None。"""
    total = 0
    for k, ch in enumerate(text):
        v = _ROMAN_VALUE[ch]
        total += -v if k + 1 < len(text) and _ROMAN_VALUE[text[k + 1]] > v else v
    rest, rebuilt = total, []
    for v, sym in _ROMAN_TABLE:
        while rest >= v:
            rebuilt.append(sym)
            rest -= v
    return total if 0 < total < 90 and "".join(rebuilt) == text else None


def _word_before(text: str, start: int) -> str:
    """start 之前的一个词（跳过空格和缩写的点：「s. V」前面的词是 s）。"""
    k = start - 1
    while k >= 0 and text[k] in " \u00a0\t":
        k -= 1
    if k >= 0 and text[k] == "." and k - 1 >= 0 and text[k - 1].isalpha():
        k -= 1
    end = k + 1
    while k >= 0 and (text[k].isalpha() or text[k] in "'’-"):
        k -= 1
    return text[k + 1:end]


def roman_spans(text: str, lang: str) -> list:
    """文本中的罗马数字：[(起点, 终点, 值, 后缀), ...]，只认大写的 I V X L。

    单个字母（I、V、X）只在前面紧跟首字母大写的词时认作罗马数字（Pedro I、Louis X），英语的 I 不认，
    首字母缩写（H.I.V.、B.V.）中的字母不认；俄语句中前面是西里尔字母的词也可以（в V веке）。
    后缀只收该语言的序数后缀（XIXe、XIº），带后缀的单个字母（Ve、Ier）不受上面的限制。
    """
    out = []
    for m in _ROMAN_RUN.finditer(string = text):
        r, suf = m.group("r"), m.group("suf")
        value = roman_value(text = r)
        if value is None or (suf and suf not in ROMAN_SUFFIXES.get(lang, ())):
            continue
        if len(r) == 1 and not suf:
            before = _word_before(text = text, start = m.start())
            if r == "L" or (lang == "en" and r == "I") or not (before or lang == "ru"):
                continue
            nxt = _INITIAL_NEXT.match(string = text, pos = m.end())
            if nxt is not None and nxt.group("c").isupper():
                # 首字母缩写（H.I.V.、B.V.、T I A）中的单个字母
                continue
            after_word = _CYRILLIC_NEXT.match(string = text, pos = m.end())
            cyrillic = lang == "ru" and ("\u0400" <= before[-1:] <= "\u04ff" or after_word is not None)
            century = before.lower().rstrip(".") in CENTURY_WORDS.get(lang, ())
            if not (cyrillic or century or (len(before) >= 2 and before[0].isupper())):
                continue
        out.append((m.start(), m.start() + len(r), value, suf))
    return out


def mark_roman(text: str, lang: str) -> str:
    """把罗马数字换成「标记 + 阿拉伯数字」，切分时成为带 roman 标记的数字记号。"""
    spans = roman_spans(text = text, lang = lang)
    for start, end, value, _ in reversed(spans):
        text = text[:start] + ROMAN_MARK + str(value) + text[end:]
    return text


def _digit_tok(digits: str, frac: bool = False) -> Tok:
    """数字记号。不带序数标记的整数既可对应基数也可对应序数（「3 May」读作 May third）。"""
    return Tok(
        kind = "digit",
        text = digits,
        frac = frac,
        kinds = CARD_ONLY if frac else CARD_OR_ORD,
    )


def _read_digits(s: str, i: int) -> int:
    while i < len(s) and _is_digit(ch = s[i]):
        i += 1
    return i


def _read_number(s: str, i: int, spec: LangSpec, ext: bool = False):
    """读取以 i 开头的数字串（含千分位和小数点），返回 (结束位置, 候选切分列表)。

    ext 为 True 时（比对），不习惯以空格分千位的语言也把「35 000」读成一个数，同时保留分开读的候选。"""
    j = _read_digits(s = s, i = i)
    groups = [s[i:j]]
    seps = []
    n = len(s)
    while j < n:
        ch = s[j]
        if ch in _NUM_SEPS and j + 1 < n and _is_digit(ch = s[j + 1]):
            k = _read_digits(s = s, i = j + 1)
            groups.append(s[j + 1:k])
            seps.append(ch)
            j = k
            continue
        if (ch in _SPACE_GROUP or (ch == " " and (spec.space_groups or ext))) and len(groups[-1]) <= 3:
            k = _read_digits(s = s, i = j + 1)
            if k - (j + 1) == 3:
                groups.append(s[j + 1:k])
                seps.append(" ")
                j = k
                continue
        break
    alts = _interpret(groups = groups, seps = seps, spec = spec)
    if ext and not spec.space_groups and " " in seps and all(sp == " " for sp in seps):
        split = [_digit_tok(digits = g) for g in groups]
        if split not in alts:
            alts.append(split)
    return j, alts


def _interpret(groups: list, seps: list, spec: LangSpec) -> list:
    """按语言习惯解释千分位与小数点；有歧义时返回多种候选（本语言习惯在前）。"""
    if not seps:
        return [[_digit_tok(digits = groups[0])]]

    def whole():
        return [_digit_tok(digits = "".join(groups))]

    def decimal(int_groups, frac):
        return [
            _digit_tok(digits = "".join(int_groups)),
            Tok(kind = "sep"),
            _digit_tok(digits = frac, frac = True),
        ]

    def split():
        return [_digit_tok(digits = g) for g in groups]

    def time_split():
        return [_digit_tok(digits = groups[0]), Tok(kind = "colon"), _digit_tok(digits = groups[1])]

    grouped = len(groups[0]) <= 3 and all(len(g) == 3 for g in groups[1:])
    if "٫" in seps:
        head = seps[:-1]
        if seps[-1] == "٫" and all(sp in ("٬", " ") for sp in head) and all(len(g) == 3 for g in groups[1:-1]):
            return [decimal(int_groups = groups[:-1], frac = groups[-1])]
        return [split()]
    if all(sp in ("٬", " ") for sp in seps):
        return [whole()] if grouped else [split()]
    last = seps[-1]
    head = seps[:-1]
    if last in ".," and seps.count(last) == 1:
        head_ok = (
            len(set(head)) <= 1
            and all(sp != last for sp in head)
            and all(len(g) == 3 for g in groups[1:-1])
            and len(groups[0]) <= 3
        )
        if head and head_ok:
            return [decimal(int_groups = groups[:-1], frac = groups[-1])]
        if not head:
            frac = groups[-1]
            dec = decimal(int_groups = groups[:-1], frac = frac)
            if len(frac) == 3 and len(groups[0]) <= 3:
                locale_decimal = (last == ",") == spec.dec_comma
                return [dec, whole()] if locale_decimal else [whole(), dec]
            alts = [dec]
            # 「14.30」也可能是时刻
            if last == "." and len(groups[0]) <= 2 and len(frac) == 2 and int(groups[0]) <= 24 and int(frac) < 60:
                alts.append(time_split())
            return alts
    if len(set(seps)) == 1 and seps[0] in ".," and grouped:
        return [whole()]
    return [split()]


# ---------------------------------------------------------------------------
# 切分
# ---------------------------------------------------------------------------

def tokenize(text: str, spec: LangSpec, ext: bool = False) -> list:
    """切分文本。ext 为 True 时（比对）货币、度数符号和数字之间的冒号单独成为记号，否则按标点处理；
    上标的平方、立方（km²、㎢）在 NFKC 之前换成私用区字符，保留为单位符号的一部分。"""
    if ext:
        text = mark_roman(text = text.translate(SQUARE_MARKS), lang = spec.code)
        text = _NUMERO.sub(repl = r"\1°", string = text)
    s = normalize_text(text = text, lang = spec.code)
    toks = []
    buf = []
    n = len(s)

    def flush():
        if not buf:
            return
        word = "".join(buf)
        buf.clear()
        if spec.scan:
            toks.extend(_scan(span = word, spec = spec, ext = ext))
        else:
            toks.append(Tok(kind = "word", text = word))

    i = 0
    roman_next = False
    while i < n:
        ch = s[i]
        if ext and ch == ROMAN_MARK:
            flush()
            roman_next = True
            i += 1
            continue
        if _is_digit(ch = ch):
            flush()
            j, alts = _read_number(
                s = s,
                i = i,
                spec = spec,
                ext = ext,
            )
            end = _attach_ordinal(
                s = s,
                j = j,
                alts = alts,
                spec = spec,
                ext = ext,
            )
            if ext and end == j + 1 and s[j] == "°":
                # 西、葡、意语的「20°」可以是序数，也可以是度数
                alts.append([_digit_tok(digits = alts[0][0].text), Tok(kind = "usym", text = "°")])
            j = end
            if roman_next:
                for alt in alts:
                    for t in alt:
                        t.roman = t.kind == "digit"
                roman_next = False
            single = len(alts) == 1 and len(alts[0]) == 1
            toks.append(alts[0][0] if single else Tok(kind = "alt", alts = alts))
            i = j
            continue
        if _is_letter(ch = ch) or (ext and ch in SQUARE_CHARS):
            buf.append(ch)
            i += 1
            continue
        if ch in _APOSTROPHES:
            i += 1
            continue
        if ch.isspace():
            if not spec.scan:
                flush()
            i += 1
            continue
        if ext and ch == "°" and "".join(buf) == "n":
            # 编号缩写 n°（numéro、número、numero）
            buf.append(ch)
            i += 1
            continue
        if ext and ch in SYMBOL_CHARS:
            text_sym = ch
            if ch == "$" and "".join(buf) in DOLLAR_PREFIXES:
                text_sym = "".join(buf) + ch
                buf.clear()
            flush()
            if ch == "°" and i + 1 < n and s[i + 1] in "cf" and not (i + 2 < n and "a" <= s[i + 2] <= "z"):
                text_sym = s[i:i + 2]
                i += 1
            toks.append(Tok(kind = "usym", text = text_sym))
            i += 1
            continue
        flush()
        nxt_digit = i + 1 < n and _is_digit(ch = s[i + 1])
        if ext and ch == ":" and nxt_digit and i and _is_digit(ch = s[i - 1]):
            toks.append(Tok(kind = "colon"))
            i += 1
            continue
        if ext and ch == "/" and nxt_digit and i and _is_digit(ch = s[i - 1]) and spec.ext is not None and spec.ext.date_month:
            toks.append(Tok(kind = "slash"))
            i += 1
            continue
        if ch in _WORD_HYPHENS or ch == "−":
            prev = s[i - 1] if i else " "
            if nxt_digit and not (prev.isalnum() or prev in _APOSTROPHES):
                toks.append(Tok(kind = "minus", text = "-"))
            elif ch != "−" and not _is_digit(ch = prev) and not nxt_digit:
                pass  # 词之间的连字符按空格处理
            elif ext and ch != "−" and _is_digit(ch = prev) and i + 1 < n and _is_letter(ch = s[i + 1]):
                pass  # 比对时数字与词之间的连字符也按空格处理：50-million、17-летний
            else:
                toks.append(_BOUND)
            i += 1
            continue
        if ch == "%":
            toks.append(Tok(kind = "pct", text = "%"))
            i += 1
            continue
        toks.append(_BOUND)
        i += 1
    flush()
    if not spec.scan and spec.multiword:
        toks = _merge_multiword(toks = toks, spec = spec)
    return toks


def _attach_ordinal(s: str, j: int, alts: list, spec: LangSpec, ext: bool = False) -> int:
    """数字后紧跟序数标记（3rd、3.、3e、3-й、3º）时，记录序数属性并吞掉标记。
    比对时数字与序数标记之间可以有撇号（土耳其语 3'üncü）。"""
    if len(alts) != 1 or len(alts[0]) != 1 or not spec.ordinal_suffixes:
        return j
    starts = [j]
    if ext and j < len(s) and s[j] in _APOSTROPHES:
        starts.append(j + 1)
    for pos in starts:
        for pattern, kinds, ext_only in spec.ordinal_suffixes:
            if ext_only and not ext:
                continue
            m = pattern.match(string = s, pos = pos)
            if m and m.end() > pos:
                alts[0][0].kinds = kinds
                return m.end()
    return j


def _scan(span: str, spec: LangSpec, ext: bool = False) -> list:
    """无空格语言：在字符串中按最长匹配找出数词语素和特殊词，其余逐字成为 char。"""
    keys = spec.scan_keys(ext = ext)
    longest = spec.max_key(ext = ext) if ext else spec._max_piece
    out = []
    i = 0
    n = len(span)
    while i < n:
        hit = None
        for length in range(min(longest, n - i), 0, -1):
            piece = span[i:i + length]
            if piece in keys:
                need = spec.lookahead.get(piece)
                if need and not span.startswith(need, i + length):
                    continue
                hit = piece
                break
        if hit:
            out.append(Tok(kind = "word", text = hit))
            i += len(hit)
        else:
            out.append(Tok(kind = "char", text = span[i]))
            i += 1
    return out


def _merge_multiword(toks: list, spec: LangSpec) -> list:
    """把词表中的多词读法（soixante dix sept、pour cent、phần trăm）合并成一个词。"""
    out = []
    i = 0
    n = len(toks)
    while i < n:
        merged = False
        if toks[i].kind == "word":
            for length in range(min(spec._max_multi, n - i), 1, -1):
                window = toks[i:i + length]
                if all(t.kind == "word" for t in window):
                    key = tuple(t.text for t in window)
                    if key in spec.multiword:
                        out.append(Tok(kind = "word", text = spec.multiword[key]))
                        i += length
                        merged = True
                        break
        if not merged:
            out.append(toks[i])
            i += 1
    return out


def _expand(toks: list) -> list:
    """展开数字写法的歧义（例如 1.234 是千分位还是小数），最多 MAX_VARIANTS 种。"""
    choices = []
    for t in toks:
        choices.append(t.alts if t.kind == "alt" else [[t]])
    variants = []
    for combo in itertools.product(*choices):
        variants.append([t for part in combo for t in part])
        if len(variants) >= MAX_VARIANTS:
            break
    return variants


# ---------------------------------------------------------------------------
# 组成数值记号
# ---------------------------------------------------------------------------

def _digit_morph(digits: str, needs = None):
    value = int(digits)
    if value == 0:
        return None
    return add(
        value = value,
        places = places_of(value = value),
        needs = needs,
        system = "digit",
    )


def parse_number(toks: list, start: int, spec: LangSpec, frac: bool = False, first = None, ext: bool = False):
    """从 start 开始尽量长地组成一个数，返回 (结束位置, 状态, 分析路径) 或 None。

    first 给出时，以该语素（阿拉伯数字）开头，组成「5 миллионов」「3万」这类混写。
    """
    frontier = []
    if first is not None:
        st = NumState(frac = frac).step(m = first, spec = spec)
        if st is not None:
            frontier.append((st, start + 1, ()))
    else:
        for a in spec.analyses(word = toks[start].text, ext = ext):
            st = NumState(frac = frac).apply(morphs = a.morphs, spec = spec)
            if st is not None:
                frontier.append((st, start + 1, (a,)))
    best = None
    seen = set()
    while frontier:
        nxt = []
        for st, j, path in frontier:
            if st.complete() and (best is None or j > best[0]):
                best = (j, st, path)
            if path and path[-1] is not None and path[-1].suffix:
                continue
            if j >= len(toks):
                continue
            tok = toks[j]
            if tok.brk:
                continue
            cands = []
            if tok.kind == "word":
                for a in spec.analyses(word = tok.text, ext = ext):
                    if a.prefix and not a.conn_prefix:
                        continue
                    if (
                        spec.split_words_need_scale
                        and st.last != "big"
                        and a.morphs[0].cls not in ("place", "big", "big_lex")
                    ):
                        # 德语数词连写：「vierzig achten」是两个词，「zwei tausend」才组成一个数
                        continue
                    s2 = st
                    if a.conn_prefix:
                        s2 = s2.step(m = CONN_CLITIC, spec = spec)
                        if s2 is None:
                            continue
                    s2 = s2.apply(morphs = a.morphs, spec = spec)
                    if s2 is not None:
                        cands.append((s2, a))
            elif (
                tok.kind == "digit"
                and spec.digits_after_big
                and st.last in ("big", "place")
                and not tok.frac
                and "card" in tok.kinds
            ):
                if st.last == "big" or len(tok.text) == max(st.last_places):
                    # 1万2000、3천800：数字填满大单位或位值词下面的各位
                    m = _digit_morph(digits = tok.text)
                else:
                    # 2千4百万、1천6백만：位值词后的数字后面还要接位值词
                    m = _digit_morph(digits = tok.text, needs = _MULT_NEEDS)
                s2 = st.step(m = m, spec = spec) if m is not None else None
                if s2 is not None:
                    cands.append((s2, None))
            for s2, a in cands:
                key = (s2.key(), j + 1)
                if key in seen:
                    continue
                seen.add(key)
                nxt.append((s2, j + 1, path + (a,)))
        frontier = nxt
    return best


def _minutes_follow(toks: list, j: int, spec: LangSpec, ext: bool = True) -> bool:
    """中文「点」后面是否是分钟：「十五」「零五」这类小数读法里不会出现的写法，或后面跟着「分」。

    「三点五」仍按小数处理；「三点五分」「三点十五」「三点零五」按时刻处理（另有按小数处理的候选）。
    """
    n = len(toks)
    if j >= n:
        return False
    tok = toks[j]
    zero_lead = False
    if tok.kind == "word" and any(m.cls == "zero" for a in spec.analyses(word = tok.text, ext = ext) for m in a.morphs):
        if j + 1 >= n or not _numeral_start(tok = toks[j + 1], spec = spec, ext = ext):
            return False
        zero_lead = True
        j += 1
        tok = toks[j]
    if tok.kind == "digit" and not tok.frac:
        value, end = int(tok.text), j + 1
    elif tok.kind == "word":
        res = parse_number(
            toks = toks,
            start = j,
            spec = spec,
            ext = ext,
        )
        if res is None:
            return False
        end, st, _ = res
        value = st.value()
    else:
        return False
    minute_word = (
        end < n
        and toks[end].text in spec.ext.minute
        and not (end + 1 < n and toks[end + 1].text == "钟")
    )
    return value < 60 and (minute_word or value >= 10 or (zero_lead and value < 10))


def _numeral_start(tok: Tok, spec: LangSpec, ext: bool = False) -> bool:
    return tok.kind == "digit" or (tok.kind == "word" and bool(spec.analyses(word = tok.text, ext = ext)))


def _is_int_num(it) -> bool:
    return isinstance(it, Num) and not it.frac


def parse_items(toks: list, spec: LangSpec, ext: bool = False, time_first: bool = True) -> list:
    """把记号组成比较单元。

    ext 为 True 时（比对）保留冒号和单位符号记号，交给 extend.py 处理；time_first 为 True 时，
    同时是小数点读法和「时」的词（中文「点」）在不超过 24 的整数后读作时刻，否则读作小数点。
    """
    items = []
    frac_ctx = False
    joiner = "" if spec.scan else " "
    i = 0
    n = len(toks)
    while i < n:
        t = toks[i]
        kind = t.kind
        if kind == "bound":
            frac_ctx = False
            i += 1
            continue
        if kind == "sep":
            items.append(Mark(kind = "sep", origin = "sym", surface = "."))
            frac_ctx = True
            i += 1
            continue
        if kind == "pct":
            items.append(Mark(kind = "pct", origin = "sym", surface = "%"))
            frac_ctx = False
            i += 1
            continue
        if kind == "minus":
            items.append(Mark(kind = "minus", origin = "sym", surface = "-"))
            i += 1
            continue
        if kind == "colon":
            if ext:
                items.append(Mark(kind = "colon", origin = "sym", surface = ":"))
            frac_ctx = False
            i += 1
            continue
        if kind == "slash":
            items.append(Mark(kind = "slash", origin = "sym", surface = "/"))
            frac_ctx = False
            i += 1
            continue
        if kind == "usym":
            items.append(Mark(kind = "usym", origin = "sym", surface = t.text))
            frac_ctx = False
            i += 1
            continue
        if kind == "char":
            items.append(Text(text = t.text))
            frac_ctx = False
            i += 1
            continue
        if kind == "digit":
            i = _digit_item(
                toks = toks,
                i = i,
                spec = spec,
                items = items,
                frac_ctx = frac_ctx,
                ext = ext,
            )
            frac_ctx = items[-1].frac if isinstance(items[-1], Num) else False
            continue
        # 普通词
        w = t.text
        nxt = toks[i + 1] if i + 1 < n else None
        nxt_num = nxt is not None and _numeral_start(tok = nxt, spec = spec, ext = ext)
        if (
            ext
            and time_first
            and spec.ext is not None
            and w in spec.sep_words
            and w in spec.ext.hour
            and items
            and _is_int_num(it = items[-1])
            and items[-1].value <= 24
            and _minutes_follow(toks = toks, j = i + 1, spec = spec)
        ):
            items.append(Mark(kind = "time", origin = "word", surface = w))
            frac_ctx = False
            i += 1
            continue
        if w in spec.sep_words and items and _is_int_num(it = items[-1]) and nxt_num:
            items.append(Mark(kind = "sep", origin = "word", surface = w.replace(" ", "")))
            frac_ctx = True
            i += 1
            continue
        if w in spec.minus_words and nxt_num:
            items.append(Mark(kind = "minus", origin = "word", surface = w.replace(" ", "")))
            i += 1
            continue
        pos = spec.pct_words.get(w)
        # 比对时数与百分号读法之间可以有连接成分（菲律宾语 anim na porsiyento）
        linked = (
            ext
            and spec.ext is not None
            and len(items) >= 2
            and isinstance(items[-1], Text)
            and items[-1].text in spec.ext.linkers
            and isinstance(items[-2], Num)
        )
        if (pos == "post" and items and (isinstance(items[-1], Num) or linked)) or (pos == "pre" and nxt_num):
            items.append(Mark(kind = "pct", origin = "word", surface = w.replace(" ", "")))
            frac_ctx = False
            i += 1
            continue
        res = parse_number(
            toks = toks,
            start = i,
            spec = spec,
            frac = frac_ctx,
            ext = ext,
        )
        if res is not None:
            end, st, path = res
            first = path[0]
            if first.prefix:
                items.append(Text(text = first.prefix))
            value = st.value()
            mixed = any(a is None for a in path)
            surface = first.surface
            for a in path[1:]:
                if a is not None:
                    surface += (a.prefix if a.conn_prefix else "") + a.surface
            items.append(Num(
                value = value,
                digits = str(value),
                origin = "mixed" if mixed else "word",
                kinds = ORD_ONLY if st.ord else CARD_ONLY,
                frac = frac_ctx,
                surface = str(value) if mixed else surface,
                display = joiner.join(tok.text for tok in toks[i:end]),
            ))
            if path[-1] is not None and path[-1].suffix:
                items.append(Text(text = path[-1].suffix))
            i = end
            continue
        items.append(Text(text = w))
        frac_ctx = False
        i += 1
    if spec.post is not None:
        items = spec.post(items = items)
    return _place_percent(items = items, spec = spec)


def _digit_item(toks: list, i: int, spec: LangSpec, items: list, frac_ctx: bool, ext: bool = False) -> int:
    """处理一个阿拉伯数字记号，返回下一个位置；后面接位值词时组成混写数。"""
    t = toks[i]
    frac = t.frac or frac_ctx
    joiner = "" if spec.scan else " "
    if not frac and t.kinds != ORD_ONLY and not t.roman:
        m = _digit_morph(digits = t.text, needs = _MULT_NEEDS)
        res = None
        if m is not None:
            res = parse_number(
                toks = toks,
                start = i,
                spec = spec,
                first = m,
                ext = ext,
            )
        if res is not None and res[0] > i + 1:
            end, st, path = res
            value = st.value()
            items.append(Num(
                value = value,
                digits = str(value),
                origin = "mixed",
                kinds = ORD_ONLY if st.ord else CARD_ONLY,
                frac = False,
                surface = str(value),
                display = joiner.join(tok.text for tok in toks[i:end]),
                lead = t.text,
            ))
            if path and path[-1] is not None and path[-1].suffix:
                # 「25 ribuan」的 -an 与读法「dua puluh lima ribuan」一样保留
                items.append(Text(text = path[-1].suffix))
            return end
    kinds = t.kinds
    end = i + 1
    if end < len(toks) and toks[end].kind == "word" and toks[end].text in spec.ord_mark_words:
        kinds = ORD_ONLY
        end += 1
    items.append(Num(
        value = int(t.text),
        digits = t.text,
        origin = "digit",
        kinds = kinds,
        frac = frac,
        surface = t.text,
        display = t.text,
        roman = t.roman,
    ))
    return end


def _group_end(items: list, k: int, step: int) -> int:
    """从 k 开始沿 step 方向找连续的数字组（Num 与小数点），返回组外第一个位置。"""
    while 0 <= k < len(items):
        it = items[k]
        if not (isinstance(it, Num) or (isinstance(it, Mark) and it.kind == "sep")):
            break
        k += step
    return k


def _place_percent(items: list, spec: LangSpec) -> list:
    """把符号 % 移到该语言读法中的位置（中文、土耳其语在数前，其余在数后）。"""
    if not any(isinstance(it, Mark) and it.kind == "pct" and it.origin == "sym" for it in items):
        return items
    out = list(items)
    k = 0
    while k < len(out):
        it = out[k]
        if isinstance(it, Mark) and it.kind == "pct" and it.origin == "sym":
            if spec.pct_symbol_prefix:
                start = _group_end(items = out, k = k - 1, step = -1) + 1
                if start < k:
                    out.pop(k)
                    out.insert(start, it)
            else:
                end = _group_end(items = out, k = k + 1, step = 1)
                before = k > 0 and isinstance(out[k - 1], Num)
                if end > k + 1 and not before:
                    out.pop(k)
                    out.insert(end - 1, it)
                    k = end - 1
        k += 1
    return out


MAX_CONN_BREAKS = 3


def _conn_breaks(toks: list, spec: LangSpec) -> list:
    """数字写法的候选，以及在数词之间的连接词处断开的候选。

    「one thousand five hundred and ten thousand」可以是 1500 and 10000，「dois mil e dois mil e dezenove」
    可以是 2000 e 2019：连接词既可以在一个数之内，也可以连接两个数，读音相同，两种都尝试。
    """
    variants = _expand(toks = toks)
    if spec.scan:
        return variants
    spots = []
    for k in range(1, len(toks) - 1):
        tok = toks[k]
        if tok.kind != "word" or not any(m.cls == "conn" for a in spec.analyses(word = tok.text, ext = True) for m in a.morphs):
            continue
        before = _numeral_start(tok = toks[k - 1], spec = spec, ext = True)
        if before and _numeral_start(tok = toks[k + 1], spec = spec, ext = True):
            spots.append(k)
    spots = spots[:MAX_CONN_BREAKS]
    choices = [[k] for k in spots] + ([spots] if len(spots) > 1 else [])
    out = list(variants)
    for chosen in choices:
        broken = list(toks)
        for k in chosen:
            brk = Tok(kind = "word", text = toks[k].text)
            brk.brk = True
            broken[k] = brk
        out.extend(_expand(toks = broken)[:2])
    return out


def analyze(text: str, spec: LangSpec, ext: bool = False) -> list:
    """返回文本的比较单元序列；数字写法有歧义时返回多种候选。

    ext 为 True 时用于比对：另外统一单位、货币、时刻的读法与符号写法，合并书写变体，
    并给读法数挂上其他切分（Num.alts）。评测打分用 ext = False。
    """
    if not ext:
        toks = tokenize(text = text, spec = spec)
        return [parse_items(toks = variant, spec = spec) for variant in _expand(toks = toks)]
    from .extend import extend_items

    toks = tokenize(text = text, spec = spec, ext = True)
    conflict = spec.ext is not None and bool(spec.sep_words & spec.ext.hour)
    out = []
    for variant in _conn_breaks(toks = toks, spec = spec):
        for time_first in ((True, False) if conflict else (True,)):
            items = parse_items(
                toks = variant,
                spec = spec,
                ext = True,
                time_first = time_first,
            )
            items = extend_items(items = items, spec = spec)
            if items not in out:
                out.append(items)
    return out[:MAX_EXT_VARIANTS]
