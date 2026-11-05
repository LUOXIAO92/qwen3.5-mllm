"""把文本切分成比较单元：普通文字、数值记号和三种符号记号。

数值记号 Num 对所有语言相同：
  value   整数值（小数的整数部分和小数部分分开成两个 Num，中间是小数点记号）
  digits  用于逐位拼接和小数部分比较的数字串，阿拉伯数字保留原样（含前导零）
  origin  word（读法）、digit（阿拉伯数字）或 mixed（「5 миллионов」这类混写）
  kinds   基数 card、序数 ord；不带序数标记的阿拉伯数字两者都允许
  frac    是否位于小数点之后
  surface 读法的比较用词形（已合并已知书写变体、去掉空格）
  split   是否是比对时另行切分出来的数（见 extend.split_variants）
符号记号 Mark：小数点 sep、百分号 pct、负号 minus，区分符号写法 sym 与读法 word。
比对时（ext = True）还有单位 unit（ids 为单位编号集合）和时刻 time 两种记号，见 units.py、extend.py。
"""

from __future__ import annotations

import itertools
import unicodedata
from dataclasses import dataclass

from .normalize import normalize_text
from .units import DOLLAR_PREFIXES, SYMBOL_CHARS
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
    # 由 spec.variants 切出的数（韩语「육십일」切成「육십」「일」）；不参与逐段拼接（年份式读法）
    split: bool = False


@dataclass(frozen = True, slots = True)
class Mark:
    kind: str
    origin: str
    surface: str
    ids: frozenset = frozenset()


class Tok:
    """切分阶段的记号。kind：word、char、digit、sep、pct、minus、bound、alt，
    比对时另有 usym（货币、度数符号）和 colon（数字之间的冒号）。"""

    __slots__ = ("kind", "text", "frac", "kinds", "alts")

    def __init__(self, kind, text = "", frac = False, kinds = CARD_ONLY, alts = None):
        self.kind = kind
        self.text = text
        self.frac = frac
        self.kinds = kinds
        self.alts = alts

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


def _read_number(s: str, i: int, spec: LangSpec):
    """读取以 i 开头的数字串（含千分位和小数点），返回 (结束位置, 候选切分列表)。"""
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
        if (ch in _SPACE_GROUP or (ch == " " and spec.space_groups)) and len(groups[-1]) <= 3:
            k = _read_digits(s = s, i = j + 1)
            if k - (j + 1) == 3:
                groups.append(s[j + 1:k])
                seps.append(" ")
                j = k
                continue
        break
    return j, _interpret(groups = groups, seps = seps, spec = spec)


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
    """切分文本。ext 为 True 时（比对）货币、度数符号和数字之间的冒号单独成为记号，否则按标点处理。"""
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
            toks.extend(_scan(span = word, spec = spec))
        else:
            toks.append(Tok(kind = "word", text = word))

    i = 0
    while i < n:
        ch = s[i]
        if _is_digit(ch = ch):
            flush()
            j, alts = _read_number(s = s, i = i, spec = spec)
            end = _attach_ordinal(
                s = s,
                j = j,
                alts = alts,
                spec = spec,
            )
            if ext and end == j + 1 and s[j] == "°":
                # 西、葡、意语的「20°」可以是序数，也可以是度数
                alts.append([_digit_tok(digits = alts[0][0].text), Tok(kind = "usym", text = "°")])
            j = end
            single = len(alts) == 1 and len(alts[0]) == 1
            toks.append(alts[0][0] if single else Tok(kind = "alt", alts = alts))
            i = j
            continue
        if _is_letter(ch = ch):
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
        if ch in _WORD_HYPHENS or ch == "−":
            prev = s[i - 1] if i else " "
            if nxt_digit and not (prev.isalnum() or prev in _APOSTROPHES):
                toks.append(Tok(kind = "minus", text = "-"))
            elif ch != "−" and not _is_digit(ch = prev) and not nxt_digit:
                pass  # 词之间的连字符按空格处理
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


def _attach_ordinal(s: str, j: int, alts: list, spec: LangSpec) -> int:
    """数字后紧跟序数标记（3rd、3.、3e、3-й、3º）时，记录序数属性并吞掉标记。"""
    if len(alts) != 1 or len(alts[0]) != 1 or not spec.ordinal_suffixes:
        return j
    for pattern, kinds in spec.ordinal_suffixes:
        m = pattern.match(string = s, pos = j)
        if m:
            alts[0][0].kinds = kinds
            return m.end()
    return j


def _scan(span: str, spec: LangSpec) -> list:
    """无空格语言：在字符串中按最长匹配找出数词语素和特殊词，其余逐字成为 char。"""
    keys = spec.scan_keys()
    out = []
    i = 0
    n = len(span)
    while i < n:
        hit = None
        for length in range(min(spec._max_piece, n - i), 0, -1):
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
    return add(value = value, places = places_of(value = value), needs = needs)


def parse_number(toks: list, start: int, spec: LangSpec, frac: bool = False, first = None):
    """从 start 开始尽量长地组成一个数，返回 (结束位置, 状态, 分析路径) 或 None。

    first 给出时，以该语素（阿拉伯数字）开头，组成「5 миллионов」「3万」这类混写。
    """
    frontier = []
    if first is not None:
        st = NumState(frac = frac).step(m = first, spec = spec)
        if st is not None:
            frontier.append((st, start + 1, ()))
    else:
        for a in spec.analyses(word = toks[start].text):
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
            cands = []
            if tok.kind == "word":
                for a in spec.analyses(word = tok.text):
                    if a.prefix and not a.conn_prefix:
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
                and st.last == "big"
                and not tok.frac
                and "card" in tok.kinds
            ):
                m = _digit_morph(digits = tok.text)
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


def _minutes_follow(toks: list, j: int, spec: LangSpec) -> bool:
    """中文「点」后面是否是分钟：「十五」「零五」这类小数读法里不会出现的写法，或后面跟着「分」。

    「三点五」仍按小数处理；「三点五分」「三点十五」「三点零五」按时刻处理（另有按小数处理的候选）。
    """
    n = len(toks)
    if j >= n:
        return False
    tok = toks[j]
    zero_lead = False
    if tok.kind == "word" and any(m.cls == "zero" for a in spec.analyses(word = tok.text) for m in a.morphs):
        if j + 1 >= n or not _numeral_start(tok = toks[j + 1], spec = spec):
            return False
        zero_lead = True
        j += 1
        tok = toks[j]
    if tok.kind == "digit" and not tok.frac:
        value, end = int(tok.text), j + 1
    elif tok.kind == "word":
        res = parse_number(toks = toks, start = j, spec = spec)
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


def _numeral_start(tok: Tok, spec: LangSpec) -> bool:
    return tok.kind == "digit" or (tok.kind == "word" and bool(spec.analyses(word = tok.text)))


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
            )
            frac_ctx = items[-1].frac if isinstance(items[-1], Num) else False
            continue
        # 普通词
        w = t.text
        nxt = toks[i + 1] if i + 1 < n else None
        nxt_num = nxt is not None and _numeral_start(tok = nxt, spec = spec)
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
        if (pos == "post" and items and isinstance(items[-1], Num)) or (pos == "pre" and nxt_num):
            items.append(Mark(kind = "pct", origin = "word", surface = w.replace(" ", "")))
            frac_ctx = False
            i += 1
            continue
        res = parse_number(
            toks = toks,
            start = i,
            spec = spec,
            frac = frac_ctx,
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


def _digit_item(toks: list, i: int, spec: LangSpec, items: list, frac_ctx: bool) -> int:
    """处理一个阿拉伯数字记号，返回下一个位置；后面接位值词时组成混写数。"""
    t = toks[i]
    frac = t.frac or frac_ctx
    joiner = "" if spec.scan else " "
    if not frac and t.kinds != ORD_ONLY:
        m = _digit_morph(digits = t.text, needs = _MULT_NEEDS)
        res = None
        if m is not None:
            res = parse_number(
                toks = toks,
                start = i,
                spec = spec,
                first = m,
            )
        if res is not None and res[0] > i + 1:
            end, st, _ = res
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


def analyze(text: str, spec: LangSpec, ext: bool = False) -> list:
    """返回文本的比较单元序列；数字写法有歧义时返回多种候选。

    ext 为 True 时用于比对：另外统一单位、货币、时刻的读法与符号写法，合并书写变体，
    并加入语言自己的候选切分（spec.variants）。评测打分用 ext = False，结果与原来相同。
    """
    if not ext:
        toks = tokenize(text = text, spec = spec)
        return [parse_items(toks = variant, spec = spec) for variant in _expand(toks = toks)]
    from .extend import extend_items

    toks = tokenize(text = text, spec = spec, ext = True)
    conflict = spec.ext is not None and bool(spec.sep_words & spec.ext.hour)
    out = []
    for variant in _expand(toks = toks):
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
    if spec.variants is not None:
        for items in list(out):
            for extra in spec.variants(items = items):
                if extra not in out:
                    out.append(extra)
    return out[:MAX_EXT_VARIANTS]
