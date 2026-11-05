"""数词解析的通用部分：语素、语言规格和位值组合器。

每种语言只提供词表（词 → 语素序列）和少量开关，组合规则对所有语言相同：
数由若干「组」构成，组内按位值从高到低相加（十位、个位不能重复占用），
遇到「千、万、million」这类大单位时，把当前组作为系数乘上去。
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from typing import Callable, NamedTuple

from .normalize import normalize_text

CARD = "card"
ORD = "ord"
CARD_ONLY = frozenset({CARD})
ORD_ONLY = frozenset({ORD})
CARD_OR_ORD = frozenset({CARD, ORD})

_EMPTY = frozenset()
_P0 = frozenset({0})
_P1 = frozenset({1})


def places_of(value: int) -> frozenset:
    """数值中非零数字所在的位（个位为 0）。"""
    digits = str(value)[::-1]
    return frozenset(k for k, d in enumerate(digits) if d != "0")


@dataclass(frozen = True, slots = True)
class Morph:
    """数词语素。

    cls 取值：
      add          加法词，value 占用 places 中的各位（个位、十几、几十、几百）
      zero         零，只能单独成数
      place        组内位值词（十、百、千、hundred），value 为 10 的幂
      big          大单位（thousand、万、million），前面的组作为系数
      big_lex      自带系数的大单位（阿拉伯语 ألفان = 2 × 1000），coef 为系数
      teen_suffix  加在单个个位数后面构成十几（印尼语 belas、阿拉伯语 عشر）
      teen_prefix  加在单个个位数前面构成十几（菲律宾语 labing）
      conn         连接词（and、und、y、e、و），只在能继续组成数时吸收
      gap          表示空位的词（中文「零」、越南语 linh），用法同连接词
      ord_mark     把前面的数变成序数的独立标记（波斯语 ام）
    """

    cls: str
    value: int = 0
    places: frozenset = _EMPTY
    ordinal: bool = False
    implicit_one: bool = True
    coef_max: int = 9
    coef: int = 1
    needs: frozenset | None = None
    # after：tens 只能紧跟十位；place_tens 只能紧跟「系数 + 十」（越南语 mươi）；any 只能跟在数内
    after: str | None = None
    not_after_tens: bool = False
    conn_after: frozenset | None = None
    conn_before: frozenset | None = None
    linker: bool = False
    fold: str | None = None
    # 数词体系（韩语 sino 汉字词、native 固有词），同一组内不能混用，见 NumState._system_ok
    system: str | None = None


def label(m: Morph) -> str:
    """语素的类别标签，用于连接词和 needs 的约束。"""
    if m.cls != "add":
        return m.cls
    return _label_places(places = m.places)


def _label_places(places: frozenset) -> str:
    if places == _P0:
        return "unit"
    if max(places) >= 2:
        return "hund"
    if 0 in places:
        return "teen"
    return "tens"


# ---------------------------------------------------------------------------
# 构造语素的辅助函数
# ---------------------------------------------------------------------------

def add(value: int, places = None, **kw) -> Morph:
    if places is None:
        places = places_of(value = value)
    return Morph(
        cls = "add",
        value = value,
        places = frozenset(places),
        **kw,
    )


def teen10(**kw) -> Morph:
    """不能再接个位数的「十」（英语 ten、俄语 десять）：同时占用十位和个位。"""
    return add(value = 10, places = {1, 0}, **kw)


def zero(**kw) -> Morph:
    return Morph(cls = "zero", **kw)


def place(exp: int, **kw) -> Morph:
    return Morph(cls = "place", value = 10 ** exp, **kw)


def big(value: int, **kw) -> Morph:
    return Morph(cls = "big", value = value, **kw)


def big_lex(coef: int, value: int, **kw) -> Morph:
    return Morph(
        cls = "big_lex",
        value = value,
        coef = coef,
        **kw,
    )


def conn(after = None, before = None, linker = False) -> Morph:
    return Morph(
        cls = "conn",
        conn_after = frozenset(after) if after is not None else None,
        conn_before = frozenset(before) if before is not None else None,
        linker = linker,
    )


def gap(before = None) -> Morph:
    return Morph(cls = "gap", conn_before = frozenset(before) if before is not None else None)


TEEN_SUFFIX = Morph(cls = "teen_suffix")
TEEN_PREFIX = Morph(cls = "teen_prefix")
ORD_MARK = Morph(cls = "ord_mark")
CONN_CLITIC = Morph(cls = "conn")


def ordinal(morph: Morph) -> Morph:
    """返回同一语素的序数形式。"""
    return dataclasses.replace(morph, ordinal = True)


def with_fold(morph: Morph, fold: str, **kw) -> Morph:
    """返回带比较用规范词形的语素，例如越南语 mốt 比较时按 một 处理。"""
    return dataclasses.replace(morph, fold = fold, **kw)


# ---------------------------------------------------------------------------
# 组合器状态
# ---------------------------------------------------------------------------

class NumState:
    """正在组合的一个数。step 返回新状态，不能接受该语素时返回 None。"""

    __slots__ = (
        "total", "last_big", "cur", "places", "last", "last_val", "last_places",
        "ord", "pending", "need", "n", "frac", "teen", "system",
    )

    def __init__(self, frac: bool = False):
        self.total = 0
        self.last_big = None
        self.cur = 0
        self.places = _EMPTY
        self.last = None
        self.last_val = 0
        self.last_places = _EMPTY
        self.ord = False
        self.pending = None
        self.need = None
        self.n = 0
        self.frac = frac
        self.teen = False
        self.system = None

    def _copy(self) -> "NumState":
        s = NumState.__new__(NumState)
        for name in NumState.__slots__:
            setattr(s, name, getattr(self, name))
        return s

    def key(self) -> tuple:
        return tuple(getattr(self, name) for name in NumState.__slots__)

    def complete(self) -> bool:
        return self.n > 0 and self.pending is None and self.need is None and not self.teen

    def value(self) -> int:
        return self.total + self.cur

    def apply(self, morphs, spec) -> "NumState | None":
        s = self
        for m in morphs:
            s = s.step(m = m, spec = spec)
            if s is None:
                return None
        return s

    def step(self, m: Morph, spec) -> "NumState | None":
        cls = m.cls
        lab = label(m = m)
        if self.last == "zero":
            return None
        if self.frac and cls not in ("add", "zero"):
            return None
        if self.need is not None and lab not in self.need:
            return None
        pend = self.pending
        if self.ord:
            allowed = (
                (spec.ordinal_all_parts and m.ordinal and cls == "add")
                or (
                    spec.ordinal_unit_head
                    and self.last == "unit"
                    and (cls in ("conn", "teen_suffix") or (pend is not None and lab == "tens"))
                )
            )
            if not allowed:
                return None
        if cls in ("conn", "gap"):
            if self.n == 0 or pend is not None or self.need is not None or self.teen:
                return None
            if m.conn_after is not None and self.last not in m.conn_after:
                return None
            s = self._copy()
            s.pending = m
            return s
        if pend is not None and pend.conn_before is not None and lab not in pend.conn_before:
            return None
        if cls == "ord_mark":
            if self.n == 0 or pend is not None or self.ord or self.teen:
                return None
            s = self._copy()
            s.ord = True
            return s
        if cls == "zero":
            if self.n:
                return None
            s = NumState(frac = self.frac)
            s.n = 1
            s.last = "zero"
            s.ord = m.ordinal
            return s
        if m.after == "tens" and (self.last_places != _P1 or pend is not None):
            return None
        if m.after == "place_tens" and (self.last_places != _P1 or self.last != "place" or pend is not None):
            return None
        if m.after == "any" and (self.n == 0 or 0 in self.places):
            return None
        if m.not_after_tens and pend is None and self.last_places == _P1:
            return None
        if cls == "add":
            return self._add(
                m = m,
                lab = lab,
                pend = pend,
                spec = spec,
            )
        if cls == "place":
            return self._place(m = m, pend = pend, spec = spec)
        if cls == "big":
            return self._big(m = m, pend = pend, spec = spec)
        if cls == "big_lex":
            return self._big_lex(m = m, pend = pend)
        if cls == "teen_suffix":
            return self._teen_suffix(pend = pend)
        if cls == "teen_prefix":
            return self._teen_prefix(pend = pend, spec = spec)
        return None

    def _fits(self, places: frozenset, lab: str, pend, spec) -> bool:
        """当前组能否再加上占用 places 的部分：位不重复，并且从高位到低位。"""
        if not self.places:
            # 当前组为空：数的开头，或刚接过大单位
            if self.last_big is not None and places and 10 ** max(places) >= self.last_big:
                return False
            return True
        if places & self.places:
            return False
        if max(places) < min(self.places):
            return True
        # 德语 einundzwanzig、阿拉伯语 خمسة وعشرون：个位在前，用连接词接十位
        return (
            spec.unit_before_tens
            and pend is not None
            and pend.cls == "conn"
            and self.last == "unit"
            and lab == "tens"
        )

    def _finish(self, s: "NumState", m: Morph) -> "NumState":
        s.pending = None
        s.need = m.needs
        s.n += 1
        if m.ordinal:
            s.ord = True
        if m.system is not None:
            s.system = m.system
        return s

    def _system_ok(self, m: Morph) -> bool:
        """同一组内不混用两套数词：韩语「십세」不是 13，而是「십」（10）加量词「세」（岁）。

        只有汉字词的百位以上后面可以接固有词（백스물셋）。大单位之后开始新的一组，不受限制。
        """
        if m.system is None or self.system is None or m.system == self.system or not self.places:
            return True
        return self.system == "sino" and m.system == "native" and self.last == "place" and self.last_val >= 100

    def _add(self, m: Morph, lab: str, pend, spec):
        if not self._system_ok(m = m):
            return None
        value, places = m.value, m.places
        if self.teen:
            value += 10
            places = frozenset({1, 0})
            lab = "teen"
        if not places:
            return None
        fits = self._fits(
            places = places,
            lab = lab,
            pend = pend,
            spec = spec,
        )
        if not fits:
            return None
        s = self._copy()
        s.cur += value
        s.places = self.places | places
        s.last = lab
        s.last_val = value
        s.last_places = places
        s.teen = False
        return self._finish(s = s, m = m)

    def _group_ok(self, places: frozenset) -> bool:
        return self.last_big is None or 10 ** max(places) < self.last_big

    def _place(self, m: Morph, pend, spec):
        if not self._system_ok(m = m):
            return None
        mult = m.value
        exp = len(str(mult)) - 1
        coef_pend_ok = pend is None or pend.linker
        if self.n > 0 and self.last in ("unit", "teen", "tens") and coef_pend_ok:
            # 刚加上的个位数作为系数：三百、five hundred
            c = self.last_val
            if self.last == "unit" or c <= m.coef_max:
                rest = self.places - self.last_places
                new_places = places_of(value = c * mult)
                ordered = not rest or (not (rest & new_places) and max(new_places) < min(rest))
                if ordered and self._group_ok(places = new_places):
                    s = self._copy()
                    s.cur = self.cur - c + c * mult
                    s.places = rest | new_places
                    s.last = "place"
                    s.last_val = c * mult
                    s.last_places = new_places
                    return self._finish(s = s, m = m)
            # 年份式读法 twenty-one hundred：整组作为系数
            if m.coef_max >= 99 and 10 <= self.cur <= m.coef_max and self.last_big is None and max(self.places) <= 1:
                new_places = places_of(value = self.cur * mult)
                s = self._copy()
                s.cur = self.cur * mult
                s.places = new_places
                s.last = "place"
                s.last_val = s.cur
                s.last_places = new_places
                return self._finish(s = s, m = m)
            if self.last == "unit":
                return None
        if not m.implicit_one:
            return None
        new_places = frozenset({exp})
        fits = self._fits(
            places = new_places,
            lab = "place",
            pend = pend,
            spec = spec,
        )
        if not fits:
            return None
        s = self._copy()
        s.cur += mult
        s.places = self.places | new_places
        s.last = "place"
        s.last_val = mult
        s.last_places = new_places
        return self._finish(s = s, m = m)

    def _big(self, m: Morph, pend, spec):
        mult = m.value
        if self.last_big is not None and mult >= self.last_big:
            # 西语 mil millones、中文 万亿：已有部分整体乘上更大的单位
            if spec.big_compound and self.n > 0 and pend is None and self.value() < mult:
                s = self._copy()
                s.total = self.value() * mult
                s.cur = 0
                s.places = _EMPTY
                s.last_big = mult
                s.last = "big"
                s.last_val = s.total
                s.last_places = _EMPTY
                s.system = None
                return self._finish(s = s, m = m)
            return None
        if self.places:
            c = self.cur
        elif self.n == 0:
            if not m.implicit_one:
                return None
            c = 1
        else:
            return None
        if c >= mult:
            return None
        s = self._copy()
        s.total = self.total + c * mult
        s.cur = 0
        s.places = _EMPTY
        s.last_big = mult
        s.last = "big"
        s.last_val = c * mult
        s.last_places = _EMPTY
        s.system = None
        return self._finish(s = s, m = m)

    def _big_lex(self, m: Morph, pend):
        if self.places or pend is not None:
            return None
        if self.last_big is not None and m.value >= self.last_big:
            return None
        s = self._copy()
        s.total = self.total + m.coef * m.value
        s.last_big = m.value
        s.last = "big"
        s.last_val = m.coef * m.value
        s.last_places = _EMPTY
        s.system = None
        return self._finish(s = s, m = m)

    def _teen_suffix(self, pend):
        if pend is not None or self.last != "unit" or self.last_places != _P0 or 1 in self.places:
            return None
        s = self._copy()
        s.cur += 10
        s.places = self.places | _P1
        s.last = "teen"
        s.last_val += 10
        s.last_places = frozenset({1, 0})
        s.need = None
        s.n += 1
        return s

    def _teen_prefix(self, pend, spec):
        fits = self._fits(
            places = frozenset({1, 0}),
            lab = "teen",
            pend = pend,
            spec = spec,
        )
        if not fits:
            return None
        s = self._copy()
        s.teen = True
        s.pending = None
        s.need = frozenset({"unit"})
        s.n += 1
        return s


# ---------------------------------------------------------------------------
# 语言规格
# ---------------------------------------------------------------------------

class Analysis(NamedTuple):
    """一个词作为数词的一种分析。prefix/suffix 是附着在数词上的非数词部分。"""

    prefix: str
    morphs: tuple
    suffix: str
    surface: str
    conn_prefix: bool = False


@dataclass
class LangSpec:
    """一种语言的数词词表和组合开关。"""

    code: str
    compound: bool = False            # 数词可以连写成一个词（德语、意大利语），按部件切分
    scan: bool = False                # 不用空格分词（中文、日语、韩语、泰语），按最长匹配扫描
    dec_comma: bool = False           # 习惯以逗号作小数点
    space_groups: bool = False        # 习惯以普通空格分千位（法语、俄语）
    unit_before_tens: bool = False    # 个位可以在十位之前（德语 und、阿拉伯语 و）
    ordinal_all_parts: bool = False   # 复合序数每一部分都是序数（西语 vigésimo primero）
    ordinal_unit_head: bool = False   # 序数个位后面还能接十位（阿拉伯语 الحادي والعشرون）
    big_compound: bool = False        # 大单位可以叠用（西语 mil millones、中文 万亿）
    digits_after_big: bool = False    # 大单位后可以接阿拉伯数字（中文 1万2000）
    pct_symbol_prefix: bool = False   # 百分号读在数前（中文 百分之、土耳其语 yüzde）
    split_affixes: Callable | None = None
    post: Callable | None = None
    # 比对时追加的候选分析（韩语「육십일」也可以是 60 加「일」），输入一串比较单元，返回其他候选
    variants: Callable | None = None
    # 比对时用的单位、货币、时刻词表和书写变体合并规则（units.Extras），由 langs.get_spec 挂上
    ext: object = None
    pieces: dict = field(default_factory = dict)
    bound: set = field(default_factory = set)
    multiword: dict = field(default_factory = dict)
    lookahead: dict = field(default_factory = dict)
    sep_words: set = field(default_factory = set)
    pct_words: dict = field(default_factory = dict)
    minus_words: set = field(default_factory = set)
    ord_mark_words: set = field(default_factory = set)
    ordinal_suffixes: list = field(default_factory = list)
    _cache: dict = field(default_factory = dict, repr = False)
    _max_piece: int = 0
    _max_multi: int = 1

    # ---- 建表 ----

    def norm(self, word: str) -> str:
        return " ".join(normalize_text(text = word, lang = self.code).split())

    def add(self, words, morphs, bound: bool = False) -> None:
        """登记词形。words 为一个词或词的列表，morphs 为一个语素或语素序列。

        同一个词可以多次登记，得到多个候选分析。bound 表示只能出现在复合词内部、后面还要接部件。
        """
        if isinstance(words, str):
            words = [words]
        seq = (morphs,) if isinstance(morphs, Morph) else tuple(morphs)
        for w in words:
            key = self.norm(word = w)
            self.pieces.setdefault(key, [])
            if seq not in self.pieces[key]:
                self.pieces[key].append(seq)
            if bound:
                self.bound.add(key)
            self._special(key = key)

    def _special(self, key: str) -> None:
        if " " in key:
            parts = tuple(key.split(" "))
            self.multiword[parts] = key
            self._max_multi = max(self._max_multi, len(parts))
        self._max_piece = max(self._max_piece, len(key))

    def add_sep(self, words) -> None:
        """登记小数点的读法（point、Komma、点）。"""
        for w in [words] if isinstance(words, str) else words:
            key = self.norm(word = w)
            self.sep_words.add(key)
            self._special(key = key)

    def add_pct(self, words, position: str = "post") -> None:
        """登记百分号的读法；position 为 post（数后）或 pre（数前，如「百分之」「yüzde」）。"""
        for w in [words] if isinstance(words, str) else words:
            key = self.norm(word = w)
            self.pct_words[key] = position
            self._special(key = key)

    def add_minus(self, words) -> None:
        """登记负号的读法。"""
        for w in [words] if isinstance(words, str) else words:
            key = self.norm(word = w)
            self.minus_words.add(key)
            self._special(key = key)

    def add_ord_mark(self, words) -> None:
        """登记独立的序数标记（波斯语 ام）。"""
        for w in [words] if isinstance(words, str) else words:
            self.ord_mark_words.add(self.norm(word = w))
            self.add(words = w, morphs = ORD_MARK)

    def add_ordinal_suffix(self, pattern: str, kinds: frozenset) -> None:
        """登记阿拉伯数字后的序数标记（3rd、3.、3e、3-й），kinds 为该写法允许的基数/序数属性。"""
        self.ordinal_suffixes.append((re.compile(pattern = pattern), kinds))

    # ---- 分析 ----

    def analyses(self, word: str) -> tuple:
        """一个词作为数词的全部分析；不是数词时返回空元组。"""
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        out = list(self._core(word = word))
        if self.split_affixes is not None:
            for prefix, stem, suffix, is_conn in self.split_affixes(word = word, spec = self):
                for a in self._core(word = stem):
                    if a.prefix or a.suffix:
                        continue
                    out.append(Analysis(
                        prefix = prefix,
                        morphs = a.morphs,
                        suffix = suffix,
                        surface = a.surface,
                        conn_prefix = is_conn,
                    ))
        result = tuple(out)
        self._cache[word] = result
        return result

    def _core(self, word: str) -> list:
        out = []
        if word in self.pieces and word not in self.bound:
            for seq in self.pieces[word]:
                out.append(Analysis(
                    prefix = "",
                    morphs = seq,
                    suffix = "",
                    surface = _surface(text = word, seq = seq),
                ))
        if self.compound and not out:
            for segs in self._segment(word = word):
                out.append(Analysis(
                    prefix = "",
                    morphs = tuple(m for _, seq in segs for m in seq),
                    suffix = "",
                    surface = "".join(_surface(text = text, seq = seq) for text, seq in segs),
                ))
        return out

    def _segment(self, word: str) -> list:
        """把复合词切成词表中的部件，部件少的切法在前。"""
        n = len(word)
        results = []
        dead = set()

        def walk(pos: int, acc: list) -> bool:
            if len(results) >= 16:
                return True
            if pos == n:
                if len(acc) >= 2:
                    results.append(list(acc))
                return True
            if pos in dead:
                return False
            reached = False
            for length in range(min(self._max_piece, n - pos), 0, -1):
                piece = word[pos:pos + length]
                seqs = self.pieces.get(piece)
                if not seqs or " " in piece:
                    continue
                if pos + length == n and piece in self.bound:
                    continue
                for seq in seqs:
                    acc.append((piece, seq))
                    reached = walk(pos = pos + length, acc = acc) or reached
                    acc.pop()
            if not reached:
                dead.add(pos)
            return reached

        walk(pos = 0, acc = [])
        results.sort(key = len)
        return results

    def scan_keys(self) -> set:
        """扫描无空格文本时要识别的全部词（数词语素和小数点、百分号、负号的读法）。"""
        keys = self._cache.get("\0scan")
        if keys is None:
            keys = set(self.pieces) | self.sep_words | set(self.pct_words) | self.minus_words
            self._cache["\0scan"] = keys
        return keys


def _surface(text: str, seq: tuple) -> str:
    if len(seq) == 1 and seq[0].fold is not None:
        return seq[0].fold
    return text.replace(" ", "")
