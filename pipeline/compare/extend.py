"""比对时对比较单元的补充处理：单位与货币、时刻、书写变体。

词表见 units.py。这里的处理两边相同，只在比对时使用（parse.analyze(ext = True)），评测打分不经过这里。

- 时刻：数字之间的冒号「14:30」、时刻式的「14.30」、「14h30」，以及不超过 24 的数后面的「时」类词
  （点、時、시、Uhr、heures、giờ……）换成时刻记号；时刻后的分钟数照常保留，「半」写成 30，
  分钟后的「分」类词和连接词记在时刻记号里（见 _emit_time）。符号写法的分钟为 00 时去掉。
- 单位与货币：紧跟在数后面的单位读法或符号换成单位记号；写在数前的货币符号（$5、Rp 5.000）
  移到这个数的后面。
- 书写变体：拉丁字母上不属于该语言字母表的变音符号去掉；葡萄牙语 m、n 前的 ó/ô、é/ê 合并
  （António、Antônio）；越南语 i/y 两种写法（Mĩ、Mỹ）和 oa、oe、uy 的声调位置（hòa、hoà）合并。
"""

from __future__ import annotations

import dataclasses
import re
import unicodedata

from .parse import Mark, Num, Text, parse_items, tokenize
from .units import fold_latin


def _is_int(it) -> bool:
    return isinstance(it, Num) and not it.frac


def _hour_num(it) -> bool:
    """可以作为时刻小时数的数：不超过 24 的整数，阿拉伯数字最多两位。"""
    if not _is_int(it) or it.value > 24 or "card" not in it.kinds:
        return False
    return it.origin == "word" or len(it.digits) <= 2


def _is_colon(it) -> bool:
    return isinstance(it, Mark) and it.kind == "colon"


def _minute_num(it, colon: bool = False) -> bool:
    """可以作为分钟的数：小于 60 的整数。冒号后必须是两位数字，「时」类词后的阿拉伯数字可以是一位（6시 5분）。"""
    if not (_is_int(it) and it.value < 60 and "card" in it.kinds):
        return False
    if it.origin == "word":
        return True
    return len(it.digits) == 2 if colon else len(it.digits) <= 2


def _take(items: list, k: int, vocab, joiner: str, max_len: int) -> tuple:
    """items[k:] 开头的若干个普通文字单元拼起来是否为 vocab 中的词，返回 (词, 结束位置)，取最长的。"""
    best = (None, k)
    text = ""
    m = k
    while m < len(items) and isinstance(items[m], Text):
        text = items[m].text if m == k else text + joiner + items[m].text
        if len(text.replace(" ", "")) > max_len:
            break
        m += 1
        if text in vocab:
            best = (text, m)
    return best


def _vocab_len(words) -> int:
    return max((len(w.replace(" ", "")) for w in words), default = 0)


def extend_items(items: list, spec) -> list:
    ext = spec.ext
    if ext is None:
        return items
    joiner = "" if spec.scan else " "
    items = _slash_dates(items = items, ext = ext)
    items = _scale_after_decimal(items = items, spec = spec)
    items = _compound_suffix(items = items, spec = spec, ext = ext)
    items = _time_and_units(items = items, ext = ext, joiner = joiner)
    items = _place_prefix(items = items)
    items = _abbreviations(items = items, ext = ext, joiner = joiner)
    items = _fold_text(items = items, spec = spec, ext = ext)
    return _attach_alts(items = items, spec = spec)


# ---------------------------------------------------------------------------
# 数词的其他切分
# ---------------------------------------------------------------------------

# 只与阿拉伯数字写法比较的读法数：surface 加上这个私用区字符，两边都是读法时不会相等
DIGIT_ONLY = "\ue020"

# 中文「千米」「千克」等以「千」开头的单位：「五千米」也可以是「五千」加「米」
ZH_KILO = {
    "千米": ("米", "m"), "千克": ("克", "g"), "千瓦": ("瓦", "watt"), "千伏": ("伏", "volt"),
    "千赫": ("赫", "hertz"), "千赫兹": ("赫兹", "hertz"),
}


def word_num(value: int, surface: str, display: str, kinds) -> Num:
    """切分出来的读法数；split = True，不参与年份式的逐段拼接（「육 십일」不能拼成 611）。"""
    return Num(
        value = value,
        digits = str(value),
        origin = "word",
        kinds = kinds,
        frac = False,
        surface = surface,
        display = display,
        split = True,
    )


def reparse_num(text: str, spec) -> Num | None:
    """一段读法单独成为一个数时返回这个数（split = True），否则返回 None。"""
    items = parse_items(toks = tokenize(text = text, spec = spec), spec = spec)
    if len(items) == 1 and isinstance(items[0], Num) and items[0].origin == "word":
        return dataclasses.replace(items[0], split = True)
    return None


def reparse(text: str, spec) -> int | None:
    """一段读法单独成为一个数时的值，否则返回 None。"""
    num = reparse_num(text = text, spec = spec)
    return None if num is None else num.value


def _context_alts(items: list, k: int, it: Num, ext) -> list:
    """阿拉伯数字或罗马数字写法在读法中多出的词：Henry VIII ↔ Henry the Eighth，11 May ↔ the eleventh of May。"""
    alts = []
    if it.roman:
        alts.extend((1, (*article.replace(" ", ""), it)) for article in ext.roman_articles)
    if ext.months and it.origin == "digit" and not it.frac and len(it.digits) <= 2 and 1 <= it.value <= 31:
        nxt = items[k + 1] if k + 1 < len(items) else None
        prev = items[k - 1] if k > 0 else None
        if isinstance(nxt, Text) and nxt.text in ext.months:
            alts.append((1, (*"the", it, *"of")))
            alts.append((1, (it, *"of")))
        if isinstance(prev, Text) and prev.text in ext.months:
            alts.append((1, (*"the", it)))
    return alts


def _attach_alts(items: list, spec) -> list:
    """给数挂上其他切分（Num.alts），比较时与另一边逐段对应，读音不变。"""
    out = []
    for k, it in enumerate(items):
        if isinstance(it, Num) and it.origin == "digit":
            alts = _context_alts(
                items = items,
                k = k,
                it = it,
                ext = spec.ext,
            )
            out.append(dataclasses.replace(it, alts = tuple(alts)) if alts else it)
            continue
        if not (isinstance(it, Num) and it.origin == "word" and not it.frac and "card" in it.kinds):
            out.append(it)
            continue
        alts = []
        if spec.splitter is not None:
            alts.extend((1, tuple(parts)) for parts in spec.splitter(num = it) if parts)
        nxt = items[k + 1] if k + 1 < len(items) else None
        if isinstance(nxt, Text) and nxt.text in spec.ext.linkers:
            # 菲律宾语 apat na taon ↔ 4 taon：读法数连同后面的 na 对应一个阿拉伯数字（na 在比较单元中逐字计）
            alts.append((1 + len(nxt.text), (dataclasses.replace(it, surface = it.surface + DIGIT_ONLY, alts = ()),)))
        if spec.code == "zh" and isinstance(nxt, Mark) and nxt.kind == "unit" and nxt.origin == "word" and nxt.surface in ZH_KILO:
            base, unit_id = ZH_KILO[nxt.surface]
            value = reparse(text = it.display + "千", spec = spec)
            if value is not None:
                head = word_num(
                    value = value,
                    surface = it.surface + "千",
                    display = it.display + "千",
                    kinds = it.kinds,
                )
                mark = Mark(
                    kind = "unit",
                    origin = "word",
                    surface = base,
                    ids = frozenset({unit_id}),
                )
                alts.append((2, (head, mark)))
        if alts:
            it = dataclasses.replace(it, alts = tuple(alts))
        out.append(it)
    return out


# ---------------------------------------------------------------------------
# 数词与后缀连写的词、缩写
# ---------------------------------------------------------------------------

def _scale_after_decimal(items: list, spec) -> list:
    """小数后的大单位词（1.7 billion、38,2 Mio.、dois vírgula cinco milhões）也成为一个数，
    这样后面的单位（dollars）和写在数前的货币符号都能找到它。"""
    out = []
    for it in items:
        if isinstance(it, Text) and out and isinstance(out[-1], Num) and out[-1].frac:
            for a in spec.analyses(word = it.text, ext = True):
                if len(a.morphs) == 1 and a.morphs[0].cls == "big" and not a.prefix and not a.suffix:
                    it = Num(
                        value = a.morphs[0].value,
                        digits = str(a.morphs[0].value),
                        origin = "word",
                        kinds = frozenset({"card"}),
                        frac = False,
                        surface = a.surface,
                        display = it.text,
                    )
                    break
        out.append(it)
    return out


def _compound_suffix(items: list, spec, ext) -> list:
    """「семнадцатилетний」「hundertjährige」「neunzehnhundertachtziger」拆成数和后缀，与 17-летний、1980er 对应。"""
    if not (ext.compound_any or ext.compound_decade):
        return items
    out = []
    for it in items:
        split = None
        if isinstance(it, Text) and " " not in it.text:
            for decade, suffixes in ((False, ext.compound_any), (True, ext.compound_decade)):
                for suffix in suffixes:
                    stem = it.text[:-len(suffix)]
                    if not it.text.endswith(suffix) or len(stem) < 3:
                        continue
                    num = _stem_num(stem = stem, spec = spec)
                    if num is not None and (not decade or (num.value >= 10 and num.value % 10 == 0)):
                        split = [dataclasses.replace(num, split = False), Text(text = suffix)]
                        break
                if split:
                    break
        out.extend(split or [it])
    return out


def _stem_num(stem: str, spec) -> Num | None:
    """词干是一个读法数时返回它；俄语的复合词干（двадцатипяти）试着切成两段。"""
    num = reparse_num(text = stem, spec = spec)
    if num is not None or spec.compound:
        return num
    for p in range(3, len(stem) - 2):
        num = reparse_num(text = stem[:p] + " " + stem[p:], spec = spec)
        if num is not None:
            return dataclasses.replace(num, display = stem)
    return None


def _abbreviations(items: list, ext, joiner: str) -> list:
    """缩写与符号（Mr.、z. B.、т. е.、&、No. 5）及其读法换成缩写记号，读法对写法时按读法集合比较。"""
    if not ext.abbr:
        return items
    out = []
    k = 0
    n = len(items)
    while k < n:
        it = items[k]
        if isinstance(it, Mark) and it.kind == "usym":
            readings = ext.abbr.get(it.surface)
            number_only = it.surface in ext.abbr_number
            if readings is not None and (not number_only or (k + 1 < n and isinstance(items[k + 1], Num))):
                out.append(Mark(
                    kind = "abbr",
                    origin = "sym",
                    surface = it.surface,
                    ids = readings,
                ))
            k += 1
            continue
        if isinstance(it, Text):
            best = None
            for vocab, origin in ((ext.abbr, "sym"), (ext.abbr_words, "word")):
                word, end = _take(
                    items = items,
                    k = k,
                    vocab = vocab,
                    joiner = joiner,
                    max_len = ext.abbr_chars,
                )
                if word is None:
                    continue
                if word in ext.abbr_number and not (end < n and isinstance(items[end], Num)):
                    continue
                if best is None or end > best[1]:
                    best = (word, end, origin, vocab[word])
            if best is not None:
                word, end, origin, readings = best
                out.append(Mark(
                    kind = "abbr",
                    origin = origin,
                    surface = word.replace(" ", ""),
                    ids = readings,
                ))
                k = end
                continue
        out.append(it)
        k += 1
    return [it for it in out if not (isinstance(it, Mark) and it.kind == "usym")]


# ---------------------------------------------------------------------------
# 斜线日期
# ---------------------------------------------------------------------------

def _digit_int(it) -> bool:
    return isinstance(it, Num) and it.origin == "digit" and not it.frac


def _slash_dates(items: list, ext) -> list:
    """越南语斜线日期：7/1 读作 bảy tháng một，5/2023 读作 năm năm 2023，17/2/2020 读作 … tháng … năm …。

    日期中的斜线换成 date 记号，记号的 ids 是读出的词（tháng 或 năm），比较时对应读法中的这个词；
    不像日期的斜线（1/2、24/7/365）去掉，与原来按标点处理相同。
    """
    if not any(isinstance(it, Mark) and it.kind == "slash" for it in items):
        return items
    out = []
    k = 0
    n = len(items)
    while k < n:
        it = items[k]
        if not (_digit_int(it = it) and k + 2 < n and isinstance(items[k + 1], Mark) and items[k + 1].kind == "slash"):
            if not (isinstance(it, Mark) and it.kind == "slash"):
                out.append(it)
            k += 1
            continue
        parts = [it]
        m = k + 1
        while m + 1 < n and isinstance(items[m], Mark) and items[m].kind == "slash" and _digit_int(it = items[m + 1]):
            parts.append(items[m + 1])
            m += 2
        values = [p.value for p in parts]
        words = None
        if len(parts) == 3 and 1 <= values[0] <= 31 and 1 <= values[1] <= 12 and len(parts[2].digits) == 4:
            words = [ext.date_month, ext.date_year]
        elif len(parts) == 2 and 1 <= values[0] <= 12 and len(parts[1].digits) == 4:
            words = [ext.date_year]
        elif len(parts) == 2 and 1 <= values[0] <= 31 and 1 <= values[1] <= 12 and len(parts[1].digits) <= 2:
            words = [ext.date_month]
        out.append(parts[0])
        for p, w in zip(parts[1:], words or [None] * (len(parts) - 1)):
            if w is not None:
                out.append(Mark(
                    kind = "date",
                    origin = "sym",
                    surface = "/",
                    ids = w,
                ))
            out.append(p)
        k = m
    return out


# ---------------------------------------------------------------------------
# 时刻与单位
# ---------------------------------------------------------------------------

def _time_and_units(items: list, ext, joiner: str) -> list:
    hour_words = ext.hour | ext.hour_sym
    hour_len = _vocab_len(words = hour_words)
    unit_len = ext.max_chars
    out = []
    k = 0
    n = len(items)
    while k < n:
        it = items[k]
        prev = out[-1] if out else None
        if isinstance(it, Mark) and it.kind == "colon":
            nxt = items[k + 1] if k + 1 < n else None
            # 「2:12:00」这类三段写法（时长、带秒的时刻）不按时刻处理
            chained = (k >= 2 and _is_colon(it = items[k - 2])) or (k + 2 < n and _is_colon(it = items[k + 2]))
            is_time = (
                not chained
                and isinstance(prev, Num)
                and prev.origin == "digit"
                and _hour_num(it = prev)
                and isinstance(nxt, Num)
                and nxt.origin == "digit"
                and _minute_num(it = nxt, colon = True)
            )
            if is_time:
                k = _emit_time(
                    items = items,
                    k = k + 1,
                    out = out,
                    ext = ext,
                    joiner = joiner,
                    origin = "sym",
                    first = ":",
                )
            else:
                k += 1
            continue
        if isinstance(it, Mark) and it.kind == "time":
            # 中文「点」在 parse 阶段已经换成时刻记号；「点钟」把后面的「钟」并进来
            word, end = _take(
                items = items,
                k = k + 1,
                vocab = {w[len(it.surface):] for w in ext.hour if w.startswith(it.surface) and w != it.surface},
                joiner = "",
                max_len = hour_len,
            )
            k = _emit_time(
                items = items,
                k = end,
                out = out,
                ext = ext,
                joiner = joiner,
                origin = it.origin,
                first = it.surface + (word or ""),
            )
            continue
        # 写在下一个数前面的货币符号（「(2002), RM1 bilion」「RM80 - RM320」）不归前一个数
        sym = it.text if isinstance(it, Text) else it.surface if isinstance(it, Mark) else None
        prefix_next = sym in ext.prefix and k + 1 < n and isinstance(items[k + 1], Num)
        if isinstance(it, Mark) and it.kind == "usym":
            entry = ext.units.get(it.surface)
            if entry is not None:
                kind = "unit" if isinstance(prev, Num) and not prefix_next else "unit_pre"
                out.append(Mark(
                    kind = kind,
                    origin = "sym",
                    surface = it.surface,
                    ids = entry[1],
                ))
            elif it.surface in ext.abbr:
                out.append(it)   # &、§ 交给缩写处理
            k += 1
            continue
        if isinstance(it, Text) and isinstance(prev, Num) and not prefix_next:
            hour, hour_end = (None, k)
            if _hour_num(it = prev):
                hour, hour_end = _take(
                    items = items,
                    k = k,
                    vocab = hour_words,
                    joiner = joiner,
                    max_len = hour_len,
                )
                conn, conn_end = _take(
                    items = items,
                    k = k,
                    vocab = ext.conn,
                    joiner = joiner,
                    max_len = 3,
                )
                minutes_follow = conn is not None and _conn_minutes(
                    items = items,
                    k = conn_end,
                    ext = ext,
                    joiner = joiner,
                )
                if hour is None and minutes_follow:
                    # 西语 las tres y quince、葡语 três e quinze：连接词在小时和分钟之间
                    hour, hour_end = conn, conn_end
            unit, unit_end = _take(
                items = items,
                k = k,
                vocab = ext.units,
                joiner = joiner,
                max_len = unit_len,
            )
            abbr, abbr_end = _take(
                items = items,
                k = k,
                vocab = ext.abbr,
                joiner = joiner,
                max_len = ext.abbr_chars,
            )
            if abbr is not None and abbr_end > unit_end:
                # 「514 v. Chr.」的 v 是缩写的一部分，不是伏特
                unit = None
            rest = None
            if unit is None and hour is None and abbr is None and ext.unit_suffixes:
                unit, rest = _unit_with_suffix(word = it.text, ext = ext)
                unit_end = k + 1
            if hour is not None and hour_end >= unit_end:
                k = _emit_time(
                    items = items,
                    k = hour_end,
                    out = out,
                    ext = ext,
                    joiner = joiner,
                    origin = "sym" if hour in ext.hour_sym else "word",
                    first = hour.replace(" ", ""),
                )
                continue
            if unit is not None:
                origin, ids = ext.units[unit]
                if origin == "sym":
                    # 「1912 M.」的 M 也可以是 monsieur
                    ids = ids | ext.abbr.get(unit, frozenset())
                out.append(Mark(
                    kind = "unit",
                    origin = origin,
                    surface = unit.replace(" ", ""),
                    ids = ids,
                ))
                if rest:
                    out.append(Text(text = rest))
                k = unit_end
                continue
        if isinstance(it, Text) and it.text in ext.prefix and k + 1 < n and isinstance(items[k + 1], Num):
            entry = ext.units.get(it.text)
            if entry is not None:
                out.append(Mark(
                    kind = "unit_pre",
                    origin = "sym",
                    surface = it.text,
                    ids = entry[1],
                ))
                k += 1
                continue
        out.append(it)
        k += 1
    return out


def _unit_with_suffix(word: str, ext) -> tuple:
    """单位词与词尾连写（土耳其语 kilometreye、km'ye、kilometrekarelik）：返回 (单位, 词尾)，不是时返回 (None, None)。
    单个字母的符号（m、g）只接两个字母以上的词尾，「5 mi」的疑问词 mi 不算米。"""
    for cut in range(len(word) - 1, 0, -1):
        unit, rest = word[:cut], word[cut:]
        if unit in ext.units and rest in ext.unit_suffixes and (len(unit) > 1 or len(rest) > 1):
            return unit, rest
    return None, None


def _conn_minutes(items: list, k: int, ext, joiner: str) -> bool:
    """没有「时」类词时，连接词后的 items[k] 只有后面跟着「分」类词才当作分钟（três e quinze minutos、
    سه و پانزده دقیقه）。其余情况连接词保持原样，由比较时冒号记号对连接词处理（core._conn_step），
    因为「19 e 20」这类写法分不清是时刻还是并列的数（séculos XIX e XX），两边必须得到同样的结论。"""
    n = len(items)
    if k >= n or not _minute_num(it = items[k]):
        return False
    word, _ = _take(
        items = items,
        k = k + 1,
        vocab = ext.minute,
        joiner = joiner,
        max_len = 8,
    )
    return word is not None


def _emit_time(
    items: list,
    k: int,
    out: list,
    ext,
    joiner: str,
    origin: str,
    first: str,
) -> int:
    """写出时刻记号及其后的分钟，返回下一个位置。

    记号的 surface 记录读出的词：小时后的词（点、Uhr、heures）、连接词（e、y）、「半」或分钟后的词（分、
    minutos），用「|」连接，两边都是读法时要求这些词相同，「三点十五分」与「三点十五」不一致。
    冒号写法的 surface 以「:」开头，冒号本身不读，后面没有别的词时可以不对应任何东西。
    """
    n = len(items)
    parts = [first]
    slot = len(out)
    out.append(None)
    half, half_end = _take(
        items = items,
        k = k,
        vocab = ext.half,
        joiner = joiner,
        max_len = 4,
    )
    if half is not None:
        parts.append(half)
        out.append(Num(
            value = 30,
            digits = "30",
            origin = "word",
            kinds = frozenset({"card"}),
            frac = False,
            surface = half,
            display = half,
        ))
        k = half_end
    else:
        conn, conn_end = _take(
            items = items,
            k = k,
            vocab = ext.conn,
            joiner = joiner,
            max_len = 3,
        )
        if conn is not None and conn_end < n and _minute_num(it = items[conn_end]):
            parts.append(conn)
            k = conn_end
        if k < n and _minute_num(it = items[k]):
            minutes = items[k]
            if minutes.origin == "digit" and minutes.digits == "00":
                out.append(Mark(kind = "time00", origin = "sym", surface = "00"))
            else:
                out.append(minutes)
            k += 1
            if minutes.origin == "word" and minutes.value == 0 and k < n and _is_int(it = items[k]) and items[k].value < 10:
                # 三点零五分：「零」与个位分开读
                out.append(items[k])
                k += 1
            word, end = _take(
                items = items,
                k = k,
                vocab = ext.minute,
                joiner = joiner,
                max_len = 8,
            )
            if word is not None:
                parts.append(word.replace(" ", ""))
                k = end
    if first == ":":
        # 14:30 Uhr：冒号写法后面的「时」类词也读出
        word, end = _take(
            items = items,
            k = k,
            vocab = ext.hour | ext.hour_sym,
            joiner = joiner,
            max_len = 8,
        )
        if word is not None:
            parts.append(word.replace(" ", ""))
            k = end
    out[slot] = Mark(kind = "time", origin = origin, surface = "|".join(parts))
    return k


def _number_end(items: list, k: int) -> int:
    """items[k] 开始的一个数（整数部分，以及小数点和小数部分）之后的位置；items[k] 不是数时返回 k。"""
    n = len(items)
    if k >= n or not isinstance(items[k], Num) or items[k].frac:
        return k
    k += 1
    if k + 1 < n and isinstance(items[k], Mark) and items[k].kind == "sep" and isinstance(items[k + 1], Num) and items[k + 1].frac:
        k += 1
        while k < n and isinstance(items[k], Num) and items[k].frac:
            k += 1
    if k < n and _scale_word(it = items[k]):
        k += 1
    return k


def _scale_word(it) -> bool:
    """单独一个大单位词（billion、millones）：值为 1000 的幂。"""
    if not (isinstance(it, Num) and it.origin == "word" and not it.frac and it.value >= 1000):
        return False
    value = it.value
    while value % 1000 == 0:
        value //= 1000
    return value == 1 and " " not in it.display


def _place_prefix(items: list) -> list:
    """写在数前的货币符号移到数后；不在数前也不在数后的单位符号恢复为普通文字或删去。"""
    if not any(isinstance(it, Mark) and it.kind == "unit_pre" for it in items):
        return items
    out = []
    k = 0
    n = len(items)
    while k < n:
        it = items[k]
        if isinstance(it, Mark) and it.kind == "unit_pre":
            end = _number_end(items = items, k = k + 1)
            mark = Mark(
                kind = "unit",
                origin = "sym",
                surface = it.surface,
                ids = it.ids,
            )
            if end > k + 1:
                out.extend(items[k + 1:end])
                out.append(mark)
                k = end
                continue
            if out and isinstance(out[-1], Num):
                out.append(mark)
            elif it.surface.isalpha():
                out.append(Text(text = it.surface))
            k += 1
            continue
        out.append(it)
        k += 1
    return out


# ---------------------------------------------------------------------------
# 书写变体
# ---------------------------------------------------------------------------

_PT_NASAL_O = re.compile(pattern = "[óô](?=[mn])")
_PT_NASAL_E = re.compile(pattern = "[éê](?=[mn])")

_VI_TONES = {"\u0300", "\u0301", "\u0303", "\u0309", "\u0323"}
_VI_IY = re.compile(pattern = "(b|c|ch|d|đ|g|gh|h|k|kh|l|m|n|ng|ngh|nh|p|ph|qu|r|s|t|th|tr|v|x)[iy]")


def vi_fold(word: str) -> str:
    """越南语音节的两种书写变体合并：i/y（Mĩ、Mỹ），oa、oe、uy 的声调位置（hòa、hoà）。"""
    decomposed = unicodedata.normalize("NFD", word)
    tones = [c for c in decomposed if c in _VI_TONES]
    if len(tones) > 1:
        return word
    tone = tones[0] if tones else ""
    base = unicodedata.normalize("NFC", "".join(c for c in decomposed if c not in _VI_TONES))
    if _VI_IY.fullmatch(string = base):
        base = base[:-1] + "i"
    elif not base.endswith(("oa", "oe", "uy")):
        return word
    return unicodedata.normalize("NFC", base + tone)


def _fold_word(text: str, code: str, native: str | None) -> str:
    if native is not None:
        text = fold_latin(text = text, native = native)
    if code == "pt":
        text = _PT_NASAL_E.sub(repl = "ê", string = _PT_NASAL_O.sub(repl = "ô", string = text))
    elif code == "vi":
        text = " ".join(vi_fold(word = w) for w in text.split(" "))
    return text


def _fold_text(items: list, spec, ext) -> list:
    """普通文字和单位、时刻读法的词形都合并书写变体（quilómetros、quilômetros）。"""
    out = []
    for it in items:
        if isinstance(it, Text):
            folded = _fold_word(text = it.text, code = spec.code, native = ext.native_letters)
            if folded != it.text:
                it = Text(text = folded)
        elif isinstance(it, Mark) and it.origin == "word" and it.kind in ("unit", "time"):
            folded = _fold_word(text = it.surface, code = spec.code, native = ext.native_letters)
            if folded != it.surface:
                it = Mark(
                    kind = it.kind,
                    origin = it.origin,
                    surface = folded,
                    ids = it.ids,
                )
        out.append(it)
    return out
