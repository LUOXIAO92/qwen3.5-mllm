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

import re
import unicodedata

from .parse import Mark, Num, Text
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
    items = _time_and_units(items = items, ext = ext, joiner = joiner)
    items = _place_prefix(items = items)
    return _fold_text(items = items, spec = spec, ext = ext)


# ---------------------------------------------------------------------------
# 数词的其他切分（供 spec.variants 使用）
# ---------------------------------------------------------------------------

MAX_SPLIT_SPOTS = 3


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


def split_variants(items: list, splitter) -> list:
    """对读法写成的数尝试其他切分，返回额外的候选。

    splitter(num) 返回该数的若干种切分，每种是一串 Num；读音不变，只是数的边界不同
    （韩语「육십일」= 60 + 일，越南语「hai trăm năm」= 200 + năm）。最多展开 MAX_SPLIT_SPOTS 个数。
    """
    spots = []
    for k, it in enumerate(items):
        if isinstance(it, Num) and it.origin == "word" and not it.frac:
            options = [opt for opt in splitter(num = it) if opt]
            if options:
                spots.append((k, options))
        if len(spots) >= MAX_SPLIT_SPOTS:
            break
    if not spots:
        return []
    out = []
    choices = [[None] + options for _, options in spots]
    positions = [k for k, _ in spots]

    def build(level: int, picked: list) -> None:
        if level == len(choices):
            if any(p is not None for p in picked):
                chosen = dict(zip(positions, picked))
                variant = []
                for k, it in enumerate(items):
                    replacement = chosen.get(k)
                    variant.extend([it] if replacement is None else replacement)
                out.append(variant)
            return
        for option in choices[level]:
            build(level = level + 1, picked = picked + [option])

    build(level = 0, picked = [])
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
        if isinstance(it, Mark) and it.kind == "usym":
            entry = ext.units.get(it.surface)
            if entry is not None:
                kind = "unit" if isinstance(prev, Num) else "unit_pre"
                out.append(Mark(
                    kind = kind,
                    origin = "sym",
                    surface = it.surface,
                    ids = entry[1],
                ))
            k += 1
            continue
        if isinstance(it, Text) and isinstance(prev, Num):
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
                if hour is None and conn is not None and conn_end < n and _minute_num(it = items[conn_end]):
                    # 西语 las tres y quince、葡语 três e quinze：连接词在小时和分钟之间
                    hour, hour_end = conn, conn_end
            unit, unit_end = _take(
                items = items,
                k = k,
                vocab = ext.units,
                joiner = joiner,
                max_len = unit_len,
            )
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
                out.append(Mark(
                    kind = "unit",
                    origin = origin,
                    surface = unit.replace(" ", ""),
                    ids = ids,
                ))
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
            if not (minutes.origin == "digit" and minutes.digits == "00"):
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
    return k


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
