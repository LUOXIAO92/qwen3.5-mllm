"""按规则切分句子，用于排除比对和把维基段落拆成整句。

规则切分的原则：宁可少切，不可错切。缩写（`U.S.`、`St.`、`geb.`、`св.`）、单个字母
（首字母、`z. B.`）、带内部句点的缩写（`J.-C.`、`e.g.`）、罗马数字（`Mieszko III.`）和
德语等语言的序数（`am 18. April`）后面的句点都不当作句末；不切的后果只是两句连在一起，
而错切会让片段以半句结尾。泰语没有句末标点，用 PyThaiNLP 的 crfcut（按空格处的统计模型）切分；没有安装 PyThaiNLP 时整段当作一句。
"""

from __future__ import annotations

import re

# 句末标点后可以跟的右引号、右括号。
_CLOSERS = "\"'”’»」』）)]】》〉"

# 东亚句末标点：后面不需要空格就切。
_CJK_END = re.compile(r"([。！？!?…]+[" + re.escape(_CLOSERS) + r"]*)")
# 西文、阿拉伯文句末标点：后面要有空白才切。
_LATIN_END = re.compile(r"([.!?…؟۔]+[" + re.escape(_CLOSERS) + r"]*)(\s+)")

# 以点结尾但不是句末的缩写（小写比较，不带点），各语言共用一张表。
_ABBREVIATIONS = {
    # 英语
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "ft", "vs", "etc", "inc", "ltd", "co",
    "corp", "bros", "fig", "figs", "al", "ed", "eds", "vol", "vols", "pp", "ca", "cf", "approx", "dept", "est",
    "gen", "col", "lt", "sgt", "capt", "cmdr", "adm", "maj", "gov", "sen", "rep", "pres", "hon", "rev", "ave",
    "blvd", "rd", "ph", "assn", "univ", "ft", "oz", "lb", "lbs", "jan", "feb", "apr", "jun", "jul", "aug",
    "sep", "sept", "oct", "nov", "dec", "op", "ch", "sec",
    # 德语
    "geb", "gest", "chr", "bzw", "usw", "ggf", "evtl", "mio", "mrd", "ca", "nr", "hrsg", "vgl", "jh", "jt",
    "std", "sek", "abb", "aufl", "bd", "kap", "tel", "usf", "inkl", "sog", "verh", "gebr", "str",
    "dr", "prof", "dipl", "ing", "mag", "ggü", "ev", "kath", "zz", "zt", "bez", "bzgl", "zzgl",
    # 法语、意大利语、西班牙语、葡萄牙语
    "m", "mm", "mme", "mlle", "pr", "ste", "av", "env", "ex", "chap", "éd", "boul", "apr", "cie", "tél",
    "sig", "sigg", "dott", "dott.ssa", "avv", "art", "pag", "cfr", "ecc", "sra", "srta", "ud", "uds", "núm",
    "pág", "dra", "profa", "ltda", "exmo", "exma", "sto", "sta", "gral", "dpto", "tel", "apdo", "vv",
    # 俄语
    "св", "г", "гг", "ул", "им", "см", "рис", "тыс", "млн", "млрд", "др", "проф", "акад", "доц", "канд",
    "ген", "полк", "кв", "вв", "оз", "обл", "пос", "ст", "стр", "корп", "тел", "руб", "коп", "англ", "нем",
    "тт", "лат", "греч", "фр", "итал", "исп", "пр", "напр", "букв", "букв", "ср", "т", "п", "д", "н", "э", "им",
    # 印尼语、马来语、土耳其语、越南语
    "bpk", "ibu", "hj", "dll", "dsb", "yg", "dkk", "tgl", "sdr", "jl", "prof", "dr", "ir", "drs",
    "bkz", "sn", "tel", "kec", "kab", "kel", "prov", "dgn", "yth", "ybs", "tsb", "spd", "skm", "tbk", "hlm", "jln",
}

# 点前是这些序数的语言：1–2 位数字加句点是序数（am 18. April），不是句末。
_ORDINAL_DOT_LANGS = {"de", "tr", "fil"}
_ORDINAL = re.compile(r"(^|\s)\d{1,2}$")
_ROMAN = re.compile(r"^[IVXLCDM]{2,}$")  # 大写罗马数字（含 XXXXVIII 这类不规范写法）
# 后面紧跟数字的缩写：No. 5、Art. 3、Nr. 7（句点后是数字，不是新句子）。
_BEFORE_NUMBER = {"no", "nr", "art", "fig", "vol", "p", "pp", "s", "ch", "sec", "nº", "n", "№"}


def split_sentences(text: str, *, lang: str) -> list[str]:
    """把文本切成句子列表，不丢字符（只丢句间空白）。"""
    text = text.strip()
    if not text:
        return []
    if lang == "th":
        return _split_thai(text)
    if lang in {"zh", "ja"}:
        return _split_cjk(text)
    return _split_latin(text, lang = lang)


def _split_thai(text: str) -> list[str]:
    try:
        from pythainlp.tokenize import sent_tokenize
    except ImportError:
        return [text]
    return [sentence.strip() for sentence in sent_tokenize(text, engine = "crfcut") if sentence.strip()]


def _split_cjk(text: str) -> list[str]:
    parts = _CJK_END.split(text)
    sentences = []
    for index in range(0, len(parts) - 1, 2):
        sentence = (parts[index] + parts[index + 1]).strip()
        if sentence:
            sentences.append(sentence)
    tail = parts[-1].strip() if len(parts) % 2 == 1 else ""
    if tail:
        sentences.append(tail)
    return sentences


def is_abbreviation(before: str, *, lang: str) -> bool:
    """句点前的文本以缩写、单个字母、带内部句点的缩写、罗马数字或序数结尾时返回 True。"""
    words = before.split()
    if not words:
        return False
    last = words[-1].strip("\"'“‘«(（[").rstrip(".")
    lowered = last.lower()
    if len(last) == 1 and last.isalpha():
        return True
    if "." in last:  # U.S.  J.-C.  e.g.  z.B.
        return True
    if lowered in _ABBREVIATIONS:
        return True
    if _ROMAN.match(last):
        return True  # Mieszko III.  Karl V.  XXXXVIII.
    if lang in _ORDINAL_DOT_LANGS and _ORDINAL.search(before.rstrip()):
        return True
    return False


def _last_word(before: str) -> str:
    words = before.split()
    return words[-1].strip("\"'“‘«(（[").rstrip(".").lower() if words else ""


def _split_latin(text: str, *, lang: str) -> list[str]:
    sentences = []
    start = 0
    for match in _LATIN_END.finditer(text):
        end = match.end(1)
        mark = match.group(1)
        if mark.rstrip(_CLOSERS) == "." and is_abbreviation(text[start:match.start(1)], lang = lang):
            continue
        following = text[match.end():match.end() + 1]
        if following and following.islower():
            continue  # 句点后接小写字母，按句内处理
        if following.isdigit() and mark == "." and _last_word(text[start:match.start(1)]) in _BEFORE_NUMBER:
            continue
        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences
