"""找出原文中需要改写成读法的片段：数字、货币和单位符号、缩写、罗马数字、网址。

find_issues(text, lang) 返回 Issue 列表，每个 Issue 记录种类、在文本中的起止位置和原文写法。
直通版只要发现任何一处就丢弃整条；基础读法先改写能处理的数字片段，再对改写结果重新检查。
"""

from __future__ import annotations

import re
import unicodedata
from typing import NamedTuple

KIND_NAMES = {
    "digit": "数字",
    "symbol": "符号",
    "abbrev": "缩写",
    "roman": "罗马数字",
    "url": "网址",
}


class Issue(NamedTuple):
    kind: str
    start: int
    end: int
    surface: str


# 数字字符：各文字的十进制数字（Nd）、上标和分数（No）、罗马数字字符（Nl）。
# 「〇」是中文、日语的数词用字，按普通文字处理。
_NUMERIC_CATEGORIES = {"Nd", "No", "Nl"}
_NOT_NUMERIC = {"〇"}

# 符号字符：货币（Sc）、数学符号（Sm）、其他符号（So，含 °、℃、№、表情符号），以及下列标点类符号
# 和序数标记 º、ª（3º、36ª）；全角的 ％＃＆＠＊、注释符 ※ 和角分符号 ′″ 也算符号。
_SYMBOL_CATEGORIES = {"Sc", "Sm", "So"}
_SYMBOL_CHARS = set("%‰‱#&@§¶*٪ºª％＃＆＠＊※′″")

# 中间带点的缩写：e.g.、z.B.、U.S.、TP.HCM、ق.م、Rev.W（每段 1–4 个字母）
_DOTTED = re.compile(pattern = r"(?<![\w.])[^\W\d_]{1,4}(?:\.[^\W\d_]{1,4})+\.?(?!\w)")
# 中间带空格的小写缩写：т. д.、z. B.、d. h.、н. э.（首字母小写，避开人名首字母 J. R.）
_SPACED = re.compile(pattern = r"(?<![\w.])([^\W\d_])\.\s([^\W\d_])\.")
_URL = re.compile(pattern = r"https?://\S+|www\.\S+|[\w.+-]+@[\w-]+\.[\w.]+")
# 罗马数字：只由 I、V、X、L 组成的 2 个及以上字母，或含 C、D、M 的 3 个及以上字母，并且是合法写法
_ROMAN_TOKEN = re.compile(pattern = r"(?<!\w)[IVXLCDM]{2,}(?!\w)")
_ROMAN_VALID = re.compile(pattern = r"M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})")
# 法语带序数后缀的罗马数字：XXe siècle、Ier、XIXe
_FR_ROMAN_ORDINAL = re.compile(pattern = r"(?<!\w)(?P<roman>[IVXLCDM]+)(?:e|er|re|ère|ème|è)(?!\w)")
# 泰文字母后直接跟句点（พ.ศ.、น.、ก.ม.）是缩写；ฯ（U+0E2F，ไปยาลน้อย）表示省略
_TH_ABBREV = re.compile(pattern = f"[{chr(0x0E01)}-{chr(0x0E4E)}]\\.|{chr(0x0E2F)}")
# 阿拉伯文、波斯文：字母后接 tatweel（U+0640，هـ、كـم，多用于缩写，偶尔用于拉长字形）、单个字母加句点（د.、م.），
# 以及单独成词的 م、ه（公历、伊斯兰历年份标记，1887 م）
_AR_LETTER = f"{chr(0x0620)}-{chr(0x064A)}{chr(0x066E)}-{chr(0x06D3)}{chr(0x06FA)}-{chr(0x06FF)}"
_AR_ABBREV = re.compile(
    pattern = rf"[{_AR_LETTER}]{chr(0x0640)}+[{_AR_LETTER}]*|(?<![\w])[{_AR_LETTER}]\.|(?<![\w])[مه](?![\w])",
)

# 带句点的常见缩写（区分大小写，按整词匹配）。只收录不会和普通词混淆的写法，
# 例如不收「No.」（英语回答 No.）和德语「Art.」（ihrer Art.）；_COMMON_SKIP 列出在某种语言里是普通词的写法。
_COMMON_ABBREV = [
    "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "St.", "Jr.", "Sr.", "Inc.", "Ltd.", "Corp.", "Co.",
    "vs.", "etc.", "Vol.", "vol.", "Fig.", "fig.", "pp.", "ca.", "approx.",
]
_COMMON_SKIP = {"fr": {"vol."}, "vi": {"ca."}}
_LANG_ABBREV = {
    "en": [
        "Mt.", "Ft.", "Rev.", "Gen.", "Col.", "Lt.", "Sgt.", "Capt.", "Gov.", "Sen.", "Rep.", "Hon.",
        "Bros.", "Dept.", "Univ.", "Ave.", "Blvd.", "Rd.", "Jan.", "Feb.", "Apr.", "Aug.", "Sept.",
        "Oct.", "Nov.", "Dec.",
    ],
    "de": [
        "Hr.", "Fr.", "Str.", "Nr.", "bzw.", "usw.", "ggf.", "evtl.", "inkl.", "vgl.", "sog.", "Mio.",
        "Mrd.", "Tsd.", "Jh.", "Jhd.", "geb.", "gest.", "Abb.", "Bd.", "Hrsg.", "Dipl.", "Ing.", "ehem.",
        "zzgl.", "bspw.", "insb.", "Tel.", "Abs.", "ü.",
    ],
    "fr": [
        "M.", "MM.", "Mme", "Mmes", "Mlle", "Mlles", "Mgr", "av.", "bd", "boul.", "cf.", "env.", "p.",
        "apr.", "Ste",
    ],
    "es": [
        "Sra.", "Srta.", "Dra.", "Ud.", "Uds.", "Vd.", "Vds.", "pág.", "págs.", "núm.", "Av.", "Avda.",
        "Dña.", "aprox.", "Lic.", "Ing.", "Sto.", "Sta.", "art.", "cap.", "sr.", "sra.", "srta.", "dr.", "dra.",
    ],
    "pt": [
        "Sra.", "Srs.", "Sras.", "Dra.", "Exmo.", "Exma.", "Ltda.", "aprox.", "séc.", "pág.", "Av.",
        "Sto.", "Sta.", "sr.", "sra.", "srs.", "sras.", "dr.", "dra.",
    ],
    "it": [
        "Sig.", "Sigg.", "Dott.", "Ing.", "Avv.", "ecc.", "pag.", "Mons.", "sec.", "Geom.", "Rag.",
        "Spett.",
    ],
    "ru": [
        "г.", "гг.", "в.", "вв.", "ок.", "тыс.", "млн", "млн.", "млрд", "млрд.", "руб.", "коп.", "см.",
        "стр.", "им.", "ул.", "д.", "кв.", "пос.", "обл.", "р.", "оз.", "с.", "ст.", "проф.", "акад.",
        "доц.", "англ.", "лат.", "греч.", "рус.", "нем.", "франц.", "род.", "ум.", "др.", "пр.", "св.",
        "т.", "чел.",
    ],
    "id": [
        "dll.", "dsb.", "dst.", "Jl.", "Jln.", "Bpk.", "Sdr.", "Sdri.", "Ir.", "Hj.", "tsb.", "Yth.",
        "yth.", "spt.", "dkk.", "hlm.", "Kab.", "Kec.", "Kel.", "Prov.", "PT.", "Tbk.",
    ],
    "ms": [
        "dll.", "dsb.", "Jl.", "Jln.", "Sdn.", "Bhd.", "Tn.", "Pn.", "En.", "Hj.", "Kg.", "Tg.", "hlm.",
    ],
    "tr": [
        "Doç.", "Av.", "vb.", "vd.", "Cad.", "Sok.", "Mah.", "Bkz.", "bkz.", "örn.", "yy.", "Alb.",
        "Yrd.", "Md.", "Apt.",
    ],
    "vi": ["TP.", "Tp.", "Q.", "P.", "TS.", "ThS.", "GS.", "PGS.", "BS.", "KS.", "TT."],
    "fil": ["Gng.", "Bb.", "Gg.", "atbp.", "Hal.", "Sto.", "Sta."],
}
# 编号缩写后接数字：No. 47、Nr. 5、意大利语 n. 10、c. 890、p. 12（单独的 No. 是英语回答，不算）
_NUMBER_ABBREV = re.compile(pattern = r"(?<![\w.])(?:No|no|Nos|nos|Nr|nr|N|[a-z])\.\s?(?=\d)")
# 德语街道名缩写：Hauptstr.、Bahnhofstr.
_DE_STREET = re.compile(pattern = r"(?<![\w.])\w*[sS]tr\.")
_ABBREV_RE = {}


def _abbrev_re(lang: str):
    """常见缩写的正则。以句点结尾的写法后面可以紧接字母（Dr.Smith），其余写法要求整词。"""
    if lang not in _ABBREV_RE:
        skip = _COMMON_SKIP.get(lang, set())
        words = set(w for w in _COMMON_ABBREV if w not in skip) | set(_LANG_ABBREV.get(lang, []))
        words = sorted(words, key = len, reverse = True)
        dotted = "|".join(re.escape(pattern = w) for w in words if w.endswith("."))
        bare = "|".join(re.escape(pattern = w) for w in words if not w.endswith("."))
        body = rf"(?:{dotted})" + (rf"|(?:{bare})(?!\w)" if bare else "")
        _ABBREV_RE[lang] = re.compile(pattern = rf"(?<![\w.])(?:{body})")
    return _ABBREV_RE[lang]


def _char_runs(text: str, test) -> list:
    """把满足 test 的连续字符合并成片段，返回 (start, end) 列表。"""
    runs = []
    start = None
    for i, ch in enumerate(text):
        if test(ch):
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(text)))
    return runs


def is_numeric_char(ch: str) -> bool:
    return unicodedata.category(ch) in _NUMERIC_CATEGORIES and ch not in _NOT_NUMERIC


def is_symbol_char(ch: str) -> bool:
    return unicodedata.category(ch) in _SYMBOL_CATEGORIES or ch in _SYMBOL_CHARS


def _after_mark(text: str, start: int) -> bool:
    """匹配起点前面是组合符号（阿拉伯文元音符号、泰文上下标等），说明起点在词中间。"""
    return start > 0 and unicodedata.category(text[start - 1]).startswith("M")


def _merge(issues: list) -> list:
    """同一种类的重叠片段只保留最长的一个（例如「т. д.」与「т.」「д.」）。"""
    issues = sorted(issues, key = lambda it: (it.start, -(it.end - it.start)))
    kept = []
    for it in issues:
        if any(k.kind == it.kind and k.start <= it.start and it.end <= k.end for k in kept):
            continue
        kept.append(it)
    return kept


def find_issues(text: str, lang: str) -> list:
    """返回文本中需要改写的片段（按起始位置排序，可能重叠）。"""
    issues = []
    for start, end in _char_runs(text = text, test = is_numeric_char):
        issues.append(Issue(kind = "digit", start = start, end = end, surface = text[start:end]))
    for start, end in _char_runs(text = text, test = is_symbol_char):
        issues.append(Issue(kind = "symbol", start = start, end = end, surface = text[start:end]))
    for m in _URL.finditer(string = text):
        issues.append(Issue(kind = "url", start = m.start(), end = m.end(), surface = m.group()))
    patterns = [_DOTTED, _NUMBER_ABBREV, _abbrev_re(lang = lang)]
    if lang == "th":
        patterns.append(_TH_ABBREV)
    if lang in ("ar", "fa"):
        patterns.append(_AR_ABBREV)
    if lang == "de":
        patterns.append(_DE_STREET)
    for pattern in patterns:
        for m in pattern.finditer(string = text):
            if not _after_mark(text = text, start = m.start()):
                issues.append(Issue(kind = "abbrev", start = m.start(), end = m.end(), surface = m.group()))
    for m in _SPACED.finditer(string = text):
        if m.group(1).islower() and not _after_mark(text = text, start = m.start()):
            issues.append(Issue(kind = "abbrev", start = m.start(), end = m.end(), surface = m.group()))
    if lang == "fr":
        for m in _FR_ROMAN_ORDINAL.finditer(string = text):
            roman = m.group("roman")
            # 单个 L、C、D、M 加 e 是普通词（Le、Ce、De、Me）
            if len(roman) == 1 and roman not in "IVX":
                continue
            if _ROMAN_VALID.fullmatch(string = roman):
                issues.append(Issue(kind = "roman", start = m.start(), end = m.end(), surface = m.group()))
    for m in _ROMAN_TOKEN.finditer(string = text):
        token = m.group()
        if not _ROMAN_VALID.fullmatch(string = token):
            continue
        if len(token) >= 3 or set(token) <= set("IVXL"):
            issues.append(Issue(kind = "roman", start = m.start(), end = m.end(), surface = token))
    issues = _merge(issues = issues)
    issues.sort(key = lambda it: (it.start, it.end))
    return issues


def describe(issues: list, limit: int = 3) -> str:
    """把问题片段写成说明，例如「含数字「1937」「30」；含缩写「Rev.W」」。"""
    grouped = {}
    for it in issues:
        grouped.setdefault(it.kind, [])
        if it.surface not in grouped[it.kind]:
            grouped[it.kind].append(it.surface)
    parts = []
    for kind in KIND_NAMES:
        if kind not in grouped:
            continue
        shown = grouped[kind][:limit]
        more = "等" if len(grouped[kind]) > limit else ""
        parts.append(f"含{KIND_NAMES[kind]}" + "".join(f"「{s}」" for s in shown) + more)
    return "；".join(parts)
