"""网页来源的整条文本规则（T03b 返工）。

独立审阅发现网页新增条目的问题率约 15–32%：单句问句标题、过短残句、小标题前缀、署名和日期线、行首节号、
缩写处断句的残留、抽取时丢掉链接文本留下的缺口、编码损坏、垃圾和赌博关键词。这些规则作用在最终条目
（整段文本）上，只在合并和核对时调用；抽取阶段按句检查，不调用它们（否则 30 字以下的短句会打断句子拼段）。

`check_web_entry` 返回丢弃原因或 None。`spam = True` 时启用赌博、外汇等关键词，只对 HPLT 和
FineWeb2-HQ-Plus 使用，Wikinews 的新闻正文里这些词可以合法出现。
"""

from __future__ import annotations

import re

from .langs import NO_SPACE_LANGS
from .split import split_sentences

# ---- 1. 单句问句标题、过短残句
_QUESTION_END = re.compile(r"[?？]['\"”’»)]*$")
_MIN_CHARS = {"zh": 10, "ja": 10, "ko": 12, "th": 20}  # 其余语言 30 字符
_MIN_WORDS = 6

# ---- 2. 小标题前缀、幻灯编号、条款编号
_LABEL_PREFIX = re.compile(r"^[^.!?:：。！？]{3,40}(?<!\d)[:：]\s")
_LABEL_PREFIX_CJK = re.compile(r"^[^。！？：]{2,20}：")
_SLIDE_NUMBER = re.compile(r"^\d+/\d+\s")
_ZH_CLAUSE = re.compile(r"^(第.{1,4}条|[（(][一二三四五六七八九十]+[）)])")

# ---- 3. 署名、日期线、摘要前缀
_BYLINE = re.compile(
    r"^[A-Za-z]{3,15}\s?[–—]\s?[A-Z]|^[A-Za-z]{3,15}\s-\s[A-Z]|^بقلم\s|^گروه\s.*\s[–—-]|^\w{3} \d{1,2}, \d{4}\s?[·•]"
)

# ---- 4. 行首节号与列表序号（"3Ang…"、"2Presiden"）
_SECTION_NUMBER = re.compile(r"^\d{1,3}\s?[A-ZÀ-Ý]")

# ---- 5. 以缩写结尾（切句器在缩写点处断句的残留）
_ABBREV_END = {
    "ms": re.compile(r"(?<!\w)(Ybhg|Datuk|Dato|Datin|Dr|Hj|Hjh|Prof|Tn|Pn|Cik|Tuan|Puan|Ir|En|Yb|Yab|Sdr|Sdri|Bhd|Sdn)\.$"),
    "id": re.compile(r"(?<!\w)(Rp|Dr|Drs|Ir|Prof|Hj|Bpk|Sdr|Kec|Kab|Jl|Tn|Ny)\.$"),
}

# ---- 6. 抽取时丢掉链接文本留下的缺口（"Forecasters at the have issued"）
_GAP_PREFIX = re.compile(r"^[,，،、;；]")
_GAP_SPACED = re.compile(r"\s[،，]\s|\s،")
_GAP_LANG = {
    "en": re.compile(r"\b(the|of|at|by|to|in|from)\s+(have|has|had|is|are|was|were|said|says|will)\b", re.IGNORECASE),
    "ko": re.compile(r"^(는|은|가|를|을|의|에서|에게)\s"),
    "ja": re.compile(r"^(は|が|を|の|に)[、，]"),
}

# ---- 7. 编码损坏
_BROKEN_TR = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü]\?[a-zçğıöşü]")
_BROKEN_APOS = re.compile(r"\s'\s?[a-z]\b|\w\s'\s(s|t|re|ll|ve|d|m)\b")

# ---- 8. 垃圾、赌博、外汇推广关键词
_SPAM = {
    "ru": r"бинарн|форекс|казино|ставк[аиу] на спорт|букмекер|кредит онлайн",
    "pt": r"forex|opções binárias|opcoes binarias|cassino|apostas esportivas|bet365|empréstimo pessoal",
    "es": r"forex|apuestas|casino|opciones binarias|préstamo|prestamos? rápido",
    "ko": r"카지노|대출|토토|바카라|슬롯|먹튀|사설",
    "ja": r"カジノ|スロット|キャッシング|消費者金融",
    "vi": r"nhà cái|cá cược|cá độ|xổ số|soi cầu|casino|game bài",
    "tr": r"bahis|casino|kumar|canlı skor|bonus veren|slot|kredi kartı borcu",
    "zh": r"厂家|彩票|赌场|赌博|博彩|娱乐城|代理加盟|批发|报价|供应商|免费咨询|价格优惠",
    "id": r"judi|togel|slot gacor|slot online|bandar|pinjol|pinjaman online|casino|poker",
    "ms": r"judi|slot|kasino|casino|pinjaman peribadi|forex|betting",
    "fil": r"sugal|online casino|slot|betting|pautang|loan",
    "th": r"สล็อต|บาคาร่า|แทงบอล|คาสิโน|หวย|เว็บพนัน|พนันออนไลน์",
    "ar": r"كازينو|قمار|مراهنات|فوركس|رهان",
    "fa": r"کازینو|شرط\s?بندی|قمار|فارکس|بت\s?بندی",
    "en": r"casino|forex|binary options|payday loan|online betting|sportsbook|viagra|cialis",
    "de": r"casino|forex|sportwetten|kredit ohne schufa|potenzmittel",
    "fr": r"casino|forex|paris sportifs|crédit rapide|options binaires",
    "it": r"casinò|casino|forex|scommesse|opzioni binarie|prestito veloce",
}
_SPAM_RE = {lang: re.compile(pattern, re.IGNORECASE) for lang, pattern in _SPAM.items()}


def check_web_entry(text: str, *, lang: str, spam: bool = False) -> str | None:
    """整条文本的网页规则；命中返回原因。"""
    n_chars = len(text)
    if lang in NO_SPACE_LANGS or lang == "ko":
        minimum = _MIN_CHARS[lang]
    else:
        minimum = 30
        if len(text.split()) < _MIN_WORDS:
            return "too_few_words"
    if n_chars < minimum:
        return "too_short_entry"
    short_limit = 25 if lang in {"zh", "ja", "ko", "th"} else 60
    if _QUESTION_END.search(text) and n_chars < short_limit and len(split_sentences(text, lang = lang)) <= 1:
        return "question_heading"
    if lang in {"zh", "ja"}:
        if _LABEL_PREFIX_CJK.search(text):
            return "label_prefix"
    elif _LABEL_PREFIX.search(text):
        return "label_prefix"
    if _SLIDE_NUMBER.search(text) or (lang == "zh" and _ZH_CLAUSE.search(text)):
        return "numbering"
    if _BYLINE.search(text):
        return "byline"
    if _SECTION_NUMBER.search(text):
        return "section_number"
    if lang in _ABBREV_END and _ABBREV_END[lang].search(text):
        return "abbreviation_end"
    if _GAP_PREFIX.search(text) or _GAP_SPACED.search(text) or (lang in _GAP_LANG and _GAP_LANG[lang].search(text)):
        return "link_gap"
    if "�" in text:
        return "encoding"
    if lang == "tr" and _BROKEN_TR.search(text):
        return "encoding"
    if lang in {"en", "id", "ms"} and _BROKEN_APOS.search(text):
        return "encoding"
    if spam and lang in _SPAM_RE and _SPAM_RE[lang].search(text):
        return "spam"
    return None
