"""按语言补充的过滤规则，来自独立审阅（949 条抽查）发现的系统性问题。

`check_language_rules` 由 `filters.check_text` 调用，返回丢弃原因或 None。每条规则都有单元测试，
阈值的理由写在对应常量旁边。
"""

from __future__ import annotations

import re

from .langs import CASED_LANGS
from .normalize import norm_key
from .split import split_sentences

# ---------------------------------------------------------------- 通用

# 引用或模板删除后的标点残留：", ,"、",."、"(,"；阿拉伯文逗号同理。
_PUNCT_RESIDUE = re.compile(r",\s*,|،\s*،|[,،]\s*\.(?!\.)|\(\s*[,;]")
# 恰好两个句点（三个是省略号，不算）：删除内容后常留下 "x.."。
_DOUBLE_DOT = re.compile(r"(?<![.…])\.\.(?!\.)")
# 空格斜杠空格：诗行分隔（Tatoeba）、双历法（阿拉伯语维基）。
_SLASH_SEPARATOR = re.compile(r"\s/\s")
# 分数式比分、战绩（4—2、3:1）出现 4 次以上视为统计行。
_SCORE = re.compile(r"\d+\s*[—–:]\s*\d+")
_BRACKET_GROUP = re.compile(r"\([^()]*\)|（[^（）]*）")
# 编号清单：多处「1–3 位数字 + 句点 + 空格 + 大写字母」（德语、土耳其语的序数除外）。
_NUMBERED_ITEM = re.compile(r"(?<![\d.])\d{1,3}\.\s+[A-ZÀ-ÖØ-ÞА-Я]")
_ORDINAL_LANGS = {"de", "tr"}

# 紧邻重复的功能词，按语言列出（删掉数值或名词后留下 "de de"）。
_REPEATED_FUNCTION = {
    "fr": "de|du|des|la|le|les|à|un|une|et|que|qui",
    "it": "di|del|della|dei|delle|il|lo|la|le|gli|un|una|a|da|in|per|con|e|che",
    "es": "de|del|el|la|los|las|un|una|y|que|en|a",
    "pt": "de|do|da|dos|das|o|a|os|as|um|uma|e|que|em",
    "en": "the|a|an|of|to|in|and|or",
    "de": "der|die|das|den|dem|des|ein|eine|und|zu|von|in",
}
_REPEATED_FUNCTION_RE = {
    lang: re.compile(rf"(?<![\w-])({words})\s+\1(?![\w-])", re.IGNORECASE) for lang, words in _REPEATED_FUNCTION.items()
}
# 实词紧邻重复（"terdapat terdapat"）：三个字母以上；这些词本身可以合法重复。
_REPEATED_WORD = re.compile(r"(?<![\w-])(\w{3,})\s+\1(?![\w-])", re.IGNORECASE)
# 只对这些语言查实词重复：其余语言（越南语、波斯语、朝鲜语、阿拉伯语、葡萄牙语等）有大量合法的叠词。
_REPEATED_WORD_LANGS = {"en", "de", "fr", "it", "es", "id", "ms", "ru", "tr", "fil"}
_REPEAT_ALLOWED = {"that", "had", "nous", "vous", "bye", "very", "knock", "ha", "haha", "pas", "sehr", "tres", "très"}

# ---------------------------------------------------------------- 各语言

# 法语、意大利语：句末是省音词（l'、d'、dell'），说明后面的词被删。
_ELIDED_END = re.compile(r"[A-Za-zÀ-ÿ]['’]\s*[.!?…]*$")
# 问号或感叹号后接小写字母（机翻字幕式断句）。
_LOWER_AFTER_QUESTION = re.compile(r"[?!][\"'”»)]*\s+[a-zà-ÿа-яё]")

# 朝鲜语：汉字、假名括注，外语对照括注，依赖后文的句尾。
_KO_FOREIGN_SCRIPT = re.compile(r"[一-鿿぀-ヿ]")
_KO_GLOSS = re.compile(r"\([^)]*[가-힣]+어\s*:")
_KO_DEPENDENT_END = re.compile(r"(다음과 같다|다음과 같습니다|아래와 같다|참조하라|참고하라|가리킨다|다음과 같이 \S+)\.?$")

# 日语：依赖后文的句尾。
_JA_DEPENDENT_END = re.compile(r"(次の通り|以下の通り|次のとおり|以下のとおり)。?$")

# 阿拉伯语：列表引言。
_AR_LIST_INTRO = re.compile(r"^(هذه\s+)?(قائمة|صفحة)\s")

# 俄语：拉丁字母与西里尔字母混在同一个词里（象棋记谱 Лf5），以及 W/L 一类统计。
_RU_MIXED_TOKEN = re.compile(r"[а-яёА-ЯЁ][A-Za-z]|[A-Za-z][а-яёА-ЯЁ]")
_RU_STATS = re.compile(r"\b[WLDwld]/[WLDwld]\b|\b[a-h][1-8]\b")

# 泰语：以连词、关系词开头的是被截断的后半句；以连词、介词结尾的是被截断的前半句；
# 句中带 " - " 或冒号的是署名、标题；词尾落在前置元音（เ แ โ ไ ใ）上是词被截断。
_TH_START = re.compile(r"^(แต่|และ|ดังนั้น|ส่วน|หรือ|ซึ่ง|โดย)")
_TH_END = re.compile(r"(และ|หรือ|ที่|ใน|จาก|กับ|(?<!ก)ว่า|แต่|โดย|เพื่อ|ซึ่ง|เมื่อ|ถ้า|เพราะ|[เแโไใ])$")
_TH_LABEL = re.compile(r"\s[-–—:]\s|:")


def _repeated_sentence(text: str, *, lang: str) -> bool:
    sentences = split_sentences(text, lang = lang)
    if len(sentences) < 2:
        return False
    keys = [norm_key(sentence, lang = lang) for sentence in sentences]
    return len(set(keys)) < len(keys)


def check_language_rules(text: str, *, lang: str, max_bracket_groups: int = 1, max_bracket_ratio: float = 0.25) -> str | None:
    """通用规则 + 各语言规则；命中返回原因。"""
    if _PUNCT_RESIDUE.search(text):
        return "punct_residue"
    if _DOUBLE_DOT.search(text):
        return "double_dot"
    if _SLASH_SEPARATOR.search(text):
        return "slash_separator"
    if len(_SCORE.findall(text)) >= 4:
        return "stats_line"

    groups = _BRACKET_GROUP.findall(text)
    if len(groups) > max_bracket_groups:
        return "many_brackets"
    if groups and sum(len(group) for group in groups) / len(text) > max_bracket_ratio:
        return "bracket_heavy"
    if lang not in _ORDINAL_LANGS and len(_NUMBERED_ITEM.findall(text)) >= 2:
        return "numbered_list"

    if lang in _REPEATED_FUNCTION_RE and _REPEATED_FUNCTION_RE[lang].search(text):
        return "repeated_function_word"
    if lang in _REPEATED_WORD_LANGS:
        for match in _REPEATED_WORD.finditer(text):
            if match.group(1).lower() not in _REPEAT_ALLOWED:
                return "repeated_word"
    if _repeated_sentence(text, lang = lang):
        return "repeated_sentence"

    if lang in {"fr", "it"} and _ELIDED_END.search(text):
        return "elided_end"
    if lang in CASED_LANGS and _LOWER_AFTER_QUESTION.search(text):
        return "lowercase_after_question"

    if lang == "ko":
        if _KO_FOREIGN_SCRIPT.search(text):
            return "ko_foreign_script"
        if _KO_GLOSS.search(text):
            return "ko_gloss"
        if _KO_DEPENDENT_END.search(text):
            return "dependent_ending"
    elif lang == "ja":
        if _JA_DEPENDENT_END.search(text):
            return "dependent_ending"
    elif lang == "ar":
        if _AR_LIST_INTRO.search(text):
            return "list_intro"
    elif lang == "ru":
        if _RU_MIXED_TOKEN.search(text) or _RU_STATS.search(text):
            return "notation"
    elif lang == "th":
        if _TH_START.search(text) or _TH_END.search(text) or _TH_LABEL.search(text):
            return "th_fragment"
    return None
