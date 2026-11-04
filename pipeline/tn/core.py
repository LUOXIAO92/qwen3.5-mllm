"""朗读稿生成的入口：清理原文、检查需要改写的片段、决定 unchanged / normalized / dropped。"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from .detect import describe, find_issues

LANGS = ("zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it", "id", "ms", "th", "vi", "fil", "ar", "tr", "fa")

# 删除的不可见字符：零宽空格、BOM、词连接符、软连字符、方向控制符。
# 零宽不连字 U+200C 在波斯语中分隔词的部件，零宽连接符 U+200D 用于组合字形，都保留。
_INVISIBLE = [0x200B, 0xFEFF, 0x2060, 0x00AD, 0x200E, 0x200F, 0x061C]
_INVISIBLE += list(range(0x202A, 0x202F)) + list(range(0x2066, 0x206A))
_REMOVE_INVISIBLE = dict.fromkeys(_INVISIBLE, None)
_SPACES = re.compile(pattern = r"\s+")


class TnResult(NamedTuple):
    script: str | None
    status: str
    note: str


@lru_cache(maxsize = 1)
def _zh_t2s() -> dict:
    """繁简单字对照表，与 pipeline/compare 共用同一份数据（来自 OpenCC，Apache-2.0）。"""
    path = Path(__file__).resolve().parent.parent / "compare" / "data" / "zh_t2s.txt"
    table = {}
    for line in path.read_text(encoding = "utf-8").splitlines():
        if len(line) == 2 and not line.startswith("#"):
            table[ord(line[0])] = line[1]
    return table


def to_simplified(text: str) -> str:
    return text.translate(_zh_t2s())


def clean(text: str, lang: str) -> str:
    """NFC、删除不可见字符、空白合并为一个空格；中文转为简体。"""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_REMOVE_INVISIBLE)
    text = _SPACES.sub(repl = " ", string = text).strip()
    if lang == "zh":
        text = to_simplified(text = text)
    return text


def normalize_entry(original: str, lang: str) -> TnResult:
    """为一条原文生成朗读稿。

    没有需要改写的片段时，朗读稿就是清理后的原文（unchanged）；有数字、符号、缩写、罗马数字或网址时
    丢弃（dropped），tn_note 写明原因。
    """
    if lang not in LANGS:
        raise ValueError(f"不支持的语言代码：{lang!r}")
    text = clean(text = original, lang = lang)
    if not text:
        return TnResult(script = None, status = "dropped", note = "原文为空")
    issues = find_issues(text = text, lang = lang)
    if issues:
        return TnResult(script = None, status = "dropped", note = describe(issues = issues))
    note = "转为简体" if lang == "zh" and text != clean(text = original, lang = "") else ""
    return TnResult(script = text, status = "unchanged", note = note)
