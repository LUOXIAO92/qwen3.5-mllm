"""FLEURS 排除清单：候选文本的规范化键与清单键相等时丢弃。"""

from __future__ import annotations

import json
from pathlib import Path

from .normalize import norm_key
from .split import split_sentences


def load_exclusion(path: str | Path) -> dict[str, set[str]]:
    """读取排除清单，按语言返回规范化键的集合。

    每行两个文本字段都转成键：`raw_transcription`（带标点的原句）和
    `transcription`（FLEURS 自带的规范化句），两者规范化后通常相同。
    """
    keys: dict[str, set[str]] = {}
    with open(path, encoding = "utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            lang = row["lang"]
            bucket = keys.setdefault(lang, set())
            for field in ("raw_transcription", "transcription"):
                key = norm_key(row[field], lang = lang)
                if key:
                    bucket.add(key)
    return keys


def is_excluded(text: str, *, lang: str, exclusion: dict[str, set[str]]) -> bool:
    """整条文本，或其中任一整句，与排除清单相等就返回 True。

    段落类条目是多句拼成的，FLEURS 的句子可能整句出现在段落里，所以逐句比较。
    """
    banned = exclusion.get(lang, set())
    if not banned:
        return False
    if norm_key(text, lang = lang) in banned:
        return True
    return any(norm_key(sentence, lang = lang) in banned for sentence in split_sentences(text, lang = lang))


def load_blocklist(path: str | Path | None = None) -> set[str]:
    """读取人工审阅排除的 source_ref 集合；默认读包内的 `data/blocklist_refs.txt`。"""
    path = Path(path) if path else Path(__file__).parent / "data" / "blocklist_refs.txt"
    refs = set()
    for line in path.read_text(encoding = "utf-8").splitlines():
        ref = line.split("#", 1)[0].strip()
        if ref:
            refs.add(ref)
    return refs
