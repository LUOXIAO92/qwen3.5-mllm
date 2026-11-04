"""去重：规范化键完全相同的只留一条，并限制同一开头的条目数量。"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from .normalize import norm_key


class Deduper:
    """按规范化键去重的过滤器，可以跨来源共用同一个实例。

    `max_per_prefix` 限制键的前 `prefix_len` 个字符相同的条目数（维基里大量
    「X 是一种……」式的重复开头会挤占样本）；设为 0 不限制。
    """

    def __init__(self, *, lang: str, max_per_prefix: int = 0, prefix_len: int = 12) -> None:
        self.lang = lang
        self.max_per_prefix = max_per_prefix
        self.prefix_len = prefix_len
        self._seen: set[str] = set()
        self._prefix_count: dict[str, int] = {}

    def add(self, text: str) -> bool:
        """第一次见到返回 True 并登记；重复返回 False。"""
        key = norm_key(text, lang = self.lang)
        if not key or key in self._seen:
            return False
        if self.max_per_prefix:
            prefix = key[: self.prefix_len]
            if self._prefix_count.get(prefix, 0) >= self.max_per_prefix:
                return False
            self._prefix_count[prefix] = self._prefix_count.get(prefix, 0) + 1
        self._seen.add(key)
        return True

    def mark_seen(self, text: str) -> None:
        """只登记不判定，用来预置已有数据。"""
        self._seen.add(norm_key(text, lang = self.lang))


def dedup(texts: Iterable[str], *, lang: str, max_per_prefix: int = 0) -> Iterator[str]:
    """对文本序列去重，保持原顺序。"""
    deduper = Deduper(lang = lang, max_per_prefix = max_per_prefix)
    for text in texts:
        if deduper.add(text):
            yield text
