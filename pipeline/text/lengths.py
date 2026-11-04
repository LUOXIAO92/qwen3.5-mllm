"""估计朗读时长与长度分档。"""

from __future__ import annotations

from .langs import CHARS_PER_SECOND, LONG_MAX_S, MEDIUM_MAX_S, MIN_DURATION_S, SHORT_MAX_S

BINS = ("short", "medium", "long")


def estimate_duration(text: str, *, lang: str) -> float:
    """按语言的估计语速（字符/秒）把字符数换算成秒。"""
    return len(text) / CHARS_PER_SECOND[lang]


def length_bin(duration_s: float) -> str | None:
    """时长落入的档位；太短或超过 30 秒返回 None。"""
    if duration_s < MIN_DURATION_S or duration_s > LONG_MAX_S:
        return None
    if duration_s <= SHORT_MAX_S:
        return "short"
    if duration_s <= MEDIUM_MAX_S:
        return "medium"
    return "long"
