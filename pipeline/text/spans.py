"""把段落切成整句，再拼成落在目标长度档内的片段。

段落里任何一句没有通过过滤（空括号、列表引子、URL 等），就在那里断开，只用前后连续干净的
句子；片段永远由完整的句子组成，不会截断半句。
"""

from __future__ import annotations

import random
from collections.abc import Callable

from .langs import LONG_MAX_S, MEDIUM_MAX_S, MIN_DURATION_S, SHORT_MAX_S
from .lengths import estimate_duration
from .split import split_sentences

_BIN_RANGES = {
    "short": (MIN_DURATION_S, SHORT_MAX_S),
    "medium": (SHORT_MAX_S, MEDIUM_MAX_S),
    "long": (MEDIUM_MAX_S, LONG_MAX_S),
}


def clean_runs(
    paragraph: str,
    *,
    lang: str,
    sentence_ok: Callable[[str], bool],
) -> list[list[str]]:
    """切句后，按 `sentence_ok` 把连续通过的句子分成若干段。"""
    runs: list[list[str]] = []
    current: list[str] = []
    for sentence in split_sentences(paragraph, lang = lang):
        if sentence_ok(sentence):
            current.append(sentence)
        else:
            if current:
                runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def join_sentences(sentences: list[str], *, lang: str) -> str:
    """中日文句子之间不加空格，其余语言加一个空格。"""
    return ("" if lang in {"zh", "ja"} else " ").join(sentences)


def build_spans(
    runs: list[list[str]],
    *,
    lang: str,
    rng: random.Random,
    bin_weights: dict[str, float],
) -> list[str]:
    """在每个连续干净的句子段里，随机选目标档位，贪心拼句直到达到目标时长。"""
    names = list(bin_weights)
    weights = [bin_weights[name] for name in names]
    spans = []
    for run in runs:
        start = 0
        while start < len(run):
            low, high = _BIN_RANGES[rng.choices(names, weights = weights)[0]]
            target = rng.uniform(low, high)
            end = start
            text = ""
            while end < len(run) and estimate_duration(text, lang = lang) < target:
                candidate = join_sentences(run[start:end + 1], lang = lang)
                if estimate_duration(candidate, lang = lang) > LONG_MAX_S and end > start:
                    break
                text = candidate
                end += 1
            spans.append(text)
            start = max(end, start + 1)
    return spans
