"""Tatoeba：读取按语言导出的 `<iso3>_sentences.tsv.bz2`（列：句子编号、语言、文本）。

Tatoeba 里人名 Tom 和 Mary 出现得极多，会让朗读稿的专名分布失衡，所以含这两个名字
（及各语言写法）的句子只按 `name_keep` 的概率保留。
"""

from __future__ import annotations

import bz2
import random
import re
from collections.abc import Iterator
from pathlib import Path

from ..langs import TATOEBA_CODES

SOURCE_NAME = "Tatoeba"
SOURCE_SLUG = "tatoeba"
SOURCE_LICENSE = "CC-BY-2.0-FR"

_NAME_LATIN = re.compile(r"\b(Tom|Tomás|Tomas|Mary|Maria|Marie|Mari)\b")
_NAME_OTHER = re.compile("Том|Мэри|Мария|トム|メアリー|メアリ|汤姆|湯姆|玛丽|瑪麗|톰|메리|توم|ماري|تام|ทอม|แมรี่")


def has_common_name(text: str) -> bool:
    """句子里是否含 Tom、Mary 一类的高频人名。"""
    return bool(_NAME_LATIN.search(text) or _NAME_OTHER.search(text))


def read_tatoeba(
    *,
    raw_dir: str | Path,
    lang: str,
    name_keep: float = 0.15,
    seed: int = 0,
) -> Iterator[tuple[str, str]]:
    """逐句产出 (句子, 定位信息)，定位信息形如 `tatoeba:<句子编号>`。"""
    rng = random.Random(seed)
    for code in TATOEBA_CODES[lang]:
        path = Path(raw_dir) / f"{code}_sentences.tsv.bz2"
        if not path.exists():
            continue
        with bz2.open(path, mode = "rt", encoding = "utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                if has_common_name(parts[2]) and rng.random() > name_keep:
                    continue
                yield parts[2], f"tatoeba:{parts[0]}"
