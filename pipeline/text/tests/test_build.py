"""合并与配额的单元测试。"""

from __future__ import annotations

import random

from pipeline.text.build import select_entries


def _rows(source: str, bin_name: str, count: int) -> list[dict]:
    return [
        {"lang": "en", "text": f"{source}{bin_name}{i}", "ref": str(i), "length_bin": bin_name, "est_duration_s": 3.0}
        for i in range(count)
    ]


def test_small_source_not_swamped():
    by_source = {"big": _rows("big", "short", 100), "small": _rows("small", "short", 5)}
    chosen = select_entries(
        by_source,
        quotas = {"short": 10, "medium": 10, "long": 10},
        max_total = 50,
        rng = random.Random(0),
    )
    small = [row for row in chosen if row["text"].startswith("small")]
    assert len(small) == 5  # 轮流取样，小来源全部入选


def test_max_total_respected_and_leftover_fill():
    by_source = {"a": _rows("a", "short", 30), "b": _rows("b", "long", 3)}
    chosen = select_entries(
        by_source,
        quotas = {"short": 10, "medium": 10, "long": 10},
        max_total = 20,
        rng = random.Random(0),
    )
    assert len(chosen) == 20
    assert sum(row["length_bin"] == "long" for row in chosen) == 3
    assert sum(row["length_bin"] == "short" for row in chosen) == 17  # 配额 10 用完后用剩余补满
