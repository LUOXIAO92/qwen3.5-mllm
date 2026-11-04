"""Common Voice 句库：读取 GitHub 仓库 `server/data/<语言目录>/` 下的句子文件。

仓库里每个语言目录有若干 `.txt`，一行一句：贡献者提交的句子、从维基抽取的句子、
Europarl 等，整体许可 CC0。`singleword-benchmark.txt` 是单词表，跳过。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ..langs import COMMON_VOICE_DIRS

SOURCE_NAME = "Common Voice 句库"
SOURCE_SLUG = "cv"
SOURCE_LICENSE = "CC0-1.0"
_SKIP_FILES = {"singleword-benchmark.txt", "LICENSE"}


def read_common_voice(*, repo_dir: str | Path, lang: str) -> Iterator[tuple[str, str]]:
    """逐行产出 (句子, 定位信息)，定位信息形如 `zh-CN/sentence-collector.txt:123`。"""
    directory = Path(repo_dir) / "server" / "data" / COMMON_VOICE_DIRS[lang]
    for path in sorted(directory.glob("*.txt")):
        if path.name in _SKIP_FILES:
            continue
        with open(path, encoding = "utf-8", errors = "replace") as handle:
            for number, line in enumerate(handle, start = 1):
                text = line.strip()
                if text:
                    yield text, f"{directory.name}/{path.name}:{number}"
