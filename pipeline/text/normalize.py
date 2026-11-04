"""文本清理与比较用的规范化键。

`clean_text` 只做不改变内容的清理，结果写入条目的 `original`。
`norm_key` 生成比较键，用于去重和 FLEURS 排除：两边都先走同一个函数再比较。
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# 清理时删除的不可见字符：零宽空格、词连接符、BOM、软连字符、方向控制符。
# 零宽不连字（ZWNJ，U+200C）在波斯语里分隔词的部件，保留。
_INVISIBLE = dict.fromkeys(
    [0x200B, 0x200D, 0x2060, 0xFEFF, 0x00AD, 0x200E, 0x200F, 0x061C]
    + list(range(0x202A, 0x202F))
    + list(range(0x2066, 0x206A)),
    None,
)
_SPACES = re.compile(r"[ \t  -   　]+")

# 比较键里额外去掉的符号：阿拉伯语元音符号和 tatweel、俄语重音符。
_KEY_STRIP = dict.fromkeys(list(range(0x064B, 0x0660)) + [0x0640, 0x0670, 0x0301], None)


def clean_text(text: str, *, lang: str) -> str:
    """去掉不可见字符，合并连续空白，统一为 NFC。俄语删除重音符 U+0301。"""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_INVISIBLE)
    if lang == "ru":
        text = text.replace(chr(0x0301), "")
    text = text.replace("\r", " ").replace("\n", " ")
    return _SPACES.sub(" ", text).strip()


@lru_cache(maxsize=1)
def _t2s():
    from opencc import OpenCC

    return OpenCC("t2s")


def norm_key(text: str, *, lang: str) -> str:
    """比较键：NFKC、小写、去掉标点符号、控制符和空白；中文先转简体。

    FLEURS 的 `transcription` 列已经小写、去标点，中日泰的字之间还带空格，
    所以键里空白一律去掉；这样两边规范化后可以直接比较是否相等。
    """
    text = unicodedata.normalize("NFKC", text)
    if lang == "zh":
        text = _t2s().convert(text)
    text = text.casefold().translate(_KEY_STRIP)
    return "".join(ch for ch in text if unicodedata.category(ch)[0] not in "PSZC")
