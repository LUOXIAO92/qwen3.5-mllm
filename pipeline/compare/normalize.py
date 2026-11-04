"""字符层面的规范化：两边文本和词表都先经过这里。

处理内容：NFKC、零宽字符和方向控制符、各种数字字形统一为 ASCII 数字、
大小写（土耳其语的 I/İ 单独处理）、俄语 ё、阿拉伯语和波斯语的元音符号、
tatweel 和字母变体、中文繁简、日语平假名与片假名。
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

ZWNJ = chr(0x200C)

# 删除的不可见字符：零宽空格、零宽连接符、词连接符、BOM、软连字符、方向控制符。
# 零宽不连字（ZWNJ）在波斯语里分隔词的部件，单独处理。
_INVISIBLE_CODES = [0x200B, 0x200D, 0x2060, 0xFEFF, 0x00AD, 0x200E, 0x200F, 0x061C]
_INVISIBLE_CODES += list(range(0x202A, 0x202F)) + list(range(0x2066, 0x206A))
_REMOVE_INVISIBLE = dict.fromkeys(_INVISIBLE_CODES, None)

# 各文字的十进制数字统一为 ASCII 数字（NFKC 已处理全角数字）。
_DIGIT_ZEROS = [
    0x0660, 0x06F0, 0x07C0, 0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6,
    0x0C66, 0x0CE6, 0x0D66, 0x0DE6, 0x0E50, 0x0ED0, 0x0F20, 0x1040, 0x1090,
    0x17E0, 0x1810, 0xFF10,
]
_DIGITS = {zero + k: str(k) for zero in _DIGIT_ZEROS for k in range(10)}
_DIGITS[0x066A] = "%"  # 阿拉伯百分号

# 阿拉伯文的元音符号、shadda、sukun、上标 alef、古兰经标注符号，以及 tatweel（U+0640）。
_AR_MARK_RANGES = [
    (0x0610, 0x061A), (0x064B, 0x065F), (0x0670, 0x0670), (0x06D6, 0x06DC),
    (0x06DF, 0x06E8), (0x06EA, 0x06ED), (0x0640, 0x0640),
]
_AR_MARKS = re.compile(pattern = "[" + "".join(f"{chr(a)}-{chr(b)}" for a, b in _AR_MARK_RANGES) + "]")
# 阿拉伯语：alef 变体 → ا，ى/ی/ئ → ي，ة/ۀ/ھ → ه，ؤ → و，ک → ك
_AR_FOLD = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي", "ی": "ي", "ئ": "ي",
    "ة": "ه", "ۀ": "ه", "ھ": "ه",
    "ؤ": "و",
    "ک": "ك",
})
# 波斯语：同样合并 alef 变体，ي/ى/ئ → ی，ك → ک
_FA_FOLD = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ي": "ی", "ى": "ی", "ئ": "ی",
    "ة": "ه", "ۀ": "ه", "ھ": "ه",
    "ؤ": "و",
    "ك": "ک",
})

_HIRAGANA_TO_KATAKANA = {c: c + 0x60 for c in range(0x3041, 0x3097)}
_HIRAGANA_TO_KATAKANA.update({0x309D: 0x30FD, 0x309E: 0x30FE})


@lru_cache(maxsize = 1)
def _zh_t2s() -> dict:
    """读取繁简单字对照表（来自 OpenCC，Apache-2.0）。"""
    table = {}
    path = Path(__file__).with_name(name = "data") / "zh_t2s.txt"
    for line in path.read_text(encoding = "utf-8").splitlines():
        if len(line) == 2 and not line.startswith("#"):
            table[ord(line[0])] = line[1]
    return table


def normalize_text(text: str, lang: str) -> str:
    """返回规范化后的文本。结果仍保留标点和空白，供后续切分使用。"""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_REMOVE_INVISIBLE)
    text = text.replace(ZWNJ, " " if lang == "fa" else "")
    text = text.translate(_DIGITS)
    if lang == "zh":
        text = text.translate(_zh_t2s())
    elif lang == "ja":
        text = text.translate(_HIRAGANA_TO_KATAKANA)
    elif lang == "ar":
        text = _AR_MARKS.sub(repl = "", string = text).translate(_AR_FOLD)
    elif lang == "fa":
        text = _AR_MARKS.sub(repl = "", string = text).translate(_FA_FOLD)
    elif lang == "tr":
        text = text.replace("I", "ı").replace("İ", "i")
    text = text.casefold()
    if lang == "ru":
        text = text.replace("ё", "е")
    # casefold 可能产生组合字符（例如 İ → i̇），再做一次 NFC。
    return unicodedata.normalize("NFC", text)
