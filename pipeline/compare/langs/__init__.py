"""各语言的数词词表。每个模块提供 build()，返回该语言的 LangSpec。"""

from __future__ import annotations

import importlib
from functools import lru_cache

LANGS = ("zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it", "id", "ms", "th", "vi", "fil", "ar", "tr", "fa")


@lru_cache(maxsize = None)
def get_spec(lang: str):
    if lang not in LANGS:
        raise ValueError(f"不支持的语言代码：{lang!r}，可用：{' '.join(LANGS)}")
    from ..units import build_extras

    module = importlib.import_module(name = f".{lang}", package = __name__)
    spec = module.build()
    spec.ext = build_extras(spec = spec)
    return spec
