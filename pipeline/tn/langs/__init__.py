"""各语言的改写规则。每个模块定义 Lang 的一个子类 LANG。"""

from __future__ import annotations

import importlib
from functools import lru_cache


@lru_cache(maxsize = None)
def get_lang(lang: str):
    module = importlib.import_module(name = f".{lang}", package = __name__)
    return module.LANG()


@lru_cache(maxsize = None)
def get_rules(lang: str) -> list:
    return get_lang(lang = lang).rules()
