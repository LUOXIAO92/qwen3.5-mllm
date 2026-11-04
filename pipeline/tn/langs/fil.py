"""菲律宾语：日常语言中他加禄语、西班牙语来源和英语三套数词随语境选用（年份、电话多用英语，时间、
金钱多用西班牙语来源词），没有可靠的规则和工具，含数字的条目全部丢弃。
"""

from __future__ import annotations

from ..rules import Skip
from .base import Lang


class Fil(Lang):
    code = "fil"

    def card(self, n: int) -> str:
        raise NotImplementedError("菲律宾语不改写数字")

    def cardinal(self, m, ctx, n):
        return Skip(reason = f"菲律宾语数词有他加禄语、西班牙语、英语三套读法，暂不处理「{m.group()}」")


LANG = Fil
