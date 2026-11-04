"""阿拉伯语：数词与被数名词的性、数、格一致尚未实现，只改写能确定格的年份，其余丢弃。

年份出现在 عام、العام、سنة、السنة 之后（3–4 位），或 في、منذ 之后（四位数 1000–2099）时为属格，
读作 ألفين وخمسة وعشرين 这样的形式。后面紧跟月份名（في 12 أغسطس）的数是日期，丢弃。
"""

from __future__ import annotations

import re

from ..numwords import ar_year_genitive
from ..rules import Rule, Skip
from .base import Lang

ARABIC_LETTERS = f"{chr(0x0620)}-{chr(0x064A)}{chr(0x066E)}-{chr(0x06D3)}{chr(0x06FA)}-{chr(0x06FF)}"


class Ar(Lang):
    code = "ar"

    def card(self, n: int) -> str:
        raise NotImplementedError("阿拉伯语只实现了年份读法")

    def extra_rules(self) -> list:
        return [
            Rule(
                pattern = re.compile(
                    pattern = rf"(?<![{ARABIC_LETTERS}])(?P<prev>عام|العام|سنة|السنة|في|منذ)\s(?P<num>\d{{3,4}})(?!\d|[.,٫٬]\d)",
                ),
                handle = self.year,
            ),
        ]

    def year(self, m, ctx):
        prev, n = m.group("prev"), int(m.group("num"))
        if prev in ("في", "منذ") and not 1000 <= n <= 2099:
            return None
        if not 100 <= n <= 2999 or m.group("num").startswith("0"):
            return None
        ctx.kind(name = "年份", surface = m.group())
        return prev + " " + ctx.num(spoken = ar_year_genitive(n = n), written = m.group("num"))

    def cardinal(self, m, ctx, n):
        return Skip(reason = f"阿拉伯语数词与名词的性、数、格一致暂不处理「{m.group()}」")


LANG = Ar
