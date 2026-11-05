"""中文、日语、韩语共用的规则。

- 中点、全角句点连接的数字有两位以上的数或后面是事件名时（九一一事件写作 9·11，二・二六事件写作 2・26，
  三・一运动写作 3・1運動）习惯逐位读或按日期读，读法不固定，丢弃；并列的一位数（1・4番線）照常改写；
- 全角连字符的范围（20－30°C）与半角范围一样丢弃；
- 拉丁字母词后隔空格的数字多为型号、版本（Windows 10、Mk 36），可能按英语读，丢弃；
- 紧挨汉字数字的数字（安倍晋三14日、弗二15）改写后会和它连成一个数，丢弃；
- 全角冒号后紧跟数字时只是标点（两个客运大楼：1号客运大楼），保留冒号，让后面的数字照常改写；
  汉字、假名后的半角冒号接数字（总部:21）多为引文页码，丢弃；
- 中文、日语里数字两侧夹在汉字、假名之间的空格（于 1903 年）改写后删除；
- 百分号也接受全角的「％」。
"""

from __future__ import annotations

import re

from ..rules import DASHES, Rule, Skip

# 汉字（扩展 A、基本区、兼容区）、平假名、片假名（含长音符）的字符范围
HAN = f"{chr(0x3400)}-{chr(0x4DBF)}{chr(0x4E00)}-{chr(0x9FFF)}{chr(0xF900)}-{chr(0xFAFF)}"
HIRAGANA = f"{chr(0x3041)}-{chr(0x3096)}"
KATAKANA = f"{chr(0x30A1)}-{chr(0x30FA)}{chr(0x30FC)}"
SPACES = " " + chr(0x3000)
# 汉字、假名、々、〇，以及中文、日语的全角标点
CJK_TEXT = HAN + HIRAGANA + KATAKANA + chr(0x3005) + chr(0x3007)
CJK_PUNCT = "，。、；：！？（）「」『』《》〈〉【】“”‘’"
MIDDLE_DOTS = "・·‧･•．"
FULLWIDTH_DASHES = "－﹣"


# 中点连接的数字后面是这些词时是事件名（二・二六事件、三・一运动），读法按日期或逐位
EVENT_WORDS = (
    "事件", "事变", "事變", "事変", "運動", "运动", "テロ", "ゼネスト", "革命", "宣言", "地震", "震災", "政変", "政变",
    "暴動", "暴动", "闘争", "斗争", "蜂起", "起义", "惨案", "晚会", "協定", "协定", "決議", "决议", "クーデター",
)


def dotted_numbers(m, ctx):
    """中点连接的数字：并列的一位数（1・4番線、1・2年）照常改写；有两位以上的数（9·11、2・26）
    或后面是事件名（3・1運動）时，读法按日期或逐位，丢弃。"""
    numbers = re.findall(pattern = r"\d+", string = m.group())
    after = ctx.text[m.end():m.end() + 6]
    if any(len(n) > 1 for n in numbers[1:]) or after.startswith(EVENT_WORDS) or "．" in m.group():
        return Skip(reason = f"中点连接的数字读法不固定「{m.group()}」")
    return None


# 汉字数字。阿拉伯数字紧挨着它们时，改写后会连成一个数（安倍晋三14日 → 安倍晋三十四日，弗二15 → 弗二十五）
HAN_NUMERALS = "〇零一二三四五六七八九十两"


def guard_rules(num: str) -> list:
    """在各语言规则之前运行的保护规则。num 是该语言的数字写法正则。"""
    return [
        Rule(
            pattern = re.compile(pattern = rf"(?<=[{HAN_NUMERALS}]){num}|{num}(?=[{HAN_NUMERALS}])"),
            handle = lambda m, ctx: Skip(reason = f"数字紧挨汉字数字，改写后会连成一个数「{adjacent(m = m, ctx = ctx)}」"),
        ),
        Rule(pattern = re.compile(pattern = rf"{num}(?:\s?[{MIDDLE_DOTS}]\s?{num})+"), handle = dotted_numbers),
        Rule(
            pattern = re.compile(pattern = rf"{num}\s*[{FULLWIDTH_DASHES}]\s*{num}"),
            handle = lambda m, ctx: Skip(reason = f"范围或编号「{m.group()}」"),
        ),
        Rule(
            # 数字带上单位或量词再接连字符的范围（600만-700만、3万～5万、1990年-1995年）
            pattern = re.compile(pattern = rf"{num}[^\W\d_]{{1,3}}\s*[{re.escape(pattern = DASHES + FULLWIDTH_DASHES)}]\s*{num}"),
            handle = lambda m, ctx: Skip(reason = f"范围或编号「{m.group()}」"),
        ),
        Rule(
            # 全角负号「－6時間」；长音符号「ー」不跟在片假名后时也是误用的负号（更にー1時間）
            pattern = re.compile(pattern = rf"(?<![\w.,])[{FULLWIDTH_DASHES}]\s?{num}|(?<![{KATAKANA}])ー{num}"),
            handle = lambda m, ctx: Skip(reason = f"负号或连字符「{m.group()}」"),
        ),
        Rule(
            pattern = re.compile(pattern = rf"(?<![A-Za-z])[A-Za-z][A-Za-z.]*\s{num}"),
            handle = lambda m, ctx: Skip(reason = f"拉丁字母词后的编号或型号「{m.group()}」"),
        ),
        Rule(
            # 公共规则的型号检查要求前面不是文字，汉字、假名后紧跟的 AH-1W、F-15 在这里补上
            pattern = re.compile(pattern = rf"(?<![A-Za-z])[A-Za-z]+[-‐‑–]{num}"),
            handle = lambda m, ctx: Skip(reason = f"型号或编号「{m.group()}」"),
        ),
    ]


def adjacent(m, ctx) -> str:
    return ctx.text[max(0, m.start() - 1):m.end() + 1]


def colon_punct_rule() -> Rule:
    """全角冒号后紧跟数字、前面不是数字：冒号是标点，原样保留，占住位置以免被当作时刻或比分丢弃。"""
    return Rule(
        pattern = re.compile(pattern = r"(?<![\d:：])：(?=\d)"),
        handle = lambda m, ctx: "：",
    )


def citation_rule(num: str) -> Rule:
    """汉字、假名后的半角冒号接数字（总部:21、寿勉成: 8288）多为维基百科的引文页码，丢弃。"""
    return Rule(
        pattern = re.compile(pattern = rf"(?<=[{CJK_TEXT}{CJK_PUNCT}])\s?:\s*{num}"),
        handle = lambda m, ctx: Skip(reason = f"冒号后的数字可能是引文页码「{m.group().strip()}」"),
    )


def space_rules() -> list:
    """删除数字与汉字、假名、全角标点之间的空格（放在最后运行，只处理数字规则没有用到的空格）。"""
    side = f"[{CJK_TEXT}{CJK_PUNCT}]"
    return [
        Rule(pattern = re.compile(pattern = rf"(?<={side})[{SPACES}]+(?=\d)"), handle = lambda m, ctx: ""),
        Rule(pattern = re.compile(pattern = rf"(?<=\d)[{SPACES}]+(?={side})"), handle = lambda m, ctx: ""),
    ]


def fullwidth_percent(rules: list) -> list:
    """把基类百分数规则的百分号字符组扩展到全角「％」。"""
    return [
        Rule(
            pattern = re.compile(pattern = rule.pattern.pattern.replace("[%٪]", "[%٪％]")),
            handle = rule.handle,
        )
        for rule in rules
    ]


def guard_currency(rules: list, names: tuple) -> list:
    """包一层基类的货币规则：符号后的数字紧跟拉丁字母（€4M、$5bn）或后面又写了货币名（$324亿美金）时丢弃。"""

    def guard(handle):
        def wrapped(m, ctx):
            after = context_after(text = ctx.text, end = m.end(), width = 4)
            if ctx.text[m.end():m.end() + 1].isascii() and ctx.text[m.end():m.end() + 1].isalpha():
                return Skip(reason = f"货币金额后接字母「{m.group()}{ctx.text[m.end()]}」")
            if after.startswith(names):
                return Skip(reason = f"货币符号与货币名重复「{m.group()}{after[:2]}」")
            return handle(m = m, ctx = ctx)

        return wrapped

    return [Rule(pattern = rule.pattern, handle = guard(handle = rule.handle)) for rule in rules]


def context_before(text: str, start: int, width: int = 6) -> str:
    """前面紧邻的几个字符（去掉空格）。"""
    return text[max(0, start - width):start].replace(" ", "")


def context_after(text: str, end: int, width: int = 6) -> str:
    return text[end:end + width].lstrip(" ")
