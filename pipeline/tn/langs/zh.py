"""中文：年份逐位读，其余按数值读；单独的「2」按后面的量词选「两」或「二」。"""

from __future__ import annotations

import re

from ..numwords import zh_cardinal, zh_digits
from ..rules import Rule, Skip
from .base import Lang

# 「2」读「两」的量词和单位（按前缀匹配后文）
LIANG_AFTER = (
    "个", "位", "名", "次", "种", "倍", "条", "只", "本", "张", "件", "天", "小时", "分钟", "秒钟", "周", "岁",
    "人", "家", "辆", "座", "架", "艘", "部", "台", "项", "支", "队", "所", "间", "口", "头", "匹", "首", "篇",
    "句", "份", "组", "场", "局", "届", "斤", "公斤", "公里", "吨", "元", "块", "角", "毛", "遍", "趟", "声",
    "下", "回", "批", "双", "对", "套", "片", "颗", "粒", "根", "棵", "枚", "封", "节", "段", "点", "百", "千",
    "万", "亿", "米", "克", "升", "名", "分",
)
# 三位以上的数后面紧跟其他汉字时多为编号（911事件、110号舰、619号道路），只在后面是这些量词、单位时按数值读
MEASURE_AFTER = tuple(w for w in LIANG_AFTER if w != "号") + (
    "多", "余", "来", "年", "月", "日", "岁", "度", "米", "千米", "公顷", "平方", "立方", "亩", "户", "册", "页",
    "字", "票", "股", "辆", "条", "卷", "集", "所", "座", "名", "位", "瓦", "千瓦", "伏", "赫", "兆", "美元", "英镑",
    "欧元", "日元", "港元", "澳门元", "新台币", "马力", "英里", "海里", "英尺", "磅", "吨", "升", "毫升", "人次",
)
# 「2」读「二」：日期、编号、序数
ER_AFTER = ("月", "日", "号")
# 后面是这些词时，整百、整千的「N年」可能表示年数
SPAN_AFTER = ("前", "后", "間", "间", "多", "来", "之久", "历史", "左右", "以上")
# 这些词后面的数字是编号，可能逐位读
CODE_BEFORE = ("编号", "编码", "代码", "电话", "号码", "邮编", "型号", "学号", "账号", "序号", "车次", "航班")


class Zh(Lang):
    code = "zh"
    point = "点"
    spaced = False
    percent = "百分之"
    percent_word_first = True
    units = {
        "km/h": ("公里每小时", "公里每小时"), "km²": ("平方公里", "平方公里"), "m²": ("平方米", "平方米"),
        "km": ("公里", "公里"), "cm": ("厘米", "厘米"), "mm": ("毫米", "毫米"), "kg": ("公斤", "公斤"),
        "m": ("米", "米"), "g": ("克", "克"), "°C": ("摄氏度", "摄氏度"), "℃": ("摄氏度", "摄氏度"),
        "°F": ("华氏度", "华氏度"), "℉": ("华氏度", "华氏度"),
    }
    currencies = {"$": ("美元", "美元"), "€": ("欧元", "欧元"), "£": ("英镑", "英镑")}
    scales = ("万", "亿")

    def card(self, n: int) -> str:
        return zh_cardinal(n = n)

    def digits(self, digits: str) -> str:
        return zh_digits(digits = digits)

    def liang(self, n, ctx):
        """量词、单位前的 2 读「两」。"""
        if n.is_int and n.value == 2:
            return ctx.num(spoken = "两", written = n.written)
        return self.say(n = n, ctx = ctx)

    def unit_number(self, m, ctx, n):
        return self.liang(n = n, ctx = ctx)

    def currency_number(self, m, ctx, n):
        return self.liang(n = n, ctx = ctx)

    def extra_rules(self) -> list:
        return [
            Rule(
                pattern = re.compile(pattern = rf"(?:{'|'.join(CODE_BEFORE)})[:：]?\s?{self.num}"),
                handle = lambda m, ctx: Skip(reason = f"编号逐位读还是按数值读不确定「{m.group()}」"),
            ),
            Rule(pattern = re.compile(pattern = rf"(?P<num>{self.num})(?P<sp>\s?)(?=年)"), handle = self.year),
            Rule(pattern = re.compile(pattern = r"(?P<h>\d{1,2})[:：](?P<mm>\d{2})(?!\d)"), handle = self.time),
        ]

    def year(self, m, ctx):
        n = self.parse(written = m.group("num"))
        if n is None or not n.is_int or n.grouped:
            return Skip(reason = f"年份写法无法确定「{m.group()}」")
        if len(n.int_digits) == 4:
            text = ctx.text
            after = text[m.end() + 1:]
            # 公元前 3000 年、2000 年前：可能是年数（三千年），也可能是年份（二零零零年）
            if text[:m.start()].rstrip().endswith("前") or n.value % 1000 == 0 or (
                n.value % 100 == 0 and after.startswith(SPAN_AFTER)
            ):
                return Skip(reason = f"四位数是年份还是年数需要上下文「{m.group()}年」")
            ctx.kind(name = "年份", surface = m.group("num"))
            return ctx.num(spoken = self.digits(digits = n.int_digits), written = n.written) + m.group("sp")
        if n.value == 2:
            return Skip(reason = "「2年」读「两年」还是「二年」需要上下文")
        ctx.kind(name = "年数", surface = m.group("num"))
        return self.say(n = n, ctx = ctx) + m.group("sp")

    def time(self, m, ctx):
        h, mm = int(m.group("h")), int(m.group("mm"))
        if h > 24 or mm > 59:
            return Skip(reason = f"时刻写法无法确定「{m.group()}」")
        hour = "两" if h == 2 else self.card(n = h)
        ctx.num(spoken = hour, written = m.group("h"))
        ctx.kind(name = "时刻", surface = m.group())
        if mm == 0:
            return hour + "点"
        minute = ("零" if mm < 10 else "") + self.card(n = mm)
        ctx.num(spoken = minute, written = m.group("mm"))
        return hour + "点" + minute + "分"

    def cardinal(self, m, ctx, n):
        long_out = self.long_number(m = m, ctx = ctx, n = n)
        if long_out is not None:
            return long_out
        after = ctx.text[m.end():]
        if n.is_int and not n.grouped and 1000 <= n.value <= 2099 and not after.lstrip().startswith(MEASURE_AFTER):
            # 《我的1919》：不带「年」的四位数可能是年份（逐位读）也可能是数量
            return Skip(reason = f"四位数是年份还是数量需要上下文「{m.group()}」")
        if n.is_int and len(n.int_digits) >= 3 and after[:1].isalpha() and not after.startswith(MEASURE_AFTER):
            return Skip(reason = f"数字后面不是量词，可能是编号「{m.group()}{after[:2]}」")
        if n.is_int and n.value == 2:
            text, start, end = ctx.text, m.start(), m.end()
            after = text[end:]
            if text[:start].endswith("第") or after.startswith(ER_AFTER):
                return ctx.num(spoken = "二", written = n.written)
            if after.startswith(LIANG_AFTER):
                return ctx.num(spoken = "两", written = n.written)
            return Skip(reason = "「2」读「两」还是「二」需要上下文")
        return self.say(n = n, ctx = ctx)


LANG = Zh
