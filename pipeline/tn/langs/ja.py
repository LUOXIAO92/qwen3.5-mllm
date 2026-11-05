"""日语：数字写成汉字数字（1950 → 千九百五十，4万1千 → 四万一千），助数词的读音（三本、六匹、一日）交给 TTS。

写法选汉字数字而不是假名：日语 ASR 输出阿拉伯数字或汉字数字（Haas-ja 写「1950年代」「4月1日」），
pipeline.compare 能把汉字数字和阿拉伯数字都换成数值比较，假名读法则无法比较。

汉字写法有歧义的情况丢弃：

- 「十分」「五分」可以读「じゅっぷん」也可以读「じゅうぶん」「ごぶ」。10 以内的「N分」只在前面是「時」「約」
  「徒歩」等、后面是「間」「後」「以内」「秒」「かかる」等时改写，「3勝1分」（引き分け）和其他情况丢弃；
  「3分の1」这类分数照常改写。
- 「97式」「747型」这类两位以上的型号常逐位读（きゅうななしき），丢弃。
- 片假名词后的数字后面不是汉字时（ベスト8、ゴルゴ13が、クハ3866は）多为型号或按英语读，丢弃；后面是汉字
  （エリザベス1世、ホーム2面、キハ47系）照常改写。拉丁字母词后的数字（Windows 10）丢弃。
- 不带量词的 1000–2099（『1984』）可能是年份也可能是书名，丢弃；括号里的（1950）按数读。
- 单独的 0（〇系、0点）有「ゼロ」「れい」两种读法，丢弃；「0時」读「〇時」，小数「0.5」读「〇点五」。
- 时刻（17:30 → 十七時三十分）只在前后有「から」「まで」「午後」「日」等时改写，经文章节「14:25」丢弃；
  「1835時」这种四位数时刻写法丢弃。
"""

from __future__ import annotations

import re

from ..numwords import ja_cardinal, ja_digits
from ..rules import Rule, Skip
from . import cjk
from .base import Lang

# 10 以内的「N分」按分钟读时前后常见的词；「3割5分」的「分」读「ぶ」，汉字写法同样不会读错
MINUTE_BEFORE = (
    "時", "約", "徒歩", "あと", "残り", "最大", "最短", "最長", "およそ", "わずか", "僅か", "毎", "数", "開始", "後半",
    "前半", "より", "から", "割",
)
MINUTE_AFTER = (
    "秒", "間", "後", "前", "以内", "以上", "未満", "程度", "ほど", "程", "ごと", "毎", "おき", "遅れ", "早く", "余り",
    "弱", "強", "足らず", "経過", "で", "かか", "を要", "しか", "ずつ", "半",
)
MINUTE_DRAW_BEFORE = ("勝", "敗")
# 时刻前后的词
TIME_BEFORE = (
    "午前", "午後", "朝", "昼", "夕方", "夜", "深夜", "未明", "毎日", "毎朝", "毎晩", "日", "曜", "曜日", "時刻",
    "時間", "当日", "翌日", "同日", "翌朝", "から", "より",
)
TIME_AFTER = (
    "まで", "以後", "以前", "以降", "から", "で", "に", "を目途", "頃", "ごろ", "過ぎ", "開始", "終了", "より",
    "台", "現在", "時点", "発", "着", "放送", "～", "〜",
)
# 两位以上的数后面是这些字时多为型号，常逐位读
MODEL_AFTER = ("式", "型")


class Ja(Lang):
    code = "ja"
    point = "点"
    spaced = False
    percent = "パーセント"
    units = {
        "km²": ("平方キロメートル", "平方キロメートル"), "m²": ("平方メートル", "平方メートル"),
        "cm²": ("平方センチメートル", "平方センチメートル"), "m³": ("立方メートル", "立方メートル"),
        "km": ("キロメートル", "キロメートル"), "cm": ("センチメートル", "センチメートル"),
        "mm": ("ミリメートル", "ミリメートル"), "nm": ("ナノメートル", "ナノメートル"),
        "μm": ("マイクロメートル", "マイクロメートル"), "µm": ("マイクロメートル", "マイクロメートル"),
        "kg": ("キログラム", "キログラム"), "mg": ("ミリグラム", "ミリグラム"), "m": ("メートル", "メートル"),
        "g": ("グラム", "グラム"), "t": ("トン", "トン"), "ml": ("ミリリットル", "ミリリットル"),
        "mL": ("ミリリットル", "ミリリットル"), "L": ("リットル", "リットル"), "ha": ("ヘクタール", "ヘクタール"),
        "kW": ("キロワット", "キロワット"), "MW": ("メガワット", "メガワット"), "W": ("ワット", "ワット"),
        "kV": ("キロボルト", "キロボルト"), "V": ("ボルト", "ボルト"), "Hz": ("ヘルツ", "ヘルツ"),
        "kHz": ("キロヘルツ", "キロヘルツ"), "MHz": ("メガヘルツ", "メガヘルツ"), "GHz": ("ギガヘルツ", "ギガヘルツ"),
        "mi": ("マイル", "マイル"), "ft": ("フィート", "フィート"), "yd": ("ヤード", "ヤード"),
        "lb": ("ポンド", "ポンド"), "kcal": ("キロカロリー", "キロカロリー"), "KB": ("キロバイト", "キロバイト"),
        "kB": ("キロバイト", "キロバイト"), "MB": ("メガバイト", "メガバイト"), "GB": ("ギガバイト", "ギガバイト"),
        "TB": ("テラバイト", "テラバイト"), "ms": ("ミリ秒", "ミリ秒"), "°C": ("度", "度"), "℃": ("度", "度"),
        "°": ("度", "度"),
    }
    currencies = {"$": ("ドル", "ドル"), "€": ("ユーロ", "ユーロ"), "£": ("ポンド", "ポンド"), "¥": ("円", "円"), "₩": ("ウォン", "ウォン")}
    scales = ("万", "億")
    # km/h 读作「時速…キロメートル」，语序要变，没有登记

    def card(self, n: int) -> str:
        return ja_cardinal(n = n)

    def digits(self, digits: str) -> str:
        return ja_digits(digits = digits)

    def rules(self) -> list:
        return super().rules() + cjk.space_rules()

    def percent_rules(self) -> list:
        return cjk.fullwidth_percent(rules = super().percent_rules())

    def currency_rules(self) -> list:
        return cjk.guard_currency(rules = super().currency_rules(), names = ("ドル", "円", "ユーロ", "ポンド", "ウォン"))

    def extra_rules(self) -> list:
        num = self.num
        return cjk.guard_rules(num = num) + [
            Rule(
                pattern = re.compile(pattern = rf"(?<=[{cjk.KATAKANA}])\s?{self.num_bare}(?![{cjk.HAN}])"),
                handle = lambda m, ctx: Skip(reason = f"片假名词后的数字可能是型号或按外语读「{ctx.text[m.start() - 1:m.end()]}」"),
            ),
            cjk.colon_punct_rule(),
            cjk.citation_rule(num = num),
            Rule(pattern = re.compile(pattern = r"(?<!\d)(?P<h>\d{1,2})[:：](?P<mm>\d{2})(?![\d:：])"), handle = self.time),
            Rule(pattern = re.compile(pattern = rf"(?P<num>{num})(?=分)"), handle = self.minutes),
            Rule(
                pattern = re.compile(pattern = r"(?<!\d)\d{3,4}(?=時)"),
                handle = lambda m, ctx: Skip(reason = f"「{m.group()}時」是军用时刻写法，读法不固定"),
            ),
            Rule(pattern = re.compile(pattern = rf"(?P<num>{num})(?=[{''.join(MODEL_AFTER)}])"), handle = self.model),
        ]

    def time(self, m, ctx):
        h, mm = int(m.group("h")), int(m.group("mm"))
        before = cjk.context_before(text = ctx.text, start = m.start())
        after = cjk.context_after(text = ctx.text, end = m.end())
        if h > 24 or mm > 59 or not (before.endswith(TIME_BEFORE) or after.startswith(TIME_AFTER)):
            return Skip(reason = f"冒号连接的数字前后没有时间词，可能是比分或章节「{m.group()}」")
        ctx.kind(name = "时刻", surface = m.group())
        hour = ctx.num(spoken = self.card(n = h), written = m.group("h"))
        if mm == 0:
            return hour + "時"
        return hour + "時" + ctx.num(spoken = self.card(n = mm), written = m.group("mm")) + "分"

    def minutes(self, m, ctx):
        """10 以内的「N分」：汉字写法可能读成「じゅうぶん」「ごぶ」，只在上下文确定是分钟或分数时改写。"""
        n = self.parse(written = m.group("num"))
        if n is None or not n.is_int or n.value > 10:
            return None
        text = ctx.text
        after = text[m.end() + 1:]
        before = cjk.context_before(text = text, start = m.start())
        if re.match(pattern = r"の\s?\d", string = after):
            return None
        if before.endswith(MINUTE_DRAW_BEFORE) or not (before.endswith(MINUTE_BEFORE) or after.startswith(MINUTE_AFTER)):
            return Skip(reason = f"「{m.group()}分」写成汉字后读音有歧义（ふん、ぶん、ぶ）")
        return None

    def model(self, m, ctx):
        n = self.parse(written = m.group("num"))
        if n is not None and n.is_int and n.value >= 10:
            return Skip(reason = f"型号可能逐位读「{m.group()}{ctx.text[m.end()]}」")
        return None

    def cardinal(self, m, ctx, n):
        long_out = self.long_number(m = m, ctx = ctx, n = n)
        if long_out is not None:
            return long_out
        text = ctx.text
        if n.is_int and n.value == 0 and not text[m.end():].startswith("時"):
            return Skip(reason = "单独的 0 有「ゼロ」「れい」两种读法")
        if n.is_int and not n.grouped and 1000 <= n.value <= 2099:
            after = text[m.end():]
            bracketed = text[:m.start()].endswith(("（", "(")) and after.startswith(("）", ")"))
            if not bracketed and not (after[:1].isalpha() and not after[:1].isascii()):
                return Skip(reason = f"不带量词的四位数可能是年份或书名「{m.group()}」")
        return self.say(n = n, ctx = ctx)


LANG = Ja
