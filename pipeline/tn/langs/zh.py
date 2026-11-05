"""中文：年份逐位读，其余按数值读；单独的「2」按后面的量词选「两」或「二」。

- 年份：「1937年」读「一九三七年」；「1982至1983年」这样共用一个「年」的年份，以及「光绪三十一年（1905）」
  括号里紧跟在「年」后的四位数也按年份读。整千的「2000年」按年份读，「3000年」「公元前8000年」可能是年数，丢弃。
- 两/二：「2个」「2人」读「两」，「2月」「2号」「第2」「2楼」读「二」，「2年后」读「两年后」，其余丢弃。
- 时刻：「下午14:30」读「十四点三十分」，只在前后有时间词（凌晨、上午、日、时间、开始、举行等）时改写，
  「以21:19获胜」「马可6:18」这类比分和经文章节丢弃。
- 百分数（含全角「％」）、小数、单位、货币（$、US$、HK$ 等）；货币符号后面又写了「美元」「美金」「元」的丢弃。
- 编号：编号、代号、电话、国道等词后面的数字，三位以上后面不跟量词的数字（911事件、110号舰），不带「年」
  又不跟量词的 1000–2099（《我的1919》），读法不确定，丢弃。
- 数字两侧夹在汉字之间的空格（于 1903 年）删除。
"""

from __future__ import annotations

import re

from ..numwords import zh_cardinal, zh_digits
from ..rules import Rule, Skip
from . import cjk
from .base import Lang

# 「2」读「两」的量词和单位（按前缀匹配后文）
LIANG_AFTER = (
    "个", "位", "名", "次", "种", "倍", "条", "只", "本", "张", "件", "天", "小时", "分钟", "秒钟", "周", "岁",
    "人", "家", "辆", "座", "架", "艘", "部", "台", "项", "支", "队", "所", "间", "口", "头", "匹", "首", "篇",
    "句", "份", "组", "场", "局", "届", "斤", "公斤", "公里", "吨", "元", "块", "角", "毛", "遍", "趟", "声",
    "下", "回", "批", "双", "对", "套", "片", "颗", "粒", "根", "棵", "枚", "封", "节", "段", "点", "百", "千",
    "万", "亿", "米", "克", "升", "分", "时", "县", "省", "国", "州", "镇", "村", "处", "室", "地", "城", "岸",
    "侧", "端", "方", "面", "派", "党", "票", "股", "成", "款", "栋", "幢", "户", "轮", "任", "周年", "种类",
    "杯", "碗", "瓶", "盒", "包", "袋", "箱", "桶", "滴", "群", "伙", "副", "扇", "列", "发", "道", "门", "样",
    "类", "千米", "厘米", "毫米", "千克", "公顷", "亩", "英里", "英尺", "海里", "美元", "欧元", "英镑", "日元",
)
# 「2」读「二」：日期、编号、楼层
ER_AFTER = ("月", "日", "号", "楼")
# 「2年」后面是这些字时读「两年」
LIANG_YEAR_AFTER = ("后", "前", "间", "内", "多", "来", "半", "时间", "以", "之", "左右", "里")
# 三位以上的数后面紧跟这些量词、单位、货币时按数值读，紧跟其他汉字时多为编号（911事件、110号舰）
MEASURE_AFTER = tuple(w for w in LIANG_AFTER if w not in ("号", "楼")) + (
    "多", "余", "来", "几", "年", "月", "日", "度", "平方", "立方", "册", "页", "字", "卷", "集", "期", "层",
    "人次", "瓦", "千瓦", "兆瓦", "伏", "赫", "兆", "港元", "港币", "澳门元", "新台币", "台币", "人民币", "卢布",
    "法郎", "马克", "里拉", "比索", "卢比", "韩元", "泰铢", "澳元", "加元", "新元", "美金", "公尺", "公分", "英寸",
    "码", "磅", "盎司", "加仑", "坪", "纳米", "微米", "毫克", "毫升", "马力", "千卡", "卡", "光年", "世纪",
    "年代", "分", "秒", "至", "到", "左右", "以上", "以下", "以内", "上下", "强", "弱",
)
# 不带「年」的 1000–2099 只在后面是真正的量词时按数值读；后面是「至」「左右」等时可能是年份
RANGE_WORDS = ("至", "到", "左右", "以上", "以下", "以内", "上下", "强", "弱")
FOUR_DIGIT_AFTER = tuple(w for w in MEASURE_AFTER if w not in RANGE_WORDS)
# 以量词开头、其实是另一个词的写法（国窖1573所属、2000家族）：后面是这些词时不算量词
NOT_MEASURE = (
    "所属", "所有", "所以", "所在", "所谓", "所示", "所述", "所得", "所需", "所用", "所提", "家族", "家庭", "人民",
    "名称", "名字", "名单", "台湾", "台北", "口号", "头条", "首都", "首次", "首届", "部分", "部门", "部长", "项目",
    "条件", "条例", "条约", "场所", "种类", "号称", "下来", "下去", "对于", "对此", "回到", "回来", "间隔", "支持",
)
# 后面是这些词时，整百的「N年」可能表示年数
SPAN_AFTER = ("前", "后", "間", "间", "多", "来", "之久", "历史", "左右", "以上")
# 前面是这些词时，整百的「1500年」「2000年」可能表示年数
SPAN_BEFORE = ("约", "近", "超过", "长达", "历时", "历经", "经过", "达", "有", "共", "前", "逾", "数", "几")
# 这些词后面的数字是编号，可能逐位读
CODE_BEFORE = (
    "编号", "编码", "代码", "代号", "番号", "电话", "号码", "邮编", "型号", "学号", "账号", "序号", "车次", "航班",
    "拨打", "热线", "频道", "房号", "门牌",
)
# 时刻前后的时间词；比分前的词
TIME_BEFORE = (
    "凌晨", "早上", "早晨", "清晨", "上午", "中午", "午间", "下午", "傍晚", "晚上", "晚间", "夜间", "夜里", "深夜",
    "午夜", "半夜", "每天", "每日", "当天", "当日", "次日", "翌日", "时间", "日", "号", "于", "在", "从", "自",
    "至", "到", "约", "晚", "早", "起",
)
TIME_AFTER = (
    "时", "前", "后", "左右", "开始", "起", "至", "到", "许", "结束", "举行", "之间", "首播", "播出", "播放",
    "发车", "出发", "抵达", "到达", "开幕", "闭幕", "截止", "为止", "召开", "进行", "正式", "停止",
)
SCORE_BEFORE = ("以", "比分", "比数", "比", "战胜", "击败", "打败", "败给", "输给", "赢", "胜", "负", "平")
YEAR_SEPS = "至到或、和及与"
# 年号后的「2年」读「二年」（民国二年、昭和二年）
REIGN_TITLES = (
    "民国", "光绪", "宣统", "同治", "咸丰", "道光", "嘉庆", "乾隆", "雍正", "康熙", "顺治", "崇祯", "天启", "万历",
    "隆庆", "嘉靖", "正德", "弘治", "成化", "永乐", "洪武", "至元", "至正", "开元", "天宝", "贞观", "建安", "建武",
    "明治", "大正", "昭和", "平成", "令和", "庆长", "宽永", "元禄", "享保", "天保", "安政", "文久", "庆应",
)


class Zh(Lang):
    code = "zh"
    point = "点"
    spaced = False
    percent = "百分之"
    percent_word_first = True
    units = {
        "km/h": ("公里每小时", "公里每小时"), "km²": ("平方公里", "平方公里"), "m²": ("平方米", "平方米"),
        "cm²": ("平方厘米", "平方厘米"), "m³": ("立方米", "立方米"), "km": ("公里", "公里"), "cm": ("厘米", "厘米"),
        "mm": ("毫米", "毫米"), "nm": ("纳米", "纳米"), "μm": ("微米", "微米"), "µm": ("微米", "微米"),
        "kg": ("公斤", "公斤"), "mg": ("毫克", "毫克"), "m": ("米", "米"), "g": ("克", "克"), "t": ("吨", "吨"),
        "ml": ("毫升", "毫升"), "mL": ("毫升", "毫升"), "L": ("升", "升"), "ha": ("公顷", "公顷"),
        "kW": ("千瓦", "千瓦"), "MW": ("兆瓦", "兆瓦"), "W": ("瓦", "瓦"), "kV": ("千伏", "千伏"), "V": ("伏", "伏"),
        "Hz": ("赫兹", "赫兹"), "kHz": ("千赫", "千赫"), "MHz": ("兆赫", "兆赫"), "mi": ("英里", "英里"),
        "ft": ("英尺", "英尺"), "yd": ("码", "码"), "lb": ("磅", "磅"), "lbs": ("磅", "磅"), "oz": ("盎司", "盎司"),
        "kcal": ("千卡", "千卡"), "ms": ("毫秒", "毫秒"), "°C": ("摄氏度", "摄氏度"), "℃": ("摄氏度", "摄氏度"),
        "°F": ("华氏度", "华氏度"), "℉": ("华氏度", "华氏度"), "°": ("度", "度"),
    }
    currencies = {
        "US$": ("美元", "美元"), "HK$": ("港元", "港元"), "NT$": ("新台币", "新台币"), "A$": ("澳元", "澳元"),
        "C$": ("加元", "加元"), "S$": ("新加坡元", "新加坡元"), "$": ("美元", "美元"), "€": ("欧元", "欧元"),
        "£": ("英镑", "英镑"),
    }
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

    # ------------------------------------------------------------------
    # 规则
    # ------------------------------------------------------------------

    def rules(self) -> list:
        return super().rules() + cjk.space_rules()

    def percent_rules(self) -> list:
        return cjk.fullwidth_percent(rules = super().percent_rules())

    def currency_rules(self) -> list:
        return cjk.guard_currency(rules = super().currency_rules(), names = ("美金", "美元", "元", "圆", "币", "镑", "欧元"))

    def extra_rules(self) -> list:
        num = self.num
        seps = re.escape(pattern = YEAR_SEPS)
        return cjk.guard_rules(num = num) + [
            cjk.colon_punct_rule(),
            cjk.citation_rule(num = num),
            Rule(
                pattern = re.compile(pattern = rf"(?<=[《〈『「“]){num}"),
                handle = self.title_start,
            ),
            Rule(
                pattern = re.compile(pattern = rf"{num}成{num}"),
                handle = lambda m, ctx: Skip(reason = f"打击率写法读法不固定「{m.group()}」"),
            ),
            Rule(
                pattern = re.compile(pattern = rf"(?:{'|'.join(CODE_BEFORE)})[:：]?\s?{num}"),
                handle = lambda m, ctx: Skip(reason = f"编号逐位读还是按数值读不确定「{m.group()}」"),
            ),
            Rule(
                pattern = re.compile(pattern = rf"(?<!\d)\d{{4}}(?:\s?[{seps}]\s?\d{{4}})+(?P<sp>\s?)(?=年)"),
                handle = self.year_list,
            ),
            Rule(pattern = re.compile(pattern = r"(?<=年[（(])\d{4}(?=[）)])"), handle = self.year_in_brackets),
            Rule(pattern = re.compile(pattern = rf"(?P<num>{num})(?P<sp>\s?)(?=年)"), handle = self.year),
            Rule(pattern = re.compile(pattern = r"(?<!\d)(?P<h>\d{1,2})[:：](?P<mm>\d{2})(?![\d:：])"), handle = self.time),
        ]

    def year_digits(self, written: str, ctx) -> str:
        ctx.kind(name = "年份", surface = written)
        return ctx.num(spoken = self.digits(digits = written), written = written)

    def year_list(self, m, ctx):
        """「1982至1983年」：共用一个「年」的几个年份都逐位读。"""
        years = re.findall(pattern = r"\d{4}", string = m.group())
        if any(not 1000 <= int(y) <= 2099 for y in years):
            return Skip(reason = f"年份写法无法确定「{m.group()}年」")
        out = m.group().replace(" ", "")
        for y in years:
            out = out.replace(y, self.year_digits(written = y, ctx = ctx), 1)
        return out

    def year_in_brackets(self, m, ctx):
        """「光绪三十一年（1905）」：紧跟在「年」后括号里的四位数是公元年份。"""
        if not 1000 <= int(m.group()) <= 2099:
            return Skip(reason = f"括号里的四位数读法不确定「{m.group()}」")
        return self.year_digits(written = m.group(), ctx = ctx)

    def year(self, m, ctx):
        n = self.parse(written = m.group("num"))
        if n is None or not n.is_int or n.grouped:
            return Skip(reason = f"年份写法无法确定「{m.group()}」")
        text = ctx.text
        if len(n.int_digits) == 4:
            after = text[m.end() + 1:]
            before = text[:m.start()].rstrip()
            ambiguous = (
                before.endswith("前")
                or (n.value % 1000 == 0 and n.value != 2000)
                or (n.value % 100 == 0 and (after.startswith(SPAN_AFTER) or before.endswith(SPAN_BEFORE)))
                or after.lstrip("的").startswith(("历史", "之久"))
            )
            # 公元前 3000 年、2000 年前、约 1500 年、1500 年的历史：可能是年数（一千五百年），也可能是年份（一五零零年）
            if ambiguous:
                return Skip(reason = f"四位数是年份还是年数需要上下文「{m.group()}年」")
            return self.year_digits(written = n.int_digits, ctx = ctx)
        if re.search(pattern = r"(?<!\d)\d{2}\s?[、，,和及或至到~]\s?$", string = text[:m.start()]):
            # 「89、90年间」：并列的两位数可能是年份简写（八九、九零年）
            return Skip(reason = f"并列的两位数年份可能是简写「{m.group()}年」")
        if n.value == 2:
            after = text[m.end() + 1:]
            if text[:m.start()].rstrip().endswith(("第",) + REIGN_TITLES) or after.startswith("级"):
                spoken = "二"
            elif after.startswith(LIANG_YEAR_AFTER):
                spoken = "两"
            else:
                return Skip(reason = "「2年」读「两年」还是「二年」需要上下文")
            ctx.kind(name = "年数", surface = m.group("num"))
            return ctx.num(spoken = spoken, written = n.written)
        ctx.kind(name = "年数", surface = m.group("num"))
        return self.say(n = n, ctx = ctx)

    def title_start(self, m, ctx):
        """「《97古惑仔》」：作品名开头的数字后面不是量词时，读法可能是简写的年份或逐位读。"""
        after = ctx.text[m.end():]
        if after.startswith(MEASURE_AFTER) or after.startswith(ER_AFTER):
            return None
        return Skip(reason = f"作品名开头的数字读法不确定「{m.group()}{after[:2]}」")

    def time(self, m, ctx):
        h, mm = int(m.group("h")), int(m.group("mm"))
        before = cjk.context_before(text = ctx.text, start = m.start())
        after = cjk.context_after(text = ctx.text, end = m.end())
        if h > 24 or mm > 59 or before.endswith(SCORE_BEFORE) or after.startswith(("的比分", "比", "获胜", "战胜")):
            return Skip(reason = f"冒号连接的数字是时刻还是比分需要上下文「{m.group()}」")
        if not (before.endswith(TIME_BEFORE) or after.startswith(TIME_AFTER)):
            return Skip(reason = f"冒号连接的数字前后没有时间词，可能是比分或章节「{m.group()}」")
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
        text, start, end = ctx.text, m.start(), m.end()
        after = text[end:]
        ordinal = text[:start].endswith("第")
        if n.is_int and len(n.int_digits) >= 3 and after.lstrip().startswith(NOT_MEASURE):
            return Skip(reason = f"数字后面不是量词，可能是编号「{m.group()}{after.lstrip()[:2]}」")
        if n.is_int and not n.grouped and 1000 <= n.value <= 2099 and not ordinal:
            if not after.lstrip().startswith(FOUR_DIGIT_AFTER):
                # 《我的1919》：不带「年」的四位数可能是年份（逐位读）也可能是数量
                return Skip(reason = f"四位数是年份还是数量需要上下文「{m.group()}」")
        if n.is_int and len(n.int_digits) >= 3 and after[:1].isalpha() and not after.startswith(MEASURE_AFTER):
            if not ordinal or after.startswith("号"):
                return Skip(reason = f"数字后面不是量词，可能是编号「{m.group()}{after[:2]}」")
        if n.is_int and n.value == 2:
            return self.two(m = m, ctx = ctx, n = n)
        return self.say(n = n, ctx = ctx)

    def two(self, m, ctx, n):
        """单独的 2：序数、日期、编号读「二」，量词前读「两」，没有把握时丢弃。"""
        text, start, end = ctx.text, m.start(), m.end()
        before, after = text[:start], text[end:]
        if re.search(pattern = r"\d\s?[、，,和及或至到~]\s?$", string = before):
            # 「1、2个」「第1、2名」：前面并列的数决定读法
            return Skip(reason = "并列的「2」读「两」还是「二」需要上下文")
        if before.endswith("第") or after.startswith(ER_AFTER):
            return ctx.num(spoken = "二", written = n.written)
        if after.startswith(LIANG_AFTER):
            return ctx.num(spoken = "两", written = n.written)
        if after[:1] in ("》", "」", "』", "”", "）", ")") or after.startswith("之"):
            # 《古惑仔2之猛龙过江》《东京巴士指南2》：作品的续集编号
            return ctx.num(spoken = "二", written = n.written)
        return Skip(reason = "「2」读「两」还是「二」需要上下文")


LANG = Zh
