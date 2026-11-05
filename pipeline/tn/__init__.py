"""朗读稿生成（实验 v1，任务 T18）：把原文改写成交给 TTS 的朗读稿。

用法
====

    python -m pipeline.tn.run --input entries.jsonl --output scripts.jsonl [--mode basic|passthrough]
    python -m pipeline.tn.review --scripts scripts.jsonl --output review.md

    from pipeline.tn import normalize_entry
    normalize_entry(original = "並於1937年徹底禁止", lang = "zh")
    # TnResult(script = "并于一九三七年彻底禁止", status = "normalized", note = "改写：年份「1937」")

输入是 pipeline/CONVENTIONS.md 的文本条目，输出在每条后补上三个字段：script（朗读稿，丢弃时为 null）、
tn_status（unchanged、normalized、dropped）和 tn_note（改写或丢弃的原因）。review 生成审阅报告：
按语言统计三种状态，用 pipeline.compare 比对改写过的条目与原文，并逐条列出原文与朗读稿。

处理流程
========

1. 清理原文（core.clean）：NFC，删除零宽空格、软连字符和方向控制符，空白合并为一个空格；中文用 OpenCC
   按词组转为简体（先 hk2t 统一香港异体字，再 tw2s）：「乾隆」「著名」保持原样，「乾淨」「看著」
   改为「干净」「看着」。
2. 找出需要改写的片段（detect.find_issues）：各文字的数字、上标和分数，货币、数学和其他符号（%、°、&、º 等），
   带句点的缩写（e.g.、т. д.、Dr.、No. 47、泰语 พ.ศ.、阿拉伯语 هـ），罗马数字，网址。没有这些片段的条目，
   朗读稿就是清理后的原文，标 unchanged。
3. passthrough（直通版）把其余条目全部丢弃。basic（基础读法，默认）按语言规则改写数字（rules.apply_rules、
   langs/）：先运行公共的保护规则，把范围（3-5）、日期写法（03.11.2026）、分数、负号、撇号后缀（5'te）、
   连续的数字（52 3048）、型号（C-47、WIN350）等没有把握的写法标为跳过；再运行各语言的年份、日期、序数、
   时刻、量词规则，以及百分数、单位、货币规则；最后决定普通数字的读法。任何一处跳过、改写后仍有数字或
   缩写，整条丢弃，tn_note 写明原因。
4. 每处读法与原写法用 pipeline.compare 比较数值（例如「一九三七」对「1937」），不一致的整条丢弃，用来
   拦住读法函数自身的错误。全部通过才标 normalized，tn_note 列出改写的类别和原文写法。

各语言的覆盖范围
================

- zh、ja、ko 共用 langs/cjk.py 的保护规则（中点连接的数字、全角范围、拉丁字母词后的编号、引文页码、全角
  ％），细则见各语言模块的说明。
- zh：年份逐位读（一九三七年，含「1982至1983年」「年（1905）」），其余按数值读；单独的 2 按量词选「两」
  或「二」；有时间词时读时刻（十四点三十分）；百分数、小数、单位、货币。年数与年份难分、编号、作品名
  开头的数丢弃。
- ja：汉字数字（千九百五十、四万一千），助数词读音交给 TTS；百分数、小数、时刻、单位、货币。读音有歧义
  的「十分」「五分」、片假名词后的编号、型号（97式）、单独的 0 丢弃。
- ko：按量词选汉字词数词（천구백사십사년）或固有词数词（세개，1–99，整百整千用汉字词）；유월、시월；
  제 序数、만/억/천 大数、번째、배；量词后面必须是助词或词尾；名称后的数字和量词不明的数丢弃。
- en：年份（nineteen sixty，按前后的词判断四位数是年份还是数量）、年代（1990s、'90s、mid-1960s）、序数、
  「October 13, 1960」「13 October 1960」（the thirteenth of October）日期、时刻、单位、货币（含 $1.1 billion）；
  缩写（etc.、e.g.、Dr. Smith、St. Louis、Jr.）、字母缩写（U.S. → U S）、&，君主序号（Henry the Eighth）和
  War、Part、Type 后的罗马数字（World War Two）。
- de：序数和日期按冠词、介词和名词的性数选词尾（am zehnten August、der dritten Liga、seit fünfundzwanzigstem
  August），年份（2001 → zweitausendeins）、年代（neunzehnhundertachtziger Jahre）、复合词（dreißig-jährige）、
  「4.40 Uhr」、Mio.、Mrd.、§、Nr. 等缩写，君主序号按格读（unter Heinrich dem Achten）；「1」在名词前仍丢弃。
- fr、es、pt、it：随名词变性的数按名词词表和阴性词尾选词形（vingt et une voix、doscientas personas、
  duas mil pessoas），名词后的编号读阳性（Fórmula uno）；世纪与君主序号（XVIIe siècle、siglo veinte、Felipe
  segundo、Louis quatorze、Pio dodicesimo）、名词前的罗马序数（Segunda Guerra Mundial）、常用缩写、以空格分千位
  的数、意语 anni '80 与 l'8 dicembre、法语时刻（21 h 30）。
- ru：用 pymorphy3 分析后面名词的格、数、性，确定基数的格（в пяти домах、более шестисот тысяч копий、
  погибли двадцать пять человек）；年份（г.、гг.、год）、日期（тринадцатого сентября）、带词尾的序数（23-м）、
  -летний 复合词、罗马数字（в девятнадцатом веке、Пётр Первый）、单位与百分号、т. е.、н. э. 等缩写。
- id、ms、th、vi、tr、fa：数词不变形，基数、年份、日期按数值读。id/ms：ke-21、1970-an、pukul 17.30、
  RM50、-10 derajat；th：14.30 น.、พ.ศ. 2500、99.50 บาท、句末误加的句点；vi：thứ nhất、tháng tư、
  lúc 6:30（sáu giờ ba mươi phút）、km 等单位、-10°C；tr：撇号后缀按元音和谐接到读法上（2006'da → iki bin
  altıda）、saat 14:30、序数（4. sınıf、2. Dünya Savaşı）、「%25」；fa：日期中的序数（دوازدهم مرداد）、15ام、
  14ساله、٪۱۰、ساعت 2:30。共用的负温度、5–6 位数规则在 langs/extras.py。
- ar：按被数名词的性（名词表）和数词的格（介词、支配属格的名词、及物动词之后）选择词形，性或格无法确定且
  影响词形时丢弃；年份、日期、百分数、单位、货币，以及 م、هـ 纪元标记。
- fil：按 FLEURS 朗读者的习惯用他加禄语数词（年份 isang libo siyam na raan at animnapu't anim、ika-18 →
  ikalabingwalo），修饰名词时加连接成分 -ng/na；「alas 6」用西班牙语来源数词（alas sais）；小数、0、分钟丢弃。

依赖与环境
==========

中文的繁简转换调用 OpenCC 1.4.2（Apache-2.0），直通版的其余语言只用标准库。基础读法的英语、德语、法语、
西班牙语、葡萄牙语、意大利语、俄语数词调用 num2words 0.5.14（LGPL-2.1，以依赖方式调用，不随项目分发），
其余语言的读法在 numwords.py 中实现。俄语另用 pymorphy3 2.0.6（MIT，词典数据来自 OpenCorpora）分析名词的格和性。
工作目录的 envs/tn 是装有这两个包的虚拟环境（uv venv 基于 Python 3.14，uv pip install num2words==0.5.14
opencc==1.4.2），俄语还需要 uv pip install pymorphy3==2.0.6。

测试
====

    python -m unittest pipeline.tn.tests.test_tn pipeline.tn.tests.test_basic pipeline.tn.tests.test_cjk \
        pipeline.tn.tests.test_european

已知局限
========

规则只覆盖常见写法，没有把握的一律丢弃，因此含数字条目的保留比例因语言而异（俄语、阿拉伯语、菲律宾语
最低）。比对模块不把单位、货币名和「点、分」的时刻读法
转成符号，这类改写在 review 中列为预期内的不一致，合成后的 ASR 核对同样会判为不一致；韩语「육십일」
（60日）、「십세」（10岁）会被比对模块读成别的数。
"""

from .core import LANGS, TnResult, clean, normalize_entry
from .detect import find_issues

__all__ = ["LANGS", "TnResult", "clean", "find_issues", "normalize_entry"]
