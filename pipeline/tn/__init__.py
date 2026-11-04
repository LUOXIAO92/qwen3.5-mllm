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

1. 清理原文（core.clean）：NFC，删除零宽空格、软连字符和方向控制符，空白合并为一个空格；中文按 OpenCC
   单字表转为简体，表与 pipeline/compare 共用。
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

- zh：年份逐位读（一九三七年），其余按数值读；单独的 2 按量词选「两」或「二」，没有把握时丢弃；百分数、
  小数、时刻（十四点三十分）、单位、货币。不带「年」的四位数、三位以上后接非量词的数（911事件）丢弃。
- ja：汉字数字（千九百五十、四万一千），量词读音交给 TTS 前端；百分数、小数、时刻、单位、货币。
- ko：按量词选汉字词数词（천구백사십사년）或固有词数词（세개，只用于 1–99）；유월、시월；后面既不是
  登记的量词也不是助词的数丢弃。
- en：年份（nineteen sixty）只在 in、since 等词之后读，年代（nineteen nineties）、序数（21st）、
  「Month D, YYYY」日期、时刻、单位、货币；「13 October」这类日在前的日期丢弃。
- de：基数、年份（neunzehnhundert…，前后文像数量时丢弃）、「14:30 Uhr」；1 和末两位为 01 的数、序数和日期要变格，丢弃。
- fr、es、pt、it：基数、年份、「1er octobre」「11 de mayo de 2025」日期、序数（2e、3º、36ª）、单位、货币；
  随名词变性的数（un/une、doscientos/doscientas、dois/duas）只在确定为阳性时改写。
- ru：只改写「2008 года」「в 1950-х годах」这类后接 год 的年份，和支配属格的词之后、复数属格名词之前的基数。
- id、ms、th、vi、tr、fa：数词不变形，基数、年份、日期按数值读；id/ms 的 ke-21，vi 的 thứ nhất、tháng tư，
  tr 的序数（4. sınıf）和「%25」，fa 日期中的序数（دوازدهم مرداد）。
- ar：只改写 عام、سنة、في、منذ 之后的年份（属格，ألفين وخمسة وعشرين），其余丢弃。
- fil：他加禄语、西班牙语、英语三套数词随语境选用，含数字的条目全部丢弃。

依赖与环境
==========

直通版只用标准库。基础读法的英语、德语、法语、西班牙语、葡萄牙语、意大利语、俄语数词调用 num2words
0.5.14（LGPL-2.1，以依赖方式调用，不随项目分发），其余语言的读法在 numwords.py 中实现。工作目录的
envs/tn 是装有 num2words 的虚拟环境（uv venv 基于 Python 3.14，uv pip install num2words==0.5.14）。

测试
====

    python -m unittest pipeline.tn.tests.test_tn pipeline.tn.tests.test_basic

已知局限
========

规则只覆盖常见写法，没有把握的一律丢弃，因此含数字条目的保留比例因语言而异（俄语、阿拉伯语、菲律宾语
最低）。中文繁简转换为单字表，「乾隆」会转成「干隆」。比对模块不把单位、货币名和「点、分」的时刻读法
转成符号，这类改写在 review 中列为预期内的不一致，合成后的 ASR 核对同样会判为不一致；韩语「육십일」
（60日）、「십세」（10岁）会被比对模块读成别的数。
"""

from .core import LANGS, TnResult, clean, normalize_entry
from .detect import find_issues

__all__ = ["LANGS", "TnResult", "clean", "find_issues", "normalize_entry"]
