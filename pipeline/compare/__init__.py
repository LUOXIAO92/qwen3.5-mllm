"""朗读稿与 ASR 识别结果的比对，以及评测打分用的规范化（实验 v1，任务 T05）。

用法
====

    from pipeline.compare import compare, normalize_for_scoring

    result = compare(script = "Он жил в пяти домах.", hypothesis = "Он жил в 5 домах.", lang = "ru")
    result.match    # True
    result.detail   # ""，不一致时为差异说明，例如「替换「пяти」→「6」」

    normalize_for_scoring(text = "Он жил в пяти домах.", lang = "ru")   # "он жил в 5 домах"

compare(script, hypothesis, lang, original = None) 返回 CompareResult(match, detail)，可以直接解包，
两个字段对应识别结果 JSONL 中的 match 和 compare_detail。给出 original（原文）时，识别结果中写成阿拉伯
数字的数值还必须能在原文中找到，用来发现朗读稿把数值写错、TTS 照读、ASR 照写的情况。
normalize_for_scoring(text, lang) 把文本规范化并把数词写成数值，供评测计算错误率；中文、日语、韩语、
泰语的结果不含空格，适合按字计算，其余语言以单个空格分词。语言代码为
zh en ja ko de fr ru pt es it id ms th vi fil ar tr fa。

比对流程
========

两边先做同一套字符规范化（normalize.py）：NFKC、删除零宽字符和方向控制符、各种数字字形统一为 ASCII
数字、大小写统一（土耳其语单独处理 I/İ）、俄语 ё 统一为 е；阿拉伯语和波斯语删除元音符号和 tatweel，
统一 alef、ya、kaf、ta marbuta 等字母变体；中文按 OpenCC 的单字表转为简体；日语平假名统一为片假名。
标点在切分时删除，比较普通文字时忽略空白。

然后把两边的数词都换成数值记号（parse.py）。数值记号的设计对所有语言相同：一个记号记录整数值、
用于逐位比较的数字串、来源（读法、阿拉伯数字或「5 миллионов」这类混写）以及基数/序数属性；
小数写成「整数部分、小数点记号、小数部分」三段，百分号和负号也各有一个记号，记号区分符号写法和读法。
各语言的读法由 langs/ 下的词表给出，包括俄语、德语、阿拉伯语等的全部格、性、数形式，再由 numerals.py
中通用的位值组合器拼成数：组内按位值从高到低相加，遇到「千、万、million」这类大单位时把前面的部分作为
系数。德语、意大利语的连写数词按词表部件切分，中文、日语、韩语、泰语按最长匹配扫描。

比较规则（core.py）如下：

- 一边是阿拉伯数字时按数值比较。年份等分段读法可以拼接对应，例如「nineteen eighty-four」对「1984」、
  「二零二六」对「2026」；小数部分逐位比较，「three point five」不会与「3.05」一致。
- 两边都是读法时，除数值相同外还要求词形一致，只合并词表中登记的书写变体（见下文）。因此变格错误
  （в пять домах 对 в пяти домах）、阴阳性错误、韩语固有词与汉字词混用、序数读成基数都会判为不一致。
- 带序数标记的数字（3rd、3e、3º、3-й）只与序数读法一致；不带标记的数字两种都可以，因为日期、年份
  常把序数写成普通数字（3 мая、1984 году、May 3）。
- 前导零只容许「09」这类两位写法，「00」「007」必须有逐位对应的读法。

比较时合并的书写变体：中文两/二、幺/一、〇/零和大写数字；日语〇/零；越南语 mốt/một、tư/bốn（mươi 之后）、
lăm/nhăm/năm（十位之后）、lẻ/linh、ngàn/nghìn、tỉ/tỷ、bẩy/bảy；泰语 เอ็ด/หนึ่ง，以及重复符 ๆ 与写出的重复。

依赖与许可
==========

只依赖 Python 标准库，支持 Python 3.10 及以上。data/zh_t2s.txt 由 OpenCC 的 TSCharacters.txt 生成
（Apache-2.0，https://github.com/BYVoid/OpenCC）。开发用的词表回归检查 tests/rbnf_check.py 需要
unicode-rbnf（MIT，数据为 Unicode 许可），只在开发环境中安装。

测试
====

    python -m unittest pipeline.compare.tests.test_compare
    python -m pipeline.compare.tests.run_cases          # 逐语言统计正例、反例
    python -m pipeline.compare.tests.rbnf_check         # 需要 unicode-rbnf

已知局限
========

比对只检查数值。识别结果写成阿拉伯数字时，读法风格和语法错误都无法发现，例如朗读稿是「в пяти домах」、
TTS 读成「в пять домах」、ASR 写成「в 5 домах」，会判为一致；年份按数值读还是分段读、该读「两」读成「二」、
韩语该用固有词用了汉字词也是如此。这些问题需要在生成朗读稿时解决。合并的书写变体读音不同，读成另一种
变体同样不会被发现。

原文核对只针对识别结果中的阿拉伯数字，并且按集合判断：数值在原文任意位置出现即可。识别结果写成读法时，
朗读稿相对原文的数值错误需要由生成朗读稿的环节检查。

以下情况会误杀：读法与符号写法语序不同（「vierzehn Uhr dreißig」对「14:30 Uhr」，「三点零五分」对「3:05」，
泰语「ร้อยละห้า」对「5%」）；货币和单位符号不转为读法（「five dollars」对「$5」，「三公里」对「3km」）；
范围中的连字符不转为「到」（「three to five」对「3-5」）；日语汉字与假名的写法不同；菲律宾语只覆盖常见的
他加禄语、英语和少量西班牙语来源数词。

日语只比较书写形式，汉字数词的读音歧义（十分、一日、八分）无法发现。阿拉伯语附着成分和冠词 ال 在数内的
处理是启发式的，双数名词（طالبان，两个学生）不视为数词。泰语没有分词，ๆ 按「重复前面一个完整音节串」近似
处理。比较时忽略空白，只差在分词或空格的结果视为一致。阿拉伯数字的歧义写法（1.234 可能是千分位或小数）
会同时尝试两种解释。差异说明显示的是规范化后的文本（小写、日语片假名等）。

各 ASR 的实际输出习惯要到 #15、#16、#17 和 #21 才知道，届时需要据实修订词表和规则。
"""

from .core import CompareResult, compare, normalize_for_scoring
from .langs import LANGS

__all__ = ["CompareResult", "LANGS", "compare", "normalize_for_scoring"]
