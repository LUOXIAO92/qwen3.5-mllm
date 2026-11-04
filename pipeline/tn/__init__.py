"""朗读稿生成（实验 v1，任务 T18）：把原文改写成交给 TTS 的朗读稿。

用法
====

    python -m pipeline.tn.run --input entries.jsonl --output scripts.jsonl

    from pipeline.tn import normalize_entry
    normalize_entry(original = "你的狗在這。", lang = "zh")
    # TnResult(script = "你的狗在这。", status = "unchanged", note = "转为简体")

输入是 pipeline/CONVENTIONS.md 的文本条目，输出在每条后补上三个字段：script（朗读稿，丢弃时为 null）、
tn_status（unchanged、normalized、dropped）和 tn_note（改写或丢弃的原因）。

处理流程
========

1. 清理原文（core.clean）：NFC，删除零宽空格、软连字符和方向控制符，空白合并为一个空格；中文按 OpenCC
   单字表转为简体，表与 pipeline/compare 共用。
2. 找出需要改写的片段（detect.find_issues）：各文字的数字、上标和分数，货币、数学和其他符号（%、°、&、@ 等），
   带句点的缩写（e.g.、т. д.、Dr.、泰语 พ.ศ.、阿拉伯语 هـ），罗马数字，网址。
3. 没有这些片段的条目，朗读稿就是清理后的原文，标 unchanged；有的条目标 dropped，tn_note 写明是哪些片段。

依赖
====

只依赖 Python 标准库，支持 Python 3.10 及以上。

测试
====

    python -m unittest pipeline.tn.tests.test_tn
"""

from .core import LANGS, TnResult, clean, normalize_entry
from .detect import find_issues

__all__ = ["LANGS", "TnResult", "clean", "find_issues", "normalize_entry"]
