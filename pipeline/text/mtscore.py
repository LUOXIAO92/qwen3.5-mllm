"""机翻腔的粗筛：字符 n-gram 语言模型的每字符交叉熵（不需要大模型，纯 Python）。

机翻和洗稿文本的字符序列不像母语者的写作：词内拼写怪、混入外语词、搭配生硬，每字符交叉熵偏高。
做法：用同语言的人写文本（Tatoeba、Common Voice、维基段落、Wikinews）训练一个 5 元字符模型
（逐级回退的线性插值，加一平滑），在留出的人写文本上定出阈值（取某个分位数），
候选文本的交叉熵超过阈值就丢弃。

代价：模型只见过训练域的写法，新闻和说明文里的专名、数字、专业词也会推高交叉熵，
所以阈值只用来去掉最差的一小部分；单靠它不能保证剔除所有机翻。
"""

from __future__ import annotations

import math
from collections import defaultdict

ORDER = 5


class CharLM:
    """字符 5 元插值模型。`train` 加入文本，`cross_entropy` 返回每字符的比特数（越高越不像训练文本）。"""

    def __init__(self, *, order: int = ORDER) -> None:
        self.order = order
        self.counts: list[dict[str, dict[str, int]]] = [defaultdict(lambda: defaultdict(int)) for _ in range(order)]
        self.totals: list[dict[str, int]] = [defaultdict(int) for _ in range(order)]
        self.vocab: set[str] = set()

    def train(self, text: str) -> None:
        padded = "\x02" * (self.order - 1) + text + "\x03"
        self.vocab.update(padded)
        for index in range(self.order - 1, len(padded)):
            char = padded[index]
            for n in range(self.order):  # n = 上下文长度
                context = padded[index - n:index]
                self.counts[n][context][char] += 1
                self.totals[n][context] += 1

    def prob(self, context: str, char: str) -> float:
        """逐级回退的插值：上下文越长越可靠，权重随该上下文出现次数增加。"""
        probability = 1.0 / (len(self.vocab) + 1)
        for n in range(self.order):
            if n > len(context):
                break
            ctx = context[len(context) - n:] if n else ""
            total = self.totals[n].get(ctx, 0)
            if not total:
                break
            count = self.counts[n][ctx].get(char, 0)
            weight = total / (total + 3.0)
            probability = weight * (count / total) + (1 - weight) * probability
        return probability

    def cross_entropy(self, text: str) -> float:
        padded = "\x02" * (self.order - 1) + text + "\x03"
        bits = 0.0
        for index in range(self.order - 1, len(padded)):
            context = padded[max(0, index - self.order + 1):index]
            bits -= math.log2(self.prob(context, padded[index]))
        return bits / (len(padded) - self.order + 1)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]
