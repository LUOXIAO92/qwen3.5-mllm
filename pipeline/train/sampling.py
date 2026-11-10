"""按来源或语言加权抽样，以及每轮的批次计划（实验 v2，任务 T1）。

每条样本有一个抽样权重，表示它在一轮中平均出现几次：权重 1 出现一次，2 出现两次，0.5 每轮有一半的机会出现，
0 不参加训练。权重由 --sample-weight 规则给出，规则写成「字段:取值=权重」，例如 lang:fa=2 让波斯语多抽一倍，
source:Common Voice 22.0=0.5 让这个来源少抽一半；一条样本命中多条规则时权重相乘，没有命中的为 1。

空目标样本（训练目标为空字符串，即无人声片段）的总量另由 empty_share 控制：先按规则算出各条样本的权重，
再把全部空目标样本的权重同乘一个系数，使它们在每轮样本中的期望占比等于 empty_share（按条数计）。
空目标样本之间的相对权重保持不变，例如 category:music=0.5 仍让音乐片段少抽一半。

每轮先按 (seed, 轮次) 决定每条样本出现几次：权重的整数部分是固定次数，小数部分是再多出现一次的概率。
得到的样本序列再按时长分组成批（data.length_grouped_batches）。批次中的每一项是 SampleKey(index, epoch,
position)：index 是清单中的行号，position 是它在本轮样本序列中的位置，同一条样本在一轮中出现两次时
position 不同。混音的随机数由 (seed, epoch, position) 决定，因此整个计划只由参数决定，断点续训时重新生成
同一轮的批次、跳过已训练的部分，得到的批次和混音都与不中断的训练相同。

所有权重都是 1 时，每轮的样本序列就是 0, 1, ..., n - 1，批次与 v1 的 length_grouped_batches 完全相同。
"""

import numpy as np

from pipeline.train.data import STREAM_OCCURRENCES, SampleKey, length_grouped_batches


def parse_weight_rules(specs):
    """把 ["lang:fa=2", "source:Common Voice 22.0=0.5"] 解析成 [(字段, 取值, 权重), ...]。

    字段和取值之间用第一个冒号分隔，取值和权重之间用最后一个等号分隔，取值中可以有空格和斜杠。
    """
    rules = []
    for spec in specs or []:
        field, separator, rest = spec.partition(":")
        value, equals, weight = rest.rpartition("=")
        if not separator or not equals or not field or not value:
            raise ValueError(f"抽样权重应写成「字段:取值=权重」：{spec!r}")
        weight = float(weight)
        if weight < 0 or not np.isfinite(weight):
            raise ValueError(f"抽样权重必须是非负的有限值：{spec!r}")
        rules.append((field, value, weight))
    return rules


def rule_weight(row, rules):
    """一条样本命中的全部规则的权重之积，没有命中时为 1。"""
    weight = 1.0
    for field, value, factor in rules:
        if field in row and str(row[field]) == value:
            weight *= factor
    return weight


def sample_weights(
    rows,
    empty,
    rules = (),
    empty_share = None,
):
    """返回每条样本的抽样权重（numpy 数组）。

    empty 标出哪些样本是空目标。empty_share 不为 None 时，把空目标样本的权重整体缩放，
    使它们的期望占比等于 empty_share；清单中没有空目标样本时不做处理。
    """
    weights = np.array(
        object = [rule_weight(row = row, rules = rules) for row in rows],
        dtype = np.float64,
    )
    if empty_share is None:
        return weights
    if not 0 <= empty_share < 1:
        raise ValueError(f"空目标占比应在 [0, 1) 内：{empty_share}")
    empty = np.asarray(a = empty, dtype = bool)
    empty_total = weights[empty].sum()
    speech_total = weights[~empty].sum()
    if empty_total == 0:
        return weights
    if speech_total == 0:
        raise ValueError("清单中只有空目标样本，无法按占比抽样。")
    # 缩放系数 k 满足 k × empty_total / (speech_total + k × empty_total) = empty_share。
    factor = empty_share * speech_total / ((1 - empty_share) * empty_total)
    weights[empty] *= factor
    return weights


def epoch_occurrences(weights, seed, epoch):
    """这一轮的样本序列：每条样本按权重出现若干次，按行号顺序排列（之后由分批函数打乱）。"""
    weights = np.asarray(a = weights, dtype = np.float64)
    whole = np.floor(weights)
    rng = np.random.default_rng(seed = [seed, epoch, STREAM_OCCURRENCES, 0])
    # 每条样本都取一个随机数，结果不依赖其他样本的权重是否为整数。
    extra = rng.random(size = len(weights)) < weights - whole
    counts = whole.astype(np.int64) + extra
    return np.repeat(a = np.arange(len(weights)), repeats = counts)


class EpochPlan:
    """每轮的抽样和分批。结果只由构造参数和轮次决定。"""

    def __init__(
        self,
        durations,
        weights,
        batch_size,
        group_batches,
        max_batch_seconds,
        seed,
    ):
        self.durations = np.asarray(a = durations, dtype = np.float64)
        self.weights = None if weights is None else np.asarray(a = weights, dtype = np.float64)
        self.batch_size = batch_size
        self.group_batches = group_batches
        self.max_batch_seconds = max_batch_seconds
        self.seed = seed

    def occurrences(self, epoch):
        if self.weights is None:
            return np.arange(len(self.durations))
        return epoch_occurrences(weights = self.weights, seed = self.seed, epoch = epoch)

    def position_batches(self, epoch):
        """返回 (本轮样本序列, 批次列表)，批次中是样本在序列中的位置。

        训练时按批次逐个换成 SampleKey（keys），不一次性为整轮几百万条样本建对象。
        """
        occurrences = self.occurrences(epoch = epoch)
        batches = length_grouped_batches(
            durations = self.durations[occurrences].tolist(),
            batch_size = self.batch_size,
            seed = self.seed,
            epoch = epoch,
            group_batches = self.group_batches,
            max_batch_seconds = self.max_batch_seconds,
        )
        return occurrences, batches

    @staticmethod
    def keys(occurrences, epoch, batch):
        return [SampleKey(index = int(occurrences[position]), epoch = epoch, position = position) for position in batch]

    def batches(self, epoch):
        """这一轮的批次列表，每个批次是 SampleKey 的列表。"""
        occurrences, batches = self.position_batches(epoch = epoch)
        return [self.keys(occurrences = occurrences, epoch = epoch, batch = batch) for batch in batches]


def expected_composition(rows, weights, empty):
    """按语言统计每轮的期望样本数和音频小时数，空目标样本单列为 empty。"""
    summary = {}
    for row, weight, is_empty in zip(rows, weights, empty):
        key = "empty" if is_empty else row["lang"]
        entry = summary.setdefault(key, {"rows": 0, "per_epoch": 0.0, "hours_per_epoch": 0.0})
        entry["rows"] += 1
        entry["per_epoch"] += float(weight)
        entry["hours_per_epoch"] += float(weight) * float(row["duration_s"]) / 3600
    total = sum(entry["per_epoch"] for entry in summary.values())
    return {
        key: {
            "rows": entry["rows"],
            "per_epoch": round(number = entry["per_epoch"], ndigits = 1),
            "share": round(number = entry["per_epoch"] / total, ndigits = 4) if total else 0.0,
            "hours_per_epoch": round(number = entry["hours_per_epoch"], ndigits = 2),
        }
        for key, entry in sorted(summary.items())
    }
