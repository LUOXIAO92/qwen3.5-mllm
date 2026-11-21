"""退化检测：gzip 压缩率、n-gram 重复率，以及循环的起点。

循环分两种，截断点取两者中较早的一个（`degeneration_onset`）。

原样循环（`loop_onset`）：同一段 token 原样反复出现。对每个位置 i 取从 i 开始的 16-gram，找它下一次出现的位置，
距离记为 d_i。一段周期为 p 的循环里，连续的一串位置都有 d_i = p。把 d_i 相同的连续位置合成一段，这段循环覆盖的长度是
「段长 + p + 15」，其中第一份之后的部分都是前文的原样复制，称为复制长度。复制长度至少 64 个 token，并且周期小于 32 时
至少出现 3 份、周期 32 及以上至少出现 2 份，才算循环；截断点是第二份开始的位置，保留第一份。

近似循环（`near_loop_onset`）：每次略有变化地反复，例如只换一个数字或词的同一句话。对每个位置记录它的 4-gram 是否在
前面 256 个 token 以内出现过（近处重复），窗口长 256 个 token；某个窗口中近处重复的比例达到 0.75 时，截断点取这个窗口里
第一个近处重复的位置。在 6,000 篇人写的文档开头（HPLT、FineWeb2-HQ-Plus、Wikinews）上，这个比例的 99.9% 分位数为
0.75，超过 0.75 的只有 5 篇，都是天气表、价格表、比分表。

16-gram、4-gram 都用 64 位多项式哈希比较，碰撞概率可以忽略。
"""

import gzip

import numpy as np

NGRAM = 16
MIN_REPEAT = 64
SHORT_PERIOD = 32
WINDOW = 256
NEAR_LOOP_THRESHOLD = 0.75
_BASE = np.uint64(1099511628211)


def ngram_hashes(ids, n = NGRAM):
    """每个位置开始的 n-gram 的 64 位哈希，长度 len(ids) - n + 1。"""
    a = np.asarray(ids, dtype = np.uint64) + np.uint64(1)
    if len(a) < n:
        return np.zeros(0, dtype = np.uint64)
    h = np.zeros(len(a) - n + 1, dtype = np.uint64)
    with np.errstate(over = "ignore"):
        for k in range(n):
            h = h * _BASE + a[k: len(a) - n + 1 + k]
    return h


def next_occurrence_distance(hashes):
    """每个 n-gram 到它下一次出现的距离，没有再出现时为 0。"""
    m = len(hashes)
    distance = np.zeros(m, dtype = np.int64)
    if m < 2:
        return distance
    order = np.lexsort((np.arange(m), hashes))
    same = hashes[order[1:]] == hashes[order[:-1]]
    first = order[:-1][same]
    distance[first] = order[1:][same] - first
    return distance


def loop_onset(ids, min_repeat = MIN_REPEAT, short_period = SHORT_PERIOD, n = NGRAM):
    """返回循环中第二份开始的位置（截断点）；没有循环返回 None。"""
    distance = next_occurrence_distance(hashes = ngram_hashes(ids = ids, n = n))
    if not distance.any():
        return None
    # 相同距离的连续位置合成一段：段的起点、长度、距离
    change = np.flatnonzero(np.diff(distance) != 0) + 1
    starts = np.concatenate(([0], change))
    lengths = np.diff(np.concatenate((starts, [len(distance)])))
    periods = distance[starts]
    repeat = lengths + n - 1  # 复制长度：整段覆盖 lengths + period + n - 1，减去第一份
    copies_ok = np.where(periods < short_period, repeat >= 2 * periods, repeat >= periods)
    hit = (periods > 0) & (repeat >= min_repeat) & copies_ok
    if not hit.any():
        return None
    onsets = starts[hit] + periods[hit]
    return int(onsets.min())


def previous_occurrence_distance(hashes):
    """每个 n-gram 到它上一次出现的距离，前文没有出现过时为 0。"""
    m = len(hashes)
    distance = np.zeros(m, dtype = np.int64)
    if m < 2:
        return distance
    order = np.lexsort((np.arange(m), hashes))
    same = hashes[order[1:]] == hashes[order[:-1]]
    later = order[1:][same]
    distance[later] = later - order[:-1][same]
    return distance


def local_repeat(ids, window = WINDOW, n = 4):
    """每个窗口起点 s 上的局部重复率：[s, s + window) 中有多少比例的 4-gram 在它前面 window 个 token 以内出现过。

    返回 (各窗口的重复率, 每个位置是否为近处重复)。人写的网页、新闻文本中这个比例的最大值不超过 0.5。
    """
    hashes = ngram_hashes(ids = ids, n = n)
    distance = previous_occurrence_distance(hashes = hashes)
    repeated = ((distance > 0) & (distance <= window)).astype(np.int64)
    if len(repeated) < window:
        return np.zeros(0), repeated
    cumulative = np.concatenate(([0], np.cumsum(repeated)))
    return (cumulative[window:] - cumulative[:-window]) / window, repeated


def near_loop_onset(ids, window = WINDOW, threshold = NEAR_LOOP_THRESHOLD):
    """近似循环（每次略有变化地反复）的起点：第一个局部重复率达到 threshold 的窗口中，第一个近处重复的 4-gram 的位置。"""
    rates, repeated = local_repeat(ids = ids, window = window)
    hit = np.flatnonzero(rates >= threshold)
    if len(hit) == 0:
        return None
    start = int(hit[0])
    return start + int(np.argmax(repeated[start:] > 0))


def degeneration_onset(ids):
    """原样循环与近似循环中较早的起点；都没有时返回 None。"""
    onsets = [onset for onset in (loop_onset(ids = ids), near_loop_onset(ids = ids)) if onset is not None]
    return min(onsets) if onsets else None


def gzip_ratio(text):
    """压缩后字节数 / 原始字节数；越低越重复，正常文本约 0.3–0.5。"""
    raw = text.encode("utf-8")
    if not raw:
        return 1.0
    return len(gzip.compress(raw, compresslevel = 6)) / len(raw)


def ngram_repeat_rate(ids, n = 4):
    """重复的 n-gram 占全部 n-gram 的比例（1 - 不同 n-gram 数 / n-gram 总数）。"""
    hashes = ngram_hashes(ids = ids, n = n)
    if len(hashes) == 0:
        return 0.0
    return 1.0 - len(np.unique(hashes)) / len(hashes)


def max_window_repeat(ids, window = 256, n = NGRAM):
    """长为 window 的窗口里，在样本前文出现过的 16-gram 所占比例的最大值；接近 1 说明这一段在复制前文。"""
    hashes = ngram_hashes(ids = ids, n = n)
    m = len(hashes)
    if m == 0:
        return 0.0
    order = np.lexsort((np.arange(m), hashes))
    seen_before = np.zeros(m, dtype = np.int64)
    same = hashes[order[1:]] == hashes[order[:-1]]
    seen_before[order[1:][same]] = 1
    if m <= window:
        return float(seen_before.mean())
    cumulative = np.concatenate(([0], np.cumsum(seen_before)))
    return float(((cumulative[window:] - cumulative[:-window]) / window).max())
