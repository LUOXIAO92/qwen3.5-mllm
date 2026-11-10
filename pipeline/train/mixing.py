"""训练时实时混入背景声（实验 v2，任务 T1）。

背景声列表是 JSONL，每行至少有 audio_path（相对列表所在目录）、duration_s 和 category。混音在数据读取进程中
完成：TranscriptionDataset 读出语音波形后调用 BackgroundMixer，按给定的随机数生成器决定这条样本是否混音、
混哪一段背景声、信噪比多少、从背景声的哪里开始截取。生成器由 (seed, 轮次, 样本在本轮中的位置) 决定，
同一条样本在续训前后得到同样的混音结果。

信噪比（SNR）是语音功率与背景声功率之比，用分贝表示：SNR = 10 × log10(语音功率 / 背景声功率)，功率是波形
平方的平均值。0 dB 时两者一样响，20 dB 时背景声的功率是语音的百分之一。混音时先把背景声截取或循环到与语音
等长，再缩放到目标信噪比后与语音相加；相加后的峰值超过 0.99 时整体缩小，信噪比不变。

背景声按类别加权抽取：每段背景声的抽中概率正比于它所属类别的权重（默认 1），例如 music=0.5 让音乐片段
的抽中概率减半。
"""

import json
from pathlib import Path

import numpy as np

from pipeline.train.data import SAMPLING_RATE, load_audio

PEAK_LIMIT = 0.99
# 功率低于这个值的波形视为静音，不混音。
SILENCE_POWER = 1e-10


def parse_category_weights(specs):
    """把 ["music=0.5", "noise=2"] 解析成 {"music": 0.5, "noise": 2.0}。"""
    weights = {}
    for spec in specs or []:
        category, equals, weight = spec.rpartition("=")
        if not equals or not category:
            raise ValueError(f"背景声类别权重应写成「类别=权重」：{spec!r}")
        weight = float(weight)
        if weight < 0 or not np.isfinite(weight):
            raise ValueError(f"背景声类别权重必须是非负的有限值：{spec!r}")
        weights[category] = weight
    return weights


def fit_length(noise, length, start):
    """从 start 开始取 length 个采样点；背景声不够长时从头循环接上。"""
    indices = (start + np.arange(length)) % len(noise)
    return noise[indices]


def mix_at_snr(speech, noise, snr_db):
    """把与语音等长的背景声缩放到目标信噪比后相加，返回 (混音波形, 背景声缩放系数, 峰值缩放系数)。

    语音或背景声几乎静音时不混音，返回原语音和 None。
    """
    speech_power = float(np.mean(a = np.square(speech, dtype = np.float64)))
    noise_power = float(np.mean(a = np.square(noise, dtype = np.float64)))
    if speech_power < SILENCE_POWER or noise_power < SILENCE_POWER:
        return speech, None, None
    scale = np.sqrt(speech_power / (noise_power * 10 ** (snr_db / 10)))
    mixed = speech.astype(np.float64) + scale * noise.astype(np.float64)
    peak = float(np.max(a = np.abs(mixed)))
    gain = PEAK_LIMIT / peak if peak > PEAK_LIMIT else 1.0
    return (mixed * gain).astype(np.float32), float(scale), gain


class BackgroundMixer:
    """按设定的比例和信噪比范围给语音样本混入背景声。"""

    def __init__(
        self,
        manifest,
        prob = 0.3,
        snr_min = 0.0,
        snr_max = 20.0,
        category_weights = None,
        audio_root = None,
    ):
        if not 0 <= prob <= 1:
            raise ValueError(f"混音比例应在 [0, 1] 内：{prob}")
        if snr_min > snr_max:
            raise ValueError(f"信噪比下限 {snr_min} 大于上限 {snr_max}")
        manifest = Path(manifest)
        root = Path(audio_root) if audio_root is not None else manifest.parent
        category_weights = category_weights or {}
        self.prob = prob
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.paths = []
        self.durations = []
        self.categories = []
        weights = []
        seen = set()
        with open(file = manifest, mode = "r", encoding = "utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(s = line)
                seen.add(row["category"])
                weight = category_weights.get(row["category"], 1.0)
                if weight == 0:
                    continue
                audio_path = Path(row["audio_path"])
                if not audio_path.is_absolute():
                    audio_path = root / audio_path
                self.paths.append(str(audio_path))
                self.durations.append(float(row["duration_s"]))
                self.categories.append(row["category"])
                weights.append(weight)
        unknown = sorted(set(category_weights) - seen)
        if unknown:
            raise ValueError(f"背景声列表中没有这些类别：{unknown}，已有的类别：{sorted(seen)}")
        if not self.paths:
            raise ValueError(f"背景声列表中没有可用的片段：{manifest}")
        weights = np.asarray(a = weights, dtype = np.float64)
        self.cumulative = np.cumsum(a = weights) / weights.sum()
        self.probabilities = weights / weights.sum()

    def __len__(self):
        return len(self.paths)

    def category_shares(self):
        """每个类别的片段数、总时长和抽中概率，打印配置用。"""
        shares = {}
        for category, duration, probability in zip(self.categories, self.durations, self.probabilities):
            entry = shares.setdefault(category, {"clips": 0, "hours": 0.0, "share": 0.0})
            entry["clips"] += 1
            entry["hours"] += duration / 3600
            entry["share"] += float(probability)
        return {
            category: {
                "clips": entry["clips"],
                "hours": round(number = entry["hours"], ndigits = 2),
                "share": round(number = entry["share"], ndigits = 4),
            }
            for category, entry in sorted(shares.items(), key = lambda item: -item[1]["share"])
        }

    def draw(self, rng):
        """取这条样本的混音决定：(是否混音, 背景声编号, 信噪比, 截取起点的比例)。

        固定取 4 个随机数，取用的顺序与是否混音无关。
        """
        uniforms = rng.random(size = 4)
        clip = int(np.searchsorted(a = self.cumulative, v = uniforms[1], side = "right"))
        # 累积概率的最后一项因浮点误差可能略小于 1。
        clip = min(clip, len(self.paths) - 1)
        snr_db = self.snr_min + (self.snr_max - self.snr_min) * uniforms[2]
        return bool(uniforms[0] < self.prob), clip, float(snr_db), float(uniforms[3])

    def __call__(self, waveform, rng):
        """返回 (波形, 混音信息)。不混音时原样返回语音和 None。"""
        mixed, clip, snr_db, start_fraction = self.draw(rng = rng)
        if not mixed or len(waveform) == 0:
            return waveform, None
        speech_seconds = len(waveform) / SAMPLING_RATE
        clip_seconds = self.durations[clip]
        if clip_seconds > speech_seconds:
            # 背景声比语音长：只解码需要的一段。
            start_s = start_fraction * (clip_seconds - speech_seconds)
            noise = load_audio(
                path = self.paths[clip],
                start_s = start_s,
                stop_s = start_s + speech_seconds,
            )
            start = 0
        else:
            noise = load_audio(path = self.paths[clip])
            start = int(start_fraction * len(noise))
        if len(noise) == 0:
            return waveform, None
        noise = fit_length(noise = noise, length = len(waveform), start = start)
        result, scale, gain = mix_at_snr(speech = waveform, noise = noise, snr_db = snr_db)
        if scale is None:
            return waveform, None
        return result, {
            "category": self.categories[clip],
            "clip": clip,
            "snr_db": snr_db,
            "gain": gain,
        }
