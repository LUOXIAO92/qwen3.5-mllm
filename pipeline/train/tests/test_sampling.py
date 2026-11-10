"""加权抽样的比例、与 v1 分批的一致性、续训一致性：python -m unittest pipeline.train.tests.test_sampling"""

import tempfile
import unittest
from collections import Counter

import numpy as np

from pipeline.train.data import TranscriptionDataset, length_grouped_batches
from pipeline.train.mixing import BackgroundMixer
from pipeline.train.sampling import (
    EpochPlan,
    epoch_occurrences,
    expected_composition,
    parse_weight_rules,
    sample_weights,
)
from pipeline.train.tests.fixtures import speech_rows, write_background
from pipeline.train.train import BatchStream


def make_rows(count, seed = 0):
    rng = np.random.default_rng(seed = seed)
    langs = ["en", "fa", "ja", "zxx"]
    rows = []
    for number in range(count):
        lang = langs[number % 4]
        rows.append({
            "sample_id": f"r-{number}",
            "lang": lang,
            "source": "FSD50K/dev" if lang == "zxx" else f"Corpus {number % 3}.0",
            "duration_s": float(rng.uniform(low = 1.0, high = 30.0)),
            "original": "" if lang == "zxx" else "text",
        })
    return rows


def flatten(batches):
    return [key for batch in batches for key in batch]


class RuleTest(unittest.TestCase):
    def test_parse(self):
        rules = parse_weight_rules(specs = ["lang:fa=2", "source:Common Voice 22.0=0.5", "source:FSD50K/dev=0"])
        self.assertEqual(rules, [("lang", "fa", 2.0), ("source", "Common Voice 22.0", 0.5), ("source", "FSD50K/dev", 0.0)])
        for bad in ("lang=fa", "lang:fa", "lang:fa=-1"):
            with self.assertRaises(ValueError):
                parse_weight_rules(specs = [bad])

    def test_rules_multiply(self):
        rows = make_rows(count = 12)
        empty = [row["original"] == "" for row in rows]
        weights = sample_weights(
            rows = rows,
            empty = empty,
            rules = parse_weight_rules(specs = ["lang:fa=2", "source:Corpus 1.0=0.5"]),
        )
        for row, weight in zip(rows, weights):
            expected = (2.0 if row["lang"] == "fa" else 1.0) * (0.5 if row["source"] == "Corpus 1.0" else 1.0)
            self.assertEqual(weight, expected)


class ProportionTest(unittest.TestCase):
    def test_unit_weights_reproduce_v1_batches(self):
        rows = make_rows(count = 3000)
        durations = [row["duration_s"] for row in rows]
        plan = EpochPlan(
            durations = durations,
            weights = np.ones(shape = len(rows)),
            batch_size = 16,
            group_batches = 8,
            max_batch_seconds = 360,
            seed = 5,
        )
        for epoch in range(3):
            v1 = length_grouped_batches(
                durations = durations,
                batch_size = 16,
                seed = 5,
                epoch = epoch,
                group_batches = 8,
                max_batch_seconds = 360,
            )
            v2 = plan.batches(epoch = epoch)
            self.assertEqual([[key.index for key in batch] for batch in v2], v1)
            self.assertEqual([[key.position for key in batch] for batch in v2], v1)

    def test_occurrence_counts_follow_weights(self):
        weights = np.array(object = [2.0, 1.0, 0.5, 0.0, 1.25] * 400)
        epochs = 50
        counts = Counter()
        for epoch in range(epochs):
            occurrences = epoch_occurrences(weights = weights, seed = 1, epoch = epoch)
            per_row = np.bincount(occurrences, minlength = len(weights))
            # 整数部分是固定次数，小数部分最多多出一次。
            self.assertTrue(np.all(per_row >= np.floor(weights)))
            self.assertTrue(np.all(per_row <= np.ceil(weights)))
            for weight, count in zip(weights, per_row):
                counts[weight] += int(count)
        for weight in (2.0, 1.0, 0.5, 0.0, 1.25):
            mean = counts[weight] / (400 * epochs)
            self.assertAlmostEqual(mean, weight, delta = 0.02)

    def test_empty_share(self):
        rows = make_rows(count = 8000)
        empty = [row["original"] == "" for row in rows]
        for rules, share in (([], 0.03), (["lang:fa=3", "lang:zxx=0.1"], 0.03), ([], 0.10), ([], 0.0)):
            weights = sample_weights(
                rows = rows,
                empty = empty,
                rules = parse_weight_rules(specs = rules),
                empty_share = share,
            )
            composition = expected_composition(rows = rows, weights = weights, empty = empty)
            self.assertAlmostEqual(composition["empty"]["share"], share, places = 4)
            realized = []
            for epoch in range(20):
                occurrences = epoch_occurrences(weights = weights, seed = 2, epoch = epoch)
                realized.append(np.mean(a = np.asarray(a = empty)[occurrences]))
            self.assertAlmostEqual(float(np.mean(a = realized)), share, delta = 0.003)
        # 语言之间的相对权重保持：fa 是 en 的 3 倍。
        weights = sample_weights(
            rows = rows,
            empty = empty,
            rules = parse_weight_rules(specs = ["lang:fa=3"]),
            empty_share = 0.03,
        )
        composition = expected_composition(rows = rows, weights = weights, empty = empty)
        self.assertAlmostEqual(composition["fa"]["per_epoch"] / composition["en"]["per_epoch"], 3.0, places = 2)

    def test_no_empty_rows_is_unchanged(self):
        rows = [row for row in make_rows(count = 400) if row["original"]]
        weights = sample_weights(rows = rows, empty = [False] * len(rows), empty_share = 0.03)
        self.assertTrue(np.all(weights == 1.0))

    def test_deterministic(self):
        rows = make_rows(count = 2000)
        empty = [row["original"] == "" for row in rows]
        weights = sample_weights(rows = rows, empty = empty, empty_share = 0.03)

        def plan():
            return EpochPlan(
                durations = [row["duration_s"] for row in rows],
                weights = weights,
                batch_size = 16,
                group_batches = 8,
                max_batch_seconds = 360,
                seed = 9,
            )

        self.assertEqual(plan().batches(epoch = 2), plan().batches(epoch = 2))
        self.assertNotEqual(plan().batches(epoch = 2), plan().batches(epoch = 3))
        # 每个位置只出现一次，所以同一条样本在一轮中出现两次时混音也不同。
        positions = [key.position for key in flatten(batches = plan().batches(epoch = 0))]
        self.assertEqual(sorted(positions), list(range(len(positions))))


def collect(items):
    """测试用的批次构造：保留样本编号、位置、混音信息和波形摘要。"""
    return [
        (
            item["sample_id"],
            item["empty"],
            None if item["mix"] is None else round(number = item["mix"]["snr_db"], ndigits = 6),
            float(np.sum(a = item["waveform"])),
        )
        for item in items
    ]


class ResumeTest(unittest.TestCase):
    """中途停止后用保存的 (轮次, 已取出的微批数) 新建数据流，后续批次与不中断时逐项相同，包括混音结果。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        manifest = write_background(directory = cls.tmp.name, clips = [("noise", 1.0), ("music", 6.0)])
        rng = np.random.default_rng(seed = 0)
        durations = [round(float(rng.uniform(low = 0.5, high = 3.0)), 3) for _ in range(40)]
        rows = speech_rows(directory = cls.tmp.name, durations = durations, empty = tuple(range(30, 40)))
        mixer = BackgroundMixer(manifest = manifest, prob = 0.5)
        cls.dataset = TranscriptionDataset(
            rows = rows,
            target = "original",
            mixer = mixer,
            seed = 11,
        )
        cls.weights = sample_weights(
            rows = rows,
            empty = cls.dataset.empty,
            rules = parse_weight_rules(specs = ["sample_id:s-0=2.5", "sample_id:s-1=0"]),
            empty_share = 0.2,
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def stream(self, num_workers, epoch = 0, offset = 0):
        stream = BatchStream(
            dataset = self.dataset,
            collator = collect,
            batch_size = 4,
            group_batches = 2,
            max_batch_seconds = 8,
            seed = 11,
            num_workers = num_workers,
            weights = self.weights,
        )
        stream.epoch, stream.offset = epoch, offset
        return stream

    def run_stream(self, stream, count):
        batches, positions = [], []
        for _ in range(count):
            batches.append(stream.next())
            positions.append((stream.epoch, stream.offset))
        return batches, positions

    def test_resume_matches_uninterrupted(self):
        total = 3 * self.stream(num_workers = 0).batches_per_epoch
        reference, positions = self.run_stream(stream = self.stream(num_workers = 0), count = total)
        mixed = sum(entry[2] is not None for batch in reference for entry in batch)
        self.assertGreater(mixed, 0)
        # 在一轮中间、正好一轮结束处和第二轮中间停止，再分别续训。
        for stop in (5, self.stream(num_workers = 0).batches_per_epoch, total - 7):
            epoch, offset = positions[stop - 1]
            for num_workers in (0, 2):
                resumed, _ = self.run_stream(
                    stream = self.stream(num_workers = num_workers, epoch = epoch, offset = offset),
                    count = total - stop,
                )
                self.assertEqual(resumed, reference[stop:], msg = f"stop {stop}, workers {num_workers}")


if __name__ == "__main__":
    unittest.main()
