"""无人声样本集的切段与划分：python -m unittest pipeline.realspeech.tests.test_negatives"""

import random
import unittest

from pipeline.realspeech.negatives import categorize, eval_quotas, plan_segments, split_recordings


def make_recording(rec_id: str, category: str, fsd_split, uploader = None, duration_s: float = 10.0) -> dict:
    return {
        "recording_id": rec_id,
        "category": category,
        "fsd_split": fsd_split,
        "uploader": uploader,
        "duration_s": duration_s,
        "path": f"/x/{rec_id}.wav",
    }


class PlanTest(unittest.TestCase):
    def test_short_dropped_and_medium_kept_whole(self):
        rng = random.Random(0)
        self.assertEqual(plan_segments(duration_s = 1.5, rng = rng, min_s = 2.0), [])
        self.assertEqual(plan_segments(duration_s = 12.0, rng = rng), [(0.0, 12.0)])

    def test_long_recording_is_cut_without_overlap(self):
        pieces = plan_segments(duration_s = 400.0, rng = random.Random(1), max_per_recording = 50)
        self.assertGreater(len(pieces), 10)
        for (start, end), (next_start, _) in zip(pieces, pieces[1:]):
            self.assertLessEqual(end, next_start + 1e-6)
        for start, end in pieces:
            self.assertGreaterEqual(end - start, 2.0)
            self.assertLessEqual(end - start, 30.0)

    def test_cap_per_recording(self):
        pieces = plan_segments(duration_s = 900.0, rng = random.Random(2), max_per_recording = 5)
        self.assertEqual(len(pieces), 5)


class CategoryTest(unittest.TestCase):
    def test_priority(self):
        self.assertEqual(categorize(labels = ["Guitar", "Musical_instrument", "Music"]), "music")
        self.assertEqual(categorize(labels = ["Car", "Vehicle"]), "vehicle")
        self.assertEqual(categorize(labels = ["noise"]), "noise")
        self.assertEqual(categorize(labels = ["Squeak"]), "other")


class QuotaTest(unittest.TestCase):
    def test_total_and_caps(self):
        quotas = eval_quotas(available = {"a": 5000, "b": 100, "c": 10}, total = 300)
        self.assertEqual(quotas["c"], 10)
        self.assertLessEqual(quotas["b"], 100)
        self.assertGreaterEqual(sum(quotas.values()), 290)
        self.assertLessEqual(sum(quotas.values()), 300)


class SplitTest(unittest.TestCase):
    def test_no_recording_or_uploader_overlap(self):
        recordings = []
        for index in range(300):
            recordings.append(make_recording(rec_id = f"d{index}", category = "music", fsd_split = "dev", uploader = f"u{index % 40}"))
        for index in range(300):
            recordings.append(make_recording(rec_id = f"e{index}", category = "animal", fsd_split = "eval", uploader = f"v{index % 25}"))
        plans = {rec["recording_id"]: [(0.0, 10.0)] for rec in recordings}
        eval_segments, train_segments = split_recordings(
            recordings = recordings,
            plans = plans,
            plans_eval = plans,
            eval_size = 50,
            musan_eval_percent = 20.0,
            seed = 0,
        )
        self.assertEqual(len(eval_segments), 50)
        eval_ids = {rec["recording_id"] for rec, _, _ in eval_segments}
        train_ids = {rec["recording_id"] for rec, _, _ in train_segments}
        self.assertFalse(eval_ids & train_ids)
        eval_uploaders = {rec["uploader"] for rec, _, _ in eval_segments}
        self.assertFalse(eval_uploaders & {rec["uploader"] for rec, _, _ in train_segments})


if __name__ == "__main__":
    unittest.main()
