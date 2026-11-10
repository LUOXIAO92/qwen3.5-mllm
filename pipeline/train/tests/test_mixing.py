"""实时混音的信噪比、比例与确定性：python -m unittest pipeline.train.tests.test_mixing"""

import tempfile
import unittest

import numpy as np

from pipeline.train.data import STREAM_MIXING, SampleKey, TranscriptionDataset
from pipeline.train.mixing import BackgroundMixer, fit_length, mix_at_snr, parse_category_weights
from pipeline.train.tests.fixtures import noise, speech_rows, tone, write_background


def snr_db(signal, residual):
    return 10 * np.log10(np.mean(a = np.square(signal, dtype = np.float64)) / np.mean(a = np.square(residual, dtype = np.float64)))


class MixAtSnrTest(unittest.TestCase):
    def test_realized_snr_matches_target(self):
        speech = tone(seconds = 3.0)
        background = noise(seconds = 3.0, seed = 0, amplitude = 0.3)
        for target in (0.0, 5.0, 12.5, 20.0):
            mixed, scale, gain = mix_at_snr(speech = speech, noise = background, snr_db = target)
            self.assertEqual(gain, 1.0)
            self.assertAlmostEqual(snr_db(signal = speech, residual = mixed - speech), target, places = 3)

    def test_peak_limit_keeps_snr(self):
        speech = tone(seconds = 2.0, amplitude = 0.9)
        background = noise(seconds = 2.0, seed = 1, amplitude = 0.5)
        mixed, scale, gain = mix_at_snr(speech = speech, noise = background, snr_db = 0.0)
        self.assertLess(gain, 1.0)
        self.assertLessEqual(float(np.max(a = np.abs(mixed))), 0.99 + 1e-6)
        self.assertAlmostEqual(snr_db(signal = gain * speech, residual = mixed - gain * speech), 0.0, places = 3)

    def test_silence_is_not_mixed(self):
        speech = np.zeros(shape = 16000, dtype = np.float32)
        mixed, scale, gain = mix_at_snr(speech = speech, noise = noise(seconds = 1.0, seed = 2), snr_db = 10.0)
        self.assertIsNone(scale)
        self.assertIs(mixed, speech)

    def test_fit_length_wraps_short_noise(self):
        clip = np.arange(5, dtype = np.float32)
        self.assertEqual(fit_length(noise = clip, length = 7, start = 3).tolist(), [3, 4, 0, 1, 2, 3, 4])
        self.assertEqual(fit_length(noise = clip, length = 2, start = 1).tolist(), [1, 2])


class MixerDrawTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.manifest = write_background(
            directory = cls.tmp.name,
            clips = [("music", 1.5), ("music", 8.0), ("noise", 2.5), ("vehicle", 12.0)],
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def draws(self, mixer, count = 20000):
        return [mixer.draw(rng = np.random.default_rng(seed = [7, 0, STREAM_MIXING, position])) for position in range(count)]

    def test_prob_and_uniform_snr(self):
        mixer = BackgroundMixer(manifest = self.manifest, prob = 0.3)
        draws = self.draws(mixer = mixer)
        mixed = np.array(object = [draw[0] for draw in draws])
        snrs = np.array(object = [draw[2] for draw in draws])
        self.assertAlmostEqual(mixed.mean(), 0.3, delta = 0.015)
        self.assertGreaterEqual(snrs.min(), 0.0)
        self.assertLessEqual(snrs.max(), 20.0)
        # 0–20 dB 均匀分布：四个 5 dB 区间各约 25%。
        counts, _ = np.histogram(a = snrs, bins = 4, range = (0.0, 20.0))
        for share in counts / len(snrs):
            self.assertAlmostEqual(share, 0.25, delta = 0.015)

    def test_category_weights(self):
        mixer = BackgroundMixer(
            manifest = self.manifest,
            category_weights = parse_category_weights(specs = ["music=0.5", "vehicle=0"]),
        )
        self.assertEqual(len(mixer), 3)
        draws = self.draws(mixer = mixer)
        music = np.mean(a = [mixer.categories[draw[1]] == "music" for draw in draws])
        # 两段音乐各 0.5、一段 noise 1：音乐的抽中概率是 1 / 2。
        self.assertAlmostEqual(music, 0.5, delta = 0.015)
        self.assertAlmostEqual(mixer.category_shares()["music"]["share"], 0.5, places = 6)
        with self.assertRaises(ValueError):
            BackgroundMixer(manifest = self.manifest, category_weights = {"speech": 1.0})


class DatasetMixingTest(unittest.TestCase):
    """经过 TranscriptionDataset 的完整路径：解码、截取或循环背景声、按信噪比相加。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        # 背景声有比语音短的（循环接上），也有比语音长的（只解码需要的一段）。
        cls.manifest = write_background(directory = cls.tmp.name, clips = [("noise", 1.2), ("music", 20.0)])
        cls.rows = speech_rows(directory = cls.tmp.name, durations = [3.0, 4.5, 2.0], empty = (2,))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def dataset(self, prob):
        mixer = BackgroundMixer(manifest = self.manifest, prob = prob)
        return TranscriptionDataset(
            rows = self.rows,
            target = "original",
            mixer = mixer,
            seed = 3,
        )

    def test_snr_of_mixed_samples(self):
        dataset = self.dataset(prob = 1.0)
        clean = TranscriptionDataset(rows = self.rows, target = "original")
        categories = set()
        for position in range(40):
            key = SampleKey(index = position % 2, epoch = 0, position = position)
            item = dataset[key]
            reference = clean[key.index]["waveform"]
            self.assertIsNotNone(item["mix"])
            self.assertEqual(len(item["waveform"]), len(reference))
            gain = item["mix"]["gain"]
            realized = snr_db(signal = gain * reference, residual = item["waveform"] - gain * reference)
            self.assertAlmostEqual(realized, item["mix"]["snr_db"], delta = 0.01)
            self.assertTrue(0.0 <= item["mix"]["snr_db"] <= 20.0)
            categories.add(item["mix"]["category"])
        self.assertEqual(categories, {"noise", "music"})

    def test_empty_target_and_plain_index_are_not_mixed(self):
        dataset = self.dataset(prob = 1.0)
        for position in range(10):
            item = dataset[SampleKey(index = 2, epoch = 0, position = position)]
            self.assertTrue(item["empty"])
            self.assertIsNone(item["mix"])
        self.assertIsNone(dataset[0]["mix"])

    def test_same_key_same_waveform(self):
        first = self.dataset(prob = 0.5)
        second = self.dataset(prob = 0.5)
        mixed = 0
        for position in range(20):
            key = SampleKey(index = 1, epoch = 4, position = position)
            a, b = first[key], second[key]
            self.assertTrue(np.array_equal(a["waveform"], b["waveform"]))
            self.assertEqual(a["mix"], b["mix"])
            mixed += a["mix"] is not None
        self.assertTrue(0 < mixed < 20)


if __name__ == "__main__":
    unittest.main()
