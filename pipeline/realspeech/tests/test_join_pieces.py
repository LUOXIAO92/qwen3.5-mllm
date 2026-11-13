"""拼回切分片段：python -m unittest pipeline.realspeech.tests.test_join_pieces（envs/mllm-eval）"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from pipeline.realspeech.join_pieces import join_one, plan
from pipeline.realspeech.materialize import SAMPLING_RATE, split_points


class JoinPiecesTest(unittest.TestCase):
    def test_join_restores_unsplit_audio(self):
        rng = np.random.default_rng(seed = 0)
        # 「说话、停顿、说话」交替的 28 秒音频，停顿处幅度很小
        segments = []
        for index in range(7):
            segments.append(0.3 * rng.standard_normal(size = 3 * SAMPLING_RATE))
            segments.append(0.001 * rng.standard_normal(size = SAMPLING_RATE))
        wave = np.concatenate(segments).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            audio_dir = Path(tmp)
            sf.write(file = str(audio_dir / "whole-reference.flac"), data = wave, samplerate = SAMPLING_RATE, format = "FLAC")
            spans = split_points(wave = wave, max_seconds = 15)
            self.assertGreater(len(spans), 1)
            for index, (begin, end) in enumerate(spans):
                sf.write(file = str(audio_dir / f"s1-p{index:02d}.flac"), data = wave[begin:end], samplerate = SAMPLING_RATE, format = "FLAC")
            jobs = plan(audio_dir = audio_dir, sample_ids = {"s1"})
            self.assertEqual(len(jobs), 1)
            sample_id, frames, error = join_one(job = jobs[0])
            self.assertIsNone(error)
            self.assertEqual(frames, len(wave))
            joined, _ = sf.read(file = str(audio_dir / "s1.flac"), dtype = "int16")
            reference, _ = sf.read(file = str(audio_dir / "whole-reference.flac"), dtype = "int16")
            # 拼回的整段与不切分直接写出的文件逐采样点相同，片段文件已删除
            self.assertTrue(np.array_equal(joined, reference))
            self.assertEqual(sorted(path.name for path in audio_dir.glob("s1-p*.flac")), [])
            # 重跑时没有片段可拼
            self.assertEqual(plan(audio_dir = audio_dir, sample_ids = {"s1"}), [])

    def test_rerun_after_interrupted_delete(self):
        """整段已就位、片段只删了一部分、还留着临时文件时，重跑只删除残留文件，整段不变。"""
        rng = np.random.default_rng(seed = 1)
        wave = (0.3 * rng.standard_normal(size = 5 * SAMPLING_RATE)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            audio_dir = Path(tmp)
            sf.write(file = str(audio_dir / "s2.flac"), data = wave, samplerate = SAMPLING_RATE, format = "FLAC")
            sf.write(file = str(audio_dir / "s2-p01.flac"), data = wave[:SAMPLING_RATE], samplerate = SAMPLING_RATE, format = "FLAC")
            sf.write(file = str(audio_dir / "s3.tmp.flac"), data = wave[:SAMPLING_RATE], samplerate = SAMPLING_RATE, format = "FLAC")
            before, _ = sf.read(file = str(audio_dir / "s2.flac"), dtype = "int16")
            jobs = plan(audio_dir = audio_dir, sample_ids = {"s2", "s3"})
            self.assertEqual(len(jobs), 1)
            sample_id, frames, error = join_one(job = jobs[0])
            self.assertIsNone(error)
            self.assertEqual(frames, len(wave))
            after, _ = sf.read(file = str(audio_dir / "s2.flac"), dtype = "int16")
            self.assertTrue(np.array_equal(before, after))
            self.assertEqual(sorted(path.name for path in audio_dir.iterdir()), ["s2.flac"])


if __name__ == "__main__":
    unittest.main()
