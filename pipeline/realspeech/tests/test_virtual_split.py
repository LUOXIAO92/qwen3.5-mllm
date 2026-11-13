"""虚拟切分：python -m unittest pipeline.realspeech.tests.test_virtual_split（envs/mllm-eval）"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from pipeline.realspeech.materialize import SAMPLING_RATE, plan_virtual_split


def speech_with_pauses(rng, blocks: int) -> np.ndarray:
    """「说话 3 秒、停顿 1 秒」交替的音频。"""
    segments = []
    for _ in range(blocks):
        segments.append(0.3 * rng.standard_normal(size = 3 * SAMPLING_RATE))
        segments.append(0.001 * rng.standard_normal(size = SAMPLING_RATE))
    return np.concatenate(segments).astype(np.float32)


def no_pause(rng, seconds: int) -> np.ndarray:
    """电平连续起伏、没有不短于 0.25 秒的安静段的音频。"""
    t = np.arange(seconds * SAMPLING_RATE) / SAMPLING_RATE
    envelope = 0.2 + 0.15 * np.sin(2 * np.pi * 4 * t)
    return (envelope * rng.standard_normal(size = len(t))).astype(np.float32)


class VirtualSplitTest(unittest.TestCase):
    def test_cuts_at_pauses_and_slices_cover_whole_file(self):
        rng = np.random.default_rng(seed = 0)
        waves = {
            "audio/long_pauses.flac": speech_with_pauses(rng = rng, blocks = 7),
            # 28 秒没有停顿：电平每秒起伏 4 次，低谷都短于 0.25 秒，只能硬切，虚拟切分保持整段
            "audio/long_no_pause.flac": no_pause(rng = rng, seconds = 28),
            "audio/short.flac": speech_with_pauses(rng = rng, blocks = 2),
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            (out_dir / "audio").mkdir()
            for relative, wave in waves.items():
                sf.write(file = str(out_dir / relative), data = wave, samplerate = SAMPLING_RATE, format = "FLAC")
            durations = {relative: len(wave) / SAMPLING_RATE for relative, wave in waves.items()}
            mapping = plan_virtual_split(
                out_dir = out_dir,
                durations = durations,
                share = 1.0,
                max_seconds = 15,
                seed = 0,
                workers = 2,
            )
            # 只有长于 15 秒的文件参与抽样；短文件不出现在映射里
            self.assertEqual(sorted(mapping), ["audio/long_no_pause.flac", "audio/long_pauses.flac"])
            self.assertEqual(mapping["audio/long_no_pause.flac"], [])
            cuts = mapping["audio/long_pauses.flac"]
            self.assertGreater(len(cuts), 0)

            # 按切点在内存中截取的各段首尾相接，拼起来就是整段文件，每段不超过 15 秒
            whole, _ = sf.read(file = str(out_dir / "audio/long_pauses.flac"), dtype = "int16")
            bounds = [0.0, *cuts, durations["audio/long_pauses.flac"]]
            parts = []
            for begin, end in zip(bounds[:-1], bounds[1:]):
                self.assertLessEqual(end - begin, 15)
                part, _ = sf.read(
                    file = str(out_dir / "audio/long_pauses.flac"),
                    start = round(begin * SAMPLING_RATE),
                    frames = round((end - begin) * SAMPLING_RATE),
                    dtype = "int16",
                )
                parts.append(part)
            self.assertTrue(np.array_equal(np.concatenate(parts), whole))

            # 映射写在 virtual_split.jsonl，重跑时直接读取，不再重新抽样
            rows = [json.loads(s = line) for line in (out_dir / "virtual_split.jsonl").read_text(encoding = "utf-8").splitlines()]
            self.assertEqual({row["audio_path"]: row["cuts_s"] for row in rows}, mapping)
            again = plan_virtual_split(
                out_dir = out_dir,
                durations = {},
                share = 0.5,
                max_seconds = 15,
                seed = 1,
                workers = 2,
            )
            self.assertEqual(again, mapping)

    def test_share_of_long_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            (out_dir / "audio").mkdir()
            rng = np.random.default_rng(seed = 1)
            durations = {}
            for index in range(10):
                relative = f"audio/s{index}.flac"
                wave = speech_with_pauses(rng = rng, blocks = 5)
                sf.write(file = str(out_dir / relative), data = wave, samplerate = SAMPLING_RATE, format = "FLAC")
                durations[relative] = len(wave) / SAMPLING_RATE
            mapping = plan_virtual_split(
                out_dir = out_dir,
                durations = durations,
                share = 0.6,
                max_seconds = 15,
                seed = 0,
                workers = 2,
            )
            # 10 个长文件抽 6 个切分，其余 4 个保持整段
            self.assertEqual(len(mapping), 6)


if __name__ == "__main__":
    unittest.main()
