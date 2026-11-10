"""LibriSpeech 读取器：python -m unittest pipeline.realspeech.tests.test_sources"""

import json
import tempfile
import unittest
from pathlib import Path

from pipeline.realspeech.sources import SOURCES, librispeech, librispeech_coverage

PC_ROWS = [
    {"duration": 3.5, "audio_filepath": "train-clean-100/19/198/19-198-0000.flac", "text": "Hello, world.", "text_raw": "\"Hello, world.\""},
    {"duration": 2.0, "audio_filepath": "train-clean-100/19/198/19-198-0001.flac", "text": "Second one.", "text_raw": "Second one."},
    {"duration": 4.0, "audio_filepath": "train-clean-100/26/495/26-495-0000.flac", "text": "Not on disk.", "text_raw": "Not on disk."},
]


def build_tree(root: Path) -> Path:
    """19-198 有 3 句（0002 没有 PC 转写），26-495 的音频不在磁盘上。"""
    subset = root / "LibriSpeech" / "train-clean-100"
    chapter = subset / "19" / "198"
    chapter.mkdir(parents = True)
    for index in range(3):
        (chapter / f"19-198-000{index}.flac").write_bytes(b"")
    (chapter / "19-198.trans.txt").write_text("19-198-0000 HELLO WORLD\n19-198-0001 SECOND ONE\n19-198-0002 THIRD ONE\n", encoding = "utf-8")
    other = subset / "26" / "495"
    other.mkdir(parents = True)
    (other / "26-495.trans.txt").write_text("26-495-0000 NOT ON DISK\n", encoding = "utf-8")
    (root / "pc").mkdir()
    (root / "pc" / "train-clean-100.json").write_text("\n".join(json.dumps(obj = row) for row in PC_ROWS) + "\n", encoding = "utf-8")
    return subset


class LibriSpeechTest(unittest.TestCase):
    def test_registered(self) -> None:
        self.assertIs(SOURCES["librispeech"], librispeech)

    def test_rows_use_pc_text_and_skip_missing_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subset = build_tree(root = Path(tmp))
            rows = list(librispeech(paths = [str(subset)], lang = "en"))
            self.assertEqual([row["uid"] for row in rows], ["19-198-0000", "19-198-0001"])
            first = rows[0]
            self.assertEqual(first["transcript"], "Hello, world.")
            self.assertEqual(first["extra"]["text_raw"], "\"Hello, world.\"")
            self.assertEqual(first["speaker"], "19")
            self.assertEqual(first["duration_s"], 3.5)
            self.assertEqual(first["lang"], "en")
            self.assertEqual(first["audio"]["kind"], "file")
            self.assertTrue(first["audio"]["path"].endswith("train-clean-100/19/198/19-198-0000.flac"))

    def test_coverage_counts_utterances_without_pc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subset = build_tree(root = Path(tmp))
            coverage = librispeech_coverage(paths = [str(subset)])
            self.assertEqual(coverage, {"train-clean-100": {"original": 4, "with_pc": 3, "without_pc": 1}})


if __name__ == "__main__":
    unittest.main()
