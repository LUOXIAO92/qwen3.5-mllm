"""音频分片的读取（shards.py）、PackedRows 中的分片位置、续训与训练计划的核对：
python -m unittest pipeline.train.tests.test_shards"""

import contextlib
import io
import os
import pickle
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.train import muon, precision, shards
from pipeline.train.data import PackedRows
from pipeline.train.sampling import EpochPlan
from pipeline.train.train import (
    PLAN_KEYS,
    RESUME_KEYS,
    parse_args,
    plan_mismatches,
    resume_mismatches,
    run_config,
)

MTIME = 1_700_000_000


def write_tar(path, members):
    """members 是 [(成员名, 字节)]，用 tarfile 写出，返回成员表 [(名, 数据起点, 长度)]。"""
    with open(file = path, mode = "wb") as raw:
        with tarfile.open(
            fileobj = raw,
            mode = "w",
            format = shards.TAR_FORMAT,
            encoding = shards.TAR_ENCODING,
        ) as archive:
            for name, data in members:
                archive.addfile(
                    tarinfo = shards.member_info(name = name, size = len(data), mtime = MTIME),
                    fileobj = io.BytesIO(initial_bytes = data),
                )
    with tarfile.open(name = path, mode = "r:", encoding = shards.TAR_ENCODING) as archive:
        return [(member.name, member.offset_data, member.size) for member in archive]


class LayoutTest(unittest.TestCase):
    def test_predicted_offsets_match_tarfile(self):
        rng = np.random.default_rng(seed = 0)
        names = [
            "a.flac",
            "dir/" + "x" * 120 + ".flac",
            "长名字/" + "音" * 60 + ".flac",
            "real-batch2/worldspeech-ko/audio/" + "y" * 200 + ".flac",
            "b.flac",
        ]
        members = [(name, rng.bytes(int(rng.integers(low = 0, high = 3000)))) for name in names]
        locations, layout_shards = shards.layout(
            members = [(name, len(data)) for name, data in members],
            shard_bytes = 10 ** 9,
            mtime = MTIME,
        )
        self.assertEqual(len(layout_shards), 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "s.tar"
            table = write_tar(path = path, members = members)
            self.assertEqual([offset for _, offset in locations], [offset for _, offset, _ in table])
            self.assertEqual([name for name, _ in members], [name for name, _, _ in table])
            self.assertEqual(layout_shards[0], (len(members), path.stat().st_size))
            data = path.read_bytes()
            for (_, offset), (_, original) in zip(locations, members):
                self.assertEqual(data[offset:offset + len(original)], original)

    def test_split_into_shards(self):
        members = [(f"f{number}.flac", 3000) for number in range(10)]
        locations, layout_shards = shards.layout(members = members, shard_bytes = 20480, mtime = MTIME)
        self.assertGreater(len(layout_shards), 1)
        self.assertTrue(all(total <= 20480 for _, total in layout_shards))
        self.assertEqual(sum(count for count, _ in layout_shards), len(members))
        numbers = [number for number, _ in locations]
        self.assertEqual(numbers, sorted(numbers))


class ReaderTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.local = root / "local"
        self.drive = root / "drive"
        self.local.mkdir()
        self.drive.mkdir()
        self.header_shards = []
        self.payloads = {}
        for number in range(6):
            name = f"stream-{number:05d}.tar"
            payload = bytes([number]) * 1000 + os.urandom(500)
            (self.drive / name).write_bytes(payload)
            self.header_shards.append({"name": name, "kind": "pool" if number == 0 else "stream", "bytes": len(payload)})
            self.payloads[number] = payload

    def tearDown(self):
        self.directory.cleanup()

    def test_local_first_then_fallback(self):
        (self.local / self.header_shards[1]["name"]).write_bytes(self.payloads[1])
        reader = shards.ShardReader(shards = self.header_shards, dirs = [self.local], fallback = self.drive)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(new_target = stderr):
            self.assertEqual(reader.read(location = (1, 10, 20)), self.payloads[1][10:30])
            self.assertEqual(reader.read(location = [2, 1000, 500]), self.payloads[2][1000:])
            self.assertEqual(reader.read(location = (2, 0, 3)), self.payloads[2][:3])
        self.assertNotIn(self.header_shards[1]["name"], stderr.getvalue())
        self.assertEqual(stderr.getvalue().count("[shard-fallback]"), 1)
        reader.close()

    def test_incomplete_local_copy_is_ignored(self):
        (self.local / self.header_shards[3]["name"]).write_bytes(self.payloads[3][:100])
        reader = shards.ShardReader(shards = self.header_shards, dirs = [self.local], fallback = self.drive)
        with contextlib.redirect_stderr(new_target = io.StringIO()):
            self.assertEqual(reader.read(location = (3, 0, 1500)), self.payloads[3])

    def test_missing(self):
        reader = shards.ShardReader(shards = self.header_shards, dirs = [self.local])
        with self.assertRaises(FileNotFoundError):
            reader.read(location = (1, 0, 10))
        with self.assertRaises(FileNotFoundError):
            reader.read(location = None)

    def test_stream_descriptors_are_limited(self):
        reader = shards.ShardReader(shards = self.header_shards, dirs = [self.drive])
        for number in range(6):
            reader.read(location = (number, 0, 10))
        self.assertEqual(list(reader._pool), [0])
        self.assertEqual(list(reader._stream), list(range(1, 6))[-shards.STREAM_OPEN:])
        reader.close()

    def test_pickle_keeps_only_names(self):
        reader = shards.ShardReader(shards = self.header_shards, dirs = [self.drive])
        reader.read(location = (1, 0, 10))
        copy = pickle.loads(pickle.dumps(obj = reader))
        self.assertEqual(copy._stream, {})
        self.assertEqual(copy.read(location = (4, 5, 5)), self.payloads[4][5:10])
        reader.close()
        copy.close()


class PackedRowsTest(unittest.TestCase):
    def test_round_trip_with_location(self):
        rows = [
            {"sample_id": "a", "lang": "en", "audio_path": "/x/a.flac", "duration_s": 1.5, "audio_shard": (3, 1536, 4000)},
            {"sample_id": "b", "lang": "ko", "audio_path": "/x/b.flac", "duration_s": 9.71, "offset_s": 8.2425, "audio_shard": (7, 512, 1)},
            {"sample_id": "c", "lang": "zxx", "audio_path": "/x/c.flac", "duration_s": 2.0},
        ]
        with tempfile.TemporaryDirectory() as directory:
            packed = PackedRows(rows = rows, targets = ["t a", "t b", ""], path = Path(directory) / "rows.bin")
            restored = pickle.loads(pickle.dumps(obj = packed))
            for index, row in enumerate(rows):
                record, target = restored.record(index = index)
                self.assertEqual(record["audio_path"], row["audio_path"])
                self.assertEqual(record["offset_s"], row.get("offset_s"))
                expected = list(row["audio_shard"]) if "audio_shard" in row else None
                self.assertEqual(record["audio_shard"], expected)
            self.assertEqual(restored.record(index = 2)[1], "")


class ConfigTest(unittest.TestCase):
    def test_new_options_do_not_block_resume(self):
        local = [
            "--model-dir", "/repo/qwen3.5-4b-mllm",
            "--train-manifest", "/data/train/manifest-v2b.jsonl",
            "--target", "original",
            "--output-dir", "/work/data/T62/v2b-4b",
            "--batch-size", "16",
            "--max-duration-s", "30",
            "--max-batch-seconds", "360",
            "--noise-manifest", "/data/nonspeech/v2/background.jsonl",
        ]
        saved = run_config(args = parse_args(argv = local))
        for key in ("audio_shards", "audio_shard_dirs", "audio_shard_fallback", "rows_cache_dir"):
            self.assertNotIn(key, RESUME_KEYS)
            # 本机开始的训练保存的配置中还没有这几项
            saved.pop(key)
        colab = run_config(args = parse_args(argv = local + [
            "--model-dir", "/content/qwen3.5-asr/qwen3.5-4b-mllm",
            "--output-dir", "/content/drive/MyDrive/runs/v2b-4b",
            "--audio-shards", "/content/drive/MyDrive/shards/index.json",
            "--audio-shard-dirs", "/content/shards/window:/content/shards/pool",
            "--audio-shard-fallback", "/content/drive/MyDrive/shards",
            "--rows-cache-dir", "/content/rows",
            "--num-workers", "16",
        ]))
        self.assertEqual(resume_mismatches(saved = saved, config = colab), [])
        self.assertEqual(plan_mismatches(saved = {key: saved[key] for key in PLAN_KEYS}, config = colab), [])
        changed = run_config(args = parse_args(argv = local + ["--train-manifest", "/elsewhere/manifest-v2b.jsonl"]))
        self.assertEqual(resume_mismatches(saved = saved, config = changed), ["train_manifest"])
        self.assertEqual(plan_mismatches(saved = {key: saved[key] for key in PLAN_KEYS}, config = changed), ["train_manifest"])

    def test_plan_keys_are_resume_keys(self):
        self.assertTrue(set(PLAN_KEYS) <= set(RESUME_KEYS))
        noise = {key for key in RESUME_KEYS if key.startswith("noise_")}
        replay = {key for key in RESUME_KEYS if key.startswith("replay_")}
        # 混音、样本格式（--prompt-format）和训练范围、精度、优化器、自生成回放（#71，加在 ASR 批次之外）不改变每一步读哪些样本。
        self.assertEqual(
            set(RESUME_KEYS) - set(PLAN_KEYS),
            noise | replay | {"train_language_model", "prompt_format"} | set(precision.KEYS) | set(muon.KEYS),
        )
        self.assertEqual(len(replay), 7)


class DigestTest(unittest.TestCase):
    def test_digest_follows_plan(self):
        rows = [{"audio_path": f"/a/{number}.flac", "duration_s": 1.0 + number % 5} for number in range(40)]
        weights = np.ones(shape = len(rows))
        weights[:4] = 2.5

        def plan(seed):
            return EpochPlan(
                durations = [row["duration_s"] for row in rows],
                weights = weights,
                batch_size = 4,
                group_batches = 2,
                max_batch_seconds = 12,
                seed = seed,
            )

        def digest(seed, grad_accum = 1):
            return shards.plan_digest(
                plan = plan(seed = seed),
                rows = rows,
                epoch = 0,
                grad_accum = grad_accum,
            )

        first = digest(seed = 1)
        self.assertEqual(first, digest(seed = 1))
        self.assertNotEqual(first, digest(seed = 2))
        self.assertNotEqual(first, digest(seed = 1, grad_accum = 2))
        steps = [step for step, _ in shards.plan_batches(plan = plan(seed = 1), epoch = 1, grad_accum = 2)]
        first_batches = len(plan(seed = 1).position_batches(epoch = 0)[1])
        self.assertEqual(steps[0], first_batches // 2)
        self.assertEqual(steps, sorted(steps))


if __name__ == "__main__":
    unittest.main()
