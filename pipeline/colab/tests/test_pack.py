"""按训练计划打包分片、核对、窗口管理和清单打包的测试：python -m unittest discover -s pipeline/colab/tests -t .

真实音频的测试读取环境变量 SHARD_TEST_MANIFEST 指向的训练清单（只读，取其中几行），没有设置时跳过。
"""

import contextlib
import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.colab import bench_storage, bundle, pack_shards, shard_window
from pipeline.train import shards
from pipeline.train.data import TranscriptionDataset, read_manifest
from pipeline.train.tests.fixtures import tone, write_flac
from pipeline.train.train import worker_context

# 训练参数：每批 4 条，空目标样本占每轮 30%（被多抽好几次，进常驻分片）。
TRAIN_ARGS = [
    "--target", "original",
    "--batch-size", "4",
    "--group-batches", "2",
    "--max-batch-seconds", "16",
    "--empty-target-share", "0.3",
    "--seed", "7",
]


def write_project(directory):
    """写出一份小训练清单：12 条语音、一段录音虚拟切成两段、3 条空目标样本。返回清单路径。"""
    directory = Path(directory)
    audio = directory / "audio"
    (audio / "speech").mkdir(parents = True)
    (audio / "nonspeech").mkdir(parents = True)
    rows = []
    for number in range(12):
        path = audio / "speech" / f"s-{number:02d}.flac"
        seconds = 1.0 + (number % 4) * 0.5
        write_flac(path = path, waveform = tone(seconds = seconds, frequency = 200.0 + 13 * number))
        rows.append({"sample_id": f"s-{number}", "audio_path": str(path), "duration_s": seconds, "lang": "en", "original": f"sentence {number}", "script": f"sentence {number}"})
    long_path = audio / "speech" / "long.flac"
    write_flac(path = long_path, waveform = tone(seconds = 5.0, frequency = 330.0))
    for piece, (offset, seconds) in enumerate(((0.0, 2.25), (2.5, 2.4))):
        rows.append({"sample_id": f"long-p{piece}", "audio_path": str(long_path), "duration_s": seconds, "offset_s": offset, "lang": "en", "original": f"piece {piece}", "script": f"piece {piece}"})
    for number in range(3):
        path = audio / "nonspeech" / f"n-{number}.flac"
        rng = np.random.default_rng(seed = number)
        write_flac(path = path, waveform = (0.05 * rng.standard_normal(size = 16000 * (number + 1))).astype(np.float32))
        rows.append({"sample_id": f"n-{number}", "audio_path": str(path), "duration_s": float(number + 1), "lang": "zxx", "original": "", "script": ""})
    manifest = directory / "manifest.jsonl"
    manifest.write_text(data = "".join(json.dumps(obj = row) + "\n" for row in rows), encoding = "utf-8")
    return manifest


def quiet(function, **kwargs):
    with contextlib.redirect_stdout(new_target = io.StringIO()):
        return function(**kwargs)


class PackTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.manifest = write_project(directory = self.root / "data")
        self.train_args = ["--train-manifest", str(self.manifest)] + TRAIN_ARGS
        self.settings = pack_shards.plan_settings(train_args = self.train_args)
        self.out = self.root / "shards"
        self.index = quiet(
            pack_shards.build_index,
            settings = self.settings,
            out_dir = self.out,
            name = "index",
            shard_bytes = 120_000,
            threads = 4,
        )
        self.header = shards.read_header(index_path = self.index)
        self.rows, self.plan = pack_shards.training_plan(settings = self.settings)

    def tearDown(self):
        self.directory.cleanup()

    def reads(self):
        return [
            (step, self.rows[index]["audio_path"])
            for step, indices in shards.plan_batches(plan = self.plan, epoch = 0, grad_accum = 1)
            for index in indices
        ]

    def test_index_follows_plan(self):
        reads = self.reads()
        counts = {}
        for _, path in reads:
            counts[path] = counts.get(path, 0) + 1
        locations = shards.load_locations(header = self.header)
        self.assertEqual(set(locations), set(counts))
        kinds = [shard["kind"] for shard in self.header["shards"]]
        for path, count in counts.items():
            self.assertEqual(kinds[locations[path][0]], "pool" if count > 1 else "stream", path)
        # 空目标样本和虚拟切分共用的录音都读了不止一次
        pool_files = {path for path, count in counts.items() if count > 1}
        self.assertTrue(any("nonspeech" in path for path in pool_files))
        self.assertTrue(any(path.endswith("long.flac") for path in pool_files))
        # 顺序分片按步数排列，每个文件落在它所在分片的步数范围内
        stream = [shard for shard in self.header["shards"] if shard["kind"] == "stream"]
        self.assertGreater(len(stream), 1)
        for earlier, later in zip(stream, stream[1:]):
            self.assertLessEqual(earlier["last_step"], later["first_step"])
        for step, path in reads:
            number = locations[path][0]
            if kinds[number] == "stream":
                self.assertLessEqual(self.header["shards"][number]["first_step"], step)
                self.assertGreaterEqual(self.header["shards"][number]["last_step"], step)
        self.assertEqual(
            self.header["plan"]["digest"],
            shards.plan_digest(
                plan = self.plan,
                rows = self.rows,
                epoch = 0,
                grad_accum = 1,
            ),
        )
        self.assertEqual(self.header["plan"]["params"]["train_manifest"], str(self.manifest))

    def test_pack_in_ranges_and_verify(self):
        count = len(self.header["shards"])
        pool = [number for number, shard in enumerate(self.header["shards"]) if shard["kind"] == "pool"]
        quiet(pack_shards.main, argv = ["pack", "--index", str(self.index), "--pool"])
        quiet(pack_shards.main, argv = ["pack", "--index", str(self.index), "--shards", f"{len(pool)}-{len(pool)}"])
        first = self.out / self.header["shards"][len(pool)]["name"]
        mtime = first.stat().st_mtime_ns
        output = io.StringIO()
        with contextlib.redirect_stdout(new_target = output):
            pack_shards.main(argv = ["pack", "--index", str(self.index), "--shards", f"0-{count - 1}"])
        self.assertIn(f"跳过 {len(pool) + 1} 个已完整的分片", output.getvalue())
        self.assertEqual(first.stat().st_mtime_ns, mtime)
        quiet(pack_shards.main, argv = ["verify", "--index", str(self.index), "--samples", "1000"])
        # 分片是标准 tar：解开的文件与原文件逐字节相同
        locations = shards.load_locations(header = self.header)
        for path, (number, offset, size) in locations.items():
            with tarfile.open(name = self.out / self.header["shards"][number]["name"], mode = "r:") as archive:
                member = archive.getmember(name = shards.member_name(path = path, root = self.header["member_root"]))
                self.assertEqual((member.offset_data, member.size), (offset, size))
                self.assertEqual(archive.extractfile(member = member).read(), Path(path).read_bytes())

    def test_verify_detects_changed_bytes(self):
        quiet(pack_shards.main, argv = ["pack", "--index", str(self.index), "--shards", f"0-{len(self.header['shards']) - 1}"])
        path, (number, offset, size) = next(iter(shards.load_locations(header = self.header).items()))
        target = self.out / self.header["shards"][number]["name"]
        data = bytearray(target.read_bytes())
        data[offset + size // 2] ^= 0xFF
        target.write_bytes(data)
        with self.assertRaises(SystemExit):
            quiet(pack_shards.main, argv = ["verify", "--index", str(self.index), "--shards", str(number), "--samples", "1000"])

    def test_dataset_reads_identical_waveforms(self):
        quiet(pack_shards.main, argv = ["pack", "--index", str(self.index), "--shards", f"0-{len(self.header['shards']) - 1}"])
        rows = read_manifest(path = self.manifest)
        plain = TranscriptionDataset(rows = [dict(row) for row in rows], target = "original")
        reader, summary = shards.attach_shards(rows = rows, header = self.header, dirs = [self.out])
        self.assertEqual(summary["rows_in_index"], len(rows))
        packed = TranscriptionDataset(
            rows = rows,
            target = "original",
            storage = self.root / "cache" / "rows.jsonl.bin",
            shards = reader,
        )
        self.assertTrue(any(row.get("offset_s") for row in rows))
        for index in range(len(rows)):
            np.testing.assert_array_equal(actual = packed[index]["waveform"], desired = plain[index]["waveform"])
        # 数据读取进程（forkserver）中同样读出
        import torch

        loader = torch.utils.data.DataLoader(
            dataset = packed,
            batch_size = None,
            num_workers = 2,
            multiprocessing_context = worker_context(num_workers = 2),
        )
        for index, item in enumerate(loader):
            np.testing.assert_array_equal(actual = np.asarray(item["waveform"]), desired = plain[index]["waveform"])

    def test_path_order(self):
        index = quiet(
            pack_shards.build_index,
            settings = self.settings,
            out_dir = self.root / "by-path",
            name = "index",
            order = "path",
            shard_bytes = 120_000,
            threads = 4,
        )
        header = shards.read_header(index_path = index)
        self.assertEqual(header["order"], "path")
        self.assertIsNone(header["plan"])
        self.assertEqual({shard["kind"] for shard in header["shards"]}, {"path"})
        paths = [path for path, _, _, _ in shards.iter_entries(header = header)]
        self.assertEqual(paths, sorted({row["audio_path"] for row in read_manifest(path = self.manifest)}))
        quiet(pack_shards.main, argv = ["pack", "--index", str(index), "--shards", "all"])
        quiet(pack_shards.main, argv = ["verify", "--index", str(index), "--samples", "1000"])
        rows = read_manifest(path = self.manifest)
        plain = TranscriptionDataset(rows = [dict(row) for row in rows], target = "original")
        reader, summary = shards.attach_shards(rows = rows, header = header, dirs = [self.root / "by-path"])
        self.assertEqual(summary["path_in_dirs"], f"{len(header['shards'])}/{len(header['shards'])}")
        packed = TranscriptionDataset(rows = rows, target = "original", shards = reader)
        for number in range(len(rows)):
            np.testing.assert_array_equal(actual = packed[number]["waveform"], desired = plain[number]["waveform"])
        with self.assertRaises(ValueError):
            quiet(
                pack_shards.build_index,
                settings = self.settings,
                out_dir = self.root / "by-path",
                name = "bad",
                order = "path",
                steps = "0-10",
            )

    def test_step_range_index(self):
        stream = [shard for shard in self.header["shards"] if shard["kind"] == "stream"]
        first, stop = stream[1]["first_step"], stream[1]["last_step"] + 1
        part = quiet(
            pack_shards.build_index,
            settings = self.settings,
            out_dir = self.out,
            name = "part",
            steps = f"{first}-{stop}",
            shard_bytes = 120_000,
            threads = 4,
        )
        header = shards.read_header(index_path = part)
        self.assertEqual(header["plan"]["steps"], [first, stop])
        self.assertEqual(header["plan"]["digest"], self.header["plan"]["digest"])
        self.assertEqual(
            [shard for shard in header["shards"] if shard["kind"] == "stream"],
            stream,
        )
        needed = {path for step, path in self.reads() if first <= step < stop}
        full = shards.load_locations(header = self.header)
        pool_kinds = {number for number, shard in enumerate(self.header["shards"]) if shard["kind"] == "pool"}
        expected_pool = {path for path in needed if full[path][0] in pool_kinds}
        locations = shards.load_locations(header = header)
        kinds = [shard["kind"] for shard in header["shards"]]
        self.assertEqual({path for path, location in locations.items() if kinds[location[0]] == "pool"}, expected_pool)
        self.assertTrue(all(shard["name"].startswith(f"pool-{first}-{stop}-") for shard in header["shards"] if shard["kind"] == "pool"))
        # 选中的分片：这段步数的顺序分片加常驻分片
        chosen = pack_shards.select_shards(header = header, pool = True, steps = f"{first}-{stop}")
        self.assertIn(len([kind for kind in kinds if kind == "pool"]) + 1, chosen)
        output = io.StringIO()
        with contextlib.redirect_stdout(new_target = output):
            pack_shards.main(argv = ["list", "--index", str(part), "--pool", "--steps", f"{first}-{stop}", "--prefix", "/d/"])
        self.assertEqual(output.getvalue().split(), [f"/d/{header['shards'][number]['name']}" for number in chosen])
        # 两个索引生成的同名顺序分片逐字节相同
        quiet(pack_shards.main, argv = ["pack", "--index", str(self.index), "--steps", f"{first}-{stop}"])
        name = stream[1]["name"]
        full_bytes = (self.out / name).read_bytes()
        (self.out / name).unlink()
        quiet(pack_shards.main, argv = ["pack", "--index", str(part), "--steps", f"{first}-{stop}"])
        self.assertEqual((self.out / name).read_bytes(), full_bytes)


class BenchTest(unittest.TestCase):
    def test_local_dir(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = pack_shards.plan_settings(train_args = ["--train-manifest", str(write_project(directory = root / "data"))] + TRAIN_ARGS)
            index = quiet(
                pack_shards.build_index,
                settings = settings,
                out_dir = root / "shards",
                name = "index",
                shard_bytes = 120_000,
                threads = 4,
            )
            quiet(pack_shards.main, argv = ["pack", "--index", str(index), "--shards", "all"])
            results = quiet(
                bench_storage.main,
                argv = [
                    "--index", str(index),
                    "--dir", f"local={root / 'shards'}",
                    "--dir", f"empty={root / 'nothing'}",
                    "--copy-shards", "2",
                    "--reads", "10",
                    "--threads", "4",
                    "--decode",
                ],
            )
        tests = {(result.get("dir"), result["test"]): result for result in results}
        self.assertIn(("local", "copy"), tests)
        self.assertEqual(tests[("local", "random")]["reads"], 10)
        self.assertIn("decode_ms_mean", tests[("local", "random")])
        self.assertIn("error", tests[("empty", "copy")])


class WindowTest(unittest.TestCase):
    """用小分片模拟训练进度，检查复制和删除的顺序。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.source = root / "drive"
        self.output = root / "run"
        self.local = root / "local"
        for directory in (self.source, self.output):
            directory.mkdir()
        shard_list = [{"name": "pool-00000.tar", "kind": "pool", "bytes": 100}]
        for number in range(8):
            shard_list.append({
                "name": f"stream-{number:05d}.tar",
                "kind": "stream",
                "bytes": 100,
                "first_step": number * 10,
                "last_step": number * 10 + 9,
            })
        for shard in shard_list:
            (self.source / shard["name"]).write_bytes(b"x" * shard["bytes"])
        self.header = {"order": "plan", "shards": shard_list, "plan": {"steps": [0, 80]}}

    def tearDown(self):
        self.directory.cleanup()

    def window(self, ahead_steps, budget):
        return shard_window.Window(
            header = self.header,
            source = self.source,
            pool_dir = self.local / "pool",
            window_dir = self.local / "window",
            ahead_steps = ahead_steps,
            budget_bytes = budget,
            margin_steps = 5,
            keep_free_bytes = 0,
        )

    def run_ticks(self, window, floor, position):
        with contextlib.redirect_stdout(new_target = io.StringIO()):
            while window.tick(floor = floor, position = position)[0]:
                pass

    def local_names(self):
        return sorted(path.name for path in (self.local / "window").glob(pattern = "*.tar"))

    def test_priorities_and_deletion(self):
        window = self.window(ahead_steps = 20, budget = 500)
        self.run_ticks(window = window, floor = 0, position = 0)
        self.assertTrue((self.local / "pool" / "pool-00000.tar").exists())
        # 第一优先覆盖第 0–19 步（2 个分片），第二优先在 500 字节预算内再取 2 个
        self.assertEqual(self.local_names(), [f"stream-{number:05d}.tar" for number in range(4)])
        self.assertEqual(window.lead(position = 0), 40)
        # 进度到第 33 步、检查点在第 30 步：最后一步早于 min(30, 33 - 5) = 28 的分片删除
        self.run_ticks(window = window, floor = 30, position = 33)
        self.assertEqual(self.local_names(), [f"stream-{number:05d}.tar" for number in range(2, 6)])
        # 检查点还停在第 0 步时不删除
        window = self.window(ahead_steps = 20, budget = 10 ** 6)
        self.run_ticks(window = window, floor = 0, position = 60)
        self.assertIn("stream-00002.tar", self.local_names())

    def test_priority_two_waits_for_priority_one(self):
        (self.source / "stream-00001.tar").unlink()
        window = self.window(ahead_steps = 20, budget = 10 ** 6)
        self.run_ticks(window = window, floor = 0, position = 0)
        self.assertEqual(self.local_names(), ["stream-00000.tar"])
        self.assertEqual(window.lead(position = 0), 10)

    def test_prepare_reads_checkpoint(self):
        (self.output / "latest").write_text(data = "checkpoint-0000030\n", encoding = "utf-8")
        (self.output / "progress.json").write_text(data = json.dumps(obj = {"step": 41, "status": "stopped"}), encoding = "utf-8")
        args = [
            "--index", "unused",
            "--source", str(self.source),
            "--pool-dir", str(self.local / "pool"),
            "--window-dir", str(self.local / "window"),
            "--output-dir", str(self.output),
            "--prepare-steps", "15",
        ]
        header = dict(self.header, index_path = "unused")
        original = shard_window.read_header
        shard_window.read_header = lambda index_path: header
        try:
            with contextlib.redirect_stdout(new_target = io.StringIO()):
                shard_window.main(argv = ["prepare"] + args)
        finally:
            shard_window.read_header = original
        # 检查点第 30 步：复制第 30–44 步的顺序分片（下标 3、4）
        self.assertEqual(self.local_names(), ["stream-00003.tar", "stream-00004.tar"])


class BundleTest(unittest.TestCase):
    def test_relative_paths_after_moving(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "work"
            audio = root / "data" / "T02" / "en" / "dev"
            audio.mkdir(parents = True)
            (root / "data" / "T10").mkdir(parents = True)
            rows = []
            for number in range(3):
                path = audio / f"{number}.flac"
                write_flac(path = path, waveform = tone(seconds = 0.5))
                rows.append({"sample_id": f"en-{number}", "audio_path": str(path), "duration_s": 0.5, "lang": "en"})
            manifest = root / "data" / "T10" / "fleurs_dev.jsonl"
            manifest.write_text(data = "".join(json.dumps(obj = row) + "\n" for row in rows), encoding = "utf-8")
            archive = Path(directory) / "dev.tar"
            bundle.write_bundle(manifest = manifest, root = root, output = archive)
            elsewhere = Path(directory) / "colab"
            with tarfile.open(name = archive) as handle:
                handle.extractall(path = elsewhere, filter = "data")
            moved = read_manifest(path = elsewhere / "data" / "T10" / "fleurs_dev.jsonl")
            for row, original in zip(moved, rows):
                self.assertTrue(row["audio_path"].startswith(str(elsewhere)))
                self.assertEqual(Path(row["audio_path"]).read_bytes(), Path(original["audio_path"]).read_bytes())

    def test_relative_list_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            listing = root / "nonspeech" / "v2"
            (listing / "train" / "audio").mkdir(parents = True)
            write_flac(path = listing / "train" / "audio" / "a.flac", waveform = tone(seconds = 0.5))
            text = json.dumps(obj = {"audio_path": "train/audio/a.flac", "duration_s": 0.5, "category": "music"}) + "\n"
            (listing / "background.jsonl").write_text(data = text, encoding = "utf-8")
            archive = Path(directory) / "noise.tar"
            bundle.write_bundle(manifest = listing / "background.jsonl", root = root, output = archive)
            with tarfile.open(name = archive) as handle:
                names = handle.getnames()
                self.assertEqual(handle.extractfile(member = "nonspeech/v2/background.jsonl").read().decode(), text)
            self.assertEqual(names, ["nonspeech/v2/background.jsonl", "nonspeech/v2/train/audio/a.flac"])


@unittest.skipUnless(os.environ.get("SHARD_TEST_MANIFEST"), "设置 SHARD_TEST_MANIFEST 后用真实音频测试")
class RealAudioTest(unittest.TestCase):
    """从真实训练清单中取几行（含虚拟切分的片段），打包后读出的字节和解码波形都与原文件相同。"""

    def test_real_files(self):
        source = Path(os.environ["SHARD_TEST_MANIFEST"])
        plain_rows = []
        split_rows = []
        with open(file = source, mode = "r", encoding = "utf-8") as handle:
            for line in handle:
                # 只解析需要的行：清单有几百万行，带 offset_s 的行在后面
                split = '"offset_s"' in line
                target = split_rows if split else plain_rows
                if len(target) < 6:
                    row = json.loads(s = line)
                    if (row.get("offset_s") is not None) == split:
                        target.append(row)
                if len(plain_rows) >= 6 and len(split_rows) >= 6:
                    break
        self.assertTrue(split_rows, "清单中没有找到带 offset_s 的行")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                data = "".join(json.dumps(obj = row, ensure_ascii = False) + "\n" for row in plain_rows + split_rows),
                encoding = "utf-8",
            )
            settings = pack_shards.plan_settings(train_args = ["--train-manifest", str(manifest), "--target", "original", "--batch-size", "2"])
            index = quiet(
                pack_shards.build_index,
                settings = settings,
                out_dir = root / "shards",
                name = "index",
                shard_bytes = 2_000_000,
                threads = 4,
            )
            header = shards.read_header(index_path = index)
            quiet(pack_shards.main, argv = ["pack", "--index", str(index), "--shards", f"0-{len(header['shards']) - 1}"])
            quiet(pack_shards.main, argv = ["verify", "--index", str(index), "--samples", "100"])
            rows = read_manifest(path = manifest)
            plain = TranscriptionDataset(rows = [dict(row) for row in rows], target = "original")
            reader, summary = shards.attach_shards(rows = rows, header = header, dirs = [root / "shards"])
            self.assertEqual(summary["rows_in_index"], len(rows))
            packed = TranscriptionDataset(rows = rows, target = "original", shards = reader)
            for index_number, row in enumerate(rows):
                number, offset, size = row["audio_shard"]
                self.assertEqual(reader.read(location = row["audio_shard"]), Path(row["audio_path"]).read_bytes())
                np.testing.assert_array_equal(actual = packed[index_number]["waveform"], desired = plain[index_number]["waveform"])
                if row.get("offset_s") is not None:
                    self.assertAlmostEqual(len(packed[index_number]["waveform"]) / 16000, row["duration_s"], delta = 0.05)


if __name__ == "__main__":
    unittest.main()
