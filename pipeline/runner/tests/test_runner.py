"""流水线测试：python -m unittest pipeline.runner.tests.test_runner

用假封装（见 fakes.py）验证任务生成规则、统计、分片，以及中途终止后续跑与一次跑完的结果完全一致。
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from pipeline.runner.config import load_config
from pipeline.runner.io_utils import read_jsonl
from pipeline.runner.plan import build_tasks, load_entries, load_voices

REPO_ROOT = Path(__file__).resolve().parents[3]
FAKES = "pipeline.runner.tests.fakes"


def make_config(texts_file: Path) -> dict:
    fake_tts = {
        "python": sys.executable,
        "module": FAKES,
        "args": ["tts"],
        "gpu": True,
        "batch_size": 2,
        "chunk_size": 3,
        "needs_ref_text": True,
    }
    fake_asr = {
        "python": sys.executable,
        "module": FAKES,
        "gpu": False,
        "chunk_size": 4,
        "env_vars": {"FAKE_TEXTS": str(texts_file)},
    }
    return {
        "seed": 7,
        "tts_models": {
            "tts_a": {**fake_tts, "langs": ["en", "ja"], "clone": True, "builtin_langs": ["en"], "aliases": ["a-model"]},
            "tts_b": {**fake_tts, "langs": ["en", "ja"], "clone": True, "builtin_langs": []},
        },
        "asr_models": {
            "asr_p": {**fake_asr, "module": FAKES, "args": ["asr", "--variant", "primary"]},
            "asr_c": {**fake_asr, "args": ["asr", "--variant", "cross"]},
        },
        "langs": {
            "en": {"tts": {"tts_a": 1, "tts_b": 1}, "asr": "asr_p", "cross_asr": "asr_c", "sec_per_char": [0.05, 0.3]},
            "ja": {"tts": {"tts_a": 1}, "asr": "asr_p", "sec_per_char": [0.05, 0.3]},
        },
    }


def make_entries() -> list:
    rows = []
    for index in range(14):
        text = f"sample sentence number {index}" if index % 2 == 0 else f"ja text {index}"
        rows.append(
            {
                "entry_id": f"{'en' if index % 2 == 0 else 'ja'}-src-{index:03d}",
                "lang": "en" if index % 2 == 0 else "ja",
                "source": "src",
                "source_license": "CC0",
                "original": text,
                "script": text,
                "tn_status": "unchanged",
            }
        )
    rows[0]["script"] = rows[0]["original"] = "this will RUNAWAY for long"
    rows[2]["script"] = rows[2]["original"] = "this one has MISMATCH inside"
    rows[4]["script"] = rows[4]["original"] = "this one has CROSSBAD inside"
    rows[6]["script"] = rows[6]["original"] = "this one has TTSFAIL inside"
    rows[1]["tn_status"] = "dropped"
    return rows


def make_voices(audio: Path) -> list:
    voices = []
    for index, (model, holdout) in enumerate([("a-model", False), ("tts_b", False), ("tts_a", False), ("tts_b", True)]):
        voices.append(
            {
                "voice_id": f"v{index}",
                "source_model": model,
                "builtin_name": f"spk{index}",
                "gender": "f",
                "ref_audio": str(audio),
                "ref_text": "reference text",
                "ref_lang": "en" if index % 2 == 0 else "ja",
                "holdout": holdout,
            }
        )
    voices[0]["ref_entry_id"] = "en-src-008"
    return voices


class Workspace:
    def __init__(self, root: Path):
        self.root = root
        self.work = root / "work"
        self.work.mkdir()
        (self.work / "gpu.lock").touch()
        self.ref = root / "ref.flac"
        self.ref.write_bytes(b"ref")
        self.entries = make_entries()
        self.texts_file = root / "texts.json"
        self.config = make_config(texts_file = self.texts_file)
        self.config_file = root / "config.json"
        self.config_file.write_text(json.dumps(self.config), encoding = "utf-8")
        self.voices = make_voices(audio = self.ref)
        self.voices_file = root / "voices.jsonl"
        self.voices_file.write_text("\n".join(json.dumps(v) for v in self.voices) + "\n", encoding = "utf-8")
        self.entries_file = root / "entries.jsonl"
        self.entries_file.write_text("\n".join(json.dumps(e) for e in self.entries) + "\n", encoding = "utf-8")
        mapping = {f"train-{e['entry_id']}": e["script"] for e in self.entries}
        mapping.update({f"test-{e['entry_id']}": e["script"] for e in self.entries})
        self.texts_file.write_text(json.dumps(mapping), encoding = "utf-8")

    def run(self, out_dir: Path, *extra: str, command: str = "run", env: dict | None = None) -> subprocess.CompletedProcess:
        full_env = dict(os.environ)
        full_env.update(env or {})
        return subprocess.run(
            [
                sys.executable, "-m", "pipeline.runner", command,
                "--config", str(self.config_file),
                "--work-dir", str(self.work),
                "--out-dir", str(out_dir),
                "--texts", str(self.entries_file),
                "--voices", str(self.voices_file),
                *extra,
            ],
            cwd = REPO_ROOT,
            env = full_env,
            capture_output = True,
            text = True,
        )


def outputs(out_dir: Path) -> dict:
    result = {}
    for name in ("tasks.jsonl", "asr_results.jsonl", "manifest.jsonl", "samples.jsonl", "stats.json"):
        result[name] = (out_dir / name).read_text(encoding = "utf-8")
    result["audio"] = {path.name: path.read_bytes() for path in sorted((out_dir / "audio").glob("*.flac"))}
    return result


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.ws = Workspace(root = self.tmp)
        self.cfg = load_config(path = self.ws.config_file, work_dir = self.ws.work)
        self.entries = load_entries(paths = [self.ws.entries_file], langs = None, limit_per_lang = None, fallback_original = False)
        self.voices = load_voices(path = self.ws.voices_file)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors = True)

    def test_train_uses_non_holdout_and_excludes_ref_entries(self):
        tasks, info = build_tasks(cfg = self.cfg, entries = self.entries, voices = self.voices, split = "train")
        self.assertTrue(all(not task["holdout"] for task in tasks))
        self.assertNotIn("en-src-008", {task["entry_id"] for task in tasks})
        self.assertNotIn("ja-src-001", {task["entry_id"] for task in tasks})  # tn_status 为 dropped
        self.assertEqual(info["excluded_ref_entries"], 1)

    def test_test_split_uses_holdout_only(self):
        tasks, _ = build_tasks(cfg = self.cfg, entries = self.entries, voices = self.voices, split = "test")
        self.assertTrue(tasks)
        self.assertEqual({task["voice_id"] for task in tasks}, {"v3"})
        self.assertTrue(all(task["holdout"] for task in tasks))
        # ja 只配了 tts_a，而留出音色属于 tts_b，tts_a 能克隆所以仍可合成
        self.assertTrue(all(task["mode"] == "clone" for task in tasks))

    def test_mode_and_cross_lingual(self):
        tasks, _ = build_tasks(cfg = self.cfg, entries = self.entries, voices = self.voices, split = "train")
        for task in tasks:
            voice = next(v for v in self.voices if v["voice_id"] == task["voice_id"])
            self.assertEqual(task["cross_lingual"], voice["ref_lang"] != task["lang"])
            builtin = (
                task["tts_model"] == "tts_a"
                and voice["source_model"] in ("a-model", "tts_a")
                and task["lang"] == "en"
            )
            self.assertEqual(task["mode"], "builtin" if builtin else "clone", task)
            if task["mode"] == "builtin":
                self.assertEqual(task["builtin_name"], voice["builtin_name"])
                self.assertNotIn("ref_audio", task)
            else:
                self.assertEqual(task["ref_text"], "reference text")

    def test_deterministic_and_order_independent(self):
        first, _ = build_tasks(cfg = self.cfg, entries = self.entries, voices = self.voices, split = "train")
        second, _ = build_tasks(
            cfg = self.cfg,
            entries = list(reversed(self.entries)),
            voices = list(reversed(self.voices)),
            split = "train",
        )
        self.assertEqual(first, second)
        self.cfg["seed"] = 8
        other, _ = build_tasks(cfg = self.cfg, entries = self.entries, voices = self.voices, split = "train")
        self.assertNotEqual([t["seed"] for t in first], [t["seed"] for t in other])


class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.ws = Workspace(root = self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors = True)

    def full_run(self, name: str, *extra: str) -> Path:
        out = self.tmp / name
        proc = self.ws.run(out, *extra)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return out

    def test_stats_and_failure_reasons(self):
        out = self.full_run("full")
        stats = json.loads((out / "stats.json").read_text(encoding = "utf-8"))
        reasons = stats["fail_reasons"]
        self.assertEqual(reasons.get("length_too_long"), 1)
        self.assertEqual(reasons.get("tts_failed"), 1)
        self.assertEqual(reasons.get("mismatch"), 1)
        self.assertEqual(reasons.get("cross_mismatch"), 1)
        self.assertEqual(stats["pending"], 0)
        self.assertEqual(set(stats["by_cross_lingual"]), {"True", "False"})
        self.assertEqual(sum(g["total"] for g in stats["by_lang"].values()), stats["decided"])
        manifest = read_jsonl(path = out / "manifest.jsonl")
        self.assertEqual(len(manifest), stats["pass"])
        self.assertEqual(
            sorted(manifest[0]),
            sorted(
                [
                    "sample_id", "audio_path", "duration_s", "lang", "original", "script", "source", "source_license",
                    "tts_model", "voice_id", "holdout", "cross_lingual", "asr_model", "cross_asr_model", "entry_id",
                    "mode", "seed",
                ]
            ),
        )
        for row in manifest:
            self.assertTrue((out / row["audio_path"]).exists())
        progress = json.loads((out / "progress.json").read_text(encoding = "utf-8"))
        stages = {info["stage"] for info in progress["stages"].values()}
        self.assertEqual(stages, {"synth", "length", "asr_primary", "asr_cross", "compare"})
        for info in progress["stages"].values():
            self.assertEqual(info["done"], info["total"])
        self.assertEqual(progress["final"]["pass"], stats["pass"])
        self.assertEqual(progress["overall"]["synth_failed"], 1)
        lines = (out / "logs" / "progress.log").read_text(encoding = "utf-8").splitlines()
        self.assertTrue(any("stage=synth" in line and "samples/s" in line and "eta=" in line for line in lines))
        # 失控音频不会送进 ASR
        asr_ids = {row["sample_id"] for row in read_jsonl(path = out / "asr_results.jsonl")}
        runaway = [r for r in read_jsonl(path = out / "samples.jsonl") if r["reason"] == "length_too_long"]
        self.assertTrue(runaway)
        self.assertFalse(asr_ids & {r["sample_id"] for r in runaway})

    def test_resume_after_stop_equals_full_run(self):
        full = self.full_run("full")
        part = self.tmp / "part"
        for calls in (1, 2, 3):
            proc = self.ws.run(part, "--stop-after-calls", str(calls))
            self.assertEqual(proc.returncode, 3, proc.stderr)
        final = self.ws.run(part)
        self.assertEqual(final.returncode, 0, final.stderr)
        self.assertEqual(outputs(full), outputs(part))

    def test_resume_after_wrapper_crash(self):
        full = self.full_run("full")
        crash = self.tmp / "crash.flag"
        crash.touch()
        part = self.tmp / "part"
        proc = self.ws.run(part, env = {"FAKE_CRASH_FILE": str(crash)})
        self.assertEqual(proc.returncode, 1, proc.stderr)
        proc = self.ws.run(part, env = {"FAKE_CRASH_FILE": str(crash)})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(outputs(full), outputs(part))

    def test_shards_combine_to_full_run(self):
        full = self.full_run("full")
        shards = self.tmp / "shards"
        first = self.ws.run(shards, "--shard", "0/2")
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertGreater(json.loads((shards / "stats.json").read_text(encoding = "utf-8"))["pending"], 0)
        second = self.ws.run(shards, "--shard", "1/2")
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(outputs(full), outputs(shards))


if __name__ == "__main__":
    unittest.main()
