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

    def run(self, out_dir: Path, *extra: str, command: str = "all", env: dict | None = None) -> subprocess.CompletedProcess:
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


    def test_prefer_builtin(self):
        self.cfg["prefer_builtin"] = True
        tasks, _ = build_tasks(cfg = self.cfg, entries = self.entries, voices = self.voices, split = "train")
        en_a = [task for task in tasks if task["tts_model"] == "tts_a" and task["lang"] == "en"]
        self.assertTrue(en_a)
        self.assertTrue(all(task["mode"] == "builtin" for task in en_a))
        # 没有内置音色可用的语言仍然克隆
        self.assertTrue(all(task["mode"] == "clone" for task in tasks if task["lang"] == "ja"))
        # 按语言覆盖：en 关闭优先内置后，会出现克隆
        self.cfg["langs"]["en"]["prefer_builtin"] = False
        tasks, _ = build_tasks(cfg = self.cfg, entries = self.entries, voices = self.voices, split = "train")
        self.assertTrue(any(task["mode"] == "clone" for task in tasks if task["tts_model"] == "tts_a" and task["lang"] == "en"))


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

    def test_resume_after_asr_chunk_size_change(self):
        full = self.full_run("full")
        part = self.tmp / "part"
        shutil.copytree(src = full, dst = part)
        # 块大小改了：旧块文件名对不上，但样本都已识别过，不应再调用 ASR 封装（FAKE_TEXTS 指向不存在的文件，一调用就失败）
        for backend in self.ws.config["asr_models"].values():
            backend["chunk_size"] = 3
            backend["env_vars"] = {"FAKE_TEXTS": str(self.tmp / "missing.json")}
        self.ws.config_file.write_text(json.dumps(self.ws.config), encoding = "utf-8")
        for role in ("primary", "cross"):
            proc = self.ws.run(part, "--role", role, command = "asr")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            progress = json.loads((part / "progress.json").read_text(encoding = "utf-8"))
            for key, info in progress["stages"].items():
                if key.startswith(f"asr_{role}:"):
                    self.assertEqual(info["done"], info["total"], key)
        self.assertEqual(self.ws.run(part, command = "compare").returncode, 0)
        self.assertEqual(self.ws.run(part, command = "manifest").returncode, 0)
        self.assertEqual(outputs(full), outputs(part))
        # 删掉一个旧块：只补识别缺的样本，结果仍与一次跑完相同
        for backend in self.ws.config["asr_models"].values():
            backend["env_vars"] = {"FAKE_TEXTS": str(self.ws.texts_file)}
        self.ws.config_file.write_text(json.dumps(self.ws.config), encoding = "utf-8")
        again = self.tmp / "again"
        shutil.copytree(src = full, dst = again)
        old = sorted((again / "state" / "asr_primary" / "asr_p").glob("chunk-*.jsonl"))
        old[0].unlink()
        for role in ("primary", "cross"):
            proc = self.ws.run(again, "--role", role, command = "asr")
            self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self.ws.run(again, command = "compare").returncode, 0)
        self.assertEqual(self.ws.run(again, command = "manifest").returncode, 0)
        self.assertEqual(outputs(full), outputs(again))

    def test_shards_combine_to_full_run(self):
        full = self.full_run("full")
        shards = self.tmp / "shards"
        self.assertEqual(self.ws.run(shards, command = "plan").returncode, 0)
        first = self.ws.run(shards, "--shard", "0/2", command = "tts")
        self.assertEqual(first.returncode, 0, first.stderr)
        # 另一片还没合成，时长检查需要全部样本，应当报输入错误
        early = self.ws.run(shards, command = "length")
        self.assertEqual(early.returncode, 2, early.stderr)
        self.assertEqual(self.ws.run(shards, "--shard", "1/2", command = "tts").returncode, 0)
        for role in ("primary", "cross"):
            self.assertEqual(self.ws.run(shards, command = "length").returncode, 0)
            for shard in ("0/2", "1/2"):
                proc = self.ws.run(shards, "--shard", shard, "--role", role, command = "asr")
                self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(self.ws.run(shards, command = "compare").returncode, 0)
        self.assertEqual(self.ws.run(shards, command = "manifest").returncode, 0)
        self.assertEqual(outputs(full), outputs(shards))

    def test_subcommands_run_separately(self):
        out = self.tmp / "steps"
        # 没有上一步的输出时，每个子命令都报输入错误（退出码 2）
        for command in ("tts", "length", "asr", "compare", "manifest"):
            proc = self.ws.run(out, command = command)
            self.assertEqual(proc.returncode, 2, f"{command}: {proc.stderr}")
        steps = [
            ("plan", [], ["tasks.jsonl"]),
            ("tts", [], ["synth_results.jsonl"]),
            ("length", [], ["length_check.jsonl"]),
            ("asr", ["--role", "primary"], ["asr_primary.jsonl"]),
            ("asr", ["--role", "cross"], None),  # 缺主 ASR 的比对结果，应当报错
            ("compare", [], ["asr_results.jsonl"]),
            ("asr", ["--role", "cross"], ["asr_cross.jsonl"]),
            ("compare", [], ["asr_results.jsonl"]),
            ("manifest", [], ["manifest.jsonl", "samples.jsonl", "stats.json", "stats.md"]),
        ]
        for command, extra, produced in steps:
            proc = self.ws.run(out, *extra, command = command)
            if produced is None:
                self.assertEqual(proc.returncode, 2, proc.stderr)
                continue
            self.assertEqual(proc.returncode, 0, f"{command}: {proc.stderr}")
            for name in produced:
                self.assertTrue((out / name).exists(), name)
        # 单独重跑任何一步，结果不变
        before = outputs(out)
        for command in ("tts", "length", "compare", "manifest"):
            self.assertEqual(self.ws.run(out, command = command).returncode, 0)
        self.assertEqual(before, outputs(out))
        self.assertEqual(outputs(self.full_run("full")), before)

    def test_models_filter_two_processes(self):
        full = self.full_run("full")
        out = self.tmp / "two"
        self.assertEqual(self.ws.run(out, command = "plan").returncode, 0)
        a = self.ws.run(out, "--models", "tts_a", command = "tts")
        self.assertEqual(a.returncode, 0, a.stderr)
        self.assertEqual(list((out / "state" / "synth").iterdir())[0].name, "tts_a")
        self.assertEqual(len(list((out / "state" / "synth").iterdir())), 1)
        b = self.ws.run(out, "--models", "tts_b", command = "tts")
        self.assertEqual(b.returncode, 0, b.stderr)
        for command, extra in (("length", []), ("asr", []), ("compare", []), ("asr", ["--role", "cross"]), ("compare", []), ("manifest", [])):
            self.assertEqual(self.ws.run(out, *extra, command = command).returncode, 0)
        self.assertEqual(outputs(full), outputs(out))

    def test_multiprocess_tts(self):
        full = self.full_run("full")
        self.ws.config["tts_models"]["tts_a"]["processes"] = 3
        self.ws.config_file.write_text(json.dumps(self.ws.config), encoding = "utf-8")
        multi = self.full_run("multi")
        self.assertEqual(outputs(full), outputs(multi))
        # 多进程中途终止后续跑也一致
        part = self.tmp / "multi_part"
        self.assertEqual(self.ws.run(part, "--stop-after-calls", "2").returncode, 3)
        self.assertEqual(self.ws.run(part).returncode, 0)
        self.assertEqual(outputs(full), outputs(part))

    def test_input_format_is_checked(self):
        out = self.tmp / "bad"
        self.assertEqual(self.ws.run(out, command = "plan").returncode, 0)
        rows = [json.loads(line) for line in (out / "tasks.jsonl").read_text(encoding = "utf-8").splitlines()]
        del rows[0]["voice_id"]
        (out / "tasks.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding = "utf-8")
        proc = self.ws.run(out, command = "tts")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("voice_id", proc.stderr)


class RealTargetsTest(unittest.TestCase):
    """真人语音的训练目标：中文一律转为简体，target_from 为 asr 时用主 ASR 结果。"""

    def test_traditional_transcript_becomes_simplified(self):
        from pipeline.runner.report import real_targets

        task = {"lang": "zh", "original": "這個軟體很好用。", "text": "這個軟體很好用。", "target_from": "transcript"}
        original, script = real_targets(task = task, primary_hypothesis = "")
        # 只换字形，不把台湾用词换成大陆用词（软体保持软体）
        self.assertEqual(original, "这个软体很好用。")
        self.assertEqual(script, original)

    def test_halfwidth_punctuation_next_to_han_becomes_fullwidth(self):
        from pipeline.runner.report import real_targets, zh_fullwidth_punctuation

        cases = {
            "不肯活下来啊,那个血,扣太多了,我不肯活下来。": "不肯活下来啊，那个血，扣太多了，我不肯活下来。",
            "真的吗?太好了!": "真的吗？太好了！",
            "原因有三: 第一; 第二.": "原因有三：第一；第二。",
            "加拿大的移民法,IRPA,从第33条一直到第43条": "加拿大的移民法，IRPA，从第33条一直到第43条",
            "约3.5亿人, 增长了2.1%": "约3.5亿人，增长了2.1%",
            "我想想...好吧": "我想想……好吧",
            "Python 3.12 和 e.g. 这类写法": "Python 3.12 和 e.g. 这类写法",
        }
        for text, expected in cases.items():
            self.assertEqual(zh_fullwidth_punctuation(text = text), expected)
        task = {"lang": "zh", "original": "這個,很好用.", "text": "這個,很好用.", "target_from": "transcript"}
        self.assertEqual(real_targets(task = task, primary_hypothesis = ""), ("这个，很好用。", "这个，很好用。"))
        # 其他语言不处理
        task = {"lang": "ja", "original": "はい,そうです.", "text": "はい,そうです.", "target_from": "transcript"}
        self.assertEqual(real_targets(task = task, primary_hypothesis = ""), ("はい,そうです.", "はい,そうです."))

    def test_asr_target_is_simplified(self):
        from pipeline.runner.report import real_targets

        task = {"lang": "zh", "original": "这个软体很好用", "text": "这个软体很好用", "target_from": "asr"}
        original, _ = real_targets(task = task, primary_hypothesis = " 這個軟體很好用。 ")
        self.assertEqual(original, "这个软体很好用。")

    def test_other_languages_unchanged(self):
        from pipeline.runner.report import real_targets

        task = {"lang": "ja", "original": "這個", "text": "這個", "target_from": "transcript"}
        self.assertEqual(real_targets(task = task, primary_hypothesis = ""), ("這個", "這個"))

    def test_human_verified_skips_asr(self):
        from pipeline.runner.report import judge

        cfg = {"langs": {"de": {"asr": "qwen3_asr", "cross_asr": "parakeet_v3"}}}
        synth = {"a": {"status": "ok"}, "b": {"status": "ok"}}
        length = {"a": {"ok": True}, "b": {"ok": True}}
        verified = {"sample_id": "a", "lang": "de", "human_verified": True}
        plain = {"sample_id": "b", "lang": "de"}
        self.assertEqual(judge(cfg = cfg, task = verified, synth = synth, length = length, asr = {}), ("pass", ""))
        self.assertEqual(judge(cfg = cfg, task = plain, synth = synth, length = length, asr = {}), ("pending", "primary_pending"))

    def test_human_verified_short_sentence_passes_length(self):
        from pipeline.runner.stages import length_verdict

        cfg = {"default_sec_per_char": [0.02, 1.0], "langs": {"zh": {"sec_per_char": None}}}
        # 「打开」两个字配 4 秒录音（前后留白）：人工转写只检查时长不为零，其他来源仍按比例判为过长
        verified = {"lang": "zh", "text": "打开", "human_verified": True}
        plain = {"lang": "zh", "text": "打开"}
        self.assertEqual(length_verdict(cfg = cfg, task = verified, record = {"duration_s": 4.0}), "")
        self.assertEqual(length_verdict(cfg = cfg, task = plain, record = {"duration_s": 4.0}), "length_too_long")
        self.assertEqual(length_verdict(cfg = cfg, task = verified, record = {"duration_s": 0}), "length_zero_duration")
        # 重新识别模式：原转写不作依据，同样只检查时长不为零
        retranscribe = {"lang": "zh", "text": "打开", "retranscribe": True}
        self.assertEqual(length_verdict(cfg = cfg, task = retranscribe, record = {"duration_s": 28.0}), "")

    def test_punctuation_applied_only_to_same_input(self):
        from pipeline.runner.report import apply_punctuation, load_punctuation

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "punctuation.jsonl"
            rows = [
                {"sample_id": "a", "input": "سلام دنیا", "output": "سلام، دنیا.", "status": "ok"},
                {"sample_id": "b", "input": "خوب است", "output": None, "status": "failed"},
            ]
            path.write_text("".join(json.dumps(obj = row, ensure_ascii = False) + "\n" for row in rows), encoding = "utf-8")
            punctuation = load_punctuation(path = path)
        self.assertEqual(
            apply_punctuation(sample_id = "a", original = "سلام دنیا", script = "سلام دنیا", punctuation = punctuation),
            ("سلام، دنیا.", "سلام، دنیا.", "سلام دنیا"),
        )
        # 训练目标变了（例如主 ASR 重跑）时不套用旧结果；未采用的结果也不套用
        self.assertEqual(
            apply_punctuation(sample_id = "a", original = "سلام", script = "سلام", punctuation = punctuation),
            ("سلام", "سلام", None),
        )
        self.assertEqual(
            apply_punctuation(sample_id = "b", original = "خوب است", script = "خوب است", punctuation = punctuation),
            ("خوب است", "خوب است", None),
        )


class MatchToleranceTest(unittest.TestCase):
    """langs.<lang>.match_max_cer：逐字不一致但字符错误率不超过容差时算通过。"""

    SCRIPT = "Ang Dead Balagtas ang una at pinakamalikhaing komiks tungkol sa mahaba at makulay na kasaysayan ng Filipinas."

    def compare(self, hypothesis, max_cer = None):
        from pipeline.runner.stages import _compare_row

        task = {"sample_id": "s1", "lang": "fil", "text": self.SCRIPT, "original": self.SCRIPT}
        row = {"sample_id": "s1", "asr_model": "omni", "asr_role": "primary", "hypothesis": hypothesis, "error": None}
        return _compare_row(task = task, row = row, max_cer = max_cer)

    def test_one_letter_passes_within_tolerance(self):
        # makulay → makuly：约 93 个字符少 1 个，约 1.1%
        hypothesis = "ang dead balagtas ang una at pinakamalikhaing komiks tungkol sa mahaba at makuly na kasaysayan ng filipinas"
        self.assertFalse(self.compare(hypothesis = hypothesis)["match"])
        record = self.compare(hypothesis = hypothesis, max_cer = 0.02)
        self.assertTrue(record["match"])
        self.assertLess(record["cer"], 0.02)
        self.assertEqual(record["max_cer"], 0.02)

    def test_larger_difference_still_fails(self):
        # 三处不同（ded balaktas、comics、makuli），超过 2%
        hypothesis = "ang ded balaktas ang una at pinakamalikhaing comics tungkol sa mahaba at makuli na kasaysayan ng filipinas"
        record = self.compare(hypothesis = hypothesis, max_cer = 0.02)
        self.assertFalse(record["match"])
        self.assertGreater(record["cer"], 0.02)


class MergeManifestsTest(unittest.TestCase):
    def write(self, directory: Path, lang: str, count: int, seconds: float) -> Path:
        directory.mkdir(parents = True)
        rows = [
            {"sample_id": f"{directory.name}-{index}", "audio_path": f"audio/{index}.flac", "lang": lang, "duration_s": seconds}
            for index in range(count)
        ]
        (directory / "manifest.jsonl").write_text(
            "".join(json.dumps(obj = row) + "\n" for row in rows),
            encoding = "utf-8",
        )
        return directory

    def test_language_cap_and_duplicates(self) -> None:
        from pipeline.runner.merge_manifests import merge

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chinese = self.write(directory = root / "zh", lang = "zh", count = 120, seconds = 60)
            german = self.write(directory = root / "de", lang = "de", count = 120, seconds = 60)
            # 同一批样本的第二份清单：sample_id 相同，只收第一次出现的
            again = root / "zh-again"
            again.mkdir()
            (again / "manifest.jsonl").write_text((chinese / "manifest.jsonl").read_text(encoding = "utf-8"), encoding = "utf-8")
            rows, report = merge(
                directories = [chinese, german, again],
                cap_hours = 1.0,
                seed = 0,
                lang_caps = {"zh": 1.5},
            )
        # 中文按单独上限 90 分钟，德语按统一上限 60 分钟；第二份中文清单全部是重复
        self.assertEqual([item["kept"] for item in report], [90, 60, 0])
        self.assertEqual(report[2]["duplicates"], 90)
        self.assertEqual(len({row["sample_id"] for row in rows}), len(rows))

    def test_cap_prefers_earlier_directories(self) -> None:
        from pipeline.runner.merge_manifests import merge

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            verified = self.write(directory = root / "cv-de", lang = "de", count = 30, seconds = 60)
            checked = self.write(directory = root / "asr-de", lang = "de", count = 30, seconds = 60)
            english = self.write(directory = root / "asr-en", lang = "en", count = 90, seconds = 60)
            rows, report = merge(
                directories = [verified, checked, english],
                cap_hours = 1.0,
                uncapped = {"en"},
                seed = 0,
            )
        # 德语上限 60 分钟：前一个目录 30 分钟全收，后一个目录只补到 30 分钟；英语不受上限
        self.assertEqual([item["kept"] for item in report], [30, 30, 90])
        self.assertEqual(sum(row["duration_s"] for row in rows if row["lang"] == "de"), 3600)
        self.assertTrue(all(Path(row["audio_path"]).is_absolute() for row in rows))

        with tempfile.TemporaryDirectory() as tmp:
            big = self.write(directory = Path(tmp) / "cv-fr", lang = "fr", count = 100, seconds = 60)
            first, _ = merge(directories = [big], cap_hours = 0.5, seed = 0)
            again, _ = merge(directories = [big], cap_hours = 0.5, seed = 0)
            other, _ = merge(directories = [big], cap_hours = 0.5, seed = 1)
        # 超出上限时随机取子集，同一个种子结果相同
        ids = [row["sample_id"] for row in first]
        self.assertEqual(len(ids), 30)
        self.assertEqual(ids, [row["sample_id"] for row in again])
        self.assertNotEqual(ids, [f"cv-fr-{index}" for index in range(30)])
        self.assertNotEqual(set(ids), {row["sample_id"] for row in other})


if __name__ == "__main__":
    unittest.main()
