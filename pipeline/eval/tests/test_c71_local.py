"""本机评测驱动程序（任务 #71，pipeline/eval/c71_local.py）与思考检查分批的单元测试：
python -m unittest pipeline.eval.tests.test_c71_local

只在 CPU 上运行，用很小的假文件：

- SafetensorsTest、DiscoverTest：按文件头核对 safetensors 是否下载完整；哪些运行的必需文件齐全。
- PlanTest：本机的命令与 c_runs.eval_items 相同，只换了 python 并给思考检查加 --fit-memory；思考检查（含参考模型）带
  --attn-implementation；两种顺序；参考模型只有三项；时间估计；冒烟比较的命令与结论的取法（已有结果优先）。
- DriverTest：已有结果的项目跳过；失败记下后接着做；超时时整个进程组结束，不留子进程；GPU 锁在子进程结束后释放。
- MainTest：先做冒烟比较再评测，结论决定全部思考检查的实现（比较失败时 sdpa）；续跑时沿用已有结果的实现；--runs-root
  与环境变量 C71_RUNS_ROOT；演习列出运行目录与冒烟比较的命令。
- SummaryTest：探针 summary.json 中的 config 不当作任务；汇总的两张表与和 C0 比较的表。
- ThinkBatchTest：思考检查按显存估算的每条用量（sdpa 与 flash_attention_2）与批大小；显存不足时对半拆开重试；参数。
- ThinkFallbackTest：flash_attention_2 载入或生成出错时整项改用 sdpa 从头重做；显存不足不算出错。
"""

import contextlib
import fcntl
import io
import json
import os
import struct
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from pipeline.colab import c_runs
from pipeline.eval import c71_local, think_check
from pipeline.eval.think_check import SPLIT_SEED_OFFSET, fit_batch_size, generate_resilient, per_sample_bytes


def write_safetensors(path, numbers = 4, truncate = 0):
    """写一个只有一个 float32 张量的 safetensors 文件；truncate 大于 0 时去掉末尾这么多字节（模拟没下载完）。"""
    header = json.dumps(obj = {"weight": {"dtype": "F32", "shape": [numbers], "data_offsets": [0, 4 * numbers]}}).encode()
    data = struct.pack("<Q", len(header)) + header + bytes(4 * numbers)
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(data[: len(data) - truncate])


def write_run(root, key, lm = True, projector = True, truncate_lm = 0, logs = True):
    run_dir = Path(root) / c_runs.run_name(key = key)
    checkpoint = run_dir / c71_local.CHECKPOINT
    if lm:
        write_safetensors(path = checkpoint / "language_model.safetensors", numbers = 16, truncate = truncate_lm)
    if projector:
        write_safetensors(path = checkpoint / "projector.safetensors")
    if logs:
        (checkpoint / "projector.json").write_text(data = "{}", encoding = "utf-8")
        (run_dir / "train_config.json").write_text(
            data = json.dumps(obj = {"init_projector": "/x/v2b-2b-cpt77-projector/checkpoint-0002700"}),
            encoding = "utf-8",
        )
        rows = [{"step": 30000, "loss": 0.43, "it_per_s": 5.0, "mem_gb": 47.0}]
        if key != "c0":
            rows[0].update(replay_loss = 1.3, replay_ratio_total = 0.5, replay_grad_share = 0.45)
            (run_dir / "replay_log.jsonl").write_text(data = "", encoding = "utf-8")
        (run_dir / "train_log.jsonl").write_text(data = "\n".join(json.dumps(obj = row) for row in rows) + "\n", encoding = "utf-8")
        dev = {"c0": 0.674, "c-mix": 0.650}.get(key, 0.66)
        (run_dir / "eval_log.jsonl").write_text(
            data = json.dumps(obj = {"step": 30000, "overall": {"loss": dev}}) + "\n",
            encoding = "utf-8",
        )
    return run_dir


def fake_paths(root):
    root = Path(root)
    return {
        "work": root,
        "model_dir": root / "model",
        "fleurs": root / "fleurs",
        "replay": root / "replay",
        "runs_root": root / "out" / "runs",
        "out_dir": root / "out",
        "eval_root": root / "out" / "eval",
        "python_audio": Path("/envs/audio/bin/python"),
        "python_text": Path("/envs/text/bin/python"),
        "gpu_lock": root / "gpu.lock",
        "think_max_new": 16384,
    }


class SafetensorsTest(unittest.TestCase):
    def test_complete_and_truncated(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good.safetensors"
            bad = Path(tmp) / "bad.safetensors"
            write_safetensors(path = good)
            write_safetensors(path = bad, truncate = 3)
            self.assertEqual(c71_local.safetensors_complete(path = good), (True, ""))
            ok, reason = c71_local.safetensors_complete(path = bad)
            self.assertFalse(ok)
            self.assertIn("没有下载完", reason)
            self.assertEqual(c71_local.safetensors_complete(path = Path(tmp) / "none.safetensors"), (False, "不存在"))
            (Path(tmp) / "short.safetensors").write_bytes(b"\x01\x02")
            self.assertFalse(c71_local.safetensors_complete(path = Path(tmp) / "short.safetensors")[0])


class DiscoverTest(unittest.TestCase):
    def test_discover(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_run(root = tmp, key = "c0")
            write_run(root = tmp, key = "c-free_raw", truncate_lm = 5)
            write_run(
                root = tmp,
                key = "c-mix",
                lm = False,
                logs = False,
            )
            runs = c71_local.discover_runs(runs_root = tmp, keys = list(c_runs.RUNS))
        self.assertTrue(runs["c0"]["ready"])
        self.assertEqual(runs["c0"]["optional_missing"], [])
        self.assertFalse(runs["c-free_raw"]["ready"])
        self.assertIn("没有下载完", runs["c-free_raw"]["missing"][0])
        self.assertFalse(runs["c-mix"]["ready"])
        self.assertEqual(len(runs["c-mix"]["missing"]), 1)
        self.assertIn("language_model.safetensors", runs["c-mix"]["missing"][0])
        self.assertIn("replay_log.jsonl", runs["c-mix"]["optional_missing"])
        self.assertEqual(len(runs["c-cont_raw"]["missing"]), 2)


class PlanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.paths = fake_paths(root = self.tmp.name)
        for key in ("c0", "c-mix"):
            write_run(root = self.paths["runs_root"], key = key)
        self.runs = c71_local.discover_runs(runs_root = self.paths["runs_root"], keys = list(c_runs.RUNS))

    def tearDown(self):
        self.tmp.cleanup()

    def test_commands_follow_c_runs(self):
        plan = c71_local.build_plan(runs = self.runs, keys = list(c_runs.RUNS), paths = self.paths)
        checkpoint = self.runs["c-mix"]["dir"] / c71_local.CHECKPOINT
        out = self.paths["eval_root"] / "c-mix"
        reference = c_runs.eval_items(
            out = out,
            projector = checkpoint / "projector.safetensors",
            language_model = checkpoint / "language_model.safetensors",
            python = "PY",
            model_dir = self.paths["model_dir"],
            fleurs_dir = self.paths["fleurs"],
            replay_dir = self.paths["replay"],
            think_max_new = 16384,
        )
        local = {entry["item"]: entry for entry in plan if entry["label"] == "c-mix"}
        self.assertEqual(set(local), {"fleurs", "probe", "lm_ppl", "replay_gap", "think"})
        for name, command, marker, _ in reference:
            entry = local[name]
            python = {"audio": "/envs/audio/bin/python", "text": "/envs/text/bin/python"}[c71_local.PYTHON_OF_ITEM[name]]
            self.assertEqual(entry["command"][0], python)
            self.assertEqual(entry["command"][1:], command[1:] + c71_local.LOCAL_EXTRA.get(name, []))
            self.assertEqual(entry["marker"], marker)
        self.assertEqual(local["think"]["command"][-1], "--fit-memory")
        self.assertIn("16384", local["think"]["command"])
        self.assertEqual(local["fleurs"]["command"][local["fleurs"]["command"].index("--prompt-format") + 1], "cpt")
        # 思考检查一律用 FlashAttention-2，参考模型也一样
        for entry in plan:
            if entry["item"] == "think":
                command = entry["command"]
                self.assertEqual(command[command.index("--attn-implementation") + 1], "flash_attention_2")
        self.assertIn("ref", {entry["label"] for entry in plan if entry["item"] == "think"})

    def test_think_attention_and_estimates(self):
        fa2 = c71_local.build_plan(runs = self.runs, keys = list(c_runs.RUNS), paths = self.paths)
        self.paths["think_attention"] = "sdpa"
        sdpa = c71_local.build_plan(runs = self.runs, keys = list(c_runs.RUNS), paths = self.paths)
        for entry in sdpa:
            if entry["item"] == "think":
                self.assertEqual(entry["command"][entry["command"].index("--attn-implementation") + 1], "sdpa")
        minutes = {(entry["label"], entry["item"]): entry["minutes"] for entry in fa2}
        slower = {(entry["label"], entry["item"]): entry["minutes"] for entry in sdpa}
        # 思考检查的估计：flash_attention_2 约 34 分钟、sdpa 约 49 分钟；运行的每一项另加从运行目录载入语言模型的用时
        self.assertAlmostEqual(minutes[("ref", "think")], 33.8, delta = 0.5)
        self.assertAlmostEqual(slower[("ref", "think")], 48.8, delta = 0.5)
        self.assertAlmostEqual(minutes[("c0", "think")] - minutes[("ref", "think")], c71_local.LM_LOAD_MINUTES, places = 6)
        self.assertAlmostEqual(minutes[("c0", "fleurs")], 22 + c71_local.LM_LOAD_MINUTES, places = 6)
        self.assertEqual(minutes[("ref", "lm_ppl")], 3)
        half = c71_local.think_minutes(max_new = 8192, attention = "flash_attention_2")
        self.assertLess(half, minutes[("ref", "think")] / 2)

    def test_smoke_decision(self):
        paths = self.paths
        entry = c71_local.smoke_entry(paths = paths)
        self.assertEqual(entry["command"][0], "/envs/text/bin/python")
        self.assertEqual(entry["command"][1:3], ["-m", "pipeline.eval.think_check"])
        self.assertIn("--smoke", entry["command"])
        self.assertNotIn("--language-model", entry["command"])
        self.assertEqual(entry["marker"], paths["out_dir"] / "think_smoke" / "summary.json")
        # 还没有冒烟比较、也没有思考检查结果：需要先比较
        self.assertIsNone(c71_local.planned_think_attention(requested = "auto", paths = paths))
        self.assertEqual(c71_local.planned_think_attention(requested = "sdpa", paths = paths)[0], "sdpa")
        for content, expected in (
            ({"use": "flash_attention_2", "reason": "一致"}, "flash_attention_2"),
            ({"use": "sdpa", "reason": "不一致"}, "sdpa"),
            ({}, "sdpa"),
        ):
            write_json(path = entry["marker"], data = content)
            self.assertEqual(c71_local.planned_think_attention(requested = "auto", paths = paths)[0], expected)
        # 已有思考检查结果时沿用它们的实现（没有记录的旧结果算 sdpa），优先于冒烟比较的结论
        write_json(path = entry["marker"], data = {"use": "sdpa"})
        write_json(path = paths["eval_root"] / "c0" / "think" / "summary.json", data = {"attn_implementation_used": "flash_attention_2"})
        use, reason = c71_local.planned_think_attention(requested = "auto", paths = paths)
        self.assertEqual(use, "flash_attention_2")
        self.assertIn("c0 flash_attention_2", reason)
        write_json(path = paths["eval_root"] / "ref" / "think" / "summary.json", data = {"closed": 0.9})
        write_json(path = paths["eval_root"] / "c-mix" / "think" / "summary.json", data = {"attn_implementation_used": "sdpa"})
        self.assertEqual(c71_local.think_attentions(eval_root = paths["eval_root"]), {"c-mix": "sdpa", "c0": "flash_attention_2", "ref": "sdpa"})
        use, reason = c71_local.planned_think_attention(requested = "auto", paths = paths)
        self.assertEqual(use, "sdpa")
        self.assertIn("取多数", reason)

    def test_filesystem(self):
        with tempfile.TemporaryDirectory() as tmp:
            mount = c71_local.filesystem(path = Path(tmp) / "not" / "yet")
        self.assertIsNotNone(mount)
        self.assertTrue(str(Path(tmp)).startswith(mount[0]))

    def test_orders(self):
        keys = list(c_runs.RUNS)
        plan = c71_local.build_plan(runs = self.runs, keys = keys, paths = self.paths)
        self.assertEqual(
            [(entry["item"], entry["label"]) for entry in plan],
            [
                ("replay_gap", "ref"), ("replay_gap", "c0"), ("replay_gap", "c-mix"),
                ("lm_ppl", "ref"), ("lm_ppl", "c0"), ("lm_ppl", "c-mix"),
                ("fleurs", "c0"), ("fleurs", "c-mix"),
                ("probe", "c0"), ("probe", "c-mix"),
                ("think", "ref"), ("think", "c0"), ("think", "c-mix"),
            ],
        )
        ref = [entry for entry in plan if entry["label"] == "ref"]
        for entry in ref:
            self.assertNotIn("--language-model", entry["command"])
            self.assertNotIn("--projector", entry["command"])
        by_run = c71_local.build_plan(
            runs = self.runs,
            keys = keys,
            paths = self.paths,
            order = "run",
            include_ref = False,
        )
        self.assertEqual([entry["label"] for entry in by_run], ["c0"] * 5 + ["c-mix"] * 5)
        self.assertEqual([entry["item"] for entry in by_run[:5]], ["fleurs", "probe", "lm_ppl", "replay_gap", "think"])


def marker_command(marker, counter, code = 0):
    """写完成标志并在 counter 中追加一行的假评测命令（code 非零时不写完成标志）。"""
    script = (
        "import sys, pathlib\n"
        f"pathlib.Path({str(counter)!r}).open('a').write('run\\n')\n"
        f"code = {code}\n"
        f"if code == 0:\n    pathlib.Path({str(marker)!r}).parent.mkdir(parents = True, exist_ok = True)\n"
        f"    pathlib.Path({str(marker)!r}).write_text('{{}}')\n"
        "sys.exit(code)\n"
    )
    return [sys.executable, "-c", script]


class DriverTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.paths = fake_paths(root = self.tmp.name)
        self.paths["out_dir"].mkdir(parents = True)
        self.driver = c71_local.Driver(paths = self.paths, busy_mib = 3000, timeout_factor = 3.0)
        patcher = mock.patch.object(target = c71_local, attribute = "gpu_used_mib", return_value = None)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def entry(self, label, item, command, marker):
        return {
            "label": label,
            "item": item,
            "command": command,
            "marker": Path(marker),
            "minutes": 1.0,
            "log": self.paths["eval_root"] / label / f"{item}.log",
        }

    def lock_is_free(self):
        with open(file = self.paths["gpu_lock"], mode = "a") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return True

    def test_resume_and_failure(self):
        root = Path(self.tmp.name)
        counter = root / "counter.txt"
        good = self.entry(
            label = "c0",
            item = "lm_ppl",
            command = marker_command(marker = root / "c0.json", counter = counter),
            marker = root / "c0.json",
        )
        bad = self.entry(
            label = "c-mix",
            item = "lm_ppl",
            command = marker_command(marker = root / "mix.json", counter = counter, code = 3),
            marker = root / "mix.json",
        )
        plan = [good, bad]
        self.assertEqual(self.driver.run_entry(entry = good, plan = plan), "ok")
        self.assertEqual(self.driver.run_entry(entry = bad, plan = plan), "failed（退出码 3）")
        self.assertTrue(self.lock_is_free())
        self.assertEqual(self.driver.run_entry(entry = good, plan = plan), "skipped")
        self.assertEqual(counter.read_text().count("run"), 2)
        results = json.loads(s = (self.paths["out_dir"] / "status.json").read_text(encoding = "utf-8"))["results"]
        self.assertEqual(results["c0:lm_ppl"]["status"], "ok")
        self.assertEqual(results["c-mix:lm_ppl"]["status"], "failed（退出码 3）")
        self.assertTrue(good["log"].exists())
        # 估计剩余时间只算还没有结果的项目，同类项目做完过的按实测用时估计
        self.assertEqual(self.driver.remaining_minutes(plan = plan), self.driver.estimate_minutes(entry = bad))
        self.assertLess(self.driver.estimate_minutes(entry = bad), 1.0)

    def test_timeout_kills_group(self):
        root = Path(self.tmp.name)
        pid_file = root / "grandchild.pid"
        script = (
            "import subprocess, pathlib, time\n"
            "child = subprocess.Popen(['sleep', '60'])\n"
            f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))\n"
            "time.sleep(60)\n"
        )
        entry = self.entry(
            label = "ref",
            item = "think",
            command = [sys.executable, "-c", script],
            marker = root / "never.json",
        )
        self.driver.min_timeout_minutes = 0.03
        self.driver.timeout_factor = 0.0
        start = time.time()
        result = self.driver.run_entry(entry = entry, plan = [entry])
        self.assertTrue(result.startswith("timeout"))
        self.assertLess(time.time() - start, 50)
        grandchild = int(pid_file.read_text())
        # 孙进程被结束后由 init 回收，回收前短暂地以僵尸进程存在
        gone = False
        for _ in range(100):
            try:
                os.kill(grandchild, 0)
            except ProcessLookupError:
                gone = True
                break
            time.sleep(0.1)
        self.assertTrue(gone)
        self.assertTrue(self.lock_is_free())


FAKE_PYTHON = """#!{python}
# 假的 python：-c 的导入检查直接通过；-m 评测模块时写出完成标志，并在 calls.txt 中记一行（模块名加全部参数）。
# 思考检查的冒烟比较按环境变量 FAKE_SMOKE_USE 给出结论（fail 时以退出码 1 失败、不写结果）；思考检查在 summary.json 中
# 记下 --attn-implementation。
import json
import os
import pathlib
import sys

args = sys.argv[1:]
if args[0] == "-c":
    sys.exit(0)
module = args[1]
with open(pathlib.Path(__file__).with_name("calls.txt"), "a") as handle:
    handle.write(" ".join(args[1:]) + "\\n")


def value(flag):
    return args[args.index(flag) + 1]


summary = {{}}
if module == "pipeline.eval.think_check" and "--smoke" in args:
    use = os.environ.get("FAKE_SMOKE_USE", "flash_attention_2")
    if use == "fail":
        sys.exit(1)
    summary = {{"use": use, "reason": "假的比较"}}
elif module == "pipeline.eval.think_check":
    summary = {{"attn_implementation_used": value("--attn-implementation")}}
if module == "pipeline.eval.lm_ppl":
    pathlib.Path(value("--output")).write_text("{{}}")
elif module != "pipeline.eval.report":
    out = pathlib.Path(value("--output-dir"))
    out.mkdir(parents = True, exist_ok = True)
    (out / "summary.json").write_text(json.dumps(summary))
    if module == "pipeline.eval.run_eval":
        (out / "hypotheses.jsonl").write_text("")
"""


def read_calls(path):
    """假的 python 记下的评测调用（不含 report）：[(模块, 参数列表)]。"""
    calls = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if parts and parts[0] != "pipeline.eval.report":
            calls.append((parts[0], parts[1:]))
    return calls


def write_inputs(root):
    """假的基本输入：模型目录、18 种语言的 FLEURS 清单与音频、五组回放评测集、假的 python。返回 main 的参数。"""
    root = Path(root)
    model = root / "model"
    model.mkdir()
    (model / "config.json").write_text(data = "{}", encoding = "utf-8")
    (model / "model.safetensors").write_bytes(b"x")
    write_json(path = model / "tokenizer_config.json", data = {"tokenizer_class": c71_local.TOKENIZER_CLASS})
    for lang in c71_local.FLEURS_LANGS:
        write_json(path = root / "fleurs" / lang / "test.jsonl", data = {"audio_path": f"{lang}/test/a.flac"})
        (root / "fleurs" / lang / "test").mkdir()
        (root / "fleurs" / lang / "test" / "a.flac").write_bytes(b"x")
    for group in c_runs.REPLAY_GROUPS:
        write_json(path = root / "replay" / group / "eval" / "samples.jsonl", data = {})
        write_json(path = root / "replay" / group / "eval" / "teacher" / "meta.json", data = {})
        (root / "replay" / group / "eval" / "raw").mkdir()
    python = root / "bin" / "python"
    python.parent.mkdir()
    python.write_text(data = FAKE_PYTHON.format(python = sys.executable), encoding = "utf-8")
    python.chmod(0o755)
    return [
        "--work-dir", str(root),
        "--dataset", str(root),
        "--model-dir", str(model),
        "--fleurs-dir", str(root / "fleurs"),
        "--replay-dir", str(root / "replay"),
        "--out-dir", str(root / "out"),
        "--py-audio", str(python),
        "--py-text", str(python),
        "--gpu-lock", str(root / "gpu.lock"),
        "--engine", "hf",
    ]


class MainTest(unittest.TestCase):
    def setUp(self):
        for signum in (c71_local.signal.SIGTERM, c71_local.signal.SIGINT):
            self.addCleanup(c71_local.signal.signal, signum, c71_local.signal.getsignal(signum))
        patcher = mock.patch.object(target = c71_local, attribute = "gpu_used_mib", return_value = None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def think_attention_of(self, calls):
        """各次思考检查（不含冒烟比较）的 --attn-implementation，并确认都带 --fit-memory。"""
        values = []
        for module, args in calls:
            if module == "pipeline.eval.think_check" and "--smoke" not in args:
                self.assertIn("--fit-memory", args)
                values.append(args[args.index("--attn-implementation") + 1])
        return values

    def test_main_runs_resumes_and_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = write_inputs(root = tmp)
            runs_root = Path(tmp) / "runs-on-f"
            argv += ["--runs-root", str(runs_root)]
            write_run(root = runs_root, key = "c0")
            write_run(root = runs_root, key = "c-mix")
            calls_path = Path(tmp) / "bin" / "calls.txt"
            # 只有两个运行下载齐全：先做冒烟比较，再做完它们和参考模型，退出码 3
            self.assertEqual(c71_local.main(argv = argv), 3)
            calls = read_calls(path = calls_path)
            self.assertEqual(len(calls), 1 + 2 * 5 + 3)
            self.assertEqual(calls[0][0], "pipeline.eval.think_check")
            self.assertIn("--smoke", calls[0][1])
            self.assertNotIn("--language-model", calls[0][1])
            self.assertEqual([module for module, _ in calls[1:4]], ["pipeline.eval.replay_gap"] * 3)
            self.assertEqual([module for module, _ in calls[-3:]], ["pipeline.eval.think_check"] * 3)
            self.assertEqual(self.think_attention_of(calls = calls), ["flash_attention_2"] * 3)
            # 运行的语言模型从 --runs-root 读
            for module, args in calls:
                if "--language-model" in args:
                    self.assertTrue(args[args.index("--language-model") + 1].startswith(str(runs_root)))
            status = json.loads(s = (Path(tmp) / "out" / "status.json").read_text(encoding = "utf-8"))
            self.assertEqual(status["phase"], "done_partial")
            self.assertEqual(status["results"]["c-mix:think"]["status"], "ok")
            self.assertEqual(status["think_attention"]["use"], "flash_attention_2")
            self.assertEqual(status["runs_root"], str(runs_root))
            self.assertIn("c-free_raw", status["runs_missing"])
            summary = (Path(tmp) / "out" / "summary" / "summary.md").read_text(encoding = "utf-8")
            self.assertIn("思考检查中语言模型全注意力层的实现：c0 flash_attention_2，c-mix flash_attention_2，ref flash_attention_2", summary)
            # 其余四个运行下载完后重新运行：已有结果的跳过，不再做冒烟比较（沿用已有结果的实现），只做新运行的 4 × 5 项，退出码 0
            for key in ("c-free_raw", "c-free_chat", "c-cont_raw", "c-cont_chat"):
                write_run(root = runs_root, key = key)
            calls_path.write_text("")
            self.assertEqual(c71_local.main(argv = argv), 0)
            calls = read_calls(path = calls_path)
            self.assertEqual(len(calls), 4 * 5)
            self.assertEqual(self.think_attention_of(calls = calls), ["flash_attention_2"] * 4)
            status = json.loads(s = (Path(tmp) / "out" / "status.json").read_text(encoding = "utf-8"))
            self.assertEqual(status["phase"], "done")
            self.assertIn("沿用已有思考检查结果", status["think_attention"]["reason"])
            summary = (Path(tmp) / "out" / "summary" / "summary.md").read_text(encoding = "utf-8")
            self.assertIn("| c-cont_chat |", summary)
            self.assertIn("## 与 C0 的比较", summary)

    def test_smoke_decides_sdpa(self):
        """冒烟比较的结论是 sdpa、或比较本身失败时，全部思考检查都用 sdpa；失败的比较下次重新做。"""
        for use, expected_reason in (("sdpa", "冒烟比较：假的比较"), ("fail", "冒烟比较没有完成")):
            with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(in_dict = os.environ, values = {"FAKE_SMOKE_USE": use}):
                argv = write_inputs(root = tmp) + ["--runs-root", str(Path(tmp) / "runs"), "--runs", "c0"]
                write_run(root = Path(tmp) / "runs", key = "c0")
                self.assertEqual(c71_local.main(argv = argv), 0)
                calls = read_calls(path = Path(tmp) / "bin" / "calls.txt")
                self.assertEqual(self.think_attention_of(calls = calls), ["sdpa"] * 2)
                status = json.loads(s = (Path(tmp) / "out" / "status.json").read_text(encoding = "utf-8"))
                self.assertEqual(status["think_attention"]["use"], "sdpa")
                self.assertIn(expected_reason, status["think_attention"]["reason"])

    def test_explicit_attention_skips_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = write_inputs(root = tmp) + ["--runs-root", str(Path(tmp) / "runs"), "--runs", "c0", "--think-attention", "sdpa"]
            write_run(root = Path(tmp) / "runs", key = "c0")
            self.assertEqual(c71_local.main(argv = argv), 0)
            calls = read_calls(path = Path(tmp) / "bin" / "calls.txt")
            self.assertFalse(any("--smoke" in args for _, args in calls))
            self.assertEqual(self.think_attention_of(calls = calls), ["sdpa"] * 2)

    def test_dry_run_shows_runs_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = Path(tmp) / "f-drive" / "c71-runs"
            base = write_inputs(root = tmp)
            argv = base + ["--runs-root", str(runs_root), "--dry-run"]
            output = io.StringIO()
            with contextlib.redirect_stdout(new_target = output):
                code = c71_local.main(argv = argv)
            text = output.getvalue()
            self.assertEqual(code, 0)
            self.assertIn(f"runs_root: {runs_root}", text)
            self.assertIn(f"运行目录 {runs_root}：还不存在", text)
            self.assertIn("先做冒烟比较", text)
            self.assertIn("[think_smoke] ", text)
            self.assertIn("--attn-implementation flash_attention_2", text)
            self.assertFalse((Path(tmp) / "out" / "status.json").exists())
            # 环境变量 C71_RUNS_ROOT 作为默认值
            with mock.patch.dict(in_dict = os.environ, values = {"C71_RUNS_ROOT": str(runs_root)}):
                args = c71_local.parse_args(argv = base)
            self.assertEqual(c71_local.resolve_paths(args = args)["runs_root"], runs_root)
            args = c71_local.parse_args(argv = base)
            if "C71_RUNS_ROOT" not in os.environ:
                self.assertEqual(c71_local.resolve_paths(args = args)["runs_root"], Path(tmp) / "out" / "runs")
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(new_target = io.StringIO()):
                c71_local.parse_args(argv = base + ["--think-attention", "eager"])


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(data = json.dumps(obj = data), encoding = "utf-8")


def fake_results(eval_root, label, gap, ppl_zh, closed, heard_en = None, rate = None):
    root = Path(eval_root) / label
    groups = {group: {"gap": gap} for group in c_runs.REPLAY_GROUPS}
    write_json(path = root / "replay_gap" / "summary.json", data = {"groups": groups, "mean_gap": gap})
    write_json(path = root / "lm_ppl.json", data = {lang: {"char_ppl": ppl_zh} for lang in c_runs.PPL_LANGS})
    write_json(path = root / "think" / "summary.json", data = {
        "closed": closed,
        "loop": 0.02,
        "capped": 0.06,
        "reasoning_tokens": {"median": 2100.0},
        "answer_tokens": {"median": 400.0},
    })
    if heard_en is not None:
        task = {"per_lang": {}, "ALL": {"verbatim_share": heard_en}}
        write_json(path = root / "probe" / "summary.json", data = {
            "config": {"model_dir": "/x", "projector": "/y", "prompt_format": ["chat", "cpt"]},
            "what_heard": {"heard_en": task, "heard_zh": task},
            "transcribe": {"transcribe": {"per_lang": {}, "ALL": {"rate_basic": rate}}, "transcribe_cpt": {"per_lang": {}, "ALL": {"rate_basic": 0.2}}},
        })


class SummaryTest(unittest.TestCase):
    def test_probe_config_is_not_a_task(self):
        overview = c_runs.probe_overview(probe = {
            "config": {"model_dir": "/x"},
            "translate": {"translate_en": {"per_lang": {}, "ALL": {"chrf": 12.0}}},
        })
        self.assertEqual(overview, {"translate": {"translate_en": {"chrf": 12.0}}})

    def test_summary_and_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = fake_paths(root = tmp)
            for key in ("c0", "c-mix"):
                write_run(root = paths["runs_root"], key = key)
            runs = c71_local.discover_runs(runs_root = paths["runs_root"], keys = list(c_runs.RUNS))
            # (标签, 回放差距, 中文每字困惑度, 写完 </think> 的比例, 「你听到了什么」逐字比例, 聊天格式转写错误率)
            for label, gap, ppl_zh, closed, heard_en, rate in (
                ("ref", 0.0001, 9.77, 0.93, None, None),
                ("c0", 0.30, 17.5, 0.60, 0.50, 0.80),
                ("c-mix", 0.10, 12.0, 0.85, 0.20, 0.70),
            ):
                fake_results(
                    eval_root = paths["eval_root"],
                    label = label,
                    gap = gap,
                    ppl_zh = ppl_zh,
                    closed = closed,
                    heard_en = heard_en,
                    rate = rate,
                )
            write_json(path = paths["out_dir"] / "summary" / "tables.json", data = {"fleurs_mean": {
                "c0": {"basic": 0.185, "runaway": 0.010},
                "c-mix": {"basic": 0.180, "runaway": 0.008},
            }})
            labels = c71_local.write_summary(
                paths = paths,
                runs = runs,
                keys = list(c_runs.RUNS),
                with_report = False,
                log = print,
            )
            text = (paths["out_dir"] / "summary" / "summary.md").read_text(encoding = "utf-8")
            data = json.loads(s = (paths["out_dir"] / "summary" / "summary.json").read_text(encoding = "utf-8"))
        self.assertEqual(labels, ["c0", "c-mix", "ref"])
        self.assertIn("projector 起点：第 2700 步", text)
        self.assertIn("| c-mix | 0.65 | 0.5 | 0.45 | 0.1000 |", text)
        rows = {line.split(" | ")[0]: line for line in text.splitlines() if line.startswith("| c-mix | 18.0")}
        self.assertEqual(len(rows), 1)
        row = next(iter(rows.values()))
        self.assertIn("18.0（-0.5）", row)
        self.assertIn("0.1000（-0.2000）", row)
        self.assertIn("12.00（0.69×）", row)
        follow = [line for line in text.splitlines() if line.startswith("| c-mix | 85.0（+25.0）")]
        self.assertEqual(len(follow), 1)
        self.assertIn("20.0（-30.0）", follow[0])
        self.assertIn("70.0（-10.0）", follow[0])
        c0_row = [line for line in text.splitlines() if line.startswith("| c0 | 18.5 |")]
        self.assertEqual(len(c0_row), 1)
        self.assertEqual(data["runs"]["c-mix"]["probe"]["what_heard"]["heard_en"]["verbatim_share"], 0.2)
        self.assertNotIn("config", data["runs"]["c-mix"]["probe"])
        self.assertEqual(data["runs"]["c0"]["train"]["dev_loss"], 0.674)


# 2B 语言模型的配置（qwen3.5-2b-mllm/config.json 的 text_config 中用到的几项）
CONFIG_2B = types.SimpleNamespace(
    layer_types = ["linear_attention"] * 3 * 6 + ["full_attention"] * 6,
    num_key_value_heads = 2,
    num_attention_heads = 8,
    head_dim = 256,
    linear_num_value_heads = 16,
    linear_num_key_heads = 16,
    linear_key_head_dim = 128,
    linear_value_head_dim = 128,
    linear_conv_kernel_dim = 4,
    vocab_size = 248320,
)


class ThinkBatchTest(unittest.TestCase):
    def test_per_sample_bytes(self):
        tokens = 219 + 16384
        expected = (
            6 * 2 * 2 * 512 * tokens            # KV 缓存
            + 2 * 512 * tokens                  # 拼接时多出的一份
            + 2 * 8 * 512 * tokens              # repeat_kv 的 K、V
            + 18 * 16 * 128 * 128 * 4           # 递推状态
            + 18 * (2 * 16 * 128 + 16 * 128) * 4 * 2
            + 4 * 248320 * 4
        )
        self.assertEqual(per_sample_bytes(config = CONFIG_2B, total_tokens = tokens), expected)
        self.assertAlmostEqual(expected / 2 ** 30, 0.355, places = 2)

    def test_per_sample_bytes_flash(self):
        tokens = 219 + 16384
        expected = (
            6 * 2 * 2 * 512 * tokens            # KV 缓存
            + 3 * 2 * 512 * tokens              # 去掉填充时 K、V 的复制（最多同时三份）
            + 18 * 16 * 128 * 128 * 4           # 递推状态
            + 18 * (2 * 16 * 128 + 16 * 128) * 4 * 2
            + 4 * 248320 * 4
        )
        flash = per_sample_bytes(config = CONFIG_2B, total_tokens = tokens, attn_implementation = "flash_attention_2")
        self.assertEqual(flash, expected)
        self.assertAlmostEqual(flash / 2 ** 30, 0.260, places = 2)
        self.assertLess(flash, per_sample_bytes(config = CONFIG_2B, total_tokens = tokens, attn_implementation = "sdpa"))
        with self.assertRaises(ValueError):
            per_sample_bytes(config = CONFIG_2B, total_tokens = tokens, attn_implementation = "eager")

    def test_fit_batch_size(self):
        gib = 2 ** 30
        # (注意力实现, 提问条数, 载入模型后的剩余显存 GiB, 应得的批大小)：RTX 5090 约剩 26 GiB，留 2 GiB → 两种实现都是
        # 3 批、每批 67 条（sdpa 最多 67 条，flash_attention_2 最多 92 条）；剩 25 GiB 时 sdpa 要 4 批，flash_attention_2 仍是 3 批；
        # 剩 28 GiB 时 flash_attention_2 才能分成 2 批；Colab G4（96 GB）一批 200 条；显存极少时至少 1 条；条数少时一批放下全部。
        cases = (
            ("sdpa", 200, 26, 67),
            ("sdpa", 200, 25, 50),
            ("flash_attention_2", 200, 26, 67),
            ("flash_attention_2", 200, 25, 67),
            ("flash_attention_2", 200, 28, 100),
            ("sdpa", 200, 88, 200),
            ("flash_attention_2", 200, 88, 200),
            ("sdpa", 200, 1, 1),
            ("flash_attention_2", 5, 88, 5),
        )
        for attention, count, free_gib, expected in cases:
            size = fit_batch_size(
                requested = 200,
                count = count,
                sample_bytes = per_sample_bytes(config = CONFIG_2B, total_tokens = 219 + 16384, attn_implementation = attention),
                free_bytes = free_gib * gib,
                reserve_bytes = 2 * gib,
            )
            self.assertEqual(size, expected, msg = f"{attention} {count} 条 {free_gib} GiB")

    def test_attention_setting_and_args(self):
        expected = {"": "sdpa", "text_config": "flash_attention_2", "audio_config": "sdpa", "vision_config": "sdpa"}
        self.assertEqual(think_check.attention_setting(name = "sdpa"), "sdpa")
        self.assertEqual(think_check.attention_setting(name = "flash_attention_2"), expected)
        # 与训练拼接批次时的设置相同（envs/mllm-eval 没有 mm_assistant_mask，导入不了训练代码时跳过这项核对）
        try:
            from pipeline.train.model import PACKED_ATTENTION
        except ImportError:
            PACKED_ATTENTION = expected
        self.assertEqual(PACKED_ATTENTION, expected)
        with self.assertRaises(ValueError):
            think_check.attention_setting(name = "eager")
        base = ["--model-dir", "m", "--replay-dir", "r", "--output-dir", "o"]
        self.assertEqual(think_check.parse_args(argv = base).attn_implementation, "sdpa")
        args = think_check.parse_args(argv = base + ["--attn-implementation", "flash_attention_2", "--smoke"])
        self.assertEqual((args.attn_implementation, args.smoke, args.smoke_prompts, args.smoke_max_new_tokens), ("flash_attention_2", True, 4, 256))
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(new_target = io.StringIO()):
            think_check.parse_args(argv = base + ["--attn-implementation", "eager"])

    def test_split_on_oom(self):
        calls = []

        def fake_generate(model, prompts, max_new_tokens, temperature, seed):
            calls.append((len(prompts), seed))
            if len(prompts) > 30:
                raise torch.OutOfMemoryError("fake")
            return [f"out-{sample_id}" for sample_id, _ in prompts]

        prompts = [(f"s{index}", [index]) for index in range(67)]
        outputs, sizes = generate_resilient(
            model = None,
            prompts = prompts,
            max_new_tokens = 8,
            temperature = 1.0,
            seed = 100,
            generate = fake_generate,
            log = lambda message: None,
        )
        self.assertEqual(outputs, [f"out-s{index}" for index in range(67)])
        self.assertEqual(sizes, [17, 17, 17, 16])
        self.assertEqual(calls[0], (67, 100))
        self.assertIn((17, 100), calls)
        self.assertIn((16, 100 + 2 * SPLIT_SEED_OFFSET), calls)

    def test_single_prompt_oom_raises(self):
        def fake_generate(model, prompts, max_new_tokens, temperature, seed):
            raise torch.OutOfMemoryError("fake")

        with self.assertRaises(torch.OutOfMemoryError):
            generate_resilient(
                model = None,
                prompts = [("s0", [1])],
                max_new_tokens = 8,
                temperature = 1.0,
                seed = 0,
                generate = fake_generate,
                log = lambda message: None,
            )


class FakeModel:
    """think_generate 测试用的假模型，只记着载入时的注意力实现。"""

    def __init__(self, attention):
        self.attention = attention


class ThinkFallbackTest(unittest.TestCase):
    """flash_attention_2 载入或生成出错时整项改用 sdpa 从头重做；显存不足仍按对半拆开处理。"""

    def setUp(self):
        self.loads = []
        self.calls = []
        self.messages = []
        self.prompts = [(f"s{index}", [index]) for index in range(10)]

    def load(self, fail_flash = False):
        def load(attention):
            self.loads.append(attention)
            if attention == "flash_attention_2" and fail_flash:
                raise ImportError("没有 flash_attn")
            return FakeModel(attention = attention)

        return load

    def think(self, load, generate, requested = "flash_attention_2"):
        return think_check.think_generate(
            load = load,
            prompts = self.prompts,
            requested = requested,
            choose_batch_size = lambda model, attention: {"flash_attention_2": 4, "sdpa": 3}[attention],
            max_new_tokens = 8,
            temperature = 1.0,
            seed = 100,
            generate = generate,
            log = self.messages.append,
        )

    def fake_generate(self, fail = None):
        """fail(model, prompts) 为真时抛出 RuntimeError。"""

        def generate(model, prompts, max_new_tokens, temperature, seed):
            self.calls.append((model.attention, len(prompts), seed))
            if fail is not None and fail(model, prompts):
                raise RuntimeError("CUDA error: no kernel image is available")
            return [f"{model.attention}-{sample_id}" for sample_id, _ in prompts]

        return generate

    def test_flash_works(self):
        result = self.think(load = self.load(), generate = self.fake_generate())
        self.assertEqual(self.loads, ["flash_attention_2"])
        self.assertEqual((result["used"], result["reason"], result["batch_size"]), ("flash_attention_2", None, 4))
        self.assertEqual(result["batches"], [4, 4, 2])
        self.assertEqual(result["outputs"], [f"flash_attention_2-s{index}" for index in range(10)])
        self.assertEqual([seed for _, _, seed in self.calls], [100, 101, 102])

    def test_load_failure_falls_back(self):
        result = self.think(load = self.load(fail_flash = True), generate = self.fake_generate())
        self.assertEqual(self.loads, ["flash_attention_2", "sdpa"])
        self.assertEqual((result["used"], result["batch_size"], result["batches"]), ("sdpa", 3, [3, 3, 3, 1]))
        self.assertIn("载入时出错：ImportError: 没有 flash_attn", result["reason"])
        self.assertEqual(result["outputs"], [f"sdpa-s{index}" for index in range(10)])

    def test_generation_failure_restarts_with_sdpa(self):
        # 第 2 批（s4 开始）在 flash_attention_2 下出错：整项用 sdpa 从第 1 批重做，批大小按 sdpa 重新估算
        failing = self.fake_generate(fail = lambda model, prompts: model.attention == "flash_attention_2" and prompts[0][0] == "s4")
        result = self.think(load = self.load(), generate = failing)
        self.assertEqual(self.loads, ["flash_attention_2", "sdpa"])
        self.assertEqual(result["used"], "sdpa")
        self.assertIn("生成时出错：RuntimeError: CUDA error", result["reason"])
        self.assertEqual(result["outputs"], [f"sdpa-s{index}" for index in range(10)])
        self.assertEqual(result["batches"], [3, 3, 3, 1])
        self.assertEqual(self.calls[:2], [("flash_attention_2", 4, 100), ("flash_attention_2", 4, 101)])
        self.assertEqual(self.calls[2:], [("sdpa", 3, 100), ("sdpa", 3, 101), ("sdpa", 3, 102), ("sdpa", 1, 103)])
        self.assertTrue(any("整项改用 sdpa 从头重做" in message for message in self.messages))

    def test_sdpa_failure_raises(self):
        failing = self.fake_generate(fail = lambda model, prompts: True)
        with self.assertRaises(RuntimeError):
            self.think(load = self.load(), generate = failing, requested = "sdpa")
        with self.assertRaises(RuntimeError):
            # flash_attention_2 出错改用 sdpa 后又出错：抛出，不再重试
            self.think(load = self.load(), generate = failing)

    def test_out_of_memory_is_not_a_fallback(self):
        def generate(model, prompts, max_new_tokens, temperature, seed):
            self.calls.append((model.attention, len(prompts), seed))
            if len(prompts) > 2:
                raise torch.OutOfMemoryError("fake")
            return [sample_id for sample_id, _ in prompts]

        result = self.think(load = self.load(), generate = generate)
        self.assertEqual(self.loads, ["flash_attention_2"])
        self.assertEqual(result["used"], "flash_attention_2")
        self.assertEqual(result["batches"], [2, 2, 2, 2, 2])
        self.assertEqual(result["outputs"], [f"s{index}" for index in range(10)])

        def always(model, prompts, max_new_tokens, temperature, seed):
            raise torch.OutOfMemoryError("fake")

        with self.assertRaises(torch.OutOfMemoryError):
            self.think(load = self.load(), generate = always)
        self.assertEqual(self.loads, ["flash_attention_2", "flash_attention_2"])


if __name__ == "__main__":
    unittest.main()
