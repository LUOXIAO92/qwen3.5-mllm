"""本机评测的 vLLM 引擎（任务 #71：pipeline/eval/vllm_host.py、vllm_generate.py 的扩展、audio_embeds.py，以及
think_check、replay_gap、c71_local、bench_local 的 --engine vllm）的单元测试：python -m unittest pipeline.eval.tests.test_vllm_engine

只在 CPU 上运行，不 import vLLM、不启动容器：

- WorkerTest：混合输入把音频向量放在音频占位符的位置；对数概率与提示对数概率的取法；每条请求覆盖采样设置；
  generate_all 用 make_prompt 构造提示、结束时调用 collect。
- HostTest：--engine vllm 的参数核对与默认任务目录；任务目录不覆盖别人的目录；结果缺条时报错。
- ThinkTest：用假的生成程序（--runner local）给出与 transformers 版相同的生成结果，两版写出的逐条结果与 token id 文件
  完全相同，汇总只差在引擎的记录；任务中每条的种子为 --seed 加序号，温度、上限、结束符与 transformers 版相同。
- ReplayTest：vLLM 的提示对数概率在计 NLL 的位置上取负，汇总与 transformers 版的汇总函数一致。
- C71Test：--engine vllm 时各项命令带 vLLM 的参数（参考模型直接加载、运行导出并保留、每项一个任务目录），运行另有严格版
  探针；默认顺序为 run；用假的 python 跑完驱动程序后，各运行的导出目录在最后一项之后删除，不做冒烟比较。
- BenchTest：简单基准的 vLLM 参数；每个模型先简单基准后 GPQA（run 顺序），导出目录只留给同一个模型的 GPQA。
"""

import json
import os
import signal
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from pipeline.colab import c_runs
from pipeline.eval import bench_local, c71_local, replay_gap, think_check, vllm_generate, vllm_host
from pipeline.eval.gpqa_vllm import EXPORT_MARKER
from pipeline.eval.tests.test_c71_local import fake_paths, read_calls, write_inputs, write_run
from pipeline.eval.tests.test_gpqa_vllm import FAKE_DOCKER, FakeEngine, write_config_dir, write_script
from pipeline.eval.tests.test_replay_evals import write_think_eval
from pipeline.train.tests.test_replay import FakeReplay

# 假的生成程序：按 FAKE_OUTPUTS（{编号: token id}）写 results.jsonl 与 engine.json；FAKE_PROMPT_LP 为真时另给每条提示的
# prompt_logprobs（第 i 个位置为 -0.01 × i）。
FAKE_WORKER = '''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
assert args[:2] == ["-m", "pipeline.eval.vllm_generate"], args
job_dir = Path(args[args.index("--job-dir") + 1])
outputs = json.loads(Path(os.environ["FAKE_OUTPUTS"]).read_text()) if os.environ.get("FAKE_OUTPUTS") else {}
requests = [json.loads(line) for line in (job_dir / "requests.jsonl").read_text().splitlines() if line.strip()]
lines = []
for r in requests:
    row = {"index": r["index"], "token_ids": outputs.get(str(r["index"]), [248046]), "finish_reason": "stop"}
    if r.get("prompt_logprobs") is not None:
        row["prompt_logprobs"] = [None] + [-0.01 * i for i in range(1, len(r["prompt_token_ids"]))]
    lines.append(json.dumps(row))
(job_dir / "results.jsonl").write_text("\\n".join(lines) + "\\n")
(job_dir / "engine.json").write_text(json.dumps({"version": "fake", "kv_cache_tokens": 1000, "max_inflight": 3, "seconds": 1.5,
    "tokens_per_second": 10.0}))
'''


class WorkerTest(unittest.TestCase):
    def test_mixed_prompt(self):
        rows = torch.arange(6, dtype = torch.float32).reshape(2, 3).to(dtype = torch.bfloat16)
        prompt = vllm_generate.mixed_prompt(token_ids = [5, 9, 9, 7], rows = rows, token_id = 9)
        self.assertEqual(prompt["prompt_token_ids"], [5, 9, 9, 7])
        self.assertEqual(prompt["prompt_is_token_ids"], [True, False, False, True])
        self.assertEqual(prompt["prompt_embeds"].dtype, torch.bfloat16)
        self.assertTrue(torch.equal(prompt["prompt_embeds"][1:3], rows))
        self.assertTrue(torch.equal(prompt["prompt_embeds"][[0, 3]], torch.zeros(size = (2, 3), dtype = torch.bfloat16)))
        with self.assertRaises(ValueError):
            vllm_generate.mixed_prompt(token_ids = [5, 9, 7], rows = rows, token_id = 9)

    def test_logprob_values(self):
        entries = [{7: SimpleNamespace(logprob = -0.5), 3: SimpleNamespace(logprob = -2.0)}, {1: SimpleNamespace(logprob = -0.1)}]
        self.assertEqual(vllm_generate.logprob_rows(entries = entries, steps = 1), [[[3, -2.0], [7, -0.5]]])
        prompt = [None, {4: SimpleNamespace(logprob = -1.5)}, {8: SimpleNamespace(logprob = -0.25), 2: SimpleNamespace(logprob = -0.1)}]
        self.assertEqual(vllm_generate.prompt_logprob_values(prompt_logprobs = prompt, prompt_ids = [1, 4, 8]), [None, -1.5, -0.25])

    def test_request_overrides(self):
        job = {"temperature": 1.0, "max_tokens": 100, "stop_token_ids": [248046, 248044]}
        request = {"max_tokens": 1, "temperature": 0.0, "logprob_token_ids": [32, 33], "prompt_logprobs": 0, "logprobs": None}
        params = vllm_generate.sampling_params(
            SamplingParams = lambda **kwargs: kwargs,
            final_only = "FINAL_ONLY",
            job = job,
            seed = 3,
            request = request,
        )
        self.assertEqual((params["max_tokens"], params["temperature"], params["seed"]), (1, 0.0, 3))
        self.assertEqual((params["logprob_token_ids"], params["prompt_logprobs"]), ([32, 33], 0))
        self.assertNotIn("logprobs", params)
        plain = vllm_generate.sampling_params(SamplingParams = lambda **kwargs: kwargs, final_only = "F", job = job, seed = 3)
        self.assertEqual((plain["max_tokens"], plain["temperature"]), (100, 1.0))

    def test_make_prompt_and_collect(self):
        engine = FakeEngine(steps = [1, 2])
        collected = []
        requests = [{"index": index, "seed": index, "prompt_token_ids": [index]} for index in range(2)]
        vllm_generate.generate_all(
            engine = engine,
            requests = requests,
            make_params = lambda request: {"index": request["index"]},
            max_inflight = 2,
            log = lambda message: None,
            make_prompt = lambda request: {"prompt_token_ids": [request["index"], 99]},
            collect = lambda index, output: collected.append(index),
        )
        self.assertEqual([prompt for _, prompt, _ in engine.added], [[0, 99], [1, 99]])
        self.assertEqual(sorted(collected), [0, 1])


class HostTest(unittest.TestCase):
    def parse(self, argv):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--language-model", default = None)
        vllm_host.add_arguments(parser = parser)
        args = parser.parse_args(args = argv)
        vllm_host.check_arguments(
            parser = parser,
            args = args,
            default_job_dir = Path("/out/vllm-job"),
            language_model = args.language_model,
        )
        return args

    def test_arguments(self):
        args = self.parse(argv = ["--engine", "vllm", "--vllm-model", "/ref-lm"])
        self.assertEqual((args.job_dir, args.docker_home), (Path("/out/vllm-job"), Path("/out/vllm-job/home")))
        self.assertEqual(args.gpu_memory_utilization, vllm_host.DEFAULT_GPU_MEMORY_UTILIZATION)
        self.assertIsNone(self.parse(argv = []).job_dir)
        for argv in (
            ["--engine", "vllm"],
            ["--engine", "vllm", "--vllm-model", "/r", "--language-model", "/lm"],
            ["--engine", "vllm", "--export-dir", "/e"],
        ):
            with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
                self.parse(argv = argv)
        self.parse(argv = ["--engine", "vllm", "--language-model", "/lm", "--export-dir", "/e", "--config-dir", "/c"])

    def test_job_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = vllm_host.prepare_job_dir(job_dir = Path(tmp) / "job")
            (job / "old.safetensors").write_bytes(b"x")
            job = vllm_host.prepare_job_dir(job_dir = job)
            self.assertEqual(sorted(path.name for path in job.iterdir()), ["job.json"])
            other = Path(tmp) / "other"
            other.mkdir()
            (other / "keep.txt").write_text("x")
            with self.assertRaises(RuntimeError):
                vllm_host.prepare_job_dir(job_dir = other)
            vllm_host.write_requests(job_dir = job, requests = [{"index": 0}], job = {"model": "m"})
            (job / "results.jsonl").write_text(json.dumps({"index": 0, "token_ids": [1], "finish_reason": "stop"}) + "\n")
            with self.assertRaises(RuntimeError):
                vllm_host.read_results(job_dir = job, indices = [0])
            (job / "engine.json").write_text("{}")
            results, engine = vllm_host.read_results(job_dir = job, indices = [0])
            self.assertEqual(results[0]["token_ids"], [1])
            with self.assertRaises(RuntimeError):
                vllm_host.read_results(job_dir = job, indices = [0, 1])


class ThinkTest(unittest.TestCase):
    def test_same_outputs_as_transformers(self):
        for signum in (signal.SIGTERM, signal.SIGINT):
            self.addCleanup(signal.signal, signum, signal.getsignal(signum))
        with tempfile.TemporaryDirectory() as tmp:
            write_think_eval(root = tmp)
            prompts = think_check.load_prompts(replay_dir = tmp, count = 5)
            outputs = [[5, 6, think_check.THINK_END_ID, 7, 248046], list(range(1000, 1012))]
            outputs_path = Path(tmp) / "outputs.json"
            outputs_path.write_text(json.dumps({str(index): ids for index, ids in enumerate(outputs)}))
            worker = write_script(path = Path(tmp) / "bin" / "worker-python", text = FAKE_WORKER)
            common = ["--model-dir", tmp, "--replay-dir", tmp, "--max-new-tokens", "12", "--seed", "7"]
            hf_args = think_check.parse_args(argv = common + ["--output-dir", str(Path(tmp) / "hf")])
            hf_args.output_dir.mkdir()
            think_check.write_outputs(
                args = hf_args,
                prompts = prompts,
                outputs = [np.asarray(a = ids) for ids in outputs],
                engine_fields = {"engine": "hf", "seconds": 1.0},
            )
            argv = common + [
                "--output-dir", str(Path(tmp) / "vllm"),
                "--engine", "vllm",
                "--vllm-model", str(Path(tmp) / "ref-lm"),
                "--runner", "local",
                "--worker-python", str(worker),
                "--job-dir", str(Path(tmp) / "job"),
                "--keep-job",
            ]
            with mock.patch.dict(in_dict = os.environ, values = {"FAKE_OUTPUTS": str(outputs_path)}):
                self.assertEqual(think_check.main(argv = argv), 0)
            for name in ("generations.jsonl", "generations.ids.i32"):
                self.assertEqual((Path(tmp) / "hf" / name).read_bytes(), (Path(tmp) / "vllm" / name).read_bytes())
            hf = json.loads((Path(tmp) / "hf" / "summary.json").read_text())
            vllm = json.loads((Path(tmp) / "vllm" / "summary.json").read_text())
            job = json.loads((Path(tmp) / "job" / "job.json").read_text())
            requests = [json.loads(line) for line in (Path(tmp) / "job" / "requests.jsonl").read_text().splitlines()]
        self.assertEqual(vllm["engine"], "vllm")
        self.assertEqual(vllm["vllm"]["version"], "fake")
        for key in ("closed", "ended", "capped", "loop", "reasoning_tokens", "answer_tokens", "tokens", "samples", "max_new_tokens"):
            self.assertEqual(hf[key], vllm[key])
        self.assertEqual([request["seed"] for request in requests], [7, 8])
        self.assertEqual([request["prompt_token_ids"] for request in requests], [ids for _, ids in prompts])
        self.assertEqual((job["temperature"], job["top_k"], job["top_p"], job["max_tokens"]), (1.0, 0, 1.0, 12))
        self.assertEqual(job["stop_token_ids"], list(think_check.STOP_IDS))
        self.assertEqual(job["max_model_len"], max(len(ids) for _, ids in prompts) + 12)


class ReplayTest(unittest.TestCase):
    def test_nll_and_aggregate(self):
        sample = {"ids": np.array([1, 2, 3, 4]), "labels": np.array([-100, -100, 3, 4])}
        nll = replay_gap.nll_from_prompt_logprobs(sample = sample, values = [None, -0.1, -0.2, -0.3])
        np.testing.assert_allclose(actual = nll, desired = [0.2, 0.3], rtol = 1e-6)
        with self.assertRaises(ValueError):
            replay_gap.nll_from_prompt_logprobs(sample = {"ids": np.array([1, 2]), "labels": np.array([1, 2])}, values = [None, -0.1])

    def test_evaluate_vllm(self):
        for signum in (signal.SIGTERM, signal.SIGINT):
            self.addCleanup(signal.signal, signum, signal.getsignal(signum))
        with tempfile.TemporaryDirectory() as tmp:
            FakeReplay(root = tmp).write_all(count = 4, split = "eval")
            worker = write_script(path = Path(tmp) / "bin" / "worker-python", text = FAKE_WORKER)
            argv = [
                "--model-dir", tmp,
                "--replay-dir", tmp,
                "--output-dir", str(Path(tmp) / "out"),
                "--engine", "vllm",
                "--vllm-model", str(Path(tmp) / "ref-lm"),
                "--runner", "local",
                "--worker-python", str(worker),
                "--job-dir", str(Path(tmp) / "job"),
            ]
            self.assertEqual(replay_gap.main(argv = argv), 0)
            summary = json.loads((Path(tmp) / "out" / "summary.json").read_text())
            self.assertFalse((Path(tmp) / "job").exists())
            # 与汇总函数直接算的结果相同：每条样本在计 NLL 的位置 p 上 NLL 为 0.01 × p
            expected = {}
            for group in replay_gap.GROUPS:
                scored = []
                for sample in replay_gap.group_samples(root = tmp, group = group, split = "eval"):
                    positions = np.nonzero(sample["labels"] != -100)[0]
                    scored.append((sample, (0.01 * positions).astype(np.float32)))
                expected[group] = replay_gap.aggregate_group(group = group, scored = scored)
        self.assertEqual(summary["engine"], "vllm")
        for group in replay_gap.GROUPS:
            self.assertEqual(summary["groups"][group]["tokens"], expected[group]["tokens"])
            self.assertAlmostEqual(summary["groups"][group]["nll"], expected[group]["nll"], places = 4)
            self.assertAlmostEqual(summary["groups"][group]["gap"], expected[group]["gap"], places = 4)


def vllm_paths(root):
    paths = fake_paths(root = root)
    paths.update({
        "engine": "vllm",
        "ref_lm": Path(root) / "ref-lm",
        "vllm_image": "vllm/vllm-openai:nightly-x86_64",
        "vllm_export_root": Path(root) / "vllm-lm",
        "vllm_job_root": Path(root) / "vllm-job",
        "vllm_home": Path(root) / "vllm-home",
        "gpu_memory_utilization": 0.9,
    })
    return paths


class C71Test(unittest.TestCase):
    def setUp(self):
        for signum in (signal.SIGTERM, signal.SIGINT):
            self.addCleanup(signal.signal, signum, signal.getsignal(signum))
        patcher = mock.patch.object(target = c71_local, attribute = "gpu_used_mib", return_value = None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = vllm_paths(root = tmp)
            write_run(root = paths["runs_root"], key = "c0")
            runs = c71_local.discover_runs(runs_root = paths["runs_root"], keys = ["c0"])
            plan = c71_local.build_plan(runs = runs, keys = ["c0"], paths = paths, order = "run")
            hf_plan = c71_local.build_plan(runs = runs, keys = ["c0"], paths = fake_paths(root = tmp), order = "run")

        def value(entry, flag):
            return entry["command"][entry["command"].index(flag) + 1]

        c0 = {entry["item"]: entry for entry in plan if entry["label"] == "c0"}
        ref = {entry["item"]: entry for entry in plan if entry["label"] == "ref"}
        self.assertEqual(list(c0), ["fleurs", "probe", "probe_strict", "lm_ppl", "replay_gap", "think"])
        self.assertEqual(set(ref), {"lm_ppl", "replay_gap", "think"})
        hf = {entry["item"]: entry for entry in hf_plan if entry["label"] == "c0"}
        self.assertNotIn("probe_strict", hf)
        for item, entry in c0.items():
            self.assertEqual(value(entry = entry, flag = "--engine"), "vllm")
            self.assertEqual(value(entry = entry, flag = "--export-dir"), str(Path(tmp) / "vllm-lm" / "c0"))
            self.assertEqual(value(entry = entry, flag = "--config-dir"), str(Path(tmp) / "ref-lm"))
            self.assertIn("--keep-export", entry["command"])
            self.assertEqual(value(entry = entry, flag = "--job-dir"), str(Path(tmp) / "vllm-job" / f"c0-{item}"))
            if item in hf:
                self.assertEqual(entry["command"][: len(hf[item]["command"])], hf[item]["command"])
        for entry in ref.values():
            self.assertEqual(value(entry = entry, flag = "--vllm-model"), str(Path(tmp) / "ref-lm"))
            self.assertNotIn("--export-dir", entry["command"])
        strict = c0["probe_strict"]["command"]
        self.assertEqual(value(entry = c0["probe_strict"], flag = "--instruction-set"), "strict")
        self.assertEqual(value(entry = c0["probe_strict"], flag = "--prompt-format"), "chat")
        self.assertEqual(strict[strict.index("--tasks") + 1: strict.index("--tasks") + 4], ["transcribe", "translate", "timestamps"])
        self.assertEqual(c0["probe_strict"]["marker"], Path(tmp) / "out" / "eval" / "c0" / "probe_strict" / "summary.json")
        # 导出只算在第一项上，vLLM 的估计比 transformers 短得多
        self.assertGreater(c0["fleurs"]["minutes"], c71_local.VLLM_MINUTES["fleurs"])
        self.assertEqual(c0["think"]["minutes"], c71_local.VLLM_MINUTES["think"])
        self.assertLess(sum(entry["minutes"] for entry in c0.values()), sum(entry["minutes"] for entry in hf.values()) / 3)

    def test_main_removes_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = write_inputs(root = tmp)
            runs_root = Path(tmp) / "runs"
            for key in ("c0", "c-mix"):
                write_run(root = runs_root, key = key)
            ref_lm = write_config_dir(path = Path(tmp) / "data" / "T70" / "ref-lm")
            (ref_lm / "model.safetensors").write_bytes(b"")
            docker_bin = Path(tmp) / "docker-bin"
            write_script(path = docker_bin / "docker", text = FAKE_DOCKER)
            exports = Path(tmp) / "tmp" / "T71c" / "vllm-lm"
            for key in ("c0", "c-mix"):
                (exports / key).mkdir(parents = True)
                (exports / key / EXPORT_MARKER).write_text("{}")
            stranger = exports / "c-free_raw"
            stranger.mkdir()
            argv += ["--runs-root", str(runs_root), "--runs", "c0,c-mix,c-free_raw", "--engine", "vllm"]
            with mock.patch.dict(in_dict = os.environ, values = {"PATH": f"{docker_bin}:{os.environ['PATH']}"}):
                self.assertEqual(c71_local.main(argv = argv), 3)
            calls = read_calls(path = Path(tmp) / "bin" / "calls.txt")
            status = json.loads((Path(tmp) / "out" / "status.json").read_text())
            remaining = sorted(path.name for path in exports.iterdir())
            log = (Path(tmp) / "out" / "driver.log").read_text()
        self.assertFalse(any("--smoke" in args for _, args in calls))
        self.assertEqual(status["order"], "run")
        self.assertEqual(status["think_attention"]["use"], "vllm")
        # 参考模型 3 项，两个运行各 6 项（含严格版探针）
        self.assertEqual(len(calls), 3 + 2 * 6)
        self.assertTrue(all("--engine" in args and args[args.index("--engine") + 1] == "vllm" for _, args in calls))
        self.assertEqual(remaining, ["c-free_raw"])
        self.assertIn("删除导出目录", log)


class BenchTest(unittest.TestCase):
    def test_simple_commands_and_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = bench_local.parse_args(argv = ["--work-dir", tmp, "--data-dir", str(Path(tmp) / "text")])
            paths = bench_local.resolve_paths(args = args)
            runs_root = Path(tmp) / "runs"
            from pipeline.eval.tests.test_text_bench import write_lm

            write_lm(runs_root = runs_root, key = "c0")
            ready, _ = bench_local.ready_runs(runs_root = runs_root, keys = ["c0"])
            plan = bench_local.build_plan(
                paths = paths,
                ready = ready,
                keys = ["c0"],
                include_ref = True,
                benches = ["mmlu_redux", "ifeval", "gpqa"],
                order = paths["order"],
            )

        def value(entry, flag):
            return entry["command"][entry["command"].index(flag) + 1]

        self.assertEqual([(entry["label"], entry["item"]) for entry in plan], [("ref", "simple"), ("ref", "gpqa"), ("c0", "simple"), ("c0", "gpqa")])
        ref_simple, ref_gpqa, c0_simple, c0_gpqa = plan
        self.assertEqual(value(entry = ref_simple, flag = "--engine"), "vllm")
        self.assertEqual(value(entry = ref_simple, flag = "--vllm-model"), str(Path(tmp) / "data" / "T70" / "ref-lm"))
        self.assertEqual(value(entry = c0_simple, flag = "--export-dir"), str(Path(tmp) / "tmp" / "T71c" / "vllm-lm" / "c0"))
        self.assertIn("--keep-export", c0_simple["command"])
        self.assertEqual(value(entry = c0_simple, flag = "--job-dir"), str(Path(tmp) / "tmp" / "T71c" / "vllm-job" / "c0-simple"))
        self.assertEqual(value(entry = c0_gpqa, flag = "--export-dir"), value(entry = c0_simple, flag = "--export-dir"))
        self.assertEqual(c0_gpqa["command"][1:3], ["-m", "pipeline.eval.gpqa_vllm"])
        # GPQA 沿用简单基准的导出，不再计导出的用时
        self.assertAlmostEqual(c0_gpqa["minutes"], ref_gpqa["minutes"])
        self.assertAlmostEqual(c0_simple["minutes"] - ref_simple["minutes"], bench_local.VLLM_EXPORT_MINUTES + c71_local.LM_LOAD_MINUTES)


if __name__ == "__main__":
    unittest.main()
