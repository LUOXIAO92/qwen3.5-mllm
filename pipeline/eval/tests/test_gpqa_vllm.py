"""GPQA 的 vLLM 版（任务 #71：pipeline/eval/gpqa_vllm.py、vllm_generate.py，replay/export_lm.py 的换权重导出，
bench_local.py 的 --gpqa-engine vllm）的单元测试：python -m unittest pipeline.eval.tests.test_gpqa_vllm

只在 CPU 上运行，不 import vLLM、不启动容器：

- ExportTest：导出时换上检查点的张量（数据类型按融合模型转换，只导出语言模型）；键或形状不一致时报错；导出目录的沿用、
  来源变化时重新导出、不覆盖也不删除别人的目录。
- SameAsTextBenchTest：用假的生成程序（--runner local）给出与 transformers 版相同的生成结果，两版写出的逐题结果和 token id
  文件完全相同，汇总只多出 vLLM 的设置；任务目录中的提示就是 text_bench 的提示，种子为 --gpqa-seed 加 CSV 的行号。
- WorkerTest：生成程序按顺序投放、同时在途不超过上限、结果按编号取回；每条的采样设置与种子；同时生成条数的上限。
- DockerTest：docker run 命令（WSL 的 GPU 设备与驱动库、只读与可写挂载、宿主机用户、无网络）；收到信号时结束生成程序并
  docker kill 容器；任务目录不覆盖别人的目录，结果缺题时报错。
- DriverTest：--gpqa-engine vllm 时参考模型与运行的命令行、默认路径、时间估计；用假的 python 与假的 docker 跑完驱动程序，
  汇总写明引擎；镜像不在时缺少基本输入。
"""

import json
import os
import signal
import stat
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from safetensors.torch import load_file, save_file

from pipeline import tokenizer_fix
from pipeline.colab import c_runs
from pipeline.eval import bench_local, c71_local, gpqa_vllm, text_bench, vllm_generate
from pipeline.eval.tests.test_text_bench import FakeTokenizer, fake_paths, write_csv, write_inputs, write_lm
from pipeline.replay import export_lm


def write_config_dir(path):
    """一个已修正分词器的小配置目录（tokenizer_fix.fix_model_dir 对它不做改动）。"""
    path = Path(path)
    path.mkdir(parents = True, exist_ok = True)
    (path / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"]}), encoding = "utf-8")
    (path / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": tokenizer_fix.TOKENIZER_CLASS}), encoding = "utf-8")
    tokenizer = {"pre_tokenizer": tokenizer_fix.PRE_TOKENIZER, "decoder": tokenizer_fix.DECODER}
    (path / "tokenizer.json").write_text(json.dumps(tokenizer), encoding = "utf-8")
    return path


def write_fused(path):
    """小的融合模型：两个语言模型张量（bfloat16 与 float32）和一个音频塔张量。"""
    path = Path(path)
    path.mkdir(parents = True, exist_ok = True)
    tensors = {
        "model.language_model.embed_tokens.weight": torch.arange(6, dtype = torch.float32).reshape(2, 3).to(dtype = torch.bfloat16),
        "model.language_model.layers.0.linear_attn.A_log": torch.zeros(4, dtype = torch.float32),
        "model.audio_tower.proj.weight": torch.ones(2, 2, dtype = torch.bfloat16),
    }
    save_file(tensors = tensors, filename = str(path / "model.safetensors"), metadata = {"format": "pt"})
    return path


def write_checkpoint(path, tensors = None):
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)
    if tensors is None:
        tensors = {
            "model.language_model.embed_tokens.weight": torch.full(size = (2, 3), fill_value = 0.5, dtype = torch.float32),
            "model.language_model.layers.0.linear_attn.A_log": torch.full(size = (4,), fill_value = -1.25, dtype = torch.bfloat16),
        }
    save_file(tensors = tensors, filename = str(path), metadata = {"format": "pt"})
    return path


def write_script(path, text):
    """写一个可执行的假程序（第一行为当前的 python）。"""
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(f"#!{sys.executable}\n" + text, encoding = "utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


class ExportTest(unittest.TestCase):
    def test_export_with_and_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            fused = write_fused(path = Path(tmp) / "fused")
            config = write_config_dir(path = Path(tmp) / "ref-lm")
            base = export_lm.export(mllm_dir = fused, original_dir = config, out_dir = Path(tmp) / "base")
            self.assertEqual(base, {"tensors": 2, "replaced": 0})
            tensors = load_file(filename = str(Path(tmp) / "base" / "model.safetensors"))
            self.assertEqual(sorted(tensors), ["model.language_model.embed_tokens.weight", "model.language_model.layers.0.linear_attn.A_log"])
            checkpoint = write_checkpoint(path = Path(tmp) / "run" / "language_model.safetensors")
            tuned = export_lm.export(
                mllm_dir = fused,
                original_dir = config,
                out_dir = Path(tmp) / "tuned",
                language_model = checkpoint,
            )
            self.assertEqual(tuned, {"tensors": 2, "replaced": 2})
            tensors = load_file(filename = str(Path(tmp) / "tuned" / "model.safetensors"))
            # 值来自检查点，数据类型按融合模型（与 transformers 的 load_state_dict 原地复制相同）
            embed = tensors["model.language_model.embed_tokens.weight"]
            self.assertEqual(embed.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(embed, torch.full(size = (2, 3), fill_value = 0.5, dtype = torch.bfloat16)))
            a_log = tensors["model.language_model.layers.0.linear_attn.A_log"]
            self.assertEqual(a_log.dtype, torch.float32)
            self.assertTrue(torch.equal(a_log, torch.full(size = (4,), fill_value = -1.25)))
            self.assertEqual(json.loads((Path(tmp) / "tuned" / "generation_config.json").read_text())["eos_token_id"], [248044, 248046])
            self.assertTrue((Path(tmp) / "tuned" / "config.json").exists())
            self.assertEqual(json.loads((Path(tmp) / "tuned" / "tokenizer_config.json").read_text())["tokenizer_class"], tokenizer_fix.TOKENIZER_CLASS)

    def test_mismatched_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            fused = write_fused(path = Path(tmp) / "fused")
            embed = torch.zeros(2, 3, dtype = torch.bfloat16)
            a_log = torch.zeros(4, dtype = torch.float32)
            cases = {
                "missing": {"model.language_model.embed_tokens.weight": embed},
                "extra": {
                    "model.language_model.embed_tokens.weight": embed,
                    "model.language_model.layers.0.linear_attn.A_log": a_log,
                    "model.language_model.lm_head.weight": embed.clone(),
                },
                "shape": {
                    "model.language_model.embed_tokens.weight": torch.zeros(3, 2, dtype = torch.bfloat16),
                    "model.language_model.layers.0.linear_attn.A_log": a_log,
                },
            }
            for name, tensors in cases.items():
                checkpoint = write_checkpoint(path = Path(tmp) / name / "language_model.safetensors", tensors = tensors)
                with self.assertRaises(ValueError, msg = name):
                    export_lm.language_model_tensors(mllm_dir = fused, language_model = checkpoint)

    def args(self, tmp, checkpoint):
        return gpqa_vllm.parse_args(argv = [
            "--model-dir", str(Path(tmp) / "fused"),
            "--language-model", str(checkpoint),
            "--export-dir", str(Path(tmp) / "vllm-lm" / "c0"),
            "--config-dir", str(Path(tmp) / "ref-lm"),
            "--data-dir", str(tmp),
            "--out-dir", str(tmp),
        ])

    def remove(self, directory, messages):
        gpqa_vllm.remove_marked(
            directory = directory,
            marker = gpqa_vllm.EXPORT_MARKER,
            log = messages.append,
            what = "导出目录",
        )

    def test_prepare_reuse_and_remove(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_fused(path = Path(tmp) / "fused")
            write_config_dir(path = Path(tmp) / "ref-lm")
            checkpoint = write_checkpoint(path = Path(tmp) / "run" / "language_model.safetensors")
            args = self.args(tmp = tmp, checkpoint = checkpoint)
            messages = []
            path, exported = gpqa_vllm.prepare_model(args = args, log = messages.append)
            self.assertTrue(exported)
            self.assertEqual(path, Path(tmp) / "vllm-lm" / "c0")
            marker = json.loads((path / gpqa_vllm.EXPORT_MARKER).read_text())
            self.assertTrue(marker["complete"])
            self.assertEqual(marker["replaced"], 2)
            # 来源相同：直接用，不再导出
            with mock.patch.object(target = export_lm, attribute = "export", side_effect = AssertionError("不应重新导出")):
                self.assertEqual(gpqa_vllm.prepare_model(args = args, log = messages.append), (path, True))
            # 检查点换了（大小不同）：删掉重新导出
            write_checkpoint(path = checkpoint, tensors = {
                "model.language_model.embed_tokens.weight": torch.ones(2, 3, dtype = torch.bfloat16),
                "model.language_model.layers.0.linear_attn.A_log": torch.ones(4, dtype = torch.float32),
            })
            gpqa_vllm.prepare_model(args = args, log = messages.append)
            tensors = load_file(filename = str(path / "model.safetensors"))
            self.assertTrue(torch.equal(tensors["model.language_model.embed_tokens.weight"], torch.ones(2, 3, dtype = torch.bfloat16)))
            self.remove(directory = path, messages = messages)
            self.assertFalse(path.exists())
            # 不是本程序导出的目录：不覆盖，也不删除
            foreign = Path(tmp) / "vllm-lm" / "c0"
            foreign.mkdir(parents = True)
            (foreign / "keep.txt").write_text("x")
            with self.assertRaises(RuntimeError):
                gpqa_vllm.prepare_model(args = args, log = messages.append)
            self.remove(directory = foreign, messages = messages)
            self.assertTrue((foreign / "keep.txt").exists())

    def test_vllm_model_is_used_directly(self):
        args = gpqa_vllm.parse_args(argv = ["--model-dir", "m", "--vllm-model", "ref-lm", "--data-dir", "d", "--out-dir", "o"])
        self.assertEqual(gpqa_vllm.prepare_model(args = args, log = print), (Path("ref-lm"), False))
        self.assertEqual(args.job_dir, Path("o") / "vllm-job")
        self.assertEqual(args.docker_home, Path("o") / "vllm-job" / "home")
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            gpqa_vllm.parse_args(argv = ["--model-dir", "m", "--data-dir", "d", "--out-dir", "o", "--language-model", "x"])
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            gpqa_vllm.parse_args(argv = ["--model-dir", "m", "--vllm-model", "r", "--export-dir", "e", "--data-dir", "d", "--out-dir", "o"])


def gpqa_csv(path, count = 5):
    rows = [
        {
            "Record ID": f"rec{i}",
            "Question": "q" * (40 - 6 * i),
            "Correct Answer": "right",
            "Incorrect Answer 1": "w1",
            "Incorrect Answer 2": "w2",
            "Incorrect Answer 3": "w3",
        }
        for i in range(count)
    ]
    write_csv(path = path, rows = rows)


def fake_generation(record_id, gold):
    """每题的假生成结果：rec0 没写完 </think>，rec1 找不到答案，rec2 答对，rec3 答错，rec4 答对且带 \\boxed。"""
    from pipeline.eval import think_check

    reasoning = [ord("x")] * (3 + int(record_id[3:]))
    if record_id == "rec0":
        return reasoning * 30
    answers = {
        "rec1": "no idea",
        "rec2": '"answer": "%s"' % gold,
        "rec3": '"answer": "%s"' % ("A" if gold != "A" else "B"),
        "rec4": "\\boxed{%s}" % gold,
    }
    return reasoning + [think_check.THINK_END_ID] + [ord(char) for char in answers[record_id]] + [think_check.STOP_IDS[0]]


class StopTokenizer(FakeTokenizer):
    """FakeTokenizer 加上结束符的编号。"""

    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": 248046, "<|endoftext|>": 248044}[token]


# 假的生成程序：按 FAKE_OUTPUTS（{题号: token id}）写 results.jsonl 与 engine.json
FAKE_WORKER = '''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
assert args[:2] == ["-m", "pipeline.eval.vllm_generate"], args
job_dir = Path(args[args.index("--job-dir") + 1])
outputs = json.loads(Path(os.environ["FAKE_OUTPUTS"]).read_text())
requests = [json.loads(line) for line in (job_dir / "requests.jsonl").read_text().splitlines() if line.strip()]
lines = [json.dumps({"index": r["index"], "token_ids": outputs[str(r["index"])], "finish_reason": "stop"}) for r in requests]
(job_dir / "results.jsonl").write_text("\\n".join(lines) + "\\n")
(job_dir / "engine.json").write_text(json.dumps({"version": "fake", "kv_cache_tokens": 1000, "max_inflight": 3, "seconds": 1.5,
    "tokens_per_second": 10.0}))
'''


class SameAsTextBenchTest(unittest.TestCase):
    def test_same_items_and_ids_as_transformers(self):
        from pipeline.eval import think_check

        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "text"
            gpqa_csv(path = data / text_bench.DATA["gpqa"])
            rows = text_bench.load_gpqa(path = data / text_bench.DATA["gpqa"])
            golds = {row["id"]: row["gold"] for row in rows}
            expected_prompts = text_bench.gpqa_prompts(tokenizer = StopTokenizer(), rows = rows)

            def hf_generate(load, prompts, requested, choose_batch_size, max_new_tokens, temperature, seed, log):
                outputs = [fake_generation(record_id = sample_id, gold = golds[sample_id]) for sample_id, _ in prompts]
                return {"outputs": outputs, "batches": [len(prompts)], "batch_size": len(prompts), "used": requested, "reason": None, "seconds": 2.0}

            outputs_path = Path(tmp) / "outputs.json"
            outputs_path.write_text(json.dumps({str(index): fake_generation(record_id = row["id"], gold = row["gold"]) for index, row in enumerate(rows)}))
            worker = write_script(path = Path(tmp) / "bin" / "worker-python", text = FAKE_WORKER)
            hf_dir = Path(tmp) / "hf"
            vllm_dir = Path(tmp) / "vllm"
            job_dir = Path(tmp) / "job"
            hf_dir.mkdir()
            common = ["--data-dir", str(data), "--gpqa-max-new-tokens", "100"]
            hf_args = text_bench.parse_args(argv = ["gpqa", "--model-dir", tmp, "--out-dir", str(hf_dir)] + common)
            argv = [
                "--model-dir", tmp,
                "--vllm-model", str(Path(tmp) / "ref-lm"),
                "--out-dir", str(vllm_dir),
                "--label", "ref",
                "--runner", "local",
                "--worker-python", str(worker),
                "--job-dir", str(job_dir),
                "--keep-job",
            ] + common
            for signum in (signal.SIGTERM, signal.SIGINT):
                self.addCleanup(signal.signal, signum, signal.getsignal(signum))
            with mock.patch.object(target = text_bench, attribute = "tokenizer_of", return_value = StopTokenizer()), \
                    mock.patch.object(target = think_check, attribute = "think_generate", side_effect = hf_generate), \
                    mock.patch.dict(in_dict = os.environ, values = {"FAKE_OUTPUTS": str(outputs_path)}):
                text_bench.run_gpqa(
                    args = hf_args,
                    data_dir = data,
                    out_dir = hf_dir,
                    log = lambda message: None,
                )
                self.assertEqual(gpqa_vllm.main(argv = argv), 0)
                # 已有结果时跳过，不再生成
                worker.write_text("raise SystemExit('不应再运行')\n")
                self.assertEqual(gpqa_vllm.main(argv = argv), 0)
            self.assertEqual((hf_dir / "gpqa.items.jsonl").read_bytes(), (vllm_dir / "gpqa.items.jsonl").read_bytes())
            self.assertEqual((hf_dir / "gpqa.ids.i32").read_bytes(), (vllm_dir / "gpqa.ids.i32").read_bytes())
            hf = json.loads((hf_dir / "gpqa.json").read_text(encoding = "utf-8"))
            vllm = json.loads((vllm_dir / "gpqa.json").read_text(encoding = "utf-8"))
            job = json.loads((job_dir / "job.json").read_text(encoding = "utf-8"))
            requests = [json.loads(line) for line in (job_dir / "requests.jsonl").read_text().splitlines()]
        # 交给 vLLM 的提示就是 text_bench 的提示，种子为 --gpqa-seed 加 CSV 中的行号
        self.assertEqual([(request["index"], request["prompt_token_ids"]) for request in requests], [(index, ids) for index, (_, ids) in enumerate(expected_prompts)])
        self.assertEqual([request["seed"] for request in requests], [1234 + index for index in range(5)])
        self.assertEqual(job["model"], str((Path(tmp) / "ref-lm").resolve()))
        self.assertEqual(job["max_model_len"], max(len(ids) for _, ids in expected_prompts) + 100)
        self.assertEqual((job["max_tokens"], job["temperature"], job["stop_token_ids"]), (100, 1.0, [248046, 248044]))
        self.assertEqual(hf["accuracy"], 0.4)
        self.assertEqual(hf["unclosed"], 0.2)
        self.assertEqual(hf["no_answer"], 0.2)
        engine_keys = {"attn_implementation_requested", "attn_implementation_used", "attn_fallback_reason", "batch_size", "batches", "seconds"}
        vllm_keys = {"engine", "seed_scheme", "language_model", "vllm", "finish_reasons", "seconds", "sampling", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty"}
        self.assertEqual({key: value for key, value in hf.items() if key not in engine_keys}, {key: value for key, value in vllm.items() if key not in vllm_keys})
        self.assertEqual(vllm["engine"], "vllm")
        self.assertEqual(vllm["finish_reasons"], {"stop": 5})
        self.assertEqual((vllm["vllm"]["runner"], vllm["vllm"]["max_inflight"], vllm["vllm"]["version"]), ("local", 3, "fake"))
        self.assertEqual(list(vllm)[: list(vllm).index("pred_share") + 1], list(hf)[: list(hf).index("pred_share") + 1])
        self.assertNotIn("right", json.dumps(vllm))

    def test_job_dir_removed_after_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "text"
            gpqa_csv(path = data / text_bench.DATA["gpqa"], count = 2)
            rows = text_bench.load_gpqa(path = data / text_bench.DATA["gpqa"])
            outputs_path = Path(tmp) / "outputs.json"
            outputs_path.write_text(json.dumps({str(index): fake_generation(record_id = row["id"], gold = row["gold"]) for index, row in enumerate(rows)}))
            worker = write_script(path = Path(tmp) / "bin" / "worker-python", text = FAKE_WORKER)
            argv = [
                "--model-dir", tmp,
                "--vllm-model", str(Path(tmp) / "ref-lm"),
                "--data-dir", str(data),
                "--out-dir", str(Path(tmp) / "out"),
                "--runner", "local",
                "--worker-python", str(worker),
            ]
            for signum in (signal.SIGTERM, signal.SIGINT):
                self.addCleanup(signal.signal, signum, signal.getsignal(signum))
            with mock.patch.object(target = text_bench, attribute = "tokenizer_of", return_value = StopTokenizer()), \
                    mock.patch.dict(in_dict = os.environ, values = {"FAKE_OUTPUTS": str(outputs_path)}):
                self.assertEqual(gpqa_vllm.main(argv = argv), 0)
                self.assertTrue((Path(tmp) / "out" / "gpqa.json").exists())
                self.assertFalse((Path(tmp) / "out" / "vllm-job").exists())
                # 生成程序出错：没有结果，任务目录保留
                worker.write_text(f"#!{sys.executable}\nraise SystemExit(5)\n")
                with self.assertRaises(RuntimeError):
                    gpqa_vllm.main(argv = argv + ["--force"])
                self.assertTrue((Path(tmp) / "out" / "vllm-job" / "job.json").exists())


class FakeEngine:
    """假的 vLLM 引擎：第 i 条在加入后的第 steps[i] 步结束，生成 [i] * (i + 1)。"""

    def __init__(self, steps):
        self.steps = steps
        self.active = {}
        self.added = []
        self.peak = 0

    def add_request(self, request_id, prompt, params):
        self.added.append((request_id, prompt["prompt_token_ids"], params))
        self.active[request_id] = [params["index"], self.steps[params["index"]]]
        self.peak = max(self.peak, len(self.active))

    def has_unfinished_requests(self):
        return bool(self.active)

    def do_log_stats_with_interval(self):
        pass

    def step(self):
        outputs = []
        for request_id, state in list(self.active.items()):
            state[1] -= 1
            finished = state[1] <= 0
            index = state[0]
            completion = SimpleNamespace(token_ids = [index] * (index + 1), finish_reason = "stop" if index % 2 == 0 else "length")
            outputs.append(SimpleNamespace(request_id = request_id, finished = finished, outputs = [completion]))
            if finished:
                del self.active[request_id]
        return outputs


class WorkerTest(unittest.TestCase):
    def requests(self, count):
        return [{"index": index, "seed": 1234 + index, "prompt_token_ids": [100 + index, 7]} for index in range(count)]

    def test_order_limit_and_results(self):
        steps = [5, 1, 1, 3, 2, 1, 4]
        engine = FakeEngine(steps = steps)
        requests = self.requests(count = len(steps))
        results = vllm_generate.generate_all(
            engine = engine,
            requests = requests,
            make_params = lambda request: {"index": request["index"], "seed": request["seed"]},
            max_inflight = 3,
            log = lambda message: None,
        )
        self.assertEqual([request_id for request_id, _, _ in engine.added], [f"r{index}" for index in range(len(steps))])
        self.assertEqual([prompt for _, prompt, _ in engine.added], [request["prompt_token_ids"] for request in requests])
        self.assertEqual([params["seed"] for _, _, params in engine.added], [1234 + index for index in range(len(steps))])
        self.assertEqual(engine.peak, 3)
        self.assertEqual(results, {index: ([index] * (index + 1), "stop" if index % 2 == 0 else "length") for index in range(len(steps))})

    def test_missing_result_raises(self):
        class Dropping(FakeEngine):
            def step(self):
                return [output for output in super().step() if output.request_id != "r1"]

        with self.assertRaises(RuntimeError):
            vllm_generate.generate_all(
                engine = Dropping(steps = [1, 1, 1]),
                requests = self.requests(count = 3),
                make_params = lambda request: {"index": request["index"]},
                max_inflight = 2,
                log = lambda message: None,
            )

    def test_sampling_params(self):
        job = {"temperature": 1.0, "max_tokens": 32768, "stop_token_ids": [248046, 248044]}
        params = vllm_generate.sampling_params(
            SamplingParams = lambda **kwargs: kwargs,
            final_only = "FINAL_ONLY",
            job = job,
            seed = 1241,
        )
        self.assertEqual(params["seed"], 1241)
        self.assertEqual(params["temperature"], 1.0)
        self.assertEqual((params["top_p"], params["top_k"], params["min_p"]), (1.0, 0, 0.0))
        self.assertEqual((params["presence_penalty"], params["frequency_penalty"], params["repetition_penalty"]), (0.0, 0.0, 1.0))
        self.assertEqual(params["max_tokens"], 32768)
        self.assertEqual(params["stop_token_ids"], [248046, 248044])
        self.assertEqual(params["output_kind"], "FINAL_ONLY")
        self.assertFalse(params["detokenize"])

    def test_inflight_limit(self):
        # (C, 倍数, 直接指定, max_num_seqs, 条数) -> 上限
        cases = {
            (49.9, 1.25, 0, 256, 198): 62,
            (49.9, 1.25, 40, 256, 198): 40,
            (49.9, 1.5, 0, 64, 198): 64,
            (49.9, 1.25, 0, 256, 8): 8,
            (0.3, 1.0, 0, 256, 8): 1,
        }
        for (concurrency, factor, explicit, max_num_seqs, count), expected in cases.items():
            limit = vllm_generate.inflight_limit(
                concurrency = concurrency,
                factor = factor,
                explicit = explicit,
                max_num_seqs = max_num_seqs,
                count = count,
            )
            self.assertEqual(limit, expected)

    def test_worker_imports_without_vllm(self):
        # 生成程序在模块层只用标准库，宿主机（以及测试）不装 vLLM 也能 import
        source = Path(vllm_generate.__file__).read_text(encoding = "utf-8")
        header = source.split("\ndef ", 1)[0]
        self.assertNotIn("import vllm", header)
        self.assertNotIn("from vllm", header)
        self.assertNotIn("import torch", header)


class DockerTest(unittest.TestCase):
    def test_command(self):
        args = gpqa_vllm.parse_args(argv = ["--model-dir", "m", "--vllm-model", "/models/ref-lm", "--data-dir", "d", "--out-dir", "o"])
        command = gpqa_vllm.docker_command(
            args = args,
            model_path = Path("/models/ref-lm"),
            job_dir = Path("/work/job/ref"),
            home = Path("/work/home"),
            name = "gpqa-vllm-ref-1",
            library_path = "/usr/local/nvidia/lib64:/usr/local/cuda/lib64",
        )
        repo = str(gpqa_vllm.REPO)

        def after(flag):
            return [command[index + 1] for index, item in enumerate(command) if item == flag]

        self.assertEqual(command[:4], ["docker", "run", "--rm", "--init"])
        self.assertEqual(after(flag = "--name"), ["gpqa-vllm-ref-1"])
        self.assertEqual(after(flag = "--device"), ["/dev/dxg"])
        self.assertEqual(after(flag = "--network"), ["none"])
        self.assertIn("--ipc=host", command)
        self.assertEqual(after(flag = "-u"), [f"{os.getuid()}:{os.getgid()}"])
        self.assertEqual(
            sorted(after(flag = "-v")),
            sorted([
                "/usr/lib/wsl:/usr/lib/wsl:ro",
                f"{repo}:{repo}:ro",
                "/models/ref-lm:/models/ref-lm:ro",
                "/work/job/ref:/work/job/ref:rw",
                "/work/home:/work/home:rw",
            ]),
        )
        environment = dict(item.split("=", 1) for item in after(flag = "-e"))
        self.assertEqual(environment["LD_LIBRARY_PATH"], "/usr/lib/wsl/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64")
        self.assertEqual(environment["HOME"], "/work/home")
        self.assertEqual(environment["PYTHONPATH"], repo)
        self.assertEqual(environment["VLLM_USE_FLASHINFER_SAMPLER"], "0")
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        self.assertEqual(after(flag = "--entrypoint"), ["python3"])
        tail = command[command.index("python3") + 1:]
        self.assertEqual(tail, [gpqa_vllm.IMAGE, "-m", "pipeline.eval.vllm_generate", "--job-dir", "/work/job/ref"])
        self.assertEqual(gpqa_vllm.container_name(label = "c-free_raw/x"), f"gpqa-vllm-c-free_raw-x-{os.getpid()}")

    def test_image_info(self):
        output = SimpleNamespace(returncode = 0, stdout = "sha256:abc\nPATH=/bin\nLD_LIBRARY_PATH=/usr/local/cuda/lib64\n\n", stderr = "")
        with mock.patch.object(target = gpqa_vllm.subprocess, attribute = "run", return_value = output):
            self.assertEqual(gpqa_vllm.image_info(image = "x"), {"id": "sha256:abc", "ld_library_path": "/usr/local/cuda/lib64"})
        with mock.patch.object(target = gpqa_vllm.subprocess, attribute = "run", return_value = SimpleNamespace(returncode = 1, stdout = "", stderr = "No such image")):
            with self.assertRaises(RuntimeError):
                gpqa_vllm.image_info(image = "x")

    def test_signal_kills_worker_and_container(self):
        def raise_stop(signum, frame):
            raise gpqa_vllm.Stop(signum = signum)

        killed = []
        real_run = gpqa_vllm.subprocess.run

        def fake_run(args, **kwargs):
            if args[:2] == ["docker", "kill"]:
                killed.append(args)
                return SimpleNamespace(returncode = 0)
            return real_run(args, **kwargs)

        previous = signal.signal(signal.SIGALRM, raise_stop)
        start = time.monotonic()
        try:
            signal.setitimer(signal.ITIMER_REAL, 0.5)
            with mock.patch.object(target = gpqa_vllm.subprocess, attribute = "run", side_effect = fake_run):
                with self.assertRaises(gpqa_vllm.Stop):
                    gpqa_vllm.run_worker(
                        command = [sys.executable, "-c", "import time; time.sleep(60)"],
                        name = "gpqa-vllm-test",
                        log = lambda message: None,
                    )
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
        self.assertEqual(killed, [["docker", "kill", "gpqa-vllm-test"]])
        self.assertLess(time.monotonic() - start, 20)

    def test_job_dir_and_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            foreign = Path(tmp) / "foreign"
            foreign.mkdir()
            (foreign / "keep.txt").write_text("x")
            with self.assertRaises(RuntimeError):
                gpqa_vllm.write_job(
                    job_dir = foreign,
                    job = {},
                    prompts = [("a", [1, 2])],
                    seed = 1,
                )
            job_dir = Path(tmp) / "job"
            # 第二次写同一个任务目录：删掉本程序的旧任务后重写
            for _ in range(2):
                gpqa_vllm.write_job(
                    job_dir = job_dir,
                    job = {"model": "m"},
                    prompts = [("a", [1, 2]), ("b", [3])],
                    seed = 10,
                )
            requests = [json.loads(line) for line in (job_dir / "requests.jsonl").read_text().splitlines()]
            self.assertEqual(requests, [{"index": 0, "seed": 10, "prompt_token_ids": [1, 2]}, {"index": 1, "seed": 11, "prompt_token_ids": [3]}])
            with self.assertRaises(RuntimeError):
                gpqa_vllm.read_results(job_dir = job_dir, count = 2)
            (job_dir / "results.jsonl").write_text(json.dumps({"index": 1, "token_ids": [5, 248046], "finish_reason": "stop"}) + "\n")
            (job_dir / "engine.json").write_text("{}")
            with self.assertRaises(RuntimeError):
                gpqa_vllm.read_results(job_dir = job_dir, count = 2)
            with open(job_dir / "results.jsonl", "a") as handle:
                handle.write(json.dumps({"index": 0, "token_ids": [7], "finish_reason": "length"}) + "\n")
            results, engine = gpqa_vllm.read_results(job_dir = job_dir, count = 2)
        self.assertEqual(results, [([7], "length"), ([5, 248046], "stop")])
        self.assertEqual(engine, {})


FAKE_DOCKER = '''import sys
args = sys.argv[1:]
if args[:2] == ["image", "inspect"]:
    if args[-1] == "missing/image":
        sys.stderr.write("Error: No such image: missing/image\\n")
        sys.exit(1)
    print("sha256:fakeimage")
    sys.exit(0)
sys.exit(3)
'''


class DriverTest(unittest.TestCase):
    def setUp(self):
        for signum in (bench_local.signal.SIGTERM, bench_local.signal.SIGINT):
            self.addCleanup(bench_local.signal.signal, signum, bench_local.signal.getsignal(signum))
        patcher = mock.patch.object(target = c71_local, attribute = "gpu_used_mib", return_value = None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def vllm_paths(self, tmp):
        paths = fake_paths(root = tmp)
        paths.update({
            "gpqa_max_new": 32768,
            "gpqa_engine": "vllm",
            "gpqa_ref_lm": Path(tmp) / "ref-lm",
            "gpqa_export_root": Path(tmp) / "vllm-lm",
            "gpqa_keep_export": False,
            "gpqa_inflight_factor": 1.25,
            "gpqa_vllm_image": "vllm/vllm-openai:nightly-x86_64",
            "gpqa_job_root": Path(tmp) / "vllm-job",
            "gpqa_docker_home": Path(tmp) / "vllm-home",
        })
        return paths

    def test_vllm_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.vllm_paths(tmp = tmp)
            write_lm(runs_root = paths["runs_root"], key = "c0")
            ready, _ = bench_local.ready_runs(runs_root = paths["runs_root"], keys = ["c0"])
            plan, hf_plan, keep_plan = [
                bench_local.build_plan(
                    paths = variant,
                    ready = ready,
                    keys = ["c0"],
                    include_ref = True,
                    benches = ["gpqa"],
                )
                for variant in (paths, dict(paths, gpqa_engine = "hf"), dict(paths, gpqa_keep_export = True))
            ]
        ref, c0 = plan

        def value(entry, flag):
            return entry["command"][entry["command"].index(flag) + 1]

        self.assertEqual(ref["command"][1:3], ["-m", "pipeline.eval.gpqa_vllm"])
        self.assertEqual(value(entry = ref, flag = "--vllm-model"), str(Path(tmp) / "ref-lm"))
        for flag in ("--language-model", "--export-dir", "--config-dir", "--keep-export"):
            self.assertNotIn(flag, ref["command"])
        self.assertEqual(value(entry = c0, flag = "--language-model"), str(ready["c0"]))
        self.assertEqual(value(entry = c0, flag = "--export-dir"), str(Path(tmp) / "vllm-lm" / "c0"))
        self.assertEqual(value(entry = c0, flag = "--config-dir"), str(Path(tmp) / "ref-lm"))
        self.assertEqual(value(entry = c0, flag = "--gpqa-max-new-tokens"), "32768")
        self.assertEqual(value(entry = c0, flag = "--inflight-factor"), "1.25")
        self.assertEqual(value(entry = c0, flag = "--image"), "vllm/vllm-openai:nightly-x86_64")
        self.assertEqual(value(entry = c0, flag = "--job-dir"), str(Path(tmp) / "vllm-job" / "c0"))
        self.assertEqual(value(entry = ref, flag = "--job-dir"), str(Path(tmp) / "vllm-job" / "ref"))
        self.assertEqual(value(entry = c0, flag = "--docker-home"), str(Path(tmp) / "vllm-home"))
        self.assertEqual(value(entry = c0, flag = "--out-dir"), str(Path(tmp) / "out" / "c0"))
        self.assertNotIn("--vllm-model", c0["command"])
        self.assertNotIn("--keep-export", c0["command"])
        self.assertIn("--keep-export", keep_plan[1]["command"])
        self.assertEqual(c0["marker"].paths, [Path(tmp) / "out" / "c0" / "gpqa.json"])
        # 估计：运行要先导出，比参考模型多一点；比 transformers 版短得多
        self.assertAlmostEqual(c0["minutes"] - ref["minutes"], bench_local.VLLM_EXPORT_MINUTES + c71_local.LM_LOAD_MINUTES)
        self.assertLess(ref["minutes"], 45)
        self.assertGreater(hf_plan[0]["minutes"], 2 * ref["minutes"])
        self.assertEqual(hf_plan[0]["command"][1:4], ["-m", "pipeline.eval.text_bench", "gpqa"])

    def test_defaults(self):
        args = bench_local.parse_args(argv = ["--work-dir", "/w", "--data-dir", "/d"])
        paths = bench_local.resolve_paths(args = args)
        self.assertEqual((paths["gpqa_engine"], paths["engine"], paths["order"]), ("vllm", "vllm", "run"))
        self.assertEqual(paths["gpqa_ref_lm"], Path("/w/data/T70/ref-lm"))
        self.assertEqual(paths["gpqa_export_root"], Path("/w/tmp/T71c/vllm-lm"))
        self.assertEqual(paths["gpqa_job_root"], Path("/w/tmp/T71c/vllm-job"))
        self.assertEqual(paths["gpqa_docker_home"], Path("/w/tmp/T71c/vllm-home"))
        self.assertEqual(paths["gpqa_vllm_image"], "vllm/vllm-openai:nightly-x86_64")
        self.assertEqual(paths["gpqa_inflight_factor"], 1.25)
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            bench_local.parse_args(argv = ["--work-dir", "/w", "--data-dir", "/d", "--gpqa-engine", "sglang"])

    def driver_inputs(self, tmp):
        argv = write_inputs(root = tmp)
        ref_lm = write_config_dir(path = Path(tmp) / "data" / "T70" / "ref-lm")
        (ref_lm / "model.safetensors").write_bytes(b"")
        gpqa = Path(tmp) / "text" / text_bench.DATA["gpqa"]
        gpqa.parent.mkdir(parents = True)
        gpqa.write_text("Record ID\n", encoding = "utf-8")
        for key in c_runs.RUNS:
            write_lm(runs_root = Path(tmp) / "runs", key = key)
        docker_bin = Path(tmp) / "docker-bin"
        write_script(path = docker_bin / "docker", text = FAKE_DOCKER)
        environment = {"FAKE_CALLS": str(Path(tmp) / "calls.txt"), "PATH": f"{docker_bin}:{os.environ['PATH']}"}
        return argv + ["--benches", "gpqa", "--gpqa-engine", "vllm", "--gpqa-max-new-tokens", "32768"], environment

    def test_main_with_vllm_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv, environment = self.driver_inputs(tmp = tmp)
            with mock.patch.dict(in_dict = os.environ, values = environment):
                self.assertEqual(bench_local.main(argv = argv), 0)
            calls = [json.loads(line) for line in Path(environment["FAKE_CALLS"]).read_text().splitlines()]
            summary = (Path(tmp) / "out" / "summary.md").read_text(encoding = "utf-8")
        self.assertEqual([call[0] for call in calls], ["pipeline.eval.gpqa_vllm"] * (1 + len(c_runs.RUNS)))
        self.assertEqual([call[call.index("--label") + 1] for call in calls], ["ref"] + list(c_runs.RUNS))
        self.assertIn("--vllm-model", calls[0])
        self.assertTrue(all("--export-dir" in call for call in calls[1:]))
        self.assertIn("| ref |  |  |  |  |  |  |  | 60.0 |", summary)
        self.assertIn("| c0 |  |  |  |  |  |  |  | 50.0（-10.0） |", summary)
        self.assertIn("最多 32,768 个新 token、vLLM 生成", summary)

    def test_missing_image_is_a_missing_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv, environment = self.driver_inputs(tmp = tmp)
            with mock.patch.dict(in_dict = os.environ, values = environment):
                self.assertEqual(bench_local.main(argv = argv + ["--gpqa-vllm-image", "missing/image"]), 2)
            self.assertFalse(Path(environment["FAKE_CALLS"]).exists())
            status = json.loads((Path(tmp) / "out" / "status.json").read_text(encoding = "utf-8"))
        self.assertEqual(status["phase"], "error")
        self.assertTrue(any("missing/image" in text for text in status["error"]))

    def test_settings_text(self):
        results = {
            "ref": {"gpqa": {"engine": "vllm", "max_new_tokens": 32768}},
            "c0": {"gpqa": {"max_new_tokens": 16384}},
        }
        self.assertEqual(bench_local.gpqa_settings_text(results = results, labels = ["ref"]), "最多 32,768 个新 token、vLLM 生成（模型卡建议 32,768）")
        text = bench_local.gpqa_settings_text(results = results, labels = ["ref", "c0"])
        self.assertIn("ref 最多 32,768 个新 token、vLLM 生成；c0 最多 16,384 个新 token、transformers 生成", text)


if __name__ == "__main__":
    unittest.main()
