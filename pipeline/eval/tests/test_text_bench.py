"""文本基准（任务 #71，pipeline/eval/text_bench.py、bench_local.py、ifeval/score.py）的单元测试：
python -m unittest pipeline.eval.tests.test_text_bench

只在 CPU 上运行，用很小的假数据：

- DataTest：等距抽题；MMMLU 各语言同一组行号、行数不同时报错；GPQA 选项打乱固定且可还原、标准答案字母随之变化。
- PromptTest：选择题的英文与中文模板；GPQA 答案的解析（取 </think> 之后最靠后的一个）。
- LetterScoreTest：右侧填充时取每条最后一个真实位置的字母对数概率；准确率、字母概率和、按组统计。
- PlanTest：先做全部模型的简单基准、最后做 GPQA；参考模型不带 --language-model；运行按语言模型权重是否完整决定。
- MainTest：用假的 python 跑完整个驱动程序：续跑时跳过已有结果、IFEval 打分、GPQA 数据缺少时跳过（退出码 3）、汇总表。
"""

import csv
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from pipeline.colab import c_runs
from pipeline.eval import bench_local, c71_local, text_bench
from pipeline.eval.tests.test_c71_local import write_safetensors


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)
    with open(file = path, mode = "w", encoding = "utf-8", newline = "") as handle:
        writer = csv.DictWriter(handle, fieldnames = list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class DataTest(unittest.TestCase):
    def test_spread(self):
        self.assertEqual(text_bench.spread(rows = list(range(10)), limit = None), list(range(10)))
        self.assertEqual(text_bench.spread(rows = list(range(10)), limit = 20), list(range(10)))
        self.assertEqual(text_bench.spread(rows = list(range(3000)), limit = 3), [0, 1000, 2000])

    def test_mmmlu_same_rows_for_every_language(self):
        indices = text_bench.mmmlu_indices(count = 14042)
        self.assertEqual(indices, text_bench.mmmlu_indices(count = 14042))
        self.assertEqual(len(set(indices)), text_bench.MMMLU_PER_LANG)
        self.assertEqual(indices, sorted(indices))
        with tempfile.TemporaryDirectory() as tmp:
            for lang, answer in (("de", "A"), ("ja", "B")):
                rows = [
                    {"": str(i), "Question": f"q{i}", "A": "None", "B": "NA", "C": "c", "D": "d", "Answer": answer, "Subject": "virology"}
                    for i in range(30)
                ]
                write_csv(path = Path(tmp) / f"mmlu_{text_bench.MMMLU_FILES[lang]}.csv", rows = rows)
            loaded = text_bench.load_mmmlu(root = Path(tmp), langs = ["de", "ja"], per_lang = 5)
            self.assertEqual(len(loaded), 10)
            de = [row["id"].split("/")[1] for row in loaded if row["group"] == "de"]
            ja = [row["id"].split("/")[1] for row in loaded if row["group"] == "ja"]
            self.assertEqual(de, ja)
            # 选项中的 "None"、"NA" 保持原样，标准答案取各语言自己的
            self.assertEqual(loaded[0]["options"][:2], ["None", "NA"])
            self.assertEqual({row["gold"] for row in loaded if row["group"] == "ja"}, {"B"})
            write_csv(path = Path(tmp) / "mmlu_JA-JP.csv", rows = rows[:20])
            with self.assertRaises(ValueError):
                text_bench.load_mmmlu(root = Path(tmp), langs = ["de", "ja"], per_lang = 5)

    def test_gpqa_permutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                {
                    "Record ID": f"rec{i}",
                    "Question": f" question {i} ",
                    "Correct Answer": "right",
                    "Incorrect Answer 1": "wrong1",
                    "Incorrect Answer 2": "wrong2",
                    "Incorrect Answer 3": "wrong3",
                }
                for i in range(40)
            ]
            write_csv(path = Path(tmp) / "gpqa.csv", rows = rows)
            first = text_bench.load_gpqa(path = Path(tmp) / "gpqa.csv")
            second = text_bench.load_gpqa(path = Path(tmp) / "gpqa.csv")
        self.assertEqual(first, second)
        golds = set()
        for row in first:
            self.assertEqual(sorted(row["permutation"]), [0, 1, 2, 3])
            self.assertEqual(row["options"][text_bench.LETTERS.index(row["gold"])], "right")
            self.assertEqual(row["question"], row["question"].strip())
            golds.add(row["gold"])
        self.assertEqual(golds, set(text_bench.LETTERS))
        self.assertEqual(text_bench.gpqa_permutation(record_id = "rec0"), first[0]["permutation"])


class PromptTest(unittest.TestCase):
    def test_templates(self):
        row = {"id": "college_physics/3", "group": "college_physics", "subject": "college_physics", "question": " Q? ", "options": ["a", "b", "c", "d"]}
        text = text_bench.mc_text(row = row, template = text_bench.EN_TEMPLATE)
        self.assertTrue(text.startswith("The following is a multiple choice question about college physics.\n\nQ?\n\nA. a\n"))
        self.assertTrue(text.endswith("Answer with only the letter (A, B, C, or D) of the correct option."))
        zh = text_bench.mc_text(row = {"id": "x/1", "group": "x", "question": "问题", "options": ["甲", "乙", "丙", "丁"]}, template = text_bench.ZH_TEMPLATE)
        self.assertIn("问题\n\nA. 甲\nB. 乙\nC. 丙\nD. 丁\n\n只回答正确选项的字母", zh)

    def test_parse_choice(self):
        cases = {
            'The final choice: {"answer": "C"}': "C",
            "\\boxed{B} ... later 'answer': 'D'": "D",
            '"answer": "A" and finally \\boxed{\\text{B}}': "B",
            "Answer: A": "A",
            "answer：（B）": "B",
            '"answer": "E"': None,
            "no letter": None,
        }
        for text, expected in cases.items():
            self.assertEqual(text_bench.parse_choice(text = text), expected, text)


class FakeLanguageModel(torch.nn.Module):
    """隐状态为 token id 对 8 取余的 one-hot；词表投影为单位矩阵，所以最后位置的对数概率在该位置的 token 上最大。"""

    def forward(self, input_ids, use_cache):
        return type("Output", (), {"last_hidden_state": torch.nn.functional.one_hot(input_ids % 8, num_classes = 8).float() * 10})()


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = FakeLanguageModel()
        self.lm_head = torch.nn.Linear(in_features = 8, out_features = 8, bias = False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.eye(8))


class LetterScoreTest(unittest.TestCase):
    def test_last_real_position_with_right_padding(self):
        # 填充 token 248044 对 8 取余为 4（D 的位置）；取错位置时会答成 D
        prompts = [[7, 7, 1], [7, 2], [7, 7, 7, 7, 3], [4]]
        logprobs = text_bench.letter_logprobs(model = FakeModel(), prompts = prompts, letters = [1, 2, 3, 4], max_batch_tokens = 1000)
        preds = [text_bench.LETTERS[max(range(4), key = lambda index: values[index])] for values in logprobs]
        self.assertEqual(preds, ["A", "B", "C", "D"])
        rows = [{"id": f"s/{i}", "group": "g1" if i < 2 else "g2", "gold": gold} for i, gold in enumerate("ABCA")]
        items, stats = text_bench.mc_summary(rows = rows, logprobs = logprobs)
        self.assertEqual([item["correct"] for item in items], [True, True, True, False])
        self.assertEqual(stats["accuracy"], 0.75)
        self.assertEqual(stats["groups"], {"g1": {"n": 2, "accuracy": 1.0}, "g2": {"n": 2, "accuracy": 0.5}})
        self.assertGreater(stats["letter_mass_mean"], 0.9)
        self.assertEqual(stats["pred_share"]["D"], 0.25)


class FakeTokenizer:
    """chat template 原样返回用户消息；每个字符一个 token（字符编码），解码时还原。"""

    def apply_chat_template(self, conversation, tokenize, add_generation_prompt, enable_thinking):
        return conversation[0]["content"]

    def __call__(self, text, add_special_tokens):
        return {"input_ids": [ord(char) for char in text]}

    def decode(self, token_ids, skip_special_tokens):
        return "".join(chr(token) for token in token_ids if token < 0x110000)


class GpqaRunTest(unittest.TestCase):
    def test_sorted_batches_restore_order_and_statuses(self):
        from pipeline.eval import think_check

        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                {
                    "Record ID": f"rec{i}",
                    "Question": "q" * (50 - 10 * i),
                    "Correct Answer": "right",
                    "Incorrect Answer 1": "w1",
                    "Incorrect Answer 2": "w2",
                    "Incorrect Answer 3": "w3",
                }
                for i in range(4)
            ]
            data = Path(tmp) / "text"
            write_csv(path = data / text_bench.DATA["gpqa"], rows = rows)
            golds = {row["id"]: row["gold"] for row in text_bench.load_gpqa(path = data / text_bench.DATA["gpqa"])}
            seen = {}

            def fake_generate(load, prompts, requested, choose_batch_size, max_new_tokens, temperature, seed, log):
                seen["order"] = [sample_id for sample_id, _ in prompts]
                outputs = []
                for sample_id, _ in prompts:
                    reasoning = [ord("x")] * 5
                    if sample_id == "rec0":
                        outputs.append(reasoning * 20)
                        continue
                    if sample_id == "rec1":
                        answer = "I cannot decide."
                    elif sample_id == "rec2":
                        answer = '{"answer": "%s"}' % golds[sample_id]
                    else:
                        answer = "\\boxed{%s}" % ("A" if golds[sample_id] != "A" else "B")
                    outputs.append(reasoning + [think_check.THINK_END_ID] + [ord(char) for char in answer] + [think_check.STOP_IDS[0]])
                return {"outputs": outputs, "batches": [4], "batch_size": 4, "used": requested, "reason": None, "seconds": 1.0}

            args = text_bench.parse_args(argv = ["gpqa", "--model-dir", tmp, "--data-dir", str(data), "--out-dir", tmp, "--gpqa-max-new-tokens", "100"])
            with mock.patch.object(target = text_bench, attribute = "tokenizer_of", return_value = FakeTokenizer()), \
                    mock.patch.object(target = think_check, attribute = "think_generate", side_effect = fake_generate):
                text_bench.run_gpqa(args = args, data_dir = data, out_dir = Path(tmp), log = lambda message: None)
            summary = json.loads((Path(tmp) / "gpqa.json").read_text(encoding = "utf-8"))
            items = [json.loads(line) for line in (Path(tmp) / "gpqa.items.jsonl").read_text(encoding = "utf-8").splitlines()]
            ids_size = (Path(tmp) / "gpqa.ids.i32").stat().st_size
        # 题目越往后越短：按提示长度排序后倒过来生成，结果按原顺序写出
        self.assertEqual(seen["order"], ["rec3", "rec2", "rec1", "rec0"])
        self.assertEqual([item["id"] for item in items], ["rec0", "rec1", "rec2", "rec3"])
        self.assertEqual([item["status"] for item in items], ["unclosed", "no_answer", "answered", "answered"])
        self.assertEqual([item["correct"] for item in items], [False, False, True, False])
        self.assertEqual(summary["accuracy"], 0.25)
        self.assertEqual(summary["unclosed"], 0.25)
        self.assertEqual(summary["no_answer"], 0.25)
        self.assertEqual(summary["accuracy_answered"], 0.5)
        self.assertEqual(summary["closed"], 0.75)
        self.assertEqual(ids_size, 4 * sum(item["tokens"] for item in items))
        # 结果中没有题目与选项的文字
        self.assertNotIn("question", items[0])
        self.assertNotIn("options", items[0])
        self.assertNotIn("right", json.dumps(items) + json.dumps(summary))


def fake_paths(root):
    root = Path(root)
    return {
        "model_dir": root / "model",
        "data_dir": root / "text",
        "runs_root": root / "runs",
        "out_dir": root / "out",
        "python_text": root / "bin" / "python",
        "python_ifeval": root / "bin" / "python-ifeval",
        "gpu_lock": root / "gpu.lock",
        "gpqa_max_new": 16384,
        "gpqa_reserve_gb": 8.0,
        "ifeval_problem": None,
    }


def write_lm(runs_root, key, truncate = 0):
    write_safetensors(path = Path(runs_root) / c_runs.run_name(key = key) / bench_local.LM_FILE, numbers = 16, truncate = truncate)


class PlanTest(unittest.TestCase):
    def test_order_and_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = fake_paths(root = tmp)
            write_lm(runs_root = paths["runs_root"], key = "c0")
            write_lm(runs_root = paths["runs_root"], key = "c-mix", truncate = 3)
            ready, missing = bench_local.ready_runs(runs_root = paths["runs_root"], keys = list(c_runs.RUNS))
            self.assertEqual(list(ready), ["c0"])
            self.assertIn("c-mix", missing)
            plan = bench_local.build_plan(
                paths = paths,
                ready = ready,
                keys = list(c_runs.RUNS),
                include_ref = True,
                benches = list(bench_local.BENCH_CHOICES),
            )
        self.assertEqual([(entry["item"], entry["label"]) for entry in plan], [("simple", "ref"), ("simple", "c0"), ("gpqa", "ref"), ("gpqa", "c0")])
        ref_simple, c0_simple, ref_gpqa, c0_gpqa = plan
        self.assertEqual(ref_simple["command"][1:7], ["-m", "pipeline.eval.text_bench", "mmlu_redux", "ceval", "mmmlu", "ifeval"])
        self.assertNotIn("--language-model", ref_simple["command"])
        self.assertEqual(c0_simple["command"][c0_simple["command"].index("--language-model") + 1], str(ready["c0"]))
        self.assertEqual(c0_gpqa["command"][3], "gpqa")
        self.assertEqual(c0_gpqa["command"][-4:], ["--gpqa-max-new-tokens", "16384", "--memory-reserve-gb", "8.0"])
        self.assertGreater(c0_gpqa["minutes"], ref_gpqa["minutes"])
        self.assertGreater(ref_gpqa["minutes"], 3 * ref_simple["minutes"])
        self.assertEqual(len(c0_simple["marker"].paths), 4)
        self.assertEqual(c0_simple["marker"].paths[-1].name, "ifeval.responses.jsonl")


FAKE_TEXT_BENCH = '''#!{python}
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if args[0] == "-c":
    sys.exit(0)
module = args[1]
with open(os.environ["FAKE_CALLS"], "a") as handle:
    handle.write(json.dumps([module] + args[2:]) + "\\n")
out = Path(args[args.index("--out-dir") + 1]) if "--out-dir" in args else None
if module == "pipeline.eval.ifeval.score":
    output = Path(args[args.index("--output") + 1])
    output.write_text(json.dumps({{"bench": "ifeval", "n": 541, "prompt_strict": 0.6, "inst_strict": 0.7, "prompt_loose": 0.65,
        "inst_loose": 0.75, "generation": {{"mean_new_tokens": 300.0, "capped": 0.01}}}}))
    sys.exit(0)
label = args[args.index("--label") + 1]
accuracy = 0.70 if label == "ref" else 0.60
benches = [arg for arg in args[2:args.index("--model-dir")]]
out.mkdir(parents = True, exist_ok = True)
for bench in benches:
    if bench == "ifeval":
        (out / "ifeval.responses.jsonl").write_text("{{}}\\n")
    elif bench == "gpqa":
        (out / "gpqa.json").write_text(json.dumps({{"accuracy": accuracy - 0.2, "closed": 0.9, "no_answer": 0.05, "capped": 0.1,
            "loop": 0.02, "reasoning_tokens": {{"median": 3000.0}}}}))
    else:
        groups = {{lang: {{"n": 500, "accuracy": accuracy}} for lang in ["ar", "de", "es", "fr", "id", "it", "ja", "ko", "pt", "zh"]}}
        (out / f"{{bench}}.json").write_text(json.dumps({{"accuracy": accuracy, "letter_mass_mean": 0.95, "groups": groups}}))
'''


def write_inputs(root):
    """假的模型目录、数据（不含 GPQA）与两个假的 python，返回驱动程序的参数。"""
    root = Path(root)
    model = root / "model"
    model.mkdir(parents = True)
    (model / "model.safetensors").write_bytes(b"")
    (model / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": c71_local.TOKENIZER_CLASS}), encoding = "utf-8")
    for bench in ("mmlu_redux", "ceval", "mmmlu", "ifeval"):
        path = root / "text" / text_bench.DATA[bench]
        path.parent.mkdir(parents = True, exist_ok = True)
        if bench == "ifeval":
            path.write_text("{}\n", encoding = "utf-8")
        else:
            path.mkdir(exist_ok = True)
    bin_dir = root / "bin"
    bin_dir.mkdir()
    for name in ("python", "python-ifeval"):
        script = bin_dir / name
        script.write_text(FAKE_TEXT_BENCH.format(python = sys.executable), encoding = "utf-8")
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return [
        "--work-dir", str(root),
        "--data-dir", str(root / "text"),
        "--model-dir", str(model),
        "--runs-root", str(root / "runs"),
        "--out-dir", str(root / "out"),
        "--py-text", str(bin_dir / "python"),
        "--py-ifeval", str(bin_dir / "python-ifeval"),
        "--gpu-lock", str(root / "gpu.lock"),
    ]


class MainTest(unittest.TestCase):
    def setUp(self):
        for signum in (bench_local.signal.SIGTERM, bench_local.signal.SIGINT):
            self.addCleanup(bench_local.signal.signal, signum, bench_local.signal.getsignal(signum))
        patcher = mock.patch.object(target = c71_local, attribute = "gpu_used_mib", return_value = None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def read_calls(self, path):
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

    def test_main_runs_scores_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = write_inputs(root = tmp)
            calls_path = Path(tmp) / "calls.txt"
            write_lm(runs_root = Path(tmp) / "runs", key = "c0")
            with mock.patch.dict(in_dict = os.environ, values = {"FAKE_CALLS": str(calls_path)}):
                # GPQA 数据不在、只有 c0 下载齐全：做参考模型与 c0 的简单基准并打分，退出码 3
                self.assertEqual(bench_local.main(argv = argv), 3)
                calls = self.read_calls(path = calls_path)
                self.assertEqual(
                    [(call[0], call[call.index("--label") + 1] if "--label" in call else None) for call in calls],
                    [
                        ("pipeline.eval.text_bench", "ref"),
                        ("pipeline.eval.ifeval.score", None),
                        ("pipeline.eval.text_bench", "c0"),
                        ("pipeline.eval.ifeval.score", None),
                    ],
                )
                log = (Path(tmp) / "out" / "driver.log").read_text(encoding = "utf-8")
                self.assertIn("gpqa 数据", log)
                summary = (Path(tmp) / "out" / "summary.md").read_text(encoding = "utf-8")
                self.assertIn("| 模型卡 | 69.2 | 65.2 | 56.9 | 61.2 |  |  |  | 51.6 |", summary)
                self.assertIn("| ref | 70.0 | 70.0 | 70.0 | 60.0 | 70.0 | 65.0 | 75.0 |  |", summary)
                self.assertIn("| c0 | 60.0（-10.0） | 60.0（-10.0） | 60.0（-10.0） | 60.0（+0.0） |", summary)
                # GPQA 数据到了、另一个运行也下载完：已有结果的跳过，补做 c-mix 的简单基准，再做三个模型的 GPQA；c-mix 以外
                # 还有四个运行没下载，退出码 3
                gpqa = Path(tmp) / "text" / text_bench.DATA["gpqa"]
                gpqa.parent.mkdir(parents = True)
                gpqa.write_text("Record ID\n", encoding = "utf-8")
                write_lm(runs_root = Path(tmp) / "runs", key = "c-mix")
                calls_path.write_text("")
                self.assertEqual(bench_local.main(argv = argv), 3)
                calls = self.read_calls(path = calls_path)
                self.assertEqual(
                    [(call[1], call[call.index("--label") + 1]) for call in calls if call[0] == "pipeline.eval.text_bench"],
                    [("mmlu_redux", "c-mix"), ("gpqa", "ref"), ("gpqa", "c0"), ("gpqa", "c-mix")],
                )
                status = json.loads((Path(tmp) / "out" / "status.json").read_text(encoding = "utf-8"))
                self.assertEqual(status["phase"], "done_partial")
                self.assertEqual(status["results"]["c-mix:gpqa"]["status"], "ok")
                summary = (Path(tmp) / "out" / "summary.md").read_text(encoding = "utf-8")
                self.assertIn("| c-mix | 60.0（-10.0） |", summary)
                self.assertIn("40.0（-10.0） |\n", summary)
                self.assertIn("| ref | 0.950 | 0.950 | 0.950 | 300 | 1.0 | 90.0 | 5.0 | 10.0 | 2.0 | 3000 |", summary)
                for key in ("c-free_raw", "c-free_chat", "c-cont_raw", "c-cont_chat"):
                    write_lm(runs_root = Path(tmp) / "runs", key = key)
                calls_path.write_text("")
                self.assertEqual(bench_local.main(argv = argv), 0)
                self.assertEqual(len(self.read_calls(path = calls_path)), 4 * 3)

    def test_missing_ifeval_python_skips_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = write_inputs(root = tmp)
            argv[argv.index("--py-ifeval") + 1] = str(Path(tmp) / "bin" / "missing-python")
            calls_path = Path(tmp) / "calls.txt"
            with mock.patch.dict(in_dict = os.environ, values = {"FAKE_CALLS": str(calls_path)}):
                self.assertEqual(bench_local.main(argv = argv + ["--runs", "c0", "--benches", "ifeval"]), 3)
            calls = self.read_calls(path = calls_path)
            self.assertEqual([call[0] for call in calls], ["pipeline.eval.text_bench"])
            status = json.loads((Path(tmp) / "out" / "status.json").read_text(encoding = "utf-8"))
            self.assertTrue(status["results"]["ref:ifeval_score"]["status"].startswith("未打分"))
            self.assertTrue((Path(tmp) / "out" / "ref" / "ifeval.responses.jsonl").exists())
            self.assertFalse((Path(tmp) / "out" / "ref" / "ifeval.json").exists())


if __name__ == "__main__":
    unittest.main()
