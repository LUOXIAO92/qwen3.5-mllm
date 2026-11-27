"""系统提示遵循评测（任务 #71）中不需要模型的部分：python -m unittest pipeline.eval.tests.test_sysprompt

条目文件的完整性、三种条件的对话、请求的排列、成对差、单个模型的汇总、语音探针中系统消息的提示、驱动程序的计划与跨模型汇总。
用 langdetect 与 nltk 重新生成条目并用官方代码判定的两项在没有这两个包的环境中跳过（用有这两个包的 python 运行）。
"""

import collections
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pipeline.eval import sysprompt_bench, sysprompt_local
from pipeline.eval.instruct_probe import probe_prompt, system_messages

HAS_IFEVAL_DEPS = bool(importlib.util.find_spec("langdetect") and importlib.util.find_spec("nltk"))


class FakeTokenizer:
    """把对话按角色拼成文字，再按字符编号，足以检查请求的排列。"""

    def apply_chat_template(self, conversation, tokenize, add_generation_prompt, enable_thinking):
        self.last = (conversation, add_generation_prompt, enable_thinking)
        return "".join(f"<{message['role']}>{message['content']}" for message in conversation) + "<assistant>"

    def __call__(self, text, add_special_tokens):
        return {"input_ids": [ord(char) for char in text]}


class FakeProcessor:
    audio_start_token = "<as>"
    audio_token = "<a>"
    audio_end_token = "<ae>"

    def apply_chat_template(self, conversation, tokenize, add_generation_prompt, enable_thinking):
        self.conversation = conversation
        return json.dumps(conversation)


class ItemsFileTest(unittest.TestCase):
    def test_items_file(self):
        items = sysprompt_bench.read_jsonl(path = sysprompt_bench.ITEMS_PATH)
        tasks = sysprompt_bench.read_jsonl(path = sysprompt_bench.TASKS_PATH)
        self.assertEqual(len(items), sysprompt_bench.PER_TYPE * len(sysprompt_bench.INSTRUCTION_TYPES))
        self.assertEqual(len({item["item_id"] for item in items}), len(items))
        counts = collections.Counter(item["instruction_id"] for item in items)
        self.assertEqual(set(counts), set(sysprompt_bench.INSTRUCTION_TYPES))
        self.assertEqual(set(counts.values()), {sysprompt_bench.PER_TYPE})
        self.assertFalse(set(counts) & set(sysprompt_bench.EXCLUDED_TYPES))
        by_task = collections.defaultdict(list)
        texts = {task["task_id"]: task["task"] for task in tasks}
        for item in items:
            by_task[item["task_id"]].append(item["instruction_id"])
            self.assertEqual(item["task"], texts[item["task_id"]])
            self.assertTrue(item["constraint"].strip())
        # 每个问题都用到，同一个问题的约束类别互不相同
        self.assertEqual(set(by_task), set(texts))
        self.assertTrue(all(len(kinds) == len(set(kinds)) for kinds in by_task.values()))
        repeat = [item for item in items if item["instruction_id"] == "combination:repeat_prompt"]
        self.assertTrue(all(item["kwargs"]["prompt_to_repeat"] == item["task"] for item in repeat))


class ConversationTest(unittest.TestCase):
    def test_three_conditions(self):
        item = {"item_id": "s001", "task": "Why is the sky blue?", "constraint": "Answer in French.", "instruction_id": "x"}
        self.assertEqual(sysprompt_bench.conversation(item = item, condition = "A"), [{"role": "user", "content": "Why is the sky blue?"}])
        self.assertEqual(
            sysprompt_bench.conversation(item = item, condition = "B"),
            [{"role": "system", "content": "Answer in French."}, {"role": "user", "content": "Why is the sky blue?"}],
        )
        self.assertEqual(
            sysprompt_bench.conversation(item = item, condition = "C"),
            [{"role": "user", "content": "Why is the sky blue?\n\nAnswer in French."}],
        )
        with self.assertRaises(ValueError):
            sysprompt_bench.conversation(item = item, condition = "D")

    def test_requests(self):
        tokenizer = FakeTokenizer()
        items = [
            {"item_id": "s001", "task": "Q1", "constraint": "K1", "instruction_id": "punctuation:no_comma"},
            {"item_id": "s002", "task": "Q2", "constraint": "K2", "instruction_id": "startend:quotation"},
        ]
        requests = sysprompt_bench.requests_of(tokenizer = tokenizer, items = items, max_new_tokens = 7)
        self.assertEqual([request["index"] for request in requests], list(range(6)))
        self.assertEqual([(request["item_id"], request["condition"]) for request in requests][:3], [("s001", "A"), ("s001", "B"), ("s001", "C")])
        decoded = "".join(chr(token) for token in requests[1]["prompt_token_ids"])
        self.assertEqual(decoded, "<system>K1<user>Q1<assistant>")
        self.assertEqual(tokenizer.last[1:], (True, False))
        self.assertTrue(all(request["max_tokens"] == 7 for request in requests))


class SummaryTest(unittest.TestCase):
    def test_paired_difference(self):
        result = sysprompt_bench.paired_difference(first = [1, 1, 0, 0], second = [1, 0, 1, 1])
        self.assertAlmostEqual(result["diff"], -0.25)
        self.assertEqual((result["first_only"], result["second_only"], result["n"]), (1, 2, 4))
        # p10 = 0.25、p01 = 0.5：方差 (0.75 − 0.0625) / 4
        self.assertAlmostEqual(result["se"], (0.6875 / 4) ** 0.5)
        self.assertIsNone(sysprompt_bench.paired_difference(first = [], second = [])["diff"])

    def test_summarize(self):
        items = [
            {"item_id": "s001", "instruction_id": "punctuation:no_comma"},
            {"item_id": "s002", "instruction_id": "startend:quotation"},
        ]
        outcomes = {("s001", "A"): (0, 0), ("s001", "B"): (1, 1), ("s001", "C"): (1, 1), ("s002", "A"): (0, 0), ("s002", "B"): (0, 1), ("s002", "C"): (1, 1)}
        rows = [
            {"item_id": item_id, "condition": condition, "instruction_id": "x", "strict": bool(strict), "loose": bool(loose)}
            for (item_id, condition), (strict, loose) in outcomes.items()
        ]
        responses = [{"item_id": item_id, "condition": condition, "response": "text", "n_new_tokens": 10, "capped": False} for item_id, condition in outcomes]
        summary = sysprompt_bench.summarize(items = items, rows = rows, responses = responses)
        self.assertEqual(summary["conditions"]["B"]["strict"], 0.5)
        self.assertEqual(summary["conditions"]["B"]["loose"], 1.0)
        self.assertEqual(summary["b_minus_c"]["strict"]["diff"], -0.5)
        self.assertEqual(summary["by_type"]["startend:quotation"]["C"]["strict"], 1.0)
        self.assertEqual(summary["by_tier0"]["punctuation"]["A"]["strict"], 0.0)
        self.assertEqual(summary["generation"]["A"]["mean_new_tokens"], 10.0)
        with self.assertRaises(ValueError):
            sysprompt_bench.summarize(items = items, rows = rows[:-1], responses = responses)


class ProbePromptTest(unittest.TestCase):
    def test_system_placement(self):
        processor = FakeProcessor()
        request = {"prompt_format": "chat", "instruction": "Transcribe."}
        probe_prompt(processor = processor, request = request, audio_path = "/a.flac", placement = "system")
        self.assertEqual(processor.conversation, system_messages(audio_path = "/a.flac", instruction = "Transcribe."))
        self.assertEqual(processor.conversation[0], {"role": "system", "content": [{"type": "text", "text": "Transcribe."}]})
        self.assertEqual(processor.conversation[1]["content"], [{"type": "audio", "audio": "/a.flac"}])
        probe_prompt(processor = processor, request = request, audio_path = "/a.flac", placement = "user")
        self.assertEqual([message["role"] for message in processor.conversation], ["user"])
        self.assertEqual(processor.conversation[0]["content"][-1], {"type": "text", "text": "Transcribe."})
        # cpt 只有音频标记，与指令的位置无关
        cpt = probe_prompt(processor = processor, request = {"prompt_format": "cpt", "instruction": None}, audio_path = "/a.flac", placement = "system")
        self.assertEqual(cpt, "<as><a><ae>")
        with self.assertRaises(ValueError):
            probe_prompt(processor = processor, request = request, audio_path = "/a.flac", placement = "assistant")


class DriverTest(unittest.TestCase):
    def paths(self, root):
        return {
            "out_dir": root / "out",
            "model_dir": root / "model",
            "items": sysprompt_bench.ITEMS_PATH,
            "fleurs": root / "fleurs",
            "runs_root": root / "runs",
            "user_probe_root": root / "eval",
            "ref_projector": root / "proj.safetensors",
            "python_text": "py-text",
            "python_audio": "py-audio",
            "python_ifeval": "py-ifeval",
            "ref_lm": root / "ref-lm",
            "vllm_image": "image",
            "vllm_export_root": root / "export",
            "vllm_job_root": root / "job",
            "vllm_home": root / "home",
            "gpu_memory_utilization": 0.9,
        }

    def test_label_entries(self):
        root = Path("/w")
        paths = self.paths(root = root)
        ref = sysprompt_local.label_entries(label = "ref", run = None, paths = paths)
        self.assertEqual([entry["item"] for entry in ref], ["text", "audio_user", "audio_system"])
        self.assertIn("--vllm-model", ref[0]["command"])
        self.assertNotIn("--language-model", ref[1]["command"])
        self.assertEqual(ref[1]["command"][ref[1]["command"].index("--projector") + 1], str(root / "proj.safetensors"))
        self.assertEqual(ref[2]["command"][ref[2]["command"].index("--instruction-placement") + 1], "system")
        run = {"dir": root / "runs" / "v2b-2b-cpt77-c0", "ready": True}
        entries = sysprompt_local.label_entries(label = "c0", run = run, paths = paths)
        self.assertEqual([entry["item"] for entry in entries], ["text", "audio_system"])
        command = entries[1]["command"]
        self.assertIn("--keep-export", command)
        self.assertEqual(command[command.index("--export-dir") + 1], str(root / "export" / "c0"))
        self.assertTrue(command[command.index("--projector") + 1].endswith("checkpoint-0030000/projector.safetensors"))
        self.assertEqual(command[command.index("--tasks") + 1:command.index("--tasks") + 3], ["transcribe", "translate"])
        rerun = sysprompt_local.label_entries(label = "c0", run = run, paths = paths, rerun_user = True)
        self.assertEqual([entry["item"] for entry in rerun], ["text", "audio_user", "audio_system"])

    def test_collect_and_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.paths(root = root)
            items = sysprompt_bench.read_jsonl(path = sysprompt_bench.ITEMS_PATH)[:3]
            for label, b_values in (("ref", (1, 1, 0)), ("c0", (0, 1, 0))):
                text = paths["out_dir"] / label / "text"
                text.mkdir(parents = True)
                rows = []
                for item, b_value in zip(items, b_values):
                    for condition in sysprompt_bench.CONDITIONS:
                        value = b_value if condition == "B" else int(condition == "C")
                        rows.append({"item_id": item["item_id"], "condition": condition, "instruction_id": item["instruction_id"], "strict": bool(value), "loose": bool(value)})
                responses = [{"item_id": row["item_id"], "condition": row["condition"], "response": "x", "n_new_tokens": 5} for row in rows]
                sysprompt_bench.write_jsonl(path = text / "scored.jsonl", rows = rows)
                scores = sysprompt_bench.summarize(items = items, rows = rows, responses = responses)
                (text / "scores.json").write_text(json.dumps(scores), encoding = "utf-8")
            probe = {
                "config": {"engine": "vllm"},
                "transcribe": {"transcribe_strict": {"ALL": {"n": 540, "rate_basic": 0.2, "empty_share": 0.0}}},
                "translate": {"translate_en_strict": {"ALL": {"n": 510, "chrf": 40.0, "verbatim_share": 0.1}}},
            }
            for path in (paths["out_dir"] / "ref" / "audio_user", paths["out_dir"] / "ref" / "audio_system", paths["out_dir"] / "c0" / "audio_system", root / "eval" / "c0" / "probe_strict"):
                path.mkdir(parents = True)
                (path / "summary.json").write_text(json.dumps(probe), encoding = "utf-8")
            data = sysprompt_local.collect(paths = paths, labels = ["ref", "c0", "c-mix"])
            self.assertEqual(data["labels"], ["ref", "c0"])
            self.assertAlmostEqual(data["text"]["c0"]["B_minus_ref"]["diff"], -1 / 3)
            self.assertEqual(data["text"]["c0"]["C_minus_ref"]["diff"], 0.0)
            # (B − C) 逐条：c0 为 (−1, 0, −1)，参考模型为 (0, 0, −1)，之差平均 −1/3
            self.assertAlmostEqual(data["text"]["c0"]["gap_minus_ref"]["diff"], -1 / 3)
            self.assertEqual(data["text"]["ref"]["gap_minus_ref"]["diff"], 0.0)
            self.assertEqual(data["audio"]["c0"]["user"]["source"], str(root / "eval" / "c0" / "probe_strict" / "summary.json"))
            self.assertEqual(data["audio"]["c0"]["system"]["values"]["translate_en_chrf"], 40.0)
            self.assertIsNone(data["audio"]["c0"]["system"]["values"]["translate_zh_chrf"])
            lines = sysprompt_local.summary_lines(data = data, paths = paths)
            self.assertTrue(any(line.startswith("| c0 | ") and "-33.3" in line for line in lines))
            self.assertTrue(any(line.startswith("| ref | ") and "40.0" in line for line in lines))


@unittest.skipUnless(HAS_IFEVAL_DEPS, "需要 langdetect 与 nltk（用有这两个包的 python 运行）")
class OfficialCheckerTest(unittest.TestCase):
    def test_build_reproduces_items(self):
        tasks = sysprompt_bench.read_jsonl(path = sysprompt_bench.TASKS_PATH)
        built = sysprompt_bench.build_items(tasks = tasks)
        saved = sysprompt_bench.read_jsonl(path = sysprompt_bench.ITEMS_PATH)
        self.assertEqual(json.loads(json.dumps(built)), saved)
        for item in saved[:40]:
            text, _ = sysprompt_bench.describe(instruction_id = item["instruction_id"], kwargs = item["kwargs"])
            self.assertEqual(text, item["constraint"])

    def test_judge(self):
        no_comma = {"item_id": "s1", "instruction_id": "punctuation:no_comma", "kwargs": {}}
        self.assertEqual(sysprompt_bench.judge(item = no_comma, response = "No commas here at all."), (True, True))
        # loose 去掉第一行后照做
        self.assertEqual(sysprompt_bench.judge(item = no_comma, response = "Sure, here:\nNo commas here at all."), (False, True))
        quotation = {"item_id": "s2", "instruction_id": "startend:quotation", "kwargs": {}}
        self.assertEqual(sysprompt_bench.judge(item = quotation, response = "\"Wrapped answer.\""), (True, True))
        self.assertEqual(sysprompt_bench.judge(item = quotation, response = "Plain answer."), (False, False))
        self.assertEqual(sysprompt_bench.judge(item = quotation, response = "   "), (False, False))


if __name__ == "__main__":
    unittest.main()
