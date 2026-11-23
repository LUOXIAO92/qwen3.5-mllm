"""IFEval 打分（任务 #71，pipeline/eval/ifeval/score.py）的单元测试：官方打分逻辑的 strict 与 loose。

需要 langdetect 与 nltk（≥ 3.9.1，带 punkt_tab 数据），项目的 envs/mllm-train、envs/mllm-eval 都没有，缺少时跳过；用有这两个包
的 python 在仓库根目录运行（不需要 torch）：<python> -m unittest pipeline.eval.tests.test_ifeval_score
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(
    importlib.util.find_spec("langdetect") and importlib.util.find_spec("nltk"),
    "IFEval 打分需要 langdetect 与 nltk（envs/mllm-train 中没有，用有这两个包的 python 运行）",
)
class IfevalScoreTest(unittest.TestCase):
    def test_strict_and_loose(self):
        from pipeline.eval.ifeval import score

        inputs = [
            {"key": 1, "prompt": "Reply in lowercase.", "instruction_id_list": ["change_case:english_lowercase"], "kwargs": [{}]},
            {"key": 2, "prompt": "No commas please.", "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]},
            {
                "key": 3,
                "prompt": "No commas and at least 2 sentences.",
                "instruction_id_list": ["punctuation:no_comma", "length_constraints:number_sentences"],
                "kwargs": [{}, {"relation": "at least", "num_sentences": 2}],
            },
        ]
        responses = [
            {"key": 1, "prompt": "Reply in lowercase.", "response": "this whole answer is written in plain lowercase english words", "n_new_tokens": 12},
            {"key": 2, "prompt": "No commas please.", "response": "Sure, here it is:\nThis line has no commas at all", "n_new_tokens": 10, "capped": False},
            {"key": 3, "prompt": "No commas and at least 2 sentences.", "response": "One, two. Three.", "n_new_tokens": 5},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.jsonl"
            input_path.write_text("".join(json.dumps(row) + "\n" for row in inputs), encoding = "utf-8")
            responses_path = Path(tmp) / "ifeval.responses.jsonl"
            responses_path.write_text("".join(json.dumps(row) + "\n" for row in responses), encoding = "utf-8")
            (Path(tmp) / "ifeval.generation.json").write_text(json.dumps({"decoding": "greedy"}), encoding = "utf-8")
            output = Path(tmp) / "ifeval.json"
            self.assertEqual(score.main(argv = ["--input", str(input_path), "--responses", str(responses_path), "--output", str(output)]), 0)
            summary = json.loads(output.read_text(encoding = "utf-8"))
            items = [json.loads(line) for line in (Path(tmp) / "ifeval.items.jsonl").read_text(encoding = "utf-8").splitlines()]
        self.assertEqual(summary["n"], 3)
        # strict：只有第 1 条照做；loose：第 2 条去掉第一行后没有逗号，也算照做
        self.assertEqual([item["strict"] for item in items], [[True], [False], [False, True]])
        self.assertEqual([item["loose"] for item in items], [[True], [True], [False, True]])
        self.assertAlmostEqual(summary["prompt_strict"], round(1 / 3, 4))
        self.assertAlmostEqual(summary["prompt_loose"], round(2 / 3, 4))
        self.assertAlmostEqual(summary["inst_strict"], 0.5)
        self.assertAlmostEqual(summary["inst_loose"], 0.75)
        self.assertEqual(summary["tiers"]["strict"]["tier0"]["punctuation"], 0.0)
        self.assertEqual(summary["generation"]["decoding"], "greedy")
        self.assertNotIn("prompt", items[0])


if __name__ == "__main__":
    unittest.main()
