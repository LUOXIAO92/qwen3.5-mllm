"""回放差距与思考能力检查（任务 #71）的单元测试：python -m unittest pipeline.eval.tests.test_replay_evals

只在 CPU 上运行：

- GapTest：按长度分批不超过上限；思考组按 </think> 分出推理与回答；右填充的一批在缩小的随机初始化模型上算出的逐 token
  NLL 与每条单独、不填充时相同（线性注意力换回 PyTorch 参考实现）；整组评测在临时回放数据上给出 NLL、参考 NLL 与差距。
- ThinkTest：从 free_chat_think 取提问（跳过没有用户轮的样本，提示停在 <think>\\n）；单条结果的指标（写完 </think>、
  正常结束、到上限、循环）；缩小的模型上左填充生成一批，结果截到第一个结束符。
"""

import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from pipeline.eval.replay_gap import batches_by_tokens, evaluate_group, score_batch, think_parts
from pipeline.eval.think_check import analyze, generate_batch, load_prompts, summarize
from pipeline.train.data import IGNORE_INDEX
from pipeline.train.tests.test_offload import MODEL_DIR, tiny_model
from pipeline.train.tests.test_replay import THINK_BETWEEN, THINK_END, USER_START, FakeReplay

HAS_MODEL = (MODEL_DIR / "config.json").exists()


@contextmanager
def reference_kernels():
    """把线性注意力的四个内核（短卷积及其逐步更新、分块与逐步递推）换回 transformers 的 PyTorch 参考实现，CPU 上可运行。"""
    from transformers.models.qwen3_5 import modeling_qwen3_5

    names = ("causal_conv1d_fn", "causal_conv1d_update", "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule")
    saved = {name: getattr(modeling_qwen3_5, name) for name in names}
    try:
        for name, function in saved.items():
            setattr(modeling_qwen3_5, name, getattr(function, "__wrapped__", function))
        yield
    finally:
        for name, function in saved.items():
            setattr(modeling_qwen3_5, name, function)


class GapTest(unittest.TestCase):
    def test_batches(self):
        lengths = [10, 500, 30, 4000, 20, 9000]
        batches = batches_by_tokens(lengths = lengths, max_batch_tokens = 1000)
        self.assertEqual(sorted(index for batch in batches for index in batch), list(range(6)))
        for batch in batches:
            if len(batch) > 1:
                self.assertLessEqual(len(batch) * max(lengths[index] for index in batch), 1000)

    def test_think_parts(self):
        ids = np.array([1, 2, 3, 4, THINK_END, 6, 7], dtype = np.int64)
        labels = np.array([IGNORE_INDEX, IGNORE_INDEX, 3, 4, THINK_END, 6, 7], dtype = np.int64)
        self.assertEqual(think_parts(sample = {"ids": ids, "labels": labels}).tolist(), [True, True, True, False, False])

    @unittest.skipUnless(HAS_MODEL, "需要阶段 1 模型目录中的代码与配置")
    def test_padded_batch_matches_single(self):
        model = tiny_model(dtype = torch.float32, device = "cpu")
        rng = np.random.default_rng(seed = 0)
        samples = []
        for length, start in ((40, 5), (23, 9), (31, 1)):
            ids = rng.integers(low = 100, high = 3000, size = length).astype(np.int64)
            labels = np.full(shape = length, fill_value = IGNORE_INDEX, dtype = np.int64)
            labels[start:] = ids[start:]
            samples.append({"ids": ids, "labels": labels, "n_loss": length - start})
        with reference_kernels():
            together = score_batch(model = model, samples = samples, chunk_size = 7)
            alone = [score_batch(model = model, samples = [sample])[0] for sample in samples]
        for got, want in zip(together, alone):
            np.testing.assert_allclose(
                actual = got,
                desired = want,
                rtol = 1e-4,
                atol = 1e-4,
            )

    @unittest.skipUnless(HAS_MODEL, "需要阶段 1 模型目录中的代码与配置")
    def test_group_summary(self):
        model = tiny_model(dtype = torch.float32, device = "cpu")
        with tempfile.TemporaryDirectory() as tmp:
            FakeReplay(root = tmp).write_all(count = 5, split = "eval")
            with reference_kernels():
                result = evaluate_group(
                    model = model,
                    root = tmp,
                    group = "free_chat_think",
                    split = "eval",
                    max_batch_tokens = 200,
                )
        self.assertEqual(result["samples"], 4)
        self.assertAlmostEqual(result["gap"], result["nll"] - result["ref_nll"], places = 4)
        self.assertEqual(set(result["parts"]), {"reasoning", "answer"})
        self.assertEqual(result["parts"]["reasoning"]["tokens"] + result["parts"]["answer"]["tokens"], result["tokens"])


def write_think_eval(root):
    """free_chat_think 评测集：第 0 条正常，第 1 条因循环丢弃（只有元数据），第 2 条用户轮没有结束，第 3 条正常。"""
    directory = Path(root) / "free_chat_think" / "eval"
    (directory / "raw").mkdir(parents = True)
    user_a, answer_a = [11, 12, 248046], [21, THINK_END, 22, 248046]
    user_b = [13, 14]
    user_c, answer_c = [15, 248046], [23, 24]
    generated = np.asarray(a = user_a + answer_a + user_b + user_c + answer_c, dtype = np.int32)
    generated.tofile(directory / "raw" / "shard-00000.ids.i32")

    def gen(spec):
        """spec 为 (角色, 在 raw shard 中的起点, token, 结束原因)。"""
        role, offset, values, finish = spec
        return {"gen": role, "offset": offset, "length": len(values), "keep": len(values), "finish_reason": finish}

    user_ok_a = gen(spec = ("user", 0, user_a, "stop"))
    answer_ok_a = gen(spec = ("assistant", 3, answer_a, "stop"))
    user_unfinished = gen(spec = ("user", 7, user_b, "length"))
    user_ok_c = gen(spec = ("user", 9, user_c, "stop"))
    answer_capped = gen(spec = ("assistant", 11, answer_c, "length"))
    pieces = {
        "s0": [{"ids": USER_START}, user_ok_a, {"ids": THINK_BETWEEN}, answer_ok_a],
        "s1": [{"ids": USER_START}],
        "s2": [{"ids": USER_START}, user_unfinished],
        "s3": [{"ids": USER_START}, user_ok_c, {"ids": THINK_BETWEEN}, answer_capped],
    }
    records = [
        {"sample_id": sample_id, "prompt_index": number, "shard": 0, "pieces": parts}
        for number, (sample_id, parts) in enumerate(pieces.items())
    ]
    with open(file = directory / "samples.jsonl", mode = "w", encoding = "utf-8") as handle:
        for record in reversed(records):
            handle.write(json.dumps(obj = record) + "\n")
    return USER_START + user_a + THINK_BETWEEN, USER_START + user_c + THINK_BETWEEN


class ThinkTest(unittest.TestCase):
    def test_prompts(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = write_think_eval(root = tmp)
            prompts = load_prompts(replay_dir = tmp, count = 5)
        self.assertEqual(prompts, [("s0", first), ("s3", second)])
        self.assertEqual(first[-6:], THINK_BETWEEN)

    def test_analyze(self):
        closed = analyze(ids = [5, 6, THINK_END, 7, 248046], max_new_tokens = 100)
        self.assertEqual(
            (closed["closed"], closed["ended"], closed["capped"], closed["reasoning_tokens"], closed["answer_tokens"], closed["loop"]),
            (True, True, False, 3, 2, False),
        )
        capped = analyze(ids = list(range(1000, 1050)), max_new_tokens = 50)
        self.assertEqual((capped["closed"], capped["ended"], capped["capped"], capped["answer_tokens"]), (False, False, True, None))
        looped = analyze(ids = list(range(2000, 2030)) * 10, max_new_tokens = 16384)
        self.assertTrue(looped["loop"])
        summary = summarize(rows = [closed, capped, looped], max_new_tokens = 100)
        self.assertAlmostEqual(summary["closed"], 1 / 3, places = 3)
        self.assertEqual(summary["reasoning_tokens"]["count"], 1)

    @unittest.skipUnless(HAS_MODEL, "需要阶段 1 模型目录中的代码与配置")
    def test_generate_batch(self):
        model = tiny_model(dtype = torch.float32, device = "cpu")
        prompts = [("a", USER_START + [11, 12, 248046] + THINK_BETWEEN), ("b", USER_START + [13, 248046] + THINK_BETWEEN)]
        with reference_kernels():
            outputs = generate_batch(
                model = model,
                prompts = prompts,
                max_new_tokens = 6,
                temperature = 1.0,
                seed = 0,
            )
            again = generate_batch(
                model = model,
                prompts = prompts,
                max_new_tokens = 6,
                temperature = 1.0,
                seed = 0,
            )
        self.assertEqual(len(outputs), 2)
        for ids, repeat in zip(outputs, again):
            self.assertLessEqual(len(ids), 6)
            self.assertEqual(ids.tolist(), repeat.tolist())
            stops = [index for index, token in enumerate(ids.tolist()) if token in (248046, 248044)]
            self.assertTrue(not stops or stops[0] == len(ids) - 1)


if __name__ == "__main__":
    unittest.main()
