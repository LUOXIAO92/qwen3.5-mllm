"""回放差距与思考能力检查（任务 #71）的单元测试：python -m unittest pipeline.eval.tests.test_replay_evals

只在 CPU 上运行：

- GapTest：按长度分批不超过上限；思考组按 </think> 分出推理与回答；右填充的一批在缩小的随机初始化模型上算出的逐 token
  NLL 与每条单独、不填充时相同（线性注意力换回 PyTorch 参考实现）；整组评测在临时回放数据上给出 NLL、参考 NLL 与差距。
- ThinkTest：从 free_chat_think 取提问（跳过没有用户轮的样本，提示停在 <think>\\n）；单条结果的指标（写完 </think>、
  正常结束、到上限、循环）；缩小的模型上左填充生成一批，结果截到第一个结束符。
- SmokeTest：冒烟比较取提问的方法与判定；缩小的模型上逐步比较两种注意力实现。CPU 上没有 flash-attn 的内核，
  emulated_flash_attention 让 transformers 的 flash_attention_2 代码路径（二维填充掩码、去掉填充、变长注意力、补回填充）
  照常运行，只把最内层的内核换成 PyTorch 写的参照实现：左填充的一批与 sdpa 逐步一致，也与每条单独生成一致；
  忽略填充掩码的错误实现会被判为不一致；整个冒烟比较在 flash_attention_2 可用与载入失败时分别给出 flash_attention_2 与 sdpa。
"""

import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from pipeline.eval import think_check
from pipeline.eval.replay_gap import batches_by_tokens, evaluate_group, score_batch, think_parts
from pipeline.eval.think_check import (
    STOP_IDS,
    analyze,
    compare_traces,
    generate_batch,
    greedy_trace,
    load_prompts,
    run_smoke,
    smoke_prompts,
    smoke_verdict,
    summarize,
)
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


def reference_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q = None, max_seqlen_k = None, dropout_p = 0.0, softmax_scale = None, causal = False):
    """flash_attn_varlen_func 的 PyTorch 参照实现：按 cu_seqlens 分段，GQA 的 K、V 头按组复制，因果掩码右下角对齐。"""
    out = torch.empty_like(q)
    for index in range(len(cu_seqlens_q) - 1):
        q_part = q[cu_seqlens_q[index]: cu_seqlens_q[index + 1]].float()
        k_part = k[cu_seqlens_k[index]: cu_seqlens_k[index + 1]].float()
        v_part = v[cu_seqlens_k[index]: cu_seqlens_k[index + 1]].float()
        groups = q_part.shape[1] // k_part.shape[1]
        k_part = k_part.repeat_interleave(repeats = groups, dim = 1)
        v_part = v_part.repeat_interleave(repeats = groups, dim = 1)
        scores = torch.einsum("qhd,khd->hqk", q_part, k_part) * softmax_scale
        if causal:
            q_len, k_len = q_part.shape[0], k_part.shape[0]
            allowed = torch.arange(end = k_len)[None, :] <= torch.arange(end = q_len)[:, None] + (k_len - q_len)
            scores = scores.masked_fill(mask = ~allowed, value = float("-inf"))
        weights = scores.softmax(dim = -1)
        out[cu_seqlens_q[index]: cu_seqlens_q[index + 1]] = torch.einsum("hqk,khd->qhd", weights, v_part).to(dtype = q.dtype)
    return out


def reference_dense(q, k, v, dropout_p = 0.0, softmax_scale = None, causal = False):
    """flash_attn_func（没有填充时用）的参照实现，形状 (批, 长度, 头数, 每头维数)。"""
    batch, q_len = q.shape[:2]
    out = reference_varlen(
        q = q.reshape(-1, *q.shape[2:]),
        k = k.reshape(-1, *k.shape[2:]),
        v = v.reshape(-1, *v.shape[2:]),
        cu_seqlens_q = torch.arange(end = batch + 1) * q_len,
        cu_seqlens_k = torch.arange(end = batch + 1) * k.shape[1],
        softmax_scale = softmax_scale,
        causal = causal,
    )
    return out.view(*q.shape)


@contextmanager
def emulated_flash_attention():
    """让 transformers 的 flash_attention_2 在 CPU 上运行：只把它载入的 flash-attn 函数换成上面的参照实现。"""
    from transformers import modeling_flash_attention_utils as utils

    saved = utils._lazy_imports

    def fake_imports(implementation, attention_wrapper = None, allow_all_kernels = False):
        return reference_dense, reference_varlen, None, utils._pad_input, utils._unpad_input

    try:
        utils._lazy_imports = fake_imports
        utils._loaded_implementation = None
        yield
    finally:
        utils._lazy_imports = saved
        utils._loaded_implementation = None


def set_text_attention(model, name):
    """直接设置语言模型全注意力层的实现（CPU 上 set_attn_implementation 不接受 flash_attention_2）。"""
    model.model.language_model.config._attn_implementation_internal = name
    return model


def ignore_padding(module, query, key, value, attention_mask, **kwargs):
    """错误的注意力实现：不管填充掩码，只做因果注意力（左填充的位置也被看到）。"""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    kwargs.pop("is_causal", None)
    return sdpa_attention_forward(
        module = module,
        query = query,
        key = key,
        value = value,
        attention_mask = None,
        is_causal = query.shape[2] > 1,
        **kwargs,
    )


SMOKE_PROMPTS = [("short", [5, 6, 7]), ("long", list(range(100, 140))), ("mid", list(range(200, 215)))]


def synthetic_trace(logprobs, tokens = None):
    """compare_traces 用的假结果：logprobs 为 (条数, 步数, top)，tokens 默认是每步的第 0 个。"""
    logprobs = np.asarray(a = logprobs, dtype = np.float32)
    rows, steps, top = logprobs.shape
    ids = np.tile(A = np.arange(top), reps = (rows, steps, 1))
    tokens = np.zeros(shape = (rows, steps), dtype = np.int64) if tokens is None else np.asarray(a = tokens)
    return {"tokens": tokens, "ids": ids, "logprobs": logprobs, "argmax": tokens.copy(), "prompt_tokens": [10] * rows}


class SmokeTest(unittest.TestCase):
    def test_smoke_prompts(self):
        prompts = [(f"p{index}", [1] * length) for index, length in enumerate([16, 11, 219, 30, 14, 25, 12])]
        picked = smoke_prompts(prompts = prompts, count = 4)
        lengths = [len(ids) for _, ids in picked]
        self.assertEqual(len(picked), 4)
        self.assertIn(11, lengths)
        self.assertIn(219, lengths)
        # 保持原来的先后顺序
        self.assertEqual([sample_id for sample_id, _ in picked], sorted((sample_id for sample_id, _ in picked), key = lambda name: int(name[1:])))
        self.assertEqual(smoke_prompts(prompts = prompts[:3], count = 4), prompts[:3])

    def test_compare_and_verdict(self):
        steps = 40
        base = np.log(np.full(shape = (2, steps, 4), fill_value = 0.2))
        base[:, :, 0] = np.log(0.4)
        tokens = np.zeros(shape = (2, steps), dtype = np.int64)
        tokens[1, 30] = STOP_IDS[0]
        reference = synthetic_trace(logprobs = base, tokens = tokens)
        same = synthetic_trace(logprobs = base + 1e-4, tokens = tokens)
        rows = compare_traces(reference = reference, candidate = same)
        self.assertEqual([row["steps"] for row in rows], [40, 31])
        self.assertLess(max(row["tv_mean"] for row in rows), 1e-3)
        self.assertTrue(smoke_verdict(rows = rows)[0])
        # 第 2 条的分布换成另一种（所选 token 的概率从 0.4 变成 0.1）：总变差 0.3，判为不一致
        shifted = base.copy()
        shifted[1, :, 0] = np.log(0.1)
        shifted[1, :, 1] = np.log(0.5)
        rows = compare_traces(reference = reference, candidate = synthetic_trace(logprobs = shifted, tokens = tokens))
        self.assertAlmostEqual(rows[1]["tv_mean"], 0.3, places = 5)
        ok, reason = smoke_verdict(rows = rows)
        self.assertFalse(ok)
        self.assertIn("平均总变差", reason)
        # 分布几乎均匀时总变差很小，所选 token 的对数概率之差仍能看出不一致
        rows = [dict(row, tv_mean = 0.0) for row in rows]
        ok, reason = smoke_verdict(rows = rows)
        self.assertFalse(ok)
        self.assertIn("对数概率之差", reason)
        self.assertFalse(smoke_verdict(rows = compare_traces(reference = reference, candidate = same), min_steps = 100)[0])
        disagree = synthetic_trace(logprobs = base, tokens = tokens)
        disagree["argmax"] = tokens + 1
        rows = compare_traces(reference = reference, candidate = disagree)
        self.assertFalse(smoke_verdict(rows = rows)[0])

    @unittest.skipUnless(HAS_MODEL, "需要阶段 1 模型目录中的代码与配置")
    def test_flash_path_matches_sdpa_with_left_padding(self):
        model = tiny_model(dtype = torch.float32, device = "cpu")
        calls = []
        with reference_kernels(), emulated_flash_attention():
            reference = greedy_trace(model = model, prompts = SMOKE_PROMPTS, max_new_tokens = 12)
            set_text_attention(model = model, name = "flash_attention_2")
            from transformers import modeling_flash_attention_utils as utils

            original = utils._upad_input

            def counting_upad(*args, **kwargs):
                calls.append(args[4])
                return original(*args, **kwargs)

            utils._upad_input = counting_upad
            try:
                forced = greedy_trace(
                    model = model,
                    prompts = SMOKE_PROMPTS,
                    max_new_tokens = 12,
                    forced = reference,
                )
                alone = [greedy_trace(model = model, prompts = [prompt], max_new_tokens = 12) for prompt in SMOKE_PROMPTS]
            finally:
                utils._upad_input = original
        # 走的是 transformers 的 flash_attention_2 去填充路径：预填充一次（查询长度 40），逐 token 生成每步一次（查询长度 1）
        self.assertEqual(calls[0], 40)
        self.assertEqual(calls[1:], [1] * (len(calls) - 1))
        self.assertEqual(len(calls), reference["tokens"].shape[1])
        rows = compare_traces(reference = reference, candidate = forced)
        for row in rows:
            self.assertLess(row["chosen_diff_max"], 1e-4)
            self.assertEqual(row["argmax_agree"], 1.0)
        self.assertTrue(smoke_verdict(rows = rows, min_steps = 10)[0])
        # 左填充的一批与每条单独生成（没有填充）一致
        for row, single in enumerate(alone):
            steps = single["tokens"].shape[1]
            np.testing.assert_array_equal(single["tokens"][0], reference["tokens"][row, :steps])
            np.testing.assert_allclose(actual = single["logprobs"][0], desired = reference["logprobs"][row, :steps], atol = 1e-4)

    @unittest.skipUnless(HAS_MODEL, "需要阶段 1 模型目录中的代码与配置")
    def test_ignoring_padding_is_detected(self):
        from transformers import AttentionInterface, AttentionMaskInterface
        from transformers.masking_utils import sdpa_mask

        AttentionInterface.register("think_test_ignore_padding", ignore_padding)
        AttentionMaskInterface.register("think_test_ignore_padding", sdpa_mask)
        model = tiny_model(dtype = torch.float32, device = "cpu")
        with reference_kernels():
            reference = greedy_trace(model = model, prompts = SMOKE_PROMPTS, max_new_tokens = 12)
            set_text_attention(model = model, name = "eager")
            eager = greedy_trace(model = model, prompts = SMOKE_PROMPTS, max_new_tokens = 12, forced = reference)
            set_text_attention(model = model, name = "think_test_ignore_padding")
            broken = greedy_trace(model = model, prompts = SMOKE_PROMPTS, max_new_tokens = 12, forced = reference)
        self.assertTrue(smoke_verdict(rows = compare_traces(reference = reference, candidate = eager), min_steps = 10)[0])
        rows = compare_traces(reference = reference, candidate = broken)
        ok, reason = smoke_verdict(rows = rows, min_steps = 10)
        self.assertFalse(ok)
        # 没有填充的最长一条不受影响，填充最多的最短一条差别最大
        self.assertLess(rows[1]["chosen_diff_max"], 1e-4)
        self.assertGreater(rows[0]["chosen_diff_mean"], 0.1)

    @unittest.skipUnless(HAS_MODEL, "需要阶段 1 模型目录中的代码与配置")
    def test_run_smoke(self):
        loads = []

        def load(attention, fail_flash = False):
            loads.append(attention)
            if attention == "flash_attention_2" and fail_flash:
                raise ImportError("没有 flash_attn")
            return set_text_attention(model = tiny_model(dtype = torch.float32, device = "cpu"), name = attention)

        # 3 条 × 12 步，超过判定要求的最少步数 SMOKE_MIN_STEPS（32）
        with reference_kernels(), emulated_flash_attention():
            summary = run_smoke(load = load, prompts = SMOKE_PROMPTS, max_new_tokens = 12, log = lambda message: None)
            failed = run_smoke(
                load = lambda attention: load(attention = attention, fail_flash = True),
                prompts = SMOKE_PROMPTS,
                max_new_tokens = 12,
                log = lambda message: None,
            )
        self.assertEqual(loads, ["sdpa", "flash_attention_2", "sdpa", "flash_attention_2"])
        self.assertEqual((summary["use"], summary["agree"], len(summary["rows"])), ("flash_attention_2", True, 3))
        self.assertEqual(summary["prompt_tokens"], [3, 40, 15])
        self.assertIn("flash_attention_2", summary["seconds"])
        self.assertEqual((failed["use"], failed["agree"], failed["rows"]), ("sdpa", False, []))
        self.assertIn("ImportError: 没有 flash_attn", failed["reason"])
        self.assertIn("结论：flash_attention_2", think_check.format_smoke_markdown(summary = summary))


if __name__ == "__main__":
    unittest.main()
