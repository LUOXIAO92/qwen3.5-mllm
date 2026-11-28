"""自生成回放（任务 #71，replay.py、model.LinearTopKL）的单元测试：python -m unittest pipeline.train.tests.test_replay

CPU 上运行（在临时目录中按回放数据的格式写出五组小样本）：

- MaskTest：各组计损失的 token：free_raw 全部生成 token；cont_raw 不含文档开头；free_chat、cont_chat 只有助手轮（含结尾
  <|im_end|>），模型自己写的用户轮不计；free_sys_chat、free_sys_chat_think 模型自己写的系统轮也不计；思考模式的助手轮含推理
  和 </think>。keep 为假的样本不用；top-32 的行与计损失的位置一一对应。设置环境变量 REPLAY_DIR（回放数据目录）时另在真实
  数据上检查每组第一条样本的边界（目录中没有的组跳过）。
- SamplerTest：比例的累计误差不超过一条样本的长度，各组的 token 份额按权重收敛，取完一遍重新打乱，结果只由种子决定；
  中途保存状态（经过 torch.save / torch.load）再续上，与不中断时取到的样本相同。
- RouteTest：超过阈值的长样本另成纯文本微批，不超过上限地装在一起，单条超过上限的自己一个微批，样本不截断。
- ReplayLogTest：Replay.extend 写出的日志能被 verify_log 按配置重新模拟核对，续训时截掉续训起点之后的行。
- PackingTest（需要阶段 1 模型目录中的 processor 配置）：回放样本接在拼接的 ASR 行（音频加文字）后面时，ASR 部分逐个
  张量不变；每条文本样本从位置 0 编号，cu_seq_lens_q、seq_idx 标出边界，计损失的位置预测的正是标签，来源编号、空目标
  标记和 kl32 的 top-32 按位置对齐；纯文本微批同样。
- KLTest：LinearTopKL（有无尾部、输出层可训练与否）的损失和梯度与 float64 全词表直接计算一致。
- TinyModelTest（CPU，缩小的随机初始化模型）：纯文本微批经 mixed_token_losses 得到的 ntp 和 kl32 损失及梯度，与同一个
  模型整段前向后在全词表上直接计算的结果一致；ASR 行后面接上回放样本，ASR 位置的损失不变；没有回放样本时
  mixed_token_losses 与 token_losses 完全相同。
- ConfigTest：以前没有回放选项的训练配置照常续训，开启回放后续训核对能发现；选项组合的检查。
- PackedGpuTest（需要 CUDA 与 flash-attn、causal-conv1d、fla 的内核，缺一时跳过；Colab 冒烟测试中运行）：拼接行中的
  回放样本与单独成批时的损失一致，ASR 样本与不接回放时一致。
"""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pipeline.train.data import IGNORE_INDEX
from pipeline.train.replay import (
    GROUP_CODES,
    LOG_FILE,
    Replay,
    ReplayGroup,
    ReplaySampler,
    append_samples,
    parse_groups,
    route,
    text_batch,
    verify_log,
)
from pipeline.train.tests.test_offload import CUDA, MODEL_DIR, tiny_batch, tiny_model
from pipeline.train.tests.test_packing import kernels_available

HAS_MODEL = (MODEL_DIR / "config.json").exists()
IM_START, IM_END, ENDOFTEXT, THINK, THINK_END = 248045, 248046, 248044, 248068, 248069
USER_START = [IM_START, 846, 198]
SYSTEM_START = [IM_START, 8678, 198]
SYSTEM_BETWEEN = [198, IM_START, 846, 198]
THINK_GROUPS = ("free_chat_think", "free_sys_chat_think")
CHAT_BETWEEN = [198, IM_START, 74455, 198, THINK, 271, THINK_END, 271]
THINK_BETWEEN = [198, IM_START, 74455, 198, THINK, 198]


class FakeReplay:
    """在临时目录中按回放数据的格式写出各组样本（raw/ 与 teacher/ 的 shard、samples.jsonl）。"""

    def __init__(self, root, seed = 0):
        self.root = Path(root)
        self.rng = np.random.default_rng(seed = seed)

    def tokens(self, count):
        return self.rng.integers(low = 1000, high = 5000, size = count).tolist()

    def write_group(self, group, specs, split = "train"):
        """specs 为 [(pieces 描述, keep)]：描述是 [("ids", [...]) 或 (角色, token 列表)]。返回每条样本期望的 (ids, labels)。"""
        directory = self.root / group / split
        (directory / "raw").mkdir(parents = True, exist_ok = True)
        (directory / "teacher").mkdir(parents = True, exist_ok = True)
        generated = []
        records = []
        expected = []
        for number, (description, keep) in enumerate(specs):
            pieces = []
            ids, labels = [], []
            for kind, values in description:
                if kind == "ids":
                    pieces.append({"ids": list(values)})
                    ids += list(values)
                    labels += [IGNORE_INDEX] * len(values)
                    continue
                offset = len(generated)
                generated += list(values)
                pieces.append({"gen": kind, "length": len(values), "offset": offset, "keep": len(values)})
                ids += list(values)
                labels += list(values) if kind in ("text", "assistant") else [IGNORE_INDEX] * len(values)
            records.append({
                "sample_id": f"{group}-{split}-{number:07d}",
                "prompt_index": number,
                "shard": 0,
                "keep": keep,
                "drop_reason": None if keep else "loop",
                "pieces": pieces,
            })
            expected.append((ids, labels, keep))
        generated = np.asarray(a = generated, dtype = np.int32)
        generated.tofile(directory / "raw" / "shard-00000.ids.i32")
        lp = -self.rng.random(size = len(generated)).astype(np.float32)
        lp.tofile(directory / "raw" / "shard-00000.lp.f32")
        # 参考模型的 top-32：互不相同的 token，概率总和小于 1（其余是尾部）
        count = len(generated)
        top_ids = np.stack([self.rng.choice(a = 5000, size = 32, replace = False) for _ in range(count)]).astype(np.int32)
        probs = self.rng.dirichlet(alpha = np.ones(shape = 33), size = count)[:, :32]
        top_lp = np.log(np.sort(probs, axis = 1)[:, ::-1]).astype(np.float16)
        top_ids.tofile(directory / "teacher" / "shard-00000.top_ids.i32")
        top_lp.tofile(directory / "teacher" / "shard-00000.top_lp.f16")
        np.zeros(shape = count, dtype = np.float32).tofile(directory / "teacher" / "shard-00000.lse.f32")
        (lp - 0.01).astype(np.float32).tofile(directory / "teacher" / "shard-00000.lp.f32")
        with open(file = directory / "samples.jsonl", mode = "w", encoding = "utf-8") as handle:
            for record in records:
                handle.write(json.dumps(obj = record) + "\n")
        return expected, top_ids, top_lp

    def write_all(self, count = 6, split = "train", long_every = 0):
        """每组各 count 条，第 1 条 keep 为假；long_every > 0 时每隔几条放一条长样本。"""
        expected = {}
        for group in GROUP_CODES:
            specs = []
            for number in range(count):
                length = 3000 if long_every and number % long_every == long_every - 1 else int(self.rng.integers(low = 20, high = 80))
                body = self.tokens(count = length)
                if group == "free_raw":
                    description = [("ids", [198]), ("text", body + [ENDOFTEXT])]
                elif group == "cont_raw":
                    description = [("ids", self.tokens(count = 12)), ("text", body + [ENDOFTEXT])]
                elif group == "free_chat":
                    description = [("ids", USER_START), ("user", self.tokens(count = 7) + [IM_END]), ("ids", CHAT_BETWEEN),
                                   ("assistant", body + [IM_END])]
                elif group == "cont_chat":
                    prompt = USER_START + self.tokens(count = 15) + [IM_END] + CHAT_BETWEEN
                    description = [("ids", prompt), ("assistant", body + [IM_END])]
                elif group == "free_sys_chat":
                    description = [("ids", SYSTEM_START), ("system", self.tokens(count = 9) + [IM_END]), ("ids", SYSTEM_BETWEEN),
                                   ("user", self.tokens(count = 7) + [IM_END]), ("ids", CHAT_BETWEEN), ("assistant", body + [IM_END])]
                elif group == "free_sys_chat_think":
                    reasoning = body[: len(body) // 2]
                    answer = body[len(body) // 2:]
                    description = [("ids", SYSTEM_START), ("system", self.tokens(count = 9) + [IM_END]), ("ids", SYSTEM_BETWEEN),
                                   ("user", self.tokens(count = 5) + [IM_END]), ("ids", THINK_BETWEEN),
                                   ("assistant", reasoning + [THINK_END, 271] + answer + [IM_END])]
                else:
                    reasoning = body[: len(body) // 2]
                    answer = body[len(body) // 2:]
                    description = [("ids", USER_START), ("user", self.tokens(count = 5) + [IM_END]), ("ids", THINK_BETWEEN),
                                   ("assistant", reasoning + [THINK_END, 271] + answer + [IM_END])]
                specs.append((description, number != 1))
            expected[group] = self.write_group(group = group, specs = specs, split = split)
        return expected


class MaskTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fake = FakeReplay(root = self.tmp.name)
        self.expected = self.fake.write_all()

    def tearDown(self):
        self.tmp.cleanup()

    def test_loss_masks_per_group(self):
        for group, (expected, top_ids, top_lp) in self.expected.items():
            reader = ReplayGroup(root = self.tmp.name, group = group)
            kept = [item for item in expected if item[2]]
            self.assertEqual(len(reader), len(kept), msg = group)
            for index, (ids, labels, _) in enumerate(kept):
                sample = reader.sample(index = index, teacher = True, reference = True)
                self.assertEqual(sample["ids"].tolist(), ids, msg = group)
                self.assertEqual(sample["labels"].tolist(), labels, msg = group)
                supervised = sample["ids"][sample["labels"] != IGNORE_INDEX].tolist()
                self.assertEqual(sample["n_loss"], len(supervised))
                self.assertEqual(int(reader.losses[index]), len(supervised))
                if group == "free_raw":
                    self.assertEqual(supervised, ids[1:])
                if group == "cont_raw":
                    self.assertEqual(supervised, ids[12:])
                if group in ("free_chat", "cont_chat") + THINK_GROUPS + ("free_sys_chat",):
                    # 只有助手轮：从模板之后开始，以 <|im_end|> 结束；系统轮、用户轮的 <|im_end|> 不计
                    self.assertEqual(supervised[-1], IM_END)
                    self.assertEqual(supervised.count(IM_END), 1)
                    between = THINK_BETWEEN if group in THINK_GROUPS else CHAT_BETWEEN
                    start = len(ids) - len(supervised)
                    self.assertEqual(ids[start - len(between): start], between)
                if group in THINK_GROUPS:
                    self.assertIn(THINK_END, supervised)
                if group.startswith("free_sys_chat"):
                    # 系统轮在最前面，三个 <|im_end|> 中只有助手轮的一个计损失
                    self.assertEqual(ids[: len(SYSTEM_START)], SYSTEM_START)
                    self.assertEqual(ids.count(IM_END), 3)
                self.assertEqual(sample["teacher_ids"].shape, (len(supervised), 32))
                self.assertEqual(sample["ref_lp"].shape, (len(supervised),))
            # top-32 的行就是这些生成 token 在 raw shard 中的位置
            sample = reader.sample(index = 0, teacher = True)
            record = sample["record"]
            rows = [
                row for piece in record["pieces"] if piece.get("gen") in ("text", "assistant")
                for row in range(piece["offset"], piece["offset"] + piece["keep"])
            ]
            self.assertTrue(np.array_equal(sample["teacher_ids"], top_ids[rows]))
            self.assertTrue(np.array_equal(sample["teacher_lp"], top_lp[rows].astype(np.float32)))

    @unittest.skipUnless(os.environ.get("REPLAY_DIR"), "设置 REPLAY_DIR 时检查真实回放数据")
    def test_real_data_boundaries(self):
        root = os.environ["REPLAY_DIR"]
        for group in GROUP_CODES:
            if not (Path(root) / group / "eval" / "samples.jsonl").exists():
                continue
            reader = ReplayGroup(root = root, group = group, split = "eval")
            sample = reader.sample(index = 0, teacher = True, reference = True)
            ids, labels = sample["ids"], sample["labels"]
            first = int(np.argmax(a = labels != IGNORE_INDEX))
            self.assertTrue((labels[first:] != IGNORE_INDEX).all(), msg = group)
            if group in ("free_chat", "cont_chat"):
                self.assertEqual(ids[first - len(CHAT_BETWEEN): first].tolist(), CHAT_BETWEEN)
            if group == "free_sys_chat":
                self.assertEqual(ids[first - len(CHAT_BETWEEN): first].tolist(), CHAT_BETWEEN)
            if group in THINK_GROUPS:
                self.assertEqual(ids[first - len(THINK_BETWEEN): first].tolist(), THINK_BETWEEN)
                self.assertIn(THINK_END, ids[first:].tolist())
            if group.startswith("free_sys_chat"):
                self.assertEqual(ids[: len(SYSTEM_START)].tolist(), SYSTEM_START)
            hits = sample["teacher_ids"] == ids[first:][:, None]
            # 参考模型给实际 token 的对数概率与 top-32 中同一个 token 的值相同（float16 误差以内）
            self.assertLess(float(np.abs(sample["teacher_lp"][hits] - sample["ref_lp"][hits.any(axis = 1)]).max()), 0.01)


class FakeGroup:
    def __init__(self, losses):
        self.losses = np.asarray(a = losses)


def fake_groups(seed = 0):
    rng = np.random.default_rng(seed = seed)
    return {
        "free_raw": FakeGroup(losses = rng.integers(low = 100, high = 16000, size = 40)),
        "free_chat": FakeGroup(losses = rng.integers(low = 50, high = 1500, size = 200)),
        "free_chat_think": FakeGroup(losses = rng.integers(low = 500, high = 9000, size = 7)),
    }


class SamplerTest(unittest.TestCase):
    def setUp(self):
        self.groups = fake_groups()
        self.weights = {"free_raw": 1.0, "free_chat": 2.0, "free_chat_think": 1.0}
        rng = np.random.default_rng(seed = 1)
        self.asr = rng.integers(low = 200, high = 1200, size = 3000).tolist()

    def sampler(self, ratio = 0.5, seed = 7):
        return ReplaySampler(
            groups = self.groups,
            weights = self.weights,
            ratio = ratio,
            seed = seed,
        )

    def test_ratio_and_shares(self):
        sampler = self.sampler()
        longest = max(int(group.losses.max()) for group in self.groups.values())
        asr_total = 0
        replay_total = 0
        for tokens in self.asr:
            picked = sampler.take(asr_tokens = tokens)
            asr_total += tokens
            replay_total += sum(int(self.groups[name].losses[index]) for name, index in picked)
            # 累计的回放 token 数在 r × ASR 与其加一条样本之间
            self.assertGreaterEqual(replay_total, 0.5 * asr_total - 1e-6)
            self.assertLess(replay_total, 0.5 * asr_total + longest)
        tokens = sampler.state["tokens"]
        total = sum(tokens.values())
        for name, weight in self.weights.items():
            share = weight / sum(self.weights.values())
            self.assertLess(abs(tokens[name] - share * total), longest, msg = name)

    def test_epochs_cover_each_sample_once_and_reshuffle(self):
        sampler = self.sampler(ratio = 5.0)
        seen = {name: [] for name in self.weights}
        for tokens in self.asr[:400]:
            for name, index in sampler.take(asr_tokens = tokens):
                seen[name].append(index)
        think = seen["free_chat_think"]
        size = len(self.groups["free_chat_think"].losses)
        self.assertGreater(len(think), 3 * size)
        passes = [think[start: start + size] for start in range(0, 3 * size, size)]
        for part in passes:
            self.assertEqual(sorted(part), list(range(size)))
        self.assertNotEqual(passes[0], passes[1])

    def test_deterministic_and_seeded(self):
        runs = []
        for seed in (7, 7, 8):
            sampler = self.sampler(seed = seed)
            runs.append([sampler.take(asr_tokens = tokens) for tokens in self.asr[:300]])
        self.assertEqual(runs[0], runs[1])
        self.assertNotEqual(runs[0], runs[2])

    def test_resume_equality(self):
        straight = self.sampler()
        want = [straight.take(asr_tokens = tokens) for tokens in self.asr]
        first = self.sampler()
        got = [first.take(asr_tokens = tokens) for tokens in self.asr[:1234]]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.pt"
            torch.save(obj = {"replay": first.state_dict()}, f = path)
            state = torch.load(f = path, weights_only = False)["replay"]
        resumed = self.sampler()
        resumed.load_state_dict(state = state)
        got += [resumed.take(asr_tokens = tokens) for tokens in self.asr[1234:]]
        self.assertEqual(want, got)
        self.assertEqual(straight.state_dict(), resumed.state_dict())

    def test_rejects_other_groups(self):
        sampler = self.sampler()
        other = ReplaySampler(
            groups = self.groups,
            weights = {"free_raw": 1.0},
            ratio = 0.5,
            seed = 7,
        )
        with self.assertRaises(ValueError):
            other.load_state_dict(state = sampler.state_dict())

    def test_parse_groups(self):
        self.assertEqual(parse_groups(spec = "free_raw=1, cont_chat=0.5,free_chat=0"), {"free_raw": 1.0, "cont_chat": 0.5})
        self.assertEqual(list(parse_groups(spec = "cont_raw,free_raw=2")), ["cont_raw", "free_raw"])
        for bad in ("prompt_chat=1", "free_raw=1,free_raw=2", "free_raw=0", "free_raw=-1"):
            with self.assertRaises(ValueError, msg = bad):
                parse_groups(spec = bad)


def fake_sample(sample_id, length, loss, code = 2):
    ids = np.arange(length, dtype = np.int64) + 1000
    labels = np.full(shape = length, fill_value = IGNORE_INDEX, dtype = np.int64)
    labels[length - loss:] = ids[length - loss:]
    return {
        "group": "free_chat",
        "code": code,
        "sample_id": sample_id,
        "ids": ids,
        "labels": labels,
        "n_tokens": length,
        "n_loss": loss,
        "teacher_ids": np.tile(np.arange(32, dtype = np.int64), (loss, 1)),
        "teacher_lp": np.full(shape = (loss, 32), fill_value = -4.0, dtype = np.float32),
    }


class RouteTest(unittest.TestCase):
    def test_long_samples_get_own_micro_batches(self):
        samples = [
            fake_sample(sample_id = "a", length = 300, loss = 200),
            fake_sample(sample_id = "b", length = 5000, loss = 4000),
            fake_sample(sample_id = "c", length = 2048, loss = 2000),
            fake_sample(sample_id = "d", length = 9000, loss = 8000),
            fake_sample(sample_id = "e", length = 20000, loss = 19000),
            fake_sample(sample_id = "f", length = 4000, loss = 3000),
        ]
        inline, batches = route(samples = samples, inline_max_tokens = 2048, micro_batch_tokens = 16640)
        self.assertEqual([sample["sample_id"] for sample in inline], ["a", "c"])
        self.assertEqual([[sample["sample_id"] for sample in batch] for batch in batches], [["b", "d"], ["e"], ["f"]])
        for batch in batches:
            if len(batch) > 1:
                self.assertLessEqual(sum(sample["n_tokens"] for sample in batch), 16640)
        built = text_batch(samples = batches[1], kl = True)
        # 超过上限的单条样本原样放进一个微批，不截断
        self.assertEqual(built["input_ids"].shape[1], 20000)
        self.assertEqual(int(built["loss_index"].shape[0]), 19000)
        self.assertEqual(built["teacher_ids"].shape, (19000, 32))


class ReplayLogTest(unittest.TestCase):
    def test_log_verifies_and_truncates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "replay"
            FakeReplay(root = root).write_all(count = 30, long_every = 5)
            output = Path(tmp) / "run"
            output.mkdir()
            spec = "free_raw=1,free_chat=1,cont_raw=1,cont_chat=1,free_chat_think=1"
            (output / "train_config.json").write_text(data = json.dumps(obj = {
                "replay_dir": str(root),
                "replay_groups": spec,
                "replay_ratio": 0.5,
                "replay_split": "train",
                "seed": 3,
            }))

            def make():
                return Replay(
                    root = root,
                    weights = parse_groups(spec = spec),
                    ratio = 0.5,
                    seed = 3,
                    loss = "kl32",
                    log_path = output / LOG_FILE,
                )

            empty_batch = make_packed_asr_stub()
            replay = make()
            asr = [40, 75, 30, 120, 66, 90, 33, 41]
            for step, tokens in enumerate(asr[:5], start = 1):
                replay.extend(
                    micro_batches = [empty_batch],
                    asr_tokens = tokens,
                    step = step,
                    probe = step == 1,
                )
            state = replay.state_dict()
            # 第 6 步记了日志后中断，从第 5 步的检查点续训：第 6 步重新训练、重新记录
            replay.extend(micro_batches = [empty_batch], asr_tokens = asr[5], step = 6)
            resumed = make()
            resumed.load_state_dict(state = state)
            resumed.truncate_log(step = 5)
            for step, tokens in enumerate(asr[5:], start = 6):
                batches, info = resumed.extend(micro_batches = [empty_batch], asr_tokens = tokens, step = step)
                self.assertEqual(info["replay_tokens"], sum(sum(batch["source_tokens"][1:]) for batch in batches))
            lines = [json.loads(s = line) for line in (output / LOG_FILE).read_text().splitlines()]
            self.assertEqual([line["step"] for line in lines], list(range(1, 9)))
            self.assertIn("probe", lines[0])
            result = verify_log(output_dir = output)
            self.assertIsNone(result["mismatch"])
            self.assertEqual(result["steps"], 8)
            # 改动一行的 ASR token 数后核对失败
            lines[2]["asr_tokens"] += 500
            (output / LOG_FILE).write_text(data = "".join(json.dumps(obj = line) + "\n" for line in lines))
            self.assertIsNotNone(verify_log(output_dir = output)["mismatch"])


def make_packed_asr_stub():
    """只有两条短文本的拼接「ASR」行（结构与 data.pack_batch 的结果相同，没有音频），日志测试用。"""
    samples = [fake_sample(sample_id = "x", length = 10, loss = 4), fake_sample(sample_id = "y", length = 7, loss = 3)]
    batch = text_batch(samples = samples)
    batch.update({
        "audio_mask": torch.zeros(size = batch["input_ids"].shape, dtype = torch.bool),
        "sample_ids": ["x", "y"],
        "empty_rows": torch.zeros(size = (2,), dtype = torch.bool),
    })
    for key in ("loss_source", "source_tokens", "replay_ids"):
        batch.pop(key)
    return batch


@unittest.skipUnless(HAS_MODEL, "需要阶段 1 模型目录中的代码与配置")
class PackingTest(unittest.TestCase):
    def setUp(self):
        from pipeline.train.data import pack_batch

        self.asr = pack_batch(batch = tiny_batch(audio_windows = (50, 800), prompt_format = "cpt"))
        self.samples = [
            fake_sample(
                sample_id = "r0",
                length = 40,
                loss = 25,
                code = 2,
            ),
            fake_sample(
                sample_id = "r1",
                length = 17,
                loss = 16,
                code = 5,
            ),
        ]

    def test_append_keeps_asr_and_marks_text(self):
        before = copy.deepcopy(self.asr)
        mixed = append_samples(batch = self.asr, samples = self.samples, kl = True)
        length = before["input_ids"].shape[1]
        rows = len(before["cu_seq_lens_q"]) - 1
        positions = before["loss_index"].shape[0]
        # 原来的 ASR 部分逐个张量不变（传入的批次也没有被改动）
        for key in ("input_ids", "labels", "audio_mask", "mm_token_type_ids", "seq_idx"):
            self.assertTrue(torch.equal(mixed[key][:, :length], before[key]), msg = key)
            self.assertTrue(torch.equal(self.asr[key], before[key]), msg = key)
        self.assertTrue(torch.equal(mixed["position_ids"][:, :, :length], before["position_ids"]))
        for key in ("loss_index", "loss_targets", "loss_rows"):
            self.assertTrue(torch.equal(mixed[key][:positions], before[key]), msg = key)
        self.assertTrue(torch.equal(mixed["cu_seq_lens_q"][: rows + 1], before["cu_seq_lens_q"]))
        for key in ("audio_padded_feature", "audio_cu_seqlens", "audio_chunk_lengths", "audio_valid_indices"):
            self.assertIs(mixed[key], self.asr[key])
        # 文本样本：从位置 0 编号，边界与行号正确，音频标记为假
        start = length
        for number, sample in enumerate(self.samples):
            end = start + sample["n_tokens"]
            row = rows + number
            self.assertEqual(int(mixed["cu_seq_lens_q"][row + 1]), end)
            self.assertTrue(torch.equal(mixed["input_ids"][0, start:end], torch.from_numpy(sample["ids"])))
            self.assertEqual(mixed["position_ids"][1, 0, start:end].tolist(), list(range(sample["n_tokens"])))
            self.assertTrue(bool((mixed["seq_idx"][0, start:end] == row).all()))
            self.assertFalse(bool(mixed["audio_mask"][0, start:end].any()))
            start = end
        self.assertEqual(mixed["input_ids"].shape[1], start)
        self.assertEqual(mixed["attention_mask"].shape[1], start)
        self.assertEqual(mixed["max_length_q"], max(before["max_length_q"], 40))
        # 计损失的位置：下一个 token 就是目标，来源编号与行号对应
        self.assertTrue(torch.equal(mixed["input_ids"][0, mixed["loss_index"] + 1], mixed["loss_targets"]))
        self.assertEqual(mixed["loss_source"][:positions].tolist(), [0] * positions)
        self.assertEqual(mixed["loss_source"][positions:].tolist(), [2] * 25 + [5] * 16)
        # 来源依次为 ASR、free_raw、free_chat、cont_raw、cont_chat、free_chat_think、free_sys_chat、free_sys_chat_think
        self.assertEqual(mixed["source_tokens"], [positions, 0, 25, 0, 0, 16, 0, 0])
        self.assertEqual(mixed["loss_rows"][positions:].tolist(), [rows] * 25 + [rows + 1] * 16)
        self.assertEqual(mixed["empty_rows"].tolist(), before["empty_rows"].tolist() + [False, False])
        # kl32：从 kl_start 起是回放位置，top-32 与之一一对应
        self.assertEqual(mixed["kl_start"], positions)
        self.assertEqual(mixed["teacher_ids"].shape, (41, 32))
        # 训练循环按 labels 数 ASR 的 token；回放部分的标签与样本相同
        self.assertEqual(int((mixed["labels"][:, 1:length] != IGNORE_INDEX).sum()), positions)

    def test_without_samples(self):
        unchanged = append_samples(batch = self.asr, samples = [], kl = True)
        self.assertNotIn("kl_start", unchanged)
        self.assertEqual(unchanged["loss_source"].tolist(), [0] * self.asr["loss_index"].shape[0])
        self.assertTrue(torch.equal(unchanged["input_ids"], self.asr["input_ids"]))

    def test_text_batch(self):
        batch = text_batch(samples = self.samples, kl = True)
        self.assertEqual(batch["cu_seq_lens_q"].tolist(), [0, 40, 57])
        self.assertEqual(batch["cu_seq_lens_q"].dtype, torch.int32)
        self.assertEqual(batch["position_ids"][0, 0].tolist(), list(range(40)) + list(range(17)))
        self.assertTrue(torch.equal(batch["input_ids"][0, batch["loss_index"] + 1], batch["loss_targets"]))
        self.assertEqual(batch["kl_start"], 0)
        self.assertNotIn("audio_padded_feature", batch)
        self.assertEqual(batch["sample_ids"], [])


def reference_kl(hidden, weight, top_ids, top_lp, tail):
    """float64 全词表直接计算的截断 KL（autograd 求梯度）。"""
    log_q = torch.log_softmax(input = hidden @ weight.t(), dim = -1)
    ref_p = top_lp.double().exp()
    log_q_top = log_q.gather(dim = -1, index = top_ids)
    loss = (ref_p * (top_lp.double() - log_q_top)).sum(dim = -1)
    if tail:
        p_tail = (1 - ref_p.sum(dim = -1)).clamp(min = 0)
        q_tail = 1 - log_q_top.exp().sum(dim = -1)
        loss = loss + p_tail * (p_tail.log() - q_tail.log())
    return loss


class KLTest(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(0)
        count, dim, vocab = 50, 16, 97
        self.hidden = torch.randn(size = (count, dim), generator = generator, dtype = torch.float64)
        self.weight = torch.randn(size = (vocab, dim), generator = generator, dtype = torch.float64)
        self.targets = torch.randint(
            low = 0,
            high = vocab,
            size = (count,),
            generator = generator,
        )
        self.scale = torch.rand(size = (count,), generator = generator, dtype = torch.float64)
        self.top_ids = torch.stack([torch.randperm(vocab, generator = generator)[:32] for _ in range(count)])
        probs = torch.distributions.Dirichlet(concentration = torch.ones(33, dtype = torch.float64)).sample(sample_shape = (count,))
        # 有的位置尾部很小，有的较大
        probs[:5, 32] = 1e-7
        probs = probs / probs.sum(dim = -1, keepdim = True)
        self.top_lp = probs[:, :32].log().float()

    def run_both(self, tail, weight_grad):
        from pipeline.train.model import linear_top_kl

        hidden = self.hidden.clone().requires_grad_(requires_grad = True)
        weight = self.weight.clone().requires_grad_(requires_grad = weight_grad)
        losses, nll, correct = linear_top_kl(
            hidden_states = hidden,
            weight = weight,
            top_ids = self.top_ids,
            top_lp = self.top_lp,
            targets = self.targets,
            tail = tail,
            chunk_size = 8,
        )
        (losses * self.scale).sum().backward()
        ref_hidden = self.hidden.clone().requires_grad_(requires_grad = True)
        ref_weight = self.weight.clone().requires_grad_(requires_grad = weight_grad)
        expected = reference_kl(
            hidden = ref_hidden,
            weight = ref_weight,
            top_ids = self.top_ids,
            top_lp = self.top_lp,
            tail = tail,
        )
        (expected * self.scale).sum().backward()
        logits = self.hidden @ self.weight.t()
        expected_nll = F.cross_entropy(input = logits, target = self.targets, reduction = "none")
        torch.testing.assert_close(
            actual = losses.double(),
            expected = expected.detach(),
            rtol = 1e-4,
            atol = 1e-5,
        )
        torch.testing.assert_close(
            actual = nll.double(),
            expected = expected_nll,
            rtol = 1e-5,
            atol = 1e-5,
        )
        self.assertTrue(torch.equal(correct, logits.argmax(dim = -1) == self.targets))
        torch.testing.assert_close(
            actual = hidden.grad,
            expected = ref_hidden.grad,
            rtol = 1e-4,
            atol = 1e-5,
        )
        if weight_grad:
            torch.testing.assert_close(
                actual = weight.grad,
                expected = ref_weight.grad,
                rtol = 1e-4,
                atol = 1e-5,
            )
        else:
            self.assertIsNone(weight.grad)
        return losses

    def test_matches_full_vocab(self):
        for tail in (False, True):
            for weight_grad in (False, True):
                with self.subTest(tail = tail, weight_grad = weight_grad):
                    self.run_both(tail = tail, weight_grad = weight_grad)

    def test_minimum_at_reference(self):
        """尾部项打开时，学生分布等于参考分布（S 外按任意比例分摊尾部）就是最小点，损失为 0。"""
        from pipeline.train.model import linear_top_kl

        vocab = 40
        weight = torch.eye(vocab, dtype = torch.float64)
        top_ids = torch.arange(32)[None]
        probs = torch.distributions.Dirichlet(concentration = torch.ones(vocab, dtype = torch.float64)).sample()
        hidden = probs.log()[None].clone().requires_grad_(requires_grad = True)
        losses, _, _ = linear_top_kl(
            hidden_states = hidden,
            weight = weight,
            top_ids = top_ids,
            top_lp = probs[:32].log()[None].float(),
            targets = torch.zeros(size = (1,), dtype = torch.long),
            tail = True,
        )
        losses.sum().backward()
        self.assertLess(abs(float(losses)), 1e-6)
        self.assertLess(float(hidden.grad.abs().max()), 1e-6)


def tiny_text_sample(kl):
    """30 个 token 的 cont_raw 样本，后 18 个计损失；kl 为真时带随机的 top-32。"""
    rng = np.random.default_rng(seed = 5)
    ids = rng.integers(low = 100, high = 3000, size = 30).astype(np.int64)
    labels = np.full(shape = 30, fill_value = IGNORE_INDEX, dtype = np.int64)
    labels[12:] = ids[12:]
    sample = {"group": "cont_raw", "code": 3, "sample_id": "t", "ids": ids, "labels": labels, "n_tokens": 30, "n_loss": 18}
    if kl:
        top_ids = np.stack([rng.choice(a = 3000, size = 32, replace = False) for _ in range(18)]).astype(np.int64)
        probs = rng.dirichlet(alpha = np.ones(shape = 33), size = 18)[:, :32]
        sample["teacher_ids"] = top_ids
        sample["teacher_lp"] = np.log(probs).astype(np.float32)
    return sample


@unittest.skipUnless(HAS_MODEL, "需要阶段 1 模型目录中的代码与配置")
class TinyModelTest(unittest.TestCase):
    """缩小的随机初始化模型（float32，CPU），语言模型参与训练。

    装了 causal-conv1d、fla 时 transformers 选用它们的 CUDA 内核，CPU 上不能运行，所以线性注意力换回 PyTorch 参考实现
    （check_packing.reference_linear_attention）。参考实现不认样本边界，这里的拼接行要么只有一条样本，要么只看行首
    的 ASR 部分（因果计算不受后文影响）。
    """

    def setUp(self):
        from pipeline.train.check_packing import reference_linear_attention

        self.reference = reference_linear_attention()
        self.reference.__enter__()

    def tearDown(self):
        self.reference.__exit__(None, None, None)

    def model(self):
        from pipeline.train.model import LANGUAGE_MODEL_PREFIXES

        model = tiny_model(dtype = torch.float32, device = "cpu")
        for name, parameter in model.named_parameters():
            if name.startswith(LANGUAGE_MODEL_PREFIXES):
                parameter.requires_grad_(requires_grad = True)
        return model

    def test_text_batch_matches_direct(self):
        from pipeline.train.model import mixed_token_losses, move_batch

        for kl, tail in ((False, False), (True, False), (True, True)):
            with self.subTest(kl = kl, tail = tail):
                sample = tiny_text_sample(kl = kl)
                model = self.model()
                inputs = move_batch(batch = text_batch(samples = [sample], kl = kl), device = "cpu", feature_dtype = torch.float32)
                losses, _, nll = mixed_token_losses(model = model, inputs = inputs, kl_tail = tail)
                losses.sum().backward()
                got = {name: parameter.grad.clone() for name, parameter in model.named_parameters() if parameter.grad is not None}

                reference = self.model()
                ids = torch.from_numpy(sample["ids"])[None]
                hidden = reference.model.language_model(input_ids = ids, use_cache = False).last_hidden_state[0]
                positions = torch.arange(12, 30)
                logits = hidden[positions - 1] @ reference.lm_head.weight.t()
                if kl:
                    expected = reference_kl(
                        hidden = hidden[positions - 1],
                        weight = reference.lm_head.weight,
                        top_ids = torch.from_numpy(sample["teacher_ids"]),
                        top_lp = torch.from_numpy(sample["teacher_lp"]),
                        tail = tail,
                    )
                else:
                    expected = F.cross_entropy(input = logits, target = ids[0, positions], reduction = "none")
                expected.sum().backward()
                torch.testing.assert_close(
                    actual = losses,
                    expected = expected.float(),
                    rtol = 1e-4,
                    atol = 1e-4,
                )
                torch.testing.assert_close(
                    actual = nll,
                    expected = F.cross_entropy(input = logits, target = ids[0, positions], reduction = "none").detach(),
                    rtol = 1e-4,
                    atol = 1e-4,
                )
                want = {name: parameter.grad for name, parameter in reference.named_parameters() if parameter.grad is not None}
                self.assertEqual(set(got), set(want))
                for name in want:
                    torch.testing.assert_close(
                        actual = got[name],
                        expected = want[name],
                        rtol = 1e-3,
                        atol = 1e-5,
                        msg = name,
                    )

    def test_asr_losses_unchanged_by_appended_replay(self):
        """回放样本接在行尾：前面的 ASR 位置只看前文，损失不变（CPU 上没有分段内核，但因果计算不受后文影响）。"""
        from pipeline.train.data import pack_batch
        from pipeline.train.model import mixed_token_losses, move_batch, token_losses

        config = tiny_model(dtype = torch.float32, device = "cpu").config.audio_config
        asr = pack_batch(batch = tiny_batch(audio_windows = (config.n_window, config.n_window_infer), prompt_format = "cpt"))
        model = self.model()
        with torch.no_grad():
            plain, plain_correct, _ = token_losses(model = model, inputs = move_batch(batch = asr, device = "cpu", feature_dtype = torch.float32))
            only_asr = append_samples(batch = asr, samples = [], kl = True)
            same, same_correct, same_nll = mixed_token_losses(model = model, inputs = move_batch(batch = only_asr, device = "cpu", feature_dtype = torch.float32))
            mixed = append_samples(batch = asr, samples = [tiny_text_sample(kl = True)], kl = True)
            losses, _, _ = mixed_token_losses(
                model = model,
                inputs = move_batch(batch = mixed, device = "cpu", feature_dtype = torch.float32),
                kl_tail = True,
            )
        # 没有回放样本时与 token_losses 完全相同
        self.assertTrue(torch.equal(plain, same))
        self.assertTrue(torch.equal(plain_correct, same_correct))
        self.assertTrue(torch.equal(plain, same_nll))
        count = plain.shape[0]
        torch.testing.assert_close(
            actual = losses[:count],
            expected = plain,
            rtol = 1e-5,
            atol = 1e-5,
        )
        self.assertEqual(losses.shape[0], count + 18)


@unittest.skipUnless(HAS_MODEL, "需要阶段 1 模型目录中的代码与配置")
class StepTest(unittest.TestCase):
    """训练循环的一步（CPU）：Replay.extend 给出的微批（ASR 行接短样本、长样本另成微批）经 replay_forward_backward。"""

    def setUp(self):
        from pipeline.train.check_packing import reference_linear_attention

        self.reference = reference_linear_attention()
        self.reference.__enter__()
        self.tmp = tempfile.TemporaryDirectory()
        FakeReplay(root = self.tmp.name).write_all(count = 12, long_every = 4)

    def tearDown(self):
        self.reference.__exit__(None, None, None)
        self.tmp.cleanup()

    def model(self):
        from pipeline.train.model import LANGUAGE_MODEL_PREFIXES

        model = tiny_model(dtype = torch.float32, device = "cpu")
        for name, parameter in model.named_parameters():
            if name.startswith(LANGUAGE_MODEL_PREFIXES):
                parameter.requires_grad_(requires_grad = True)
        return model

    def test_step_accounting_and_gradient(self):
        import contextlib

        from pipeline.train.data import pack_batch
        from pipeline.train.model import mixed_token_losses, move_batch, token_losses
        from pipeline.train.train import replay_forward_backward

        config = tiny_model(dtype = torch.float32, device = "cpu").config.audio_config
        asr = pack_batch(batch = tiny_batch(audio_windows = (config.n_window, config.n_window_infer), prompt_format = "cpt"))
        asr_tokens = int((asr["labels"][:, 1:] != IGNORE_INDEX).sum())
        for loss, tail in (("ntp", False), ("kl32", True)):
            with self.subTest(loss = loss):
                replay = Replay(
                    root = self.tmp.name,
                    weights = parse_groups(spec = "free_raw=1,free_chat=1,free_chat_think=1"),
                    ratio = 40.0,
                    seed = 0,
                    loss = loss,
                    inline_max_tokens = 100,
                    micro_batch_tokens = 4000,
                )
                batches, info = replay.extend(micro_batches = [asr], asr_tokens = asr_tokens, step = 1)
                self.assertGreaterEqual(len(batches), 2, msg = "应有长样本另成的纯文本微批")
                self.assertEqual(info["long_batches"], len(batches) - 1)
                self.assertEqual(info["replay_tokens"], sum(sum(batch["source_tokens"][1:]) for batch in batches))
                n_tokens = replay.normalizer(asr_tokens = asr_tokens, replay_tokens = info["replay_tokens"])
                self.assertEqual(n_tokens, asr_tokens + info["replay_tokens"])

                model = self.model()
                totals = replay_forward_backward(
                    model = model,
                    micro_batches = batches,
                    n_tokens = n_tokens,
                    device = "cpu",
                    feature_dtype = torch.float32,
                    forward_context = contextlib.nullcontext,
                    kl_tail = tail,
                )
                got = {name: parameter.grad.clone() for name, parameter in model.named_parameters() if parameter.grad is not None}

                reference = self.model()
                expected_sources = [0.0] * len(totals["loss"])
                for batch in batches:
                    inputs = move_batch(batch = batch, device = "cpu", feature_dtype = torch.float32)
                    losses, _, _ = mixed_token_losses(model = reference, inputs = inputs, kl_tail = tail)
                    (losses.sum() / n_tokens).backward()
                    for code, value in zip(batch["loss_source"].tolist(), losses.detach().tolist()):
                        expected_sources[code] += value
                for name, parameter in reference.named_parameters():
                    if parameter.grad is not None:
                        torch.testing.assert_close(
                            actual = got[name],
                            expected = parameter.grad,
                            rtol = 1e-4,
                            atol = 1e-6,
                            msg = name,
                        )
                # 各来源的损失之和（float32 累加，与 Python 的双精度求和相差在 float32 舍入以内）
                for got_value, want in zip(totals["loss"], expected_sources):
                    self.assertLess(abs(got_value - want), 1e-5 * max(1.0, abs(want)))
                # ASR 部分的损失之和与没有回放时相同；空目标样本只在 ASR 中，按原来的行号统计
                with torch.no_grad():
                    plain, _, rows = token_losses(model = reference, inputs = move_batch(batch = asr, device = "cpu", feature_dtype = torch.float32))
                self.assertLess(abs(totals["loss"][0] - float(plain.sum())), 1e-5 * float(plain.sum()))
                expected_empty = float((plain * asr["empty_rows"][rows]).sum())
                self.assertGreater(expected_empty, 0.0)
                self.assertLess(abs(totals["empty_loss"] - expected_empty), 1e-4 * max(1.0, expected_empty))
                self.assertLess(abs(totals["objective"] - sum(totals["loss"])), 1e-5 * totals["objective"])


class ConfigTest(unittest.TestCase):
    ARGV = [
        "--model-dir", "/repo/qwen3.5-2b-mllm",
        "--train-manifest", "/data/train/manifest-v2b.jsonl",
        "--target", "original",
        "--output-dir", "/work/runs/c",
        "--train-language-model",
        "--language-model-lr", "1e-5",
        "--pack-sequences",
        "--precompute-metadata",
    ]
    REPLAY = ["--replay-dir", "/data/replay-noloop", "--replay-groups", "free_raw=1", "--replay-ratio", "0.5"]

    def test_old_configs_resume_without_replay(self):
        from pipeline.train.train import parse_args, resume_mismatches, run_config

        saved = run_config(args = parse_args(argv = self.ARGV))
        for key in [key for key in saved if key.startswith("replay_")]:
            saved.pop(key)
        self.assertEqual(resume_mismatches(saved = saved, config = run_config(args = parse_args(argv = self.ARGV))), [])
        with_replay = run_config(args = parse_args(argv = self.ARGV + self.REPLAY))
        self.assertEqual(
            resume_mismatches(saved = saved, config = with_replay),
            ["replay_dir", "replay_groups", "replay_ratio"],
        )
        kl = run_config(args = parse_args(argv = self.ARGV + self.REPLAY + ["--replay-loss", "kl32"]))
        self.assertEqual(resume_mismatches(saved = with_replay, config = kl), ["replay_loss"])
        # 样本放在哪个微批、冒烟测试的检查不影响损失，不在续训核对中，也不写进配置
        moved = run_config(args = parse_args(argv = self.ARGV + self.REPLAY + [
            "--replay-inline-max-tokens", "512",
            "--replay-micro-batch-tokens", "8192",
            "--replay-probe-longest",
        ]))
        self.assertEqual(resume_mismatches(saved = with_replay, config = moved), [])
        self.assertNotIn("replay_probe_longest", moved)

    def test_argument_checks(self):
        from pipeline.train.train import check_replay_args, parse_args

        check_replay_args(args = parse_args(argv = self.ARGV))
        check_replay_args(args = parse_args(argv = self.ARGV + self.REPLAY + ["--replay-loss", "kl32", "--replay-kl-tail"]))
        # 语言模型参数在主机内存中更新时一步内各微批的梯度在主机上累加（placement.py），可以与回放同用。
        for host in (["--offload-optimizer"], ["--gradient-checkpointing", "--lm-blocks-to-swap", "8"]):
            check_replay_args(args = parse_args(argv = self.ARGV + self.REPLAY + host))
        bad = [
            self.ARGV + ["--replay-groups", "free_raw=1"],
            self.ARGV + ["--replay-loss", "kl32"],
            self.ARGV + ["--replay-dir", "/data/replay-noloop", "--replay-ratio", "0.5"],
            self.ARGV + self.REPLAY + ["--replay-kl-tail"],
            self.ARGV + self.REPLAY[:-1] + ["0"],
            [arg for arg in self.ARGV if arg != "--pack-sequences"] + self.REPLAY,
            [arg for arg in self.ARGV if arg not in ("--train-language-model", "--language-model-lr", "1e-5")] + self.REPLAY,
        ]
        for argv in bad:
            with self.assertRaises(SystemExit, msg = argv):
                check_replay_args(args = parse_args(argv = argv))


@unittest.skipUnless(
    CUDA and HAS_MODEL and kernels_available(),
    "需要 CUDA、阶段 1 模型目录中的代码与配置，以及 flash-attn、causal-conv1d、fla",
)
class PackedGpuTest(unittest.TestCase):
    """拼接内核按样本边界分段：同一行中的音频样本与回放样本互不影响。"""

    def test_mixed_row_matches_separate(self):
        from pipeline.train.data import pack_batch
        from pipeline.train.model import PACKED_ATTENTION, check_packing_kernels, mixed_token_losses, move_batch

        model = tiny_model(dtype = torch.bfloat16)
        model.set_attn_implementation(attn_implementation = PACKED_ATTENTION)
        check_packing_kernels(model = model)
        config = model.config.audio_config
        asr = pack_batch(batch = tiny_batch(audio_windows = (config.n_window, config.n_window_infer), prompt_format = "cpt"))
        samples = [tiny_text_sample(kl = True), fake_sample(
            sample_id = "g",
            length = 50,
            loss = 20,
            code = 5,
        )]
        samples[1]["ids"] = samples[1]["ids"] % 3000 + 100
        samples[1]["labels"] = np.where(samples[1]["labels"] == IGNORE_INDEX, IGNORE_INDEX, samples[1]["ids"])

        def losses(batch, tail = True):
            with torch.no_grad():
                inputs = move_batch(batch = batch, device = "cuda", feature_dtype = torch.bfloat16)
                return mixed_token_losses(model = model, inputs = inputs, kl_tail = tail)[0].float()

        mixed = losses(batch = append_samples(batch = asr, samples = samples, kl = True))
        alone_asr = losses(batch = append_samples(batch = asr, samples = [], kl = True))
        alone_text = losses(batch = text_batch(samples = samples, kl = True))
        count = alone_asr.shape[0]
        # 拼接行变长后矩阵乘法可能换一种分块，bfloat16 下允许舍入误差
        torch.testing.assert_close(
            actual = mixed[:count],
            expected = alone_asr,
            rtol = 2e-2,
            atol = 2e-2,
        )
        torch.testing.assert_close(
            actual = mixed[count:],
            expected = alone_text,
            rtol = 2e-2,
            atol = 2e-2,
        )


if __name__ == "__main__":
    unittest.main()
