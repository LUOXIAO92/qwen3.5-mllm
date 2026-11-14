"""梯度检查点、块流式加载、激活值卸载、fp8 与变长音频注意力的一致性：python -m unittest pipeline.train.tests.test_offload

需要 CUDA；没有 GPU 时跳过。前一部分在一组小的冻结层上比较输出和传回可训练层的梯度，后一部分在按阶段 1
配置缩小的随机初始化模型上比较 token_losses 的损失和 projector 的梯度。
"""

import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pipeline.train.offload import ActivationOffload, FrozenBlocks
from pipeline.train.quant import FP8Linear, quantize_fp8

CUDA = torch.cuda.is_available()
MODEL_DIR = Path(__file__).resolve().parents[3] / "qwen3.5-0.8b-mllm"


class Block(nn.Module):
    """两种形状交替的小层，检验缓冲池按形状复用。"""

    def __init__(self, dim, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape = dim)
        self.up = nn.Linear(in_features = dim, out_features = hidden)
        self.down = nn.Linear(in_features = hidden, out_features = dim)

    def forward(self, hidden_states, scale = None):
        output = self.down(torch.nn.functional.gelu(input = self.up(self.norm(hidden_states))))
        if scale is not None:
            output = output * scale
        return hidden_states + output


class Stack(nn.Module):
    def __init__(self, dim = 32, count = 6):
        super().__init__()
        self.projector = nn.Linear(in_features = dim, out_features = dim)
        self.layers = nn.ModuleList(modules = [Block(dim = dim, hidden = 64 if index % 2 else 96) for index in range(count)])

    def forward(self, inputs, scale):
        hidden_states = self.projector(inputs)
        for layer in self.layers:
            hidden_states = layer(hidden_states, scale = scale)
        return hidden_states


def build_stack():
    torch.manual_seed(seed = 0)
    model = Stack().to(device = "cuda")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(requires_grad = name.startswith("projector."))
    return model


def train_steps(model, steps = 3, offload = False):
    """前向、反传若干次，返回每次的输出和 projector 权重的梯度。"""
    generator = torch.Generator().manual_seed(1)
    inputs = torch.randn(size = (4, 10, 32), generator = generator).to(device = "cuda")
    scale = torch.full(size = (32,), fill_value = 0.5, device = "cuda")
    outputs, grads = [], []
    for _ in range(steps):
        model.projector.weight.grad = None
        context = ActivationOffload(min_bytes = 0) if offload else torch.enable_grad()
        with context:
            output = model(inputs = inputs, scale = scale)
        output.square().sum().backward()
        torch.cuda.synchronize()
        outputs.append(output.detach().clone())
        grads.append(model.projector.weight.grad.clone())
    return outputs, grads


@unittest.skipUnless(CUDA, "需要 CUDA")
class FrozenBlocksTest(unittest.TestCase):
    def assert_same(self, reference, result, label):
        for want, got in zip(reference[0] + reference[1], result[0] + result[1]):
            self.assertTrue(torch.allclose(input = want, other = got, atol = 1e-5, rtol = 1e-5), msg = f"{label}: {(want - got).abs().max()}")

    def setUp(self):
        self.reference = train_steps(model = build_stack())

    def run_with(self, checkpoint, swap_count, offload = False, no_prefetch = False):
        model = build_stack()
        stack = FrozenBlocks(
            blocks = model.layers,
            checkpoint = checkpoint,
            swap_count = swap_count,
        )
        if no_prefetch:
            stack.swap.prefetch = lambda index: None
        result = train_steps(model = model, offload = offload)
        return model, stack, result

    def test_checkpoint(self):
        _, _, result = self.run_with(checkpoint = True, swap_count = 0)
        self.assert_same(reference = self.reference, result = result, label = "checkpoint")

    def test_streaming(self):
        for count in (2, 6):
            for no_prefetch in (False, True):
                model, stack, result = self.run_with(checkpoint = True, swap_count = count, no_prefetch = no_prefetch)
                self.assert_same(reference = self.reference, result = result, label = f"swap {count} no_prefetch {no_prefetch}")
                # 走完前向和反传后，流式层都已释放，参数指回锁页主机内存。
                self.assertEqual(stack.swap.resident, {})
                for index in range(6 - count, 6):
                    self.assertEqual(model.layers[index].up.weight.device.type, "cpu")
                    self.assertTrue(model.layers[index].up.weight.is_pinned())
                stack.remove()
                self.assertEqual(model.layers[5].up.weight.device.type, "cuda")
                self.assert_same(reference = self.reference, result = train_steps(model = model), label = "removed")

    def test_window_and_pool(self):
        model = build_stack()
        stack = FrozenBlocks(
            blocks = model.layers,
            checkpoint = True,
            swap_count = 6,
        )
        seen = []
        allocations = []
        upload = stack.swap._upload

        def counting_upload(index):
            before = {key: len(bucket) for key, bucket in stack.swap.pool.items()}
            upload(index = index)
            key = stack.swap._key(host = stack.swap.host[index])
            allocations.append(before.get(key, 0) == 0)
            seen.append(len(stack.swap.resident))

        stack.swap._upload = counting_upload
        train_steps(model = model, steps = 2)
        # 同时驻留的流式层不超过两层；两种形状各自从缓冲池复用，大多数上传不新申请显存。
        self.assertLessEqual(max(seen), 2)
        self.assertLessEqual(sum(allocations), 4)

    def test_offload_activations(self):
        _, _, result = self.run_with(checkpoint = True, swap_count = 0, offload = True)
        self.assert_same(reference = self.reference, result = result, label = "offload")
        _, _, result = self.run_with(checkpoint = True, swap_count = 6, offload = True)
        self.assert_same(reference = self.reference, result = result, label = "offload + swap")

    def test_no_grad_streaming(self):
        model = build_stack()
        inputs = torch.randn(size = (4, 10, 32), device = "cuda")
        with torch.no_grad():
            want = model(inputs = inputs, scale = None)
        FrozenBlocks(blocks = model.layers, swap_count = 6)
        with torch.no_grad():
            got = model(inputs = inputs, scale = None)
        self.assertTrue(torch.equal(input = want, other = got))

    def test_swap_needs_checkpoint_under_grad(self):
        model = build_stack()
        FrozenBlocks(blocks = model.layers, checkpoint = False, swap_count = 2)
        with self.assertRaises(RuntimeError):
            train_steps(model = model, steps = 1)


@unittest.skipUnless(CUDA, "需要 CUDA")
class FP8Test(unittest.TestCase):
    def test_close_to_bf16(self):
        torch.manual_seed(seed = 0)
        linear = nn.Linear(in_features = 256, out_features = 128).to(device = "cuda", dtype = torch.bfloat16).requires_grad_(requires_grad = False)
        inputs = torch.randn(size = (64, 256), device = "cuda", dtype = torch.bfloat16, requires_grad = True)
        want = linear(inputs)
        layer = FP8Linear(linear = linear)
        got = layer(inputs)
        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
        relative = float((got.float() - want.float()).norm() / want.float().norm())
        self.assertLess(relative, 0.05)
        got.sum().backward()
        self.assertIsNotNone(inputs.grad)

    def test_only_frozen_replaced(self):
        model = Stack().to(device = "cuda")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(requires_grad = name.startswith("projector."))
        self.assertEqual(quantize_fp8(module = model), 12)
        self.assertNotIsInstance(model.projector, FP8Linear)


def tiny_model(dtype, device = "cuda"):
    """按阶段 1 的配置缩小的随机初始化模型：语言模型 4 层（3 层线性注意力加 1 层全注意力），音频塔 4 层。"""
    from transformers import AutoConfig, AutoModelForImageTextToText

    config = AutoConfig.from_pretrained(pretrained_model_name_or_path = str(MODEL_DIR), trust_remote_code = True)
    text = config.text_config
    text.num_hidden_layers = 4
    text.layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    text.hidden_size = 128
    text.intermediate_size = 256
    text.num_attention_heads = 2
    text.num_key_value_heads = 1
    # 全注意力层的旋转位置编码按 mrope_section 分段，head_dim × partial_rotary_factor 必须保持原值 64。
    text.head_dim = 256
    text.linear_num_key_heads = 2
    text.linear_num_value_heads = 2
    text.linear_key_head_dim = 64
    text.linear_value_head_dim = 64
    audio = config.audio_config
    audio.encoder_layers = 4
    audio.d_model = 128
    audio.encoder_attention_heads = 2
    audio.encoder_ffn_dim = 256
    audio.output_dim = 128
    audio.downsample_hidden_size = 32
    config.vision_config.depth = 1
    config.vision_config.out_hidden_size = 128
    torch.manual_seed(seed = 0)
    model = AutoModelForImageTextToText.from_config(config = config, trust_remote_code = True, dtype = dtype)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(requires_grad = name.startswith("model.audio_projector."))
    model.model.audio_projector.float()
    return model.eval().to(device = device)


def tiny_batch(audio_windows = None):
    from transformers import AutoProcessor

    from pipeline.train.data import TrainCollator

    processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path = str(MODEL_DIR), trust_remote_code = True)
    rng = np.random.default_rng(seed = 0)
    items = []
    # 长度不同，含一段超过 8 秒、跨两个注意力窗口的音频，以及一个空目标样本。
    for number, seconds in enumerate((3.2, 9.5, 1.7)):
        items.append({
            "sample_id": f"t-{number}",
            "lang": "en",
            "audio_path": f"t-{number}.flac",
            "duration_s": seconds,
            "waveform": (0.1 * rng.standard_normal(size = int(seconds * 16000))).astype(np.float32),
            "target": "" if number == 2 else f"sentence number {number}",
            "empty": number == 2,
            "mix": None,
        })
    return TrainCollator(processor = processor, audio_windows = audio_windows)(items = items)


def without_metadata(batch):
    from pipeline.train.model import METADATA_KEYS

    return {key: value for key, value in batch.items() if key not in METADATA_KEYS + ("audio_max_seqlen",)}


@unittest.skipUnless((MODEL_DIR / "config.json").exists(), "需要阶段 1 模型目录中的代码与配置")
class PrecomputedMetadataTest(unittest.TestCase):
    """读取进程中预先算好的位置编号与音频分块，与模型代码在前向中算出的结果逐位相同（CPU 上即可检查）。"""

    def test_matches_model_code(self):
        model = tiny_model(dtype = torch.float32, device = "cpu")
        config = model.config.audio_config
        batch = tiny_batch(audio_windows = (config.n_window, config.n_window_infer))
        inner = model.model
        positions, _ = inner.get_rope_index(
            input_ids = batch["input_ids"],
            mm_token_type_ids = batch["mm_token_type_ids"],
            attention_mask = batch["attention_mask"],
        )
        self.assertTrue(torch.equal(input = positions, other = batch["position_ids"]))
        with torch.no_grad():
            want = inner.get_audio_features(
                input_features = batch["input_features"],
                feature_attention_mask = batch["feature_attention_mask"],
            )
            audio = inner.audio_tower(
                input_features = None,
                feature_lens = None,
                padded_feature = batch["audio_padded_feature"],
                chunk_lengths = batch["audio_chunk_lengths"],
                valid_indices = batch["audio_valid_indices"],
                cu_seqlens = batch["audio_cu_seqlens"],
                max_seqlen = batch["audio_max_seqlen"],
            )
            got = inner.audio_projector(audio_features = audio.last_hidden_state)
        self.assertTrue(torch.equal(input = want, other = got))
        # 计损失的位置与按标签掩码取出的位置相同。
        shifted = batch["labels"][:, 1:]
        mask = shifted != -100
        self.assertTrue(torch.equal(input = batch["loss_targets"], other = shifted[mask]))
        self.assertTrue(torch.equal(input = batch["loss_rows"], other = mask.nonzero()[:, 0]))
        self.assertEqual(int(batch["audio_mask"].sum()), int(want.shape[0]))


@unittest.skipUnless(CUDA and (MODEL_DIR / "config.json").exists(), "需要 CUDA 和阶段 1 模型目录中的代码与配置")
class TinyModelTest(unittest.TestCase):
    """缩小的完整模型上，各选项的损失和 projector 梯度与原实现一致。"""

    @classmethod
    def setUpClass(cls):
        cls.full_batch = tiny_batch(audio_windows = (50, 800))
        cls.batch = without_metadata(batch = cls.full_batch)

    def losses_and_grad(self, options, dtype):
        from pipeline.train.model import move_batch, projector_parameters, token_losses
        from pipeline.train.optimize import apply

        model = tiny_model(dtype = dtype)
        context = None
        if options is not None:
            context = apply(model = model, options = options)
        batch = self.full_batch if options is not None and options.get("precompute_metadata") else self.batch
        inputs = move_batch(batch = batch, device = "cuda", feature_dtype = model.model.audio_tower.dtype)
        results = []
        for _ in range(2):
            for parameter in projector_parameters(model = model):
                parameter.grad = None
            with (context.forward_context() if context else torch.enable_grad()):
                losses, _, _ = token_losses(model = model, inputs = inputs)
            losses.sum().backward()
            grad = torch.cat(tensors = [parameter.grad.flatten() for parameter in projector_parameters(model = model)])
            results.append((losses.detach().clone(), grad.clone()))
        return results

    def options(self, **changes):
        base = {
            "gradient_checkpointing": False,
            "lm_blocks_to_swap": 0,
            "audio_blocks_to_swap": 0,
            "offload_activations": False,
            "fp8_frozen": "none",
            "compile": "none",
            "audio_attention": "sdpa",
            "precompute_metadata": False,
            "memory_fraction": None,
        }
        base.update(changes)
        return base

    def compare(self, reference, result, atol, label):
        for (want_loss, want_grad), (got_loss, got_grad) in zip(reference, result):
            self.assertTrue(torch.allclose(input = want_loss, other = got_loss, atol = atol, rtol = atol), msg = f"{label} loss {(want_loss - got_loss).abs().max()}")
            relative = float((want_grad - got_grad).norm() / want_grad.norm())
            self.assertLess(relative, atol * 10, msg = f"{label} grad")

    def test_exact_options_float32(self):
        reference = self.losses_and_grad(options = None, dtype = torch.float32)
        cases = {
            "off": self.options(),
            "checkpoint": self.options(gradient_checkpointing = True),
            "swap": self.options(gradient_checkpointing = True, lm_blocks_to_swap = 4, audio_blocks_to_swap = 4),
            "offload": self.options(gradient_checkpointing = True, offload_activations = True, lm_blocks_to_swap = 2),
            "precompute": self.options(precompute_metadata = True),
            "precompute + checkpoint + swap": self.options(
                precompute_metadata = True,
                gradient_checkpointing = True,
                lm_blocks_to_swap = 3,
                audio_blocks_to_swap = 4,
            ),
        }
        for label, options in cases.items():
            self.compare(reference = reference, result = self.losses_and_grad(options = options, dtype = torch.float32), atol = 1e-4, label = label)

    def test_degrade_steps_keep_results(self):
        """显存不足时的逐步退让（检查点层数、流式加载层数逐步增加）不改变损失和梯度，最后返回 None。"""
        from pipeline.train.model import move_batch, projector_parameters, token_losses
        from pipeline.train.optimize import apply

        model = tiny_model(dtype = torch.float32)
        applied = apply(model = model, options = self.options(precompute_metadata = True))
        inputs = move_batch(batch = self.full_batch, device = "cuda", feature_dtype = model.model.audio_tower.dtype)

        def run():
            for parameter in projector_parameters(model = model):
                parameter.grad = None
            with applied.forward_context():
                losses, _, _ = token_losses(model = model, inputs = inputs)
            losses.sum().backward()
            grad = torch.cat(tensors = [parameter.grad.flatten() for parameter in projector_parameters(model = model)])
            return [(losses.detach().clone(), grad.clone())]

        reference = run()
        changes = []
        while (change := applied.degrade()) is not None:
            changes.append(change)
            self.compare(reference = reference, result = run(), atol = 1e-4, label = f"degrade {change}")
        # 语言模型 4 层、音频塔 4 层：检查点 1→4 层、语言模型流式 1→4 层、音频塔流式 1→4 层，共 12 步。
        self.assertEqual(len(changes), 12)
        self.assertEqual(applied.options["checkpoint_layers"], 4)
        self.assertEqual(applied.options["audio_blocks_to_swap"], 4)

    def test_varlen_attention_bfloat16(self):
        reference = self.losses_and_grad(options = None, dtype = torch.bfloat16)
        for precompute in (False, True):
            options = self.options(audio_attention = "varlen", gradient_checkpointing = True, precompute_metadata = precompute)
            result = self.losses_and_grad(options = options, dtype = torch.bfloat16)
            self.compare(reference = reference, result = result, atol = 2e-2, label = f"varlen precompute {precompute}")


if __name__ == "__main__":
    unittest.main()
