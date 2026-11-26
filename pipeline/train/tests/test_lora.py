"""语言模型 LoRA（任务 #71，lora.py）的单元测试：python -m unittest pipeline.train.tests.test_lora

在 CPU 上运行：B 为 0 时输出与原模型逐位相同；合并后的权重与不合并的前向一致；LoRA 参数量等于公式；梯度只到 LoRA 参数，
梯度检查点下与不做检查点时相同；检查点的保存与载入往返一致，合并后的 language_model.safetensors 能被评测程序载入；
续训配置的核对与选项组合的检查。用到缩小的完整模型的几项需要阶段 1 模型目录中的代码与配置。
"""

import argparse
import math
import tempfile
import unittest
from pathlib import Path

import torch

from pipeline.eval.tests.test_replay_evals import reference_kernels
from pipeline.train import lora
from pipeline.train.checkpoint import LANGUAGE_MODEL_FILE, LORA_FILE, save_checkpoint
from pipeline.train.model import LANGUAGE_MODEL_PREFIXES, move_batch, token_losses
from pipeline.train.offload import FrozenBlocks
from pipeline.train.tests.test_offload import MODEL_DIR, tiny_batch, tiny_model
from pipeline.train.train import (
    REQUIRED_PLACEHOLDERS,
    RESUME_KEYS,
    check_language_model_args,
    check_replay_args,
    parse_args,
    resume_mismatches,
    run_config,
)

HAS_MODEL = (MODEL_DIR / "config.json").exists()
# 缩小模型（tiny_model）各解码层的线性层：3 层线性注意力加 1 层全注意力，每层都有 MLP。
LINEAR_ATTENTION = ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b", "linear_attn.in_proj_a", "linear_attn.out_proj")
FULL_ATTENTION = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
MLP = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


class Stack(torch.nn.Module):
    """两层线性层的小网络，layers 与解码层列表的接口相同。"""

    def __init__(self, dtype = torch.float32):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(in_features = 8, out_features = 12), torch.nn.SiLU(), torch.nn.Linear(in_features = 12, out_features = 8, bias = False))
            for _ in range(2)
        ]).to(dtype = dtype)
        for parameter in self.parameters():
            parameter.requires_grad_(requires_grad = False)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = inputs + layer(inputs)
        return inputs


def attach_stack(model, rank = 3, alpha = None, targets = "all", seed = 0):
    return lora.attach(
        model = model,
        rank = rank,
        alpha = rank if alpha is None else alpha,
        targets = targets,
        seed = seed,
        layers = model.layers,
        prefix = "layers",
    )


def randomize_b(model, seed = 1):
    """给 B 赋随机值（训练开始时 B 为 0，增量为 0，测不出合并是否正确）。"""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in lora.named_parameters(model = model):
            if name.endswith(".lora_B"):
                parameter.copy_(0.05 * torch.randn(size = parameter.shape, generator = generator))


class LoRALinearTest(unittest.TestCase):
    def test_zero_b_matches_base(self):
        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(seed = 0)
            model = Stack(dtype = dtype)
            inputs = torch.randn(size = (5, 8)).to(dtype = dtype)
            with torch.no_grad():
                before = model(inputs)
            attach_stack(model = model)
            with torch.no_grad():
                after = model(inputs)
            self.assertTrue(torch.equal(input = before, other = after), dtype)

    def test_merge_matches_unmerged(self):
        torch.manual_seed(seed = 0)
        model = Stack()
        inputs = torch.randn(size = (5, 8))
        with torch.no_grad():
            base = model(inputs)
        attach_stack(model = model, rank = 3, alpha = 6)
        randomize_b(model = model)
        with torch.no_grad():
            unmerged = model(inputs)
        state = lora.merged_state(model = model, prefixes = ("layers.",), dtype = torch.float32)
        merged = Stack()
        merged.load_state_dict(state_dict = state)
        with torch.no_grad():
            got = merged(inputs)
        self.assertGreater(float((unmerged - base).abs().max()), 1e-3)
        torch.testing.assert_close(actual = got, expected = unmerged, rtol = 1e-5, atol = 1e-5)
        # 合并时 ΔW = (α / r) B A。
        module = lora.lora_modules(model = model)["layers.0.0"]
        torch.testing.assert_close(
            actual = state["layers.0.0.weight"],
            expected = module.weight + 2.0 * module.lora_B.detach() @ module.lora_A.detach(),
        )

    def test_parameter_count_and_scaling(self):
        model = Stack()
        named = attach_stack(model = model, rank = 3, alpha = 6)
        # 每层两个线性层：8 → 12 和 12 → 8，各 3 × (8 + 12) 个参数，两层共 4 个。
        self.assertEqual(sum(parameter.numel() for _, parameter in named), 4 * 3 * (8 + 12))
        modules = lora.lora_modules(model = model)
        self.assertEqual(lora.expected_count(modules = modules.values(), rank = 3), 240)
        self.assertEqual({module.lora_scaling for module in modules.values()}, {2.0})
        self.assertEqual([name for name, _ in named][:2], ["layers.0.0.lora_A", "layers.0.0.lora_B"])
        for _, parameter in named:
            self.assertEqual(parameter.dtype, torch.float32)
            self.assertTrue(parameter.requires_grad)

    def test_init(self):
        first, second, other = Stack(), Stack(), Stack()
        attach_stack(model = first, seed = 7)
        attach_stack(model = second, seed = 7)
        attach_stack(model = other, seed = 8)
        a_first = {name: parameter.detach() for name, parameter in lora.named_parameters(model = first)}
        a_second = dict(lora.named_parameters(model = second))
        a_other = dict(lora.named_parameters(model = other))
        name = "layers.0.0.lora_A"
        self.assertTrue(torch.equal(input = a_first[name], other = a_second[name]))
        self.assertFalse(torch.equal(input = a_first[name], other = a_other[name]))
        # kaiming_uniform_(a = √5)：每个元素在 ±1/√输入维度 之间。
        self.assertLessEqual(float(a_first[name].abs().max()), 1 / math.sqrt(8))
        self.assertTrue(torch.equal(input = a_first["layers.0.0.lora_B"], other = torch.zeros(size = (12, 3))))

    def test_grads_only_on_lora(self):
        torch.manual_seed(seed = 0)
        model = Stack()
        attach_stack(model = model)
        model(torch.randn(size = (5, 8))).square().sum().backward()
        for name, parameter in model.named_parameters():
            if name.endswith(lora.LORA_SUFFIXES):
                self.assertIsNotNone(parameter.grad, name)
            else:
                self.assertIsNone(parameter.grad, name)
        grads = {name: parameter.grad for name, parameter in lora.named_parameters(model = model)}
        # B 为 0 时 A 的梯度为 0，B 的梯度不为 0。
        self.assertEqual(float(grads["layers.0.0.lora_A"].abs().max()), 0.0)
        self.assertGreater(float(grads["layers.0.0.lora_B"].abs().max()), 0.0)

    def test_targets(self):
        model = Stack()
        selected = lora.select_targets(layers = model.layers, spec = "2", prefix = "layers")
        self.assertEqual([name for name, _ in selected], ["layers.0.2", "layers.1.2"])
        with self.assertRaises(ValueError):
            lora.select_targets(layers = model.layers, spec = "q_proj", prefix = "layers")
        with self.assertRaises(ValueError):
            lora.parse_targets(spec = "all,q_proj")
        attach_stack(model = model)
        with self.assertRaises(RuntimeError):
            attach_stack(model = model)

    def test_state_round_trip(self):
        torch.manual_seed(seed = 0)
        model = Stack()
        attach_stack(model = model)
        randomize_b(model = model)
        state = lora.state_dict(model = model)
        fresh = Stack()
        fresh.load_state_dict(state_dict = model.state_dict(), strict = False)
        attach_stack(model = fresh, seed = 99)
        lora.load_state(model = fresh, state = state)
        inputs = torch.randn(size = (5, 8))
        with torch.no_grad():
            self.assertTrue(torch.equal(input = fresh(inputs), other = model(inputs)))
        with self.assertRaises(ValueError):
            lora.load_state(model = fresh, state = {name: tensor for name, tensor in state.items() if "layers.1." not in name})
        wrong = dict(state)
        wrong["layers.0.0.lora_A"] = torch.zeros(size = (4, 8))
        with self.assertRaises(ValueError):
            lora.load_state(model = fresh, state = wrong)

    def test_check_state(self):
        model = Stack()
        attach_stack(model = model)
        expected = lora.base_language_model_shapes(model = model, prefixes = ("layers.",))
        state = lora.merged_state(model = model, prefixes = ("layers.",))
        self.assertNotIn("layers.0.0.lora_A", state)
        lora.check_state(state = state, expected = expected)
        with self.assertRaises(ValueError):
            lora.check_state(state = {key: value for key, value in state.items() if key != "layers.0.0.weight"}, expected = expected)
        with self.assertRaises(ValueError):
            lora.check_state(state = {**state, "layers.0.0.weight": state["layers.0.0.weight"].float()}, expected = expected)
        with self.assertRaises(ValueError):
            lora.check_state(state = {**state, "layers.0.0.weight": state["layers.0.0.weight"][:, :4]}, expected = expected)

    def test_rank_for_share(self):
        model = Stack()
        modules = [module for _, module in lora.select_targets(layers = model.layers, spec = "all", prefix = "layers")]
        # 每个秩单位 4 × 20 = 80 个参数；总数 1,000 的 40% 是 400，对应秩 5。
        self.assertEqual(lora.rank_for_share(modules = modules, total = 1000, share = 0.4), (5, 80))


def lora_args(extra = ()):
    return parse_args(argv = REQUIRED_PLACEHOLDERS + list(extra))


class ArgumentsTest(unittest.TestCase):
    def test_defaults_and_resume(self):
        self.assertTrue(set(lora.KEYS) <= set(RESUME_KEYS))
        args = lora_args(extra = ["--language-model-lora-rank", "32"])
        check_language_model_args(args = args)
        self.assertEqual(args.language_model_lora_alpha, 32.0)
        self.assertEqual(args.language_model_lora_lr, lora.DEFAULT_LR)
        config = run_config(args = args)
        # 以前的训练配置中没有 LoRA 的几项，按默认值（不用 LoRA）核对：不用 LoRA 的训练照常续训，LoRA 的训练不能接着它。
        old = {key: value for key, value in run_config(args = lora_args()).items() if key not in lora.KEYS}
        self.assertEqual(resume_mismatches(saved = old, config = run_config(args = lora_args())), [])
        self.assertEqual(resume_mismatches(saved = old, config = config), ["language_model_lora_rank", "language_model_lora_alpha"])
        self.assertEqual(resume_mismatches(saved = config, config = config), [])
        changed = run_config(args = lora_args(extra = ["--language-model-lora-rank", "32", "--language-model-lora-alpha", "64"]))
        self.assertEqual(resume_mismatches(saved = config, config = changed), ["language_model_lora_alpha"])

    def test_invalid_combinations(self):
        for extra in (
            ["--language-model-lora-rank", "8", "--train-language-model", "--language-model-lr", "1e-5"],
            ["--language-model-lora-alpha", "8"],
            ["--language-model-lora-lr", "1e-4"],
            ["--language-model-lora-rank", "8", "--optimizer", "muon"],
            ["--language-model-lora-rank", "8", "--fp8-frozen", "lm"],
            ["--language-model-lora-rank", "8", "--language-model-lora-targets", "nothing,all"],
            ["--language-model-lora-rank", "-1"],
        ):
            with self.assertRaises(SystemExit, msg = extra):
                check_language_model_args(args = lora_args(extra = extra))

    def test_replay_allowed_with_lora(self):
        replay = ["--replay-dir", "x", "--replay-groups", "cont_raw=1", "--replay-ratio", "0.5", "--pack-sequences"]
        args = lora_args(extra = ["--language-model-lora-rank", "8"] + replay)
        check_language_model_args(args = args)
        check_replay_args(args = args)
        with self.assertRaises(SystemExit):
            check_replay_args(args = lora_args(extra = replay))


def tiny_lora_model(rank = 4, seed = 0):
    """缩小的完整模型（float32，CPU）加 LoRA；返回 (模型, LoRA 参数)。"""
    model = tiny_model(dtype = torch.float32, device = "cpu")
    named = lora.attach(
        model = model,
        rank = rank,
        alpha = rank,
        seed = seed,
    )
    return model, named


def batch_inputs(model):
    config = model.config.audio_config
    batch = tiny_batch(audio_windows = (config.n_window, config.n_window_infer), prompt_format = "cpt")
    return batch, move_batch(batch = batch, device = "cpu", feature_dtype = torch.float32)


@unittest.skipUnless(HAS_MODEL, "需要阶段 1 模型目录中的代码与配置")
class TinyModelTest(unittest.TestCase):
    """缩小的完整模型（Qwen3.5 的线性注意力层和全注意力层都有）上的 LoRA。"""

    @classmethod
    def setUpClass(cls):
        # 线性注意力的短卷积与递推换回 PyTorch 参考实现，CPU 上可以前向和反传。
        cls.enterClassContext(cm = reference_kernels())
        cls.base = tiny_model(dtype = torch.float32, device = "cpu")
        cls.batch, cls.inputs = batch_inputs(model = cls.base)
        with torch.no_grad():
            cls.base_losses, _, _ = token_losses(model = cls.base, inputs = cls.inputs)

    def test_targets_in_hybrid_layers(self):
        model, named = tiny_lora_model()
        modules = lora.lora_modules(model = model)
        layer_types = model.config.text_config.layer_types
        expected = []
        for index, kind in enumerate(layer_types):
            inner = LINEAR_ATTENTION if kind == "linear_attention" else FULL_ATTENTION
            expected += [f"model.language_model.layers.{index}.{name}" for name in inner + MLP]
        self.assertEqual(sorted(modules), sorted(expected))
        count = sum(parameter.numel() for _, parameter in named)
        self.assertEqual(count, sum(4 * (module.in_features + module.out_features) for module in modules.values()))
        # 词嵌入兼输出层、归一化层、短卷积等不加 LoRA，也不训练。
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.assertEqual(trainable, {name for name, _ in named} | {name for name, _ in model.named_parameters() if name.startswith("model.audio_projector.")})

    def test_zero_b_matches_base(self):
        model, _ = tiny_lora_model()
        with torch.no_grad():
            losses, _, _ = token_losses(model = model, inputs = self.inputs)
        self.assertTrue(torch.equal(input = losses, other = self.base_losses))

    def test_merged_language_model_loads_in_eval(self):
        from pipeline.eval.run_eval import load_language_model

        model, _ = tiny_lora_model()
        randomize_b(model = model)
        with torch.no_grad():
            losses, _, _ = token_losses(model = model, inputs = self.inputs)
        self.assertGreater(float((losses - self.base_losses).abs().max()), 1e-3)
        expected = lora.base_language_model_shapes(model = model, prefixes = LANGUAGE_MODEL_PREFIXES)
        bf16 = lora.merged_state(model = model, prefixes = LANGUAGE_MODEL_PREFIXES)
        lora.check_state(state = bf16, expected = expected)
        self.assertEqual(set(bf16), {name for name, _ in self.base.named_parameters() if name.startswith(LANGUAGE_MODEL_PREFIXES)})
        with tempfile.TemporaryDirectory() as directory:
            from safetensors.torch import save_file

            path = Path(directory) / LANGUAGE_MODEL_FILE
            save_file(tensors = lora.merged_state(model = model, prefixes = LANGUAGE_MODEL_PREFIXES, dtype = torch.float32), filename = str(path))
            fresh = tiny_model(dtype = torch.float32, device = "cpu")
            # 评测程序载入的是不带 LoRA 的模型：键必须恰好是语言模型的全部参数（与输出层共用的词嵌入一份）。
            load_language_model(model = fresh, language_model_path = path)
        with torch.no_grad():
            got, _, _ = token_losses(model = fresh, inputs = self.inputs)
        torch.testing.assert_close(actual = got, expected = losses, rtol = 1e-4, atol = 1e-4)

    def test_gradient_checkpointing_and_text_only_batch(self):
        """梯度检查点下 LoRA 参数的梯度与不做检查点时相同；纯文本输入（词嵌入冻结）时第 0 层也做检查点。"""
        from unittest import mock

        from pipeline.train import offload

        grads = []
        for checkpoint in (False, True):
            model, named = tiny_lora_model()
            randomize_b(model = model)
            handle = lora.require_input_grads(model = model)
            blocks = FrozenBlocks(blocks = model.model.language_model.layers, checkpoint = checkpoint, device = "cpu")
            with mock.patch.object(target = offload._CheckpointBlock, attribute = "apply", wraps = offload._CheckpointBlock.apply) as spy:
                losses, _, _ = token_losses(model = model, inputs = self.inputs)
                losses.sum().backward()
                # 回放的纯文本微批：只有文字嵌入，词嵌入冻结。
                outputs = model.model.language_model(input_ids = self.inputs["input_ids"][:, :12], use_cache = False)
                outputs.last_hidden_state.square().mean().backward()
            layers = len(model.model.language_model.layers)
            self.assertEqual(spy.call_count, 2 * layers if checkpoint else 0)
            grads.append({name: parameter.grad.clone() for name, parameter in named})
            blocks.remove()
            handle.remove()
        for name in grads[0]:
            torch.testing.assert_close(actual = grads[1][name], expected = grads[0][name], rtol = 1e-5, atol = 1e-6, msg = name)
            self.assertGreater(float(grads[0][name].abs().max()), 0.0, name)

    def test_checkpoint_round_trip(self):
        model, _ = tiny_lora_model()
        randomize_b(model = model)
        with torch.no_grad():
            losses, _, _ = token_losses(model = model, inputs = self.inputs)
        with tempfile.TemporaryDirectory() as directory:
            for step in (1, 2, 3):
                saved = save_checkpoint(
                    output_dir = directory,
                    step = step,
                    model = model,
                    info = {},
                    trainer_state = {"step": step},
                    language_model = lora.merged_state(model = model, prefixes = LANGUAGE_MODEL_PREFIXES),
                    lora = lora.state_dict(model = model),
                    keep_language_models = 2,
                )
            kept = sorted(path.parent.name for path in Path(directory).glob(pattern = f"checkpoint-*/{LANGUAGE_MODEL_FILE}"))
            self.assertEqual(kept, ["checkpoint-0000002", "checkpoint-0000003"])
            self.assertEqual(len(list(Path(directory).glob(pattern = f"checkpoint-*/{LORA_FILE}"))), 3)
            fresh, _ = tiny_lora_model(seed = 5)
            lora.load_state(model = fresh, state = lora.read_state(path = saved))
        with torch.no_grad():
            got, _, _ = token_losses(model = fresh, inputs = self.inputs)
        self.assertTrue(torch.equal(input = got, other = losses))


if __name__ == "__main__":
    unittest.main()
