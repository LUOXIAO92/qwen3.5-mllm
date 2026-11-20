"""全量微调精度选项（任务 #66，precision.py、master.py、torchao_adamw.py）的单元测试：

    python -m unittest pipeline.train.tests.test_precision

选项的组合与续训核对、主权重与直接更新两种方式的梯度精度在 CPU 上运行；torchao 的 AdamW（状态格式、续训时保持状态
精度、随机舍入）需要 torchao，没有时跳过，有 GPU 时在 GPU 上运行，否则在 CPU 上（torch.compile 需要 C++ 编译器）。
"""

import importlib.util
import io
import unittest

import torch

from pipeline.train import precision
from pipeline.train.master import DirectWeights, MasterWeights
from pipeline.train.train import RESUME_KEYS, check_language_model_args, parse_args, resume_mismatches, run_config

HAS_TORCHAO = importlib.util.find_spec(name = "torchao") is not None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ARGV = [
    "--model-dir", "/repo/qwen3.5-4b-mllm",
    "--train-manifest", "/data/train/manifest-v2b.jsonl",
    "--target", "original",
    "--output-dir", "/work/data/T66/full",
]
FULL = ["--train-language-model", "--language-model-lr", "1e-5"]


def options(master = "fp32", grad = "fp32", rounding = "nearest", states = "fp32"):
    return {
        "language_model_master_weights": master,
        "language_model_grad_dtype": grad,
        "language_model_rounding": rounding,
        "language_model_optimizer_states": states,
    }


class OptionsTest(unittest.TestCase):
    def test_defaults_are_the_previous_recipe(self):
        args = parse_args(argv = ARGV + FULL)
        self.assertEqual(precision.options_from_args(args = args), options())
        self.assertTrue(precision.uses_torch_adamw(options = options()))
        self.assertEqual(precision.bytes_per_parameter(options = options())["total"], 18.0)

    def test_old_configs_resume_with_the_defaults(self):
        saved = run_config(args = parse_args(argv = ARGV + FULL))
        for key in precision.KEYS:
            self.assertIn(key, RESUME_KEYS)
            saved.pop(key)
        self.assertEqual(resume_mismatches(saved = saved, config = run_config(args = parse_args(argv = ARGV + FULL))), [])
        changed = run_config(args = parse_args(argv = ARGV + FULL + [
            "--language-model-master-weights", "none",
            "--language-model-grad-dtype", "bf16",
            "--language-model-optimizer-states", "8bit",
        ]))
        self.assertEqual(
            resume_mismatches(saved = saved, config = changed),
            ["language_model_master_weights", "language_model_grad_dtype", "language_model_optimizer_states"],
        )

    def test_invalid_combinations(self):
        with self.assertRaises(SystemExit):
            check_language_model_args(args = parse_args(argv = ARGV + FULL + ["--language-model-rounding", "stochastic"]))
        with self.assertRaises(SystemExit):
            check_language_model_args(args = parse_args(argv = ARGV + ["--language-model-optimizer-states", "bf16"]))
        check_language_model_args(args = parse_args(argv = ARGV + FULL + [
            "--language-model-master-weights", "none",
            "--language-model-grad-dtype", "bf16",
            "--language-model-optimizer-states", "bf16",
        ]))

    def test_implementation_and_memory(self):
        self.assertTrue(precision.uses_torch_adamw(options = options(master = "none", grad = "bf16", states = "bf16")))
        for case in (
            options(grad = "bf16"),
            options(states = "8bit"),
            options(master = "none", grad = "bf16"),
            options(master = "none", grad = "bf16", rounding = "stochastic", states = "bf16"),
        ):
            self.assertFalse(precision.uses_torch_adamw(options = case), msg = case)
        cases = {
            (("fp32", "bf16", "nearest", "fp32")): 16.0,
            (("none", "bf16", "nearest", "fp32")): 12.0,
            (("none", "bf16", "stochastic", "bf16")): 8.0,
            (("none", "fp32", "stochastic", "bf16")): 10.0,
        }
        for (master, grad, rounding, states), total in cases.items():
            parts = precision.bytes_per_parameter(options = options(master = master, grad = grad, rounding = rounding, states = states))
            self.assertEqual(parts["total"], total)
        eight = precision.bytes_per_parameter(options = options(master = "none", grad = "bf16", states = "8bit"))
        self.assertAlmostEqual(eight["total"], 6.03125)


def bf16_layer():
    torch.manual_seed(seed = 0)
    layer = torch.nn.Linear(in_features = 8, out_features = 4).to(dtype = torch.bfloat16)
    return layer, list(layer.named_parameters())


def backward_twice(layer):
    inputs = torch.randn(size = (3, 8), dtype = torch.bfloat16, generator = torch.Generator().manual_seed(1))
    for _ in range(2):
        layer(inputs).float().sum().backward()


class GradDtypeTest(unittest.TestCase):
    def reference_grads(self):
        layer, named = bf16_layer()
        backward_twice(layer = layer)
        return {name: parameter.grad.clone() for name, parameter in named}

    def test_master_with_bf16_grads(self):
        reference = self.reference_grads()
        layer, named = bf16_layer()
        weights = MasterWeights(named_parameters = named, grad_dtype = torch.bfloat16)
        backward_twice(layer = layer)
        for name, master in zip(weights.names, weights.masters):
            self.assertEqual(master.dtype, torch.float32)
            self.assertEqual(master.grad.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(input = master.grad, other = reference[name]), msg = name)
        for _, parameter in named:
            self.assertIsNone(parameter.grad)

    def test_direct_weights(self):
        reference = self.reference_grads()
        for dtype in (torch.float32, torch.bfloat16):
            layer, named = bf16_layer()
            weights = DirectWeights(named_parameters = named, grad_dtype = dtype)
            backward_twice(layer = layer)
            for name, parameter in named:
                self.assertEqual(parameter.dtype, torch.bfloat16)
                self.assertEqual(parameter.grad.dtype, dtype)
                # 两次反传：float32 中累加与 bfloat16 中累加相差不超过 bfloat16 的舍入。
                torch.testing.assert_close(actual = parameter.grad.float(), expected = reference[name].float(), rtol = 1e-2, atol = 1e-2)
            self.assertEqual(weights.without_grad(), [])
            weights.remove()

    def test_direct_weights_resume(self):
        layer, named = bf16_layer()
        weights = DirectWeights(named_parameters = named)
        with torch.no_grad():
            for _, parameter in named:
                parameter.add_(0.25)
        saved = {name: tensor.clone() for name, tensor in weights.state_dict().items()}
        _, named2 = bf16_layer()
        DirectWeights(named_parameters = named2, state = saved)
        for name, parameter in named2:
            self.assertTrue(torch.equal(input = parameter.detach(), other = saved[name]), msg = name)
        with self.assertRaises(ValueError):
            DirectWeights(named_parameters = named2, state = {name: tensor.float() for name, tensor in saved.items()})


@unittest.skipUnless(HAS_TORCHAO, "需要 torchao")
class TorchaoAdamWTest(unittest.TestCase):
    def build(self, states, stochastic, master_dtype = torch.bfloat16):
        """projector 组一个 float32 参数，语言模型组一个大参数（可量化）和一个小参数（不量化）。"""
        generator = torch.Generator().manual_seed(0)
        projector = torch.nn.Parameter(torch.randn(size = (64, 64), generator = generator).to(device = DEVICE))
        big = torch.nn.Parameter(torch.randn(size = (64, 64), generator = generator).to(device = DEVICE, dtype = master_dtype))
        small = torch.nn.Parameter(torch.randn(size = (16,), generator = generator).to(device = DEVICE, dtype = master_dtype))
        optimizer = precision.make_optimizer(
            projector_params = [projector],
            language_model_params = [big, small],
            lr = 1e-3,
            language_model_lr = 1e-2,
            weight_decay = 0.0,
            options = options(
                master = "fp32" if master_dtype == torch.float32 else "none",
                grad = "bf16",
                rounding = "stochastic" if stochastic else "nearest",
                states = states,
            ),
        )
        return optimizer, [projector, big, small]

    def step(self, optimizer, params, seed):
        generator = torch.Generator().manual_seed(seed)
        for parameter in params:
            grad = torch.randn(size = parameter.shape, generator = generator).to(device = DEVICE)
            parameter.grad = grad.to(dtype = parameter.dtype)
        optimizer.step()
        optimizer.zero_grad(set_to_none = True)

    def test_state_formats(self):
        from torchao.optim.subclass_8bit import OptimState8bit

        expected = {
            "fp32": (torch.float32, torch.float32),
            "bf16": (torch.bfloat16, torch.bfloat16),
        }
        for states, (big_dtype, small_dtype) in expected.items():
            optimizer, (projector, big, small) = self.build(states = states, stochastic = True)
            self.step(optimizer = optimizer, params = [projector, big, small], seed = 1)
            self.assertEqual(optimizer.state[projector]["exp_avg"].dtype, torch.float32)
            self.assertIs(type(optimizer.state[big]["exp_avg"]), torch.Tensor)
            self.assertEqual(optimizer.state[big]["exp_avg_sq"].dtype, big_dtype)
            self.assertEqual(optimizer.state[small]["exp_avg"].dtype, small_dtype)
        optimizer, (projector, big, small) = self.build(states = "8bit", stochastic = True)
        self.step(optimizer = optimizer, params = [projector, big, small], seed = 1)
        self.assertEqual(optimizer.state[projector]["exp_avg"].dtype, torch.float32)
        self.assertIsInstance(optimizer.state[big]["exp_avg"], OptimState8bit)
        # 少于 4096 个元素的张量与 torchao 一样按参数的精度存。
        self.assertIs(type(optimizer.state[small]["exp_avg"]), torch.Tensor)
        self.assertEqual(optimizer.state[small]["exp_avg"].dtype, torch.bfloat16)

    def save_and_reload(self, optimizer):
        buffer = io.BytesIO()
        torch.save(obj = optimizer.state_dict(), f = buffer)
        buffer.seek(0)
        return torch.load(f = buffer, weights_only = False)

    def rng_state(self):
        return torch.cuda.get_rng_state() if DEVICE == "cuda" else torch.get_rng_state()

    def set_rng_state(self, state):
        if DEVICE == "cuda":
            torch.cuda.set_rng_state(new_state = state)
        else:
            torch.set_rng_state(new_state = state)

    def test_resume_is_exact(self):
        """连续四步与「两步后保存、新建优化器载入、再两步」逐位相同，包括随机舍入和 float32 状态。"""
        for states, master_dtype in (("fp32", torch.bfloat16), ("bf16", torch.bfloat16), ("8bit", torch.bfloat16), ("bf16", torch.float32)):
            torch.manual_seed(seed = 7)
            if DEVICE == "cuda":
                torch.cuda.manual_seed(seed = 7)
            optimizer, params = self.build(states = states, stochastic = master_dtype == torch.bfloat16, master_dtype = master_dtype)
            for seed in range(4):
                self.step(optimizer = optimizer, params = params, seed = seed)
            straight = [parameter.detach().clone() for parameter in params]

            torch.manual_seed(seed = 7)
            if DEVICE == "cuda":
                torch.cuda.manual_seed(seed = 7)
            optimizer, params = self.build(states = states, stochastic = master_dtype == torch.bfloat16, master_dtype = master_dtype)
            for seed in range(2):
                self.step(optimizer = optimizer, params = params, seed = seed)
            saved = self.save_and_reload(optimizer = optimizer)
            weights = [parameter.detach().clone() for parameter in params]
            rng = self.rng_state()
            dtypes = {index: {key: value.dtype for key, value in state.items()} for index, state in saved["state"].items()}

            resumed, params2 = self.build(states = states, stochastic = master_dtype == torch.bfloat16, master_dtype = master_dtype)
            with torch.no_grad():
                for parameter, value in zip(params2, weights):
                    parameter.copy_(value)
            resumed.load_state_dict(state_dict = saved)
            self.set_rng_state(state = rng)
            for index, parameter in enumerate(params2):
                for key, value in resumed.state[parameter].items():
                    self.assertEqual(value.dtype, dtypes[index][key], msg = (states, index, key))
            for seed in range(2, 4):
                self.step(optimizer = resumed, params = params2, seed = seed)
            for index, (expected, parameter) in enumerate(zip(straight, params2)):
                self.assertTrue(torch.equal(input = parameter.detach(), other = expected), msg = (states, index))

    def test_stochastic_rounding_keeps_small_updates(self):
        """权重为 1 的 bfloat16 参数，每步的更新约为学习率 1e-3，小于下方相邻两个 bfloat16 数的间距（2^-8）的一半：
        就近舍入时权重不动，随机舍入时按比例下移，平均下移量接近 1e-3。"""
        changes = {}
        for stochastic in (False, True):
            weight = torch.nn.Parameter(torch.ones(size = (128, 128), device = DEVICE, dtype = torch.bfloat16))
            optimizer = precision.make_optimizer(
                projector_params = [],
                language_model_params = [weight],
                lr = 1e-3,
                language_model_lr = 1e-3,
                weight_decay = 0.0,
                options = options(master = "none", grad = "bf16", rounding = "stochastic" if stochastic else "nearest", states = "fp32"),
            )
            weight.grad = torch.ones_like(input = weight)
            optimizer.step()
            changes[stochastic] = float((weight.detach().float() - 1).mean())
        self.assertEqual(changes[False], 0.0)
        self.assertAlmostEqual(changes[True], -1e-3, delta = 2e-4)


if __name__ == "__main__":
    unittest.main()
