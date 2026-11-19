"""全量微调（任务 #66）的单元测试：python -m unittest pipeline.train.tests.test_full_finetune

在 CPU 上运行：分块交叉熵对隐状态和输出层权重的梯度与直接计算一致；主权重收到 float32 梯度、更新后写回模型参数、
续训时恢复；以前的训练配置没有 train_language_model 时照常续训。
"""

import unittest

import torch
import torch.nn.functional as F

from pipeline.train.master import MasterWeights
from pipeline.train.model import LinearCrossEntropy
from pipeline.train.train import RESUME_KEYS, parse_args, resume_mismatches, run_config


def reference_grads(hidden, weight, targets, scale):
    """先投影再 F.cross_entropy，按 scale 加权求和后反传。"""
    hidden = hidden.detach().clone().requires_grad_(requires_grad = True)
    weight = weight.detach().clone().requires_grad_(requires_grad = True)
    losses = F.cross_entropy(input = hidden @ weight.t(), target = targets, reduction = "none")
    (losses * scale).sum().backward()
    return losses.detach(), hidden.grad, weight.grad


class LinearCrossEntropyTest(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(0)
        self.hidden = torch.randn(size = (50, 16), generator = generator, dtype = torch.float64)
        self.weight = torch.randn(size = (37, 16), generator = generator, dtype = torch.float64)
        self.targets = torch.randint(low = 0, high = 37, size = (50,), generator = generator)
        self.scale = torch.rand(size = (50,), generator = generator, dtype = torch.float64)

    def run_chunked(self, weight_grad):
        hidden = self.hidden.clone().requires_grad_(requires_grad = True)
        weight = self.weight.clone().requires_grad_(requires_grad = weight_grad)
        losses, correct = LinearCrossEntropy.apply(hidden, weight, self.targets, 8)
        (losses * self.scale).sum().backward()
        return losses.detach(), hidden.grad, weight.grad

    def test_frozen_weight(self):
        losses, grad_hidden, grad_weight = self.run_chunked(weight_grad = False)
        expected_losses, expected_hidden, _ = reference_grads(
            hidden = self.hidden,
            weight = self.weight,
            targets = self.targets,
            scale = self.scale,
        )
        self.assertIsNone(grad_weight)
        # 分数按 float32 计算，与 float64 的参考值相差在 float32 舍入以内。
        torch.testing.assert_close(actual = losses.double(), expected = expected_losses, rtol = 1e-5, atol = 1e-5)
        torch.testing.assert_close(actual = grad_hidden, expected = expected_hidden, rtol = 1e-5, atol = 1e-5)

    def test_trainable_weight(self):
        losses, grad_hidden, grad_weight = self.run_chunked(weight_grad = True)
        expected_losses, expected_hidden, expected_weight = reference_grads(
            hidden = self.hidden,
            weight = self.weight,
            targets = self.targets,
            scale = self.scale,
        )
        torch.testing.assert_close(actual = losses.double(), expected = expected_losses, rtol = 1e-5, atol = 1e-5)
        torch.testing.assert_close(actual = grad_hidden, expected = expected_hidden, rtol = 1e-5, atol = 1e-5)
        torch.testing.assert_close(actual = grad_weight, expected = expected_weight, rtol = 1e-5, atol = 1e-5)


class MasterWeightsTest(unittest.TestCase):
    def build(self):
        torch.manual_seed(seed = 0)
        layer = torch.nn.Linear(in_features = 8, out_features = 4).to(dtype = torch.bfloat16)
        return layer, [(name, parameter) for name, parameter in layer.named_parameters()]

    def test_grad_moves_to_master_and_update_is_written_back(self):
        layer, named = self.build()
        masters = MasterWeights(named_parameters = named)
        optimizer = torch.optim.SGD(params = masters.optimizer_params(), lr = 0.5)
        inputs = torch.randn(size = (3, 8), dtype = torch.bfloat16)
        for _ in range(2):
            layer(inputs).float().sum().backward()
        for (name, parameter), master in zip(named, masters.masters):
            self.assertIsNone(parameter.grad, msg = name)
            self.assertEqual(master.grad.dtype, torch.float32)
        self.assertEqual(masters.without_grad(), [])
        expected = [(master - 0.5 * master.grad).detach().clone() for master in masters.masters]
        optimizer.step()
        masters.copy_to_model()
        for (name, parameter), master, value in zip(named, masters.masters, expected):
            torch.testing.assert_close(actual = master.detach(), expected = value)
            self.assertTrue(torch.equal(input = parameter.detach(), other = value.to(dtype = torch.bfloat16)), msg = name)

    def test_resume_restores_masters_and_model(self):
        layer, named = self.build()
        masters = MasterWeights(named_parameters = named)
        with torch.no_grad():
            for master in masters.masters:
                master.add_(1e-3)
        masters.copy_to_model()
        saved = {name: tensor.clone() for name, tensor in masters.state_dict().items()}
        state = masters.model_state()
        masters.remove()
        layer2, named2 = self.build()
        restored = MasterWeights(named_parameters = named2, state = saved)
        for name, master in zip(restored.names, restored.masters):
            self.assertTrue(torch.equal(input = master.detach(), other = saved[name]), msg = name)
        for name, parameter in named2:
            self.assertTrue(torch.equal(input = parameter.detach(), other = state[name]), msg = name)

    def test_resume_rejects_other_keys(self):
        _, named = self.build()
        with self.assertRaises(ValueError):
            MasterWeights(named_parameters = named, state = {"weight": torch.zeros(size = (4, 8))})


class ConfigTest(unittest.TestCase):
    ARGV = [
        "--model-dir", "/repo/qwen3.5-2b-mllm",
        "--train-manifest", "/data/train/manifest-v2b.jsonl",
        "--target", "original",
        "--output-dir", "/work/data/T61/v2b-2b",
    ]

    def test_old_configs_resume_without_the_new_options(self):
        saved = run_config(args = parse_args(argv = self.ARGV))
        for key in ("train_language_model", "language_model_lr", "init_projector"):
            saved.pop(key)
        self.assertEqual(resume_mismatches(saved = saved, config = run_config(args = parse_args(argv = self.ARGV))), [])
        full = run_config(args = parse_args(argv = self.ARGV + ["--train-language-model", "--language-model-lr", "1e-5"]))
        self.assertEqual(resume_mismatches(saved = saved, config = full), ["train_language_model"])

    def test_learning_rates_and_init_are_not_resume_keys(self):
        for key in ("lr", "language_model_lr", "init_projector"):
            self.assertNotIn(key, RESUME_KEYS)
        saved = run_config(args = parse_args(argv = self.ARGV + [
            "--train-language-model",
            "--language-model-lr", "1e-5",
            "--init-projector", "/work/data/T61/v2b-2b/checkpoint-0005000",
        ]))
        resumed = run_config(args = parse_args(argv = self.ARGV + ["--train-language-model", "--language-model-lr", "2e-5"]))
        self.assertEqual(resume_mismatches(saved = saved, config = resumed), [])


if __name__ == "__main__":
    unittest.main()
