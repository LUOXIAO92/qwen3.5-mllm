"""语言模型参数在主机内存中更新（任务 #66，placement.py）的单元测试：python -m unittest pipeline.train.tests.test_placement

HostAdamW 与显存中的 AdamW 的一致性、续训逐位相同、随机舍入，以及 LanguageModelUpdate 在缩小的模型上三种去处（显存、
流式加载的层、卸载优化器）的损失、梯度范数、更新后的权重和规范格式的续训状态。需要 CUDA 和 torchao，没有时跳过。
"""

import importlib.util
import io
import unittest

import torch

from pipeline.train import precision
from pipeline.train.tests.test_offload import MODEL_DIR, TinyModelTest, tiny_batch, tiny_model

CUDA = torch.cuda.is_available()
HAS_TORCHAO = importlib.util.find_spec(name = "torchao") is not None


def precision_options(master = "fp32", grad = "fp32", rounding = "nearest", states = "fp32"):
    return {
        "language_model_master_weights": master,
        "language_model_grad_dtype": grad,
        "language_model_rounding": rounding,
        "language_model_optimizer_states": states,
    }


class HostProblemTest(unittest.TestCase):
    def test_supported_combinations(self):
        self.assertIsNone(precision.host_problem(options = precision_options(), grad_accum = 1))
        self.assertIsNone(precision.host_problem(options = precision_options(grad = "bf16"), grad_accum = 1))
        stochastic = precision_options(master = "none", grad = "bf16", rounding = "stochastic")
        self.assertIsNone(precision.host_problem(options = stochastic, grad_accum = 1))
        for states in ("bf16", "8bit", "4bit", "fp8"):
            self.assertIsNotNone(precision.host_problem(options = precision_options(states = states), grad_accum = 1))
        self.assertIsNotNone(precision.host_problem(options = precision_options(), grad_accum = 2))


def random_case(seed, shape = (300, 70)):
    """一个 bfloat16 权重（主机，锁页）和三步的 bfloat16 梯度。"""
    generator = torch.Generator().manual_seed(seed)
    weight = (0.05 * torch.randn(size = shape, generator = generator)).to(dtype = torch.bfloat16)
    grads = [(0.01 * torch.randn(size = shape, generator = generator)).to(dtype = torch.bfloat16) for _ in range(3)]
    return weight, grads


@unittest.skipUnless(CUDA and HAS_TORCHAO, "需要 CUDA 和 torchao")
class HostAdamWTest(unittest.TestCase):
    def host(self, weight, master, stochastic = False, scale = 1.0):
        from pipeline.train.placement import HostAdamW, HostContext

        context = HostContext(seed = 3)
        context.scale.fill_(scale)
        host = weight.clone().pin_memory()
        host.grad = torch.empty_like(input = host).pin_memory()
        optimizer = HostAdamW(
            params = [host],
            lr = 1e-3,
            betas = (0.9, 0.999),
            eps = 1e-8,
            weight_decay = 0.01,
            master = master,
            stochastic = stochastic,
            context = context,
        )
        return host, optimizer

    def test_master_matches_torch_fused_adamw(self):
        """有主权重：与显存中 torch 的 fused AdamW 更新 float32 主权重（梯度按裁剪系数缩放）在 float32 舍入以内一致，
        bfloat16 权重是主权重的就近舍入。"""
        weight, grads = random_case(seed = 0)
        host, optimizer = self.host(weight = weight, master = True, scale = 0.5)
        master = torch.nn.Parameter(weight.float().cuda())
        reference = torch.optim.AdamW(params = [master], lr = 1e-3, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 0.01, fused = True)
        for grad in grads:
            host.grad.copy_(grad)
            optimizer.step()
            master.grad = grad.float().cuda() * 0.5
            reference.step()
        got = optimizer.masters[host]
        torch.testing.assert_close(actual = got, expected = master.detach().cpu(), rtol = 1e-6, atol = 1e-8)
        self.assertTrue(torch.equal(input = host, other = got.to(dtype = torch.bfloat16)))
        state = optimizer.state[host]
        # 二阶矩的系数 1 - β2 两边按不同精度算出（float32 与 float64 中的 1 - 0.999 相对差约 5e-5）。
        torch.testing.assert_close(actual = state["exp_avg_sq"], expected = reference.state[master]["exp_avg_sq"].cpu(), rtol = 1e-4, atol = 1e-12)
        self.assertEqual(float(state["step"]), 3.0)

    def test_direct_matches_torchao(self):
        """没有主权重、就近舍入：与显存中 torchao 实现（precision.make_optimizer 的同一组合）几乎处处相同，最多差 1 个
        bfloat16 舍入间距。"""
        weight, grads = random_case(seed = 1)
        host, optimizer = self.host(weight = weight, master = False)
        device = torch.nn.Parameter(weight.cuda())
        reference = precision.make_optimizer(
            projector_params = [],
            language_model_params = [device],
            lr = 1e-3,
            language_model_lr = 1e-3,
            weight_decay = 0.01,
            options = precision_options(master = "none", grad = "bf16", states = "fp32"),
        )
        for grad in grads:
            host.grad.copy_(grad)
            optimizer.step()
            device.grad = grad.cuda()
            reference.step()
        got, want = host.float(), device.detach().cpu().float()
        differ = float((got != want).float().mean())
        self.assertLess(differ, 1e-3)
        spacing = torch.finfo(torch.bfloat16).eps * want.abs().clamp(min = 1e-3)
        self.assertTrue(bool(((got - want).abs() <= spacing * 1.01).all()))

    def test_stochastic_rounding_keeps_small_updates(self):
        """权重为 1、每步更新约 1e-3（小于 bfloat16 间距的一半）：就近舍入时不动，随机舍入时平均下移约 1e-3。"""
        changes = {}
        for stochastic in (False, True):
            host, optimizer = self.host(weight = torch.ones(size = (128, 128), dtype = torch.bfloat16), master = False, stochastic = stochastic)
            optimizer.param_groups[0]["weight_decay"] = 0.0
            host.grad.fill_(1.0)
            optimizer.step()
            changes[stochastic] = float((host.float() - 1).mean())
        self.assertEqual(changes[False], 0.0)
        self.assertAlmostEqual(changes[True], -1e-3, delta = 2e-4)

    def test_resume_is_exact(self):
        """连续三步，与「一步后取出状态（含主权重与随机数状态）、新建优化器载入、再两步」逐位相同。"""
        for master, stochastic in ((True, False), (False, True)):
            weight, grads = random_case(seed = 2)
            host, optimizer = self.host(weight = weight, master = master, stochastic = stochastic)
            for grad in grads:
                host.grad.copy_(grad)
                optimizer.step()
            straight = host.clone()

            host, optimizer = self.host(weight = weight, master = master, stochastic = stochastic)
            host.grad.copy_(grads[0])
            optimizer.step()
            buffer = io.BytesIO()
            torch.save(obj = {
                "entry": optimizer.state[host],
                "master": optimizer.master_of(param = host),
                "weight": host,
                "rng": optimizer.context.rng,
            }, f = buffer)
            buffer.seek(0)
            saved = torch.load(f = buffer, weights_only = False)
            resumed_host, resumed = self.host(weight = saved["weight"], master = master, stochastic = stochastic)
            resumed.load_entry(param = resumed_host, entry = saved["entry"])
            if master:
                resumed.set_master(param = resumed_host, value = saved["master"])
            resumed.context.rng = saved["rng"]
            for grad in grads[1:]:
                resumed_host.grad.copy_(grad)
                resumed.step()
            self.assertTrue(torch.equal(input = resumed_host, other = straight), msg = (master, stochastic))


@unittest.skipUnless(CUDA and HAS_TORCHAO and (MODEL_DIR / "config.json").exists(), "需要 CUDA、torchao 和阶段 1 模型目录中的代码与配置")
class LanguageModelUpdateTest(unittest.TestCase):
    """缩小的 bfloat16 模型：三种去处的训练与只在显存中训练一致（第一步的损失和梯度范数相同，之后在 AdamW 的
    CPU / GPU 舍入差别以内），规范格式的续训状态可以在不同去处之间互相续训。"""

    @classmethod
    def setUpClass(cls):
        cls.batch = tiny_batch(audio_windows = (50, 800))

    def build(
        self,
        options,
        precision_choice,
        offload,
        state = None,
    ):
        from pipeline.train.model import LANGUAGE_MODEL_PREFIXES, language_model_parameters, move_batch, projector_parameters
        from pipeline.train.optimize import apply
        from pipeline.train.placement import LanguageModelUpdate

        model = tiny_model(dtype = torch.bfloat16)
        for name, parameter in model.named_parameters():
            if name.startswith(LANGUAGE_MODEL_PREFIXES):
                parameter.requires_grad_(requires_grad = True)
        applied = apply(model = model, options = options, train_language_model = True)
        update = LanguageModelUpdate(
            named_parameters = language_model_parameters(model = model),
            projector_params = projector_parameters(model = model),
            options = precision_choice,
            lr = 1e-3,
            language_model_lr = 1e-4,
            weight_decay = 0.0,
            applied = applied,
            offload_optimizer = offload,
            seed = 0,
            state = state,
        )
        inputs = move_batch(batch = self.batch, device = "cuda", feature_dtype = model.model.audio_tower.dtype)
        return model, applied, update, inputs

    def train(
        self,
        model,
        applied,
        update,
        inputs,
        steps,
    ):
        from pipeline.train.model import token_losses

        losses, norms = [], []
        for _ in range(steps):
            with applied.forward_context():
                loss, _, _ = token_losses(model = model, inputs = inputs)
            loss.sum().backward()
            self.assertEqual(update.without_grad(), [])
            norms.append(float(update.clip_grad_norm(max_norm = 1.0)))
            update.optimizer.step()
            update.copy_to_model()
            update.optimizer.zero_grad()
            losses.append(float(loss.detach().sum()))
        return losses, norms

    def options(self, **changes):
        base = TinyModelTest.options(self)
        base.update({"precompute_metadata": True, "gradient_checkpointing": True})
        base.update(changes)
        return base

    PLACEMENTS = {
        "streamed 2": ({"lm_blocks_to_swap": 2}, False),
        "offload optimizer": ({}, True),
        "streamed 4 + offload + activations": ({"lm_blocks_to_swap": 4, "offload_activations": True}, True),
    }

    def assert_close_weights(self, want, got, label, tolerance):
        """更新后的 bfloat16 权重：不同的元素占比很小，差别不超过几个舍入间距。"""
        differ = sum(int((want[name] != got[name]).sum()) for name in want)
        total = sum(tensor.numel() for tensor in want.values())
        self.assertLess(differ / total, tolerance, msg = label)
        for name in want:
            gap = (want[name].float() - got[name].float()).abs()
            spacing = torch.finfo(torch.bfloat16).eps * want[name].float().abs().clamp(min = 1e-2)
            self.assertTrue(bool((gap <= 4 * spacing).all()), msg = f"{label} {name}")

    def test_placements_match_gpu(self):
        for precision_choice, tolerance in (
            (precision_options(), 0.02),
            (precision_options(master = "none", grad = "bf16", rounding = "stochastic"), 1.0),
        ):
            model, applied, update, inputs = self.build(options = self.options(), precision_choice = precision_choice, offload = False)
            reference = self.train(model = model, applied = applied, update = update, inputs = inputs, steps = 2)
            reference_weights = update.model_state()
            reference_state = update.optimizer.state_dict()
            for label, (changes, offload) in self.PLACEMENTS.items():
                label = f"{precision_choice['language_model_master_weights']} {label}"
                model, applied, update, inputs = self.build(options = self.options(**changes), precision_choice = precision_choice, offload = offload)
                losses, norms = self.train(model = model, applied = applied, update = update, inputs = inputs, steps = 2)
                self.assertEqual(losses[0], reference[0][0], msg = label)
                self.assertAlmostEqual(norms[0], reference[1][0], delta = 1e-5 * reference[1][0], msg = label)
                self.assert_close_weights(want = reference_weights, got = update.model_state(), label = label, tolerance = tolerance)
                state = update.optimizer.state_dict()
                # 规范格式：与只在显存中训练时的键、形状、精度相同。
                self.assertEqual(sorted(state["state"]), sorted(reference_state["state"]), msg = label)
                self.assertEqual([group["params"] for group in state["param_groups"]], [group["params"] for group in reference_state["param_groups"]])
                for index, entry in reference_state["state"].items():
                    for key, value in entry.items():
                        self.assertEqual(state["state"][index][key].shape, value.shape, msg = (label, index, key))
                        self.assertEqual(state["state"][index][key].dtype, value.dtype, msg = (label, index, key))
                self.assertEqual(sorted(update.weights_state()), sorted(name for name, _ in update.named))

    def test_resume_across_placements(self):
        """卸载与流式加载时保存的状态在只用显存时续训，反过来也可以；结果与不中断的训练在舍入差别以内。同一种去处
        保存、新建、载入后再训练一步，与不中断的训练只差 GPU 内核本身的不确定性（这里不打开确定性模式）。"""
        from pipeline.train.model import load_projector, projector_state_dict

        choice = precision_options()
        model, applied, update, inputs = self.build(options = self.options(), precision_choice = choice, offload = False)
        self.train(model = model, applied = applied, update = update, inputs = inputs, steps = 3)
        straight = update.model_state()
        streamed = ({"lm_blocks_to_swap": 2, "offload_activations": True}, True)
        gpu = ({}, False)
        for (first, second) in ((streamed, gpu), (gpu, streamed), (streamed, streamed)):
            model, applied, update, inputs = self.build(options = self.options(**first[0]), precision_choice = choice, offload = first[1])
            self.train(model = model, applied = applied, update = update, inputs = inputs, steps = 2)
            buffer = io.BytesIO()
            torch.save(obj = {
                "weights": update.weights_state(),
                "optimizer": update.optimizer.state_dict(),
                "projector": projector_state_dict(model = model),
            }, f = buffer)
            buffer.seek(0)
            saved = torch.load(f = buffer, weights_only = False, map_location = "cpu")
            model, applied, update, inputs = self.build(
                options = self.options(**second[0]),
                precision_choice = choice,
                offload = second[1],
                state = saved["weights"],
            )
            load_projector(model = model, state = saved["projector"])
            update.optimizer.load_state_dict(state_dict = saved["optimizer"])
            self.train(model = model, applied = applied, update = update, inputs = inputs, steps = 1)
            self.assert_close_weights(want = straight, got = update.model_state(), label = f"{first} -> {second}", tolerance = 0.02)

    def test_rebuild_after_degrade(self):
        """显存不足时把更多层改为流式加载（rebuild）：主权重和 AdamW 状态原样搬到主机，之后的训练与不调整时一致。"""
        choice = precision_options()
        model, applied, update, inputs = self.build(options = self.options(), precision_choice = choice, offload = False)
        self.train(model = model, applied = applied, update = update, inputs = inputs, steps = 3)
        straight = update.model_state()
        model, applied, update, inputs = self.build(options = self.options(), precision_choice = choice, offload = False)
        self.train(model = model, applied = applied, update = update, inputs = inputs, steps = 1)
        before = update.weights_state()
        before = {name: tensor.detach().cpu().clone() for name, tensor in before.items()}
        while "lm_blocks_to_swap" not in (change := applied.degrade()):
            pass
        update.rebuild()
        self.assertEqual(change, {"lm_blocks_to_swap": 1})
        self.assertTrue(update.streamed_named)
        for _, parameter in update.streamed_named:
            self.assertEqual(parameter.device.type, "cpu")
        after = update.weights_state()
        for name, tensor in before.items():
            self.assertTrue(torch.equal(input = after[name].detach().cpu(), other = tensor), msg = name)
        self.train(model = model, applied = applied, update = update, inputs = inputs, steps = 2)
        self.assert_close_weights(want = straight, got = update.model_state(), label = "rebuild", tolerance = 0.02)


if __name__ == "__main__":
    unittest.main()
