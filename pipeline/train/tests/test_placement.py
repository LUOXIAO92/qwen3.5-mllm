"""语言模型参数在主机内存中更新（任务 #66，placement.py）的单元测试：python -m unittest pipeline.train.tests.test_placement

HostAdamW 与显存中的 AdamW 的一致性、续训逐位相同、随机舍入，以及 LanguageModelUpdate 在缩小的模型上三种去处（显存、
流式加载的层、卸载优化器）的损失、梯度范数、更新后的权重和规范格式的续训状态。需要 CUDA 和 torchao，没有时跳过。

一步有多个微批时主机上的梯度累积（任务 #71）：AccumulateTest（CPU）核对累加的加法与显存中 MasterWeights 的钩子逐位相同、
HostAdamW 读累加缓冲时与 torch 的 CPU fused AdamW 逐位相同；HostAccumulationTest（CUDA、确定性模式）在缩小的模型上每步
三个微批，核对累加出的梯度与只在显存中训练时逐位相同、裁剪用全局范数、一个微批时与以前逐位相同、续训逐位相同；
HostReplayTest（另需拼接内核）走 train.replay_forward_backward 的回放微批。
"""

import importlib.util
import io
import tempfile
import unittest

import numpy as np
import torch

from pipeline.train import precision
from pipeline.train.tests.test_offload import MODEL_DIR, TinyModelTest, tiny_batch, tiny_model
from pipeline.train.tests.test_packing import kernels_available
from pipeline.train.tests.test_replay import FakeReplay

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
        self.assertIsNone(precision.host_problem(options = precision_options()))
        self.assertIsNone(precision.host_problem(options = precision_options(grad = "bf16")))
        stochastic = precision_options(master = "none", grad = "bf16", rounding = "stochastic")
        self.assertIsNone(precision.host_problem(options = stochastic))
        for states in ("bf16", "8bit", "4bit", "fp8"):
            self.assertIsNotNone(precision.host_problem(options = precision_options(states = states)))


class AccumulateTest(unittest.TestCase):
    """主机上的梯度累积（CPU）：accumulate 与显存中 MasterWeights 钩子的加法逐位相同；HostAdamW 读累加缓冲时与 torch 的
    CPU fused AdamW 对同一个梯度（累加后乘裁剪系数）的更新逐位相同。"""

    def test_matches_master_weights_hook(self):
        from pipeline.train.placement import CHUNK, accumulate

        # 跨块边界的长度，三个微批的 bfloat16 梯度，量级相差很大，让 float32 加法确有舍入。
        generator = torch.Generator().manual_seed(0)
        shape = (CHUNK // 512 + 3, 517)
        grads = [
            (scale * torch.randn(size = shape, generator = generator)).to(dtype = torch.bfloat16)
            for scale in (1.0, 1e-3, 30.0)
        ]
        buffer = torch.empty(size = (CHUNK,), dtype = torch.float32)
        for dtype in (torch.float32, torch.bfloat16):
            target = torch.empty(size = shape, dtype = dtype)
            for number, grad in enumerate(grads):
                accumulate(target = target, source = grad, first = number == 0, buffer = buffer)
            # MasterWeights 的钩子：第一次 master.grad = grad.to(梯度精度)，之后 master.grad.add_(grad)。
            expected = grads[0].to(dtype = dtype)
            for grad in grads[1:]:
                expected.add_(grad)
            self.assertTrue(torch.equal(input = target, other = expected), msg = str(dtype))
            # 加法确有舍入：换一个相加的顺序，结果就不同。
            reverse = grads[-1].to(dtype = dtype)
            for grad in grads[-2::-1]:
                reverse.add_(grad)
            self.assertFalse(torch.equal(input = target, other = reverse), msg = str(dtype))

    def test_host_adamw_reads_accumulated(self):
        from pipeline.train.placement import HostAdamW, HostContext

        weight, grads = random_case(seed = 4)
        context = HostContext(seed = 0)
        context.scale.fill_(0.37)
        host = weight.clone()
        # 搬运缓冲里放的是最后一个微批的梯度；有累加时 HostAdamW 应当读累加缓冲。
        host.grad = grads[-1].clone()
        optimizer = HostAdamW(
            params = [host],
            lr = 1e-3,
            betas = (0.9, 0.999),
            eps = 1e-8,
            weight_decay = 0.01,
            master = True,
            stochastic = False,
            context = context,
        )
        master = torch.nn.Parameter(weight.float())
        reference = torch.optim.AdamW(params = [master], lr = 1e-3, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 0.01, fused = True)
        for step in range(3):
            accumulated = grads[step % 3].float() + grads[(step + 1) % 3].float()
            context.accumulated = {id(host): accumulated}
            optimizer.step()
            context.accumulated = None
            master.grad = accumulated * context.scale
            reference.step()
            self.assertTrue(torch.equal(input = optimizer.masters[host], other = master.detach()), msg = step)
            self.assertTrue(torch.equal(input = host, other = master.detach().to(dtype = torch.bfloat16)), msg = step)
        # 这一步没有得到梯度的参数（不在累加缓冲中）不更新，与显存中梯度为 None 的参数相同。
        before = host.clone()
        context.accumulated = {}
        optimizer.step()
        context.accumulated = None
        self.assertTrue(torch.equal(input = host, other = before))
        self.assertEqual(float(optimizer.state[host]["step"]), 3.0)


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


def micro_batches(count, seed = 1):
    """一步内的 count 个微批：各 2–3 条长短不一的音频（第 2 个微批含一个空目标样本），内容互不相同。"""
    from transformers import AutoProcessor

    from pipeline.train.data import TrainCollator

    processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path = str(MODEL_DIR), trust_remote_code = True)
    collator = TrainCollator(processor = processor, audio_windows = (50, 800))
    rng = np.random.default_rng(seed = seed)
    batches = []
    for number in range(count):
        items = []
        for index in range(2 + number % 2):
            seconds = float(rng.uniform(low = 1.0, high = 9.0))
            empty = number == 1 and index == 0
            items.append({
                "sample_id": f"m{number}-{index}",
                "lang": "en",
                "audio_path": f"m{number}-{index}.flac",
                "duration_s": seconds,
                "waveform": (0.1 * rng.standard_normal(size = int(seconds * 16000))).astype(np.float32),
                "target": "" if empty else f"micro batch {number} sample {index}",
                "empty": empty,
                "mix": None,
            })
        batches.append(collator(items = items))
    return batches


def language_model_grads(update):
    """这一步裁剪前全部语言模型参数的梯度 {名称: CPU 张量}：显存中的在主权重（或参数）上；主机上的在累加缓冲中，
    一步只有一次反传时在搬运缓冲中。"""
    torch.cuda.synchronize()
    weights = update.weights
    pairs = list(zip(weights.names, weights.masters)) + weights.direct if hasattr(weights, "masters") else weights.named
    grads = {name: tensor.grad.detach().cpu().clone() for name, tensor in pairs}
    for name, (_, host) in update.host_entries.items():
        source = update.accumulators[name] if update.accumulating else host.grad
        grads[name] = source.detach().clone()
    return grads


def cpu_tree(value):
    """优化器状态等嵌套结构中的张量复制到 CPU（续训核对用）。"""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(value = item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(value = item) for item in value)
    return value


def max_difference(want, got):
    """两组同名张量的最大绝对差、最大相对差（相对于该张量的最大绝对值）和不同元素的占比。"""
    absolute = relative = 0.0
    differ = total = 0
    for name, expected in want.items():
        expected = expected.float()
        actual = got[name].float()
        gap = (expected - actual).abs()
        absolute = max(absolute, float(gap.max()))
        relative = max(relative, float(gap.max() / expected.abs().max().clamp(min = 1e-30)))
        differ += int((gap > 0).sum())
        total += gap.numel()
    return {"max_abs": absolute, "max_rel": relative, "differ_fraction": differ / max(1, total)}


@unittest.skipUnless(CUDA and HAS_TORCHAO and (MODEL_DIR / "config.json").exists(), "需要 CUDA、torchao 和阶段 1 模型目录中的代码与配置")
class HostAccumulationTest(unittest.TestCase):
    """一步有多个微批时主机上的梯度累积（任务 #71）。缩小的 bfloat16 模型，每步三个微批；确定性模式、线性注意力用 PyTorch
    参考实现，同一份权重上各次运行的每个微批的梯度逐位相同，因此可以逐位比较：

    - 累加出的梯度等于各微批搬到主机的梯度依次相加，也与只在显存中训练时 MasterWeights / DirectWeights 累加的梯度逐位相同；
    - 裁剪用累加后梯度的全局范数（与显存中的总范数只差 CPU、GPU 求和顺序的舍入），系数作用于主机上的梯度；
    - 几步之后的权重与只在显存中训练一致（在 CPU 与 GPU 的 AdamW 舍入差别以内，见 LanguageModelUpdateTest）；
    - 一个微批时与以前的做法（不调用 begin_step、end_micro_batch）逐位相同；停下再续训与不中断的训练逐位相同。

    精度组合取默认、fp32 主权重 + bf16 梯度、none + bf16 梯度 + 随机舍入。none + fp32 梯度不在逐位比较之列：显存中
    DirectWeights 给参数设 grad_dtype = float32，autograd 在 float32 中合并词嵌入兼输出层的两处梯度、求广播参数（dt_bias）
    的梯度，主机上的参数收到的是 bfloat16 中算出的结果，一个微批时两者就有舍入量级的差别（与累加无关）。
    """

    # 裁剪上限取得小，让每一步都确实裁剪。
    MAX_NORM = 0.01
    CHOICES = {
        "default": precision_options(),
        "bf16 grads": precision_options(grad = "bf16"),
        "none + bf16 + stochastic": precision_options(master = "none", grad = "bf16", rounding = "stochastic"),
    }
    PLACEMENTS = LanguageModelUpdateTest.PLACEMENTS

    @classmethod
    def setUpClass(cls):
        cls.batches = micro_batches(count = 3)

    def setUp(self):
        import os

        from pipeline.train.check_packing import reference_linear_attention

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(mode = True)
        self.linear_attention = reference_linear_attention()
        self.linear_attention.__enter__()

    def tearDown(self):
        self.linear_attention.__exit__(None, None, None)
        torch.use_deterministic_algorithms(mode = False)

    def build(self, changes, offload, choice, state = None):
        from pipeline.train.model import LANGUAGE_MODEL_PREFIXES, language_model_parameters, move_batch, projector_parameters
        from pipeline.train.optimize import apply
        from pipeline.train.placement import LanguageModelUpdate

        model = tiny_model(dtype = torch.bfloat16)
        for name, parameter in model.named_parameters():
            if name.startswith(LANGUAGE_MODEL_PREFIXES):
                parameter.requires_grad_(requires_grad = True)
        options = {**TinyModelTest.options(self), "precompute_metadata": True, "gradient_checkpointing": True, **changes}
        applied = apply(model = model, options = options, train_language_model = True)
        update = LanguageModelUpdate(
            named_parameters = language_model_parameters(model = model),
            projector_params = projector_parameters(model = model),
            options = choice,
            lr = 1e-3,
            language_model_lr = 1e-4,
            weight_decay = 0.01,
            applied = applied,
            offload_optimizer = offload,
            seed = 0,
            state = state,
        )
        inputs = [move_batch(batch = batch, device = "cuda", feature_dtype = model.model.audio_tower.dtype) for batch in self.batches]
        return model, applied, update, inputs

    def train(
        self,
        model,
        applied,
        update,
        inputs,
        steps,
        snapshots = None,
        begin = True,
    ):
        """每步依次前向反传各个微批，损失除以整步的计损失 token 数（与 train.py 相同）。返回每步的损失之和、总范数和第一步
        裁剪前的语言模型梯度。snapshots 为 list 时记下第一步每个微批搬到主机的梯度；begin 为假时按以前的做法，不调用
        begin_step、end_micro_batch（只用于一个微批）。"""
        from pipeline.train.model import token_losses

        n_tokens = sum(int((item["labels"][:, 1:] != -100).sum()) for item in inputs)
        losses, norms, first = [], [], None
        for step in range(steps):
            if begin:
                update.begin_step(micro_batches = len(inputs))
            total = 0.0
            for item in inputs:
                with applied.forward_context():
                    loss, _, _ = token_losses(model = model, inputs = item)
                (loss.sum() / n_tokens).backward()
                if snapshots is not None and step == 0:
                    torch.cuda.synchronize()
                    snapshots.append({name: host.grad.detach().clone() for name, (_, host) in update.host_entries.items()})
                if begin:
                    update.end_micro_batch()
                total += float(loss.detach().sum())
            if step == 0:
                first = language_model_grads(update = update)
            norm = update.clip_grad_norm(max_norm = self.MAX_NORM)
            if update.host_entries:
                # 主机上的梯度乘的系数就是显存中 clip_grads_with_norm_ 用的系数，并且确实在裁剪。
                coefficient = torch.clamp(input = self.MAX_NORM / (norm + 1e-6), max = 1.0).cpu()
                self.assertTrue(torch.equal(input = update.context.scale, other = coefficient))
                self.assertLess(float(coefficient), 1.0)
            norms.append(float(norm))
            update.optimizer.step()
            update.copy_to_model()
            update.optimizer.zero_grad()
            losses.append(total)
        return losses, norms, first

    def test_accumulation_matches_gpu(self):
        for choice_label, choice in self.CHOICES.items():
            model, applied, update, inputs = self.build(changes = {}, offload = False, choice = choice)
            want_losses, want_norms, want_grads = self.train(model = model, applied = applied, update = update, inputs = inputs, steps = 3)
            want_weights = update.model_state()
            for label, (changes, offload) in self.PLACEMENTS.items():
                label = f"{choice_label} {label}"
                snapshots = []
                model, applied, update, inputs = self.build(changes = changes, offload = offload, choice = choice)
                losses, norms, grads = self.train(
                    model = model,
                    applied = applied,
                    update = update,
                    inputs = inputs,
                    steps = 3,
                    snapshots = snapshots,
                )
                self.assertEqual(len(snapshots), 3)
                self.assertTrue(update.host_entries, msg = label)
                for name in update.host_entries:
                    # 各微批搬到主机的梯度依次相加（MasterWeights 钩子的做法）
                    expected = snapshots[0][name].to(dtype = grads[name].dtype)
                    for snapshot in snapshots[1:]:
                        expected.add_(snapshot[name])
                    self.assertTrue(torch.equal(input = grads[name], other = expected), msg = f"{label} {name}")
                # 全部语言模型参数的累加梯度与只在显存中训练时逐位相同
                self.assertEqual(set(grads), set(want_grads))
                for name, want in want_grads.items():
                    self.assertTrue(torch.equal(input = grads[name], other = want), msg = f"{label} {name}")
                self.assertEqual(losses[0], want_losses[0], msg = label)
                self.assertAlmostEqual(norms[0], want_norms[0], delta = 1e-6 * want_norms[0], msg = label)
                for got, want in zip(losses[1:], want_losses[1:]):
                    self.assertAlmostEqual(got, want, delta = 1e-3 * abs(want), msg = label)
                tolerance = 1.0 if choice["language_model_rounding"] == "stochastic" else 0.02
                LanguageModelUpdateTest.assert_close_weights(
                    self,
                    want = want_weights,
                    got = update.model_state(),
                    label = label,
                    tolerance = tolerance,
                )

    def test_single_micro_batch_unchanged(self):
        """一步只有一个微批：调用 begin_step(1) 与 end_micro_batch 时与以前的做法逐位相同（主机上不另做累加）。"""
        for label, (changes, offload) in self.PLACEMENTS.items():
            results = []
            for begin in (False, True):
                model, applied, update, inputs = self.build(changes = changes, offload = offload, choice = precision_options())
                losses, norms, _ = self.train(
                    model = model,
                    applied = applied,
                    update = update,
                    inputs = inputs[:1],
                    steps = 3,
                    begin = begin,
                )
                self.assertEqual(update.accumulators, {}, msg = label)
                results.append((losses, norms, update.model_state(), cpu_tree(value = update.weights_state())))
            self.assertEqual(results[0][:2], results[1][:2], msg = label)
            for first, second in zip(results[0][2:], results[1][2:]):
                for name in first:
                    self.assertTrue(torch.equal(input = first[name], other = second[name]), msg = f"{label} {name}")

    def assert_same_tree(self, want, got, label):
        if isinstance(want, torch.Tensor):
            self.assertTrue(torch.equal(input = want, other = got), msg = label)
        elif isinstance(want, dict):
            self.assertEqual(set(want), set(got), msg = label)
            for key in want:
                self.assert_same_tree(want = want[key], got = got[key], label = f"{label}.{key}")
        elif isinstance(want, (list, tuple)):
            self.assertEqual(len(want), len(got), msg = label)
            for index, (first, second) in enumerate(zip(want, got)):
                self.assert_same_tree(want = first, got = second, label = f"{label}[{index}]")
        else:
            self.assertEqual(want, got, msg = label)

    def test_resume_is_exact(self):
        """三个微批的主机累积：连续 4 步，与「2 步后保存权重、规范格式的优化器状态、projector 和随机数状态，新建、载入后
        再 2 步」逐位相同（流式 2 层 + 卸载优化器 + 激活值卸载；默认精度与无主权重的随机舍入）。"""
        from pipeline.train.model import load_projector, projector_state_dict

        changes, offload = {"lm_blocks_to_swap": 2, "offload_activations": True}, True
        for choice_label in ("default", "none + bf16 + stochastic"):
            choice = self.CHOICES[choice_label]
            model, applied, update, inputs = self.build(changes = changes, offload = offload, choice = choice)
            straight_losses, _, _ = self.train(model = model, applied = applied, update = update, inputs = inputs, steps = 4)
            straight = {
                "model": update.model_state(),
                "weights": cpu_tree(value = update.weights_state()),
                "optimizer": cpu_tree(value = update.optimizer.state_dict()),
                "projector": cpu_tree(value = projector_state_dict(model = model)),
            }

            model, applied, update, inputs = self.build(changes = changes, offload = offload, choice = choice)
            first_losses, _, _ = self.train(model = model, applied = applied, update = update, inputs = inputs, steps = 2)
            buffer = io.BytesIO()
            torch.save(obj = {
                "weights": update.weights_state(),
                "optimizer": update.optimizer.state_dict(),
                "projector": projector_state_dict(model = model),
                "host_rng": update.context.rng,
                "cuda_rng": torch.cuda.get_rng_state(),
                "cpu_rng": torch.get_rng_state(),
            }, f = buffer)
            update.remove()
            buffer.seek(0)
            saved = torch.load(f = buffer, weights_only = False, map_location = "cpu")
            model, applied, update, inputs = self.build(
                changes = changes,
                offload = offload,
                choice = choice,
                state = saved["weights"],
            )
            load_projector(model = model, state = saved["projector"])
            update.optimizer.load_state_dict(state_dict = saved["optimizer"])
            update.context.rng = saved["host_rng"]
            torch.cuda.set_rng_state(new_state = saved["cuda_rng"])
            torch.set_rng_state(new_state = saved["cpu_rng"])
            second_losses, _, _ = self.train(model = model, applied = applied, update = update, inputs = inputs, steps = 2)
            self.assertEqual(first_losses + second_losses, straight_losses, msg = choice_label)
            resumed = {
                "model": update.model_state(),
                "weights": cpu_tree(value = update.weights_state()),
                "optimizer": cpu_tree(value = update.optimizer.state_dict()),
                "projector": cpu_tree(value = projector_state_dict(model = model)),
            }
            self.assert_same_tree(want = straight, got = resumed, label = choice_label)


@unittest.skipUnless(
    CUDA and HAS_TORCHAO and (MODEL_DIR / "config.json").exists() and kernels_available(),
    "需要 CUDA、torchao、阶段 1 模型目录中的代码与配置，以及 flash-attn、causal-conv1d、fla",
)
class HostReplayTest(unittest.TestCase):
    """开启回放的一步（train.replay_forward_backward，拼接批次、kl32 加尾部项）：ASR 行接短样本，长样本另成纯文本微批，
    语言模型参数在主机内存中更新时各微批的梯度在主机上累加，与只在显存中训练一致。拼接内核（flash-attn、causal-conv1d、fla）
    的反传不确定，两次运行的梯度本身就有舍入量级的差别，这里按相对误差比较；主机上的累加仍与搬来的各微批梯度逐位相同。"""

    MAX_NORM = 0.01

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        FakeReplay(root = self.tmp.name).write_all(count = 12, long_every = 4)

    def tearDown(self):
        self.tmp.cleanup()

    def run_steps(self, changes, offload, steps):
        from pipeline.train.data import pack_batch
        from pipeline.train.model import LANGUAGE_MODEL_PREFIXES, PACKED_ATTENTION, language_model_parameters, projector_parameters
        from pipeline.train.optimize import apply
        from pipeline.train.placement import LanguageModelUpdate
        from pipeline.train.replay import Replay, parse_groups
        from pipeline.train.train import replay_forward_backward

        model = tiny_model(dtype = torch.bfloat16)
        model.set_attn_implementation(attn_implementation = PACKED_ATTENTION)
        for name, parameter in model.named_parameters():
            if name.startswith(LANGUAGE_MODEL_PREFIXES):
                parameter.requires_grad_(requires_grad = True)
        options = {
            **TinyModelTest.options(self),
            "precompute_metadata": True,
            "pack_sequences": True,
            "gradient_checkpointing": True,
            **changes,
        }
        applied = apply(model = model, options = options, train_language_model = True)
        update = LanguageModelUpdate(
            named_parameters = language_model_parameters(model = model),
            projector_params = projector_parameters(model = model),
            options = precision_options(),
            lr = 1e-3,
            language_model_lr = 1e-4,
            weight_decay = 0.0,
            applied = applied,
            offload_optimizer = offload,
            seed = 0,
        )
        config = model.config.audio_config
        asr = pack_batch(batch = tiny_batch(audio_windows = (config.n_window, config.n_window_infer), prompt_format = "cpt"))
        asr_tokens = int((asr["labels"][:, 1:] != -100).sum())
        replay = Replay(
            root = self.tmp.name,
            weights = parse_groups(spec = "free_raw=1,cont_raw=1,free_chat_think=1"),
            ratio = 40.0,
            seed = 0,
            loss = "kl32",
            inline_max_tokens = 100,
            micro_batch_tokens = 4000,
        )
        counts, snapshots, firsts, norms = [], [], [], []
        for step in range(steps):
            batches, info = replay.extend(micro_batches = [asr], asr_tokens = asr_tokens, step = step + 1)
            counts.append(len(batches))
            step_snapshots = []

            def after_backward():
                torch.cuda.synchronize()
                step_snapshots.append({name: host.grad.detach().clone() for name, (_, host) in update.host_entries.items()})
                update.end_micro_batch()

            update.begin_step(micro_batches = len(batches))
            replay_forward_backward(
                model = model,
                micro_batches = batches,
                n_tokens = replay.normalizer(asr_tokens = asr_tokens, replay_tokens = info["replay_tokens"]),
                device = "cuda",
                feature_dtype = model.model.audio_tower.dtype,
                forward_context = applied.forward_context,
                kl_tail = True,
                after_backward = after_backward,
            )
            snapshots.append(step_snapshots)
            firsts.append(language_model_grads(update = update))
            norms.append(float(update.clip_grad_norm(max_norm = self.MAX_NORM)))
            update.optimizer.step()
            update.copy_to_model()
            update.optimizer.zero_grad()
        return counts, snapshots, firsts, norms, update.model_state(), update

    def test_replay_steps_match_gpu(self):
        counts, _, want_grads, want_norms, want_weights, _ = self.run_steps(changes = {}, offload = False, steps = 3)
        # 第 1 步取到长样本、另成纯文本微批；之后预算为负，有的步只有 ASR 一个微批（两种情况都走到）。
        self.assertGreaterEqual(counts[0], 2, msg = counts)
        self.assertIn(1, counts, msg = counts)
        for label, (changes, offload) in LanguageModelUpdateTest.PLACEMENTS.items():
            got_counts, snapshots, grads, norms, weights, update = self.run_steps(changes = changes, offload = offload, steps = 3)
            self.assertEqual(got_counts, counts, msg = label)
            for step, step_snapshots in enumerate(snapshots):
                self.assertEqual(len(step_snapshots), counts[step], msg = label)
                for name in update.host_entries:
                    expected = step_snapshots[0][name].float()
                    for snapshot in step_snapshots[1:]:
                        expected.add_(snapshot[name])
                    self.assertTrue(torch.equal(input = grads[step][name].float(), other = expected), msg = f"{label} 第 {step + 1} 步 {name}")
            # 第一步（权重相同）：累加出的梯度与只在显存中训练时只差拼接内核反传本身的不确定性。两次只在显存中的运行之间，
            # 线性注意力门控参数（A_log、dt_bias，几个元素、正负相消的和）的梯度就相差约 13%，所以逐个张量只作宽松的检查，
            # 全部梯度合在一起的相对差要小。
            difference = total = 0.0
            for name, want in want_grads[0].items():
                gap = float((grads[0][name].float() - want.float()).norm())
                size = float(want.float().norm())
                self.assertLess(gap, 0.5 * max(size, 1e-12), msg = f"{label} {name}")
                difference += gap ** 2
                total += size ** 2
            self.assertLess((difference / total) ** 0.5, 1e-2, msg = label)
            self.assertAlmostEqual(norms[0], want_norms[0], delta = 1e-4 * want_norms[0], msg = label)
            LanguageModelUpdateTest.assert_close_weights(
                self,
                want = want_weights,
                got = weights,
                label = label,
                tolerance = 0.05,
            )


if __name__ == "__main__":
    unittest.main()
