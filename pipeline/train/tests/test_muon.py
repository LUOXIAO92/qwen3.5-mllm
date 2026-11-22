"""Muon 优化器（任务 #78，muon.py）的单元测试：python -m unittest pipeline.train.tests.test_muon

CPU 上运行：选项与续训核对、不合法组合的报错；按 Qwen3.5-2B 的结构在 meta 设备上建模型（不需要权重），检查参数分组；
GroupStateMuon 与 torch.optim.Muon 逐位相同，以及其余精度组合的计算和续训时保持动量精度；缩小的模型上只训练 projector
和全量微调几步，与直接用 torch.optim.Muon 加 AdamW 写成的参照逐位相同，停下再续训与不中断的训练逐位相同。语言模型的真实
前向在本机要用 GPU 上的内核（causal-conv1d），这里的损失只依赖参数本身（synthetic_loss），检查的是优化器的部分。
--offload-optimizer 加 Muon（OffloadTest）需要 CUDA 和 torchao，没有时跳过。
"""

import importlib.util
import io
import unittest

import torch

from pipeline.train import muon
from pipeline.train.optimize import Applied, validate
from pipeline.train.tests.test_offload import MODEL_DIR, tiny_batch, tiny_model
from pipeline.train.train import RESUME_KEYS, check_language_model_args, parse_args, peak_lrs, resume_mismatches, run_config

HAS_TORCHAO = importlib.util.find_spec(name = "torchao") is not None
CUDA = torch.cuda.is_available()
HAS_CONFIG = (MODEL_DIR / "config.json").exists()

ARGV = [
    "--model-dir", "/repo/qwen3.5-2b-mllm",
    "--train-manifest", "/data/train/manifest-v2b.jsonl",
    "--target", "original",
    "--output-dir", "/work/data/T78/muon",
]
FULL = ["--train-language-model", "--language-model-lr", "1e-4"]
MUON = ["--optimizer", "muon"]
LR = 1e-3
LANGUAGE_MODEL_LR = 1e-4
WEIGHT_DECAY = 0.01
# 2B 的语言模型中各类参数（在 meta 设备上按真实配置建出的模型中数得）。
SIZES_2B = {
    "projector_adamw": (2, 4096),
    "projector_muon": (2, 8388608),
    "language_model_adamw": (134, 509108032),
    "language_model_muon": (186, 1372717056),
}
MUON_SUFFIXES = (
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.in_proj_a.weight",
    "linear_attn.out_proj.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "audio_projector.linear_1.weight",
    "audio_projector.linear_2.weight",
)


def precision_options(master = "fp32", grad = "fp32", rounding = "nearest", states = "fp32"):
    return {
        "language_model_master_weights": master,
        "language_model_grad_dtype": grad,
        "language_model_rounding": rounding,
        "language_model_optimizer_states": states,
    }


def check(argv):
    check_language_model_args(args = parse_args(argv = ARGV + argv))


class OptionsTest(unittest.TestCase):
    def test_defaults_keep_adamw(self):
        args = parse_args(argv = ARGV + FULL)
        self.assertEqual(args.optimizer, "adamw")
        check_language_model_args(args = args)
        self.assertEqual(peak_lrs(args = args, muon_settings = None), [1e-3, 1e-4])
        self.assertEqual(muon.lr_labels(train_language_model = True, optimizer = "adamw"), ["lr", "lm_lr"])

    def test_resume_keys(self):
        for key in muon.KEYS:
            self.assertIn(key, RESUME_KEYS)
        for key in ("muon_lr", "language_model_muon_lr", "muon_momentum", "muon_ns_steps"):
            self.assertNotIn(key, RESUME_KEYS)
        saved = run_config(args = parse_args(argv = ARGV + FULL))
        for key in muon.DEFAULTS:
            saved.pop(key)
        self.assertEqual(resume_mismatches(saved = saved, config = run_config(args = parse_args(argv = ARGV + FULL))), [])
        changed = run_config(args = parse_args(argv = ARGV + FULL + MUON + ["--muon-momentum-dtype", "bf16"]))
        self.assertEqual(resume_mismatches(saved = saved, config = changed), ["optimizer", "muon_momentum_dtype"])

    def test_invalid_combinations(self):
        cases = [
            ["--muon-momentum", "0.9"],
            ["--no-muon-nesterov"],
            ["--muon-lr", "0.02"],
            MUON + ["--muon-lr-scaling", "original"],
            MUON + FULL + ["--muon-lr-scaling", "original", "--muon-lr", "0.02"],
            MUON + ["--muon-momentum-dtype", "bf16"],
            MUON + ["--language-model-muon-lr", "1e-5"],
            MUON + ["--muon-ns-steps", "0"],
            MUON + ["--muon-momentum", "1.0"],
            MUON + ["--muon-weight-decay", "-0.1"],
        ]
        for argv in cases:
            with self.assertRaises(SystemExit, msg = argv):
                check(argv = argv)
        check(argv = MUON)
        check(argv = MUON + FULL + ["--muon-momentum-dtype", "bf16", "--offload-optimizer"])
        check(argv = MUON + FULL + [
            "--muon-lr-scaling", "original",
            "--muon-lr", "0.02",
            "--language-model-muon-lr", "0.002",
        ])
        # AdamW 部分照常可用低比特状态。
        check(argv = MUON + FULL + [
            "--language-model-master-weights", "none",
            "--language-model-grad-dtype", "bf16",
            "--language-model-rounding", "stochastic",
            "--language-model-optimizer-states", "8bit",
        ])

    def test_streaming_is_rejected_for_muon(self):
        options = {
            "gradient_checkpointing": True,
            "checkpoint_layers": None,
            "lm_blocks_to_swap": 2,
            "audio_blocks_to_swap": 0,
            "offload_activations": False,
            "fp8_frozen": "none",
            "compile": "none",
            "precompute_metadata": True,
            "offload_optimizer": False,
        }
        problem = muon.stream_problem(optimizer = "muon", train_language_model = True)
        with self.assertRaisesRegex(ValueError, "--optimizer muon"):
            validate(
                options = options,
                total = 4,
                train_language_model = True,
                stream_problem = problem,
            )
        # 只训练 projector 时语言模型冻结，流式加载照常可用；--offload-optimizer 不受这一项限制。
        self.assertIsNone(muon.stream_problem(optimizer = "muon", train_language_model = False))
        validate(options = options, total = 4, train_language_model = False)
        validate(
            options = {**options, "lm_blocks_to_swap": 0, "offload_optimizer": True},
            total = 4,
            train_language_model = True,
            stream_problem = problem,
        )
        applied = Applied(
            model = None,
            options = {**options, "lm_blocks_to_swap": 0},
            device = "cpu",
            train_language_model = True,
            stream_problem = problem,
        )
        self.assertFalse(applied.lm_streamable)

    def test_learning_rates_and_notice(self):
        args = parse_args(argv = ARGV + FULL + MUON + ["--muon-lr", "2e-3"])
        settings = settings_for(args = args, names = frozenset(), mask = ())
        self.assertEqual(peak_lrs(args = args, muon_settings = settings), [1e-3, 1e-4, 2e-3, 1e-4])
        self.assertEqual(
            muon.lr_labels(train_language_model = True, optimizer = "muon"),
            ["lr", "lm_lr", "muon_lr", "lm_muon_lr"],
        )
        groups = [{"lr": 1e-3}, {"ns_steps": 5, "momentum": 0.9, "nesterov": True, "adjust_lr_fn": "match_rms_adamw", "weight_decay": 0.0}]
        self.assertIn("'momentum': 0.9", muon.resume_notice(param_groups = groups, settings = settings))
        groups[1]["momentum"] = 0.95
        self.assertIsNone(muon.resume_notice(param_groups = groups, settings = settings))


def settings_for(args, names, mask):
    return muon.Settings(
        names = names,
        projector_mask = tuple(mask),
        lr = args.muon_lr if args.muon_lr is not None else args.lr,
        language_model_lr = args.language_model_muon_lr if args.language_model_muon_lr is not None else args.language_model_lr,
        momentum = args.muon_momentum,
        nesterov = args.muon_nesterov,
        ns_steps = args.muon_ns_steps,
        lr_scaling = args.muon_lr_scaling,
        weight_decay = args.muon_weight_decay if args.muon_weight_decay is not None else args.weight_decay,
        momentum_dtype = args.muon_momentum_dtype,
    )


@unittest.skipUnless(HAS_CONFIG, "需要阶段 1 模型目录中的代码与配置")
class GroupingTest(unittest.TestCase):
    """Qwen3.5-2B 的结构（阶段 1 的 0.8B 配置改成 2B 的语言模型尺寸，在 meta 设备上建，不分配内存）。"""

    @classmethod
    def setUpClass(cls):
        from transformers import AutoConfig, AutoModelForImageTextToText

        from pipeline.train.model import LANGUAGE_MODEL_PREFIXES, PROJECTOR_PREFIX

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path = str(MODEL_DIR), trust_remote_code = True)
        config.text_config.hidden_size = 2048
        config.text_config.intermediate_size = 6144
        config.vision_config.out_hidden_size = 2048
        with torch.device("meta"):
            model = AutoModelForImageTextToText.from_config(config = config, trust_remote_code = True, dtype = torch.bfloat16)
        model.model.audio_projector.float()
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(requires_grad = name.startswith(PROJECTOR_PREFIX) or name.startswith(LANGUAGE_MODEL_PREFIXES))
        cls.model = model
        cls.projector_names = [name for name, _ in model.named_parameters() if name.startswith(PROJECTOR_PREFIX)]
        cls.language_names = [name for name, _ in model.named_parameters() if name.startswith(LANGUAGE_MODEL_PREFIXES)]
        cls.args = parse_args(argv = ARGV + FULL + MUON)

    def test_qwen35_2b_groups(self):
        model = self.model
        self.assertTrue(model.lm_head.weight is model.model.language_model.embed_tokens.weight)
        settings = muon.settings_from_args(
            args = self.args,
            model = model,
            projector_names = self.projector_names,
            language_model_names = self.language_names,
        )
        kinds = muon.classify(model = model, names = self.projector_names + self.language_names)
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        # 每个可训练参数恰好属于一组，冻结的参数（音频塔、视觉塔）不在其中。
        self.assertEqual(set(kinds), trainable)
        self.assertEqual(settings.names, {name for name, kind in kinds.items() if kind == muon.HIDDEN_MATRIX})
        self.assertFalse(any(name.startswith(("model.audio_tower.", "model.visual.")) for name in kinds))
        parameters = dict(model.named_parameters())
        for name in settings.names:
            self.assertTrue(name.endswith(MUON_SUFFIXES), msg = name)
            self.assertEqual(parameters[name].ndim, 2, msg = name)
        for name in trainable - settings.names:
            parameter = parameters[name]
            if kinds[name] == "embedding":
                self.assertEqual(name, "model.language_model.embed_tokens.weight")
            elif kinds[name] == "conv":
                self.assertTrue(name.endswith("linear_attn.conv1d.weight") and parameter.ndim == 3, msg = name)
            else:
                self.assertEqual(kinds[name], "vector", msg = name)
                self.assertEqual(parameter.ndim, 1, msg = name)
        self.assertEqual(settings.projector_mask, (True, False, True, False))
        info = muon.describe(
            named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad],
            kinds = kinds,
            settings = settings,
            projector_names = self.projector_names,
            options = precision_options(),
        )
        self.assertEqual({key: (value["tensors"], value["parameters"]) for key, value in info["groups"].items()}, SIZES_2B)
        self.assertEqual(info["adamw_kinds"]["embedding"], {"tensors": 1, "parameters": 248320 * 2048})
        self.assertEqual(info["adamw_kinds"]["conv"]["tensors"], 18)
        self.assertEqual(info["bytes_per_parameter"]["muon"]["total"], 14.0)
        self.assertAlmostEqual(info["gb"], (1372717056 * 14 + 509108032 * 18) / 1e9, places = 2)

    def test_projector_only_groups(self):
        args = parse_args(argv = ARGV + MUON)
        settings = muon.settings_from_args(
            args = args,
            model = self.model,
            projector_names = self.projector_names,
            language_model_names = [],
        )
        self.assertEqual(sorted(settings.names), ["model.audio_projector.linear_1.weight", "model.audio_projector.linear_2.weight"])
        self.assertIsNone(muon.settings_from_args(
            args = parse_args(argv = ARGV),
            model = self.model,
            projector_names = self.projector_names,
            language_model_names = [],
        ))


def matrices(seed, shapes, dtype = torch.float32):
    generator = torch.Generator().manual_seed(seed)
    return [(0.1 * torch.randn(size = shape, generator = generator)).to(dtype = dtype) for shape in shapes]


SHAPES = ((24, 40), (40, 24), (16, 16))


def muon_kwargs(**changes):
    base = {
        "lr": 0.02,
        "weight_decay": 0.1,
        "momentum": 0.95,
        "nesterov": True,
        "ns_steps": 5,
        "adjust_lr_fn": "match_rms_adamw",
    }
    base.update(changes)
    return base


def run_steps(optimizer, params, grads_per_step, grad_dtype = None):
    """grad_dtype 与参数精度不同时，与 master.py 一样设置参数的 grad_dtype，再交给它这个精度的梯度。"""
    for grads in grads_per_step:
        for parameter, grad in zip(params, grads):
            dtype = grad_dtype or parameter.dtype
            if dtype != parameter.dtype:
                parameter.grad_dtype = dtype
            parameter.grad = grad.to(dtype = dtype).clone()
        optimizer.step()


class GroupStateMuonTest(unittest.TestCase):
    def setUp(self):
        self.grads = [matrices(seed = 10 + step, shapes = SHAPES) for step in range(3)]

    def reference(self, kwargs, rounded = False):
        """torch.optim.Muon 在 float32 中的结果；rounded 为真时每步之后把权重就近舍入到 bfloat16（模拟直接更新 bfloat16 参数）。"""
        params = [torch.nn.Parameter(value) for value in matrices(seed = 0, shapes = SHAPES)]
        if rounded:
            with torch.no_grad():
                for parameter in params:
                    parameter.copy_(parameter.bfloat16())
        optimizer = torch.optim.Muon(params = params, **kwargs)
        history = []
        for grads in self.grads:
            for parameter, grad in zip(params, grads):
                parameter.grad = grad.clone()
            optimizer.step()
            if rounded:
                with torch.no_grad():
                    for parameter in params:
                        parameter.copy_(parameter.bfloat16())
            history.append([parameter.detach().clone() for parameter in params])
        return history, optimizer

    def test_float32_matches_torch(self):
        for kwargs in (muon_kwargs(), muon_kwargs(adjust_lr_fn = "original", nesterov = False, weight_decay = 0.0)):
            history, _ = self.reference(kwargs = kwargs)
            params = [torch.nn.Parameter(value) for value in matrices(seed = 0, shapes = SHAPES)]
            optimizer = muon.GroupStateMuon(params = params, **kwargs)
            run_steps(optimizer = optimizer, params = params, grads_per_step = self.grads)
            for want, got in zip(history[-1], params):
                self.assertTrue(torch.equal(input = want, other = got.detach()), msg = kwargs)

    def test_bf16_grads_on_float32_master(self):
        """主权重 float32、梯度 bfloat16：与 torch 用转成 float32 的同一份梯度逐位相同。"""
        self.grads = [[grad.bfloat16().float() for grad in grads] for grads in self.grads]
        history, _ = self.reference(kwargs = muon_kwargs())
        params = [torch.nn.Parameter(value) for value in matrices(seed = 0, shapes = SHAPES)]
        optimizer = muon.GroupStateMuon(params = params, **muon_kwargs())
        run_steps(
            optimizer = optimizer,
            params = params,
            grads_per_step = self.grads,
            grad_dtype = torch.bfloat16,
        )
        for want, got in zip(history[-1], params):
            self.assertTrue(torch.equal(input = want, other = got.detach()))
        self.assertEqual(optimizer.state[params[0]]["momentum_buffer"].dtype, torch.float32)

    def test_bf16_momentum(self):
        """bfloat16 动量：按存储精度舍入；第一步的更新用未舍入的 float32 动量，与 torch 逐位相同，之后在 bfloat16 的误差以内。"""
        history, reference = self.reference(kwargs = muon_kwargs())
        params = [torch.nn.Parameter(value) for value in matrices(seed = 0, shapes = SHAPES)]
        optimizer = muon.GroupStateMuon(params = [{"params": params, "momentum_dtype": "bf16"}], **muon_kwargs())
        run_steps(optimizer = optimizer, params = params, grads_per_step = self.grads[:1])
        for want, got in zip(history[0], params):
            self.assertTrue(torch.equal(input = want, other = got.detach()))
        buffer = optimizer.state[params[0]]["momentum_buffer"]
        self.assertEqual(buffer.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(input = buffer, other = (0.05 * self.grads[0][0]).bfloat16()))
        run_steps(optimizer = optimizer, params = params, grads_per_step = self.grads[1:])
        for want, got in zip(history[-1], params):
            torch.testing.assert_close(
                actual = got.detach(),
                expected = want,
                rtol = 0,
                atol = 2e-3,
            )

    def test_bf16_parameters_nearest(self):
        """直接更新 bfloat16 参数、float32 梯度和动量、就近舍入：等于 float32 中算出后舍入。"""
        history, _ = self.reference(kwargs = muon_kwargs(), rounded = True)
        params = [torch.nn.Parameter(value) for value in matrices(seed = 0, shapes = SHAPES, dtype = torch.bfloat16)]
        optimizer = muon.GroupStateMuon(params = [{"params": params, "momentum_dtype": "fp32"}], **muon_kwargs())
        run_steps(
            optimizer = optimizer,
            params = params,
            grads_per_step = self.grads,
            grad_dtype = torch.float32,
        )
        for want, got in zip(history[-1], params):
            self.assertEqual(got.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(input = want.bfloat16(), other = got.detach()))

    def test_pure_bf16_uses_torch(self):
        """bfloat16 参数、梯度、动量且就近舍入：直接用 torch 的 Muon 计算，与 torch.optim.Muon 逐位相同。"""
        kwargs = muon_kwargs()
        expected = [torch.nn.Parameter(value) for value in matrices(seed = 0, shapes = SHAPES, dtype = torch.bfloat16)]
        run_steps(optimizer = torch.optim.Muon(params = expected, **kwargs), params = expected, grads_per_step = self.grads)
        params = [torch.nn.Parameter(value) for value in matrices(seed = 0, shapes = SHAPES, dtype = torch.bfloat16)]
        optimizer = muon.GroupStateMuon(params = [{"params": params, "momentum_dtype": "bf16"}], **kwargs)
        run_steps(optimizer = optimizer, params = params, grads_per_step = self.grads)
        for want, got in zip(expected, params):
            self.assertTrue(torch.equal(input = want.detach(), other = got.detach()))

    @unittest.skipUnless(HAS_TORCHAO, "需要 torchao")
    def test_stochastic_rounding(self):
        """更新远小于 bfloat16 的间距时，就近舍入几乎不动；随机舍入取上下两个相邻的 bfloat16 数之一，平均等于 float32 的结果。"""
        shape = (64, 1024)
        start = (1.0 + 0.01 * torch.randn(size = shape, generator = torch.Generator().manual_seed(0))).bfloat16()
        grad = torch.randn(size = shape, generator = torch.Generator().manual_seed(1))
        kwargs = muon_kwargs(lr = 4e-3, weight_decay = 0.0)
        exact = torch.nn.Parameter(start.float())
        exact.grad = grad.clone()
        torch.optim.Muon(params = [exact], **kwargs).step()
        results = {}
        for stochastic in (False, True):
            torch.manual_seed(seed = 5)
            parameter = torch.nn.Parameter(start.clone())
            parameter.grad_dtype = torch.float32
            parameter.grad = grad.clone()
            muon.GroupStateMuon(params = [{"params": [parameter], "stochastic": stochastic}], **kwargs).step()
            results[stochastic] = parameter.detach().float()
        change = (exact.detach() - start.float()).abs().mean()
        nearest_moved = (results[False] != start.float()).float().mean()
        stochastic_moved = (results[True] != start.float()).float().mean()
        self.assertLess(float(nearest_moved), 0.01)
        self.assertGreater(float(stochastic_moved), 0.05)
        # 每个元素都是 float32 结果的上下两个相邻 bfloat16 数之一。
        spacing = torch.finfo(torch.bfloat16).eps * 2
        self.assertTrue(bool(((results[True] - exact.detach()).abs() < spacing).all()))
        bias = (results[True] - exact.detach()).mean().abs()
        self.assertLess(float(bias), 0.1 * float(change))

    def test_resume_keeps_momentum_dtype(self):
        for param_dtype, momentum_dtype in ((torch.float32, "bf16"), (torch.bfloat16, "fp32")):
            def build():
                params = [torch.nn.Parameter(value) for value in matrices(seed = 0, shapes = SHAPES, dtype = param_dtype)]
                optimizer = muon.GroupStateMuon(params = [{"params": params, "momentum_dtype": momentum_dtype}], **muon_kwargs())
                return params, optimizer

            straight, optimizer = build()
            run_steps(
                optimizer = optimizer,
                params = straight,
                grads_per_step = self.grads,
                grad_dtype = torch.float32,
            )
            params, optimizer = build()
            run_steps(
                optimizer = optimizer,
                params = params,
                grads_per_step = self.grads[:2],
                grad_dtype = torch.float32,
            )
            buffer = io.BytesIO()
            torch.save(obj = {"params": [parameter.detach() for parameter in params], "optimizer": optimizer.state_dict()}, f = buffer)
            buffer.seek(0)
            saved = torch.load(f = buffer, weights_only = False)
            resumed, optimizer = build()
            with torch.no_grad():
                for parameter, value in zip(resumed, saved["params"]):
                    parameter.copy_(value)
            optimizer.load_state_dict(state_dict = saved["optimizer"])
            self.assertEqual(optimizer.state[resumed[0]]["momentum_buffer"].dtype, muon.MOMENTUM_DTYPES[momentum_dtype])
            run_steps(
                optimizer = optimizer,
                params = resumed,
                grads_per_step = self.grads[2:],
                grad_dtype = torch.float32,
            )
            for want, got in zip(straight, resumed):
                self.assertTrue(torch.equal(input = want.detach(), other = got.detach()), msg = (param_dtype, momentum_dtype))

    def test_rejects_non_matrices(self):
        with self.assertRaises(ValueError):
            muon.GroupStateMuon(params = [torch.nn.Parameter(torch.zeros(size = (4,)))], **muon_kwargs())


class MuonAdamWTest(unittest.TestCase):
    def test_groups_schedule_and_state_format(self):
        weight = torch.nn.Parameter(torch.randn(size = (8, 6)))
        bias = torch.nn.Parameter(torch.zeros(size = (8,)))
        optimizer = muon.MuonAdamW(
            adamw = torch.optim.AdamW(params = [bias], lr = 1e-3),
            muon = torch.optim.Muon(params = [weight], lr = 2e-3),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer = optimizer, lr_lambda = lambda step: 0.5 ** step)
        self.assertEqual(scheduler.base_lrs, [1e-3, 2e-3])
        weight.grad = torch.ones_like(input = weight)
        bias.grad = torch.ones_like(input = bias)
        optimizer.step()
        scheduler.step()
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [5e-4, 1e-3])
        state = optimizer.state_dict()
        self.assertEqual(sorted(state), ["adamw", "muon"])
        self.assertIn("momentum_buffer", state["muon"]["state"][0])
        with self.assertRaisesRegex(ValueError, "--optimizer muon"):
            optimizer.load_state_dict(state_dict = state["adamw"])
        optimizer.load_state_dict(state_dict = state)
        optimizer.zero_grad()
        self.assertIsNone(weight.grad)


# 损失只依赖参数：每个参数趋向一个随步数变化的随机目标，梯度为 p − c·target。
TARGET_SCALES = (0.1, -0.05, 0.08, 0.02)


def synthetic_loss(named, targets, step):
    total = torch.zeros(size = ())
    for name, parameter in named:
        if name not in targets:
            targets[name] = torch.randn(size = parameter.shape, generator = torch.Generator().manual_seed(len(targets)))
        total = total + 0.5 * ((parameter.float() - TARGET_SCALES[step % 4] * targets[name]) ** 2).sum()
    return total


def lr_lambda(step):
    return min(1.0, (step + 1) / 2)


@unittest.skipUnless(HAS_CONFIG, "需要阶段 1 模型目录中的代码与配置")
class TinyModelTest(unittest.TestCase):
    """缩小的 Qwen3.5 结构（语言模型 4 层、词表 1024）：只训练 projector 与全量微调。"""

    def model(self, train_language_model):
        from pipeline.train.model import LANGUAGE_MODEL_PREFIXES, PROJECTOR_PREFIX

        model = tiny_model(dtype = torch.bfloat16, device = "cpu", vocab_size = 1024)
        for name, parameter in model.named_parameters():
            if train_language_model and name.startswith(LANGUAGE_MODEL_PREFIXES):
                parameter.requires_grad_(requires_grad = True)
        projector = [(name, parameter) for name, parameter in model.named_parameters() if name.startswith(PROJECTOR_PREFIX)]
        language = [(name, parameter) for name, parameter in model.named_parameters() if name.startswith(LANGUAGE_MODEL_PREFIXES)]
        return model, projector, language if train_language_model else []

    def settings(self, model, projector, language, momentum_dtype = "fp32"):
        argv = ARGV + MUON + ["--weight-decay", str(WEIGHT_DECAY)]
        if language:
            argv += FULL + ["--muon-momentum-dtype", momentum_dtype]
        return muon.settings_from_args(
            args = parse_args(argv = argv),
            model = model,
            projector_names = [name for name, _ in projector],
            language_model_names = [name for name, _ in language],
        )

    # 只训练 projector

    def test_projector_only_matches_reference(self):
        model, projector, _ = self.model(train_language_model = False)
        settings = self.settings(model = model, projector = projector, language = [])
        params = [parameter for _, parameter in projector]
        optimizer = muon.projector_optimizer(
            params = params,
            settings = settings,
            lr = LR,
            weight_decay = WEIGHT_DECAY,
        )
        self.assertEqual([len(group["params"]) for group in optimizer.param_groups], [2, 2])
        got = self.train_projector(optimizer = optimizer, projector = projector, steps = 3)

        model, projector, _ = self.model(train_language_model = False)
        weights = [parameter for name, parameter in projector if name.endswith("weight")]
        biases = [parameter for name, parameter in projector if name.endswith("bias")]
        reference = muon.MuonAdamW(
            adamw = torch.optim.AdamW(
                params = biases,
                lr = LR,
                betas = (0.9, 0.999),
                weight_decay = WEIGHT_DECAY,
            ),
            muon = torch.optim.Muon(
                params = weights,
                lr = LR,
                weight_decay = WEIGHT_DECAY,
                adjust_lr_fn = "match_rms_adamw",
            ),
        )
        want = self.train_projector(optimizer = reference, projector = projector, steps = 3)
        for name in want:
            self.assertTrue(torch.equal(input = want[name], other = got[name]), msg = name)

    def train_projector(self, optimizer, projector, steps, start = 0, scheduler = None):
        scheduler = scheduler or torch.optim.lr_scheduler.LambdaLR(optimizer = optimizer, lr_lambda = lr_lambda)
        targets = {}
        for step in range(start, start + steps):
            synthetic_loss(named = projector, targets = targets, step = step).backward()
            torch.nn.utils.clip_grad_norm_(parameters = [parameter for _, parameter in projector], max_norm = 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none = True)
        return {name: parameter.detach().clone() for name, parameter in projector}

    def test_projector_only_resume(self):
        def build():
            model, projector, _ = self.model(train_language_model = False)
            settings = self.settings(model = model, projector = projector, language = [])
            optimizer = muon.projector_optimizer(
                params = [parameter for _, parameter in projector],
                settings = settings,
                lr = LR,
                weight_decay = WEIGHT_DECAY,
            )
            return projector, optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer = optimizer, lr_lambda = lr_lambda)

        projector, optimizer, scheduler = build()
        straight = self.train_projector(
            optimizer = optimizer,
            projector = projector,
            steps = 3,
            scheduler = scheduler,
        )
        projector, optimizer, scheduler = build()
        self.train_projector(
            optimizer = optimizer,
            projector = projector,
            steps = 2,
            scheduler = scheduler,
        )
        saved = roundtrip(obj = {
            "projector": {name: parameter.detach() for name, parameter in projector},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        })
        projector, optimizer, scheduler = build()
        with torch.no_grad():
            for name, parameter in projector:
                parameter.copy_(saved["projector"][name])
        optimizer.load_state_dict(state_dict = saved["optimizer"])
        scheduler.load_state_dict(state_dict = saved["scheduler"])
        resumed = self.train_projector(
            optimizer = optimizer,
            projector = projector,
            steps = 1,
            start = 2,
            scheduler = scheduler,
        )
        for name in straight:
            self.assertTrue(torch.equal(input = straight[name], other = resumed[name]), msg = name)

    # 全量微调（placement.LanguageModelUpdate，只在显存中的那条路径，这里在 CPU 上运行）

    def update(self, choice, momentum_dtype = "fp32", state = None):
        from pipeline.train.placement import LanguageModelUpdate

        model, projector, language = self.model(train_language_model = True)
        settings = self.settings(
            model = model,
            projector = projector,
            language = language,
            momentum_dtype = momentum_dtype,
        )
        update = LanguageModelUpdate(
            named_parameters = language,
            projector_params = [parameter for _, parameter in projector],
            options = choice,
            lr = LR,
            language_model_lr = LANGUAGE_MODEL_LR,
            weight_decay = WEIGHT_DECAY,
            applied = None,
            seed = 0,
            state = state,
            muon_settings = settings,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer = update.optimizer, lr_lambda = lr_lambda)
        return model, projector + language, update, scheduler, settings

    def train_update(self, named, update, scheduler, steps, start = 0):
        targets = {}
        for step in range(start, start + steps):
            synthetic_loss(named = named, targets = targets, step = step).backward()
            self.assertEqual(update.without_grad(), [])
            update.clip_grad_norm(max_norm = 1.0)
            update.optimizer.step()
            update.copy_to_model()
            scheduler.step()
            update.optimizer.zero_grad()
        return {name: parameter.detach().clone() for name, parameter in named}

    def test_full_finetune_matches_reference(self):
        """默认精度（float32 主权重、梯度、状态和动量）：与手写的参照逐位相同。参照：float32 主权重；torch 的 fused AdamW
        两组（projector 的偏置、语言模型的词嵌入等）加 torch.optim.Muon 两组（projector 与语言模型的二维隐藏层矩阵）。"""
        model, named, update, scheduler, settings = self.update(choice = precision_options())
        groups = update.optimizer.param_groups
        self.assertEqual(len(groups), 4)
        counted = sum(len(group["params"]) for group in groups)
        self.assertEqual(counted, len(named))
        muon_tensors = {id(tensor) for group in groups[2:] for tensor in group["params"]}
        for name, master in zip(update.weights.names, update.weights.masters):
            self.assertEqual(id(master) in muon_tensors, name in settings.names, msg = name)
        got = self.train_update(
            named = named,
            update = update,
            scheduler = scheduler,
            steps = 3,
        )
        self.assertEqual(scheduler.get_last_lr(), [LR, LANGUAGE_MODEL_LR, LR, LANGUAGE_MODEL_LR])

        model, projector, language = self.model(train_language_model = True)
        masters = {name: torch.nn.Parameter(parameter.detach().float().clone()) for name, parameter in language}
        projector_adamw = [parameter for name, parameter in projector if name not in settings.names]
        projector_muon = [parameter for name, parameter in projector if name in settings.names]
        adamw = torch.optim.AdamW(
            params = [
                {"params": projector_adamw, "lr": LR},
                {"params": [masters[name] for name, _ in language if name not in settings.names], "lr": LANGUAGE_MODEL_LR},
            ],
            lr = LR,
            betas = (0.9, 0.999),
            weight_decay = WEIGHT_DECAY,
            fused = True,
        )
        reference = torch.optim.Muon(
            params = [
                {"params": projector_muon, "lr": LR},
                {"params": [masters[name] for name, _ in language if name in settings.names], "lr": LANGUAGE_MODEL_LR},
            ],
            lr = LR,
            weight_decay = WEIGHT_DECAY,
            adjust_lr_fn = "match_rms_adamw",
        )
        schedulers = [torch.optim.lr_scheduler.LambdaLR(optimizer = item, lr_lambda = lr_lambda) for item in (adamw, reference)]
        targets = {}
        for step in range(3):
            synthetic_loss(named = projector + language, targets = targets, step = step).backward()
            for name, parameter in language:
                masters[name].grad = parameter.grad.float()
                parameter.grad = None
            torch.nn.utils.clip_grad_norm_(
                parameters = [parameter for _, parameter in projector] + [masters[name] for name, _ in language],
                max_norm = 1.0,
            )
            adamw.step()
            reference.step()
            with torch.no_grad():
                for name, parameter in language:
                    parameter.copy_(masters[name])
            for item in schedulers:
                item.step()
            adamw.zero_grad(set_to_none = True)
            reference.zero_grad(set_to_none = True)
        for name, parameter in projector + language:
            self.assertTrue(torch.equal(input = parameter.detach(), other = got[name]), msg = name)
        state = update.optimizer.state_dict()
        self.assertEqual(sorted(state), ["adamw", "muon"])
        self.assertEqual(len(state["muon"]["state"]), len(settings.names))

    COMBINATIONS = {
        "default": (precision_options(), "fp32"),
        "pure bf16": (precision_options(master = "none", grad = "bf16", states = "bf16"), "bf16"),
        "fp32 master, bf16 grads, 8bit AdamW states, bf16 momentum": (precision_options(grad = "bf16", states = "8bit"), "bf16"),
        "none, stochastic, bf16 momentum": (precision_options(master = "none", rounding = "stochastic"), "bf16"),
    }

    def test_full_finetune_resume(self):
        """停在第 2 步、保存、新建后载入，再训练 1 步，与不中断地训练 3 步逐位相同；动量保持保存时的精度。"""
        for label, (choice, momentum_dtype) in self.COMBINATIONS.items():
            if not HAS_TORCHAO and label != "default" and label != "pure bf16":
                continue
            torch.manual_seed(seed = 7)
            model, named, update, scheduler, _ = self.update(choice = choice, momentum_dtype = momentum_dtype)
            straight = self.train_update(
                named = named,
                update = update,
                scheduler = scheduler,
                steps = 3,
            )

            torch.manual_seed(seed = 7)
            model, named, update, scheduler, _ = self.update(choice = choice, momentum_dtype = momentum_dtype)
            self.train_update(
                named = named,
                update = update,
                scheduler = scheduler,
                steps = 2,
            )
            saved = roundtrip(obj = {
                "weights": update.weights_state(),
                "optimizer": update.optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "projector": {name: parameter.detach() for name, parameter in named if name.startswith("model.audio_projector.")},
                "rng": torch.get_rng_state(),
            })
            for entry in saved["optimizer"]["muon"]["state"].values():
                self.assertIn(entry["momentum_buffer"].dtype, (torch.float32, muon.MOMENTUM_DTYPES[momentum_dtype]))
            model, named, update, scheduler, _ = self.update(choice = choice, momentum_dtype = momentum_dtype, state = saved["weights"])
            with torch.no_grad():
                for name, parameter in named:
                    if name in saved["projector"]:
                        parameter.copy_(saved["projector"][name])
            update.optimizer.load_state_dict(state_dict = saved["optimizer"])
            scheduler.load_state_dict(state_dict = saved["scheduler"])
            torch.set_rng_state(new_state = saved["rng"])
            language_muon = update.gpu.muon.param_groups[1]
            for tensor in language_muon["params"]:
                self.assertEqual(update.gpu.muon.state[tensor]["momentum_buffer"].dtype, muon.MOMENTUM_DTYPES[momentum_dtype], msg = label)
            resumed = self.train_update(
                named = named,
                update = update,
                scheduler = scheduler,
                steps = 1,
                start = 2,
            )
            for name in straight:
                self.assertTrue(torch.equal(input = straight[name], other = resumed[name]), msg = f"{label}: {name}")

    def test_adamw_path_unchanged(self):
        """--optimizer adamw（muon_settings 为 None）：与以前的组建方式（MasterWeights 加 precision.make_optimizer）逐位相同，
        续训状态仍是 torch 优化器的格式。"""
        from pipeline.train import precision
        from pipeline.train.master import MasterWeights
        from pipeline.train.placement import LanguageModelUpdate

        results = []
        for build_directly in (False, True):
            model, projector, language = self.model(train_language_model = True)
            projector_params = [parameter for _, parameter in projector]
            if build_directly:
                weights = MasterWeights(named_parameters = language)
                optimizer = precision.make_optimizer(
                    projector_params = projector_params,
                    language_model_params = weights.optimizer_params(),
                    lr = LR,
                    language_model_lr = LANGUAGE_MODEL_LR,
                    weight_decay = WEIGHT_DECAY,
                    options = precision_options(),
                )
                clip = projector_params + weights.optimizer_params()
            else:
                update = LanguageModelUpdate(
                    named_parameters = language,
                    projector_params = projector_params,
                    options = precision_options(),
                    lr = LR,
                    language_model_lr = LANGUAGE_MODEL_LR,
                    weight_decay = WEIGHT_DECAY,
                    applied = None,
                )
                weights, optimizer = update.weights, update.optimizer
                clip = projector_params + weights.optimizer_params()
            targets = {}
            for step in range(2):
                synthetic_loss(named = projector + language, targets = targets, step = step).backward()
                torch.nn.utils.clip_grad_norm_(parameters = clip, max_norm = 1.0)
                optimizer.step()
                weights.copy_to_model()
                optimizer.zero_grad(set_to_none = True)
            state = optimizer.state_dict()
            self.assertEqual(sorted(state), ["param_groups", "state"])
            results.append(({name: parameter.detach().clone() for name, parameter in projector + language}, state))
        (first, first_state), (second, second_state) = results
        for name in first:
            self.assertTrue(torch.equal(input = first[name], other = second[name]), msg = name)
        self.assertEqual(sorted(first_state["state"]), sorted(second_state["state"]))

    def test_rejects_adamw_checkpoint(self):
        model, named, update, scheduler, _ = self.update(choice = precision_options())
        adamw_only = update.optimizer.state_dict()["adamw"]
        with self.assertRaisesRegex(ValueError, "--optimizer muon"):
            update.optimizer.load_state_dict(state_dict = adamw_only)


def roundtrip(obj):
    buffer = io.BytesIO()
    torch.save(obj = obj, f = buffer)
    buffer.seek(0)
    return torch.load(f = buffer, weights_only = False)


@unittest.skipUnless(CUDA and HAS_TORCHAO and HAS_CONFIG, "需要 CUDA、torchao 和阶段 1 模型目录中的代码与配置")
class OffloadTest(unittest.TestCase):
    """--offload-optimizer 加 Muon：AdamW 部分中元素不少于 4096 个的语言模型参数（词嵌入等）卸载到主机，Muon 部分留在显存中；
    真实前向（缩小的模型）。与只在显存中训练的 Muon 第一步的损失相同，之后在 CPU / GPU 上 AdamW 的舍入差别以内；
    续训状态的格式相同，可以互相续训。"""

    @classmethod
    def setUpClass(cls):
        cls.batch = tiny_batch(audio_windows = (50, 800))

    def build(self, offload, state = None):
        from pipeline.train.model import LANGUAGE_MODEL_PREFIXES, PROJECTOR_PREFIX, language_model_parameters, move_batch, projector_parameters
        from pipeline.train.optimize import apply
        from pipeline.train.placement import LanguageModelUpdate
        from pipeline.train.tests.test_placement import LanguageModelUpdateTest

        model = tiny_model(dtype = torch.bfloat16)
        for name, parameter in model.named_parameters():
            if name.startswith(LANGUAGE_MODEL_PREFIXES):
                parameter.requires_grad_(requires_grad = True)
        options = LanguageModelUpdateTest.options(self)
        options["offload_optimizer"] = offload
        problem = muon.stream_problem(optimizer = "muon", train_language_model = True)
        applied = apply(
            model = model,
            options = options,
            train_language_model = True,
            stream_problem = problem,
        )
        projector_names = [name for name, _ in model.named_parameters() if name.startswith(PROJECTOR_PREFIX)]
        language = language_model_parameters(model = model)
        settings = muon.settings_from_args(
            args = parse_args(argv = ARGV + FULL + MUON),
            model = model,
            projector_names = projector_names,
            language_model_names = [name for name, _ in language],
        )
        update = LanguageModelUpdate(
            named_parameters = language,
            projector_params = projector_parameters(model = model),
            options = precision_options(),
            lr = LR,
            language_model_lr = LANGUAGE_MODEL_LR,
            weight_decay = 0.0,
            applied = applied,
            offload_optimizer = offload,
            seed = 0,
            state = state,
            muon_settings = settings,
        )
        inputs = move_batch(batch = self.batch, device = "cuda", feature_dtype = model.model.audio_tower.dtype)
        return model, applied, update, inputs, settings

    def train(self, model, applied, update, inputs, steps):
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

    def test_offload_matches_gpu(self):
        model, applied, update, inputs, settings = self.build(offload = False)
        reference = self.train(
            model = model,
            applied = applied,
            update = update,
            inputs = inputs,
            steps = 2,
        )
        reference_weights = update.model_state()
        reference_state = update.optimizer.state_dict()
        model, applied, update, inputs, settings = self.build(offload = True)
        self.assertTrue(update.offloaded_named)
        for name, _ in update.offloaded_named:
            self.assertNotIn(name, settings.names)
        losses, norms = self.train(
            model = model,
            applied = applied,
            update = update,
            inputs = inputs,
            steps = 2,
        )
        self.assertEqual(losses[0], reference[0][0])
        self.assertAlmostEqual(norms[0], reference[1][0], delta = 1e-5 * reference[1][0])
        weights = update.model_state()
        for name in reference_weights:
            gap = (reference_weights[name].float() - weights[name].float()).abs()
            spacing = torch.finfo(torch.bfloat16).eps * reference_weights[name].float().abs().clamp(min = 1e-2)
            self.assertTrue(bool((gap <= 4 * spacing).all()), msg = name)
        state = update.optimizer.state_dict()
        self.assertEqual(sorted(state), ["adamw", "muon"])
        self.assertEqual(sorted(state["adamw"]["state"]), sorted(reference_state["adamw"]["state"]))
        self.assertEqual(sorted(state["muon"]["state"]), sorted(reference_state["muon"]["state"]))
        # 只在显存中训练的状态可以载入卸载优化器的训练，反过来也可以。
        saved = roundtrip(obj = {"weights": update.weights_state(), "optimizer": state})
        for offload in (False, True):
            model, applied, update, inputs, settings = self.build(offload = offload, state = saved["weights"])
            update.optimizer.load_state_dict(state_dict = saved["optimizer"])
            self.train(
                model = model,
                applied = applied,
                update = update,
                inputs = inputs,
                steps = 1,
            )


if __name__ == "__main__":
    unittest.main()
