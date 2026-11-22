"""Muon 优化器（任务 #78，train.py 的 --optimizer muon）。

Muon 只更新网络隐藏层的二维权重矩阵：每个矩阵维护一份动量，把（Nesterov）动量用 Newton–Schulz 迭代近似正交化，
也就是把矩阵的奇异值都拉到 1 附近，再用这个矩阵更新权重。这里直接用 PyTorch 2.13 自带的 torch.optim.Muon，
它的一步是（g 为梯度，μ 为 --muon-momentum）：

    buf ← buf + (1 − μ)·(g − buf)                 # 即 μ·buf + (1 − μ)·g，torch 用 lerp 实现
    u   ← g + μ·(buf − g)（Nesterov，默认）或 buf
    O   ← Newton–Schulz(u)                        # bfloat16 中迭代 --muon-ns-steps 次，系数 (3.4445, −4.7750, 2.0315)
    w   ← w·(1 − lr·weight_decay) − lr·s·O        # s 由 --muon-lr-scaling 决定，见下文

它只接受二维参数（构造时检查），每个参数只有一个状态 momentum_buffer。

参数分组（classify）
====================

可训练参数中，nn.Linear 的二维权重（词嵌入和输出层除外）用 Muon，其余全部用 AdamW，AdamW 的设置与只用 AdamW 时相同
（--lr、--language-model-lr、--weight-decay、精度选项）。冻结的参数（音频塔、视觉塔，只训练 projector 时的语言模型）
不进入优化器。按 Qwen3.5 的结构：

- projector：linear_1.weight、linear_2.weight 用 Muon，两个偏置用 AdamW。
- 语言模型，用 Muon：线性注意力层（Gated DeltaNet）的 in_proj_qkv、in_proj_z、in_proj_b、in_proj_a、out_proj，全注意力层的
  q_proj、k_proj、v_proj、o_proj，各层 MLP 的 gate_proj、up_proj、down_proj。
- 语言模型，用 AdamW：词嵌入（与输出层共用一个权重，tie_word_embeddings；不共用时输出层的权重也归 AdamW），各归一化层的
  权重、线性注意力的 A_log 和 dt_bias（一维），短卷积 conv1d 的权重（三维，(通道, 1, 4)）。

几个矩阵是拼起来的：in_proj_qkv 是 Q、K、V 三个投影上下拼成的一个矩阵（2B 为 6144 × 2048），全注意力层的 q_proj 是查询
和输出门（attn_output_gate）拼成的矩阵（2B 为 4096 × 2048）。Muon 把拼起来的矩阵当作一个整体正交化；Keller Jordan 的
说明提到对 Q、K、V 分别正交化效果更好，这里沿用 torch 的实现，没有拆开。in_proj_b、in_proj_a 每个头一行（2B 为 16 × 2048），
也按二维矩阵用 Muon。训练开始时打印各组的张量个数和参数个数（describe）。

学习率（--muon-lr-scaling，即 torch 的 adjust_lr_fn）
====================================================

A、B 为矩阵的行数和列数。正交化后的 O 的均方根约为 1/√max(A, B)，两种缩放：

- match_rms_adamw（默认）：s = 0.2·√max(A, B)，更新的均方根约为 0.2·lr，与 AdamW 更新的均方根（通常 0.2–0.4 倍学习率）
  相当，可以沿用为 AdamW 调好的学习率和权重衰减（Moonshot 的做法）。--muon-lr、--language-model-muon-lr 不给时，Muon
  两组的峰值学习率就是 --lr、--language-model-lr。
- original：s = √max(1, A/B)（Keller Jordan 的写法），更新的均方根约为 lr/√B（2B 中输入为 2048 维的矩阵约 0.022·lr），
  学习率的量级与 AdamW 不同，所以必须给出 --muon-lr（训练语言模型时还有 --language-model-muon-lr）。

权重衰减乘的是未缩放的学习率（torch 的实现，与 Moonshot 论文的式 (4) 相同）。--muon-weight-decay 不给时等于 --weight-decay。
Muon 各组与 AdamW 各组按同一条预热加余弦的曲线变化，各组的学习率都是本组峰值乘以同一个系数。

精度（GroupStateMuon）
=====================

projector 的 Muon 组始终是 float32 权重、float32 梯度、float32 动量。语言模型的 Muon 组与 AdamW 组一样受精度选项控制
（precision.py）：优化器更新 float32 主权重或直接更新 bfloat16 参数（--language-model-master-weights），梯度为 fp32 或 bf16
（--language-model-grad-dtype），没有主权重时就近或随机舍入（--language-model-rounding）。动量的存储由
--muon-momentum-dtype 决定（fp32 或 bf16，只作用于语言模型的 Muon 组）；--language-model-optimizer-states 只决定 AdamW 部分
（词嵌入等）的状态格式，8bit、4bit、fp8 照常可用。

权重、梯度、动量是同一种精度且不随机舍入时（默认组合，以及 none + bf16 梯度 + bf16 动量 + 就近舍入），直接调用 torch 的
Muon 计算（torch.optim._muon.muon），与 torch.optim.Muon 逐位相同。其余组合逐个张量在 float32 中计算：梯度转成 float32，
动量在 float32 中更新后按存储精度写回（就近舍入），Nesterov 的插值用更新后未舍入的 float32 动量（与 torchao 的 AdamW 保留
float32 副本用于更新的做法相同），正交化照样在 bfloat16 中进行；新的权重在 float32 中算出，float32 参数直接写入，bfloat16
参数按设置就近或随机舍入（torchao 的 _fp32_to_bf16_sr，随机数来自参数所在设备的默认生成器，CUDA 上随 trainer_state.pt 保存
和恢复）。bfloat16 动量：μ = 0.95 时每步变化 5%，比 bfloat16 的相对舍入间距（0.4%–0.8%）大得多，AdamW 二阶矩那样被舍掉
而停住的问题（pipeline/train/__init__.py「全量微调」中的 e）只在梯度几乎不变、动量已经接近梯度时才会出现。

续训时 torch 的 Optimizer.load_state_dict 会把动量转成参数的精度；GroupStateMuon.load_state_dict 按保存时的精度放回。

与其他选项的组合
================

- 只训练 projector：AdamW（两个偏置，--lr）加 Muon（两个权重矩阵）。
- 全量微调（--train-language-model）：AdamW 两组（projector、语言模型）加 Muon 两组（projector、语言模型）。
- 梯度裁剪（--max-grad-norm）照常作用于全部可训练参数的梯度，裁剪之后再交给两个优化器。
- --offload-optimizer：只卸载 AdamW 部分中元素不少于 4096 个的语言模型参数（词嵌入兼输出层、短卷积的权重），Muon 部分的
  主权重、梯度和动量留在显存中（Newton–Schulz 迭代是大矩阵乘法，在 CPU 上太慢）。
- --lm-blocks-to-swap（全量微调时流式加载可训练的层，参数在主机内存中由 CPU 更新）不支持：这些层的 Muon 参数也要在显存中
  正交化，没有实现。显存不足时的退让因此跳过「把语言模型的层改为流式加载」这一档。只训练 projector 时语言模型冻结，
  --lm-blocks-to-swap 照常可用。
- 拼接批次（--pack-sequences）、梯度检查点、激活值卸载、--stop-at-step、--grad-accum 与优化器无关，照常可用。

续训。--optimizer 和 --muon-momentum-dtype 改变优化器状态的格式，在 train.RESUME_KEYS 中；以前的训练配置中没有它们，按默认值
（adamw、fp32）核对。续训状态 trainer_state.pt 的 optimizer 为 {"adamw": AdamW 部分, "muon": Muon 部分}：AdamW 部分的格式
与只用 AdamW 时相同（全量微调时按 placement.py 的规范格式编号，只含 AdamW 的参数），Muon 部分是 torch 的 state_dict（projector
一组、语言模型一组，每个参数一个 momentum_buffer）。动量、Nesterov、迭代次数、缩放方式和权重衰减保存在 Muon 各组中，续训时
与学习率一样沿用检查点中的数值，命令行给出的不同时只打印提示。

显存
====

每个参数的字节数（bytes_per_parameter）：Muon 部分为 bfloat16 权重 2、主权重 4 或 0、梯度 4 或 2、动量 4 或 2；AdamW 部分与
precision.bytes_per_parameter 相同（两个状态）。默认精度下 Muon 部分每个参数 14 字节，AdamW 为 18 字节。表格与各尺寸的参数
个数见 pipeline/train/__init__.py「Muon 优化器」。

计算量。每个 A × B 矩阵每次迭代约 4m²n + 2m³ 次浮点运算（m = min(A, B)，n = max(A, B)），乘迭代次数。训练开始时打印每步的
总量（newton_schulz_tflop_per_step）。它与每步的 token 数无关：每步 token 少时（本项目每批 16 条、360 秒音频约八千个 token），
它与语言模型线性层的前向加反传（6 × 参数个数 × token 数）相当。
"""

import argparse
import dataclasses

import torch
from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz
from torch.optim._muon import muon as muon_update
from torch.optim.optimizer import _to_scalar

OPTIMIZERS = ("adamw", "muon")
LR_SCALINGS = ("match_rms_adamw", "original")
MOMENTUM_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
DEFAULTS = {
    "optimizer": "adamw",
    "muon_lr": None,
    "language_model_muon_lr": None,
    "muon_lr_scaling": "match_rms_adamw",
    "muon_momentum": 0.95,
    "muon_nesterov": True,
    "muon_ns_steps": 5,
    "muon_weight_decay": None,
    "muon_momentum_dtype": "fp32",
}
# 续训核对项（train.RESUME_KEYS 的一部分）：决定优化器状态的格式。其余几项保存在优化器的参数组中，续训时沿用检查点。
KEYS = ("optimizer", "muon_momentum_dtype")
# 参数的类别（classify）。只有 hidden_matrix 用 Muon。
HIDDEN_MATRIX = "hidden_matrix"
# Muon 参数组中保存、续训时沿用检查点的设置：参数组中的键 -> Settings 中的属性。
GROUP_SETTINGS = {
    "momentum": "momentum",
    "nesterov": "nesterov",
    "ns_steps": "ns_steps",
    "adjust_lr_fn": "lr_scaling",
    "weight_decay": "weight_decay",
}


def add_arguments(parser):
    group = parser.add_argument_group(title = "优化器（muon.py）")
    group.add_argument(
        "--optimizer",
        default = DEFAULTS["optimizer"],
        choices = OPTIMIZERS,
        help = "adamw：全部可训练参数用 AdamW（默认）；muon：二维隐藏层权重矩阵用 Muon，其余用 AdamW",
    )
    group.add_argument(
        "--muon-lr",
        type = float,
        default = DEFAULTS["muon_lr"],
        help = "projector 的 Muon 组的峰值学习率；不给时与 --lr 相同（只在 --muon-lr-scaling match_rms_adamw 时允许不给）",
    )
    group.add_argument(
        "--language-model-muon-lr",
        type = float,
        default = DEFAULTS["language_model_muon_lr"],
        help = "语言模型的 Muon 组的峰值学习率；不给时与 --language-model-lr 相同（同上）",
    )
    group.add_argument(
        "--muon-lr-scaling",
        default = DEFAULTS["muon_lr_scaling"],
        choices = LR_SCALINGS,
        help = (
            "按矩阵形状缩放 Muon 的更新（torch 的 adjust_lr_fn）：match_rms_adamw 乘 0.2·√max(行, 列)，更新幅度与 AdamW 相当，"
            "可沿用 AdamW 的学习率（默认）；original 乘 √max(1, 行/列)，需要另给 --muon-lr"
        ),
    )
    group.add_argument(
        "--muon-momentum",
        type = float,
        default = DEFAULTS["muon_momentum"],
        help = "Muon 的动量系数（默认 0.95）",
    )
    group.add_argument(
        "--muon-nesterov",
        action = argparse.BooleanOptionalAction,
        default = DEFAULTS["muon_nesterov"],
        help = "Nesterov 动量（默认打开，--no-muon-nesterov 关闭）",
    )
    group.add_argument(
        "--muon-ns-steps",
        type = int,
        default = DEFAULTS["muon_ns_steps"],
        help = "Newton–Schulz 迭代次数（默认 5）",
    )
    group.add_argument(
        "--muon-weight-decay",
        type = float,
        default = DEFAULTS["muon_weight_decay"],
        help = "Muon 组的权重衰减；不给时与 --weight-decay 相同",
    )
    group.add_argument(
        "--muon-momentum-dtype",
        default = DEFAULTS["muon_momentum_dtype"],
        choices = tuple(MOMENTUM_DTYPES),
        help = "语言模型的 Muon 组动量的存储精度：fp32（默认）或 bf16（需要 --train-language-model；projector 组始终 fp32）",
    )
    return group


def validate(args):
    """检查优化器选项的组合，不合法时抛出 ValueError（信息写明冲突的选项）。"""
    changed = [key for key in DEFAULTS if key != "optimizer" and getattr(args, key) != DEFAULTS[key]]
    if args.optimizer != "muon":
        if changed:
            flags = ", ".join("--" + ("no-muon-nesterov" if key == "muon_nesterov" else key.replace("_", "-")) for key in changed)
            raise ValueError(f"{flags} 只在 --optimizer muon 时使用")
        return
    if not 0.0 <= args.muon_momentum < 1.0:
        raise ValueError(f"--muon-momentum 必须在 [0, 1) 中（现在是 {args.muon_momentum}）")
    if not 1 <= args.muon_ns_steps < 100:
        raise ValueError(f"--muon-ns-steps 必须在 1 到 99 之间（现在是 {args.muon_ns_steps}，torch.optim.Muon 要求少于 100）")
    for key in ("muon_lr", "language_model_muon_lr", "muon_weight_decay"):
        value = getattr(args, key)
        if value is not None and value < 0:
            raise ValueError(f"--{key.replace('_', '-')} 不能为负（现在是 {value}）")
    if args.language_model_muon_lr is not None and not args.train_language_model:
        raise ValueError("--language-model-muon-lr 只作用于语言模型的 Muon 组，需要 --train-language-model")
    if args.muon_momentum_dtype != DEFAULTS["muon_momentum_dtype"] and not args.train_language_model:
        raise ValueError("--muon-momentum-dtype 只作用于语言模型的 Muon 组（projector 组始终 fp32），需要 --train-language-model")
    if args.muon_lr_scaling == "original":
        missing = ["--muon-lr"] if args.muon_lr is None else []
        if args.train_language_model and args.language_model_muon_lr is None:
            missing.append("--language-model-muon-lr")
        if missing:
            raise ValueError(
                f"--muon-lr-scaling original 时 Muon 更新的幅度与 AdamW 不同，不能沿用 --lr、--language-model-lr，"
                f"需要给出 {'、'.join(missing)}"
            )


def stream_problem(optimizer, train_language_model):
    """全量微调时语言模型的层不能流式加载（--lm-blocks-to-swap，参数在主机内存中更新）的原因；可以时返回 None。"""
    if optimizer == "muon" and train_language_model:
        return (
            "--optimizer muon 的 Newton–Schulz 迭代要在显存中进行，流式加载的层的参数在主机内存中由 CPU 更新，没有实现；"
            "可以改用 --offload-optimizer（只卸载 AdamW 部分）或梯度检查点"
        )
    return None


def classify(model, names):
    """names 中每个可训练参数的类别 {名称: 类别}：hidden_matrix（nn.Linear 的二维权重，用 Muon）、embedding（词嵌入、输出层）、
    vector（一维：归一化层的权重、偏置、A_log、dt_bias）、conv（三维及以上：短卷积的权重）、other（其余二维参数），后四类用 AdamW。"""
    embeddings = set()
    for module in (model.get_input_embeddings(), model.get_output_embeddings()):
        weight = getattr(module, "weight", None) if module is not None else None
        if weight is not None:
            embeddings.add(id(weight))
    linear = {id(module.weight) for module in model.modules() if isinstance(module, torch.nn.Linear)}
    parameters = dict(model.named_parameters())
    kinds = {}
    for name in names:
        parameter = parameters[name]
        if id(parameter) in embeddings:
            kinds[name] = "embedding"
        elif parameter.ndim <= 1:
            kinds[name] = "vector"
        elif parameter.ndim >= 3:
            kinds[name] = "conv"
        elif id(parameter) in linear:
            kinds[name] = HIDDEN_MATRIX
        else:
            kinds[name] = "other"
    return kinds


@dataclasses.dataclass(frozen = True)
class Settings:
    """Muon 部分的设置。names 是用 Muon 的参数名；projector_mask 与 model.projector_parameters 的顺序对应，真为 Muon。
    lr、language_model_lr 是两组 Muon 的峰值学习率（已按 --muon-lr、--lr 等解析）。"""

    names: frozenset
    projector_mask: tuple
    lr: float
    language_model_lr: float
    momentum: float
    nesterov: bool
    ns_steps: int
    lr_scaling: str
    weight_decay: float
    momentum_dtype: str


def settings_from_args(args, model, projector_names, language_model_names):
    """--optimizer muon 时的 Settings，否则返回 None。projector_names 与 model.projector_parameters 的顺序相同。"""
    if args.optimizer != "muon":
        return None
    kinds = classify(model = model, names = list(projector_names) + list(language_model_names))
    names = frozenset(name for name, kind in kinds.items() if kind == HIDDEN_MATRIX)
    return Settings(
        names = names,
        projector_mask = tuple(name in names for name in projector_names),
        lr = args.muon_lr if args.muon_lr is not None else args.lr,
        language_model_lr = (
            args.language_model_muon_lr if args.language_model_muon_lr is not None else args.language_model_lr
        ),
        momentum = args.muon_momentum,
        nesterov = args.muon_nesterov,
        ns_steps = args.muon_ns_steps,
        lr_scaling = args.muon_lr_scaling,
        weight_decay = args.muon_weight_decay if args.muon_weight_decay is not None else args.weight_decay,
        momentum_dtype = args.muon_momentum_dtype,
    )


def split(items, mask):
    """按 mask 把 items 分成 (AdamW 部分, Muon 部分)，各自保持原来的顺序。"""
    adamw = [item for item, is_muon in zip(items, mask, strict = True) if not is_muon]
    muon = [item for item, is_muon in zip(items, mask, strict = True) if is_muon]
    return adamw, muon


class GroupStateMuon(torch.optim.Muon):
    """torch.optim.Muon，参数组可另设 "momentum_dtype"（fp32、bf16，缺省 fp32）和 "stochastic"（bfloat16 参数是否随机舍入，
    缺省否）。权重、梯度、动量同一精度且不随机舍入的参数直接交给 torch 的 Muon 计算，其余逐个张量在 float32 中计算
    （见模块说明「精度」）。"""

    def __init__(
        self,
        params,
        lr,
        weight_decay,
        momentum,
        nesterov,
        ns_steps,
        adjust_lr_fn,
    ):
        super().__init__(
            params = params,
            lr = lr,
            weight_decay = weight_decay,
            momentum = momentum,
            nesterov = nesterov,
            ns_steps = ns_steps,
            adjust_lr_fn = adjust_lr_fn,
        )

    def add_param_group(self, param_group):
        dtype = param_group.setdefault("momentum_dtype", "fp32")
        if dtype not in MOMENTUM_DTYPES:
            raise ValueError(f"未知的动量精度 {dtype}")
        param_group.setdefault("stochastic", False)
        super().add_param_group(param_group = param_group)

    @torch.no_grad()
    def step(self, closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params, grads, buffers = [], [], []
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if torch.is_complex(input = parameter) or parameter.grad.is_sparse:
                    raise RuntimeError("Muon 不支持复数参数和稀疏梯度")
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        input = parameter,
                        dtype = MOMENTUM_DTYPES[group["momentum_dtype"]],
                        memory_format = torch.preserve_format,
                    )
                buffer = state["momentum_buffer"]
                stochastic = group["stochastic"] and parameter.dtype == torch.bfloat16
                if parameter.grad.dtype == buffer.dtype == parameter.dtype and not stochastic:
                    params.append(parameter)
                    grads.append(parameter.grad)
                    buffers.append(buffer)
                else:
                    self._mixed_update(
                        parameter = parameter,
                        buffer = buffer,
                        group = group,
                        stochastic = stochastic,
                    )
            if params:
                muon_update(
                    params = params,
                    grads = grads,
                    muon_momentum_bufs = buffers,
                    lr = group["lr"],
                    weight_decay = group["weight_decay"],
                    momentum = group["momentum"],
                    nesterov = group["nesterov"],
                    ns_coefficients = group["ns_coefficients"],
                    eps = group["eps"],
                    ns_steps = group["ns_steps"],
                    adjust_lr_fn = group["adjust_lr_fn"],
                    has_complex = False,
                )
        return loss

    @staticmethod
    def _mixed_update(parameter, buffer, group, stochastic):
        """一个参数的 Muon 更新，计算在 float32 中进行（正交化在 bfloat16 中，与 torch 相同）。"""
        lr = _to_scalar(group["lr"])
        momentum = group["momentum"]
        grad = parameter.grad.float()
        if buffer.dtype == torch.float32:
            buffer.lerp_(end = grad, weight = 1 - momentum)
            buffer_f32 = buffer
        else:
            buffer_f32 = buffer.float().lerp_(end = grad, weight = 1 - momentum)
            buffer.copy_(buffer_f32)
        update = grad.lerp(end = buffer_f32, weight = momentum) if group["nesterov"] else buffer_f32
        update = _zeropower_via_newtonschulz(
            grad = update,
            ns_coefficients = group["ns_coefficients"],
            ns_steps = group["ns_steps"],
            eps = group["eps"],
        )
        adjusted_lr = _adjust_lr(lr = lr, adjust_lr_fn = group["adjust_lr_fn"], param_shape = parameter.shape)
        if parameter.dtype == torch.float32:
            parameter.mul_(other = 1 - lr * group["weight_decay"])
            parameter.add_(other = update, alpha = -adjusted_lr)
            return
        weight = parameter.float()
        weight.mul_(other = 1 - lr * group["weight_decay"])
        weight.add_(other = update, alpha = -adjusted_lr)
        if stochastic:
            from torchao.optim.quant_utils import _fp32_to_bf16_sr

            parameter.copy_(_fp32_to_bf16_sr(_x_f32 = weight))
        else:
            parameter.copy_(weight)

    def load_state_dict(self, state_dict):
        """载入后把动量放回保存时的精度（torch 的 Optimizer.load_state_dict 会转成参数的精度）。"""
        super().load_state_dict(state_dict = state_dict)
        saved_ids = [index for group in state_dict["param_groups"] for index in group["params"]]
        params = [parameter for group in self.param_groups for parameter in group["params"]]
        for index, parameter in zip(saved_ids, params, strict = True):
            value = state_dict["state"].get(index, {}).get("momentum_buffer")
            if value is not None and self.state[parameter]["momentum_buffer"].dtype != value.dtype:
                self.state[parameter]["momentum_buffer"] = value.to(device = parameter.device)


def make_optimizer(projector_params, language_model_params, settings, stochastic = False):
    """Muon 部分：projector 一组（settings.lr，float32 动量）；language_model_params 不为 None 时再加语言模型一组
    （settings.language_model_lr，动量精度 settings.momentum_dtype，stochastic 为真时 bfloat16 参数随机舍入）。"""
    groups = [{"params": list(projector_params), "lr": settings.lr, "momentum_dtype": "fp32", "stochastic": False}]
    if language_model_params is not None:
        groups.append({
            "params": list(language_model_params),
            "lr": settings.language_model_lr,
            "momentum_dtype": settings.momentum_dtype,
            "stochastic": stochastic,
        })
    return GroupStateMuon(
        params = groups,
        lr = settings.lr,
        weight_decay = settings.weight_decay,
        momentum = settings.momentum,
        nesterov = settings.nesterov,
        ns_steps = settings.ns_steps,
        adjust_lr_fn = settings.lr_scaling,
    )


class MuonAdamW(torch.optim.Optimizer):
    """AdamW 与 Muon 合在一起交给学习率调度器和训练循环：param_groups 先列 AdamW 的各组，再列 Muon 的各组（同一批字典，
    调度器直接改写它们）；step、zero_grad 依次作用于两者；state_dict 为 {"adamw": ..., "muon": ...}。
    与 placement.UpdateOptimizer 一样不调用 Optimizer.__init__，只为让学习率调度器接受它。"""

    def __init__(self, adamw, muon):
        self.adamw = adamw
        self.muon = muon

    @property
    def param_groups(self):
        return self.adamw.param_groups + self.muon.param_groups

    def step(self, closure = None):
        self.adamw.step()
        self.muon.step()

    def zero_grad(self, set_to_none = True):
        self.adamw.zero_grad(set_to_none = set_to_none)
        self.muon.zero_grad(set_to_none = set_to_none)

    def state_dict(self):
        return {"adamw": self.adamw.state_dict(), "muon": self.muon.state_dict()}

    def load_state_dict(self, state_dict):
        check_state_format(state_dict = state_dict)
        self.adamw.load_state_dict(state_dict = state_dict["adamw"])
        self.muon.load_state_dict(state_dict = state_dict["muon"])


def check_state_format(state_dict):
    """续训的优化器状态必须是 Muon 加 AdamW 的格式（{"adamw": ..., "muon": ...}）。"""
    if set(state_dict) != {"adamw", "muon"}:
        raise ValueError(f"检查点中的优化器状态不是 --optimizer muon 的格式（键为 {sorted(state_dict)}），不能用 Muon 续训")


def projector_optimizer(params, settings, lr, weight_decay):
    """只训练 projector 时的优化器：偏置用 AdamW（与只用 AdamW 时的构造相同），两个权重矩阵用 Muon。"""
    adamw_params, muon_params = split(items = params, mask = settings.projector_mask)
    adamw = torch.optim.AdamW(
        params = adamw_params,
        lr = lr,
        betas = (0.9, 0.999),
        weight_decay = weight_decay,
    )
    return MuonAdamW(
        adamw = adamw,
        muon = make_optimizer(projector_params = muon_params, language_model_params = None, settings = settings),
    )


def peak_lrs(settings, train_language_model):
    """Muon 各组的峰值学习率，按 param_groups 中的顺序。"""
    return [settings.lr] + ([settings.language_model_lr] if train_language_model else [])


def lr_labels(train_language_model, optimizer):
    """训练日志中各参数组学习率的字段名，按 param_groups 的顺序：lr、lm_lr 是 AdamW 的两组，muon_lr、lm_muon_lr 是 Muon 的两组。"""
    labels = ["lr"] + (["lm_lr"] if train_language_model else [])
    if optimizer == "muon":
        labels += ["muon_lr"] + (["lm_muon_lr"] if train_language_model else [])
    return labels


def resume_notice(param_groups, settings):
    """续训后 Muon 各组的设置（取自检查点）与命令行不同时返回提示，相同时返回 None。"""
    saved = {}
    for group in param_groups:
        if "ns_steps" not in group:
            continue
        for key, attribute in GROUP_SETTINGS.items():
            if group[key] != getattr(settings, attribute):
                saved[key] = group[key]
    if not saved:
        return None
    requested = {key: getattr(settings, GROUP_SETTINGS[key]) for key in saved}
    return f"续训沿用检查点中 Muon 组的设置 {saved}，命令行给出的 {requested} 不起作用。"


def bytes_per_parameter(options, momentum_dtype):
    """语言模型 Muon 部分每个参数在训练中常驻显存的字节数，按来源分开（与 precision.bytes_per_parameter 对应，状态只有动量）。"""
    parts = {
        "weights": 2.0,
        "master": 4.0 if options["language_model_master_weights"] == "fp32" else 0.0,
        "grads": 4.0 if options["language_model_grad_dtype"] == "fp32" else 2.0,
        "states": 4.0 if momentum_dtype == "fp32" else 2.0,
    }
    parts["total"] = sum(parts.values())
    return parts


def newton_schulz_flops(shape, steps):
    """一个矩阵一次正交化的浮点运算次数：每次迭代 4m²n + 2m³（m、n 为较短、较长的一边）。"""
    m, n = sorted(shape)
    return steps * (4 * m * m * n + 2 * m ** 3)


def describe(named, kinds, settings, projector_names, options = None, offload_optimizer = False):
    """训练开始时打印的分组概况：各组（projector / 语言模型 × AdamW / Muon）的张量个数和参数个数，AdamW 部分按类别的个数，
    每步 Newton–Schulz 的计算量，学习率与设置；训练语言模型时（options 为精度选项）还有两部分每个参数的字节数和显存估计
    （GB，10^9 字节；--offload-optimizer 时 AdamW 部分卸载到主机的参数不计）。named 为全部可训练参数 [(名称, 参数)]。"""
    projector = set(projector_names)
    groups = {}
    by_kind = {}
    flops = 0
    for name, parameter in named:
        part = "projector" if name in projector else "language_model"
        method = "muon" if name in settings.names else "adamw"
        entry = groups.setdefault(f"{part}_{method}", {"tensors": 0, "parameters": 0})
        entry["tensors"] += 1
        entry["parameters"] += parameter.numel()
        if method == "muon":
            flops += newton_schulz_flops(shape = tuple(parameter.shape), steps = settings.ns_steps)
        else:
            kind = by_kind.setdefault(kinds[name], {"tensors": 0, "parameters": 0})
            kind["tensors"] += 1
            kind["parameters"] += parameter.numel()
    order = ("projector_adamw", "projector_muon", "language_model_adamw", "language_model_muon")
    result = {
        "groups": {key: groups[key] for key in order if key in groups},
        "adamw_kinds": by_kind,
        "newton_schulz_tflop_per_step": round(number = flops / 1e12, ndigits = 3),
        "lr": settings.lr,
        "language_model_lr": settings.language_model_lr if options is not None else None,
        "lr_scaling": settings.lr_scaling,
        "momentum": settings.momentum,
        "nesterov": settings.nesterov,
        "ns_steps": settings.ns_steps,
        "weight_decay": settings.weight_decay,
    }
    if options is not None:
        from pipeline.train import precision
        from pipeline.train.placement import HOST_MIN_NUMEL

        muon_bytes = bytes_per_parameter(options = options, momentum_dtype = settings.momentum_dtype)
        adamw_bytes = precision.bytes_per_parameter(options = options)
        muon_count = groups.get("language_model_muon", {}).get("parameters", 0)
        adamw_count = groups.get("language_model_adamw", {}).get("parameters", 0)
        offloaded = 0
        if offload_optimizer:
            offloaded = sum(
                parameter.numel() for name, parameter in named
                if name not in projector and name not in settings.names and parameter.numel() >= HOST_MIN_NUMEL
            )
        result["momentum_dtype"] = settings.momentum_dtype
        result["bytes_per_parameter"] = {
            "muon": {key: round(number = value, ndigits = 3) for key, value in muon_bytes.items()},
            "adamw": {key: round(number = value, ndigits = 3) for key, value in adamw_bytes.items()},
        }
        result["gb"] = round(
            number = (
                muon_count * muon_bytes["total"]
                + (adamw_count - offloaded) * adamw_bytes["total"]
                + offloaded * adamw_bytes["weights"]
            ) / 1e9,
            ndigits = 2,
        )
    return result
