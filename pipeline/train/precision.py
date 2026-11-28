"""全量微调时语言模型参数组的精度选项（任务 #66）。

语言模型的参数在模型中是 bfloat16，前向和反传都用它们计算。下面四项决定优化器怎样更新这些参数，彼此独立，
默认值就是以前写死的做法（float32 主权重、float32 梯度、float32 AdamW 状态，每个参数 18 字节）：

- --language-model-master-weights fp32|none：fp32 时优化器更新另存的 float32 主权重，更新后舍入成 bfloat16
  写回模型（master.MasterWeights）；none 时优化器直接更新模型中的 bfloat16 参数，不另存主权重（master.DirectWeights）。
- --language-model-grad-dtype fp32|bf16：交给优化器的梯度的精度。fp32 时反传得到的 bfloat16 梯度转成 float32 再累加；
  bf16 时不转换。梯度累积（--grad-accum 大于 1，或开启回放时长样本另成的微批）时各微批的梯度在这个精度中相加，bf16 时
  每次相加都舍入到 bfloat16，累积的微批越多误差越大；本项目的设置是 --grad-accum 1，不开回放时一步只有一次反传，两者算出的
  梯度相同，bf16 只省显存。参数在主机内存中更新时累加也按这个精度进行（placement.py「梯度累积」）。
- --language-model-rounding nearest|stochastic：没有主权重时，AdamW 在 float32 中算出新的权重，再写回 bfloat16 参数时
  怎样舍入。nearest 就近舍入：学习率小时每步的更新常常小于 bfloat16 相邻两个数的间距（权重为 1 时约 0.0078），
  被舍掉，权重不动。stochastic 随机舍入：按与上下两个 bfloat16 数的距离随机取其一，期望等于 float32 的结果，
  小的更新按比例体现出来（arXiv 2502.20566）。只能与 --language-model-master-weights none 一起用：有主权重时更新
  在 float32 中累积，写回模型就近舍入即可。
- --language-model-optimizer-states fp32|bf16|8bit|4bit|fp8：AdamW 两个状态（一阶矩、二阶矩）的存储格式。8bit、
  4bit、fp8 是 torchao 的分块量化状态（AdamW8bit、AdamW4bit、AdamWFp8），每块 256、128、256 个数共用一个 float32
  缩放系数；与 torchao 相同，元素少于 4096 个或不能被块大小整除的张量（归一化层的权重等）按参数自身的精度存。
  状态的计算都在 float32 中进行，只是存储时舍入或量化。

projector 参数组很小（几百万个参数），始终是 float32 权重、float32 梯度、float32 状态，不受这几项影响。

--optimizer muon（任务 #78，muon.py）时语言模型的二维隐藏层权重矩阵由 Muon 更新：前三项同样作用于这些矩阵，
--language-model-optimizer-states 只决定其余参数（AdamW 部分）的状态，Muon 动量的精度由 --muon-momentum-dtype 决定。

实现。torch 的 fused AdamW 要求权重、梯度和状态是同一种精度，所以只在两种组合中使用：默认的 fp32 主权重 + fp32 梯度
+ fp32 状态（以前的做法，续训逐位不变），以及 none + bf16 梯度 + bf16 状态 + 就近舍入（纯 bfloat16 的 AdamW）。
其余组合用 torchao（0.18.0）的 AdamW 实现（torchao_adamw.py）：每个张量在 float32 中计算，状态按格式存放，随机舍入
用 torchao 的 bf16_stochastic_round。torchao 只在用到时导入。

这几项改变优化器更新哪一份权重和状态的格式，列入续训核对（train.RESUME_KEYS）；以前的训练配置中没有这几项，
按默认值核对，照常续训。

显存（每个语言模型参数，字节）：bfloat16 的模型参数 2（只训练 projector 时就在显存中），float32 主权重 4，
梯度 4 或 2，AdamW 状态 8（fp32）、4（bf16）、约 2（8bit、fp8）、约 1（4bit）。见 bytes_per_parameter 和
pipeline/train/__init__.py「全量微调」中的表。
"""

import torch

MASTER_WEIGHTS = ("fp32", "none")
GRAD_DTYPES = ("fp32", "bf16")
ROUNDINGS = ("nearest", "stochastic")
OPTIMIZER_STATES = ("fp32", "bf16", "8bit", "4bit", "fp8")
DEFAULTS = {
    "language_model_master_weights": "fp32",
    "language_model_grad_dtype": "fp32",
    "language_model_rounding": "nearest",
    "language_model_optimizer_states": "fp32",
}
# 续训核对项（train.RESUME_KEYS 的一部分）。
KEYS = tuple(DEFAULTS)
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
# 每个参数的 AdamW 状态（两个）占的字节数。量化格式另有每块一个 float32 缩放系数，两个状态各一份。
STATE_BYTES = {
    "fp32": 8.0,
    "bf16": 4.0,
    "8bit": 2.0 + 2 * 4 / 256,
    "4bit": 1.0 + 2 * 4 / 128,
    "fp8": 2.0 + 2 * 4 / 256,
}


def add_arguments(parser):
    group = parser.add_argument_group(title = "全量微调的精度（precision.py，只作用于语言模型参数组）")
    group.add_argument(
        "--language-model-master-weights",
        default = DEFAULTS["language_model_master_weights"],
        choices = MASTER_WEIGHTS,
        help = "fp32：优化器更新 float32 主权重，再写回 bfloat16 参数（默认）；none：直接更新模型中的 bfloat16 参数",
    )
    group.add_argument(
        "--language-model-grad-dtype",
        default = DEFAULTS["language_model_grad_dtype"],
        choices = GRAD_DTYPES,
        help = "交给优化器的梯度精度：fp32（默认，梯度累积在 float32 中相加）或 bf16（不转换）",
    )
    group.add_argument(
        "--language-model-rounding",
        default = DEFAULTS["language_model_rounding"],
        choices = ROUNDINGS,
        help = "没有主权重时更新写回 bfloat16 参数的舍入：nearest 就近（默认），stochastic 随机舍入（torchao）",
    )
    group.add_argument(
        "--language-model-optimizer-states",
        default = DEFAULTS["language_model_optimizer_states"],
        choices = OPTIMIZER_STATES,
        help = "AdamW 状态的存储：fp32（默认）、bf16，或 torchao 的分块量化 8bit、4bit、fp8",
    )
    return group


def options_from_args(args):
    return {key: getattr(args, key) for key in KEYS}


def weight_dtype(options):
    """优化器更新的那份权重的精度：有主权重时 float32，否则是模型参数的 bfloat16。"""
    return "fp32" if options["language_model_master_weights"] == "fp32" else "bf16"


def uses_torch_adamw(options):
    """能否用 torch 的 fused AdamW：权重、梯度、状态同一精度，且就近舍入。"""
    dtype = weight_dtype(options = options)
    return (
        options["language_model_rounding"] == "nearest"
        and options["language_model_grad_dtype"] == dtype
        and options["language_model_optimizer_states"] == dtype
    )


def validate(options, train_language_model):
    """检查选项的组合，不合法时抛出 ValueError。不训练语言模型时这几项必须保持默认值。"""
    changed = [key for key in KEYS if options[key] != DEFAULTS[key]]
    if not train_language_model:
        if changed:
            flags = ", ".join("--" + key.replace("_", "-") for key in changed)
            raise ValueError(f"{flags} 只作用于语言模型参数组，需要 --train-language-model")
        return
    if options["language_model_rounding"] == "stochastic" and options["language_model_master_weights"] != "none":
        raise ValueError(
            "随机舍入（--language-model-rounding stochastic）作用于直接更新的 bfloat16 参数，"
            "需要 --language-model-master-weights none；有 float32 主权重时更新在主权重中累积，写回模型用就近舍入"
        )
    if not uses_torch_adamw(options = options):
        try:
            import torchao.optim  # noqa: F401
        except ImportError as error:
            raise ValueError(
                f"这组精度设置 {options} 需要 torchao（pip install torchao==0.18.0）：{error}"
            ) from error


def host_problem(options):
    """语言模型参数在主机内存中更新（流式加载的层、--offload-optimizer，placement.py）时这组设置有什么不支持，返回原因；
    支持时返回 None。

    支持：fp32 主权重 + fp32 状态（梯度精度任意，舍入只能就近），以及 none + fp32 状态 + 就近或随机舍入。一步有多个微批
    （回放的长样本、--grad-accum 大于 1）时梯度在主机内存中按梯度精度累加（placement.py「梯度累积」）。
    """
    if options["language_model_optimizer_states"] != "fp32":
        return (
            f"语言模型参数在主机内存中更新时 AdamW 状态只支持 fp32（现在是 {options['language_model_optimizer_states']}）："
            "bf16 和量化状态是为省显存设计的，状态放在主机内存时不需要，也没有实现"
        )
    return None


def bytes_per_parameter(options):
    """每个语言模型参数在训练中常驻显存的字节数，按来源分开。

    梯度是一步反传结束到优化器更新之间同时存在的那一份：有主权重且梯度为 fp32 时，bfloat16 的梯度在反传中
    算出就转入主权重的梯度并释放，只算 float32 的 4 字节。不含优化器更新时逐个张量的临时显存。
    """
    parts = {
        "weights": 2.0,
        "master": 4.0 if options["language_model_master_weights"] == "fp32" else 0.0,
        "grads": 4.0 if options["language_model_grad_dtype"] == "fp32" else 2.0,
        "states": STATE_BYTES[options["language_model_optimizer_states"]],
    }
    parts["total"] = sum(parts.values())
    return parts


def describe(options, parameters):
    """训练开始时打印的概况：实现、每个参数的字节数、按参数个数算出的显存（GB，10^9 字节）。"""
    per_parameter = bytes_per_parameter(options = options)
    return {
        **options,
        "optimizer": "torch fused AdamW" if uses_torch_adamw(options = options) else "torchao AdamW",
        "bytes_per_parameter": {key: round(number = value, ndigits = 3) for key, value in per_parameter.items()},
        "parameters": parameters,
        "gb": round(number = per_parameter["total"] * parameters / 1e9, ndigits = 2),
    }


def make_optimizer(
    projector_params,
    language_model_params,
    lr,
    language_model_lr,
    weight_decay,
    options,
):
    """训练语言模型时的 AdamW：projector 一组（lr），语言模型一组（language_model_lr）。

    能用 torch 的 fused AdamW 时用它：它逐个张量就地更新；默认的 foreach 实现会为整组参数临时再分配一份与二阶矩
    一样大的显存（2B 约 7.5 GB）。其余组合用 torchao 的实现，projector 组的状态固定为 float32。
    """
    if uses_torch_adamw(options = options):
        return torch.optim.AdamW(
            params = [
                {"params": projector_params, "lr": lr},
                {"params": language_model_params, "lr": language_model_lr},
            ],
            lr = lr,
            betas = (0.9, 0.999),
            weight_decay = weight_decay,
            fused = True,
        )
    from pipeline.train.torchao_adamw import GroupStateAdamW

    return GroupStateAdamW(
        params = [
            {"params": projector_params, "lr": lr, "states": "fp32"},
            {"params": language_model_params, "lr": language_model_lr, "states": options["language_model_optimizer_states"]},
        ],
        lr = lr,
        betas = (0.9, 0.999),
        eps = 1e-8,
        weight_decay = weight_decay,
        bf16_stochastic_round = options["language_model_rounding"] == "stochastic",
    )

