"""语言模型的 LoRA（任务 #71）：基座权重冻结，只训练各解码层线性层上的低秩增量和 projector。

用途：与全量微调（--train-language-model）对照，看只训练低秩增量能否在保留同样 ASR 能力的同时减轻语言模型的遗忘。
做法与 PEFT（moss-image-trainer 用的 LoraConfig）相同，自己实现，不依赖 peft：

    y = x Wᵀ + (α / r) · (x Aᵀ) Bᵀ，A 形状 (r, 输入维度)，B 形状 (输出维度, r)

- 初始化：A 按 kaiming_uniform_(a = √5)（与 nn.Linear 的默认初始化相同，每个元素在 ±1/√输入维度 之间均匀分布），
  B 为 0，所以训练开始时模型与原模型逐位相同。A 的随机数用 --seed 单独建的生成器，不动全局随机数状态。
- 缩放：默认 α = r（--language-model-lora-alpha，缩放 α / r = 1，与 moss-image-trainer 的默认 rank 32、alpha 32 相同），
  合并时 ΔW = B A 不另乘系数。
- 精度：A、B 以 float32 保存（优化器直接更新它们，相当于全量微调的 float32 主权重），前向时转成输入的精度（bfloat16）
  计算，梯度经类型转换回到 float32。基座权重保持 bfloat16、不求梯度。
- 范围（--language-model-lora-targets，默认 all）：语言模型各解码层中的全部 nn.Linear。Qwen3.5 的 Gated DeltaNet 层
  （线性注意力）是 linear_attn 的 in_proj_qkv、in_proj_z、in_proj_b、in_proj_a、out_proj，全注意力层是 self_attn 的
  q_proj（含输出门）、k_proj、v_proj、o_proj，每层另有 mlp 的 gate_proj、up_proj、down_proj。词嵌入兼输出层（两者共用
  一个权重）、归一化层、短卷积 conv1d、A_log、dt_bias 不加 LoRA，保持冻结；model.LinearCrossEntropy 和 LinearTopKL 拿到的
  输出层权重不需要梯度，只对隐状态求梯度。也可以只给一部分名字，例如 q_proj,k_proj,v_proj,o_proj（按模块名的最后一段
  或层内的完整路径匹配，例如 self_attn.q_proj）。
- 参数量：每个加 LoRA 的线性层 r × (输入维度 + 输出维度)。2B（24 层：18 层线性注意力、6 层全注意力）全部线性层时每个
  秩单位 1,051,200 个参数：线性注意力层 20,512 × 18，全注意力层 15,360 × 6，MLP 24,576 × 24。r = 32 为 33,638,400 个
  （语言模型 1,881,825,088 个参数的 1.79%），r = 90 为 94,608,000 个（5.03%）。in_proj_a、in_proj_b 的输出只有 16 维，
  r 超过 16 时它们的增量最多只有 16 的秩，多出的参数没有作用，按 PEFT 的做法照样计数。

LoRA 层的实现是给原来的 nn.Linear 换成子类 LoRALinear、在原模块上加 lora_A、lora_B 两个参数，基座权重的名字不变
（例如 model.language_model.layers.0.mlp.gate_proj.weight），LoRA 参数的名字是同一模块下的 .lora_A、.lora_B。

检查点（train.py）：每个检查点多存 lora.safetensors（A、B，float32，续训时载入），并写出合并后的
language_model.safetensors：键名、形状与全量微调的检查点完全相同（model.language_model. 开头的全部语言模型参数，
与输出层共用的词嵌入只有一份），bfloat16，W + (α / r) B A 先以 float32 算出再转成 bfloat16。评测程序
（pipeline/eval/run_eval.py --language-model、c71_local、bench_local）不用改动。写出前核对键、形状与精度。
--language-model-lora-keep-merged K 只在最近 K 个检查点中保留合并后的文件（0 表示全部保留），删掉的可以用下面的
merge 命令从 lora.safetensors 重新生成。

命令行（在仓库根目录运行）：

    # 从检查点的 lora.safetensors 重新生成合并后的 language_model.safetensors（CPU 即可，约 1 分钟）
    python -m pipeline.train.lora merge --model-dir <阶段 1 模型目录> --checkpoint <检查点目录>
    # 核对一个 language_model.safetensors 的键、形状和精度与全量微调的检查点（或模型本身）一致
    python -m pipeline.train.lora check --file <language_model.safetensors> --reference <全量微调的 language_model.safetensors>
"""

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from pipeline.train.checkpoint import INFO_FILE, LANGUAGE_MODEL_FILE, LORA_FILE

LORA_SUFFIXES = (".lora_A", ".lora_B")
DEFAULT_TARGETS = "all"
# LoRA 参数组的默认峰值学习率：全量微调语言模型 1e-5 的 10 倍，与 projector 的 1e-4 相同（见 add_arguments）。
DEFAULT_LR = 1e-4
# 续训时必须与原训练一致的选项（train.RESUME_KEYS）：它们决定训练哪些参数、损失怎样随参数变化。
KEYS = ("language_model_lora_rank", "language_model_lora_alpha", "language_model_lora_targets")
# 语言模型解码层所在的位置（阶段 1 融合模型）。
LAYERS_PREFIX = "model.language_model.layers"


def add_arguments(parser):
    group = parser.add_argument_group(title = "语言模型的 LoRA（lora.py）")
    group.add_argument(
        "--language-model-lora-rank",
        type = int,
        default = 0,
        help = "语言模型各解码层线性层上 LoRA 的秩 r，0 表示不用 LoRA；不能与 --train-language-model 同用",
    )
    group.add_argument(
        "--language-model-lora-alpha",
        type = float,
        default = None,
        help = "LoRA 的 α，增量乘以 α / r；默认等于秩（缩放为 1）",
    )
    group.add_argument(
        "--language-model-lora-targets",
        default = DEFAULT_TARGETS,
        help = "加 LoRA 的线性层：all（默认，各解码层中的全部 nn.Linear）或逗号分隔的模块名，例如 q_proj,k_proj,v_proj,o_proj",
    )
    group.add_argument(
        "--language-model-lora-lr",
        type = float,
        default = None,
        help = (
            f"LoRA 参数组的峰值学习率，默认 {DEFAULT_LR:g}（全量微调语言模型 1e-5 的 10 倍，与 projector 相同）；"
            "与 --lr 一样，续训沿用检查点中保存的峰值学习率"
        ),
    )
    group.add_argument(
        "--language-model-lora-keep-merged",
        type = int,
        default = 0,
        help = "只在最近这么多个检查点中保留合并后的 language_model.safetensors，0 表示全部保留",
    )
    return group


def enabled(args):
    return args.language_model_lora_rank > 0


def validate(args):
    """检查 LoRA 选项与其他选项的组合，不合法时抛出 ValueError；合法时把 α、学习率的默认值填入 args。"""
    rank = args.language_model_lora_rank
    if rank < 0:
        raise ValueError(f"--language-model-lora-rank 不能为负（现在是 {rank}）")
    if rank == 0:
        given = [
            flag for flag, value, default in (
                ("--language-model-lora-alpha", args.language_model_lora_alpha, None),
                ("--language-model-lora-targets", args.language_model_lora_targets, DEFAULT_TARGETS),
                ("--language-model-lora-lr", args.language_model_lora_lr, None),
                ("--language-model-lora-keep-merged", args.language_model_lora_keep_merged, 0),
            )
            if value != default
        ]
        if given:
            raise ValueError(f"{', '.join(given)} 只在 --language-model-lora-rank 大于 0 时使用")
        return
    if args.train_language_model:
        raise ValueError("--language-model-lora-rank 与 --train-language-model 不能同用：前者冻结基座只训练低秩增量，后者训练全部参数")
    if getattr(args, "optimizer", "adamw") != "adamw":
        raise ValueError("LoRA 只用 AdamW（--optimizer adamw）")
    if getattr(args, "fp8_frozen", "none") in ("lm", "all"):
        raise ValueError("LoRA 没有与冻结权重的 fp8（--fp8-frozen lm、all）一起验证过，不能同用")
    if getattr(args, "compile", "none") in ("lm", "all"):
        raise ValueError("LoRA 没有与 --compile lm、all 一起验证过，不能同用")
    if args.language_model_lora_keep_merged < 0:
        raise ValueError("--language-model-lora-keep-merged 不能为负")
    parse_targets(spec = args.language_model_lora_targets)
    if args.language_model_lora_alpha is None:
        args.language_model_lora_alpha = float(rank)
    if not args.language_model_lora_alpha > 0:
        raise ValueError(f"--language-model-lora-alpha 应大于 0（现在是 {args.language_model_lora_alpha}）")
    if args.language_model_lora_lr is None:
        args.language_model_lora_lr = DEFAULT_LR
    if not args.language_model_lora_lr > 0:
        raise ValueError(f"--language-model-lora-lr 应大于 0（现在是 {args.language_model_lora_lr}）")


def parse_targets(spec):
    """「all」或逗号分隔的模块名 → None（全部）或名字的集合。"""
    names = [item.strip() for item in str(spec).split(sep = ",") if item.strip()]
    if not names:
        raise ValueError(f"--language-model-lora-targets 为空：{spec!r}")
    if names == [DEFAULT_TARGETS]:
        return None
    if DEFAULT_TARGETS in names:
        raise ValueError(f"--language-model-lora-targets 中 all 不能与其他名字同时给出：{spec!r}")
    return set(names)


class LoRALinear(torch.nn.Linear):
    """加了低秩增量的 nn.Linear：y = x Wᵀ + b + (α / r) (x Aᵀ) Bᵀ。

    由 convert 把原来的 nn.Linear 换成这个子类，基座的 weight、bias 不变；lora_A (r, 输入维度)、lora_B (输出维度, r) 为
    float32 参数，前向时转成输入的精度计算。
    """

    def forward(self, input):
        output = F.linear(input = input, weight = self.weight, bias = self.bias)
        down = F.linear(input = input, weight = self.lora_A.to(dtype = input.dtype))
        up = F.linear(input = down, weight = self.lora_B.to(dtype = input.dtype))
        if self.lora_scaling != 1.0:
            up = up * self.lora_scaling
        return output + up

    def delta_weight(self):
        """合并用的增量 (α / r) B A，float32，形状与 weight 相同。"""
        return (self.lora_B.detach().float() @ self.lora_A.detach().float()) * self.lora_scaling

    def extra_repr(self):
        return f"{super().extra_repr()}, lora_rank={self.lora_A.shape[0]}, lora_scaling={self.lora_scaling:g}"


def convert(
    module,
    rank,
    alpha,
    generator,
):
    """把一个 nn.Linear 原地换成 LoRALinear：A 按 kaiming_uniform_(a = √5) 初始化（用 generator），B 为 0。"""
    if type(module) is not torch.nn.Linear:
        raise TypeError(f"只能给 nn.Linear 加 LoRA：{type(module).__name__}")
    device = module.weight.device
    down = torch.empty(size = (rank, module.in_features), dtype = torch.float32)
    torch.nn.init.kaiming_uniform_(tensor = down, a = math.sqrt(5), generator = generator)
    module.__class__ = LoRALinear
    module.lora_A = torch.nn.Parameter(data = down.to(device = device))
    module.lora_B = torch.nn.Parameter(data = torch.zeros(size = (module.out_features, rank), dtype = torch.float32, device = device))
    module.lora_scaling = float(alpha) / rank


def select_targets(layers, spec, prefix = LAYERS_PREFIX):
    """各解码层中要加 LoRA 的 nn.Linear：[(完整模块名, 模块)]，按层号和层内顺序。

    spec 为 all 时取全部 nn.Linear，否则按模块名的最后一段或层内的完整路径匹配；给出的名字一个也没匹配到时抛出 ValueError。
    """
    wanted = parse_targets(spec = spec)
    selected = []
    seen = set()
    available = set()
    for index, layer in enumerate(layers):
        for name, module in layer.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            available.add(name)
            available.add(name.rsplit(sep = ".", maxsplit = 1)[-1])
            hits = None if wanted is None else {name, name.rsplit(sep = ".", maxsplit = 1)[-1]} & wanted
            if wanted is not None and not hits:
                continue
            if hits:
                seen.update(hits)
            selected.append((f"{prefix}.{index}.{name}", module))
    if wanted is not None and wanted - seen:
        raise ValueError(f"--language-model-lora-targets 中这些名字没有匹配到线性层：{sorted(wanted - seen)}，可选 {sorted(available)}")
    if not selected:
        raise ValueError("没有可以加 LoRA 的线性层")
    return selected


def attach(
    model,
    rank,
    alpha,
    targets = DEFAULT_TARGETS,
    seed = 0,
    layers = None,
    prefix = LAYERS_PREFIX,
):
    """给语言模型各解码层的线性层加 LoRA，返回 LoRA 参数 [(名称, 参数)]，顺序与 model.named_parameters 相同。

    layers 不给时取阶段 1 融合模型的 model.model.language_model.layers；单元测试可以传入别的层列表和对应的 prefix。
    基座参数的 requires_grad 不动（调用者事先冻结），LoRA 参数需要梯度。
    """
    if layers is None:
        layers = model.model.language_model.layers
    if any(isinstance(module, LoRALinear) for module in layers.modules()):
        raise RuntimeError("这些层已经加过 LoRA")
    generator = torch.Generator().manual_seed(seed)
    for _, module in select_targets(layers = layers, spec = targets, prefix = prefix):
        convert(
            module = module,
            rank = rank,
            alpha = alpha,
            generator = generator,
        )
    return named_parameters(model = model)


def named_parameters(model):
    """模型中的 LoRA 参数 [(名称, 参数)]。"""
    return [(name, parameter) for name, parameter in model.named_parameters() if name.endswith(LORA_SUFFIXES)]


def lora_modules(model):
    """{模块名: LoRALinear}。"""
    return {name: module for name, module in model.named_modules() if isinstance(module, LoRALinear)}


def expected_count(modules, rank):
    """按公式算出的 LoRA 参数量 Σ r × (输入维度 + 输出维度)，modules 为 nn.Linear 的列表。"""
    return sum(rank * (module.in_features + module.out_features) for module in modules)


def require_input_grads(model):
    """让语言模型的词嵌入输出需要梯度（PEFT 的 enable_input_require_grads）。

    LoRA 时词嵌入冻结，回放的纯文本微批进入第 0 层时隐状态不需要梯度，offload.FrozenBlocks 就不对这一层做梯度检查点
    （流式加载时还会报错）；第 0 层里的 LoRA 参数照样有梯度，但这一层的行为与其余层不同。让词嵌入的输出需要梯度后，
    各层一律按设置处理，代价只是多求一次第 0 层输入的梯度。返回钩子的句柄。
    """
    def hook(module, inputs, output):
        if torch.is_grad_enabled() and not output.requires_grad:
            output.requires_grad_(requires_grad = True)
        return output

    return model.get_input_embeddings().register_forward_hook(hook = hook)


def state_dict(model):
    """LoRA 参数 {名称: CPU 上的 float32 张量}，写成 lora.safetensors。"""
    return {
        name: parameter.detach().to(device = "cpu", dtype = torch.float32).contiguous()
        for name, parameter in named_parameters(model = model)
    }


def load_state(model, state):
    """把 lora.safetensors 的内容载入模型的 LoRA 参数；键必须恰好是模型的全部 LoRA 参数，形状一致。"""
    expected = dict(named_parameters(model = model))
    missing = sorted(set(expected) - set(state))
    extra = sorted(set(state) - set(expected))
    if missing or extra:
        raise ValueError(f"LoRA 权重的键与模型不一致。缺少：{missing[:5]}；多余：{extra[:5]}")
    with torch.no_grad():
        for name, parameter in expected.items():
            tensor = state[name]
            if tuple(tensor.shape) != tuple(parameter.shape):
                raise ValueError(f"{name} 形状为 {tuple(tensor.shape)}，模型中为 {tuple(parameter.shape)}")
            parameter.copy_(tensor.to(device = parameter.device, dtype = parameter.dtype))


def read_state(path):
    """读取 lora.safetensors；path 可以是文件，也可以是检查点目录。"""
    from safetensors.torch import load_file

    path = Path(path)
    if path.is_dir():
        path = path / LORA_FILE
    return load_file(filename = str(path), device = "cpu")


def base_language_model_shapes(model, prefixes):
    """语言模型的基座参数 {名称: (形状, 精度)}，不含 LoRA 参数；即全量微调检查点中 language_model.safetensors 的键。"""
    return {
        name: (tuple(parameter.shape), parameter.dtype)
        for name, parameter in model.named_parameters()
        if name.startswith(prefixes) and not name.endswith(LORA_SUFFIXES)
    }


def merged_state(model, prefixes, dtype = torch.bfloat16):
    """合并后的语言模型权重 {名称: CPU 上的 dtype 张量}：加 LoRA 的线性层为 W + (α / r) B A（float32 中相加再转成 dtype），
    其余参数原样。键与全量微调的 language_model.safetensors 相同（prefixes 为 model.LANGUAGE_MODEL_PREFIXES）。"""
    merged_weights = {f"{name}.weight": module for name, module in lora_modules(model = model).items()}
    state = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not name.startswith(prefixes) or name.endswith(LORA_SUFFIXES):
                continue
            module = merged_weights.pop(name, None)
            tensor = parameter.detach()
            if module is not None:
                tensor = tensor.float() + module.delta_weight().to(device = tensor.device)
            state[name] = tensor.to(device = "cpu", dtype = dtype).contiguous()
    if merged_weights:
        raise RuntimeError(f"这些 LoRA 层的基座权重不在语言模型参数中：{sorted(merged_weights)[:5]}")
    return state


def check_state(state, expected, dtype = torch.bfloat16):
    """核对合并后的权重：键恰好是 expected（base_language_model_shapes）的键，形状相同，精度都是 dtype。不一致时抛出 ValueError。"""
    missing = sorted(set(expected) - set(state))
    extra = sorted(set(state) - set(expected))
    if missing or extra:
        raise ValueError(f"合并后的语言模型权重与模型的键不一致。缺少：{missing[:5]}；多余：{extra[:5]}")
    for name, (shape, _) in expected.items():
        tensor = state[name]
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} 形状为 {tuple(tensor.shape)}，模型中为 {shape}")
        if tensor.dtype != dtype:
            raise ValueError(f"{name} 精度为 {tensor.dtype}，应为 {dtype}")


def describe(model, rank, alpha, targets, language_model_parameters):
    """训练开始时打印的概况：秩、α、缩放、加 LoRA 的层数（按层内名字）、参数量与占语言模型参数的比例。"""
    modules = lora_modules(model = model)
    kinds = {}
    for name in modules:
        relative = name.split(sep = ".layers.", maxsplit = 1)[-1].split(sep = ".", maxsplit = 1)[-1]
        kinds[relative] = kinds.get(relative, 0) + 1
    count = sum(parameter.numel() for _, parameter in named_parameters(model = model))
    return {
        "rank": rank,
        "alpha": alpha,
        "scaling": alpha / rank,
        "targets": targets,
        "modules": len(modules),
        "modules_by_name": kinds,
        "parameters": count,
        "parameters_formula": expected_count(modules = modules.values(), rank = rank),
        "language_model_parameters": language_model_parameters,
        "share_of_language_model": round(number = count / language_model_parameters, ndigits = 5) if language_model_parameters else None,
    }


def rank_for_share(modules, total, share):
    """使 LoRA 参数量最接近 total × share 的秩：每个秩单位的参数量为 Σ (输入维度 + 输出维度)。"""
    per_rank = sum(module.in_features + module.out_features for module in modules)
    return max(1, round(total * share / per_rank)), per_rank


def safetensors_header(path):
    """读取 safetensors 文件头：{键: (形状, 精度名)}，不读张量。"""
    with open(file = path, mode = "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(s = handle.read(size))
    header.pop("__metadata__", None)
    return {key: (tuple(value["shape"]), value["dtype"]) for key, value in header.items()}


def compare_headers(actual, reference):
    """两个文件头的差异 [说明]，相同时为空列表。"""
    problems = []
    missing = sorted(set(reference) - set(actual))
    extra = sorted(set(actual) - set(reference))
    if missing:
        problems.append(f"缺少 {len(missing)} 个键：{missing[:5]}")
    if extra:
        problems.append(f"多出 {len(extra)} 个键：{extra[:5]}")
    for key in sorted(set(actual) & set(reference)):
        if actual[key] != reference[key]:
            problems.append(f"{key}：{actual[key]}，参照为 {reference[key]}")
    return problems


def model_header(model_dir):
    """按模型目录的配置在 meta 设备上建模型，得到全量微调 language_model.safetensors 应有的 {键: (形状, 'BF16')}。"""
    from transformers import AutoConfig, AutoModelForImageTextToText

    from pipeline.train.model import LANGUAGE_MODEL_PREFIXES

    config = AutoConfig.from_pretrained(pretrained_model_name_or_path = str(model_dir), trust_remote_code = True)
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(config = config, trust_remote_code = True)
    return {
        name: (tuple(parameter.shape), "BF16")
        for name, parameter in model.named_parameters()
        if name.startswith(LANGUAGE_MODEL_PREFIXES)
    }


def merge_checkpoint(model_dir, checkpoint, output = None):
    """从检查点的 lora.safetensors 重新生成合并后的 language_model.safetensors（在 CPU 上），返回写出的路径。

    秩、α、范围取自检查点 projector.json 中的训练配置。
    """
    from safetensors.torch import save_file

    from pipeline.train.model import LANGUAGE_MODEL_PREFIXES, load_model

    checkpoint = Path(checkpoint)
    config = json.loads(s = (checkpoint / INFO_FILE).read_text(encoding = "utf-8"))["config"]
    model, _ = load_model(model_dir = model_dir, device = "cpu")
    attach(
        model = model,
        rank = config["language_model_lora_rank"],
        alpha = config["language_model_lora_alpha"],
        targets = config["language_model_lora_targets"],
        seed = config["seed"],
    )
    load_state(model = model, state = read_state(path = checkpoint))
    state = merged_state(model = model, prefixes = LANGUAGE_MODEL_PREFIXES)
    check_state(state = state, expected = base_language_model_shapes(model = model, prefixes = LANGUAGE_MODEL_PREFIXES))
    target = Path(output) if output else checkpoint / LANGUAGE_MODEL_FILE
    staging = target.with_name(target.name + ".tmp")
    save_file(tensors = state, filename = str(staging), metadata = {"format": "pt"})
    staging.replace(target = target)
    return target


def main(argv = None):
    parser = argparse.ArgumentParser(description = "LoRA 检查点：合并出 language_model.safetensors，或核对它的键与形状")
    commands = parser.add_subparsers(dest = "command", required = True)
    merge = commands.add_parser("merge", help = "从 lora.safetensors 重新生成合并后的 language_model.safetensors")
    merge.add_argument("--model-dir", required = True, help = "阶段 1 模型目录")
    merge.add_argument("--checkpoint", required = True, help = "检查点目录（含 lora.safetensors、projector.json）")
    merge.add_argument("--output", default = None, help = "输出文件，默认 <检查点目录>/language_model.safetensors")
    check = commands.add_parser("check", help = "核对 language_model.safetensors 的键、形状与精度")
    check.add_argument("--file", required = True, help = "要核对的 language_model.safetensors")
    check.add_argument("--reference", default = None, help = "参照：全量微调检查点的 language_model.safetensors")
    check.add_argument("--model-dir", default = None, help = "参照：按这个模型目录的配置算出应有的键与形状")
    args = parser.parse_args(args = argv)
    if args.command == "merge":
        path = merge_checkpoint(model_dir = args.model_dir, checkpoint = args.checkpoint, output = args.output)
        print(f"已写出 {path}")
        return 0
    if args.reference is None and args.model_dir is None:
        parser.error("check 需要 --reference 或 --model-dir")
    actual = safetensors_header(path = args.file)
    results = {}
    if args.reference is not None:
        results["reference"] = compare_headers(actual = actual, reference = safetensors_header(path = args.reference))
    if args.model_dir is not None:
        results["model_dir"] = compare_headers(actual = actual, reference = model_header(model_dir = args.model_dir))
    print(json.dumps(obj = {"tensors": len(actual), "problems": results}, ensure_ascii = False, indent = 1))
    return 1 if any(results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
