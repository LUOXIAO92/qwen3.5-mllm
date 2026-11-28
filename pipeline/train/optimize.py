"""训练的显存与速度选项（实验 v2，任务 T59）。

每一项单独开关，都只改变冻结部分怎样计算和存放，不改变训练的数据、批次、损失和学习率：同一批数据上，
除 --compile 和 --fp8-frozen 外，损失和 projector 的梯度与原实现逐位相同；--compile 的差异与 bfloat16 本身的
舍入误差同一量级，--fp8-frozen 的差异明显更大。层数和维度都从模型读取，0.8B、2B、4B 通用。默认值按 RTX 5090
上真实 v2a 清单的实测选出，测量方法、结果和三种规模下的建议见 pipeline/train/__init__.py「显存与速度选项」。

- --audio-attention sdpa|varlen（默认 varlen）：音频塔的注意力。音频塔把每段音频切成至多约 8 秒的窗口，各窗口
  内部做注意力。原实现（sdpa）在 Python 中逐个窗口调用一次注意力，每层还要把窗口长度取回 CPU；varlen 用
  torch.nn.attention.varlen.varlen_attn（FlashAttention 的变长接口）一层只调用一次，结果与原实现逐位相同。
- --precompute-metadata（默认打开）：语言模型的位置编号、音频塔的分块与窗口边界、计损失的位置，原实现在前向中
  逐条样本计算，每条都要把 GPU 上的数取回 CPU，CPU 反复停下来等 GPU，批越大越慢。打开后这些量在数据读取进程中
  用同样的函数在 CPU 上算好（data.batch_metadata），前向直接使用（model.token_losses_precomputed），结果逐位相同。
- --gradient-checkpointing（默认关闭）：语言模型全部解码层做梯度检查点（offload.py）。前向时每层只保存输入，
  反传时重算，代价是语言模型多算一次前向。0.8B 上每个 token 保存的中间结果从 2.05 MB 降到 0.06 MB。
- --checkpoint-layers N：只对语言模型末尾 N 层做梯度检查点，其余层照常保存中间结果，在显存与重算之间折中。
  每少做一层检查点，每个 token 多占这一层的中间结果（0.8B 约 0.085 MB，4B 约 0.18 MB），每步少重算一层。
- --lm-blocks-to-swap N、--audio-blocks-to-swap N：把语言模型 / 音频塔末尾 N 层的冻结权重放在锁页主机内存中，
  训练时按层流式上传（offload.py），N 大于层数时就是全部层。语言模型流式加载的层必须都做梯度检查点。
  全量微调时（任务 #66）语言模型的这些层要更新：权重只在主机内存中，反传算完一层后梯度搬到主机，主权重和 AdamW 状态
  也在主机内存中，由 CPU 更新（placement.py）。
- --offload-activations：前向时把 autograd 保存的张量（主要是梯度检查点留下的层输入）搬到锁页主机内存，
  反传时搬回（offload.py）。需要梯度检查点。--activation-prefetch（默认打开，任务 #66）：层输入由 BoundaryOffload 管理，
  反传算第 i 层时提前上传第 i - 1 层的输入；--no-activation-prefetch 时与以前一样，用到时才上传。
- --offload-optimizer（任务 #66，只用于全量微调）：常驻显存的语言模型参数的梯度、主权重和 AdamW 状态放在主机内存中，
  显存中只留 bfloat16 权重；梯度在反传中一算出就搬走，CPU 更新后把新权重拷回显存（torchao 的 CPUOffloadOptimizer，
  placement.py）。
- --fp8-frozen none|lm|audio|all（默认 none）：冻结层的线性权重存为 fp8（quant.py）。
- --compile none|lm|audio|all（默认 none）：用 torch.compile 编译语言模型解码层 / 音频塔编码层的前向（形状可变，
  dynamic = True）。fla 的线性注意力内核不参与编译。每个进程开始时要先编译几分钟。0.8B 上速度再快约 12%，测得的
  几批与冻结部分用 float32 计算的结果相比误差比原实现还小；但 4B 上训练时梯度范数约 8000（原实现约 20），只编译
  音频塔时 projector 梯度也与原实现相差 72%。音频塔在三种规模中相同，问题应与批次形状有关，原因未查清，不建议使用。
- --memory-fraction F：PyTorch 最多使用显存总量的 F 倍。WSL 下显存用满后会悄悄借用内存继续运行、速度骤降
  而不报错；设上限后超出时报 out of memory，训练程序据此逐步增加做梯度检查点的层、再逐步改为流式加载，
  重试同一批（Applied.degrade、train.py），训练结果不变。
- --loss-impl chunked|torch（默认 chunked，任务 #66）：计损失位置的词表投影加交叉熵用本项目的分块实现
  （model.LinearCrossEntropy）还是 torch.nn.functional.linear_cross_entropy 的分块实现，两者算的是同一个量，
  差异在 bfloat16 舍入以内。
- --pack-sequences（默认关闭，任务 #66）：每批的样本首尾相接拼成一行、去掉填充（data.pack_batch），每步的样本
  不变。语言模型的全注意力层改用 flash_attention_2，按样本边界分段计算；线性注意力层的短卷积和递推用 causal-conv1d
  与 fla 的内核分段计算。三者缺一时各条样本会互相串扰，训练程序开始前核对（model.check_packing_kernels）。需要
  --precompute-metadata，不能与 --compile 同用。

视觉塔在本项目中从不使用，始终留在 CPU 上，不占显存。

全量微调（train.py 的 --train-language-model，任务 #66）时语言模型的权重要更新：梯度检查点照常可用（反传重算
一层时同时求这一层参数的梯度）；--lm-blocks-to-swap 和 --offload-optimizer 把语言模型参数放到主机内存中更新，
只支持 fp32 的 AdamW 状态（precision.host_problem，其余组合不允许），一步有多个微批时梯度在主机内存中累加
（placement.py「梯度累积」）；--fp8-frozen lm/all 只适用于
冻结的权重，不能使用。显存不足时的退让先增加做梯度检查点的层，再逐步把语言模型的层改为流式加载（参数改在主机内存中
更新，主权重和 AdamW 状态原样搬过去，placement.LanguageModelUpdate.rebuild；精度设置不支持时跳过这一档），最后改为
流式加载音频塔。--optimizer muon（任务 #78）时全量微调不能流式加载语言模型的层（muon.stream_problem），退让同样跳过
这一档；--offload-optimizer 只卸载 AdamW 部分。

续训。--offload-activations、--activation-prefetch、--lm-blocks-to-swap、--offload-optimizer 与其余各项一样不在续训
核对项中：它们只改变激活值、权重、梯度和优化器状态放在显存还是主机内存中，算法不变，续训状态的格式也不变（placement.py
把主机上的状态并回只在显存中训练时的格式），所以续训时可以打开或关闭。层输入和权重的搬运是逐位拷贝，激活值卸载的结果与
不卸载时逐位相同；CPU 上的 AdamW 与显存中的 AdamW 舍入不同，主机内存中更新的训练与只在显存中的训练只在舍入误差上不同。
"""

import argparse
import contextlib
import math

import torch

from pipeline.train.offload import ActivationOffload, BoundaryOffload, FrozenBlocks
from pipeline.train.quant import quantize_fp8

PARTS = ("none", "lm", "audio", "all")
# 这些选项不影响数据与损失，续训时可以改变；记录在训练配置中便于对照速度。
OPTION_KEYS = (
    "gradient_checkpointing",
    "checkpoint_layers",
    "lm_blocks_to_swap",
    "audio_blocks_to_swap",
    "offload_activations",
    "fp8_frozen",
    "compile",
    "audio_attention",
    "precompute_metadata",
    "memory_fraction",
    "loss_impl",
    "pack_sequences",
    "activation_prefetch",
    "offload_optimizer",
)


def add_arguments(parser):
    group = parser.add_argument_group(title = "显存与速度（optimize.py）")
    group.add_argument(
        "--gradient-checkpointing",
        action = argparse.BooleanOptionalAction,
        default = False,
        help = "对语言模型全部解码层做梯度检查点（默认关闭；显存不足时训练程序会逐步打开）",
    )
    group.add_argument(
        "--checkpoint-layers",
        type = int,
        default = None,
        help = "只对语言模型末尾这么多层做梯度检查点（0 到层数），给出时优先于 --gradient-checkpointing",
    )
    group.add_argument(
        "--lm-blocks-to-swap",
        type = int,
        default = 0,
        help = (
            "语言模型末尾多少层的权重放在主机内存中按层流式加载，0 表示全部常驻显存；全量微调时这些层的梯度、"
            "主权重和 AdamW 状态也在主机内存中，由 CPU 更新（placement.py）"
        ),
    )
    group.add_argument(
        "--audio-blocks-to-swap",
        type = int,
        default = 0,
        help = "音频塔末尾多少层的冻结权重放在主机内存中按层流式加载",
    )
    group.add_argument(
        "--offload-activations",
        action = "store_true",
        help = "前向时把保存给反传的激活值搬到锁页主机内存（需要梯度检查点）",
    )
    group.add_argument(
        "--activation-prefetch",
        action = argparse.BooleanOptionalAction,
        default = True,
        help = "激活值卸载时，反传算一层的同时提前上传前一层的层边界（默认打开；关闭时用到才上传）",
    )
    group.add_argument(
        "--offload-optimizer",
        action = "store_true",
        help = (
            "全量微调时常驻显存的语言模型参数的梯度、主权重和 AdamW 状态放在主机内存中，由 CPU 更新"
            "（torchao 的 CPUOffloadOptimizer，placement.py）；显存中只留 bfloat16 权重"
        ),
    )
    group.add_argument(
        "--fp8-frozen",
        default = "none",
        choices = PARTS,
        help = "冻结线性层的权重存为 fp8：none、lm（语言模型解码层）、audio（音频塔编码层）、all",
    )
    group.add_argument(
        "--compile",
        default = "none",
        choices = PARTS,
        help = "用 torch.compile 编译的部分：none、lm、audio、all",
    )
    group.add_argument(
        "--audio-attention",
        default = "varlen",
        choices = ("sdpa", "varlen"),
        help = "音频塔窗口注意力的实现：sdpa 逐窗口调用（原实现），varlen 一层调用一次",
    )
    group.add_argument(
        "--precompute-metadata",
        action = argparse.BooleanOptionalAction,
        default = True,
        help = "在数据读取进程中预先算好位置编号、音频分块和计损失的位置（默认打开），前向中不让 CPU 等 GPU",
    )
    group.add_argument(
        "--memory-fraction",
        type = float,
        default = None,
        help = "PyTorch 最多使用的显存比例，超出时报 out of memory 而不是借用内存",
    )
    group.add_argument(
        "--loss-impl",
        default = "chunked",
        choices = ("chunked", "torch"),
        help = "词表投影加交叉熵的实现：chunked 本项目的分块实现，torch 用 torch.nn.functional.linear_cross_entropy",
    )
    group.add_argument(
        "--pack-sequences",
        action = argparse.BooleanOptionalAction,
        default = False,
        help = "每批样本首尾相接拼成一行、去掉填充（需要 flash-attn、causal-conv1d、flash-linear-attention 的内核）",
    )
    return group


def options_from_args(args):
    return {key: getattr(args, key) for key in OPTION_KEYS}


def checkpoint_count(options, total):
    """做梯度检查点的语言模型层数：--checkpoint-layers 给出时用它，否则按 --gradient-checkpointing 全部或不做。"""
    if options.get("checkpoint_layers") is not None:
        return max(0, min(total, options["checkpoint_layers"]))
    return total if options["gradient_checkpointing"] else 0


def validate(
    options,
    total,
    train_language_model = False,
    host_problem = None,
    stream_problem = None,
):
    """检查选项的组合，不合法时抛出 ValueError。

    host_problem 是全量微调时语言模型参数不能在主机内存中更新的原因（precision.host_problem），为 None 时可以：
    它决定全量微调能否流式加载语言模型的层（--lm-blocks-to-swap）和打开 --offload-optimizer。stream_problem 是只针对
    流式加载的另一个原因（例如 --optimizer muon，muon.stream_problem），为 None 时只看 host_problem。
    """
    stream_problem = stream_problem if stream_problem is not None else host_problem
    count = checkpoint_count(options = options, total = total)
    if min(options["lm_blocks_to_swap"], total) > count:
        raise ValueError("语言模型流式加载的层（--lm-blocks-to-swap）必须都做梯度检查点")
    if options["offload_activations"] and count == 0:
        raise ValueError("--offload-activations 需要梯度检查点")
    if options.get("offload_optimizer") and not train_language_model:
        raise ValueError("--offload-optimizer 只作用于语言模型参数组，需要 --train-language-model")
    if train_language_model and stream_problem is not None and options["lm_blocks_to_swap"] > 0:
        raise ValueError(f"全量微调时流式加载语言模型的层（--lm-blocks-to-swap）不支持这组设置：{stream_problem}")
    if train_language_model and host_problem is not None and options.get("offload_optimizer"):
        raise ValueError(f"--offload-optimizer 不支持这组设置：{host_problem}")
    if train_language_model and _covers(choice = options["fp8_frozen"], part = "lm"):
        raise ValueError("训练语言模型时不能把语言模型的权重存为 fp8（--fp8-frozen lm 或 all）")
    if options.get("pack_sequences"):
        if not options["precompute_metadata"]:
            raise ValueError("--pack-sequences 需要 --precompute-metadata（样本边界在数据读取进程中算好）")
        if options["compile"] != "none":
            raise ValueError("--pack-sequences 没有与 --compile 一起验证过，不能同用")
    if train_language_model and options["lm_blocks_to_swap"] > 0 and _covers(choice = options["compile"], part = "lm"):
        raise ValueError("全量微调时流式加载语言模型的层没有与 --compile lm、all 一起验证过，不能同用")


def _covers(choice, part):
    return choice == "all" or choice == part


def audio_window_tokens(audio_tower):
    """音频塔一个注意力窗口最多的 token 数（n_window_infer 帧经卷积下采样后的长度）。"""
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import _get_feat_extract_output_lengths

    config = audio_tower.config
    chunk = int(_get_feat_extract_output_lengths(input_lengths = torch.tensor(data = 2 * config.n_window), n_window = config.n_window))
    return chunk * (config.n_window_infer // (2 * config.n_window))


def varlen_audio_attention(self, hidden_states, cu_seqlens, max_seqlen = None, **kwargs):
    """Qwen3OmniMoeAudioAttention.forward 的变长版本：全部窗口一次算完。

    q、k、v 的投影与原实现相同；窗口边界 cu_seqlens 直接交给 varlen_attn。没有预先算好的最长窗口 max_seqlen
    时用窗口长度的上界，免得每层把窗口长度取回 CPU；上界只影响内核的分块数，不影响结果。
    """
    from torch.nn.attention.varlen import varlen_attn

    seq_length, _ = hidden_states.size()
    shape = (seq_length, self.num_heads, -1)
    query = self.q_proj(input = hidden_states).reshape(shape = shape)
    key = self.k_proj(input = hidden_states).reshape(shape = shape)
    value = self.v_proj(input = hidden_states).reshape(shape = shape)
    longest = self.window_tokens if max_seqlen is None else max_seqlen
    output = varlen_attn(
        query = query,
        key = key,
        value = value,
        cu_seq_q = cu_seqlens,
        cu_seq_k = cu_seqlens,
        max_q = longest,
        max_k = longest,
        scale = self.scaling,
    )
    output = output.reshape(shape = (seq_length, -1)).contiguous()
    return self.out_proj(input = output)


def pin_device_and_dtype(module, reference):
    """让 module.device、module.dtype 跟随 reference（一个常驻显存、计算精度的参数）。

    transformers 的 PreTrainedModel 按第一个参数取 device 和 dtype，模型代码据此搬运音频塔的输入。
    音频塔的第一个参数在第 0 个编码层里：流式加载时它在 CPU 上，fp8 时它是 float8，都会把输入搬错。
    这里给音频塔换一个只改这两个属性的子类。
    """
    base = type(module)
    module.__class__ = type(
        base.__name__,
        (base,),
        {
            "device": property(fget = lambda self: reference.device),
            "dtype": property(fget = lambda self: reference.dtype),
        },
    )


class Applied:
    """apply 的结果：记录各项实际生效的设置，提供前向用的上下文，以及显存不足时的退让。"""

    def __init__(
        self,
        model,
        options,
        device,
        train_language_model = False,
        host_problem = None,
        stream_problem = None,
    ):
        self.model = model
        self.options = dict(options)
        self.device = device
        self.train_language_model = train_language_model
        # 全量微调时语言模型的层能否流式加载（参数在主机内存中更新，placement.py）；不能时显存不足的退让跳过这一档。
        stream_problem = stream_problem if stream_problem is not None else host_problem
        self.lm_streamable = not train_language_model or stream_problem is None
        self.lm = None
        self.audio = None
        self.lm_forwards = None
        self.audio_forwards = None

    def lm_layers(self):
        return self.model.model.language_model.layers

    def audio_layers(self):
        return self.model.model.audio_tower.layers

    def install_blocks(self):
        """按当前设置接管语言模型和音频塔各层的前向。"""
        options = self.options
        boundaries = None
        if options["offload_activations"] and options.get("activation_prefetch", True):
            boundaries = BoundaryOffload(device = self.device)
        self.lm = FrozenBlocks(
            blocks = self.lm_layers(),
            checkpoint = checkpoint_count(options = options, total = len(self.lm_layers())),
            swap_count = options["lm_blocks_to_swap"],
            device = self.device,
            trainable = self.train_language_model,
            boundaries = boundaries,
        )
        self.audio = FrozenBlocks(
            blocks = self.audio_layers(),
            checkpoint = False,
            swap_count = options["audio_blocks_to_swap"],
            device = self.device,
        )

    def remove_blocks(self):
        for stack in (self.lm, self.audio):
            if stack is not None:
                stack.remove()
        self.lm = self.audio = None

    @contextlib.contextmanager
    def forward_context(self):
        """训练前向用的上下文：打开激活值卸载时，做检查点的层的输入交给 BoundaryOffload（提前上传），其余保存的张量
        由 ActivationOffload 的钩子搬到主机内存。"""
        if not self.options["offload_activations"]:
            yield
            return
        self.lm.offloading = True
        try:
            with ActivationOffload(device = self.device):
                yield
        finally:
            self.lm.offloading = False

    def degrade(self):
        """显存不足时退一步：先逐步增加做梯度检查点的语言模型层，再逐步把语言模型、音频塔的层改为流式加载。

        每步增加该组层数的四分之一（向上取整），层数和维度都从模型读取。返回这一步改动的设置；已经无法再省时
        返回 None。只改变权重和激活值放在哪里，下一次重试同一批数据，训练结果不变。全量微调时语言模型的层改为流式
        加载后，这些层的参数改在主机内存中更新，调用者要让 placement.LanguageModelUpdate.rebuild 重新安排；
        精度设置不支持主机内存中的更新时（lm_streamable 为假）跳过这一档。
        """
        options = self.options
        lm_total, audio_total = len(self.lm_layers()), len(self.audio_layers())
        checkpointed = checkpoint_count(options = options, total = lm_total)
        lm_step, audio_step = math.ceil(lm_total / 4), math.ceil(audio_total / 4)
        if checkpointed < lm_total:
            change = {"checkpoint_layers": min(lm_total, checkpointed + lm_step)}
        elif self.lm_streamable and options["lm_blocks_to_swap"] < lm_total:
            change = {"lm_blocks_to_swap": min(lm_total, options["lm_blocks_to_swap"] + lm_step)}
        elif options["audio_blocks_to_swap"] < audio_total:
            change = {"audio_blocks_to_swap": min(audio_total, options["audio_blocks_to_swap"] + audio_step)}
        else:
            return None
        torch.cuda.synchronize(device = self.device)
        self.remove_blocks()
        options.update(change)
        self.install_blocks()
        torch.cuda.empty_cache()
        return change

    def describe(self):
        return {
            **self.options,
            "lm_offloaded_gb": round(number = (self.lm.offloaded_bytes() if self.lm else 0) / 2 ** 30, ndigits = 3),
            "audio_offloaded_gb": round(number = (self.audio.offloaded_bytes() if self.audio else 0) / 2 ** 30, ndigits = 3),
        }


def apply(
    model,
    options,
    device = "cuda",
    train_language_model = False,
    host_problem = None,
    stream_problem = None,
):
    """按 options 设置模型（模型已在 device 上），返回 Applied。

    顺序：视觉塔移回 CPU → fp8 → 音频注意力 → torch.compile → 接管各层前向（检查点、流式加载）。
    编译的是各层原来的前向，流式加载和检查点的包装在编译范围之外。train_language_model 为真时语言模型的权重
    要更新，不允许只适用于冻结权重的设置；host_problem、stream_problem 见 validate。
    """
    validate(
        options = options,
        total = len(model.model.language_model.layers),
        train_language_model = train_language_model,
        host_problem = host_problem,
        stream_problem = stream_problem,
    )
    inner = model.model
    if getattr(inner, "visual", None) is not None:
        inner.visual.to(device = "cpu")
    dtype = inner.language_model.layers[0].mlp.down_proj.weight.dtype
    if _covers(choice = options["fp8_frozen"], part = "lm"):
        quantize_fp8(module = inner.language_model.layers, compute_dtype = dtype)
    if _covers(choice = options["fp8_frozen"], part = "audio"):
        quantize_fp8(module = inner.audio_tower.layers, compute_dtype = dtype)
    pin_device_and_dtype(module = inner.audio_tower, reference = inner.audio_tower.conv2d1.weight)
    if options["audio_attention"] == "varlen":
        window = audio_window_tokens(audio_tower = inner.audio_tower)
        for layer in inner.audio_tower.layers:
            attention = layer.self_attn
            attention.window_tokens = window
            attention.forward = varlen_audio_attention.__get__(attention)
    if options["compile"] != "none":
        parts = []
        if _covers(choice = options["compile"], part = "lm"):
            parts.extend(inner.language_model.layers)
        if _covers(choice = options["compile"], part = "audio"):
            if options["audio_attention"] != "varlen":
                raise ValueError("编译音频塔需要 --audio-attention varlen：逐窗口的实现每层都要把长度取回 CPU，无法编译")
            parts.extend(inner.audio_tower.layers)
        # 两类解码层、有无梯度（反传重算在 autograd 的线程中进行）、有无填充掩码，以及 fla 内核前后分开编译的
        # 片段按层各自缓存，默认的 8 次重编译上限不够，超出的部分会退回不编译。上限按区域设置：
        # torch._dynamo.config 中的设置只对当前线程有效，管不到反传线程。
        for layer in parts:
            layer.forward = torch.compile(
                model = layer.forward,
                dynamic = True,
                recompile_limit = 128,
            )
    if options["memory_fraction"] is not None:
        torch.cuda.set_per_process_memory_fraction(fraction = options["memory_fraction"])
    applied = Applied(
        model = model,
        options = options,
        device = device,
        train_language_model = train_language_model,
        host_problem = host_problem,
        stream_problem = stream_problem,
    )
    applied.install_blocks()
    torch.cuda.empty_cache()
    return applied
