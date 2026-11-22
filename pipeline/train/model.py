"""加载阶段 1 模型、冻结训练范围以外的参数、计算目标文本上的损失。

默认只训练 projector；全量微调（任务 #66，train.py 的 --train-language-model）时语言模型的全部参数也参与训练，
音频塔和视觉塔始终冻结。

词表投影加交叉熵有两种实现（--loss-impl，任务 #66）：chunked 是本文件的 LinearCrossEntropy（默认），torch 是
torch.nn.functional.linear_cross_entropy 的分块实现，两者算的是同一个量。批次可以首尾相接拼成一行、去掉填充
（--pack-sequences，data.pack_batch），这时 token_losses_precomputed 把各条样本的边界交给语言模型的各层，
见 check_packing_kernels。
"""

import hashlib
import inspect

import torch
import torch.nn.functional as F

from pipeline.tokenizer_fix import check_tokenizer
from pipeline.train.data import IGNORE_INDEX

LOSS_IMPLS = ("chunked", "torch")

PROJECTOR_PREFIX = "model.audio_projector."
# 语言模型的参数：词嵌入、解码层和最后的归一化都在 model.language_model 下。输出层 lm_head 与词嵌入共用同一个
# 权重（tie_word_embeddings），named_parameters 只以 model.language_model.embed_tokens.weight 给出一次；
# 不共用权重的模型中 lm_head.weight 是单独的参数，也属于语言模型。
LANGUAGE_MODEL_PREFIXES = ("model.language_model.", "lm_head.")
_RAW_DTYPES = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
MODEL_INPUT_KEYS = (
    "input_ids",
    "attention_mask",
    "mm_token_type_ids",
    "input_features",
    "feature_attention_mask",
)
# data.batch_metadata 预先算好的张量（--precompute-metadata），有时一并搬到 GPU。
METADATA_KEYS = (
    "position_ids",
    "audio_mask",
    "audio_padded_feature",
    "audio_chunk_lengths",
    "audio_valid_indices",
    "audio_cu_seqlens",
    "loss_index",
    "loss_targets",
    "loss_rows",
)
# data.pack_batch 给出的样本边界（--pack-sequences）：各条样本在拼接后的一行中的起点（最后一项是总长度）和每个
# token 属于第几条样本，搬到 GPU；最长一条的长度 max_length_q 是 Python 整数，原样传给 FlashAttention。
PACK_KEYS = ("cu_seq_lens_q", "seq_idx")
# 自生成回放（任务 #71，replay.py）加入的张量：每个计损失位置的来源（0 是 ASR，回放各组从 1 开始），kl32 时回放位置上
# 参考模型的 top-32 token 与对数概率。kl_start（Python 整数，kl32 时从这个位置起是回放）原样传入。
REPLAY_KEYS = ("loss_source", "teacher_ids", "teacher_lp")
# 拼接批次时语言模型全注意力层的实现。音频塔用自己的变长注意力（optimize.py），视觉塔不使用，都保持 sdpa。
PACKED_ATTENTION = {
    "": "sdpa",
    "text_config": "flash_attention_2",
    "audio_config": "sdpa",
    "vision_config": "sdpa",
}


def load_model(
    model_dir,
    device = "cuda",
    dtype = torch.bfloat16,
    attn_implementation = "sdpa",
    train_language_model = False,
):
    """加载阶段 1 完整模型，让 model.audio_projector 参与训练；train_language_model 为真时语言模型也参与训练。

    冻结部分（音频塔、视觉塔，只训练 projector 时还有语言模型）按 dtype 加载，projector 转为 float32，
    这样优化器更新的主权重保持 float32 精度。模型内部会把音频塔输出转成 projector 的
    精度，再把 projector 输出转成语言模型的精度，两处转换都在模型代码中完成。
    训练语言模型时它的参数仍按 dtype 加载并用于计算，优化器怎样更新它们见 master.py 和 precision.py。
    attn_implementation 也可以是按子配置给出的字典，拼接批次时用 PACKED_ATTENTION。
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path = str(model_dir),
        trust_remote_code = True,
    )
    # 分词规则必须与原始 Qwen3.5 相同（#77），否则泰语、阿拉伯语等的训练目标会按另一种切法分词
    check_tokenizer(tokenizer = processor, source = model_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        pretrained_model_name_or_path = str(model_dir),
        trust_remote_code = True,
        dtype = dtype,
        attn_implementation = attn_implementation,
    )
    model.model.audio_projector.float()
    for name, parameter in model.named_parameters():
        trainable = name.startswith(PROJECTOR_PREFIX) or (train_language_model and name.startswith(LANGUAGE_MODEL_PREFIXES))
        parameter.requires_grad_(requires_grad = trainable)
    # 冻结部分没有 dropout 需要关闭以外的训练行为，整个模型保持 eval 模式，前向结果确定。
    # 语言模型的 attention_dropout 为 0，训练它时也不需要 train 模式。
    model.eval()
    model.to(device = device)
    return model, processor


def projector_parameters(model):
    return [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith(PROJECTOR_PREFIX)
    ]


def language_model_parameters(model):
    """语言模型的全部参数 [(名称, 参数)]，与输出层共用的词嵌入只出现一次。"""
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith(LANGUAGE_MODEL_PREFIXES)
    ]


def projector_state_dict(model):
    """取出 projector 的权重，键名与完整模型的 state_dict 一致，统一存为 float32。"""
    return {
        name: tensor.detach().to(device = "cpu", dtype = torch.float32).contiguous()
        for name, tensor in model.state_dict().items()
        if name.startswith(PROJECTOR_PREFIX)
    }


def load_projector(model, state):
    """把 projector 权重载入模型；键必须恰好是模型全部的 projector 参数，形状一致。"""
    expected = {
        name: tensor.shape
        for name, tensor in model.state_dict().items()
        if name.startswith(PROJECTOR_PREFIX)
    }
    missing = sorted(set(expected) - set(state))
    extra = sorted(set(state) - set(expected))
    if missing or extra:
        raise ValueError(f"projector 权重的键与模型不一致。缺少：{missing}；多余：{extra}")
    for name, tensor in state.items():
        if tuple(tensor.shape) != tuple(expected[name]):
            raise ValueError(f"{name} 形状为 {tuple(tensor.shape)}，模型中为 {tuple(expected[name])}")
    result = model.load_state_dict(state_dict = state, strict = False)
    if result.unexpected_keys:
        raise ValueError(f"意外的键：{result.unexpected_keys}")


def frozen_fingerprint(model, trainable = ()):
    """对 projector 和 trainable 中的参数以外的全部参数和缓冲区逐个计算 SHA-256，用来确认训练前后没有变化。

    trainable 是另外参与训练的参数名（训练语言模型时为语言模型的参数），它们不参与核对。
    """
    skip = set(trainable)
    digests = {}
    tensors = list(model.named_parameters()) + list(model.named_buffers())
    for name, tensor in tensors:
        if name.startswith(PROJECTOR_PREFIX) or name in skip:
            continue
        data = tensor.detach().contiguous().to(device = "cpu")
        # 按字节宽度换成整数视图再取字节，bfloat16 等 numpy 不支持的类型也能逐位比较。
        raw = data.view(dtype = _RAW_DTYPES[data.dtype.itemsize])
        digests[name] = hashlib.sha256(data = raw.numpy().tobytes()).hexdigest()
    return digests


def move_batch(batch, device, feature_dtype):
    """把模型输入搬到 GPU；梅尔特征转成音频塔的精度。

    有预先算好的元数据时，音频塔直接用分好块的特征 audio_padded_feature，整段特征 input_features 不再上传。
    回放的纯文本微批（replay.text_batch）没有音频的各项，跳过；回放加入的 REPLAY_KEYS 和 kl_start 有则一并搬过去。
    """
    precomputed = "position_ids" in batch
    packed = "cu_seq_lens_q" in batch
    text_only = precomputed and "audio_padded_feature" not in batch
    moved = {}
    for key in MODEL_INPUT_KEYS + (METADATA_KEYS if precomputed else ()) + (PACK_KEYS if packed else ()) + REPLAY_KEYS:
        if precomputed and key == "input_features":
            continue
        if key not in batch and (key in REPLAY_KEYS or text_only):
            continue
        tensor = batch[key].to(device = device, non_blocking = True)
        if key in ("input_features", "audio_padded_feature"):
            tensor = tensor.to(dtype = feature_dtype)
        moved[key] = tensor
    moved["labels"] = batch["labels"].to(device = device, non_blocking = True)
    if precomputed and "audio_max_seqlen" in batch:
        moved["audio_max_seqlen"] = batch["audio_max_seqlen"]
    if packed:
        moved["max_length_q"] = batch["max_length_q"]
    if "kl_start" in batch:
        moved["kl_start"] = batch["kl_start"]
    return moved


def _resolved_function(function):
    """transformers 的 use_kernel_func_from_hub_with_fallback 在导入模型代码时选定的实现。

    它先尝试导入原始的包（causal_conv1d、fla），导入失败时悄悄换成 PyTorch 写的参考实现；选定的函数存在包装函数的
    闭包变量 implementation 中。
    """
    closure = inspect.getclosurevars(func = function).nonlocals
    return closure.get("implementation", function)


def packing_kernels(model):
    """拼接批次时各层实际使用的实现：{名称: (实现的模块.函数名, 是否为能按样本边界分段的内核)}。

    - full_attention：语言模型全注意力层的实现，须是 flash_attention_2，按 cu_seq_lens_q 分段计算（varlen）。
    - causal_conv1d：线性注意力层（Gated DeltaNet）的短卷积，须是 causal_conv1d 包的内核，按 seq_idx 在样本边界处
      重新开始；PyTorch 参考实现不认 seq_idx，会让每条样本开头的几个 token 看到上一条样本的末尾。
    - gated_delta_rule：线性注意力的递推，须是 fla 的 chunk_gated_delta_rule，按 cu_seqlens 在样本边界处清空状态；
      参考实现不认 cu_seqlens，状态会从上一条样本一直带到下一条。
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5

    attention = model.model.language_model.config._attn_implementation
    conv = _resolved_function(function = modeling_qwen3_5.causal_conv1d_fn)
    delta = _resolved_function(function = modeling_qwen3_5.torch_chunk_gated_delta_rule)
    return {
        "full_attention": (attention, attention == "flash_attention_2"),
        "causal_conv1d": (
            f"{conv.__module__}.{conv.__name__}",
            conv.__module__.startswith("causal_conv1d") and "seq_idx" in inspect.signature(obj = conv).parameters,
        ),
        "gated_delta_rule": (
            f"{delta.__module__}.{delta.__name__}",
            delta.__module__.startswith("fla.") and "cu_seqlens" in inspect.signature(obj = delta).parameters,
        ),
    }


def check_packing_kernels(model):
    """拼接批次前确认三处都用真正按样本边界分段的内核，否则停止：退回参考实现时各条样本会互相串扰，且不报错。"""
    kernels = packing_kernels(model = model)
    missing = {name: implementation for name, (implementation, ok) in kernels.items() if not ok}
    if missing:
        raise RuntimeError(
            "--pack-sequences 需要 flash-attn（全注意力层用 flash_attention_2）、causal-conv1d 和 flash-linear-attention 的内核，"
            f"这些层退回了不认样本边界的实现：{missing}"
        )
    return {name: implementation for name, (implementation, _) in kernels.items()}


class LinearCrossEntropy(torch.autograd.Function):
    """把词表投影和交叉熵合在一起分块计算，返回每个位置的损失和预测是否正确。

    词表约 25 万，一次算出全部位置的 float32 分数和它的梯度，批量大时要占十几 GB 显存。
    输出层冻结时只需要对隐状态求梯度，所以前向时逐块算出分数、损失以及损失对隐状态的梯度，
    分数用完即丢；反传时把保存的梯度乘上传回的系数。数值与先投影再 F.cross_entropy 相同。

    训练语言模型时输出层的权重（与词嵌入共用）也需要梯度。权重梯度是各位置「(softmax − one-hot) × 传回的系数」
    与隐状态的乘积之和，要等反传时才知道系数，所以前向只算损失并记下每个位置的 logsumexp；反传时逐块重算分数，
    同时求隐状态和权重的梯度。重算只在计损失的位置进行（每批几百到几千个），比语言模型本身的计算小得多。
    """

    @staticmethod
    def forward(ctx, hidden_states, weight, targets, chunk_size):
        count = hidden_states.shape[0]
        losses = torch.empty(size = (count,), device = hidden_states.device, dtype = torch.float32)
        correct = torch.empty(size = (count,), device = hidden_states.device, dtype = torch.bool)
        weight_grad = ctx.needs_input_grad[1]
        need_grad = ctx.needs_input_grad[0] and not weight_grad
        grads = torch.empty_like(input = hidden_states) if need_grad else None
        normalizers = torch.empty(size = (count,), device = hidden_states.device, dtype = torch.float32) if weight_grad else None
        for start in range(0, count, chunk_size):
            end = min(start + chunk_size, count)
            logits = F.linear(input = hidden_states[start:end], weight = weight).float()
            target = targets[start:end]
            log_normalizer = torch.logsumexp(input = logits, dim = -1)
            target_logits = logits.gather(dim = -1, index = target[:, None]).squeeze(dim = -1)
            losses[start:end] = log_normalizer - target_logits
            correct[start:end] = logits.argmax(dim = -1) == target
            if weight_grad:
                normalizers[start:end] = log_normalizer
            if need_grad:
                # 交叉熵对分数的梯度是 softmax 减去目标位置的 one-hot，原地计算以节省显存。
                probs = logits.sub_(other = log_normalizer[:, None]).exp_()
                rows = torch.arange(end - start, device = probs.device)
                probs[rows, target] -= 1
                grads[start:end] = probs.to(dtype = weight.dtype) @ weight
        ctx.weight_grad = weight_grad
        ctx.chunk_size = chunk_size
        if weight_grad:
            ctx.save_for_backward(hidden_states, weight, targets, normalizers)
        else:
            ctx.save_for_backward(grads)
        ctx.mark_non_differentiable(correct)
        return losses, correct

    @staticmethod
    def backward(ctx, grad_losses, grad_correct):
        if not ctx.weight_grad:
            (grads,) = ctx.saved_tensors
            grad_hidden = (grads.float() * grad_losses[:, None]).to(dtype = grads.dtype)
            return grad_hidden, None, None, None
        hidden_states, weight, targets, normalizers = ctx.saved_tensors
        count = hidden_states.shape[0]
        grad_hidden = torch.empty_like(input = hidden_states) if ctx.needs_input_grad[0] else None
        grad_weight = None
        for start in range(0, count, ctx.chunk_size):
            end = min(start + ctx.chunk_size, count)
            chunk = hidden_states[start:end]
            logits = F.linear(input = chunk, weight = weight).float()
            probs = logits.sub_(other = normalizers[start:end, None]).exp_()
            rows = torch.arange(end - start, device = probs.device)
            probs[rows, targets[start:end]] -= 1
            scaled = probs.mul_(other = grad_losses[start:end, None]).to(dtype = weight.dtype)
            if grad_hidden is not None:
                grad_hidden[start:end] = scaled @ weight
            # 权重梯度与整个输出层一样大（词表 × 隐藏维度），各块直接累加在同一个张量上，不另留 float32 副本。
            if grad_weight is None:
                grad_weight = scaled.t() @ chunk
            else:
                grad_weight.addmm_(mat1 = scaled.t(), mat2 = chunk)
        return grad_hidden, grad_weight, None, None


def torch_linear_cross_entropy(hidden_states, weight, targets, chunk_size):
    """用 torch.nn.functional.linear_cross_entropy 的分块实现算每个位置的损失，再另算预测是否正确。

    options 给出 batch_chunk_size 时每块 chunk_size 行（与 LinearCrossEntropy 相同）；acc_policy 保持 auto，
    bfloat16 输入在 CUDA 上取 compact：分数与梯度在 float32 中累加，输出层权重的梯度逐块直接累加，不另留整份
    float32 副本。不给 batch_chunk_size 时 auto 按「分数块与输入一样大」再除以 2 分块，每批几百个计损失的位置、
    词表 25 万时每块只有几行，内核调用很多。reduction = "none" 时前向只算损失，反传时逐块重算分数再求梯度。
    返回的损失是输入的精度（bfloat16），这里转成 float32 只是统一类型。函数不给出最大分数的位置，预测是否正确
    在不求梯度的情况下逐块重算一遍分数得到（与 LinearCrossEntropy 一样按 bfloat16 分数取最大值）。
    """
    losses = F.linear_cross_entropy(
        input = hidden_states,
        linear_weight = weight,
        target = targets,
        reduction = "none",
        options = torch.nn.LinearCrossEntropyOptions(batch_chunk_size = chunk_size),
    )
    count = hidden_states.shape[0]
    correct = torch.empty(size = (count,), device = hidden_states.device, dtype = torch.bool)
    with torch.no_grad():
        for start in range(0, count, chunk_size):
            end = min(start + chunk_size, count)
            logits = F.linear(input = hidden_states[start:end], weight = weight)
            correct[start:end] = logits.argmax(dim = -1) == targets[start:end]
    return losses.float(), correct


def linear_cross_entropy(
    hidden_states,
    weight,
    targets,
    chunk_size = 1024,
    impl = "chunked",
):
    """计损失位置的词表投影加交叉熵，返回 (每个位置的损失 float32, 预测是否正确)。impl 取 LOSS_IMPLS 之一。"""
    if impl == "chunked":
        return LinearCrossEntropy.apply(hidden_states, weight, targets, chunk_size)
    if impl == "torch":
        return torch_linear_cross_entropy(
            hidden_states = hidden_states,
            weight = weight,
            targets = targets,
            chunk_size = chunk_size,
        )
    raise ValueError(f"未知的损失实现：{impl}，可选 {LOSS_IMPLS}")


class LinearTopKL(torch.autograd.Function):
    """回放的 kl32 损失（任务 #71）：词表投影与「对参考模型 top-32 的有偏截断前向 KL」合在一起分块计算。

    每个位置给出参考模型概率最高的 32 个 token S（top_ids）和它们的对数概率 top_lp（logit − log-sum-exp，回放数据的
    teacher/top_lp.f16），p_ref,i = exp(top_lp_i)。学生的对数概率 log p_θ,i = z_i − lse(z)，z 是这一位置的分数。

        KL_head = Σ_{i∈S} p_ref,i (log p_ref,i − log p_θ,i)
        tail 为真时再加  p_tail,ref (log p_tail,ref − log p_tail,θ)，
        p_tail,ref = max(0, 1 − Σ_{i∈S} p_ref,i)，p_tail,θ = 1 − Σ_{i∈S} p_θ,i

    log p_tail,θ 按 lse(z 中去掉 S 的部分) − lse(z) 计算，与 log(1 − Σ p_θ,i) 相等，尾部很小时不会因相减失去精度。
    不加尾部时损失的最小点是把 S 上的参考分布重新归一化（学生在 S 之外的概率为 0）；加上尾部时最小点是学生在 S 上
    等于参考分布、S 之外的总量等于 p_tail,ref。p_ref,i log p_ref,i 用 xlogy 计算，概率为 0 时为 0。

    对分数的梯度（乘以传回的系数）：不加尾部时 g_j = P_S q_j − p_ref,j [j∈S]，P_S = Σ_{i∈S} p_ref,i，q = softmax(z)；
    加上尾部时 S 内 g_j = q_j − p_ref,j，S 外 g_j = q_j − p_tail,ref q_j / T，T = Σ_{j∉S} q_j（即 S 外按学生自己的
    尾部分布分摊 p_tail,ref 的交叉熵）。与 LinearCrossEntropy 的权重可训练时一样，前向只记下每个位置的 log-sum-exp
    （加尾部时再记 S 外的 log-sum-exp），反传时逐块重算分数，同时求隐状态和输出层权重的梯度，不留整块的分数。

    另外返回每个位置对实际 token（targets，回放中保存的采样 token）的负对数似然和预测是否正确，只用于日志，不求梯度。
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states,
        weight,
        top_ids,
        top_lp,
        targets,
        tail,
        chunk_size,
    ):
        count = hidden_states.shape[0]
        device = hidden_states.device
        losses = torch.empty(size = (count,), device = device, dtype = torch.float32)
        nll = torch.empty(size = (count,), device = device, dtype = torch.float32)
        correct = torch.empty(size = (count,), device = device, dtype = torch.bool)
        normalizers = torch.empty(size = (count,), device = device, dtype = torch.float32)
        tail_normalizers = torch.empty(size = (count,), device = device, dtype = torch.float32) if tail else None
        ref_lp = top_lp.float()
        ref_p = ref_lp.exp()
        ref_mass = ref_p.sum(dim = -1)
        tail_mass = (1 - ref_mass).clamp(min = 0) if tail else None
        entropy_part = torch.special.xlogy(ref_p, ref_p).sum(dim = -1)
        for start in range(0, count, chunk_size):
            end = min(start + chunk_size, count)
            logits = F.linear(input = hidden_states[start:end], weight = weight).float()
            ids = top_ids[start:end]
            target = targets[start:end]
            log_normalizer = torch.logsumexp(input = logits, dim = -1)
            log_q = logits.gather(dim = -1, index = ids) - log_normalizer[:, None]
            loss = entropy_part[start:end] - (ref_p[start:end] * log_q).sum(dim = -1)
            nll[start:end] = log_normalizer - logits.gather(dim = -1, index = target[:, None]).squeeze(dim = -1)
            correct[start:end] = logits.argmax(dim = -1) == target
            normalizers[start:end] = log_normalizer
            if tail:
                logits.scatter_(dim = -1, index = ids, value = float("-inf"))
                tail_normalizer = torch.logsumexp(input = logits, dim = -1)
                mass = tail_mass[start:end]
                log_tail = tail_normalizer - log_normalizer
                loss = loss + torch.special.xlogy(mass, mass) - mass * log_tail
                tail_normalizers[start:end] = tail_normalizer
            losses[start:end] = loss
        ctx.tail = tail
        ctx.chunk_size = chunk_size
        if tail:
            ctx.save_for_backward(hidden_states, weight, top_ids, ref_p, normalizers, tail_normalizers, tail_mass)
        else:
            ctx.save_for_backward(hidden_states, weight, top_ids, ref_p, normalizers, ref_mass)
        ctx.mark_non_differentiable(nll, correct)
        return losses, nll, correct

    @staticmethod
    def backward(ctx, grad_losses, grad_nll, grad_correct):
        if ctx.tail:
            hidden_states, weight, top_ids, ref_p, normalizers, tail_normalizers, tail_mass = ctx.saved_tensors
        else:
            hidden_states, weight, top_ids, ref_p, normalizers, ref_mass = ctx.saved_tensors
        count = hidden_states.shape[0]
        grad_hidden = torch.empty_like(input = hidden_states) if ctx.needs_input_grad[0] else None
        grad_weight = None
        for start in range(0, count, ctx.chunk_size):
            end = min(start + ctx.chunk_size, count)
            chunk = hidden_states[start:end]
            ids = top_ids[start:end]
            logits = F.linear(input = chunk, weight = weight).float()
            log_normalizer = normalizers[start:end, None]
            if ctx.tail:
                q_top = (logits.gather(dim = -1, index = ids) - log_normalizer).exp()
                tail_normalizer = tail_normalizers[start:end, None]
                # S 外：q_j − p_tail q_j / T = (q_j / T)(T − p_tail)，q_j / T = exp(z_j − lse_tail)；S 内另外写入。
                logits.scatter_(dim = -1, index = ids, value = float("-inf"))
                grads = logits.sub_(other = tail_normalizer).exp_()
                tail_share = (tail_normalizer - log_normalizer).exp()
                grads.mul_(other = tail_share - tail_mass[start:end, None])
                grads.scatter_(dim = -1, index = ids, src = q_top - ref_p[start:end])
            else:
                grads = logits.sub_(other = log_normalizer).exp_()
                q_top = grads.gather(dim = -1, index = ids)
                mass = ref_mass[start:end, None]
                grads.mul_(other = mass)
                grads.scatter_(dim = -1, index = ids, src = mass * q_top - ref_p[start:end])
            scaled = grads.mul_(other = grad_losses[start:end, None]).to(dtype = weight.dtype)
            if grad_hidden is not None:
                grad_hidden[start:end] = scaled @ weight
            if ctx.needs_input_grad[1]:
                if grad_weight is None:
                    grad_weight = scaled.t() @ chunk
                else:
                    grad_weight.addmm_(mat1 = scaled.t(), mat2 = chunk)
        return grad_hidden, grad_weight, None, None, None, None, None


def linear_top_kl(
    hidden_states,
    weight,
    top_ids,
    top_lp,
    targets,
    tail = False,
    chunk_size = 1024,
):
    """回放位置的 kl32 损失，返回 (每个位置的 KL float32, 对实际 token 的负对数似然, 预测是否正确)，见 LinearTopKL。"""
    return LinearTopKL.apply(hidden_states, weight, top_ids, top_lp, targets, tail, chunk_size)


def hidden_states_precomputed(model, inputs):
    """语言模型最后一层的隐状态（形状同 input_ids 加隐藏维度），使用 data.batch_metadata 预先算好的位置与索引。

    依次做阶段 1 模型 forward 中的各步：文字嵌入、音频塔（传入预先算好的分块与窗口）、projector、把音频特征
    填入音频标记的位置、语言模型（传入预先算好的位置编号）。各步调用的模块和精度转换与模型代码相同，结果逐位
    一致；省去的只是模型代码中逐条样本的计算和长度核对（音频标记数与音频特征数是否一致，在 CPU 上已由 processor
    保证）。

    拼接的批次（data.pack_batch，带 cu_seq_lens_q）只有一行、没有填充，不建注意力掩码，改为把样本边界按
    transformers 无填充训练的约定（DataCollatorWithFlattening(return_flash_attn_kwargs = True, return_seq_idx = True)
    给出的几项）传给语言模型：全注意力层（flash_attention_2）按 cu_seq_lens_q/k、max_length_q/k 分段计算，
    线性注意力层的递推（fla）取 cu_seq_lens_q 作 cu_seqlens，短卷积（causal_conv1d）按 seq_idx 分段。

    回放的纯文本微批（replay.text_batch，没有 audio_padded_feature）不经过音频塔和 projector，只有文字嵌入。
    """
    from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask

    inner = model.model
    embeds = inner.get_input_embeddings()(input = inputs["input_ids"])
    boundaries = {}
    if "cu_seq_lens_q" in inputs:
        masks = {"full_attention": None, "linear_attention": None}
        boundaries = {
            "cu_seq_lens_q": inputs["cu_seq_lens_q"],
            "cu_seq_lens_k": inputs["cu_seq_lens_q"],
            "max_length_q": inputs["max_length_q"],
            "max_length_k": inputs["max_length_q"],
            "seq_idx": inputs["seq_idx"],
        }
    else:
        # 语言模型的注意力掩码在排入音频塔之前算好（与语言模型 forward 中的调用相同）：掩码函数要检查有没有填充，
        # 会让 CPU 等 GPU，放在这里只需等文字嵌入，不必等整个音频塔算完。
        mask_arguments = {
            "config": inner.language_model.config,
            "inputs_embeds": embeds,
            "attention_mask": inputs["attention_mask"],
            "past_key_values": None,
            "position_ids": None,
        }
        masks = {
            "full_attention": create_causal_mask(**mask_arguments),
            "linear_attention": create_recurrent_attention_mask(**mask_arguments),
        }
    if "audio_padded_feature" in inputs:
        audio = inner.audio_tower(
            input_features = None,
            feature_lens = None,
            padded_feature = inputs["audio_padded_feature"],
            chunk_lengths = inputs["audio_chunk_lengths"],
            valid_indices = inputs["audio_valid_indices"],
            cu_seqlens = inputs["audio_cu_seqlens"],
            max_seqlen = inputs["audio_max_seqlen"],
        )
        projector_weight = inner.audio_projector.linear_1.weight
        features = inner.audio_projector(audio_features = audio.last_hidden_state.to(dtype = projector_weight.dtype))
        embeds = embeds.masked_scatter(
            mask = inputs["audio_mask"].unsqueeze(dim = -1),
            source = features.to(dtype = embeds.dtype),
        )
    outputs = inner.language_model(
        input_ids = None,
        position_ids = inputs["position_ids"],
        attention_mask = masks,
        inputs_embeds = embeds,
        use_cache = False,
        **boundaries,
    )
    return outputs.last_hidden_state


def token_losses_precomputed(
    model,
    inputs,
    chunk_size = 1024,
    loss_impl = "chunked",
):
    """与 token_losses 相同的计算，使用 data.batch_metadata 预先算好的位置与索引，前向中不让 CPU 等 GPU。

    隐状态由 hidden_states_precomputed 算出，再在计损失的位置上做词表投影与交叉熵。
    """
    hidden_states = hidden_states_precomputed(model = model, inputs = inputs)
    flat = hidden_states.reshape(shape = (-1, hidden_states.shape[-1])).index_select(dim = 0, index = inputs["loss_index"])
    losses, correct = linear_cross_entropy(
        hidden_states = flat,
        weight = model.lm_head.weight,
        targets = inputs["loss_targets"],
        chunk_size = chunk_size,
        impl = loss_impl,
    )
    return losses, correct, inputs["loss_rows"]


def token_losses(
    model,
    inputs,
    chunk_size = 1024,
    loss_impl = "chunked",
):
    """前向一次，返回每个计损失位置的 (损失, 是否预测正确, 所在的样本行号)。

    音频塔的参数和输入都不需要梯度，autograd 不会为它保存中间结果，效果等同于 no_grad；
    梯度从损失经过冻结的语言模型传到输入嵌入中的音频位置，再传回 projector。训练语言模型时，
    同一次反传也给语言模型的各层、输出层和词嵌入算出梯度。
    语言模型只输出最后一层隐状态，词表投影只在计损失的位置上分块进行。
    批次带有预先算好的元数据时改用 token_losses_precomputed（拼接的批次总是带着元数据）。
    loss_impl 选择词表投影加交叉熵的实现（LOSS_IMPLS）。
    """
    if "position_ids" in inputs:
        return token_losses_precomputed(
            model = model,
            inputs = inputs,
            chunk_size = chunk_size,
            loss_impl = loss_impl,
        )
    outputs = model.model(
        input_ids = inputs["input_ids"],
        attention_mask = inputs["attention_mask"],
        mm_token_type_ids = inputs["mm_token_type_ids"],
        input_features = inputs["input_features"],
        feature_attention_mask = inputs["feature_attention_mask"],
        use_cache = False,
    )
    # 第 t 个位置的输出预测第 t + 1 个 token，所以隐状态去掉最后一位、标签去掉第一位。
    hidden_states = outputs.last_hidden_state[:, :-1]
    shifted_labels = inputs["labels"][:, 1:]
    loss_mask = shifted_labels != IGNORE_INDEX
    losses, correct = linear_cross_entropy(
        hidden_states = hidden_states[loss_mask],
        weight = model.lm_head.weight,
        targets = shifted_labels[loss_mask],
        chunk_size = chunk_size,
        impl = loss_impl,
    )
    rows = loss_mask.nonzero()[:, 0]
    return losses, correct, rows


def mixed_token_losses(
    model,
    inputs,
    chunk_size = 1024,
    loss_impl = "chunked",
    kl_tail = False,
):
    """带回放的拼接批次（replay.append_samples、replay.text_batch）的损失，返回 (损失, 是否预测正确, 负对数似然)，
    三者都按计损失的位置排列，顺序与 inputs["loss_source"] 相同。

    没有 kl_start 时（--replay-loss ntp，或 kl32 时这一批没有回放样本）全部位置都是交叉熵，负对数似然就是损失本身。
    有 kl_start 时，前 kl_start 个位置（ASR）是交叉熵，之后的位置（回放）是 linear_top_kl 的 kl32 损失，负对数似然
    另外算出，只用于日志。两部分都在同一次语言模型前向的隐状态上计算。
    """
    hidden_states = hidden_states_precomputed(model = model, inputs = inputs)
    flat = hidden_states.reshape(shape = (-1, hidden_states.shape[-1])).index_select(dim = 0, index = inputs["loss_index"])
    targets = inputs["loss_targets"]
    weight = model.lm_head.weight
    kl_start = inputs.get("kl_start")
    if kl_start is None:
        losses, correct = linear_cross_entropy(
            hidden_states = flat,
            weight = weight,
            targets = targets,
            chunk_size = chunk_size,
            impl = loss_impl,
        )
        return losses, correct, losses.detach()
    parts = []
    if kl_start > 0:
        ce_losses, ce_correct = linear_cross_entropy(
            hidden_states = flat[:kl_start],
            weight = weight,
            targets = targets[:kl_start],
            chunk_size = chunk_size,
            impl = loss_impl,
        )
        parts.append((ce_losses, ce_correct, ce_losses.detach()))
    kl_losses, kl_nll, kl_correct = linear_top_kl(
        hidden_states = flat[kl_start:],
        weight = weight,
        top_ids = inputs["teacher_ids"],
        top_lp = inputs["teacher_lp"],
        targets = targets[kl_start:],
        tail = kl_tail,
        chunk_size = chunk_size,
    )
    parts.append((kl_losses, kl_correct, kl_nll))
    if len(parts) == 1:
        return parts[0]
    return tuple(torch.cat(tensors = [part[index] for part in parts]) for index in range(3))


def target_loss(model, inputs, loss_impl = "chunked"):
    """返回 (损失之和, 计损失的 token 数, 预测正确的 token 数)。"""
    losses, correct, _ = token_losses(model = model, inputs = inputs, loss_impl = loss_impl)
    return losses.sum(), losses.numel(), correct.sum()
