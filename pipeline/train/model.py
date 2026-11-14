"""加载阶段 1 模型、冻结 projector 以外的参数、计算目标文本上的损失。"""

import hashlib

import torch
import torch.nn.functional as F

from pipeline.train.data import IGNORE_INDEX

PROJECTOR_PREFIX = "model.audio_projector."
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


def load_model(
    model_dir,
    device = "cuda",
    dtype = torch.bfloat16,
    attn_implementation = "sdpa",
):
    """加载阶段 1 完整模型，只让 model.audio_projector 参与训练。

    冻结部分（音频塔、视觉塔、语言模型）按 dtype 加载，projector 转为 float32，
    这样优化器更新的主权重保持 float32 精度。模型内部会把音频塔输出转成 projector 的
    精度，再把 projector 输出转成语言模型的精度，两处转换都在模型代码中完成。
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path = str(model_dir),
        trust_remote_code = True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        pretrained_model_name_or_path = str(model_dir),
        trust_remote_code = True,
        dtype = dtype,
        attn_implementation = attn_implementation,
    )
    model.model.audio_projector.float()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(requires_grad = name.startswith(PROJECTOR_PREFIX))
    # 冻结部分没有 dropout 需要关闭以外的训练行为，整个模型保持 eval 模式，前向结果确定。
    model.eval()
    model.to(device = device)
    return model, processor


def projector_parameters(model):
    return [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith(PROJECTOR_PREFIX)
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


def frozen_fingerprint(model):
    """对 projector 以外的全部参数和缓冲区逐个计算 SHA-256，用来确认训练前后没有变化。"""
    digests = {}
    tensors = list(model.named_parameters()) + list(model.named_buffers())
    for name, tensor in tensors:
        if name.startswith(PROJECTOR_PREFIX):
            continue
        data = tensor.detach().contiguous().to(device = "cpu")
        # 按字节宽度换成整数视图再取字节，bfloat16 等 numpy 不支持的类型也能逐位比较。
        raw = data.view(dtype = _RAW_DTYPES[data.dtype.itemsize])
        digests[name] = hashlib.sha256(data = raw.numpy().tobytes()).hexdigest()
    return digests


def move_batch(batch, device, feature_dtype):
    """把模型输入搬到 GPU；梅尔特征转成音频塔的精度。

    有预先算好的元数据时，音频塔直接用分好块的特征 audio_padded_feature，整段特征 input_features 不再上传。
    """
    precomputed = "position_ids" in batch
    moved = {}
    for key in MODEL_INPUT_KEYS + (METADATA_KEYS if precomputed else ()):
        if precomputed and key == "input_features":
            continue
        tensor = batch[key].to(device = device, non_blocking = True)
        if key in ("input_features", "audio_padded_feature"):
            tensor = tensor.to(dtype = feature_dtype)
        moved[key] = tensor
    moved["labels"] = batch["labels"].to(device = device, non_blocking = True)
    if precomputed:
        moved["audio_max_seqlen"] = batch["audio_max_seqlen"]
    return moved


class LinearCrossEntropy(torch.autograd.Function):
    """把词表投影和交叉熵合在一起分块计算，返回每个位置的损失和预测是否正确。

    词表约 25 万，一次算出全部位置的 float32 分数和它的梯度，批量大时要占十几 GB 显存。
    输出层冻结，只需要对隐状态求梯度，所以前向时逐块算出分数、损失以及损失对隐状态的梯度，
    分数用完即丢；反传时把保存的梯度乘上传回的系数。数值与先投影再 F.cross_entropy 相同。
    """

    @staticmethod
    def forward(ctx, hidden_states, weight, targets, chunk_size):
        count = hidden_states.shape[0]
        losses = torch.empty(size = (count,), device = hidden_states.device, dtype = torch.float32)
        correct = torch.empty(size = (count,), device = hidden_states.device, dtype = torch.bool)
        need_grad = ctx.needs_input_grad[0]
        grads = torch.empty_like(input = hidden_states) if need_grad else None
        for start in range(0, count, chunk_size):
            end = min(start + chunk_size, count)
            logits = F.linear(input = hidden_states[start:end], weight = weight).float()
            target = targets[start:end]
            log_normalizer = torch.logsumexp(input = logits, dim = -1)
            target_logits = logits.gather(dim = -1, index = target[:, None]).squeeze(dim = -1)
            losses[start:end] = log_normalizer - target_logits
            correct[start:end] = logits.argmax(dim = -1) == target
            if need_grad:
                # 交叉熵对分数的梯度是 softmax 减去目标位置的 one-hot，原地计算以节省显存。
                probs = logits.sub_(other = log_normalizer[:, None]).exp_()
                rows = torch.arange(end - start, device = probs.device)
                probs[rows, target] -= 1
                grads[start:end] = probs.to(dtype = weight.dtype) @ weight
        ctx.save_for_backward(grads)
        ctx.mark_non_differentiable(correct)
        return losses, correct

    @staticmethod
    def backward(ctx, grad_losses, grad_correct):
        (grads,) = ctx.saved_tensors
        grad_hidden = (grads.float() * grad_losses[:, None]).to(dtype = grads.dtype)
        return grad_hidden, None, None, None


def token_losses_precomputed(model, inputs, chunk_size = 1024):
    """与 token_losses 相同的计算，使用 data.batch_metadata 预先算好的位置与索引，前向中不让 CPU 等 GPU。

    依次做阶段 1 模型 forward 中的各步：文字嵌入、音频塔（传入预先算好的分块与窗口）、projector、把音频特征
    填入音频标记的位置、语言模型（传入预先算好的位置编号），再在计损失的位置上做词表投影与交叉熵。
    各步调用的模块和精度转换与模型代码相同，结果逐位一致；省去的只是模型代码中逐条样本的计算和长度核对
    （音频标记数与音频特征数是否一致，在 CPU 上已由 processor 保证）。
    """
    from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask

    inner = model.model
    embeds = inner.get_input_embeddings()(input = inputs["input_ids"])
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
    )
    hidden_states = outputs.last_hidden_state
    flat = hidden_states.reshape(shape = (-1, hidden_states.shape[-1])).index_select(dim = 0, index = inputs["loss_index"])
    losses, correct = LinearCrossEntropy.apply(
        flat,
        model.lm_head.weight,
        inputs["loss_targets"],
        chunk_size,
    )
    return losses, correct, inputs["loss_rows"]


def token_losses(model, inputs, chunk_size = 1024):
    """前向一次，返回每个计损失位置的 (损失, 是否预测正确, 所在的样本行号)。

    音频塔的参数和输入都不需要梯度，autograd 不会为它保存中间结果，效果等同于 no_grad；
    梯度从损失经过冻结的语言模型传到输入嵌入中的音频位置，再传回 projector。
    语言模型只输出最后一层隐状态，词表投影只在计损失的位置上分块进行。
    批次带有预先算好的元数据时改用 token_losses_precomputed。
    """
    if "position_ids" in inputs:
        return token_losses_precomputed(model = model, inputs = inputs, chunk_size = chunk_size)
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
    losses, correct = LinearCrossEntropy.apply(
        hidden_states[loss_mask],
        model.lm_head.weight,
        shifted_labels[loss_mask],
        chunk_size,
    )
    rows = loss_mask.nonzero()[:, 0]
    return losses, correct, rows


def target_loss(model, inputs):
    """返回 (损失之和, 计损失的 token 数, 预测正确的 token 数)。"""
    losses, correct, _ = token_losses(model = model, inputs = inputs)
    return losses.sum(), losses.numel(), correct.sum()
