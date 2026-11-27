"""思考能力检查（任务 #71）：用固定的 200 条提问开启思考生成，看 </think> 能否写完、推理与回答多长、是否陷入循环。

    flock <gpu.lock> python -m pipeline.eval.think_check --model-dir <融合模型目录> \\
        [--language-model <检查点>/language_model.safetensors] \\
        --replay-dir <回放目录> --output-dir <输出目录> [--attn-implementation flash_attention_2] [--fit-memory]
    flock <gpu.lock> python -m pipeline.eval.think_check --model-dir <融合模型目录> \\
        --replay-dir <回放目录> --output-dir <输出目录> --smoke

提问：回放数据 free_chat_think 评测集（不参与任何训练）中参考模型自己写的用户提问，按提示顺序取前 --count 条（默认
200）用户轮正常结束、token 保存在 raw/ 中的样本；生成时因循环整条丢弃的样本只记了元数据、没有用户轮的 token，不在
其中（评测集 615 条中有 39 条）。提示就是生成这组数据时的写法，直接用保存的 token id：
<|im_start|>user\\n{用户提问}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n，即聊天模板 enable_thinking = True
时的生成提示，没有系统消息。200 条提示长 11–219 个 token（中位数 16）。

生成：温度 --temperature（默认 1）、不设 top-k / top-p，与回放数据的采样设置相同，参考模型在同样设置下的结果可以直接与
回放数据的统计（#70：无循环样本中 93% 写完 </think>，推理中位数约 2,170 token）对照；每批开始前按 --seed 固定随机数。
最多 --max-new-tokens（默认 16,384，与回放数据相同）个新 token，在 <|im_end|> 或 <|endoftext|> 处停止；左填充，transformers
的 generate。批内全部样本结束或到上限时这一批结束，所以一批的用时由最长的样本决定：参考模型约 6% 的样本一直写到上限，
200 条一批时几乎每批都要生成满 16k 步。--max-new-tokens 降低时，超过上限才写完 </think> 的样本记为没有写完，比较
不同模型时用同一个上限。

注意力实现（--attn-implementation，默认 sdpa）：只改变语言模型 6 个全注意力层的计算。18 个线性注意力层（Gated DeltaNet）
不经过这项选择，始终用 fla 与 causal-conv1d 的内核；音频塔与视觉塔不参与文本生成，保持 sdpa。flash_attention_2 按
FLASH_ATTENTION 载入（与训练拼接批次时 pipeline/train/model.py 的 PACKED_ATTENTION 相同），用 flash-attn 的
FlashAttention-2。transformers 5.16.1 中左填充的处理：预填充时，全注意力层由 flash_attention_mask 得到二维填充掩码，
_upad_input 去掉填充位置，按每条的真实长度做变长注意力（flash_attn_varlen_func）；逐 token 生成时掩码同样是二维的，每一步
按它取出各条有效的 K、V，再做变长注意力，每条只有一个查询。线性注意力层的掩码（create_recurrent_attention_mask）与注意力
实现无关：预填充时把填充位置的隐状态置零，递推状态因此保持为零；逐 token 生成时新 token 都有效，不用掩码。两种实现算的是
同一个注意力，只差在 bfloat16 的舍入。flash_attention_2 载入失败，或者生成时出错（显存不足以外的错误）时，整项改用 sdpa
从头重做；summary.json 的 attn_implementation_used 记下实际的实现，attn_fallback_reason 记下原因。比较各模型时要用同一种
实现：本机驱动程序（pipeline/eval/c71_local.py）先做一次冒烟比较，再给所有模型统一指定。

显存：生成到上限时一条样本占用的显存主要是全注意力层的 KV 缓存，另有一层注意力计算时的临时张量和线性注意力层的递推
状态。临时张量取决于注意力实现：sdpa 在带掩码（左填充）时把 K、V 复制到全部 8 个查询头（repeat_kv）；flash_attention_2
每一步把这一层的 K、V 排成按 token 连续的张量，再取出有效的部分，各复制两次，没有按查询头的复制。2B 模型、提示补齐到 219
个 token、生成 16,384 个 token 时，每条 sdpa 约 0.355 GiB、flash_attention_2 约 0.260 GiB（其中 KV 缓存 0.19 GiB）。一批
200 条分别约需 71 GiB 与 52 GiB，Colab G4（96 GB）放得下，RTX 5090（32 GB）放不下：载入 5.3 GiB 的融合模型后约剩 26 GiB，
两种实现都分成 3 批、每批 67 条（flash_attention_2 每批最多约 92 条，分成 2 批需要约 28 GiB 的剩余显存）。--fit-memory
在载入模型后按剩余显存（减去 --memory-reserve-gb）和实际使用的注意力实现估算一批最多放几条（per_sample_bytes），超过
--batch-size 时把提问平均分成几批。批的划分只改变随机数流（每批开始前按 --seed 加批序号固定随机数），提问、温度和上限都
不变，比较时统计意义相同。任何一批显存不足时，这一批对半拆开重试（前一半沿用这一批的种子，后一半加 SPLIT_SEED_OFFSET）；
summary.json 的 batches 记下实际的各批条数。

冒烟比较（--smoke，输出 summary.json 与 summary.md）：从前 --count 条提问中按长度等距取 --smoke-prompts 条（含最短与最长，
左填充最多），用 sdpa 贪心生成最多 --smoke-max-new-tokens 个 token，记下每一步概率最高的 SMOKE_TOP 个 token 的对数概率；
再用 flash_attention_2 重新载入，每一步强制选 sdpa 选出的 token，在完全相同的前缀上比较两边的分布：每一步的总变差距离
（这些 token 与其余部分合起来算，0 为相同，1 为完全不同）、所选 token 的对数概率之差、概率最高的 token 是否相同。每条的
平均总变差不超过 SMOKE_TV_MAX、每条所选 token 对数概率之差的平均不超过 SMOKE_CHOSEN_MAX（单位 nat）、概率最高的 token
相同的比例不低于 SMOKE_AGREE_MIN、比较的步数不少于 SMOKE_MIN_STEPS 时，summary.json 的 use 为 flash_attention_2，否则为
sdpa；flash_attention_2 载入或运行出错时也是 sdpa。填充处理有错时，填充多的几条看到的上下文不同，这几条的差别远大于
bfloat16 舍入带来的差别。

指标（summary.json、summary.md；逐条结果在 generations.jsonl，生成的 token id 在 generations.ids.i32）：
- closed：写出了 </think> 的比例；ended：在上限之前写出结束符的比例；capped：写到上限的比例。
- loop：生成内容出现循环（原样或近似，pipeline/replay/degeneration.py 的 degeneration_onset，与回放数据丢弃循环的规则
  相同）的比例。
- reasoning_tokens：写完 </think> 的样本中推理的 token 数（含 </think>）；answer_tokens：写完 </think> 且正常结束的样本中
  回答的 token 数（含结束符）。给出平均、中位数和 90% 分位数。

vLLM 引擎（--engine vllm，用户 2026-11-27 决定本机评测改用 vLLM 的 docker 镜像，见 pipeline/eval/vllm_host.py）：提问、
温度、上限、结束符、指标和输出文件都不变，生成换成 vLLM 的连续批处理（一条写完补进一条，不再按批等最长的样本）。每条
提问有自己的种子（--seed 加提问的序号）：vLLM 的采样噪声只由种子和位置决定，与同时生成的其他提问无关；transformers 版
每批共用一个随机数流，两版从同一个分布采样、随机数不同，逐条结果不同，统计量应在抽样误差之内一致。summary.json 中
engine 为 vllm，vllm 记下设置与用时；注意力实现由 vLLM 自己选择（attn_implementation_used 为 None）。
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np

THINK_END_ID = 248069
STOP_IDS = (248046, 248044)
PAD_ID = 248044
# 显存不足、一批对半拆开时，后一半的随机数种子加上这个数，与其他批的种子（--seed 加批序号）不重合。
SPLIT_SEED_OFFSET = 10000
ATTENTION_CHOICES = ("sdpa", "flash_attention_2")
# flash_attention_2 时传给 from_pretrained 的设置：键为子配置名，"" 是整个模型。只有语言模型（text_config）用
# FlashAttention-2，与 pipeline/train/model.py 的 PACKED_ATTENTION 相同（这里不导入训练代码，它依赖 mm_assistant_mask）。
FLASH_ATTENTION = {
    "": "sdpa",
    "text_config": "flash_attention_2",
    "audio_config": "sdpa",
    "vision_config": "sdpa",
}
# 冒烟比较的设置与判定标准（见开头「冒烟比较」）。
SMOKE_PROMPTS = 4
SMOKE_MAX_NEW = 256
SMOKE_TOP = 32
SMOKE_TV_MAX = 0.05
SMOKE_CHOSEN_MAX = 0.1
SMOKE_AGREE_MIN = 0.9
SMOKE_MIN_STEPS = 32


def load_prompts(replay_dir, count, split = "eval"):
    """free_chat_think 中前 count 条有用户轮的样本：[(sample_id, 提示 token id 列表)]。"""
    from pipeline.replay.store import ShardReader

    directory = Path(replay_dir) / "free_chat_think" / split
    reader = ShardReader(raw_dir = directory / "raw")
    records = []
    with open(file = directory / "samples.jsonl", encoding = "utf-8") as handle:
        for line in handle:
            records.append(json.loads(s = line))
    records.sort(key = lambda record: record["prompt_index"])
    prompts = []
    for record in records:
        pieces = record["pieces"]
        roles = [piece.get("gen") for piece in pieces]
        if "user" not in roles:
            continue
        user = pieces[roles.index("user")]
        if user.get("finish_reason") != "stop" or roles.index("user") + 1 >= len(pieces) or "ids" not in pieces[roles.index("user") + 1]:
            continue
        ids = []
        for piece in pieces[: roles.index("user") + 2]:
            if "gen" in piece:
                ids += np.asarray(a = reader.piece(number = int(record["shard"]), piece = piece)[: int(piece.get("keep", piece["length"]))]).tolist()
            else:
                ids += piece["ids"]
        prompts.append((record["sample_id"], [int(token) for token in ids]))
        if len(prompts) == count:
            break
    return prompts


def attention_setting(name):
    """--attn-implementation 的取值换成 from_pretrained 的 attn_implementation：sdpa 时整个模型用 sdpa，flash_attention_2
    时为 FLASH_ATTENTION（只有语言模型的全注意力层用 FlashAttention-2）。"""
    if name == "sdpa":
        return "sdpa"
    if name == "flash_attention_2":
        return dict(FLASH_ATTENTION)
    raise ValueError(f"未知的注意力实现 {name}，可选 {ATTENTION_CHOICES}")


def text_attention(model):
    """语言模型全注意力层实际使用的实现（语言模型子配置的 _attn_implementation）。"""
    return model.model.language_model.config._attn_implementation


def load_for_attention(model_dir, language_model, attention):
    """按 attention 载入融合模型（全量微调时换上 language_model），并确认语言模型的全注意力层用的正是它。"""
    from pipeline.eval.run_eval import load_model

    model, _ = load_model(
        model_dir = model_dir,
        projector_path = None,
        language_model_path = language_model,
        attn_implementation = attention_setting(name = attention),
    )
    used = text_attention(model = model)
    if used != attention:
        raise RuntimeError(f"语言模型的注意力实现是 {used}，要求的是 {attention}")
    return model


def release_memory():
    """丢掉模型的引用之后调用：回收 Python 对象并把空出的显存还给 CUDA。"""
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def left_padded(prompts):
    """左填充成一批：(input_ids, attention_mask, 补齐后的提示长度)，都在 CPU 上。"""
    import torch

    longest = max(len(ids) for _, ids in prompts)
    input_ids = torch.full(size = (len(prompts), longest), fill_value = PAD_ID, dtype = torch.long)
    attention_mask = torch.zeros(size = (len(prompts), longest), dtype = torch.long)
    for row, (_, ids) in enumerate(prompts):
        input_ids[row, longest - len(ids):] = torch.tensor(data = ids)
        attention_mask[row, longest - len(ids):] = 1
    return input_ids, attention_mask, longest


def generate_batch(model, prompts, max_new_tokens, temperature, seed):
    """左填充生成一批，返回每条新生成的 token（到第一个结束符为止，含结束符）。"""
    import torch

    device = next(model.parameters()).device
    input_ids, attention_mask, longest = left_padded(prompts = prompts)
    torch.manual_seed(seed = seed)
    with torch.inference_mode():
        output = model.generate(
            input_ids = input_ids.to(device = device),
            attention_mask = attention_mask.to(device = device),
            mm_token_type_ids = torch.zeros_like(input = input_ids).to(device = device),
            max_new_tokens = max_new_tokens,
            do_sample = True,
            temperature = temperature,
            top_k = 0,
            top_p = 1.0,
            use_cache = True,
            pad_token_id = PAD_ID,
            eos_token_id = list(STOP_IDS),
        )
    results = []
    for row in output[:, longest:].cpu().numpy():
        stops = np.nonzero(np.isin(element = row, test_elements = STOP_IDS))[0]
        results.append(row[: stops[0] + 1] if len(stops) else row)
    return results


def per_sample_bytes(config, total_tokens, attn_implementation = "sdpa", sampling_copies = 4):
    """一条样本生成到 total_tokens（提示加新 token）时占用的显存估计（字节），config 为语言模型的 text_config。

    - 全注意力层的 KV 缓存（bfloat16）：层数 × 2（K、V）× KV 头数 × 每头维数 × 2 字节 × token 数；
    - 一层注意力计算时的临时张量（以一份 K 的大小为单位，KV 头数 × 每头维数 × 2 字节 × token 数）：
      sdpa 为缓存拼接（torch.cat）时多出的一份，加上左填充时把 K、V 复制到全部查询头（repeat_kv）的两份；
      flash_attention_2 为 _upad_input 中 K、V 各自先排成按 token 连续的一份、再取出有效部分的一份，最多同时存在三份
      （取出的 K，与 V 的两份），缓存拼接在这之前完成，不同时存在；
    - 线性注意力层的递推状态（float32，值头数 × 键维数 × 值维数）与短卷积状态（bfloat16）；
    - 采样时一个位置的词表分布，按 sampling_copies 份 float32 计。
    """
    layer_types = list(config.layer_types)
    full = layer_types.count("full_attention")
    linear = len(layer_types) - full
    head_bytes = config.head_dim * 2
    kv_bytes = config.num_key_value_heads * head_bytes * total_tokens
    kv_cache = full * 2 * kv_bytes
    if attn_implementation == "flash_attention_2":
        temporary = 3 * kv_bytes
    elif attn_implementation == "sdpa":
        temporary = kv_bytes
        if config.num_attention_heads != config.num_key_value_heads:
            temporary += 2 * config.num_attention_heads * head_bytes * total_tokens
    else:
        raise ValueError(f"未知的注意力实现 {attn_implementation}")
    recurrent = linear * config.linear_num_value_heads * config.linear_key_head_dim * config.linear_value_head_dim * 4
    conv_dim = 2 * config.linear_num_key_heads * config.linear_key_head_dim + config.linear_num_value_heads * config.linear_value_head_dim
    conv = linear * conv_dim * config.linear_conv_kernel_dim * 2
    sampling = sampling_copies * config.vocab_size * 4
    return kv_cache + temporary + recurrent + conv + sampling


def fit_batch_size(requested, count, sample_bytes, free_bytes, reserve_bytes):
    """剩余显存放得下的批大小：一批最多 (free_bytes − reserve_bytes) ÷ sample_bytes 条（至少 1 条），不超过 requested；
    需要分成几批时，把 count 条平均分开（例如 200 条、最多 92 条时分成 3 批，每批 67 条）。"""
    cap = max(1, int((free_bytes - reserve_bytes) // sample_bytes))
    size = max(1, min(requested, cap, count))
    batches = math.ceil(count / size)
    return math.ceil(count / batches)


def generate_resilient(model, prompts, max_new_tokens, temperature, seed, generate = None, log = print):
    """生成一批；显存不足时把这一批对半拆开重试（前一半用 seed，后一半用 seed + SPLIT_SEED_OFFSET），一直拆到单条。

    返回 (结果列表, 实际的各批条数)。generate 默认为 generate_batch，测试时可以换成假的函数。
    """
    import torch

    generate = generate_batch if generate is None else generate
    try:
        outputs = generate(
            model = model,
            prompts = prompts,
            max_new_tokens = max_new_tokens,
            temperature = temperature,
            seed = seed,
        )
        return list(outputs), [len(prompts)]
    except torch.OutOfMemoryError:
        if len(prompts) == 1:
            raise
    # 异常对象引用着出错时的栈帧，在 except 之外清理，生成到一半的缓存才能释放
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    half = (len(prompts) + 1) // 2
    log(f"{len(prompts)} 条一批显存不足，拆成 {half} 条与 {len(prompts) - half} 条重试")
    first, first_sizes = generate_resilient(
        model = model,
        prompts = prompts[:half],
        max_new_tokens = max_new_tokens,
        temperature = temperature,
        seed = seed,
        generate = generate,
        log = log,
    )
    second, second_sizes = generate_resilient(
        model = model,
        prompts = prompts[half:],
        max_new_tokens = max_new_tokens,
        temperature = temperature,
        seed = seed + SPLIT_SEED_OFFSET,
        generate = generate,
        log = log,
    )
    return first + second, first_sizes + second_sizes


def generate_all(model, prompts, batch_size, max_new_tokens, temperature, seed, generate = None, log = print):
    """按 batch_size 分批生成全部提问（第 number 批用种子 seed + number），返回 (按提问顺序的结果, 实际的各批条数)。"""
    outputs = []
    batches = []
    start = time.perf_counter()
    for number, begin in enumerate(range(0, len(prompts), batch_size)):
        batch = prompts[begin: begin + batch_size]
        results, sizes = generate_resilient(
            model = model,
            prompts = batch,
            max_new_tokens = max_new_tokens,
            temperature = temperature,
            seed = seed + number,
            generate = generate,
            log = log,
        )
        outputs += results
        batches += sizes
        log(f"第 {number + 1} 批 {len(batch)} 条完成，已用 {time.perf_counter() - start:.0f} 秒")
    return outputs, batches


def load_with_fallback(load, requested, log = print):
    """按 requested 载入模型；flash_attention_2 载入出错时改用 sdpa。返回 (模型, 实际的实现, 改用的原因或 None)。

    load(attention = 名称) 返回模型，测试时可以换成假的函数。
    """
    failure = None
    try:
        return load(attention = requested), requested, None
    except Exception as error:
        if requested == "sdpa":
            raise
        failure = f"{type(error).__name__}: {error}"[:500]
    release_memory()
    log(f"flash_attention_2 载入失败（{failure}），改用 sdpa")
    return load(attention = "sdpa"), "sdpa", f"载入时出错：{failure}"


def think_generate(load, prompts, requested, choose_batch_size, max_new_tokens, temperature, seed, generate = None, log = print):
    """载入模型并生成全部提问；flash_attention_2 载入或生成时出错（显存不足以外）时，整项改用 sdpa 从头重做，
    同一个模型的结果始终来自同一种实现。

    choose_batch_size(model = 模型, attention = 实现) 返回批大小。返回 {"outputs", "batches", "batch_size", "used", "reason",
    "seconds"}，seconds 为最后一次（成功的）生成的用时。
    """
    import torch

    model, used, reason = load_with_fallback(load = load, requested = requested, log = log)
    log(f"注意力实现：{used}（语言模型的全注意力层）" + (f"；要求 {requested}，{reason}" if reason else ""))
    while True:
        batch_size = choose_batch_size(model = model, attention = used)
        start = time.perf_counter()
        failure = None
        try:
            outputs, batches = generate_all(
                model = model,
                prompts = prompts,
                batch_size = batch_size,
                max_new_tokens = max_new_tokens,
                temperature = temperature,
                seed = seed,
                generate = generate,
                log = log,
            )
        except torch.OutOfMemoryError:
            raise
        except Exception as error:
            if used == "sdpa":
                raise
            failure = f"{type(error).__name__}: {error}"[:500]
        if failure is None:
            return {
                "outputs": outputs,
                "batches": batches,
                "batch_size": batch_size,
                "used": used,
                "reason": reason,
                "seconds": round(number = time.perf_counter() - start, ndigits = 1),
            }
        # 异常对象已随 except 块结束释放，这里丢掉模型，生成到一半的缓存与模型的显存才能还回去
        log(f"flash_attention_2 生成时出错（{failure}），整项改用 sdpa 从头重做")
        model = None
        release_memory()
        model = load(attention = "sdpa")
        used = "sdpa"
        reason = f"生成时出错：{failure}"
        log("注意力实现：sdpa（语言模型的全注意力层）")


def smoke_prompts(prompts, count):
    """冒烟比较用的提问：按提示长度排序后等距取 count 条（含最短与最长），保持原来的先后顺序。"""
    if count >= len(prompts):
        return list(prompts)
    order = sorted(range(len(prompts)), key = lambda index: (len(prompts[index][1]), index))
    if count == 1:
        picks = {order[-1]}
    else:
        picks = {order[round(number = step * (len(order) - 1) / (count - 1))] for step in range(count)}
    return [prompts[index] for index in sorted(picks)]


def greedy_trace(model, prompts, max_new_tokens, forced = None, top = SMOKE_TOP):
    """左填充贪心生成一批，逐步记下 float32 的对数概率。

    forced 为 None 时记下每一步概率最高的 top 个 token（ids）与它们的对数概率（logprobs）；forced 为另一次 greedy_trace 的
    结果时，每一步强制选 forced 在这一步生成的 token，记下 forced 的 ids 处的对数概率，前缀因此与 forced 完全相同。
    两种情况都记下每一步模型自己概率最高的 token（argmax）。返回 {"tokens": (条数, 步数), "argmax": (条数, 步数),
    "logprobs": (条数, 步数, top), "ids": (条数, 步数, top)（只在 forced 为 None 时）, "prompt_tokens", "seconds"}。
    """
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList

    forced_tokens = None if forced is None else torch.as_tensor(data = forced["tokens"])
    reference_ids = None if forced is None else torch.as_tensor(data = forced["ids"])

    class Trace(LogitsProcessor):
        def __init__(self):
            self.ids = []
            self.logprobs = []
            self.argmax = []

        def __call__(self, input_ids, scores):
            step = len(self.argmax)
            logprobs = torch.log_softmax(input = scores.float(), dim = -1)
            self.argmax.append(scores.argmax(dim = -1).cpu())
            if forced_tokens is None:
                values, ids = torch.topk(input = logprobs, k = top, dim = -1)
                self.ids.append(ids.cpu())
                self.logprobs.append(values.cpu())
                return scores
            if step >= forced_tokens.shape[1]:
                raise RuntimeError(f"强制生成走到第 {step + 1} 步，超过了参照的 {forced_tokens.shape[1]} 步")
            ids = reference_ids[:, step].to(device = scores.device)
            self.logprobs.append(torch.gather(input = logprobs, dim = -1, index = ids).cpu())
            target = forced_tokens[:, step].to(device = scores.device)
            only = torch.full_like(input = scores, fill_value = float("-inf"))
            only.scatter_(dim = -1, index = target[:, None], value = 0.0)
            return only

    trace = Trace()
    device = next(model.parameters()).device
    input_ids, attention_mask, longest = left_padded(prompts = prompts)
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            input_ids = input_ids.to(device = device),
            attention_mask = attention_mask.to(device = device),
            mm_token_type_ids = torch.zeros_like(input = input_ids).to(device = device),
            max_new_tokens = max_new_tokens,
            do_sample = False,
            use_cache = True,
            pad_token_id = PAD_ID,
            eos_token_id = list(STOP_IDS),
            logits_processor = LogitsProcessorList([trace]),
        )
    result = {
        "tokens": output[:, longest:].cpu().numpy(),
        "argmax": torch.stack(tensors = trace.argmax, dim = 1).numpy(),
        "logprobs": torch.stack(tensors = trace.logprobs, dim = 1).numpy(),
        "prompt_tokens": [len(ids) for _, ids in prompts],
        "seconds": round(number = time.perf_counter() - start, ndigits = 1),
    }
    if forced is None:
        result["ids"] = torch.stack(tensors = trace.ids, dim = 1).numpy()
    return result


def compare_traces(reference, candidate):
    """在相同前缀上逐步比较两次 greedy_trace（candidate 强制跟随 reference）。每条只比到 reference 的第一个结束符（含）。

    返回每条的 {"prompt_tokens", "steps", "tv_mean", "tv_max", "chosen_diff_mean", "chosen_diff_max", "argmax_agree"}：
    tv 为每一步的总变差距离（reference 的前 top 个 token 各自的概率差，加上其余部分合计的概率差，之和的一半），
    chosen_diff 为所选 token（reference 概率最高的 token）的对数概率之差的绝对值，argmax_agree 为 candidate 概率最高的
    token 与所选 token 相同的比例。
    """
    rows = []
    for row, tokens in enumerate(reference["tokens"]):
        stops = np.nonzero(np.isin(element = tokens, test_elements = STOP_IDS))[0]
        length = int(stops[0]) + 1 if len(stops) else len(tokens)
        ref_lp = reference["logprobs"][row, :length].astype(np.float64)
        cand_lp = candidate["logprobs"][row, :length].astype(np.float64)
        p_ref = np.exp(ref_lp)
        p_cand = np.exp(cand_lp)
        rest = np.abs(np.clip(a = 1 - p_ref.sum(axis = -1), a_min = 0, a_max = None) - np.clip(a = 1 - p_cand.sum(axis = -1), a_min = 0, a_max = None))
        tv = 0.5 * (np.abs(p_ref - p_cand).sum(axis = -1) + rest)
        chosen = np.abs(ref_lp[:, 0] - cand_lp[:, 0])
        agree = candidate["argmax"][row, :length] == tokens[:length]
        rows.append({
            "prompt_tokens": int(reference["prompt_tokens"][row]),
            "steps": length,
            "tv_mean": round(number = float(tv.mean()), ndigits = 5),
            "tv_max": round(number = float(tv.max()), ndigits = 5),
            "chosen_diff_mean": round(number = float(chosen.mean()), ndigits = 5),
            "chosen_diff_max": round(number = float(chosen.max()), ndigits = 5),
            "argmax_agree": round(number = float(agree.mean()), ndigits = 4),
        })
    return rows


def smoke_verdict(rows, tv_max = SMOKE_TV_MAX, chosen_max = SMOKE_CHOSEN_MAX, agree_min = SMOKE_AGREE_MIN, min_steps = SMOKE_MIN_STEPS):
    """按 compare_traces 的结果判断两种实现是否一致：返回 (是否一致, 说明)。

    每条的平均总变差不超过 tv_max、所选 token 对数概率之差的平均不超过 chosen_max（分布接近均匀时总变差反映不出上下文的
    差别，对数概率之差仍然能），概率最高的 token 相同的比例不低于 agree_min，比较的步数不少于 min_steps。
    """
    steps = sum(row["steps"] for row in rows)
    if steps < min_steps:
        return False, f"只比较了 {steps} 步，少于 {min_steps} 步"
    agree = sum(row["argmax_agree"] * row["steps"] for row in rows) / steps
    bad_tv = [row for row in rows if row["tv_mean"] > tv_max]
    if bad_tv:
        return False, "平均总变差超过 {}：{}".format(tv_max, "，".join(f"提示 {row['prompt_tokens']} token 的一条 {row['tv_mean']:.3f}" for row in bad_tv))
    bad_chosen = [row for row in rows if row["chosen_diff_mean"] > chosen_max]
    if bad_chosen:
        return False, "所选 token 的对数概率之差平均超过 {}：{}".format(
            chosen_max,
            "，".join(f"提示 {row['prompt_tokens']} token 的一条 {row['chosen_diff_mean']:.3f}" for row in bad_chosen),
        )
    if agree < agree_min:
        return False, f"概率最高的 token 相同的比例 {agree:.3f}，低于 {agree_min}"
    worst_tv = max(row["tv_mean"] for row in rows)
    worst_chosen = max(row["chosen_diff_mean"] for row in rows)
    return True, (
        f"{len(rows)} 条共 {steps} 步，每条平均总变差最大 {worst_tv:.4f}，所选 token 对数概率之差的平均最大 {worst_chosen:.4f}，"
        f"概率最高的 token 相同的比例 {agree:.3f}"
    )


def run_smoke(load, prompts, max_new_tokens, log = print):
    """冒烟比较：sdpa 贪心生成作参照，flash_attention_2 强制跟随比较分布。返回写进 summary.json 的内容（use 为结论）。

    flash_attention_2 载入或运行出错（包括显存不足）时结论为 sdpa；sdpa 参照本身出错时抛出异常。
    """
    model = load(attention = "sdpa")
    reference = greedy_trace(model = model, prompts = prompts, max_new_tokens = max_new_tokens)
    model = None
    release_memory()
    log(f"sdpa 参照：{len(prompts)} 条，提示 {reference['prompt_tokens']} token，生成 {reference['tokens'].shape[1]} 步，用时 {reference['seconds']} 秒")
    candidate = None
    failure = None
    try:
        model = load(attention = "flash_attention_2")
        candidate = greedy_trace(
            model = model,
            prompts = prompts,
            max_new_tokens = max_new_tokens,
            forced = reference,
        )
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"[:500]
    model = None
    release_memory()
    summary = {
        "smoke": True,
        "prompts": [sample_id for sample_id, _ in prompts],
        "prompt_tokens": reference["prompt_tokens"],
        "max_new_tokens": max_new_tokens,
        "thresholds": {
            "tv_max": SMOKE_TV_MAX,
            "chosen_max": SMOKE_CHOSEN_MAX,
            "agree_min": SMOKE_AGREE_MIN,
            "min_steps": SMOKE_MIN_STEPS,
            "top": SMOKE_TOP,
        },
        "seconds": {"sdpa": reference["seconds"]},
    }
    if failure is not None:
        summary.update(use = "sdpa", agree = False, reason = f"flash_attention_2 不能运行：{failure}", rows = [])
    else:
        rows = compare_traces(reference = reference, candidate = candidate)
        ok, reason = smoke_verdict(rows = rows)
        summary["seconds"]["flash_attention_2"] = candidate["seconds"]
        summary.update(use = "flash_attention_2" if ok else "sdpa", agree = ok, reason = reason, rows = rows)
    log(f"冒烟比较的结论：{summary['use']}（{summary['reason']}）")
    return summary


def format_smoke_markdown(summary):
    lines = [
        f"思考检查的冒烟比较：sdpa 与 flash_attention_2，{len(summary['prompts'])} 条提问，贪心最多 {summary['max_new_tokens']} 个 token",
        "",
        f"结论：{summary['use']}（{summary['reason']}）",
        "",
    ]
    if summary["rows"]:
        lines += [
            "| 提示 token | 比较步数 | 平均总变差 | 最大总变差 | 所选 token 对数概率差（平均 / 最大） | 最高 token 相同 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in summary["rows"]:
            lines.append(
                f"| {row['prompt_tokens']} | {row['steps']} | {row['tv_mean']:.4f} | {row['tv_max']:.4f} | "
                f"{row['chosen_diff_mean']:.4f} / {row['chosen_diff_max']:.4f} | {100 * row['argmax_agree']:.1f}% |"
            )
        lines.append("")
    return "\n".join(lines)


def analyze(ids, max_new_tokens):
    """一条生成结果的各项指标。"""
    from pipeline.replay.degeneration import degeneration_onset

    ids = np.asarray(a = ids, dtype = np.int64)
    ended = len(ids) > 0 and int(ids[-1]) in STOP_IDS
    ends = np.nonzero(ids == THINK_END_ID)[0]
    closed = len(ends) > 0
    reasoning = int(ends[0]) + 1 if closed else None
    return {
        "tokens": int(len(ids)),
        "ended": bool(ended),
        "capped": bool(not ended and len(ids) >= max_new_tokens),
        "closed": bool(closed),
        "reasoning_tokens": reasoning,
        "answer_tokens": int(len(ids)) - reasoning if closed and ended else None,
        "loop": degeneration_onset(ids = ids) is not None,
    }


def distribution(values):
    if not values:
        return None
    values = np.asarray(a = values, dtype = np.float64)
    return {
        "count": int(len(values)),
        "mean": round(number = float(values.mean()), ndigits = 1),
        "median": round(number = float(np.median(a = values)), ndigits = 1),
        "p90": round(number = float(np.quantile(a = values, q = 0.9)), ndigits = 1),
    }


def summarize(rows, max_new_tokens):
    count = len(rows)
    return {
        "samples": count,
        "max_new_tokens": max_new_tokens,
        "closed": round(number = sum(row["closed"] for row in rows) / count, ndigits = 4),
        "ended": round(number = sum(row["ended"] for row in rows) / count, ndigits = 4),
        "capped": round(number = sum(row["capped"] for row in rows) / count, ndigits = 4),
        "loop": round(number = sum(row["loop"] for row in rows) / count, ndigits = 4),
        "closed_without_loop": round(number = sum(row["closed"] and not row["loop"] for row in rows) / count, ndigits = 4),
        "reasoning_tokens": distribution(values = [row["reasoning_tokens"] for row in rows if row["closed"]]),
        "answer_tokens": distribution(values = [row["answer_tokens"] for row in rows if row["answer_tokens"] is not None]),
        "tokens": distribution(values = [row["tokens"] for row in rows]),
    }


def format_markdown(summary):
    def pct(value):
        return f"{100 * value:.1f}%"

    def dist(item):
        return "—" if item is None else f"{item['mean']:.0f} / {item['median']:.0f} / {item['p90']:.0f}"

    attention = "vLLM" if summary.get("engine") == "vllm" else f"全注意力层 {summary.get('attn_implementation_used', 'sdpa')}"
    return "\n".join([
        f"模型：{summary['model']}（{summary['samples']} 条提问，温度 {summary['temperature']}，上限 {summary['max_new_tokens']}，"
        f"{attention}）",
        "",
        "| 写完 </think> | 正常结束 | 到上限 | 循环 | 推理 token（平均 / 中位 / 90%） | 回答 token（平均 / 中位 / 90%） |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {pct(summary['closed'])} | {pct(summary['ended'])} | {pct(summary['capped'])} | {pct(summary['loop'])}"
        f" | {dist(summary['reasoning_tokens'])} | {dist(summary['answer_tokens'])} |",
        "",
    ])


def parse_args(argv = None):
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model-dir",
        required = True,
        type = Path,
        help = "融合模型目录",
    )
    parser.add_argument(
        "--language-model",
        default = None,
        type = Path,
        help = "全量微调检查点中的 language_model.safetensors；不给时是参考模型",
    )
    parser.add_argument(
        "--replay-dir",
        required = True,
        type = Path,
        help = "回放数据目录（replay-noloop）",
    )
    parser.add_argument(
        "--count",
        type = int,
        default = 200,
        help = "提问条数",
    )
    parser.add_argument("--max-new-tokens", type = int, default = 16384)
    parser.add_argument(
        "--batch-size",
        type = int,
        default = 200,
        help = "每批条数（一批的用时由最长的样本决定，越大越省时）",
    )
    parser.add_argument("--temperature", type = float, default = 1.0)
    parser.add_argument("--seed", type = int, default = 1234)
    parser.add_argument(
        "--attn-implementation",
        default = "sdpa",
        choices = ATTENTION_CHOICES,
        help = "语言模型全注意力层的实现（见开头「注意力实现」）；flash_attention_2 出错时自动改用 sdpa",
    )
    parser.add_argument(
        "--fit-memory",
        action = "store_true",
        help = "载入模型后按剩余显存估算一批放得下几条，放不下 --batch-size 时平均分成几批（见开头「显存」）",
    )
    parser.add_argument(
        "--memory-reserve-gb",
        type = float,
        default = 2.0,
        help = "--fit-memory 估算时留出的显存（GiB）",
    )
    parser.add_argument(
        "--smoke",
        action = "store_true",
        help = "只做 sdpa 与 flash_attention_2 的冒烟比较（见开头「冒烟比较」），不做思考检查",
    )
    parser.add_argument(
        "--smoke-prompts",
        type = int,
        default = SMOKE_PROMPTS,
        help = "冒烟比较的提问条数",
    )
    parser.add_argument(
        "--smoke-max-new-tokens",
        type = int,
        default = SMOKE_MAX_NEW,
        help = "冒烟比较最多生成的 token 数",
    )
    parser.add_argument("--output-dir", required = True, type = Path)
    from pipeline.eval import vllm_host

    vllm_host.add_arguments(parser = parser)
    args = parser.parse_args(args = argv)
    vllm_host.check_arguments(
        parser = parser,
        args = args,
        default_job_dir = args.output_dir / "vllm-job",
        language_model = args.language_model,
    )
    if args.engine == "vllm" and args.smoke:
        parser.error("--smoke 只比较 transformers 的两种注意力实现，不用 --engine vllm")
    return args


def vllm_generate_all(args, prompts, log):
    """用 vLLM 生成全部提问（每条的种子为 --seed 加序号），返回 (按提问顺序的结果, vLLM 的设置与用时)。"""
    from pipeline.eval import vllm_host

    requests = [
        {"index": index, "seed": args.seed + index, "prompt_token_ids": [int(token) for token in ids]}
        for index, (_, ids) in enumerate(prompts)
    ]
    settings = vllm_host.job_settings(
        args = args,
        max_model_len = max(len(ids) for _, ids in prompts) + args.max_new_tokens,
        max_tokens = args.max_new_tokens,
        temperature = args.temperature,
        top_p = 1.0,
        top_k = 0,
        min_p = 0.0,
        stop_token_ids = list(STOP_IDS),
        seed = args.seed,
    )
    results, record = vllm_host.run_job(
        args = args,
        model_dir = args.model_dir,
        language_model = args.language_model,
        requests = requests,
        settings = settings,
        kind = "think",
        label = args.output_dir.parent.name,
        log = log,
    )
    return [np.asarray(a = results[index]["token_ids"], dtype = np.int64) for index in range(len(prompts))], record


def main(argv = None):
    args = parse_args(argv = argv)
    args.output_dir.mkdir(parents = True, exist_ok = True)
    prompts = load_prompts(replay_dir = args.replay_dir, count = args.count)
    if len(prompts) < args.count:
        print(f"只有 {len(prompts)} 条可用的提问（要求 {args.count}）", flush = True)

    def log(message):
        print(message, flush = True)

    if args.engine == "vllm":
        outputs, record = vllm_generate_all(args = args, prompts = prompts, log = log)
        engine_fields = {
            "engine": "vllm",
            "seed_scheme": "per prompt: seed + prompt index",
            "attn_implementation_requested": None,
            "attn_implementation_used": None,
            "attn_fallback_reason": None,
            "batch_size": None,
            "batches": None,
            "seconds": record.get("seconds"),
            "vllm": record,
        }
        return write_outputs(args = args, prompts = prompts, outputs = outputs, engine_fields = engine_fields)

    def load(attention):
        return load_for_attention(
            model_dir = args.model_dir,
            language_model = args.language_model,
            attention = attention,
        )

    if args.smoke:
        summary = run_smoke(
            load = load,
            prompts = smoke_prompts(prompts = prompts, count = args.smoke_prompts),
            max_new_tokens = args.smoke_max_new_tokens,
            log = log,
        )
        summary["model"] = str(args.language_model or args.model_dir)
        (args.output_dir / "summary.json").write_text(data = json.dumps(obj = summary, ensure_ascii = False, indent = 1) + "\n", encoding = "utf-8")
        (args.output_dir / "summary.md").write_text(data = format_smoke_markdown(summary = summary), encoding = "utf-8")
        return 0

    def choose_batch_size(model, attention):
        if not args.fit_memory:
            return args.batch_size
        import torch

        free_bytes, _ = torch.cuda.mem_get_info()
        sample_bytes = per_sample_bytes(
            config = model.config.text_config,
            total_tokens = max(len(ids) for _, ids in prompts) + args.max_new_tokens,
            attn_implementation = attention,
        )
        size = fit_batch_size(
            requested = args.batch_size,
            count = len(prompts),
            sample_bytes = sample_bytes,
            free_bytes = free_bytes,
            reserve_bytes = int(args.memory_reserve_gb * 2 ** 30),
        )
        log(
            f"剩余显存 {free_bytes / 2 ** 30:.1f} GiB，{attention} 每条约 {sample_bytes / 2 ** 30:.3f} GiB，"
            f"批大小 {size}（要求 {args.batch_size}）"
        )
        return size

    result = think_generate(
        load = load,
        prompts = prompts,
        requested = args.attn_implementation,
        choose_batch_size = choose_batch_size,
        max_new_tokens = args.max_new_tokens,
        temperature = args.temperature,
        seed = args.seed,
        log = log,
    )
    engine_fields = {
        "engine": "hf",
        "attn_implementation_requested": args.attn_implementation,
        "attn_implementation_used": result["used"],
        "attn_fallback_reason": result["reason"],
        "batch_size": result["batch_size"],
        "batches": result["batches"],
        "seconds": result["seconds"],
    }
    return write_outputs(args = args, prompts = prompts, outputs = result["outputs"], engine_fields = engine_fields)


def write_outputs(args, prompts, outputs, engine_fields):
    """逐条指标、generations.jsonl、generations.ids.i32、summary.json 与 summary.md（两种引擎相同，engine_fields 为引擎的记录）。"""
    rows = []
    generated = []
    for (sample_id, prompt), ids in zip(prompts, outputs):
        row = {"sample_id": sample_id, "prompt_tokens": len(prompt), **analyze(ids = ids, max_new_tokens = args.max_new_tokens)}
        rows.append(row)
        generated.append(np.asarray(a = ids, dtype = np.int32))
    offset = 0
    with open(file = args.output_dir / "generations.jsonl", mode = "w", encoding = "utf-8") as handle:
        for row, ids in zip(rows, generated):
            row["offset"] = offset
            offset += len(ids)
            handle.write(json.dumps(obj = row, ensure_ascii = False) + "\n")
    np.concatenate(generated).astype(np.int32).tofile(args.output_dir / "generations.ids.i32")
    summary = {
        "model": str(args.language_model or args.model_dir),
        "temperature": args.temperature,
        "seed": args.seed,
        **engine_fields,
        **summarize(rows = rows, max_new_tokens = args.max_new_tokens),
    }
    (args.output_dir / "summary.json").write_text(data = json.dumps(obj = summary, ensure_ascii = False, indent = 1) + "\n", encoding = "utf-8")
    (args.output_dir / "summary.md").write_text(data = format_markdown(summary = summary), encoding = "utf-8")
    print(json.dumps(obj = summary, ensure_ascii = False), flush = True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
