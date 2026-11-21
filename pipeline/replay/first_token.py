"""只有一个 token 的提示：第一个生成 token 由这里按参考模型的分布抽取，再把两个 token 交给 vLLM。

vLLM 0.28 的 Qwen3.5（线性注意力层 GDN）有一个问题：新请求的提示只有一个 token 时，调度器把它当作解码步，
线性注意力层直接从这个请求分到的状态槽读取递推状态，而这个槽位可能还留着上一个请求的状态（`gdn_attn.py` 调用
`split_decodes_and_prefills` 时没有像 Mamba 那样传 `treat_short_extends_as_decodes = False`）。槽位第一次使用时
状态为零，结果正确；槽位被用过之后，生成的就是乱码。试验中同一个提示 `\\n` 在新槽位上第一个 token 的对数概率与
transformers 一致（-1.758 对 -1.744），在用过的槽位上变成 -1.5 到 -4.2 之间的随机值。两个 token 以上的提示走预填充，
不受影响。

free_raw 的提示只有一个换行，所以第一个生成 token 在这里用 transformers 前向一次、按完整分布抽取（与从同一分布采样
等价），同时记下它的对数概率和 top-20；vLLM 收到的提示是「换行 + 第一个 token」，从第二个 token 开始生成。
"""

import numpy as np

from pipeline.replay.store import TOP_K


def first_token_logprobs(model_dir, start_ids):
    """每个单 token 提示之后下一个 token 的完整对数概率分布，形状 [len(start_ids), 词表大小]（float64）。

    用 transformers 加载参考模型目录（export_lm 的输出），只取其中的语言模型在 GPU 上前向一次（线性注意力层的卷积内核
    只支持 CUDA）。
    """
    import torch
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        pretrained_model_name_or_path = model_dir,
        dtype = torch.bfloat16,
    )
    model.eval()
    model.to(device = "cuda")
    with torch.inference_mode():
        input_ids = torch.tensor([[token] for token in start_ids], device = "cuda")
        hidden = model.model.language_model(input_ids = input_ids, use_cache = False).last_hidden_state[:, -1]
        logits = (hidden @ model.lm_head.weight.T).double()
        return torch.log_softmax(logits, dim = -1).cpu().numpy()


def sample_first(logprobs, rng):
    """从一条完整分布中抽一个 token，返回 prompts.jsonl 中 `first` 字段的内容。"""
    probs = np.exp(logprobs)
    token = int(rng.choice(len(probs), p = probs / probs.sum()))
    top = np.argsort(-logprobs)[:TOP_K]
    return {
        "id": token,
        "logprob": float(logprobs[token]),
        "topk_ids": [int(i) for i in top],
        "topk_lp": [round(float(logprobs[i]), 6) for i in top],
    }
