"""参考模型在回放评测文本上的每 token 负对数似然（NLL），作为以后「与参考模型的差距」的基线。

    flock <工作目录>/gpu.lock python -m pipeline.replay.nll --mllm <阶段 1 融合模型目录> --replay-dir <回放目录>

在 `envs/mllm-train` 中运行（transformers 与 fla 内核）。模型用训练的加载函数 `pipeline.train.model.load_model`
加载，只取其中的语言模型做纯文本前向，精度与训练相同（bfloat16）。文本按训练时的写法重新渲染、重新分词：

- free_raw：开头的换行（或随机 token）+ 生成文本；cont_raw：文档开头 + 续写。正常结束的样本末尾带上结束符。
- free_chat、cont_chat：用 `prompts.render_training`（与 ASR 训练同一套 chat template 设置）渲染用户与助手内容。

计 NLL 的 token：无模板两组为生成部分（含正常结束时的结束符）；带模板两组为助手内容和结尾的 `<|im_end|>`，
free_chat 的用户消息和它的 `<|im_end|>` 另计一项。被截断的样本没有自然结束，不计结尾的结束符。
按字符位置把 token 分到各部分：token 的起点落在哪一段，就属于哪一段。

输出 `<回放目录>/<组>/<划分>/nll.jsonl`，每行一个样本：各部分的 token 数与 NLL 之和。
汇总（按 token 加权的平均 NLL、续写两组按文档语言分开）由 `report` 完成。
"""

import argparse
import json
from pathlib import Path

import torch

from pipeline.replay.prompts import CONTINUE_INSTRUCTION, GROUPS, render_training

IM_END = "<|im_end|>"


def training_text(row, group):
    """返回 (渲染后的文本, {部分名: [(起, 止), ...]})，区间是字符位置。"""
    last_stop = None
    for piece in row["pieces"]:
        if "gen" in piece:
            last_stop = piece.get("stop_token")
    if group in ("free_raw", "cont_raw"):
        start = row["start_text"] if group == "free_raw" else row["prefix_text"]
        ending = ""
        if row["ended"]:
            ending = "<|im_end|>" if last_stop == 248046 else "<|endoftext|>"
        text = start + row["generated_text"] + ending
        return text, {"generated": [(len(start), len(text))]}
    user = row["user_text"] if group == "free_chat" else f"{CONTINUE_INSTRUCTION}\n\n{row['prefix_text']}"
    assistant = row["assistant_text"]
    # 与训练的 build_training_text 一样用 processor 渲染
    text = render_training(tokenizer = _processor, user = user, assistant = assistant)
    spans = {}
    # 模板会去掉内容首尾的空白
    user_clean, assistant_clean = user.strip(), assistant.strip()
    user_start = text.index("<|im_start|>user\n") + len("<|im_start|>user\n")
    if text[user_start: user_start + len(user_clean)] != user_clean:
        return None, None
    if group == "free_chat":
        spans["user"] = [(user_start, user_start + len(user_clean) + len(IM_END))]
    end = text.rindex(IM_END)
    assistant_start = end - len(assistant_clean)
    if text[assistant_start:end] != assistant_clean:
        return None, None
    ended = row["ended"] and last_stop == 248046
    spans["assistant"] = [(assistant_start, end + (len(IM_END) if ended else 0))]
    return text, spans


_processor = None
_tokenizer = None


@torch.inference_mode()
def sample_nll(model, input_ids, positions, chunk = 2048):
    """positions 位置上 token 的 NLL（由前一个位置的输出预测）。"""
    hidden = model.model.language_model(input_ids = input_ids[None], use_cache = False).last_hidden_state[0]
    weight = model.lm_head.weight
    out = torch.empty(len(positions), dtype = torch.float32, device = hidden.device)
    for start in range(0, len(positions), chunk):
        index = positions[start: start + chunk]
        logits = (hidden[index - 1] @ weight.T).float()
        out[start: start + chunk] = torch.logsumexp(logits, dim = -1) - logits.gather(1, input_ids[index][:, None])[:, 0]
    return out


def main():
    global _processor, _tokenizer
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mllm", required = True, help = "阶段 1 融合模型目录（训练用的模型）")
    parser.add_argument("--replay-dir", required = True)
    parser.add_argument("--groups", nargs = "+", default = list(GROUPS))
    parser.add_argument("--split", default = "eval")
    args = parser.parse_args()

    from pipeline.train.model import load_model

    model, processor = load_model(model_dir = args.mllm, device = "cuda", dtype = torch.bfloat16, attn_implementation = "sdpa")
    _processor = processor
    _tokenizer = processor.tokenizer
    for group in args.groups:
        split_dir = Path(args.replay_dir) / group / args.split
        with open(split_dir / "samples.jsonl", encoding = "utf-8") as handle:
            rows = [row for row in map(json.loads, handle) if row["keep"]]
        out_rows = []
        skipped = 0
        for row in rows:
            text, spans = training_text(row = row, group = group)
            if text is None:
                skipped += 1
                continue
            encoding = _tokenizer(text, add_special_tokens = False, return_offsets_mapping = True)
            ids = torch.tensor(encoding["input_ids"], device = "cuda")
            starts = [offset[0] for offset in encoding["offset_mapping"]]
            parts = {name: [t for t in range(1, len(starts)) if any(a <= starts[t] < b for a, b in ranges)] for name, ranges in spans.items()}
            positions = sorted(t for indices in parts.values() for t in indices)
            if not positions:
                skipped += 1
                continue
            nll = sample_nll(model = model, input_ids = ids, positions = torch.tensor(positions, device = "cuda")).cpu()
            by_position = dict(zip(positions, nll.tolist()))
            result = {"sample_id": row["sample_id"], "lang": row.get("lang"), "doc_lang": row.get("doc_lang"), "n_tokens": len(starts)}
            for name, indices in parts.items():
                result[f"{name}_tokens"] = len(indices)
                result[f"{name}_nll_sum"] = round(sum(by_position[t] for t in indices), 4)
            out_rows.append(result)
        with open(split_dir / "nll.jsonl", "w", encoding = "utf-8") as handle:
            for result in out_rows:
                handle.write(json.dumps(result, ensure_ascii = False) + "\n")
        key = "assistant" if group in ("free_chat", "cont_chat") else "generated"
        total = sum(result[f"{key}_tokens"] for result in out_rows)
        nll_sum = sum(result[f"{key}_nll_sum"] for result in out_rows)
        print(f"{group}/{args.split}: {len(out_rows)} 条（跳过 {skipped}），{key} {total} token，平均 NLL {nll_sum / max(total, 1):.4f}", flush = True)


if __name__ == "__main__":
    main()
