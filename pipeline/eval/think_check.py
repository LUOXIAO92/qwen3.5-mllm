"""思考能力检查（任务 #71）：用固定的 200 条提问开启思考生成，看 </think> 能否写完、推理与回答多长、是否陷入循环。

    flock <gpu.lock> python -m pipeline.eval.think_check --model-dir <融合模型目录> \\
        [--language-model <检查点>/language_model.safetensors] \\
        --replay-dir <回放目录> --output-dir <输出目录>

提问：回放数据 free_chat_think 评测集（不参与任何训练）中参考模型自己写的用户提问，按提示顺序取前 --count 条（默认
200）用户轮正常结束、token 保存在 raw/ 中的样本；生成时因循环整条丢弃的样本只记了元数据、没有用户轮的 token，不在
其中（评测集 615 条中有 39 条）。提示就是生成这组数据时的写法，直接用保存的 token id：
<|im_start|>user\\n{用户提问}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n，即聊天模板 enable_thinking = True
时的生成提示，没有系统消息。

生成：温度 --temperature（默认 1）、不设 top-k / top-p，与回放数据的采样设置相同，参考模型在同样设置下的结果可以直接与
回放数据的统计（#70：无循环样本中 93% 写完 </think>，推理中位数约 2,170 token）对照；每批开始前按 --seed 固定随机数。
最多 --max-new-tokens（默认 16,384，与回放数据相同）个新 token，在 <|im_end|> 或 <|endoftext|> 处停止；左填充，transformers
的 generate。批内全部样本结束或到上限时这一批结束，所以一批的用时由最长的样本决定：参考模型约 6% 的样本一直写到上限，
200 条一批时几乎每批都要生成满 16k 步。--max-new-tokens 降低时，超过上限才写完 </think> 的样本记为没有写完，比较
不同模型时用同一个上限。

指标（summary.json、summary.md；逐条结果在 generations.jsonl，生成的 token id 在 generations.ids.i32）：
- closed：写出了 </think> 的比例；ended：在上限之前写出结束符的比例；capped：写到上限的比例。
- loop：生成内容出现循环（原样或近似，pipeline/replay/degeneration.py 的 degeneration_onset，与回放数据丢弃循环的规则
  相同）的比例。
- reasoning_tokens：写完 </think> 的样本中推理的 token 数（含 </think>）；answer_tokens：写完 </think> 且正常结束的样本中
  回答的 token 数（含结束符）。给出平均、中位数和 90% 分位数。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

THINK_END_ID = 248069
STOP_IDS = (248046, 248044)
PAD_ID = 248044


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


def generate_batch(model, prompts, max_new_tokens, temperature, seed):
    """左填充生成一批，返回每条新生成的 token（到第一个结束符为止，含结束符）。"""
    import torch

    device = next(model.parameters()).device
    longest = max(len(ids) for _, ids in prompts)
    input_ids = torch.full(size = (len(prompts), longest), fill_value = PAD_ID, dtype = torch.long)
    attention_mask = torch.zeros(size = (len(prompts), longest), dtype = torch.long)
    for row, (_, ids) in enumerate(prompts):
        input_ids[row, longest - len(ids):] = torch.tensor(data = ids)
        attention_mask[row, longest - len(ids):] = 1
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

    return "\n".join([
        f"模型：{summary['model']}（{summary['samples']} 条提问，温度 {summary['temperature']}，上限 {summary['max_new_tokens']}）",
        "",
        "| 写完 </think> | 正常结束 | 到上限 | 循环 | 推理 token（平均 / 中位 / 90%） | 回答 token（平均 / 中位 / 90%） |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {pct(summary['closed'])} | {pct(summary['ended'])} | {pct(summary['capped'])} | {pct(summary['loop'])}"
        f" | {dist(summary['reasoning_tokens'])} | {dist(summary['answer_tokens'])} |",
        "",
    ])


def main(argv = None):
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
    parser.add_argument("--output-dir", required = True, type = Path)
    args = parser.parse_args(args = argv)

    from pipeline.eval.run_eval import load_model

    args.output_dir.mkdir(parents = True, exist_ok = True)
    prompts = load_prompts(replay_dir = args.replay_dir, count = args.count)
    if len(prompts) < args.count:
        print(f"只有 {len(prompts)} 条可用的提问（要求 {args.count}）", flush = True)
    model, _ = load_model(model_dir = args.model_dir, projector_path = None, language_model_path = args.language_model)
    rows = []
    generated = []
    start = time.perf_counter()
    for number, begin in enumerate(range(0, len(prompts), args.batch_size)):
        batch = prompts[begin: begin + args.batch_size]
        outputs = generate_batch(
            model = model,
            prompts = batch,
            max_new_tokens = args.max_new_tokens,
            temperature = args.temperature,
            seed = args.seed + number,
        )
        for (sample_id, prompt), ids in zip(batch, outputs):
            row = {"sample_id": sample_id, "prompt_tokens": len(prompt), **analyze(ids = ids, max_new_tokens = args.max_new_tokens)}
            rows.append(row)
            generated.append(np.asarray(a = ids, dtype = np.int32))
        print(f"第 {number + 1} 批 {len(batch)} 条完成，已用 {time.perf_counter() - start:.0f} 秒", flush = True)
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
        "seconds": round(number = time.perf_counter() - start, ndigits = 1),
        **summarize(rows = rows, max_new_tokens = args.max_new_tokens),
    }
    (args.output_dir / "summary.json").write_text(data = json.dumps(obj = summary, ensure_ascii = False, indent = 1) + "\n", encoding = "utf-8")
    (args.output_dir / "summary.md").write_text(data = format_markdown(summary = summary), encoding = "utf-8")
    print(json.dumps(obj = summary, ensure_ascii = False), flush = True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
