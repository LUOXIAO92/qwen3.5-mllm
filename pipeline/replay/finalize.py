"""整理生成结果：在循环（原样或近似，见 `degeneration.py`）起点截断、解码文本、计算退化统计，写出每个划分的最终索引 `samples.jsonl`。

    python -m pipeline.replay.finalize --model <参考模型目录> --replay-dir <回放目录> [--no-filter | --drop-loops]

`--no-filter`（不筛选的回放目录）时不截断、不丢弃，每条样本按生成结果完整保留（`keep` 都为真，生成片段的 `keep` 等于
`length`）；循环起点照常计算，只作为元数据：`loop_onset` 记录第一个出现循环的片段、位置和种类，`pre_loop_tokens` 是
循环起点之前的生成 token 数（即筛选模式会保留的数量），`status` 保留生成时的状态（free_chat 用户消息没有正常结束时为
`no_user_end`）。

`--drop-loops`（丢弃循环的回放目录 `replay-noloop/`）时同样不截断：生成时已经整条丢弃的样本（索引状态 `loop`，只有
元数据）写成 `keep` 为假、`drop_reason` 为 `loop` 的行，`loop_onset` 取自生成时的记录，`n_gen` 是花掉的生成 token 数；
其余样本按同一套规则再完整检查一次，仍发现循环的也整条不用（正常情况下为 0 条）。

在 `envs/mllm-eval` 中运行（只用 CPU）。二进制数组不复制也不改写：`samples.jsonl` 中每个生成片段的 `keep`
给出截断后保留的 token 数，读取时取 `[offset, offset + keep)`。每行字段：

- `sample_id`、`prompt_index`、`shard`、`pieces`（同 shard 索引，生成片段多一个 `keep`）
- `keep`（是否作为回放样本使用，不筛选时都为真）、`drop_reason`（`no_user_end` 用户消息没有正常结束；`user_loop` 用户消息中出现循环；
  `empty` 截断后没有剩下生成 token）
- `n_gen`、`n_kept`（生成 token 数与截断后保留数）、`truncated`、`ended`（最后保留的 token 是结束符）
- `gzip_ratio`、`rep4`（4-gram 重复率）、`window_repeat`（256 token 窗口内复制前文的最大比例），都在保留的生成 token 上计算
- `gen_nll`：最后一个生成片段（free_chat 为助手回答）保留部分的平均每 token 负对数似然，取自生成时记录的对数概率；
  明显偏高（例如超过 4）的多是各种文字混杂、不成句的文本
- 文本：`generated_text`（无模板的两组：保留的生成文本，不含结束符）；`user_text`、`assistant_text`（带模板的两组：
  用户与助手内容，不含模板和结束符）；续写两组另有 `prefix_text` 与来源信息（`doc_lang`、`source`、`source_ref` 等）
- `code_like`：文本像代码（含代码围栏，或括号、等号、分号等符号占非空白字符的 8% 以上）

语言字段 `lang` 由 `langid` 一步补上。
"""

import argparse
import json
import re
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from pipeline.replay.degeneration import gzip_ratio, loop_onset, max_window_repeat, near_loop_onset, ngram_repeat_rate
from pipeline.replay.prompts import ALL_GROUPS, SPLITS
from pipeline.replay.store import ShardReader, read_all_records
from pipeline.tokenizer_fix import check_tokenizer

_CODE_SYMBOLS = set("{}[]();=<>_/\\|$#")
_tokenizer = None


def code_like(text):
    compact = [char for char in text if not char.isspace()]
    if "```" in text:
        return True
    if not compact:
        return False
    return sum(char in _CODE_SYMBOLS for char in compact) / len(compact) > 0.08


def mean_nll(reader, shard, piece):
    if piece is None or not piece["keep"]:
        return None
    lp = np.asarray(reader.piece(number = shard, piece = piece, name = "lp"))[: piece["keep"]]
    return round(float(-lp.mean()), 4)


def decode(ids):
    return _tokenizer.decode(list(map(int, ids)), skip_special_tokens = False)


def init_worker(model_dir):
    global _tokenizer
    from transformers import AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(model_dir)


def prompt_meta(row, prompt, group):
    """free_raw 的开头 token，续写两组的文档开头与来源信息。"""
    if group == "free_raw":
        row["start_ids"] = prompt["prompt_ids"]
        row["start_text"] = decode(ids = prompt["prompt_ids"])
    if group in ("cont_raw", "cont_chat"):
        for key in ("prefix_id", "source", "source_license", "source_ref", "doc_id", "prefix_text", "prefix_tokens"):
            row[key] = prompt[key]
        row["doc_lang"] = prompt["lang"]
    if group == "prompt_chat":
        for key in ("part", "kind", "source", "source_license", "source_lang", "source_ref", "user_text", "user_tokens", "context_tokens"):
            if key in prompt:
                row[key] = prompt[key]
        row["prompt_lang"] = prompt.get("lang") or "unknown"


def process(job):
    """一个样本：截断、解码、统计。"""
    record, prompt, group, raw_dir, stop_ids = job["record"], job["prompt"], job["group"], job["raw_dir"], set(job["stop_ids"])
    no_filter = job["no_filter"]
    drop_loops = job["drop_loops"]
    reader = ShardReader(raw_dir = raw_dir)
    row = {
        "sample_id": record["sample_id"],
        "prompt_index": record["prompt_index"],
        "shard": record["shard"],
        "keep": True,
        "drop_reason": None,
    }
    if drop_loops:
        row["selection"] = "noloop"
        row["status"] = record["status"]
        if record.get("origin"):
            row["origin"] = record["origin"]
    if record["status"] == "loop":
        # 生成时发现循环、整条丢弃的样本（generate --drop-loops 或 noloop 导入）：只有元数据
        row.update(
            keep = False,
            drop_reason = "loop",
            pieces = record["pieces"],
            n_gen = record["n_gen"],
            n_kept = 0,
            truncated = False,
            loop_onset = record["loop"],
            ended = False,
            finish_reason = record["loop"].get("finish_reason"),
        )
        prompt_meta(row = row, prompt = prompt, group = group)
        return row
    gen_pieces = [piece for piece in record["pieces"] if "gen" in piece]
    arrays = [np.asarray(reader.piece(number = record["shard"], piece = piece)) for piece in gen_pieces]
    keeps = []
    truncated = False
    onset_at = None
    pre_loop = 0
    stop = False
    for piece, ids in zip(gen_pieces, arrays):
        if stop:
            keeps.append(0)
            continue
        exact, near = loop_onset(ids = ids), near_loop_onset(ids = ids)
        if exact is None and near is None:
            keeps.append(len(ids))
            pre_loop += len(ids)
            continue
        onset = min(position for position in (exact, near) if position is not None)
        truncated = True
        onset_at = {"role": piece["gen"], "position": onset, "kind": "exact" if onset == exact else "near"}
        keeps.append(onset)
        pre_loop += onset
        stop = True
    if no_filter or drop_loops:
        # 不筛选、丢弃循环两种模式都不截断：所有生成 token 都保留；丢弃循环模式下再检查出循环的样本整条不用
        keeps = [len(ids) for ids in arrays]
        loop_found = truncated
        truncated = False
        row["status"] = record["status"]
        row["pre_loop_tokens"] = pre_loop
        if drop_loops and loop_found:
            row.update(keep = False, drop_reason = "loop")
    elif record["status"] != "ok":
        row.update(keep = False, drop_reason = record["status"])
    elif truncated and onset_at["role"] == "user":
        row.update(keep = False, drop_reason = "user_loop")
    pieces = []
    index = 0
    for piece in record["pieces"]:
        if "gen" in piece:
            piece = dict(piece, keep = keeps[index])
            index += 1
        pieces.append(piece)
    kept_arrays = [ids[:keep] for ids, keep in zip(arrays, keeps)]
    kept_all = np.concatenate(kept_arrays) if kept_arrays else np.zeros(0, dtype = np.int32)
    n_kept = int(len(kept_all))
    if row["keep"] and n_kept == 0:
        row.update(keep = False, drop_reason = "empty")
    last = kept_arrays[-1] if kept_arrays else np.zeros(0, dtype = np.int32)
    ended = bool(not truncated and len(last) and int(last[-1]) in stop_ids)

    def content(ids):
        ids = list(map(int, ids))
        if ids and ids[-1] in stop_ids:
            ids = ids[:-1]
        return decode(ids = ids)

    if group in ("free_raw", "cont_raw"):
        generated = content(ids = kept_arrays[0])
        row["generated_text"] = generated
        stats_text = generated
    else:
        user = content(ids = kept_arrays[0]) if group == "free_chat" else None
        assistant = content(ids = kept_arrays[-1]) if (group in ("cont_chat", "prompt_chat") or len(kept_arrays) > 1) else ""
        if group == "free_chat":
            row["user_text"] = user
        row["assistant_text"] = assistant
        stats_text = assistant
    prompt_meta(row = row, prompt = prompt, group = group)
    row.update(
        pieces = pieces,
        n_gen = int(sum(len(ids) for ids in arrays)),
        n_kept = n_kept,
        truncated = truncated,
        loop_onset = onset_at,
        ended = ended,
        finish_reason = gen_pieces[-1]["finish_reason"] if gen_pieces else None,
        gzip_ratio = round(gzip_ratio(text = stats_text), 4),
        rep4 = round(ngram_repeat_rate(ids = kept_all, n = 4), 4),
        window_repeat = round(max_window_repeat(ids = kept_all), 4),
        code_like = code_like(text = stats_text),
        gen_nll = mean_nll(reader = reader, shard = record["shard"], piece = pieces[-1] if "gen" in pieces[-1] else None),
    )
    return row


def finalize_split(replay_dir, group, split, model_dir, jobs, no_filter = False, drop_loops = False):
    split_dir = Path(replay_dir) / group / split
    raw_dir = split_dir / "raw"
    if not raw_dir.exists():
        return None
    with open(split_dir / "prompts.jsonl", encoding = "utf-8") as handle:
        prompts = {row["prompt_index"]: row for row in map(json.loads, handle)}
    records = sorted(read_all_records(raw_dir = raw_dir), key = lambda record: record["prompt_index"])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    check_tokenizer(tokenizer = tokenizer, source = model_dir)
    stop_ids = [tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")]
    tasks = [
        {
            "record": record,
            "prompt": prompts[record["prompt_index"]],
            "group": group,
            "raw_dir": str(raw_dir),
            "stop_ids": stop_ids,
            "no_filter": no_filter,
            "drop_loops": drop_loops,
        }
        for record in records
    ]
    with get_context("spawn").Pool(processes = jobs, initializer = init_worker, initargs = (model_dir,)) as pool:
        rows = pool.map(process, tasks, chunksize = 64)
    for row in rows:
        row["group"] = group
        row["split"] = split
    tmp = split_dir / "samples.jsonl.tmp"
    with open(tmp, "w", encoding = "utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii = False) + "\n")
    tmp.replace(split_dir / "samples.jsonl")
    kept = [row for row in rows if row["keep"]]
    summary = {
        "samples": len(rows),
        "kept": len(kept),
        "dropped": dict(Counter(row["drop_reason"] for row in rows if not row["keep"])),
        "truncated": sum(row["truncated"] for row in kept),
        "with_loop": sum(row["loop_onset"] is not None for row in kept),
        "dropped_tokens": sum(row["n_gen"] for row in rows if not row["keep"]),
        "gen_tokens": sum(row["n_gen"] for row in rows),
        "kept_tokens": sum(row["n_kept"] for row in kept),
    }
    print(f"{group}/{split}: {summary}", flush = True)
    return summary


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required = True, help = "参考模型目录（取分词器）")
    parser.add_argument("--replay-dir", required = True)
    parser.add_argument("--groups", nargs = "+", default = list(ALL_GROUPS))
    parser.add_argument("--splits", nargs = "+", default = list(SPLITS))
    parser.add_argument("--jobs", type = int, default = 8)
    parser.add_argument("--no-filter", action = "store_true", help = "不截断、不丢弃，循环只作为元数据")
    parser.add_argument("--drop-loops", action = "store_true", help = "不截断；有循环的样本整条不用（generate --drop-loops 的目录）")
    args = parser.parse_args()
    for group in args.groups:
        for split in args.splits:
            finalize_split(
                replay_dir = args.replay_dir,
                group = group,
                split = split,
                model_dir = args.model,
                jobs = args.jobs,
                no_filter = args.no_filter,
                drop_loops = args.drop_loops,
            )


if __name__ == "__main__":
    main()
