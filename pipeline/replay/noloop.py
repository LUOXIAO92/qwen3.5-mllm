"""由不筛选的回放目录建立「丢弃循环」的回放目录：没有循环的样本原样复制过来，有循环的样本整条丢弃，只记元数据。

    python -m pipeline.replay.noloop --source <replay-unfiltered> --out <replay-noloop>

在 `envs/mllm-eval` 中运行（只用 CPU）。不筛选目录中的文件只读不改。对每个（组, 划分）：

- 复制 `prompts.jsonl` 和 `templates.json`，提示的编号与不筛选目录相同；
- 已经生成完的样本按 finalize 的规则逐段检查循环（`degeneration.loop_onset` 与 `near_loop_onset`）。没有循环的样本把
  token id 和对数概率写进新目录的 `raw/shard-00000`，索引行与原来相同，另加 `origin`（原 shard 编号）；有循环的样本
  只写一行元数据：状态 `loop`、`loop`（片段、位置、种类）、`n_gen`（花掉的生成 token 数），没有生成片段；
- 不筛选目录中暂停在途的样本（`raw/inflight.pkl`）复制过来，由 `generate --drop-loops` 接着生成，出现循环时同样丢弃。

之后用 `generate --drop-loops` 在新目录中续跑：按提示清单的顺序接着投放，每条提示只采样一次，所以最终的样本集合是提示
清单前若干条中没有循环的全部样本。新目录的 `raw/` 中只有保留样本的 token，参考模型 top-32 一步（teacher）只处理这些样本。
"""

import argparse
import json
import shutil
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from pipeline.replay.degeneration import loop_onset, near_loop_onset
from pipeline.replay.prompts import GROUPS, SPLITS
from pipeline.replay.store import ShardReader, ShardWriter, list_shards, read_all_records


def find_loop(job):
    """样本中第一个出现循环的生成片段：{"role", "position", "kind"}；没有循环返回 None。"""
    record, raw_dir = job
    reader = ShardReader(raw_dir = raw_dir)
    for piece in record["pieces"]:
        if "gen" not in piece:
            continue
        ids = np.asarray(reader.piece(number = record["shard"], piece = piece))
        exact, near = loop_onset(ids = ids), near_loop_onset(ids = ids)
        if exact is None and near is None:
            continue
        position = min(value for value in (exact, near) if value is not None)
        return {"role": piece["gen"], "position": int(position), "kind": "exact" if position == exact else "near", "finish_reason": piece["finish_reason"]}
    return None


def import_split(source, out, group, split, jobs):
    source_dir = Path(source) / group / split
    out_dir = Path(out) / group / split
    source_raw = source_dir / "raw"
    out_raw = out_dir / "raw"
    if out_raw.exists() and list_shards(raw_dir = out_raw):
        print(f"{group}/{split}: 已导入，跳过", flush = True)
        return None
    out_dir.mkdir(parents = True, exist_ok = True)
    shutil.copy2(src = source_dir / "prompts.jsonl", dst = out_dir / "prompts.jsonl")
    if not source_raw.exists() or not list_shards(raw_dir = source_raw):
        print(f"{group}/{split}: 不筛选目录中还没有生成结果，只复制提示", flush = True)
        return None
    out_raw.mkdir(parents = True, exist_ok = True)
    records = sorted(read_all_records(raw_dir = source_raw), key = lambda record: record["prompt_index"])
    with get_context("spawn").Pool(processes = jobs) as pool:
        loops = pool.map(find_loop, [(record, str(source_raw)) for record in records], chunksize = 16)
    reader = ShardReader(raw_dir = source_raw)
    writer = ShardWriter(raw_dir = out_raw, topk = False)
    kept = dropped = kept_tokens = dropped_tokens = 0
    for record, loop in zip(records, loops):
        n_gen = sum(piece["length"] for piece in record["pieces"] if "gen" in piece)
        base = {"sample_id": record["sample_id"], "prompt_index": record["prompt_index"], "origin": {"root": Path(source).name, "shard": record["shard"]}}
        if loop is not None:
            writer.add_record({**base, "status": "loop", "n_gen": n_gen, "kept_gen": 0, "pieces": record["pieces"][:1], "loop": loop})
            dropped += 1
            dropped_tokens += n_gen
            continue
        pieces = []
        for piece in record["pieces"]:
            if "gen" in piece:
                offset = writer.add_gen(
                    ids = np.asarray(reader.piece(number = record["shard"], piece = piece)),
                    lp = np.asarray(reader.piece(number = record["shard"], piece = piece, name = "lp")),
                )
                piece = dict(piece, offset = offset)
            pieces.append(piece)
        writer.add_record({**base, "status": record["status"], "n_gen": n_gen, "kept_gen": n_gen, "pieces": pieces})
        kept += 1
        kept_tokens += n_gen
    writer.close()
    inflight = source_raw / "inflight.pkl"
    if inflight.exists():
        shutil.copy2(src = inflight, dst = out_raw / "inflight.pkl")
    summary = {"kept": kept, "kept_tokens": kept_tokens, "dropped": dropped, "dropped_tokens": dropped_tokens, "inflight_copied": inflight.exists()}
    with open(out_raw / "import.json", "w", encoding = "utf-8") as handle:
        json.dump({"source": str(Path(source).name), **summary}, handle, ensure_ascii = False, indent = 2)
    print(f"{group}/{split}: {summary}", flush = True)
    return summary


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required = True, help = "不筛选的回放目录")
    parser.add_argument("--out", required = True, help = "丢弃循环的回放目录")
    parser.add_argument("--groups", nargs = "+", default = list(GROUPS))
    parser.add_argument("--splits", nargs = "+", default = list(SPLITS))
    parser.add_argument("--jobs", type = int, default = 8)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents = True, exist_ok = True)
    shutil.copy2(src = Path(args.source) / "templates.json", dst = out / "templates.json")
    for group in args.groups:
        for split in args.splits:
            import_split(
                source = args.source,
                out = args.out,
                group = group,
                split = split,
                jobs = args.jobs,
            )


if __name__ == "__main__":
    main()
