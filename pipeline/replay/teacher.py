"""参考模型对回放文本的 top-32 分布：把每条保留的样本（提示 + 生成，就是保存的 token id）整条送进参考模型前向一次（teacher forcing），
在每个生成位置保存参考模型概率最高的 32 个 token、它们的对数概率和整个词表上的 log-sum-exp，供以后试 top-k 截断的近似 KL 损失。

    python -m pipeline.replay.teacher --mllm <阶段 1 融合模型目录> --replay-dir <回放目录> \
        --plan free_raw:eval,free_chat:eval --estimate        # 只统计 token 数与需要的磁盘空间
    flock <工作目录>/gpu.lock python -m pipeline.replay.teacher --mllm <阶段 1 融合模型目录> --replay-dir <回放目录> \
        --plan free_raw:eval,free_chat:eval --time-budget 720
    flock <工作目录>/gpu.lock python -m pipeline.replay.teacher --mllm <阶段 1 融合模型目录> --replay-dir <回放目录> \
        --plan free_raw:eval --verify 8                       # 抽样与全词表直接计算对照

在 `envs/mllm-train` 中运行（transformers 与 fla 内核）。模型用 transformers 加载阶段 1 融合模型（与 ASR 训练同一份权重、
同一套实现），只取其中的语言模型，精度 bfloat16。输出层的矩阵乘法用 bfloat16 输入、float32 输出（`torch.mm(...,
out_dtype = torch.float32)`），log-sum-exp 与 top-32 都在 float32 上计算。

存放格式：每个划分的 `teacher/` 目录，与 `raw/` 的 shard 一一对应，按生成 token 平铺，第 i 行对应 raw shard
二进制数组中的第 i 个生成 token（同一个 `offset`），即参考模型在这个 token 之前的全部 token（提示、模板、前面的生成）
条件下给出的下一个 token 的分布：

    teacher/shard-<编号>.top_ids.i32   # 概率最高的 32 个 token id，按概率从高到低（int32，每行 32 个）
    teacher/shard-<编号>.top_lp.f16    # 这 32 个 token 的对数概率 logit - lse（float16，每行 32 个）
    teacher/shard-<编号>.lse.f32       # 整个词表上的 log-sum-exp（float32）
    teacher/shard-<编号>.lp.f32        # 实际生成的 token 的对数概率（float32），即参考模型在这条文本上的负对数似然取负
    teacher/progress.jsonl             # 已完成的样本，断点续跑用
    teacher/meta.json                  # 格式说明与计算设置

每个生成 token 占 32 × 4 + 32 × 2 + 4 + 4 = 200 字节。由存下的数可以还原：top-32 的概率 p_i = exp(top_lp_i)；
top-32 之外的尾部概率 1 - Σ_i p_i；原始 logit = top_lp_i + lse（temperature 不为 1 时可用 logit 重新归一化，但尾部只有
总量没有细节）。对数概率存成 float16，概率最高的几个 token（对数概率接近 0）精度最高：|对数概率| < 1 时误差不超过 5e-4，
< 8 时不超过 4e-3。

批次：样本按长度排序，凑成填充后不超过 `--max-batch-tokens` 个 token 的批次，右侧填充、不用注意力掩码。因果注意力
和线性注意力（GDN）都只看前文，右侧填充不影响前面位置的结果。
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from pipeline.replay.generate import parse_plan
from pipeline.replay.store import ShardReader, list_shards, read_all_records, shard_paths

TOP = 32
SUFFIXES = {
    "top_ids": (".top_ids.i32", np.int32, TOP),
    "top_lp": (".top_lp.f16", np.float16, TOP),
    "lse": (".lse.f32", np.float32, 1),
    "lp": (".lp.f32", np.float32, 1),
}
BYTES_PER_TOKEN = sum(np.dtype(dtype).itemsize * width for _, dtype, width in SUFFIXES.values())
PAD_ID = 248044


def teacher_paths(teacher_dir, number):
    base = Path(teacher_dir) / f"shard-{number:05d}"
    return {name: Path(str(base) + suffix) for name, (suffix, _, _) in SUFFIXES.items()}


def raw_tokens(raw_dir, number):
    """raw shard 中生成 token 的总数（ids.i32 的长度）。"""
    _, paths = shard_paths(raw_dir = raw_dir, number = number)
    return paths["ids"].stat().st_size // 4


def build_sequence(record, reader):
    """样本的完整 token 序列（提示、模板与生成片段依次拼接），以及各生成片段 (shard 中的起点, 长度, 在序列中的起点)。"""
    ids = []
    pieces = []
    for piece in record["pieces"]:
        if "gen" in piece:
            tokens = np.asarray(reader.piece(number = record["shard"], piece = piece))
            pieces.append((piece["offset"], piece["length"], len(ids)))
            ids.extend(tokens.tolist())
        else:
            ids.extend(piece["ids"])
    return ids, pieces


# 为真时 free_chat_think 中没有写出 </think> 的样本（samples.jsonl 的 think_closed 不为真）也不算（--exclude-unclosed-think）
EXCLUDE_UNCLOSED_THINK = False


def kept_records(split_dir):
    """要计算 top-32 的样本：有生成片段，并且 finalize 写出的 samples.jsonl 中 keep 为真（丢弃循环的目录中有循环的样本不算）。"""
    split_dir = Path(split_dir)
    raw_dir = split_dir / "raw"
    if not raw_dir.exists():
        return []
    records = [record for record in read_all_records(raw_dir = raw_dir) if any("gen" in piece for piece in record["pieces"])]
    samples_path = split_dir / "samples.jsonl"
    if not samples_path.exists():
        raise SystemExit(f"{split_dir} 还没有 samples.jsonl，先运行 finalize")
    exclude_unclosed = EXCLUDE_UNCLOSED_THINK and split_dir.parent.name == "free_chat_think"
    with open(samples_path, encoding = "utf-8") as handle:
        keep = {row["sample_id"]: row["keep"] and (not exclude_unclosed or row.get("think_closed") is True) for row in map(json.loads, handle)}
    missing = [record["sample_id"] for record in records if record["sample_id"] not in keep]
    if missing:
        raise SystemExit(f"{split_dir} 的 samples.jsonl 比生成结果旧（缺 {len(missing)} 条），先重新运行 finalize")
    return [record for record in records if keep[record["sample_id"]]]


def split_tokens(replay_dir, group, split):
    records = kept_records(split_dir = Path(replay_dir) / group / split)
    return len(records), sum(piece["length"] for record in records for piece in record["pieces"] if "gen" in piece)


def free_bytes(path):
    path = Path(path)
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


class Teacher:
    def __init__(self, args):
        import torch
        from transformers import AutoModelForImageTextToText

        self.torch = torch
        self.args = args
        t0 = time.time()
        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name_or_path = args.mllm,
            trust_remote_code = True,
            dtype = torch.bfloat16,
            attn_implementation = "sdpa",
        )
        model.eval()
        model.to(device = "cuda")
        self.language_model = model.model.language_model
        self.weight = model.lm_head.weight
        self.load_seconds = time.time() - t0
        print(f"模型加载 {self.load_seconds:.0f} 秒", flush = True)

    def hidden(self, batch):
        """一批右侧填充的序列，返回最后一层的隐状态 [B, L, H]。"""
        torch = self.torch
        width = max(len(ids) for ids in batch)
        input_ids = torch.full((len(batch), width), PAD_ID, dtype = torch.long)
        for row, ids in enumerate(batch):
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype = torch.long)
        input_ids = input_ids.to(device = "cuda")
        with torch.inference_mode():
            return self.language_model(input_ids = input_ids, use_cache = False).last_hidden_state, input_ids

    def head(self, states, targets):
        """states [T, H] 位置上的下一个 token 分布：(top-32 id, top-32 对数概率, lse, 目标 token 的对数概率)。"""
        torch = self.torch
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(states), self.args.head_chunk):
                logits = torch.mm(states[start: start + self.args.head_chunk], self.weight.T, out_dtype = torch.float32)
                lse = torch.logsumexp(logits, dim = -1)
                top = torch.topk(logits, k = TOP, dim = -1)
                target_logit = logits.gather(1, targets[start: start + self.args.head_chunk, None])[:, 0]
                outputs.append((top.indices.int(), (top.values - lse[:, None]).half(), lse, target_logit - lse))
                del logits
        return [torch.cat([part[k] for part in outputs]).cpu().numpy() for k in range(4)]

    def run_split(self, group, split, deadline):
        args = self.args
        split_dir = Path(args.replay_dir) / group / split
        raw_dir = split_dir / "raw"
        teacher_dir = split_dir / "teacher"
        teacher_dir.mkdir(parents = True, exist_ok = True)
        records = kept_records(split_dir = split_dir)
        progress_path = teacher_dir / "progress.jsonl"
        done = set()
        if progress_path.exists():
            with open(progress_path, encoding = "utf-8") as handle:
                for line in handle:
                    if line.endswith("\n"):
                        done.update(json.loads(line)["sample_ids"])
        pending = [record for record in records if record["sample_id"] not in done]
        if not pending:
            print(f"[{group}/{split}] 全部 {len(records)} 条已完成", flush = True)
            return True
        # 预先分配与 raw shard 等长的数组；新文件以稀疏方式创建，写入时才占用磁盘
        needed = 0
        for number in list_shards(raw_dir = raw_dir):
            n = raw_tokens(raw_dir = raw_dir, number = number)
            for name, path in teacher_paths(teacher_dir = teacher_dir, number = number).items():
                _, dtype, width = SUFFIXES[name]
                size = n * width * np.dtype(dtype).itemsize
                if not path.exists() or path.stat().st_size != size:
                    with open(path, "ab") as handle:
                        handle.truncate(size)
                needed += size
        pending_tokens = sum(piece["length"] for record in pending for piece in record["pieces"] if "gen" in piece)
        if free_bytes(path = teacher_dir) - pending_tokens * BYTES_PER_TOKEN < args.min_free_gb * 1e9:
            raise SystemExit(f"[{group}/{split}] 写完后磁盘剩余将少于 {args.min_free_gb} GB，停止")
        meta = {
            "top_k": TOP,
            "files": {name: {"suffix": suffix, "dtype": np.dtype(dtype).name, "width": width} for name, (suffix, dtype, width) in SUFFIXES.items()},
            "bytes_per_token": BYTES_PER_TOKEN,
            "top_lp": "logit - lse，按概率从高到低",
            "model": Path(args.mllm).name,
            "dtype": "bfloat16 前向，输出层 bfloat16 输入、float32 输出",
        }
        (teacher_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii = False, indent = 2) + "\n", encoding = "utf-8")

        reader = ShardReader(raw_dir = raw_dir)
        lengths = {record["sample_id"]: len(build_sequence(record = record, reader = reader)[0]) for record in pending}
        pending.sort(key = lambda record: -lengths[record["sample_id"]])
        batches = []
        batch = []
        for record in pending:
            width = lengths[(batch or [record])[0]["sample_id"]]
            if batch and width * (len(batch) + 1) > args.max_batch_tokens:
                batches.append(batch)
                batch = []
            batch.append(record)
        if batch:
            batches.append(batch)

        outputs = {}

        def output(number):
            if number not in outputs:
                arrays = {}
                for name, path in teacher_paths(teacher_dir = teacher_dir, number = number).items():
                    _, dtype, width = SUFFIXES[name]
                    data = np.memmap(path, dtype = dtype, mode = "r+")
                    arrays[name] = data.reshape(-1, width) if width > 1 else data
                outputs[number] = arrays
            return outputs[number]

        torch = self.torch
        t_start = time.time()
        t_log = t_start
        tokens = 0
        samples = 0
        print(f"[{group}/{split}] 开始：已完成 {len(done)} 条，剩余 {len(pending)} 条 {pending_tokens} 个生成 token，{len(batches)} 批", flush = True)
        for batch in batches:
            if time.time() >= deadline:
                break
            sequences = []
            spans = []
            for row, record in enumerate(batch):
                ids, pieces = build_sequence(record = record, reader = reader)
                sequences.append(ids)
                spans.extend((row, record["shard"], offset, length, start) for offset, length, start in pieces)
            states, input_ids = self.hidden(batch = sequences)
            rows = torch.cat([torch.full((length,), row, dtype = torch.long) for row, _, _, length, _ in spans])
            positions = torch.cat([torch.arange(start, start + length) for _, _, _, length, start in spans])
            rows, positions = rows.to(device = "cuda"), positions.to(device = "cuda")
            top_ids, top_lp, lse, lp = self.head(states = states[rows, positions - 1], targets = input_ids[rows, positions])
            del states
            cursor = 0
            for _, number, offset, length, _ in spans:
                arrays = output(number = number)
                arrays["top_ids"][offset: offset + length] = top_ids[cursor: cursor + length]
                arrays["top_lp"][offset: offset + length] = top_lp[cursor: cursor + length]
                arrays["lse"][offset: offset + length] = lse[cursor: cursor + length]
                arrays["lp"][offset: offset + length] = lp[cursor: cursor + length]
                cursor += length
            for arrays in outputs.values():
                for data in arrays.values():
                    data.flush()
            with open(progress_path, "a", encoding = "utf-8") as handle:
                handle.write(json.dumps({"sample_ids": [record["sample_id"] for record in batch]}) + "\n")
            tokens += cursor
            samples += len(batch)
            now = time.time()
            if now - t_log >= args.log_every:
                t_log = now
                print(f"[{group}/{split}] {now - t_start:.0f}s 完成 {samples} 条，{tokens} 个生成 token（{tokens / (now - t_start):.0f} tok/s）", flush = True)
        elapsed = time.time() - t_start
        run = {
            "group": group,
            "split": split,
            "samples": samples,
            "tokens": tokens,
            "seconds": round(elapsed, 1),
            "tokens_per_second": round(tokens / max(elapsed, 1e-9), 1),
            "load_seconds": round(self.load_seconds, 1),
            "max_batch_tokens": args.max_batch_tokens,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(teacher_dir / "runs.jsonl", "a", encoding = "utf-8") as handle:
            handle.write(json.dumps(run, ensure_ascii = False) + "\n")
        print(f"[{group}/{split}] 本次 {elapsed:.0f}s，完成 {samples} 条，{tokens} 个生成 token（{tokens / max(elapsed, 1e-9):.0f} tok/s）", flush = True)
        return samples == len(pending)

    def verify(self, group, split, count, seed = 0, per_sample = 1024):
        """抽 count 条已完成的样本单条前向（不填充），每条随机取至多 per_sample 个生成位置，在 float64 上算整个词表的
        log softmax，与保存的数对照；同时在同一份隐状态上重算保存算法（float32 输出层），单独检查输出层与 top-32 的计算。"""
        torch = self.torch
        split_dir = Path(self.args.replay_dir) / group / split
        raw_dir = split_dir / "raw"
        teacher_dir = split_dir / "teacher"
        done = set()
        with open(teacher_dir / "progress.jsonl", encoding = "utf-8") as handle:
            for line in handle:
                done.update(json.loads(line)["sample_ids"])
        records = [record for record in kept_records(split_dir = split_dir) if record["sample_id"] in done]
        rng = np.random.default_rng(seed = seed)
        picks = [records[index] for index in rng.choice(len(records), size = min(count, len(records)), replace = False)]
        reader = ShardReader(raw_dir = raw_dir)
        weight64 = self.weight.double()
        diffs = {}
        id_mismatch = 0
        id_mismatch_same_hidden = 0
        # top-32 集合不同时，全词表分布中「在直接计算的前 32 里、不在保存的前 32 里」的 token 的概率之和
        swapped_mass = []
        checked = 0
        vllm_diff = []
        for record in picks:
            ids, pieces = build_sequence(record = record, reader = reader)
            states, input_ids = self.hidden(batch = [ids])
            # 序列中全部生成位置，以及它们在 teacher 数组中的行号
            positions = np.concatenate([np.arange(start, start + length) for _, length, start in pieces])
            rows = np.concatenate([np.arange(offset, offset + length) for offset, length, _ in pieces])
            chosen = np.sort(rng.choice(len(positions), size = min(per_sample, len(positions)), replace = False))
            positions, rows = positions[chosen], rows[chosen]
            stored = {name: np.asarray(data) for name, data in self.stored(teacher_dir = teacher_dir, number = record["shard"]).items()}
            stored = {name: data[rows] for name, data in stored.items()}
            index = torch.as_tensor(positions, device = "cuda")
            hidden = states[0, index - 1]
            targets = input_ids[0, index]
            same_ids, same_lp, same_lse, same_target = self.head(states = hidden, targets = targets)
            direct_ids, direct_lp, direct_lse, direct_target = [], [], [], []
            with torch.inference_mode():
                for start in range(0, len(hidden), 256):
                    logits = hidden[start: start + 256].double() @ weight64.T
                    lse = torch.logsumexp(logits, dim = -1)
                    top = torch.topk(logits, k = TOP, dim = -1)
                    direct_ids.append(top.indices.cpu().numpy())
                    direct_lp.append((top.values - lse[:, None]).cpu().numpy())
                    direct_lse.append(lse.cpu().numpy())
                    direct_target.append((logits.gather(1, targets[start: start + 256, None])[:, 0] - lse).cpu().numpy())
            direct_ids, direct_lp, direct_lse, direct_target = map(np.concatenate, (direct_ids, direct_lp, direct_lse, direct_target))
            # top-32 的 id 按集合比较（并列时顺序可以不同），对数概率按排序后的值比较
            id_mismatch += sum(set(a) != set(b) for a, b in zip(stored["top_ids"].tolist(), direct_ids.tolist()))
            for kept_ids, ids_row, lp_row in zip(stored["top_ids"].tolist(), direct_ids.tolist(), direct_lp.tolist()):
                missing = set(ids_row) - set(kept_ids)
                if missing:
                    swapped_mass.append(sum(np.exp(value) for token, value in zip(ids_row, lp_row) if token in missing))
            id_mismatch_same_hidden += sum(set(a) != set(b) for a, b in zip(same_ids.tolist(), direct_ids.tolist()))
            pairs = {
                "stored_vs_direct_top_lp": (stored["top_lp"], direct_lp),
                "stored_vs_direct_lse": (stored["lse"], direct_lse),
                "stored_vs_direct_lp": (stored["lp"], direct_target),
                "same_hidden_top_lp": (same_lp, direct_lp),
                "same_hidden_lse": (same_lse, direct_lse),
                "same_hidden_lp": (same_target, direct_target),
            }
            for key, (a, b) in pairs.items():
                diffs.setdefault(key, []).append(np.abs(a.astype(np.float64) - b).ravel())
            sampled_lp = np.asarray(reader.arrays(number = record["shard"])["lp"])[rows]
            vllm_diff.append(stored["lp"] - sampled_lp)
            checked += len(rows)
        vllm_diff = np.concatenate(vllm_diff)
        diffs = {key: np.concatenate(values) for key, values in diffs.items()}
        result = {
            "group": group,
            "split": split,
            "samples": len(picks),
            "positions": checked,
            "top_id_set_mismatch": id_mismatch,
            "top_id_set_mismatch_same_hidden": id_mismatch_same_hidden,
            "swapped_mass": {"max": round(float(max(swapped_mass, default = 0.0)), 6), "mean": round(float(np.mean(swapped_mass)) if swapped_mass else 0.0, 6)},
            "max_abs_diff": {key: round(float(value.max()), 6) for key, value in diffs.items()},
            "mean_abs_diff": {key: round(float(value.mean()), 6) for key, value in diffs.items()},
            "stored_lp_minus_vllm_lp": {
                "mean": round(float(vllm_diff.mean()), 5),
                "mean_abs": round(float(np.abs(vllm_diff).mean()), 5),
                "p99_abs": round(float(np.percentile(np.abs(vllm_diff), 99)), 5),
            },
        }
        print(json.dumps(result, ensure_ascii = False), flush = True)
        with open(teacher_dir / "verify.jsonl", "a", encoding = "utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii = False) + "\n")
        return result

    @staticmethod
    def stored(teacher_dir, number):
        """读取一个 teacher shard 的四个数组（内存映射）。"""
        arrays = {}
        for name, path in teacher_paths(teacher_dir = teacher_dir, number = number).items():
            _, dtype, width = SUFFIXES[name]
            data = np.memmap(path, dtype = dtype, mode = "r")
            arrays[name] = data.reshape(-1, width) if width > 1 else data
        return arrays


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mllm", required = True, help = "阶段 1 融合模型目录（训练用的模型）")
    parser.add_argument("--replay-dir", required = True)
    parser.add_argument("--plan", required = True, help = "逗号分隔的 组:划分，按顺序处理")
    parser.add_argument("--estimate", action = "store_true", help = "只统计各划分的生成 token 数和需要的磁盘空间，不加载模型")
    parser.add_argument("--verify", type = int, default = 0, help = "抽这么多条已完成的样本与全词表直接计算对照，不写数据")
    parser.add_argument("--time-budget", type = float, default = 720, help = "秒；超过后写完当前批次就退出，下次运行续跑")
    parser.add_argument("--max-batch-tokens", type = int, default = 65536, help = "一批填充后的 token 数上限")
    parser.add_argument("--head-chunk", type = int, default = 4096, help = "输出层每次计算的位置数（float32 logits 约 4 GB）")
    parser.add_argument("--min-free-gb", type = float, default = 100, help = "写完后数据盘至少剩余的空间")
    parser.add_argument("--exclude-unclosed-think", action = "store_true", help = "free_chat_think 中没有写出 </think> 的样本不计算")
    parser.add_argument("--log-every", type = float, default = 60)
    args = parser.parse_args()
    # 计划中的「部分」只用于生成，这里按（组, 划分）去重
    plan = list(dict.fromkeys((group, split) for group, split, _ in parse_plan(plan = args.plan)))
    global EXCLUDE_UNCLOSED_THINK
    EXCLUDE_UNCLOSED_THINK = args.exclude_unclosed_think

    if args.estimate:
        total = 0
        rows = []
        for group, split in plan:
            samples, tokens = split_tokens(replay_dir = args.replay_dir, group = group, split = split)
            total += tokens
            rows.append({"group": group, "split": split, "samples": samples, "gen_tokens": tokens, "teacher_bytes": tokens * BYTES_PER_TOKEN})
            print(f"{group}/{split}: {samples} 条，{tokens:,} 个生成 token，需要 {tokens * BYTES_PER_TOKEN / 1e9:.2f} GB", flush = True)
        free = free_bytes(path = args.replay_dir)
        print(f"合计 {total:,} 个生成 token，需要 {total * BYTES_PER_TOKEN / 1e9:.2f} GB（每 token {BYTES_PER_TOKEN} 字节）；数据盘剩余 {free / 1e9:.1f} GB", flush = True)
        print(json.dumps({"splits": rows, "total_tokens": total, "total_bytes": total * BYTES_PER_TOKEN, "free_bytes": free}), flush = True)
        return

    teacher = Teacher(args = args)
    if args.verify:
        for group, split in plan:
            teacher.verify(group = group, split = split, count = args.verify)
        return
    deadline = time.time() + args.time_budget
    all_done = True
    for group, split in plan:
        if time.time() >= deadline:
            all_done = False
            break
        finished = teacher.run_split(group = group, split = split, deadline = deadline)
        all_done = all_done and finished
        if not finished:
            break
    print("全部完成" if all_done else "时间用完，下次运行续跑", flush = True)
    sys.exit(0 if all_done else 3)


if __name__ == "__main__":
    main()
