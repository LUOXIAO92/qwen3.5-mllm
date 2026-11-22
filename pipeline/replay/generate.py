"""用 vLLM 从参考模型采样回放文本，逐个样本写入 shard（格式见 `store.py`），可以随时中断、再次运行时续跑。

    flock <工作目录>/gpu.lock python -m pipeline.replay.generate --model <参考模型目录> --replay-dir <回放目录> \
        --plan free_raw:eval,free_chat:eval,cont_raw:eval,cont_chat:eval --target eval=1000000,train=12000000 \
        --time-budget 720

在 `envs/mllm-eval` 中运行。采样设置：温度 1，不设 top-k、top-p、min-p，没有重复惩罚，每个样本最多生成 16,384 个
token（续写的提示另计），遇到 `<|im_end|>` 或 `<|endoftext|>` 停止，结束符保留在生成结果中。每个生成位置同时保存
参考模型的 top-20 token 与对数概率（`logprobs = 20`，vLLM 默认的 raw_logprobs，即温度处理之前的 log softmax）。

循环提前停止：生成过程中每多出 `--check-every` 个 token，就用 `degeneration.degeneration_onset` 检查一次。发现循环（原样或近似）
后再多生成 `--loop-margin` 个 token 就中止这个请求，`finish_reason` 记为 `loop`。最终的截断由 finalize 按同样的规则重新计算，
起点之前的 token 与不中止时完全相同；多留的余量让起点更早、但确认得更晚的循环（周期较长）也能被找到。free_chat 的
用户消息出现循环时整条样本不用，立即中止，状态记为 `user_loop`。不加这一步时，无模板两组的大部分算力花在循环上（试运行中
cont_raw 生成的 token 只有 23% 在循环起点之前）。

`--plan` 按顺序处理各个（组, 划分）。每个划分的进度以「保留的生成 token 数」（循环起点之前的生成 token）计，达到目标后
停止投放新提示。提示按 prompts.jsonl 的顺序投放，已完成的跳过；运行超过 `--time-budget` 秒后停止投放，等在途的样本全部
完成再退出，所以单次持有 GPU 锁的时间约为模型加载 + time-budget + 收尾。退出码 0 表示计划全部完成，3 表示还要续跑。

只有一个 token 的提示（free_raw）不直接交给 vLLM：第一个生成 token 已由 prompts 按参考模型的分布抽好（`first` 字段，
原因见 `first_token.py`），这里把它放在生成结果的最前面，vLLM 从两个 token 的提示接着生成。

free_chat 分两段生成：先从 `<|im_start|>user\\n` 生成用户消息，以 `<|im_end|>` 结束后补上
`\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`，再生成助手回答，两段合计不超过 16,384 个 token。
用户消息没有以 `<|im_end|>` 结束（达到长度上限或以 `<|endoftext|>` 结束）的样本记为 `no_user_end`，不生成回答。
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from pipeline.replay.degeneration import degeneration_onset
from pipeline.replay.store import TOP_K, ShardWriter, read_all_records, repair
from pipeline.tokenizer_fix import check_tokenizer

MAX_NEW_TOKENS = 16384
# 估计平均保留 token 数时的先验：相当于 20 条各 1000 token 的样本
PRIOR_SAMPLES = 20


def parse_plan(plan):
    return [tuple(item.split(":")) for item in plan.split(",") if item]


def parse_targets(text):
    return {key: int(float(value)) for key, value in (item.split("=") for item in text.split(","))}


class Stream:
    """一个请求已经生成的 token 与对数概率，按 DELTA 输出逐步累积。"""

    def __init__(self, state, role):
        self.state = state
        self.role = role
        self.token_ids = []
        self.flat_ids = []
        self.flat_lp = []
        self.length = 0
        self.next_check = 0
        self.abort_at = None

    def add(self, completion):
        """追加一步的输出：采样到的 token，以及每个位置的 1 + 20 个 (token, 对数概率)，第一个是采样到的 token 自身。"""
        if not completion.token_ids:
            return
        self.token_ids.extend(completion.token_ids)
        self.flat_ids.extend(completion.logprobs.token_ids)
        self.flat_lp.extend(completion.logprobs.logprobs)
        self.length = len(self.token_ids)

    def add_first(self, first):
        """提示清单中预先抽好的第一个生成 token（见 first_token.py），按 vLLM 输出的同样排列放在最前面。"""
        self.token_ids.append(first["id"])
        self.flat_ids.extend([first["id"]] + first["topk_ids"])
        self.flat_lp.extend([first["logprob"]] + first["topk_lp"])
        self.length = len(self.token_ids)

    def arrays(self):
        """(采样 token, 它的对数概率, top-20 token, top-20 对数概率)。"""
        width = TOP_K + 1
        ids = np.asarray(self.token_ids, dtype = np.int32)
        all_ids = np.asarray(self.flat_ids, dtype = np.int32).reshape(-1, width)
        all_lp = np.asarray(self.flat_lp, dtype = np.float32).reshape(-1, width)
        assert all_ids.shape[0] == len(ids) and np.array_equal(all_ids[:, 0], ids), "每个位置应有 1 + 20 个对数概率，第一个为采样到的 token"
        return ids, all_lp[:, 0], all_ids[:, 1:], all_lp[:, 1:]

    def ids(self):
        return np.asarray(self.token_ids, dtype = np.int32)


class Generator:
    def __init__(self, args):
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import RequestOutputKind

        self.args = args
        self.SamplingParams = SamplingParams
        self.delta = RequestOutputKind.DELTA
        t0 = time.time()
        self.llm = LLM(
            model = args.model,
            dtype = "bfloat16",
            language_model_only = True,
            gpu_memory_utilization = args.gpu_memory_utilization,
            max_model_len = args.max_new_tokens + 512,
            max_num_seqs = args.max_num_seqs,
            max_num_batched_tokens = args.max_num_batched_tokens,
            enable_prefix_caching = False,
            generation_config = "vllm",
            seed = args.seed,
            disable_log_stats = False,  # 定期打印吞吐、KV 缓存占用和抢占次数
        )
        self.load_seconds = time.time() - t0
        self.engine = self.llm.llm_engine
        tokenizer = self.llm.get_tokenizer()
        # 提示是 prompts 一步写好的 token 编号；这里确认参考模型目录的分词器也已修正（#77）
        check_tokenizer(tokenizer = tokenizer, source = args.model)
        self.stop_ids = [tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")]
        self.im_end = self.stop_ids[0]
        print(f"模型加载 {self.load_seconds:.0f} 秒，结束符 {self.stop_ids}", flush = True)

    def params(self, max_tokens):
        return self.SamplingParams(
            n = 1,
            temperature = 1.0,
            top_p = 1.0,
            top_k = 0,
            min_p = 0.0,
            presence_penalty = 0.0,
            frequency_penalty = 0.0,
            repetition_penalty = 1.0,
            max_tokens = max_tokens,
            stop_token_ids = self.stop_ids,
            logprobs = TOP_K,
            flat_logprobs = True,
            detokenize = False,
            output_kind = self.delta,
        )

    def run_split(self, group, split, target, deadline):
        args = self.args
        split_dir = Path(args.replay_dir) / group / split
        raw_dir = split_dir / "raw"
        raw_dir.mkdir(parents = True, exist_ok = True)
        repair(raw_dir = raw_dir)
        records = read_all_records(raw_dir = raw_dir)
        done = {record["prompt_index"] for record in records}
        kept_total = sum(record["kept_gen"] for record in records)
        if kept_total >= target:
            print(f"[{group}/{split}] 已有 {kept_total} 个保留 token，达到目标 {target}，跳过", flush = True)
            return True
        with open(split_dir / "prompts.jsonl", encoding = "utf-8") as handle:
            prompts = [json.loads(line) for line in handle]
        pending = (row for row in prompts if row["prompt_index"] not in done)
        between_ids = None
        if group == "free_chat":
            templates = json.loads((Path(args.replay_dir) / "templates.json").read_text(encoding = "utf-8"))
            between_ids = templates["free_chat"]["between_ids"]
        main_role = "assistant" if group in ("free_chat", "cont_chat") else "text"

        writer = None
        streams = {}
        counters = {"samples": 0, "gen_tokens": 0, "loop_aborts": 0}
        mean_kept = (kept_total + 1000.0 * PRIOR_SAMPLES) / (len(records) + PRIOR_SAMPLES)
        finished_samples = len(records)
        exhausted = False
        t_start = time.time()
        t_log = t_start
        print(f"[{group}/{split}] 开始：已完成 {len(records)} 条，保留 token {kept_total} / {target}", flush = True)

        def submit(state, phase, prompt_ids, max_tokens, role, first = None):
            request_id = f"{state['row']['prompt_index']}-{phase}"
            stream = Stream(state = state, role = role)
            streams[request_id] = stream
            if first is not None:
                stream.add_first(first = first)
                if first["id"] in self.stop_ids:
                    piece_done(request_id = request_id, finish_reason = "stop")
                    return
                prompt_ids = prompt_ids + [first["id"]]
                max_tokens -= 1
            self.engine.add_request(request_id = request_id, prompt = {"prompt_token_ids": prompt_ids}, params = self.params(max_tokens = max_tokens))

        def close_piece(stream, finish_reason):
            """一个请求结束：把它的数组和片段信息记到样本上，返回生成的 token。"""
            arrays = stream.arrays()
            ids = arrays[0]
            stream.state["arrays"].append(arrays)
            stream.state["roles"].append(stream.role)
            stream.state["pieces"].append(
                {
                    "gen": stream.role,
                    "length": int(len(ids)),
                    "finish_reason": finish_reason,
                    "stop_token": int(ids[-1]) if finish_reason == "stop" and len(ids) else None,
                }
            )
            counters["gen_tokens"] += len(ids)
            return ids

        def finish_sample(state, status, kept):
            nonlocal writer, kept_total, finished_samples, mean_kept
            if writer is None:
                writer = ShardWriter(raw_dir = raw_dir)
            pieces = []
            gen_index = 0
            for piece in state["pieces"]:
                if "gen" in piece:
                    sampled, sampled_lp, topk_ids, topk_lp = state["arrays"][gen_index]
                    gen_index += 1
                    offset = writer.add_gen(
                        ids = sampled,
                        lp = sampled_lp,
                        topk_ids = topk_ids,
                        topk_lp = topk_lp,
                    )
                    piece = dict(piece, offset = offset)
                pieces.append(piece)
            row = state["row"]
            writer.add_record(
                {
                    "sample_id": f"{group}-{split}-{row['prompt_index']:07d}",
                    "prompt_index": row["prompt_index"],
                    "status": status,
                    "n_gen": int(sum(len(arrays[0]) for arrays in state["arrays"])),
                    "kept_gen": int(kept),
                    "pieces": pieces,
                }
            )
            kept_total += kept
            counters["samples"] += 1
            finished_samples += 1
            # 按全部已完成样本的平均值估计在途样本；短样本先完成，滑动平均会偏低，导致投放过多
            mean_kept = (kept_total + 1000.0 * PRIOR_SAMPLES) / (finished_samples + PRIOR_SAMPLES)

        def piece_done(request_id, finish_reason):
            """请求结束（正常停止、达到上限或因循环中止）后的处理：free_chat 用户消息正常结束时接着生成回答，否则样本完成。"""
            stream = streams.pop(request_id)
            state = stream.state
            ids = close_piece(stream = stream, finish_reason = finish_reason)
            onset = degeneration_onset(ids = ids)
            if stream.role == "user":
                if onset is not None:
                    finish_sample(state = state, status = "user_loop", kept = 0)
                    return
                used = len(ids)
                if finish_reason == "stop" and used and ids[-1] == self.im_end and used < args.max_new_tokens:
                    state["pieces"].append({"ids": between_ids})
                    state["used"] = used
                    prompt_ids = state["row"]["prompt_ids"] + ids.tolist() + between_ids
                    submit(
                        state = state,
                        phase = 2,
                        prompt_ids = prompt_ids,
                        max_tokens = args.max_new_tokens - used,
                        role = "assistant",
                    )
                    return
                finish_sample(state = state, status = "no_user_end", kept = 0)
                return
            kept = state.get("used", 0) + (onset if onset is not None else len(ids))
            finish_sample(state = state, status = "ok", kept = kept)

        while True:
            # 在途样本的保留 token 估计：已生成的长度与平均值中较大的一个（free_chat 两段按同一条样本计）
            lengths = {}
            for stream in streams.values():
                index = stream.state["row"]["prompt_index"]
                lengths[index] = max(lengths.get(index, 0), stream.state.get("used", 0) + stream.length)
            projected = kept_total + sum(max(length, mean_kept) for length in lengths.values())
            feeding = time.time() < deadline and projected < target and not exhausted
            while feeding and len(streams) < args.max_inflight:
                row = next(pending, None)
                if row is None:
                    exhausted = True
                    break
                state = {"row": row, "pieces": [{"ids": row["prompt_ids"]}], "arrays": [], "roles": []}
                role = "user" if group == "free_chat" else main_role
                submit(
                    state = state,
                    phase = 1,
                    prompt_ids = row["prompt_ids"],
                    max_tokens = args.max_new_tokens,
                    role = role,
                    first = row.get("first"),
                )
                projected += mean_kept
                feeding = projected < target
            if not self.engine.has_unfinished_requests():
                break
            self.engine.do_log_stats_with_interval()
            to_abort = []
            for output in self.engine.step():
                stream = streams.get(output.request_id)
                if stream is None:
                    continue
                completion = output.outputs[0]
                stream.add(completion = completion)
                if output.finished:
                    piece_done(request_id = output.request_id, finish_reason = completion.finish_reason)
                    continue
                if stream.abort_at is None and stream.length >= stream.next_check:
                    stream.next_check = stream.length + args.check_every
                    onset = degeneration_onset(ids = stream.ids())
                    if onset is not None:
                        stream.abort_at = onset if stream.role == "user" else onset + args.loop_margin
                if stream.abort_at is not None and stream.length >= stream.abort_at:
                    to_abort.append(output.request_id)
            if to_abort:
                self.engine.abort_request(request_ids = to_abort)
                for request_id in to_abort:
                    counters["loop_aborts"] += 1
                    piece_done(request_id = request_id, finish_reason = "loop")
            now = time.time()
            if now - t_log >= args.log_every:
                t_log = now
                elapsed = now - t_start
                print(
                    f"[{group}/{split}] {elapsed:.0f}s 完成 {counters['samples']} 条，生成 {counters['gen_tokens']} token，"
                    f"因循环中止 {counters['loop_aborts']}，在途 {len(streams)}，保留 {kept_total} / {target}",
                    flush = True,
                )
        elapsed = time.time() - t_start
        if writer is not None:
            writer.close()
            run = {
                "shard": writer.number,
                "group": group,
                "split": split,
                "samples": counters["samples"],
                "gen_tokens": counters["gen_tokens"],
                "loop_aborts": counters["loop_aborts"],
                "seconds": round(elapsed, 1),
                "tokens_per_second": round(counters["gen_tokens"] / max(elapsed, 1e-9), 1),
                "load_seconds": round(self.load_seconds, 1),
                "max_num_seqs": args.max_num_seqs,
                "check_every": args.check_every,
                "loop_margin": args.loop_margin,
                "seed": args.seed,
                "model": os.path.basename(os.path.normpath(args.model)),
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(raw_dir / "runs.jsonl", "a", encoding = "utf-8") as handle:
                handle.write(json.dumps(run, ensure_ascii = False) + "\n")
        print(
            f"[{group}/{split}] 本次 {elapsed:.0f}s，完成 {counters['samples']} 条，生成 {counters['gen_tokens']} token"
            f"（{counters['gen_tokens'] / max(elapsed, 1e-9):.0f} tok/s），因循环中止 {counters['loop_aborts']}，"
            f"保留累计 {kept_total} / {target}",
            flush = True,
        )
        return kept_total >= target or exhausted


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required = True, help = "参考模型目录（export_lm 的输出）")
    parser.add_argument("--replay-dir", required = True)
    parser.add_argument("--plan", required = True, help = "逗号分隔的 组:划分，按顺序处理")
    parser.add_argument("--target", default = "eval=1000000,train=12000000", help = "每个划分保留的生成 token 目标")
    parser.add_argument("--time-budget", type = float, default = 720, help = "秒；超过后停止投放新提示，等在途样本完成后退出")
    parser.add_argument("--max-new-tokens", type = int, default = MAX_NEW_TOKENS)
    parser.add_argument("--max-num-seqs", type = int, default = 256, help = "并发序列数；KV 缓存约 180 万 token。不提前中止循环时 256 条长样本会频繁抢占重算，中止后 256 条约 1.3–1.5 万 token/s")
    parser.add_argument("--max-inflight", type = int, default = 0, help = "在途请求上限，默认 max-num-seqs 的 1.5 倍")
    parser.add_argument("--max-num-batched-tokens", type = int, default = 4096)
    parser.add_argument("--gpu-memory-utilization", type = float, default = 0.88)
    parser.add_argument("--check-every", type = int, default = 256, help = "每生成这么多 token 检查一次循环")
    parser.add_argument("--loop-margin", type = int, default = 1024, help = "发现循环后，在起点之后再生成这么多 token 才中止")
    parser.add_argument("--log-every", type = float, default = 60)
    parser.add_argument("--seed", type = int, default = None, help = "vLLM 采样种子，默认取当前时间")
    args = parser.parse_args()
    if args.seed is None:
        args.seed = int(time.time()) % 1_000_000
    if args.max_inflight <= 0:
        args.max_inflight = int(args.max_num_seqs * 1.5)

    generator = Generator(args = args)
    deadline = time.time() + args.time_budget
    targets = parse_targets(text = args.target)
    all_done = True
    for group, split in parse_plan(plan = args.plan):
        if time.time() >= deadline:
            all_done = False
            break
        finished = generator.run_split(
            group = group,
            split = split,
            target = targets[split],
            deadline = deadline,
        )
        all_done = all_done and finished
    print("全部完成" if all_done else "时间用完，下次运行续跑", flush = True)
    # 退出码：0 全部完成，3 还有剩余（run.sh 据此释放 GPU 锁后续跑）
    sys.exit(0 if all_done else 3)


if __name__ == "__main__":
    main()
