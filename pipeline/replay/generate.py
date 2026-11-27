"""用 vLLM 从参考模型采样回放文本，逐个样本写入 shard（格式见 `store.py`），可以随时中断、再次运行时续跑。

    flock <工作目录>/gpu.lock python -m pipeline.replay.generate --model <参考模型目录> --replay-dir <回放目录> \
        --plan free_raw:eval,free_chat:eval,cont_raw:eval,cont_chat:eval --target eval=1000000,train=12000000 \
        --time-budget 720

在 `envs/mllm-eval` 中运行。采样设置：温度 1，不设 top-k、top-p、min-p，没有重复惩罚，每个样本最多生成 16,384 个
token（续写的提示另计），遇到 `<|im_end|>` 或 `<|endoftext|>` 停止，结束符保留在生成结果中。每个生成位置保存采样到的
token 和它的对数概率；`--topk 20`（默认）时另存参考模型的 top-20 token 与对数概率，`--topk 0` 时不存（vLLM 的对数概率
都是 raw_logprobs，即温度处理之前的 log softmax）。

两种模式：

- 筛选（默认，#70 的 `replay/`）：生成过程中每多出 `--check-every` 个 token，就用 `degeneration.degeneration_onset`
  检查一次。发现循环（原样或近似）后再多生成 `--loop-margin` 个 token 就中止这个请求，`finish_reason` 记为 `loop`。
  最终的截断由 finalize 按同样的规则重新计算，起点之前的 token 与不中止时完全相同；多留的余量让起点更早、但确认得更晚的
  循环（周期较长）也能被找到。free_chat 的用户消息出现循环时整条样本不用，立即中止，状态记为 `user_loop`。不加这一步时，
  无模板两组的大部分算力花在循环上（试运行中 cont_raw 生成的 token 只有 23% 在循环起点之前）。
- 不筛选（`--no-filter`，`replay-unfiltered/`，按 arXiv 2605.26097 的纯自回放）：不检查循环、不提前中止，每条样本都生成
  到结束符或 16,384 个 token 的上限，全部保留。循环只在 finalize 中作为元数据记录。
- 丢弃循环（`--drop-loops`，`replay-noloop/`）：检查方法与筛选模式相同，发现循环就立即中止，整条样本丢弃，不截断；
  索引中只记一行元数据（状态 `loop`、循环出现的片段、位置、种类和花掉的 token 数），不写 token。没有循环的样本生成到
  结束符或上限，原样保留；每段生成结束时再按 finalize 的规则完整检查一次。循环检测只看前文：在前缀上发现的循环，
  在完整序列上同样存在，所以提前中止丢弃得到的保留集合与「全部生成完再丢弃有循环的样本」完全相同。

`--plan` 按顺序处理各个（组, 划分）。每个划分的进度以「保留的生成 token 数」计（筛选模式为循环起点之前的生成 token，
不筛选模式为全部生成 token，丢弃循环模式为没有循环的样本的全部生成 token），达到目标后停止投放新提示，等在途的样本全部完成。提示按 prompts.jsonl 的顺序投放，
已完成的跳过，所以最终的样本是提示清单的前若干条，长样本与短样本一样完整保留。

运行超过 `--time-budget` 秒后，在途的请求全部中止，已生成的部分存进 `raw/inflight.pkl`，进程退出；下次运行先把这些样本
接着生成：提示为原提示加已生成的 token，上限扣除已生成的数量。每个 token 仍按参考模型在前文条件下的分布采样，接着生成的
样本与一次生成完的样本分布相同。这样单次持有 GPU 锁的时间约为模型加载 + time-budget，不必等最长的样本写完。
退出码 0 表示计划全部完成，3 表示还要续跑。

只有一个 token 的提示（free_raw）不直接交给 vLLM：第一个生成 token 已由 prompts 按参考模型的分布抽好（`first` 字段，
原因见 `first_token.py`），这里把它放在生成结果的最前面，vLLM 从两个 token 的提示接着生成。

free_chat 分两段生成：先从 `<|im_start|>user\\n` 生成用户消息，以 `<|im_end|>` 结束后补上
`\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`，再生成助手回答，两段合计不超过 16,384 个 token。
用户消息没有以 `<|im_end|>` 结束（达到长度上限或以 `<|endoftext|>` 结束）的样本记为 `no_user_end`，不生成回答；
筛选模式不用这些样本，不筛选模式原样保留（只有用户消息）。

free_sys_chat、free_sys_chat_think（#71）分三段生成：先从 `<|im_start|>system\\n` 生成系统提示，以 `<|im_end|>` 结束后补上
`\\n<|im_start|>user\\n`，再生成用户消息，之后与 free_chat、free_chat_think 相同（补上的 token 都取自 `templates.json`）。三段合计
不超过 16,384 个 token；系统提示没有以 `<|im_end|>` 结束的样本记为 `no_system_end`，系统提示中出现循环（筛选模式）记为
`system_loop`。每一段的提示都是前面全部片段（提示、已生成的 token、补上的模板）依次拼接。
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from pipeline.replay.degeneration import degeneration_onset, loop_onset, near_loop_onset
from pipeline.replay.groups import CHAT_GROUPS, SELF_SYSTEM_GROUPS, THINK_GROUPS, self_roles
from pipeline.replay.store import TOP_K, ShardReader, ShardWriter, read_all_records, repair
from pipeline.tokenizer_fix import check_tokenizer

MAX_NEW_TOKENS = 16384
# 估计平均保留 token 数时的先验：相当于 20 条各 --prior-mean 个 token 的样本
PRIOR_SAMPLES = 20


def parse_plan(plan):
    """「组:划分」或「组:划分@部分」的列表，返回 (组, 划分, 部分或 None)。"""
    items = []
    for item in plan.split(","):
        if not item:
            continue
        name, _, part = item.partition("@")
        group, split = name.split(":")
        items.append((group, split, part or None))
    return items


def parse_targets(text):
    return {key: int(float(value)) for key, value in (item.split("=") for item in text.split(","))}


class Stream:
    """一个请求已经生成的 token 与对数概率，按 DELTA 输出逐步累积。

    prompt_ids 是这个请求的提示（不含预先抽好的第一个 token），budget 是它最多生成的 token 数（含第一个 token）；
    中途暂停后用 prompt_ids + token_ids 作提示、budget - length 作上限接着生成。
    """

    def __init__(self, state, role, prompt_ids, budget, topk):
        self.state = state
        self.role = role
        self.prompt_ids = list(prompt_ids)
        self.budget = budget
        self.width = topk + 1
        self.token_ids = []
        self.flat_ids = []
        self.flat_lp = []
        self.length = 0
        self.next_check = 0
        self.abort_at = None

    def add(self, completion):
        """追加一步的输出：采样到的 token，以及每个位置的 1 + k 个 (token, 对数概率)，第一个是采样到的 token 自身。"""
        if not completion.token_ids:
            return
        self.token_ids.extend(completion.token_ids)
        self.flat_ids.extend(completion.logprobs.token_ids)
        self.flat_lp.extend(completion.logprobs.logprobs)
        self.length = len(self.token_ids)

    def add_first(self, first):
        """提示清单中预先抽好的第一个生成 token（见 first_token.py），按 vLLM 输出的同样排列放在最前面。"""
        self.token_ids.append(first["id"])
        self.flat_ids.extend([first["id"]] + first["topk_ids"][: self.width - 1])
        self.flat_lp.extend([first["logprob"]] + first["topk_lp"][: self.width - 1])
        self.length = len(self.token_ids)

    def arrays(self):
        """(采样 token, 它的对数概率, top-k token, top-k 对数概率)；不存 top-k 时后两项为 None。"""
        width = self.width
        ids = np.asarray(self.token_ids, dtype = np.int32)
        all_ids = np.asarray(self.flat_ids, dtype = np.int32).reshape(-1, width)
        all_lp = np.asarray(self.flat_lp, dtype = np.float32).reshape(-1, width)
        assert all_ids.shape[0] == len(ids) and np.array_equal(all_ids[:, 0], ids), f"每个位置应有 {width} 个对数概率，第一个为采样到的 token"
        if width == 1:
            return ids, all_lp[:, 0], None, None
        return ids, all_lp[:, 0], all_ids[:, 1:], all_lp[:, 1:]

    def ids(self):
        return np.asarray(self.token_ids, dtype = np.int32)

    @classmethod
    def restore(cls, fields):
        """由 inflight.pkl 中存的属性字典（只含列表、字典和 numpy 数组）还原。"""
        stream = cls.__new__(cls)
        stream.__dict__.update(fields)
        return stream


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
            max_model_len = args.max_model_len or args.max_new_tokens + 512,
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
        self.think_end = tokenizer.convert_tokens_to_ids("</think>")
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
            logprobs = self.args.topk,
            flat_logprobs = True,
            detokenize = False,
            output_kind = self.delta,
        )

    def run_split(self, group, split, target, deadline, part = None):
        """生成一个（组, 划分）；part 不为空时只投放提示清单中 `part` 字段等于它的提示，进度也只计这一部分。"""
        args = self.args
        split_dir = Path(args.replay_dir) / group / split
        raw_dir = split_dir / "raw"
        raw_dir.mkdir(parents = True, exist_ok = True)
        repair(raw_dir = raw_dir)
        all_records = read_all_records(raw_dir = raw_dir)
        done = {record["prompt_index"] for record in all_records}
        records = [record for record in all_records if part is None or record.get("part") == part]
        # 思考模式的组（free_chat_think、free_sys_chat_think）加 --count-closed-think 时，只有写出 </think> 的样本计入进度
        # （没有写出的照样保留）
        count_closed = args.count_closed_think and group in THINK_GROUPS
        reader = ShardReader(raw_dir = raw_dir)

        def closed_in_shard(record):
            pieces = [piece for piece in record["pieces"] if piece.get("gen") == "assistant"]
            return bool(pieces) and self.think_end in np.asarray(reader.piece(number = record["shard"], piece = pieces[-1])).tolist()

        def progress(record):
            if not count_closed or record["status"] == "loop":
                return record["kept_gen"]
            closed = record.get("think_closed")
            if closed is None:
                closed = closed_in_shard(record = record)
            return record["kept_gen"] if closed else 0

        kept_total = sum(progress(record = record) for record in records)
        inflight_path = raw_dir / ("inflight.pkl" if part is None else f"inflight-{part}.pkl")
        label = f"{group}/{split}" + (f"@{part}" if part else "")
        suspended = []
        if inflight_path.exists():
            with open(inflight_path, "rb") as handle:
                # 上次进程中途退出时，其中一部分可能已经写完，按 prompt_index 去掉
                saved = pickle.load(handle)
            suspended = [Stream.restore(fields = fields) for fields in saved if fields["state"]["row"]["prompt_index"] not in done]
        if kept_total >= target and not suspended:
            print(f"[{label}] 已有 {kept_total} 个保留 token，达到目标 {target}，跳过", flush = True)
            inflight_path.unlink(missing_ok = True)
            return True
        with open(split_dir / "prompts.jsonl", encoding = "utf-8") as handle:
            prompts = [json.loads(line) for line in handle]
        started = done | {stream.state["row"]["prompt_index"] for stream in suspended}
        pending = (row for row in prompts if row["prompt_index"] not in started and (part is None or row.get("part") == part))
        # 模型自己写的各段依次的角色，以及每段（以 <|im_end|> 结束）之后补上的模板 token
        roles = self_roles(group = group)
        after_ids = {}
        if roles:
            templates = json.loads((Path(args.replay_dir) / "templates.json").read_text(encoding = "utf-8"))
            after_ids["user"] = templates[group]["between_ids"]
            if group in SELF_SYSTEM_GROUPS:
                after_ids["system"] = templates[group]["system_between_ids"]
        main_role = "assistant" if group in CHAT_GROUPS else "text"

        writer = None
        streams = {}
        counters = {"samples": 0, "gen_tokens": 0, "decoded": 0, "loop_aborts": 0, "dropped": 0, "resumed": len(suspended), "suspended": 0}
        prior = parse_targets(text = args.prior_mean).get(group, 1000) if "=" in args.prior_mean else float(args.prior_mean)
        default_inflight = int(args.max_num_seqs * 1.5)
        if "=" in args.max_inflight:
            max_inflight = parse_targets(text = args.max_inflight).get(group, default_inflight)
        else:
            max_inflight = int(args.max_inflight) if int(args.max_inflight) > 0 else default_inflight
        mean_kept = (kept_total + prior * PRIOR_SAMPLES) / (len(records) + PRIOR_SAMPLES)
        finished_samples = len(records)
        exhausted = False
        t_start = time.time()
        t_log = t_start
        print(
            f"[{label}] 开始：已完成 {len(records)} 条，保留 token {kept_total} / {target}，接着生成上次暂停的 {len(suspended)} 条",
            flush = True,
        )

        def request(request_id, stream):
            """把 stream 交给 vLLM：提示为原提示加已生成的 token，上限扣除已生成的数量。"""
            streams[request_id] = stream
            remaining = stream.budget - stream.length
            if remaining <= 0:
                piece_done(request_id = request_id, finish_reason = "length")
                return
            prompt_ids = stream.prompt_ids + stream.token_ids
            self.engine.add_request(request_id = request_id, prompt = {"prompt_token_ids": prompt_ids}, params = self.params(max_tokens = remaining))

        def phase_of(role):
            """请求编号中的段号：模型自己写的各段依次为 1、2、3，其他组只有一段。"""
            return roles.index(role) + 1 if roles else 1

        def context_ids(state):
            """样本到目前为止的全部 token：直接给定的片段（提示、模板）与已生成的片段依次拼接。"""
            ids = []
            gen_index = 0
            for piece in state["pieces"]:
                if "gen" in piece:
                    ids.extend(state["arrays"][gen_index][0].tolist())
                    gen_index += 1
                else:
                    ids.extend(piece["ids"])
            return ids

        def submit(state, phase, prompt_ids, max_tokens, role, first = None):
            request_id = f"{state['row']['prompt_index']}-{phase}"
            stream = Stream(
                state = state,
                role = role,
                prompt_ids = prompt_ids,
                budget = max_tokens,
                topk = args.topk,
            )
            if first is not None:
                stream.add_first(first = first)
                if first["id"] in self.stop_ids:
                    streams[request_id] = stream
                    piece_done(request_id = request_id, finish_reason = "stop")
                    return
            request(request_id = request_id, stream = stream)

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

        def finish_sample(state, status, kept, loop = None):
            """写出一条完成的样本。loop 不为空时（--drop-loops 发现循环）只写元数据：提示片段、循环位置与花掉的 token 数。"""
            nonlocal writer, kept_total, finished_samples, mean_kept
            if writer is None:
                writer = ShardWriter(raw_dir = raw_dir, topk = args.topk > 0)
            pieces = []
            gen_index = 0
            for piece in state["pieces"] if loop is None else state["pieces"][:1]:
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
            record = {
                "sample_id": f"{group}-{split}-{row['prompt_index']:07d}",
                "prompt_index": row["prompt_index"],
                **({"part": row["part"]} if "part" in row else {}),
                "status": status,
                "n_gen": int(sum(len(arrays[0]) for arrays in state["arrays"])),
                "kept_gen": int(kept),
                "pieces": pieces,
            }
            if loop is not None:
                record["loop"] = loop
                counters["dropped"] += 1
            counted = kept
            if group in THINK_GROUPS and loop is None:
                # 助手回答（思考 + 回答）中是否写出了 </think>；没有助手回答的样本记为否
                answer = [arrays[0] for arrays, role in zip(state["arrays"], state["roles"]) if role == "assistant"]
                record["think_closed"] = bool(answer) and self.think_end in answer[-1].tolist()
                if count_closed and not record["think_closed"]:
                    counted = 0
            writer.add_record(record)
            kept_total += counted
            counters["samples"] += 1
            finished_samples += 1
            # 按全部已完成样本的平均值估计在途样本；短样本先完成，滑动平均会偏低，导致投放过多
            mean_kept = (kept_total + prior * PRIOR_SAMPLES) / (finished_samples + PRIOR_SAMPLES)

        def piece_done(request_id, finish_reason):
            """请求结束（正常停止、达到上限或因循环中止）后的处理：模型自己写的系统提示、用户消息正常结束时补上模板，
            接着生成下一段，否则样本完成。"""
            stream = streams.pop(request_id)
            state = stream.state
            ids = close_piece(stream = stream, finish_reason = finish_reason)
            if args.drop_loops:
                # 用 finalize 的同一套规则检查这一段；有循环（原样或近似）就整条丢弃，只记元数据
                exact, near = loop_onset(ids = ids), near_loop_onset(ids = ids)
                if exact is not None or near is not None:
                    position = min(value for value in (exact, near) if value is not None)
                    loop = {"role": stream.role, "position": int(position), "kind": "exact" if position == exact else "near", "finish_reason": finish_reason}
                    finish_sample(state = state, status = "loop", kept = 0, loop = loop)
                    return
            onset = None if args.no_filter or args.drop_loops else degeneration_onset(ids = ids)
            if stream.role in after_ids:
                # 系统提示或用户消息：正常以 <|im_end|> 结束时补上模板，接着生成下一段
                if onset is not None:
                    finish_sample(state = state, status = f"{stream.role}_loop", kept = 0)
                    return
                used = state.get("used", 0) + len(ids)
                if finish_reason == "stop" and len(ids) and ids[-1] == self.im_end and used < args.max_new_tokens:
                    state["pieces"].append({"ids": after_ids[stream.role]})
                    state["used"] = used
                    next_role = roles[roles.index(stream.role) + 1]
                    submit(
                        state = state,
                        phase = phase_of(role = next_role),
                        prompt_ids = context_ids(state = state),
                        max_tokens = args.max_new_tokens - used,
                        role = next_role,
                    )
                    return
                status = f"no_{stream.role}_end"
                finish_sample(state = state, status = status, kept = used if args.no_filter or args.drop_loops else 0)
                return
            kept = state.get("used", 0) + (onset if onset is not None else len(ids))
            finish_sample(state = state, status = "ok", kept = kept)

        for stream in suspended:
            request(request_id = f"{stream.state['row']['prompt_index']}-{phase_of(role = stream.role)}", stream = stream)

        while True:
            if time.time() >= deadline and streams:
                # 时间用完：中止在途请求，已生成的部分存起来，下次运行接着生成
                self.engine.abort_request(request_ids = list(streams))
                counters["suspended"] = len(streams)
                tmp = inflight_path.with_suffix(".pkl.tmp")
                with open(tmp, "wb") as handle:
                    pickle.dump([vars(stream) for stream in streams.values()], handle)
                tmp.replace(inflight_path)
                streams.clear()
                break
            # 在途样本的保留 token 估计：已生成的长度与平均值中较大的一个（模型自己写的几段按同一条样本计）
            lengths = {}
            for stream in streams.values():
                index = stream.state["row"]["prompt_index"]
                lengths[index] = max(lengths.get(index, 0), stream.state.get("used", 0) + stream.length)
            projected = kept_total + sum(max(length, mean_kept) for length in lengths.values())
            feeding = time.time() < deadline and projected < target and not exhausted
            while feeding and len(streams) < max_inflight:
                row = next(pending, None)
                if row is None:
                    exhausted = True
                    break
                state = {"row": row, "pieces": [{"ids": row["prompt_ids"]}], "arrays": [], "roles": []}
                role = roles[0] if roles else main_role
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
                counters["decoded"] += len(completion.token_ids)
                stream.add(completion = completion)
                if output.finished:
                    piece_done(request_id = output.request_id, finish_reason = completion.finish_reason)
                    continue
                if args.no_filter:
                    continue
                if stream.abort_at is None and stream.length >= stream.next_check:
                    stream.next_check = stream.length + args.check_every
                    onset = degeneration_onset(ids = stream.ids())
                    if onset is not None:
                        if args.drop_loops:
                            stream.abort_at = stream.length  # 整条丢弃，立即中止
                        else:
                            stream.abort_at = onset if stream.role in after_ids else onset + args.loop_margin
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
                    f"[{label}] {elapsed:.0f}s 完成 {counters['samples']} 条，本次解码 {counters['decoded']} token"
                    f"（{counters['decoded'] / max(elapsed, 1e-9):.0f} tok/s），因循环中止 {counters['loop_aborts']}，"
                    f"在途 {len(streams)}，保留 {kept_total} / {target}",
                    flush = True,
                )
        elapsed = time.time() - t_start
        if not counters["suspended"]:
            inflight_path.unlink(missing_ok = True)
        if writer is not None or counters["decoded"]:
            shard = None
            if writer is not None:
                writer.close()
                shard = writer.number
            run = {
                "shard": shard,
                "group": group,
                "split": split,
                "part": part,
                "samples": counters["samples"],
                "gen_tokens": counters["gen_tokens"],
                "decoded_tokens": counters["decoded"],
                "loop_aborts": counters["loop_aborts"],
                "dropped": counters["dropped"],
                "resumed": counters["resumed"],
                "suspended": counters["suspended"],
                "seconds": round(elapsed, 1),
                "tokens_per_second": round(counters["decoded"] / max(elapsed, 1e-9), 1),
                "load_seconds": round(self.load_seconds, 1),
                "max_num_seqs": args.max_num_seqs,
                "max_inflight": max_inflight,
                "no_filter": args.no_filter,
                "drop_loops": args.drop_loops,
                "topk": args.topk,
                "check_every": args.check_every,
                "loop_margin": args.loop_margin,
                "seed": args.seed,
                "model": os.path.basename(os.path.normpath(args.model)),
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(raw_dir / "runs.jsonl", "a", encoding = "utf-8") as handle:
                handle.write(json.dumps(run, ensure_ascii = False) + "\n")
        print(
            f"[{label}] 本次 {elapsed:.0f}s，完成 {counters['samples']} 条，解码 {counters['decoded']} token"
            f"（{counters['decoded'] / max(elapsed, 1e-9):.0f} tok/s），因循环中止 {counters['loop_aborts']}，"
            f"暂停 {counters['suspended']} 条，保留累计 {kept_total} / {target}",
            flush = True,
        )
        return not counters["suspended"] and (kept_total >= target or exhausted)


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required = True, help = "参考模型目录（export_lm 的输出）")
    parser.add_argument("--replay-dir", required = True)
    parser.add_argument("--plan", required = True, help = "逗号分隔的 组:划分，按顺序处理")
    parser.add_argument(
        "--target",
        default = "eval=1000000,train=12000000",
        help = "每个划分保留的生成 token 目标；可以按组覆盖，例如 eval=1000000,train=30000000,free_raw:train=10000000",
    )
    parser.add_argument("--time-budget", type = float, default = 720, help = "秒；超过后暂停在途样本并退出，下次运行接着生成")
    parser.add_argument(
        "--prior-mean",
        default = "1000",
        help = "估计在途样本最终长度时用的每条样本平均保留 token 数的先验，一个数或按组给出（如 free_raw=10000,free_chat=850）；"
        "在途的样本都要写完，先验过低时开头投放过多，划分的 token 数会超出目标",
    )
    parser.add_argument("--no-filter", action = "store_true", help = "不检查循环、不提前中止，每条样本生成到结束符或上限并全部保留")
    parser.add_argument(
        "--drop-loops",
        action = "store_true",
        help = "发现循环（原样或近似）就立即中止并整条丢弃，只记元数据；其余样本生成到结束符或上限并原样保留，不截断",
    )
    parser.add_argument(
        "--count-closed-think",
        action = "store_true",
        help = "思考模式的组（free_chat_think、free_sys_chat_think）的进度只计写出 </think> 的样本；没有写出的样本照样保留，"
        "索引中 think_closed 为假",
    )
    parser.add_argument("--topk", type = int, default = TOP_K, help = f"每个生成位置另存的 top-k 对数概率个数，只能取 {TOP_K} 或 0（不存）")
    parser.add_argument("--max-new-tokens", type = int, default = MAX_NEW_TOKENS)
    parser.add_argument("--max-model-len", type = int, default = 0, help = "提示加生成的最大长度，默认 max-new-tokens + 512；带工具调用上下文的长提示需要调大")
    parser.add_argument("--max-num-seqs", type = int, default = 256, help = "并发序列数；KV 缓存约 177 万 token")
    parser.add_argument(
        "--max-inflight",
        default = "0",
        help = "在途请求上限，一个数或按组给出（如 free_raw=160,cont_raw=160），默认 max-num-seqs 的 1.5 倍。"
        "不筛选时无模板两组的样本大多写到 16k 上限，在途样本的 KV 缓存超过容量（约 177 万 token）后 vLLM 反复抢占、重算，"
        "吞吐降到约 2,000 token/s，这两组要把在途样本数限制在容量以内",
    )
    parser.add_argument("--max-num-batched-tokens", type = int, default = 4096)
    parser.add_argument("--gpu-memory-utilization", type = float, default = 0.88)
    parser.add_argument("--check-every", type = int, default = 256, help = "每生成这么多 token 检查一次循环（筛选模式）")
    parser.add_argument("--loop-margin", type = int, default = 1024, help = "发现循环后，在起点之后再生成这么多 token 才中止（筛选模式）")
    parser.add_argument("--log-every", type = float, default = 60)
    parser.add_argument("--seed", type = int, default = None, help = "vLLM 采样种子，默认取当前时间")
    args = parser.parse_args()
    if args.topk not in (0, TOP_K):
        parser.error(f"--topk 只能取 {TOP_K} 或 0")
    if args.no_filter and args.drop_loops:
        parser.error("--no-filter 与 --drop-loops 只能选一个")
    if args.seed is None:
        args.seed = int(time.time()) % 1_000_000

    generator = Generator(args = args)
    deadline = time.time() + args.time_budget
    targets = parse_targets(text = args.target)
    all_done = True
    for group, split, part in parse_plan(plan = args.plan):
        if time.time() >= deadline:
            all_done = False
            break
        finished = generator.run_split(
            group = group,
            split = split,
            target = targets.get(f"{group}:{split}@{part}", targets.get(f"{group}:{split}", targets.get(split))),
            deadline = deadline,
            part = part,
        )
        all_done = all_done and finished
        if not finished:
            # 按计划的顺序完成：这个划分没有完成时不开始后面的划分
            break
    print("全部完成" if all_done else "时间用完，下次运行续跑", flush = True)
    # 退出码：0 全部完成，3 还有剩余（run.sh 据此释放 GPU 锁后续跑）
    sys.exit(0 if all_done else 3)


if __name__ == "__main__":
    main()
