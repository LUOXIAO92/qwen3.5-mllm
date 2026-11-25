"""在 vLLM 容器中运行的生成程序（任务 #71 的 GPQA，由 pipeline/eval/gpqa_vllm.py 启动）：读一个任务目录中的提示
（token id）、每条的随机数种子和采样设置，用 vLLM 生成，写回每条生成的 token id 与结束原因。

    python3 -m pipeline.eval.vllm_generate --job-dir <任务目录>

只依赖标准库、vLLM 和 transformers（vLLM 镜像中都有），不读题目文字、不打分；解析答案和统计都在宿主机上由 gpqa_vllm
完成。用户 2026-11-25 决定在 docker 镜像 vllm/vllm-openai:nightly-x86_64（vLLM 0.34.0rc1.dev1235、torch 2.14.0+cu130、
transformers 5.23.0）中运行，启动方式见 gpqa_vllm.py。

任务目录（gpqa_vllm 写好）：
- job.json：model（vLLM 加载的语言模型目录）、max_tokens、temperature、stop_token_ids、gpu_memory_utilization、
  max_model_len、max_num_seqs、max_num_batched_tokens、inflight_factor、max_inflight、seed（引擎的全局种子）、log_every；
- requests.jsonl：每行一条 {"index", "seed", "prompt_token_ids"}，按这个顺序交给 vLLM。

输出（写在同一目录）：results.jsonl（每行 {"index", "token_ids", "finish_reason"}，按 index 排序）和 engine.json（vLLM
版本、KV 缓存容量、同时生成的条数、载入与生成的用时、生成 token 数）。engine.json 最后写，有它表示全部完成。

载入：bfloat16，language_model_only（按原始架构 Qwen3_5ForConditionalGeneration 只建语言模型），不用前缀缓存，
generation_config = "vllm"（不从模型目录读默认采样参数）。分词器只用来核对：用 transformers 载入模型目录的分词器，
按 #77 检查分词规则（pipeline/tokenizer_fix.check_tokenizer），并确认 <|im_end|>、<|endoftext|> 的编号就是 job.json 的
stop_token_ids；提示已经是 token id，vLLM 不再分词、不再套模板，生成结果也不解码（detokenize = False）。

采样：温度 temperature，top_p 1、top_k 不限、min_p 0、没有惩罚，每条一个样本，最多 max_tokens 个新 token，生成到
stop_token_ids 中任何一个时停止，结束符留在结果中。每条请求带自己的种子：vLLM 0.34 的采样是 Gumbel-max，噪声由
（种子, 位置, 词表编号）的哈希算出，同一条请求的随机数只取决于种子和位置，与同时生成的其他请求、被抢占后重算都无关。

同时生成的条数：载入后读 vLLM 报告的 KV 缓存容量（cache_config.kv_cache_size_tokens）与「每条都用满 max_model_len 时
能同时放下几条」（kv_cache_max_concurrency，记为 C）；同时交给 vLLM 的请求最多 floor(C × inflight_factor) 条
（max_inflight 大于 0 时直接用它），其余排队，一条结束补进一条（见 gpqa_vllm.py 开头的说明）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")


def inflight_limit(concurrency: float, factor: float, explicit: int, max_num_seqs: int, count: int) -> int:
    """同时交给 vLLM 的请求数上限：explicit 大于 0 时用它，否则为 floor(C × factor)；不超过 max_num_seqs 与请求数，至少 1。"""
    limit = explicit if explicit > 0 else math.floor(concurrency * factor)
    return max(1, min(limit, max_num_seqs, count))


def sampling_params(SamplingParams, final_only, job: dict, seed: int):
    """一条请求的采样设置：温度 job["temperature"]，不设 top-k、top-p、min-p 和惩罚，带自己的种子，只在结束时返回结果。"""
    return SamplingParams(
        n = 1,
        temperature = job["temperature"],
        top_p = 1.0,
        top_k = 0,
        min_p = 0.0,
        presence_penalty = 0.0,
        frequency_penalty = 0.0,
        repetition_penalty = 1.0,
        seed = seed,
        max_tokens = job["max_tokens"],
        stop_token_ids = list(job["stop_token_ids"]),
        detokenize = False,
        output_kind = final_only,
    )


def generate_all(engine, requests: list, make_params, max_inflight: int, log, log_every: float = 60.0) -> dict:
    """requests 为 [{"index", "seed", "prompt_token_ids"}]，按顺序交给 engine，同时最多 max_inflight 条。

    返回 {index: (生成的 token id, finish_reason)}。engine 需要有 vLLM LLMEngine 的 add_request、step、
    has_unfinished_requests 与 do_log_stats_with_interval（测试时换成假的）；make_params(request = 一条请求) 返回它的采样设置。
    """
    pending = list(reversed(requests))
    names = {}
    results = {}
    tokens = 0
    start = time.perf_counter()
    last_log = start
    while pending or engine.has_unfinished_requests():
        while pending and len(names) - len(results) < max_inflight:
            request = pending.pop()
            request_id = f"r{request['index']}"
            names[request_id] = request["index"]
            engine.add_request(
                request_id = request_id,
                prompt = {"prompt_token_ids": list(request["prompt_token_ids"])},
                params = make_params(request = request),
            )
        engine.do_log_stats_with_interval()
        for output in engine.step():
            if not output.finished:
                continue
            completion = output.outputs[0]
            ids = [int(token) for token in completion.token_ids]
            results[names[output.request_id]] = (ids, completion.finish_reason)
            tokens += len(ids)
        now = time.perf_counter()
        if now - last_log >= log_every:
            last_log = now
            log(
                f"{now - start:.0f} 秒：写完 {len(results)} / {len(requests)} 条（{tokens} 个生成 token，"
                f"{tokens / max(now - start, 1e-9):.0f} token/s），在途 {len(names) - len(results)}，排队 {len(pending)}"
            )
    missing = [request["index"] for request in requests if request["index"] not in results]
    if missing:
        raise RuntimeError(f"有 {len(missing)} 条没有得到结果：{missing[:10]}")
    return results


def read_job(job_dir: Path) -> tuple:
    job = json.loads((job_dir / "job.json").read_text(encoding = "utf-8"))
    with open(file = job_dir / "requests.jsonl", encoding = "utf-8") as handle:
        requests = [json.loads(s = line) for line in handle if line.strip()]
    return job, requests


def write_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(data = text, encoding = "utf-8")
    temporary.replace(target = path)


def check_model_tokenizer(model: str, stop_ids: list) -> None:
    """模型目录的分词器按 #77 的规则分词，结束符编号与任务一致；不一致时报错。"""
    from transformers import AutoTokenizer

    from pipeline.tokenizer_fix import check_tokenizer

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = model)
    check_tokenizer(tokenizer = tokenizer, source = model)
    found = [tokenizer.convert_tokens_to_ids(token) for token in STOP_TOKENS]
    if list(found) != list(stop_ids):
        raise RuntimeError(f"{model} 的结束符 {found} 与任务的 stop_token_ids {stop_ids} 不同")


def run(job_dir: Path, log) -> None:
    """载入 vLLM、按 KV 缓存容量定下同时生成的条数，生成全部请求，写 results.jsonl 与 engine.json。"""
    job, requests = read_job(job_dir = job_dir)
    check_model_tokenizer(model = job["model"], stop_ids = job["stop_token_ids"])
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    start = time.perf_counter()
    llm = LLM(
        model = job["model"],
        dtype = "bfloat16",
        language_model_only = True,
        gpu_memory_utilization = job["gpu_memory_utilization"],
        max_model_len = job["max_model_len"],
        max_num_seqs = job["max_num_seqs"],
        max_num_batched_tokens = job["max_num_batched_tokens"],
        enable_prefix_caching = False,
        generation_config = "vllm",
        seed = job["seed"],
        disable_log_stats = False,
    )
    load_seconds = round(number = time.perf_counter() - start, ndigits = 1)
    cache = llm.llm_engine.vllm_config.cache_config
    kv_tokens = getattr(cache, "kv_cache_size_tokens", None)
    concurrency = getattr(cache, "kv_cache_max_concurrency", None)
    if concurrency is None:
        if not kv_tokens:
            raise RuntimeError("vLLM 没有报告 KV 缓存容量（cache_config.kv_cache_size_tokens），请用 --max-inflight 直接指定")
        concurrency = kv_tokens / job["max_model_len"]
    max_inflight = inflight_limit(
        concurrency = concurrency,
        factor = job["inflight_factor"],
        explicit = job["max_inflight"],
        max_num_seqs = job["max_num_seqs"],
        count = len(requests),
    )
    log(
        f"vLLM {vllm.__version__} 载入 {load_seconds} 秒：KV 缓存 {kv_tokens:,} token，每条 {job['max_model_len']:,} token 时"
        f"同时放得下 {concurrency:.1f} 条；同时生成最多 {max_inflight} 条，共 {len(requests)} 条"
    )

    def make_params(request):
        return sampling_params(
            SamplingParams = SamplingParams,
            final_only = RequestOutputKind.FINAL_ONLY,
            job = job,
            seed = request["seed"],
        )

    start = time.perf_counter()
    results = generate_all(
        engine = llm.llm_engine,
        requests = requests,
        make_params = make_params,
        max_inflight = max_inflight,
        log = log,
        log_every = job.get("log_every", 60.0),
    )
    seconds = round(number = time.perf_counter() - start, ndigits = 1)
    lines = [
        json.dumps(obj = {"index": index, "token_ids": ids, "finish_reason": reason})
        for index, (ids, reason) in sorted(results.items())
    ]
    write_atomic(path = job_dir / "results.jsonl", text = "\n".join(lines) + "\n")
    tokens = sum(len(ids) for ids, _ in results.values())
    engine = {
        "version": vllm.__version__,
        "kv_cache_tokens": kv_tokens,
        "max_concurrency": round(number = float(concurrency), ndigits = 2),
        "max_inflight": max_inflight,
        "flashinfer_sampler": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") != "0",
        "load_seconds": load_seconds,
        "seconds": seconds,
        "generated_tokens": tokens,
        "tokens_per_second": round(number = tokens / max(seconds, 1e-9), ndigits = 1),
    }
    write_atomic(path = job_dir / "engine.json", text = json.dumps(obj = engine, indent = 1) + "\n")
    log(f"生成 {len(results)} 条，{tokens} 个 token，{seconds} 秒（{engine['tokens_per_second']:.0f} token/s）")


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job-dir", required = True, type = Path, help = "任务目录（job.json 与 requests.jsonl）")
    args = parser.parse_args(args = argv)

    def log(message):
        print(f"{time.strftime('%H:%M:%S')} {message}", flush = True)

    run(job_dir = args.job_dir, log = log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
