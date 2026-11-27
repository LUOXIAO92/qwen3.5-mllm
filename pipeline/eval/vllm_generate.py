"""在 vLLM 容器中运行的生成程序（任务 #71 的 GPQA，由 pipeline/eval/gpqa_vllm.py 启动）：读一个任务目录中的提示
（token id）、每条的随机数种子和采样设置，用 vLLM 生成，写回每条生成的 token id 与结束原因。

    python3 -m pipeline.eval.vllm_generate --job-dir <任务目录>

只依赖标准库、vLLM、transformers、torch 与 safetensors（vLLM 镜像中都有），不读题目文字、不打分；解析答案和统计都在宿主机上由 gpqa_vllm
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

其他评测（任务 #71 的本机评测改用 vLLM，宿主机一侧见 pipeline/eval/vllm_host.py）另用到下面几项，都可以不给：

- job.json 的 enable_prompt_embeds（音频评测：提示中音频位置的向量由宿主机算好）、embeds_token_id（音频占位符
  <|audio_pad|> 的编号）、max_logprobs（每步最多返回几个对数概率）、logprobs_mode（vLLM 的 logprobs_mode，核对首步 logits
  时用 raw_logits）、logprob_steps（生成的前几步记下对数概率）；
- requests.jsonl 每条可以覆盖 job.json 的 max_tokens、temperature，另可给 logprobs（每步概率最高的几个）、
  logprob_token_ids（只要这几个 token 的对数概率，选择题用）、prompt_logprobs（0：提示中每个位置实际 token 的对数概率，
  困惑度与回放差距用）、embeds（{"file", "key"}：任务目录中 safetensors 文件里这条的音频向量，行数等于提示中
  embeds_token_id 的个数）。

带 embeds 的请求用 vLLM 的混合输入（prompt_token_ids 加 prompt_embeds 与 prompt_is_token_ids）：音频占位符的位置换成
宿主机算好的向量，其余位置仍按 token id 查词嵌入，与融合模型 forward 中 masked_scatter 的结果相同。位置编码不受影响：
vLLM 对没有图像、视频的请求按 token 顺序给一维位置（M-RoPE 三个分量相同），与融合模型把音频 token 当作文字排位置
（modeling_qwen3_5_mllm.get_rope_index）一致。

results.jsonl 每行另有 logprobs（[[[token id, 对数概率], ...] 每步一个列表]，只在请求了 logprobs 或 logprob_token_ids 时）
与 prompt_logprobs（[None, 第 2 个 token 的对数概率, ...]，只在请求了 prompt_logprobs 时）。
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


def sampling_params(SamplingParams, final_only, job: dict, seed: int, request: dict | None = None):
    """一条请求的采样设置：温度 job["temperature"]；top_p、top_k、min_p、presence_penalty、repetition_penalty 取 job 中的值
    （没有时为 1、不限、0、0、1，即不设），带自己的种子，只在结束时返回结果。

    request 中的 max_tokens、temperature 覆盖 job 的值；logprobs、logprob_token_ids、prompt_logprobs 有值时一并交给 vLLM。
    """
    request = request or {}
    extra = {key: request[key] for key in ("logprobs", "logprob_token_ids", "prompt_logprobs") if request.get(key) is not None}
    return SamplingParams(
        n = 1,
        temperature = request.get("temperature", job["temperature"]),
        top_p = job.get("top_p", 1.0),
        top_k = job.get("top_k", 0),
        min_p = job.get("min_p", 0.0),
        presence_penalty = job.get("presence_penalty", 0.0),
        frequency_penalty = 0.0,
        repetition_penalty = job.get("repetition_penalty", 1.0),
        seed = seed,
        max_tokens = request.get("max_tokens", job["max_tokens"]),
        stop_token_ids = list(job["stop_token_ids"]),
        detokenize = False,
        output_kind = final_only,
        **extra,
    )


def generate_all(
    engine,
    requests: list,
    make_params,
    max_inflight: int,
    log,
    log_every: float = 60.0,
    make_prompt = None,
    collect = None,
) -> dict:
    """requests 为 [{"index", "seed", "prompt_token_ids"}]，按顺序交给 engine，同时最多 max_inflight 条。

    返回 {index: (生成的 token id, finish_reason)}。engine 需要有 vLLM LLMEngine 的 add_request、step、
    has_unfinished_requests 与 do_log_stats_with_interval（测试时换成假的）；make_params(request = 一条请求) 返回它的采样设置；
    make_prompt(request = 一条请求) 返回交给 vLLM 的提示（不给时为 {"prompt_token_ids": ...}）；collect(index = 编号,
    output = vLLM 的结果) 在每条结束时调用，用来另外取出对数概率。
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
                prompt = {"prompt_token_ids": list(request["prompt_token_ids"])} if make_prompt is None else make_prompt(request = request),
                params = make_params(request = request),
            )
        engine.do_log_stats_with_interval()
        for output in engine.step():
            if not output.finished:
                continue
            completion = output.outputs[0]
            ids = [int(token) for token in completion.token_ids]
            results[names[output.request_id]] = (ids, completion.finish_reason)
            if collect is not None:
                collect(index = names[output.request_id], output = output)
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


def logprob_rows(entries, steps: int) -> list:
    """vLLM 每步的 {token id: Logprob} 换成 [[token id, 对数概率], ...]（按 token id 排列），只取前 steps 步。"""
    rows = []
    for entry in list(entries or [])[:steps]:
        rows.append(sorted([int(token), float(value.logprob)] for token, value in entry.items()))
    return rows


def prompt_logprob_values(prompt_logprobs, prompt_ids: list) -> list:
    """提示中每个位置实际 token 的对数概率：第一个位置没有（None），其余取 vLLM 返回的 {token id: Logprob} 中这个 token 的值。"""
    values = [None]
    for entry, token in zip(list(prompt_logprobs)[1:], list(prompt_ids)[1:]):
        values.append(float(entry[int(token)].logprob))
    return values


class EmbedsReader:
    """任务目录中 safetensors 文件里各条请求的音频向量，用到时才读（文件保持打开）。"""

    def __init__(self, job_dir: Path):
        self.job_dir = Path(job_dir)
        self.handles = {}

    def get(self, file: str, key: str):
        from safetensors import safe_open

        if file not in self.handles:
            self.handles[file] = safe_open(filename = str(self.job_dir / file), framework = "pt", device = "cpu")
        return self.handles[file].get_tensor(key)


def mixed_prompt(token_ids: list, rows, token_id: int) -> dict:
    """混合输入：token_ids 中等于 token_id（音频占位符）的位置依次换成 rows 的各行，其余位置用 token id。

    prompt_embeds 与提示等长（token 位置的行不用，置零），prompt_is_token_ids 为真处按 token id 查词嵌入。
    """
    import torch

    is_token = [int(token) != token_id for token in token_ids]
    count = len(is_token) - sum(is_token)
    if count != rows.shape[0]:
        raise ValueError(f"提示中有 {count} 个音频占位符，向量有 {rows.shape[0]} 行")
    full = torch.zeros(size = (len(token_ids), rows.shape[1]), dtype = rows.dtype)
    full[torch.tensor(data = [not flag for flag in is_token], dtype = torch.bool)] = rows
    return {"prompt_token_ids": [int(token) for token in token_ids], "prompt_embeds": full, "prompt_is_token_ids": is_token}


def run(job_dir: Path, log) -> None:
    """载入 vLLM、按 KV 缓存容量定下同时生成的条数，生成全部请求，写 results.jsonl 与 engine.json。"""
    job, requests = read_job(job_dir = job_dir)
    check_model_tokenizer(model = job["model"], stop_ids = job["stop_token_ids"])
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    options = {}
    if job.get("enable_prompt_embeds"):
        options["enable_prompt_embeds"] = True
    if job.get("max_logprobs") is not None:
        options["max_logprobs"] = int(job["max_logprobs"])
    if job.get("logprobs_mode"):
        options["logprobs_mode"] = job["logprobs_mode"]
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
        **options,
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
            request = request,
        )

    embeds = EmbedsReader(job_dir = job_dir)

    def make_prompt(request):
        if request.get("embeds") is None:
            return {"prompt_token_ids": list(request["prompt_token_ids"])}
        return mixed_prompt(
            token_ids = request["prompt_token_ids"],
            rows = embeds.get(file = request["embeds"]["file"], key = request["embeds"]["key"]),
            token_id = int(job["embeds_token_id"]),
        )

    by_index = {request["index"]: request for request in requests}
    extras = {}
    steps = int(job.get("logprob_steps", 1))

    def collect(index, output):
        request = by_index[index]
        extra = {}
        if request.get("logprobs") is not None or request.get("logprob_token_ids") is not None:
            extra["logprobs"] = logprob_rows(entries = output.outputs[0].logprobs, steps = steps)
        if request.get("prompt_logprobs") is not None:
            extra["prompt_logprobs"] = prompt_logprob_values(
                prompt_logprobs = output.prompt_logprobs,
                prompt_ids = request["prompt_token_ids"],
            )
        if extra:
            extras[index] = extra

    start = time.perf_counter()
    results = generate_all(
        engine = llm.llm_engine,
        requests = requests,
        make_params = make_params,
        max_inflight = max_inflight,
        log = log,
        log_every = job.get("log_every", 60.0),
        make_prompt = make_prompt,
        collect = collect,
    )
    seconds = round(number = time.perf_counter() - start, ndigits = 1)
    lines = [
        json.dumps(obj = {"index": index, "token_ids": ids, "finish_reason": reason, **extras.get(index, {})})
        for index, (ids, reason) in sorted(results.items())
    ]
    write_atomic(path = job_dir / "results.jsonl", text = "\n".join(lines) + "\n")
    tokens = sum(len(ids) for ids, _ in results.values())
    prompt_tokens = sum(len(request["prompt_token_ids"]) for request in requests)
    engine = {
        "version": vllm.__version__,
        "kv_cache_tokens": kv_tokens,
        "max_concurrency": round(number = float(concurrency), ndigits = 2),
        "max_inflight": max_inflight,
        "flashinfer_sampler": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") != "0",
        "enable_prompt_embeds": bool(job.get("enable_prompt_embeds")),
        "load_seconds": load_seconds,
        "seconds": seconds,
        "prompt_tokens": prompt_tokens,
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
