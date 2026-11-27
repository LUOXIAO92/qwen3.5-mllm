"""本机评测的 vLLM 引擎（任务 #71）：宿主机一侧的共用部分。

用户 2026-11-27 决定：C 组各模型的本机评测一律用 vLLM 的 docker 镜像，把 GPU 用满、尽快出结果，不再用 transformers 的
generate。各评测程序（think_check、text_bench、run_eval、instruct_probe、replay_gap、lm_ppl）加 --engine vllm 时，提示、
打分和输出文件都与 transformers 版相同，只把模型的计算交给容器中的 pipeline/eval/vllm_generate.py；这里是它们共用的
部分：命令行参数、准备 vLLM 加载的语言模型目录、写任务目录、启动生成程序和读回结果。

模型、导出与 docker 的做法与 GPQA 的 vLLM 版相同（pipeline/eval/gpqa_vllm.py，函数直接沿用）：参考模型直接加载
--vllm-model（已导出的参考模型目录）；运行先导出到 --export-dir（融合模型中的语言模型张量换成 --language-model 中的
同名张量，配置与分词器取自 --config-dir），评测后删除，--keep-export 时保留（同一个模型的几项评测共用一份导出，最后一项
之后由驱动程序删除）；导出目录中的 export.json 记下来源，来源相同时直接用。生成程序在镜像中运行（WSL：/dev/dxg 与
/usr/lib/wsl，宿主机用户，--network none，--ipc=host，--init），仓库与模型目录只读挂载，任务目录与 --docker-home 可写；
收到 SIGTERM / SIGINT 或出错时 docker kill 容器，不留占着 GPU 的容器。

vLLM 设置：bfloat16，language_model_only，不用前缀缓存，generation_config = "vllm"，--gpu-memory-utilization（默认
0.9），--max-num-batched-tokens（默认 4096），--max-num-seqs（默认 256）。同时交给 vLLM 的请求数按 KV 缓存容量定
（vllm_generate.inflight_limit：每条用满 max_model_len 时放得下的条数乘 --inflight-factor，不超过 --max-num-seqs）。

任务目录（--job-dir）：job.json、requests.jsonl，音频评测另有宿主机算好的音频向量（safetensors）；结果为 results.jsonl 与
engine.json。完成后删除（--keep-job 保留；出错时保留，便于查看）。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import sys
import time
from pathlib import Path

from pipeline.eval import gpqa_vllm
from pipeline.eval.gpqa_vllm import EXPORT_MARKER, IMAGE, JOB_FILE, Stop
from pipeline.eval.text_bench import write_atomic

ENGINES = ("hf", "vllm")
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9


def add_arguments(parser, max_num_seqs: int = 256, inflight_factor: float = 1.25) -> None:
    """加上 --engine 与 vLLM 引擎的参数（--engine 默认 hf，即原来的 transformers 版）。"""
    group = parser.add_argument_group(title = "vLLM 引擎（--engine vllm，见 pipeline/eval/vllm_host.py）")
    group.add_argument("--engine", default = "hf", choices = ENGINES, help = "hf：transformers；vllm：vLLM 的 docker 镜像")
    group.add_argument("--vllm-model", default = None, type = Path, help = "直接加载的语言模型目录（参考模型：已导出的 ref-lm）")
    group.add_argument("--export-dir", default = None, type = Path, help = "没有 --vllm-model 时把语言模型导出到这里")
    group.add_argument("--config-dir", default = None, type = Path, help = "导出时复制配置与分词器文件的目录（已导出的参考模型目录）")
    group.add_argument("--keep-export", action = "store_true", help = "评测后保留导出目录（同一模型的后续评测共用）")
    group.add_argument("--job-dir", default = None, type = Path, help = "与生成程序交换数据的任务目录，默认 <输出目录>/vllm-job")
    group.add_argument("--keep-job", action = "store_true", help = "评测后保留任务目录")
    group.add_argument("--runner", default = "docker", choices = ("docker", "local"), help = "生成程序在 docker 镜像中运行，或用 --worker-python 直接运行")
    group.add_argument("--image", default = IMAGE, help = "vLLM 的 docker 镜像")
    group.add_argument("--docker-home", default = None, type = Path, help = "容器中的 HOME（vLLM、Triton 等的编译缓存），默认 <任务目录>/home")
    group.add_argument("--worker-python", default = sys.executable, help = "--runner local 时运行生成程序的 python（需要装有 vLLM）")
    group.add_argument("--gpu-memory-utilization", default = DEFAULT_GPU_MEMORY_UTILIZATION, type = float)
    group.add_argument("--max-num-seqs", default = max_num_seqs, type = int, help = "vLLM 调度器的序列数上限")
    group.add_argument("--max-num-batched-tokens", default = 4096, type = int)
    group.add_argument("--inflight-factor", default = inflight_factor, type = float, help = "同时交给 vLLM 的请求数为 KV 缓存放得下的满长条数的这么多倍")
    group.add_argument("--max-inflight", default = 0, type = int, help = "直接指定同时交给 vLLM 的请求数（大于 0 时代替 --inflight-factor）")
    group.add_argument("--log-every", default = 60.0, type = float, help = "生成程序每隔这么多秒打印一次进度")


def check_arguments(parser, args, default_job_dir: Path, language_model: Path | None) -> None:
    """--engine vllm 时核对模型来源（--vllm-model，或 --export-dir 加 --config-dir），补上任务目录与容器 HOME 的默认值。"""
    if args.engine != "vllm":
        return
    if args.vllm_model is not None and (language_model is not None or args.export_dir is not None):
        parser.error("--vllm-model 与 --language-model、--export-dir 只能选一边")
    if args.vllm_model is None and (args.export_dir is None or args.config_dir is None):
        parser.error("--engine vllm 且没有 --vllm-model 时需要 --export-dir 与 --config-dir")
    if args.job_dir is None:
        args.job_dir = Path(default_job_dir)
    if args.docker_home is None:
        args.docker_home = Path(args.job_dir) / "home"


def job_settings(args, max_model_len: int, **fields) -> dict:
    """写进 job.json 的通用设置；fields 为各评测自己的设置（max_tokens、temperature、stop_token_ids 等）。"""
    settings = {
        "temperature": 0.0,
        "seed": 0,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": int(max_model_len),
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "inflight_factor": args.inflight_factor,
        "max_inflight": args.max_inflight,
        "log_every": args.log_every,
    }
    settings.update(fields)
    return settings


def prepare_job_dir(job_dir: Path) -> Path:
    """建立空的任务目录；目录中已有本程序的旧任务（有 job.json）时先删掉，别人的非空目录不覆盖。"""
    job_dir = Path(job_dir)
    if (job_dir / JOB_FILE).exists():
        shutil.rmtree(path = job_dir)
    elif job_dir.exists() and any(job_dir.iterdir()):
        raise RuntimeError(f"{job_dir} 已存在、不是本程序的任务目录（没有 {JOB_FILE}），不覆盖")
    job_dir.mkdir(parents = True, exist_ok = True)
    # 先写一个占位的 job.json，出错中断时目录也能被识别为任务目录、下次删掉重来
    (job_dir / JOB_FILE).write_text(data = "{}\n", encoding = "utf-8")
    return job_dir


def write_requests(job_dir: Path, requests: list, job: dict) -> None:
    """写 requests.jsonl 与 job.json（job.json 最后写）。每条请求至少有 index、seed、prompt_token_ids。"""
    lines = [json.dumps(obj = request, ensure_ascii = False) for request in requests]
    write_atomic(path = Path(job_dir) / "requests.jsonl", text = "\n".join(lines) + "\n")
    write_atomic(path = Path(job_dir) / JOB_FILE, text = json.dumps(obj = job, indent = 1) + "\n")


def read_results(job_dir: Path, indices: list) -> tuple:
    """读生成程序的输出：({index: 结果行}, engine.json 的内容)。缺少任何一条时报错。"""
    engine_path = Path(job_dir) / "engine.json"
    if not engine_path.exists():
        raise RuntimeError(f"生成程序没有写出 {engine_path}")
    results = {}
    with open(file = Path(job_dir) / "results.jsonl", encoding = "utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(s = line)
                results[int(row["index"])] = row
    missing = [index for index in indices if index not in results]
    if missing:
        raise RuntimeError(f"有 {len(missing)} 条没有结果：{missing[:10]}")
    return results, json.loads(s = engine_path.read_text(encoding = "utf-8"))


def container_name(kind: str, label: str | None) -> str:
    """容器名：vllm-<评测>-<标签>-<进程号>，只留 docker 允许的字符。"""
    clean = re.sub(pattern = r"[^A-Za-z0-9_.-]", repl = "-", string = f"{kind}-{label or 'model'}")
    return f"vllm-{clean}-{os.getpid()}"


def worker_command(args, model_path: Path, kind: str, label: str | None) -> tuple:
    """(生成程序的命令, 容器名或 None, 镜像 id 或 None)。"""
    if args.runner == "local":
        return gpqa_vllm.local_command(args = args, job_dir = args.job_dir), None, None
    info = gpqa_vllm.image_info(image = args.image)
    name = container_name(kind = kind, label = label)
    command = gpqa_vllm.docker_command(
        args = args,
        model_path = model_path,
        job_dir = args.job_dir,
        home = args.docker_home,
        name = name,
        library_path = info["ld_library_path"],
    )
    return command, name, info["id"]


def model_args(args, model_dir: Path, language_model: Path | None):
    """gpqa_vllm.prepare_model 需要的参数（各评测程序的 --model-dir、--language-model 名称相同，这里显式传入）。"""
    from types import SimpleNamespace

    return SimpleNamespace(
        vllm_model = args.vllm_model,
        export_dir = args.export_dir,
        config_dir = args.config_dir,
        keep_export = args.keep_export,
        model_dir = model_dir,
        language_model = language_model,
    )


def run_job(
    args,
    model_dir: Path,
    language_model: Path | None,
    requests: list | None,
    settings: dict | None,
    kind: str,
    label: str | None,
    log,
    prepare = None,
) -> tuple:
    """导出（或直接用）语言模型，写任务目录，在容器中运行生成程序，读回结果。

    settings 为 job_settings 的结果。音频评测的请求要先在任务目录中算好音频向量才能确定：这时 requests、settings 给 None，
    prepare(job_dir = 任务目录) 写好音频向量并返回 (requests, settings)。返回 (results {index: 结果行}, record)，record 为
    写进各评测汇总的 vLLM 设置（vllm_record）。收到 SIGTERM / SIGINT 时抛出 gpqa_vllm.Stop（容器已结束）；导出目录在没有
    --keep-export 时删除，任务目录在成功且没有 --keep-job 时删除。
    """

    def on_signal(signum, frame):
        raise Stop(signum = signum)

    previous = {signum: signal.signal(signum, on_signal) for signum in (signal.SIGTERM, signal.SIGINT)}
    model_path = None
    exported = False
    start = time.perf_counter()
    try:
        model_path, exported = gpqa_vllm.prepare_model(
            args = model_args(args = args, model_dir = model_dir, language_model = language_model),
            log = log,
        )
        job_dir = prepare_job_dir(job_dir = args.job_dir)
        if prepare is not None:
            requests, settings = prepare(job_dir = job_dir)
        job = {"model": str(Path(model_path).resolve()), **settings}
        write_requests(job_dir = job_dir, requests = requests, job = job)
        Path(args.docker_home).mkdir(parents = True, exist_ok = True)
        command, name, image_id = worker_command(
            args = args,
            model_path = model_path,
            kind = kind,
            label = label,
        )
        log(f"启动生成程序（{args.runner}{'，容器 ' + name if name else ''}），{len(requests)} 条请求")
        code = gpqa_vllm.run_worker(command = command, name = name, log = log)
        if code != 0:
            raise RuntimeError(f"生成程序退出码 {code}，任务目录 {args.job_dir} 保留")
        results, engine = read_results(job_dir = job_dir, indices = [request["index"] for request in requests])
    except BaseException:
        # 出错时任务目录保留便于查看，但音频向量（FLEURS 约 9 GB）先删掉，不占系统盘
        for path in Path(args.job_dir).glob(pattern = "embeds-*.safetensors"):
            path.unlink()
        raise
    finally:
        if exported and not args.keep_export:
            gpqa_vllm.remove_marked(
                directory = model_path,
                marker = EXPORT_MARKER,
                log = log,
                what = "导出目录",
            )
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    if not args.keep_job:
        gpqa_vllm.remove_marked(
            directory = args.job_dir,
            marker = JOB_FILE,
            log = log,
            what = "任务目录",
        )
    record = vllm_record(
        args = args,
        job = job,
        engine = engine,
        image_id = image_id,
        model_path = model_path,
        seconds = time.perf_counter() - start,
    )
    return results, record


def vllm_record(args, job: dict, engine: dict, image_id: str | None, model_path: Path, seconds: float) -> dict:
    """写进汇总的 vLLM 设置与用时（seconds 为含导出、载入的总用时）。"""
    return {
        "runner": args.runner,
        "image": args.image if args.runner == "docker" else None,
        "image_id": image_id,
        "model": str(model_path),
        "dtype": "bfloat16",
        "gpu_memory_utilization": job["gpu_memory_utilization"],
        "max_model_len": job["max_model_len"],
        "max_num_seqs": job["max_num_seqs"],
        "max_num_batched_tokens": job["max_num_batched_tokens"],
        "enable_prefix_caching": False,
        "inflight_factor": None if job["max_inflight"] > 0 else job["inflight_factor"],
        "total_seconds": round(number = seconds, ndigits = 1),
        **engine,
    }


def stop_ids_of(tokenizer) -> list:
    """融合模型分词器的结束符编号（<|im_end|>、<|endoftext|>），与 think_check.STOP_IDS 核对。"""
    from pipeline.eval.think_check import STOP_IDS
    from pipeline.eval.vllm_generate import STOP_TOKENS

    found = [tokenizer.convert_tokens_to_ids(token) for token in STOP_TOKENS]
    if list(found) != list(STOP_IDS):
        raise RuntimeError(f"分词器的结束符 {found} 与 think_check.STOP_IDS {list(STOP_IDS)} 不同")
    return list(found)
