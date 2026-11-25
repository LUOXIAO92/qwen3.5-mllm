"""GPQA Diamond（思考模式）的 vLLM 版本（任务 #71）：提示、选项顺序、采样设置、答案解析、指标和输出文件都与
text_bench 的 gpqa 相同，只把生成换成 vLLM 的连续批处理。生成在 vLLM 的 docker 镜像中进行，其余都在宿主机上。

    flock <gpu.lock> python -m pipeline.eval.gpqa_vllm --model-dir <融合模型目录> --vllm-model <参考模型目录> \\
        --data-dir $DATASET/text --out-dir <输出目录>/ref --label ref --gpqa-max-new-tokens 32768 \\
        --job-dir <临时目录>/vllm-job/ref --docker-home <临时目录>/vllm-home
    flock <gpu.lock> python -m pipeline.eval.gpqa_vllm --model-dir <融合模型目录> \\
        --language-model <检查点>/language_model.safetensors --export-dir <临时目录>/vllm-lm/<模型标签> \\
        --config-dir <参考模型目录> --data-dir $DATASET/text --out-dir <输出目录>/<模型标签> --label <模型标签> \\
        --gpqa-max-new-tokens 32768 --job-dir <临时目录>/vllm-job/<模型标签> --docker-home <临时目录>/vllm-home

加 --dry-run 时只构造提示、列出长度、vLLM 设置和 docker 命令，不导出、不启动容器、不占用 GPU。本机驱动程序
bench_local.py 加 --gpqa-engine vllm 时用这个程序做 GPQA。

为什么换引擎：transformers 的 generate 左填充分批生成，一批里已经写完的样本仍占着位置，要等整批走到上限，每一步还要
复制 K、V。上限 32,768 时每个模型分 7 批、每批 29 条，约 3.5 小时。vLLM 一条写完就补进下一条，KV 缓存按块分配，
不复制。

分工：宿主机（envs/mllm-train）读题、构造提示、导出模型、解析答案和统计，都调用 text_bench 的函数；docker 镜像（用户
2026-11-25 决定用 vllm/vllm-openai:nightly-x86_64，vLLM 0.34.0rc1.dev1235，--image 可改）中只运行
pipeline/eval/vllm_generate.py：读任务目录（--job-dir）中的 token id 提示、每题的种子和采样设置，写回生成的 token id
与结束原因。任务目录在评测完成后删除（--keep-job 保留；出错时保留，便于查看）。

模型：vLLM 按原始架构 Qwen3_5ForConditionalGeneration 加载只含语言模型的目录（pipeline/replay/export_lm.py 的输出，
language_model_only）。参考模型直接用已经导出的目录（--vllm-model，即回放文本用的参考模型）。全量微调的运行先导出到
--export-dir：融合模型中的语言模型张量逐个换成检查点 language_model.safetensors 中的同名张量（键与形状必须完全一致，与
transformers 版换权重的要求相同），配置与分词器文件从 --config-dir（已导出的参考模型目录，分词器已按 #77 修正）复制。
导出目录约 3.8 GB，c0 实测 41 秒，评测结束后删除（--keep-export 保留）。目录中的 export.json 记下来源；中途被结束时
导出目录会留下，下次来源相同时直接用；不是本程序导出的目录不覆盖、不删除。

与 text_bench 相同的部分都直接调用它的函数：题目与选项顺序（load_gpqa，按 GPQA_OPTION_SEED 与 Record ID 固定打乱）、
用户消息（gpqa_text）、提示（用融合模型目录的分词器与 chat template 展开，enable_thinking = True，提示结尾是
`<|im_start|>assistant\\n<think>\\n`），以 token id 交给 vLLM，vLLM 不再套模板、不再分词；答案只在 </think> 之后找
（parse_choice）；逐题结果与汇总指标（write_gpqa，循环检测为 pipeline/replay/degeneration.py 的 degeneration_onset）。

采样：温度 --gpqa-temperature（默认 1），top_p 1、top_k 不限、min_p 0、没有任何惩罚，每题采样一次，最多
--gpqa-max-new-tokens 个新 token，在 <|im_end|> 或 <|endoftext|> 处停止，结束符留在结果中（与 transformers 版截到第一个
结束符、含结束符相同）。随机数：transformers 版每批开始前按 --gpqa-seed 加批序号固定全局随机数，一题用到的随机数取决于
它分在哪一批、批里还有哪些题；vLLM 版每题的种子为 --gpqa-seed 加题号（题目在 gpqa_diamond.csv 中的行号，从 0 起）。
vLLM 0.34 的采样是 Gumbel-max，噪声由（种子, 位置, 词表编号）的哈希算出，一题的随机数只取决于它的种子和位置，与同时
生成的其他题、被抢占后重算都无关。两版从同一个分布采样，随机数不同，同一个模型两版的回答不会逐 token 相同；vLLM 版
重跑时，同一题只在批的组成不同带来的 bfloat16 舍入上有出入。

vLLM 设置：bfloat16，language_model_only，不用前缀缓存，generation_config = "vllm"（不从模型目录读默认采样参数），
--gpu-memory-utilization 0.88，--max-num-batched-tokens 4096（与回放文本 pipeline/replay/generate.py 相同）。
max_model_len 为最长提示加 --gpqa-max-new-tokens：vLLM 在提示加生成达到 max_model_len 时也会停止，不能更小。回放文本
在 vLLM 0.28 上用到的两个绕法，在 0.34 中的情况：FlashInfer 采样只在设了 top-k 或 top-p 时使用，这里用不到，仍设
VLLM_USE_FLASHINFER_SAMPLER=0，省去它启动时的编译检查；只有一个 token 的提示读到别的请求留下的递推状态
（replay/first_token.py），0.34 的 gdn_attn 已经像 Mamba 一样传 treat_short_extends_as_decodes = False，GPQA 的提示也
都有七十多到两千多个 token，不受影响。

docker：WSL 上 --gpus all 不能用（NVML 报 GPU access blocked by the operating system），改为把 /dev/dxg 和 /usr/lib/wsl
传进容器，LD_LIBRARY_PATH 在镜像原有的值前面加上 /usr/lib/wsl/lib。仓库目录与模型目录只读挂载，任务目录与 --docker-home
（容器中的 HOME，存 vLLM、Triton、torch.compile 的缓存，跨模型共用可以加快载入）可写挂载，都挂在与宿主机相同的路径上；
以宿主机用户（uid:gid）运行，--network none，--ipc=host（vLLM 的进程间通信用共享内存），--init（容器中的 1 号进程转发
信号）。容器名为 gpqa-vllm-<标签>-<进程号>；本程序收到 SIGTERM / SIGINT 或出错时 docker kill 这个容器，不留占着 GPU
的容器。--runner local 时不用 docker，用 --worker-python 直接运行同一个生成程序（需要装有 vLLM 的 python）。

同时生成的题数：载入后读出 vLLM 算好的 KV 缓存容量（token 数）和「每条都用满 max_model_len 时能同时放下几条」
（记为 C），同时交给 vLLM 的请求最多 floor(C × --inflight-factor) 条（默认 1.25 倍；--max-inflight 直接指定），其余排队，
一条结束补进一条。vLLM 0.28 在 RTX 5090 上 0.88 时 KV 缓存约 177 万 token（回放文本实测），最长提示 2,779 token、上限
32,768 时 C 约 50，同时约 62 条；0.34 的容量以运行时报告的为准。在途样本的 KV 缓存超过容量时，vLLM 抢占最后放进来的
请求、以后从头重算；GPQA 大多数题写到上限，放进的题数远超 C 时会反复抢占重算、吞吐骤降（回放文本中 256 条 16k 的
样本时降到约 2,000 token/s），所以只多放四分之一，早写完的题空出的位置上补进来的新题还很短，用掉多出的那部分。vLLM
每 10 秒打印一次统计（吞吐、在途与排队的请求数、KV 缓存占用），有抢占时同一行带 Preemptions。用时主要花在每一步读
KV 缓存上（全注意力层 6 层、每 token 12 KiB），总量只取决于各题写了多长，与同时生成的题数关系不大：上限 32,768、
大多数题写到上限时约 600 万个生成 token，估计约半小时（另加载入，运行的模型另加导出）。

输出（--out-dir，与 text_bench 相同）：gpqa.json（汇总，另有 engine = "vllm"、vllm 的设置与 KV 缓存容量、
finish_reasons）、gpqa.items.jsonl（逐题）、gpqa.ids.i32（生成的 token id，只存本机）。gpqa.json 已存在时跳过（--force 重做）。
日志不打印题目和模型的回答。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from pipeline.eval import text_bench
from pipeline.eval.vllm_generate import STOP_TOKENS, inflight_limit

REPO = Path(__file__).resolve().parents[2]
IMAGE = "vllm/vllm-openai:nightly-x86_64"
EXPORT_MARKER = "export.json"
JOB_FILE = "job.json"
# vLLM 0.28、RTX 5090、--gpu-memory-utilization 0.88、max_num_batched_tokens 4096 时报告的 KV 缓存容量（#70 回放文本实测），
# 只用于 --dry-run 的估计；正式运行用载入后 vLLM 报告的值。
KV_TOKENS_ESTIMATE = 1_773_597
# 容器中访问 GPU（WSL）：/dev/dxg 设备与 /usr/lib/wsl 下的驱动库
WSL_DEVICE = "/dev/dxg"
WSL_LIB = "/usr/lib/wsl"


class Stop(Exception):
    def __init__(self, signum):
        super().__init__(f"收到信号 {signum}")
        self.signum = signum


# ---------------------------------------------------------------- 导出


def export_source(args) -> dict:
    """导出目录的来源：融合模型目录、检查点（大小与修改时间）和配置目录。来源相同的导出可以直接用。"""
    source = {
        "model_dir": str(Path(args.model_dir).resolve()),
        "language_model": None,
        "config_dir": str(Path(args.config_dir).resolve()),
    }
    if args.language_model is not None:
        stat = Path(args.language_model).stat()
        source["language_model"] = str(Path(args.language_model).resolve())
        source["language_model_bytes"] = stat.st_size
        source["language_model_mtime"] = int(stat.st_mtime)
    return source


def write_marker(export_dir: Path, source: dict, complete: bool, extra: dict | None = None) -> None:
    data = {"source": source, "complete": complete, "at": time.strftime("%Y-%m-%d %H:%M:%S"), **(extra or {})}
    text_bench.write_atomic(path = export_dir / EXPORT_MARKER, text = json.dumps(obj = data, ensure_ascii = False, indent = 1) + "\n")


def prepare_model(args, log) -> tuple:
    """vLLM 要加载的目录：(目录, 是否是本程序导出的)。给了 --vllm-model 时直接用它；否则导出到 --export-dir。"""
    if args.vllm_model is not None:
        return Path(args.vllm_model), False
    from pipeline.replay.export_lm import export

    export_dir = Path(args.export_dir)
    source = export_source(args = args)
    marker = export_dir / EXPORT_MARKER
    if marker.exists():
        info = json.loads(s = marker.read_text(encoding = "utf-8"))
        if info.get("complete") and info.get("source") == source:
            log(f"沿用已导出的 {export_dir}")
            return export_dir, True
        log(f"{export_dir} 是来源不同或没有导出完的目录，删除后重新导出")
        shutil.rmtree(path = export_dir)
    elif export_dir.exists() and any(export_dir.iterdir()):
        raise RuntimeError(f"{export_dir} 已存在、不是本程序导出的目录（没有 {EXPORT_MARKER}），不覆盖")
    export_dir.mkdir(parents = True, exist_ok = True)
    write_marker(export_dir = export_dir, source = source, complete = False)
    start = time.perf_counter()
    counts = export(
        mllm_dir = args.model_dir,
        original_dir = args.config_dir,
        out_dir = export_dir,
        language_model = args.language_model,
    )
    seconds = round(number = time.perf_counter() - start, ndigits = 1)
    write_marker(
        export_dir = export_dir,
        source = source,
        complete = True,
        extra = {**counts, "seconds": seconds},
    )
    log(f"导出 {export_dir}：{counts['tensors']} 个张量，换成检查点的 {counts['replaced']} 个，{seconds} 秒")
    return export_dir, True


def remove_marked(directory: Path, marker: str, log, what: str) -> None:
    """删除本程序建立的目录（有标志文件 marker 才删）。"""
    if (Path(directory) / marker).exists():
        shutil.rmtree(path = directory)
        log(f"删除{what} {directory}")


# ---------------------------------------------------------------- 任务目录


def request_seed(seed: int, index: int) -> int:
    """第 index 题（gpqa_diamond.csv 中的行号，从 0 起）的随机数种子。"""
    return seed + index


def job_settings(args, model_path: Path, prompts: list) -> dict:
    """写进 job.json 的生成设置（vllm_generate 读取）。"""
    from pipeline.eval.think_check import STOP_IDS

    return {
        "model": str(Path(model_path).resolve()),
        "max_tokens": args.gpqa_max_new_tokens,
        "temperature": args.gpqa_temperature,
        "stop_token_ids": list(STOP_IDS),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": max(len(ids) for _, ids in prompts) + args.gpqa_max_new_tokens,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "inflight_factor": args.inflight_factor,
        "max_inflight": args.max_inflight,
        "seed": args.gpqa_seed,
        "log_every": args.log_every,
    }


def write_job(job_dir: Path, job: dict, prompts: list, seed: int) -> None:
    """建立任务目录，写 job.json 与 requests.jsonl（题号、种子、提示 token id）。目录已有本程序的旧任务时先删掉。"""
    job_dir = Path(job_dir)
    if (job_dir / JOB_FILE).exists():
        shutil.rmtree(path = job_dir)
    elif job_dir.exists() and any(job_dir.iterdir()):
        raise RuntimeError(f"{job_dir} 已存在、不是本程序的任务目录（没有 {JOB_FILE}），不覆盖")
    job_dir.mkdir(parents = True, exist_ok = True)
    lines = [
        json.dumps(obj = {"index": index, "seed": request_seed(seed = seed, index = index), "prompt_token_ids": list(ids)})
        for index, (_, ids) in enumerate(prompts)
    ]
    text_bench.write_atomic(path = job_dir / "requests.jsonl", text = "\n".join(lines) + "\n")
    text_bench.write_atomic(path = job_dir / JOB_FILE, text = json.dumps(obj = job, indent = 1) + "\n")


def read_results(job_dir: Path, count: int) -> tuple:
    """读生成程序的输出：(按题号排列的 [(token id, finish_reason)], engine.json 的内容)。"""
    engine_path = Path(job_dir) / "engine.json"
    if not engine_path.exists():
        raise RuntimeError(f"生成程序没有写出 {engine_path}")
    results = {}
    with open(file = Path(job_dir) / "results.jsonl", encoding = "utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(s = line)
                results[int(row["index"])] = ([int(token) for token in row["token_ids"]], row["finish_reason"])
    missing = [index for index in range(count) if index not in results]
    if missing:
        raise RuntimeError(f"有 {len(missing)} 题没有结果：{missing[:10]}")
    return [results[index] for index in range(count)], json.loads(s = engine_path.read_text(encoding = "utf-8"))


# ---------------------------------------------------------------- 生成程序的启动


def image_info(image: str) -> dict:
    """镜像的 id 与它原有的 LD_LIBRARY_PATH（docker image inspect，不启动容器）。"""
    result = subprocess.run(
        args = ["docker", "image", "inspect", "--format", "{{.Id}}\n{{range .Config.Env}}{{println .}}{{end}}", image],
        capture_output = True,
        text = True,
        check = False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"docker image inspect {image} 失败：{result.stderr.strip()[:300]}")
    lines = result.stdout.splitlines()
    library = next((line.split("=", 1)[1] for line in lines[1:] if line.startswith("LD_LIBRARY_PATH=")), "")
    return {"id": lines[0].strip() if lines else None, "ld_library_path": library}


def container_name(label: str | None) -> str:
    """容器名：gpqa-vllm-<标签>-<进程号>，只留 docker 允许的字符。"""
    clean = re.sub(pattern = r"[^A-Za-z0-9_.-]", repl = "-", string = label or "model")
    return f"gpqa-vllm-{clean}-{os.getpid()}"


def docker_command(args, model_path: Path, job_dir: Path, home: Path, name: str, library_path: str) -> list:
    """在镜像中运行 vllm_generate 的 docker run 命令（见开头「docker」）。"""
    library = f"{WSL_LIB}/lib" + (f":{library_path}" if library_path else "")
    command = [
        "docker", "run", "--rm", "--init",
        "--name", name,
        "--device", WSL_DEVICE,
        "-v", f"{WSL_LIB}:{WSL_LIB}:ro",
        "--ipc=host",
        "--network", "none",
        "-u", f"{os.getuid()}:{os.getgid()}",
    ]
    environment = {
        "LD_LIBRARY_PATH": library,
        "HOME": str(home),
        "USER": "vllm",
        "LOGNAME": "vllm",
        "PYTHONPATH": str(REPO),
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1",
        "VLLM_NO_USAGE_STATS": "1",
        "DO_NOT_TRACK": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    }
    for key, value in environment.items():
        command += ["-e", f"{key}={value}"]
    for path, mode in ((REPO, "ro"), (Path(model_path).resolve(), "ro"), (Path(job_dir).resolve(), "rw"), (Path(home).resolve(), "rw")):
        command += ["-v", f"{path}:{path}:{mode}"]
    command += [
        "-w", str(REPO),
        "--entrypoint", "python3",
        args.image,
        "-m", "pipeline.eval.vllm_generate",
        "--job-dir", str(Path(job_dir).resolve()),
    ]
    return command


def local_command(args, job_dir: Path) -> list:
    return [str(args.worker_python), "-m", "pipeline.eval.vllm_generate", "--job-dir", str(Path(job_dir).resolve())]


def run_worker(command: list, name: str | None, log) -> int:
    """运行生成程序（输出接在本程序的输出上），返回退出码。中途出错或收到信号时结束它；docker 时另外 docker kill 容器。"""
    env = dict(
        os.environ,
        VLLM_USE_FLASHINFER_SAMPLER = "0",
        PYTHONPATH = str(REPO),
        PYTHONUNBUFFERED = "1",
    )
    process = subprocess.Popen(
        args = command,
        cwd = REPO,
        stdin = subprocess.DEVNULL,
        env = env,
    )
    try:
        return process.wait()
    finally:
        if process.poll() is None:
            if name is not None:
                log(f"结束容器 {name}")
                subprocess.run(args = ["docker", "kill", name], capture_output = True, check = False)
            process.terminate()
            try:
                process.wait(timeout = 30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


# ---------------------------------------------------------------- 入口


def plan_text(prompts: list, args, command: list | None) -> str:
    """--dry-run 的说明：提示长度、max_model_len、按 KV_TOKENS_ESTIMATE 估计的同时生成题数与生成程序的命令。"""
    lengths = sorted(len(ids) for _, ids in prompts)
    max_model_len = lengths[-1] + args.gpqa_max_new_tokens
    concurrency = KV_TOKENS_ESTIMATE / max_model_len
    limit = inflight_limit(
        concurrency = concurrency,
        factor = args.inflight_factor,
        explicit = args.max_inflight,
        max_num_seqs = args.max_num_seqs,
        count = len(prompts),
    )
    if args.vllm_model is not None:
        model = f"直接加载 {args.vllm_model}"
    else:
        model = f"导出到 {args.export_dir}（{'换上 ' + str(args.language_model) if args.language_model else '融合模型中的语言模型'}，配置取自 {args.config_dir}），"
        model += "评测后保留" if args.keep_export else "评测后删除"
    return "\n".join([
        f"{len(prompts)} 题，提示长度最短 {lengths[0]}、中位数 {lengths[len(lengths) // 2]}、最长 {lengths[-1]} 个 token",
        f"max_model_len {max_model_len}，按 KV 缓存约 {KV_TOKENS_ESTIMATE:,} token 估计同时放得下 {concurrency:.1f} 条，同时生成最多 {limit} 题",
        f"采样：温度 {args.gpqa_temperature}，每题种子 {args.gpqa_seed} + 题号，最多 {args.gpqa_max_new_tokens} 个新 token",
        f"模型：{model}",
        f"任务目录：{args.job_dir}（{'评测后保留' if args.keep_job else '评测后删除'}）",
        "生成程序：" + (" ".join(command) if command else "（docker 镜像信息取不到）"),
    ])


def parse_args(argv: list | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required = True, type = Path, help = "融合模型目录：分词器与 chat template 用于构造提示和解码回答，导出时取语言模型张量")
    parser.add_argument("--language-model", default = None, type = Path, help = "全量微调检查点中的 language_model.safetensors；不给时是参考模型")
    parser.add_argument("--vllm-model", default = None, type = Path, help = "直接加载的语言模型目录（export_lm 的输出，如参考模型）")
    parser.add_argument("--export-dir", default = None, type = Path, help = "没有 --vllm-model 时导出到这里，评测后删除")
    parser.add_argument("--config-dir", default = None, type = Path, help = "导出时复制配置与分词器文件的目录（已导出的参考模型目录）")
    parser.add_argument("--keep-export", action = "store_true", help = "评测后保留导出目录")
    parser.add_argument("--data-dir", required = True, type = Path, help = "文本数据集目录（$DATASET/text）")
    parser.add_argument("--out-dir", required = True, type = Path, help = "这个模型的输出目录")
    parser.add_argument("--label", default = None, help = "模型标签（用于日志与容器名）")
    parser.add_argument("--job-dir", default = None, type = Path, help = "与生成程序交换提示和结果的任务目录，默认 <输出目录>/vllm-job")
    parser.add_argument("--keep-job", action = "store_true", help = "评测后保留任务目录")
    parser.add_argument("--runner", default = "docker", choices = ("docker", "local"), help = "生成程序在 docker 镜像中运行，或用 --worker-python 直接运行")
    parser.add_argument("--image", default = IMAGE, help = "vLLM 的 docker 镜像")
    parser.add_argument("--docker-home", default = None, type = Path, help = "容器中的 HOME（存 vLLM、Triton 等的缓存），默认 <任务目录>/home")
    parser.add_argument("--worker-python", default = sys.executable, help = "--runner local 时运行生成程序的 python（需要装有 vLLM）")
    parser.add_argument("--gpqa-limit", default = None, type = int, help = "只做前这么多题")
    parser.add_argument("--gpqa-max-new-tokens", default = 16384, type = int)
    parser.add_argument("--gpqa-temperature", default = 1.0, type = float)
    parser.add_argument("--gpqa-seed", default = 1234, type = int, help = "每题的种子为它加题号")
    parser.add_argument("--gpu-memory-utilization", default = 0.88, type = float)
    parser.add_argument("--max-num-seqs", default = 256, type = int, help = "vLLM 调度器的序列数上限（实际同时生成的题数见 --inflight-factor）")
    parser.add_argument("--max-num-batched-tokens", default = 4096, type = int)
    parser.add_argument("--inflight-factor", default = 1.25, type = float, help = "同时生成的题数为「每条用满 max_model_len 时 KV 缓存放得下的条数」的这么多倍")
    parser.add_argument("--max-inflight", default = 0, type = int, help = "直接指定同时生成的题数（大于 0 时代替 --inflight-factor）")
    parser.add_argument("--log-every", default = 60.0, type = float, help = "每隔这么多秒打印一次进度")
    parser.add_argument("--force", action = "store_true", help = "已有结果也重做")
    parser.add_argument("--dry-run", action = "store_true", help = "只构造提示、列出设置与命令，不导出、不启动生成")
    args = parser.parse_args(args = argv)
    if args.vllm_model is not None and (args.language_model is not None or args.export_dir is not None):
        parser.error("--vllm-model 与 --language-model、--export-dir 只能选一边")
    if args.vllm_model is None and (args.export_dir is None or args.config_dir is None):
        parser.error("没有 --vllm-model 时需要 --export-dir 与 --config-dir")
    if args.job_dir is None:
        args.job_dir = args.out_dir / "vllm-job"
    if args.docker_home is None:
        args.docker_home = args.job_dir / "home"
    return args


def worker_command(args, model_path: Path) -> tuple:
    """(生成程序的命令, 容器名或 None, 镜像 id 或 None)。"""
    if args.runner == "local":
        return local_command(args = args, job_dir = args.job_dir), None, None
    info = image_info(image = args.image)
    name = container_name(label = args.label)
    command = docker_command(
        args = args,
        model_path = model_path,
        job_dir = args.job_dir,
        home = args.docker_home,
        name = name,
        library_path = info["ld_library_path"],
    )
    return command, name, info["id"]


def main(argv: list | None = None) -> int:
    args = parse_args(argv = argv)

    def log(message):
        print(f"{time.strftime('%H:%M:%S')} {message}", flush = True)

    marker = text_bench.marker(bench = "gpqa", out_dir = args.out_dir)
    if marker.exists() and not args.force and not args.dry_run:
        log(f"已有结果 {marker}，跳过")
        return 0
    data = args.data_dir / text_bench.DATA["gpqa"]
    if not data.exists():
        log(f"缺少数据：{data}")
        return 2
    rows = text_bench.gpqa_rows(data_dir = args.data_dir, limit = args.gpqa_limit)
    tokenizer = text_bench.tokenizer_of(model_dir = args.model_dir)
    prompts = text_bench.gpqa_prompts(tokenizer = tokenizer, rows = rows)
    stop_ids = [tokenizer.convert_tokens_to_ids(token) for token in STOP_TOKENS]
    log(f"模型 {args.label or args.language_model or args.vllm_model}；GPQA {len(rows)} 题，vLLM（{args.runner}）")
    if args.dry_run:
        model_path = args.vllm_model if args.vllm_model is not None else args.export_dir
        try:
            command, _, _ = worker_command(args = args, model_path = model_path)
        except RuntimeError as error:
            log(str(error))
            command = None
        print(plan_text(prompts = prompts, args = args, command = command), flush = True)
        return 0

    def on_signal(signum, frame):
        raise Stop(signum = signum)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    args.out_dir.mkdir(parents = True, exist_ok = True)
    model_path = None
    exported = False
    try:
        model_path, exported = prepare_model(args = args, log = log)
        job = job_settings(args = args, model_path = model_path, prompts = prompts)
        if job["stop_token_ids"] != stop_ids:
            raise RuntimeError(f"融合模型分词器的结束符 {stop_ids} 与 think_check.STOP_IDS {job['stop_token_ids']} 不同")
        write_job(
            job_dir = args.job_dir,
            job = job,
            prompts = prompts,
            seed = args.gpqa_seed,
        )
        Path(args.docker_home).mkdir(parents = True, exist_ok = True)
        command, name, image_id = worker_command(args = args, model_path = model_path)
        log(f"启动生成程序（{args.runner}{'，容器 ' + name if name else ''}）")
        code = run_worker(command = command, name = name, log = log)
        if code != 0:
            raise RuntimeError(f"生成程序退出码 {code}，任务目录 {args.job_dir} 保留")
        results, engine = read_results(job_dir = args.job_dir, count = len(prompts))
    except Stop as stop:
        log(f"停下：{stop}")
        return 128 + stop.signum
    finally:
        if exported and not args.keep_export:
            remove_marked(
                directory = model_path,
                marker = EXPORT_MARKER,
                log = log,
                what = "导出目录",
            )
    reasons = {}
    for _, reason in results:
        reasons[str(reason)] = reasons.get(str(reason), 0) + 1
    vllm = {
        "runner": args.runner,
        "image": args.image if args.runner == "docker" else None,
        "image_id": image_id,
        "model": str(model_path),
        "dtype": "bfloat16",
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": job["max_model_len"],
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_prefix_caching": False,
        "inflight_factor": None if args.max_inflight > 0 else args.inflight_factor,
        **engine,
    }
    settings = {
        "temperature": args.gpqa_temperature,
        "seed": args.gpqa_seed,
        "seed_scheme": "per question: seed + row index in gpqa_diamond.csv",
        "option_seed": text_bench.GPQA_OPTION_SEED,
        "engine": "vllm",
        "language_model": None if args.language_model is None else str(args.language_model),
        "vllm": vllm,
        "finish_reasons": dict(sorted(reasons.items())),
        "seconds": engine.get("seconds"),
    }
    summary = text_bench.write_gpqa(
        rows = rows,
        prompts = prompts,
        outputs = [ids for ids, _ in results],
        tokenizer = tokenizer,
        max_new_tokens = args.gpqa_max_new_tokens,
        out_dir = args.out_dir,
        settings = settings,
    )
    if not args.keep_job:
        remove_marked(
            directory = args.job_dir,
            marker = JOB_FILE,
            log = log,
            what = "任务目录",
        )
    log(text_bench.gpqa_log_line(summary = summary, engine = "vllm") + f"，{engine.get('tokens_per_second')} token/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
