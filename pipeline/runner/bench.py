"""批次测量：对一个 TTS 或 ASR 模型扫一组批次，记录吞吐、峰值显存和 GPU 利用率（实验 v1，任务 T20）。

用法（在仓库根目录运行，GPU 命令经 gpu.lock）::

    # TTS：用朗读稿和音色库为该模型生成 N 个任务，对每个批次各调用一次封装
    python -m pipeline.runner.bench tts --model omnivoice --config pipeline/runner/config_v1.yaml \\
        --work-dir <工作目录> --out-dir <输出目录> --texts <条目> --scripts <朗读稿> --voices <voices.jsonl> \\
        --batch-sizes 8 16 32 --n 256 [--mode clone|builtin] [--processes 1]

    # ASR：用已有的合成音频（audio.jsonl：sample_id、audio_path、lang；不够 N 条时重复使用）
    python -m pipeline.runner.bench asr --model qwen3_asr --config ... --work-dir ... --out-dir ... \\
        --audio <audio.jsonl> --batch-sizes 16 32 64 --n 256

每个批次先跑一次只有 2 条的调用测出启动（加载模型）时间，再跑 N 条，吞吐按「N / (总用时 - 启动时间)」计算；启动时间有几秒的波动，批次之间也可以直接比较 wall_s。
峰值显存是调用期间 nvidia-smi 每 0.25 秒取样的最大 memory.used 减去调用前的基线；WSL 下显存超过物理容量不会报错，
而是溢出到内存并使速度骤降，所以峰值接近 32 GB 的批次即使跑完也不选。结果追加写入 ``<out-dir>/bench.jsonl``，
并打印 markdown 表。合成出的音频在每次测量后删除，只保留结果。
"""

import argparse
import copy
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

from .config import load_config
from .io_utils import read_jsonl, write_jsonl_atomic
from .plan import build_tasks, load_entries, load_voices
from .stages import GpuLock, RunnerError, run_wrapper


OVERFLOW_MIB = 31000


class GpuSampler:
    """后台线程：每 0.25 秒读一次 nvidia-smi 的显存与利用率。"""

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target = self._run, daemon = True)

    @staticmethod
    def read() -> tuple:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output = True,
            text = True,
        ).stdout.strip().splitlines()[0]
        memory, util = (float(part) for part in out.split(","))
        return memory, util

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(self.read())
            except Exception:
                pass
            self._stop.wait(timeout = self.interval)

    def __enter__(self):
        self.baseline = self.read()[0]
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()

    def summary(self) -> dict:
        memory = [item[0] for item in self.samples] or [self.baseline]
        util = [item[1] for item in self.samples] or [0.0]
        return {
            "baseline_mib": self.baseline,
            "peak_mib": max(memory),
            "peak_delta_mib": max(memory) - self.baseline,
            "util_mean": round(sum(util) / len(util), 1),
            "util_max": max(util),
        }


def model_tasks(cfg: dict, model: str, args: argparse.Namespace) -> list:
    """用该模型单独生成任务：把每种它支持的语言的权重都指向它。"""
    cfg = copy.deepcopy(cfg)
    spec = cfg["tts_models"][model]
    for lang in spec.get("langs", []):
        cfg.setdefault("langs", {}).setdefault(lang, {})["tts"] = {model: 1}
    cfg["prefer_builtin"] = args.mode == "builtin"
    for lang_cfg in cfg["langs"].values():
        lang_cfg.pop("prefer_builtin", None)
    entries = load_entries(
        paths = args.texts,
        langs = None,
        limit_per_lang = None,
        fallback_original = False,
        scripts = args.scripts,
    )
    tasks, _ = build_tasks(cfg = cfg, entries = entries, voices = load_voices(path = args.voices), split = "train")
    tasks = [task for task in tasks if task["tts_model"] == model and (args.mode is None or task["mode"] == args.mode)]
    # 各语言轮流取，保证 N 条覆盖该模型支持的全部语言
    by_lang = {}
    for task in tasks:
        by_lang.setdefault(task["lang"], []).append(task)
    picked = []
    while len(picked) < args.n and any(by_lang.values()):
        for lang in sorted(by_lang):
            if by_lang[lang] and len(picked) < args.n:
                picked.append(by_lang[lang].pop(0))
    return picked


def tts_jobs(tasks: list) -> list:
    jobs = []
    for task in tasks:
        job = {key: task[key] for key in ("sample_id", "entry_id", "lang", "text", "mode", "voice_id", "ref_lang", "seed")}
        for key in ("builtin_name", "ref_audio", "ref_text"):
            if task.get(key):
                job[key] = task[key]
        jobs.append(job)
    return jobs


def asr_jobs(path: Path, n: int, langs: list | None) -> list:
    rows = [row for row in read_jsonl(path = path) if not langs or row["lang"] in langs]
    if not rows:
        raise RunnerError("没有可用的音频")
    jobs = []
    index = 0
    while len(jobs) < n:
        row = rows[index % len(rows)]
        jobs.append({**row, "sample_id": f"{row['sample_id']}~{index}"})
        index += 1
    return jobs


def run_once(kind: str, name: str, backend: dict, work_dir: Path, jobs: list, batch: int, tag: str, base: Path, processes: int) -> dict:
    """跑一次封装调用（processes 大于 1 时把 jobs 平分给多个同时运行的进程，整体持一次 gpu.lock）。"""
    call_dir = base / "calls" / tag
    if call_dir.exists():
        shutil.rmtree(call_dir)
    parts = [jobs[i::processes] for i in range(processes)]
    procs = []
    for index, part in enumerate(parts):
        part_dir = call_dir / f"p{index}"
        part_dir.mkdir(parents = True)
        write_jsonl_atomic(path = part_dir / "jobs.jsonl", rows = part)
        procs.append(part_dir)

    def call(part_dir: Path, lock: bool) -> None:
        if kind == "tts":
            args = ["--input", part_dir / "jobs.jsonl", "--output-dir", part_dir / "out", "--batch-size", batch]
        else:
            args = ["--input", part_dir / "jobs.jsonl", "--output", part_dir / "results.jsonl", "--batch-size", batch]
        run_wrapper(
            backend_name = name,
            backend = backend,
            work_dir = work_dir,
            args = [*args, *backend.get("args", [])],
            log_path = base / "logs" / f"{name}.log",
            lock = lock,
        )

    record = {"model": name, "kind": kind, "batch_size": batch, "n": len(jobs), "processes": processes, "ok": False}
    errors = []
    gpu = backend.get("gpu", True)
    context = GpuLock(path = work_dir / "gpu.lock") if (gpu and processes > 1) else _Null()
    with context:
        with GpuSampler() as sampler:
            started = time.monotonic()
            threads = []

            def guarded(part_dir: Path) -> None:
                try:
                    call(part_dir = part_dir, lock = processes == 1)
                except Exception as error:
                    errors.append(str(error))

            for part_dir in procs:
                thread = threading.Thread(target = guarded, args = (part_dir,))
                thread.start()
                threads.append(thread)
            for thread in threads:
                thread.join()
            wall = time.monotonic() - started
    record.update(wall_s = round(wall, 2), **sampler.summary())
    if errors:
        record["error"] = errors[0]
        shutil.rmtree(call_dir, ignore_errors = True)
        return record
    ok = 0
    audio_s = 0.0
    worst = 0.0
    for part_dir in procs:
        path = part_dir / ("out/results.jsonl" if kind == "tts" else "results.jsonl")
        for row in read_jsonl(path = path):
            if kind == "tts":
                if row.get("status") == "ok":
                    ok += 1
                    audio_s += row.get("duration_s") or 0
                    chars = max(1, len("".join(row["text"].split())))
                    worst = max(worst, (row.get("duration_s") or 0) / chars)
            elif not row.get("error") and row.get("hypothesis"):
                ok += 1
    record.update(ok = True, n_ok = ok, audio_s = round(audio_s, 1), max_sec_per_char = round(worst, 3))
    shutil.rmtree(call_dir, ignore_errors = True)
    return record


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description = "批次测量")
    parser.add_argument("kind", choices = ["tts", "asr"])
    parser.add_argument("--model", required = True)
    parser.add_argument("--config", required = True, type = Path)
    parser.add_argument("--work-dir", required = True, type = Path)
    parser.add_argument("--out-dir", required = True, type = Path)
    parser.add_argument("--batch-sizes", nargs = "+", type = int, required = True)
    parser.add_argument("--n", type = int, default = 256)
    parser.add_argument("--processes", type = int, default = 1)
    parser.add_argument("--texts", nargs = "+", type = Path)
    parser.add_argument("--scripts", nargs = "+", type = Path)
    parser.add_argument("--voices", type = Path)
    parser.add_argument("--mode", choices = ["clone", "builtin"], default = None)
    parser.add_argument("--audio", type = Path)
    parser.add_argument("--langs", nargs = "+", default = None, help = "asr：只用这些语言的音频")
    parser.add_argument("--note", default = "")
    args = parser.parse_args()

    cfg = load_config(path = args.config, work_dir = args.work_dir)
    backend = cfg["tts_models" if args.kind == "tts" else "asr_models"][args.model]
    args.out_dir.mkdir(parents = True, exist_ok = True)
    if args.kind == "tts":
        jobs = tts_jobs(tasks = model_tasks(cfg = cfg, model = args.model, args = args))
    else:
        jobs = asr_jobs(path = args.audio, n = args.n, langs = args.langs)
    print(f"{args.model}: {len(jobs)} 条", flush = True)

    for batch in args.batch_sizes:
        # 每个批次先测一次启动时间（2 条的调用），再测 N 条
        probe = run_once(
            kind = args.kind, name = args.model, backend = backend, work_dir = args.work_dir,
            jobs = jobs[:2], batch = batch, tag = "startup", base = args.out_dir, processes = 1,
        )
        startup = probe["wall_s"] if probe["ok"] else 0.0
        print(f"启动时间 {startup} s", flush = True)
        record = run_once(
            kind = args.kind, name = args.model, backend = backend, work_dir = args.work_dir,
            jobs = jobs, batch = batch, tag = f"b{batch}", base = args.out_dir, processes = args.processes,
        )
        record["startup_s"] = startup
        record["note"] = args.note
        if record["ok"]:
            effective = max(record["wall_s"] - startup, 1e-6)
            record["samples_per_s"] = round(record["n_ok"] / effective, 3)
            record["audio_s_per_s"] = round(record["audio_s"] / effective, 2) if args.kind == "tts" else None
        record["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with (args.out_dir / "bench.jsonl").open(mode = "a", encoding = "utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii = False) + "\n")
        print(json.dumps(record, ensure_ascii = False), flush = True)
        if not record["ok"] or record["peak_mib"] >= OVERFLOW_MIB:
            break  # 失败，或显存接近 32 GB（WSL 下会溢出到内存），更大的批次不再测


if __name__ == "__main__":
    main()
