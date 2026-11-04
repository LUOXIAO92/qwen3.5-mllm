"""流水线的执行阶段：TTS 合成、长度检查、ASR 识别与比对。

每个阶段把任务按固定方式切块（块文件名由块内 sample_id 的哈希决定），块跑完才原子地写出块文件。
重新运行时已有块文件的块直接跳过，没有的块整块重做，所以中途终止后续跑的结果和一次跑完相同。
"""

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from pipeline.compare import compare

from .config import sec_per_char
from .validate import InputError, load_checked
from .io_utils import (
    chunk_name,
    load_chunk_dir,
    read_jsonl,
    split_chunks,
    write_jsonl_atomic,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class StopRun(Exception):
    """达到 --stop-after-calls 设定的封装调用次数（用于模拟中途终止）。"""


class RunnerError(Exception):
    """封装调用失败，需要人工查看日志。"""


class Budget:
    def __init__(self, calls: int | None):
        self.calls = calls

    def consume(self) -> None:
        if self.calls is None:
            return
        if self.calls <= 0:
            raise StopRun("已达到 stop-after-calls")
        self.calls -= 1


class Layout:
    """输出目录中的固定位置。"""

    def __init__(self, out_dir: Path):
        self.out = Path(out_dir)
        self.audio = self.out / "audio"
        self.state = self.out / "state"
        self.work = self.out / "work"
        self.logs = self.out / "logs"
        self.tasks = self.out / "tasks.jsonl"
        self.synth = self.out / "synth_results.jsonl"
        self.length = self.out / "length_check.jsonl"
        self.asr_results = self.out / "asr_results.jsonl"

    def asr_raw(self, role: str) -> Path:
        return self.out / f"asr_{role}.jsonl"

    def synth_dir(self, model: str) -> Path:
        return self.state / "synth" / model

    def asr_dir(self, role: str, model: str) -> Path:
        return self.state / f"asr_{role}" / model


def run_wrapper(
    backend_name: str,
    backend: dict,
    work_dir: Path,
    args: list,
    log_path: Path,
    progress = None,
) -> None:
    """在封装自己的环境里以子进程运行；需要 GPU 的封装经 gpu.lock 排队。"""
    work_dir = Path(work_dir)
    python = backend.get("python") or str(Path(backend["env"]) / "bin" / "python")
    python_path = Path(python)
    if not python_path.is_absolute():
        python_path = work_dir / python_path
    cmd = []
    if backend.get("gpu", True):
        cmd += ["flock", str(work_dir / "gpu.lock")]
    cmd += [str(python_path), "-m", backend["module"], *[str(item) for item in args]]
    env = dict(os.environ)
    extra = dict(backend.get("env_vars") or {})
    paths = [str(REPO_ROOT)]
    if extra.get("PYTHONPATH"):
        paths.append(extra.pop("PYTHONPATH"))
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env.update({key: str(value) for key, value in extra.items()})
    log_path.parent.mkdir(parents = True, exist_ok = True)
    with log_path.open(mode = "a", encoding = "utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        if progress is not None:
            with progress.heartbeat():
                proc = subprocess.run(cmd, cwd = REPO_ROOT, env = env, stdout = log, stderr = subprocess.STDOUT)
        else:
            proc = subprocess.run(cmd, cwd = REPO_ROOT, env = env, stdout = log, stderr = subprocess.STDOUT)
    if proc.returncode != 0:
        raise RunnerError(f"{backend_name} 退出码 {proc.returncode}，日志 {log_path}")


# ---------------------------------------------------------------- TTS


def tts_stage(
    cfg: dict,
    tasks: list,
    layout: Layout,
    work_dir: Path,
    budget: Budget,
    progress = None,
) -> None:
    by_model = {}
    for task in tasks:
        by_model.setdefault(task["tts_model"], []).append(task)
    for model in sorted(by_model):
        backend = cfg["tts_models"][model]
        chunk_size = backend.get("chunk_size", cfg.get("synth_chunk_size", 64))
        chunks = split_chunks(items = sorted(by_model[model], key = lambda row: row["sample_id"]), size = chunk_size)
        targets = [
            layout.synth_dir(model = model) / chunk_name(sample_ids = [row["sample_id"] for row in chunk])
            for chunk in chunks
        ]
        if progress is not None:
            progress.start(
                key = f"synth:{model}",
                stage = "synth",
                total = len(by_model[model]),
                done = sum(len(chunk) for chunk, target in zip(chunks, targets) if target.exists()),
                counters = _synth_counters(targets = targets),
            )
        for chunk, target in zip(chunks, targets):
            if target.exists():
                continue
            budget.consume()
            records = _synth_chunk(
                model = model,
                backend = backend,
                chunk = chunk,
                target = target,
                layout = layout,
                work_dir = work_dir,
                progress = progress,
            )
            if progress is not None:
                ok = [rec for rec in records if rec["status"] == "ok"]
                progress.advance(
                    key = f"synth:{model}",
                    n = len(records),
                    audio_s = sum(rec["duration_s"] or 0 for rec in ok),
                    counters = {"ok": len(ok), "failed": len(records) - len(ok)},
                )
    merge_synth(cfg = cfg, layout = layout)


def _synth_chunk(
    model: str,
    backend: dict,
    chunk: list,
    target: Path,
    layout: Layout,
    work_dir: Path,
    progress = None,
) -> list:
    job_dir = layout.work / f"synth-{model}-{target.stem}"
    if job_dir.exists():
        shutil.rmtree(job_dir)
    out_dir = job_dir / "out"
    out_dir.mkdir(parents = True)
    jobs = []
    for task in chunk:
        job = {
            key: task[key]
            for key in ("sample_id", "entry_id", "lang", "text", "mode", "voice_id", "ref_lang", "seed")
        }
        for key in ("builtin_name", "ref_audio", "ref_text"):
            if task.get(key):
                job[key] = task[key]
        jobs.append(job)
    write_jsonl_atomic(path = job_dir / "jobs.jsonl", rows = jobs)
    run_wrapper(
        backend_name = model,
        backend = backend,
        work_dir = work_dir,
        args = [
            "--input", job_dir / "jobs.jsonl",
            "--output-dir", out_dir,
            "--batch-size", backend.get("batch_size", 8),
            *backend.get("args", []),
        ],
        log_path = layout.logs / f"synth-{model}.log",
        progress = progress,
    )
    results = {row["sample_id"]: row for row in read_jsonl(path = out_dir / "results.jsonl")}
    if not results:
        raise RunnerError(f"{model} 没有写出 results.jsonl，日志 {layout.logs / f'synth-{model}.log'}")
    layout.audio.mkdir(parents = True, exist_ok = True)
    records = []
    for task in chunk:
        result = results.get(task["sample_id"])
        record = {
            "sample_id": task["sample_id"],
            "tts_model": model,
            "mode": task["mode"],
            "audio_path": None,
            "duration_s": None,
            "status": "failed",
            "error": "tts_no_result",
        }
        if result is not None:
            record["status"] = result.get("status", "failed")
            record["error"] = result.get("error")
            flac = out_dir / f"{task['sample_id']}.flac"
            if record["status"] == "ok":
                if flac.exists():
                    os.replace(src = flac, dst = layout.audio / flac.name)
                    record["audio_path"] = f"audio/{flac.name}"
                    record["duration_s"] = result.get("duration_s")
                else:
                    record["status"] = "failed"
                    record["error"] = "tts_audio_missing"
        records.append(record)
    write_jsonl_atomic(path = target, rows = records)
    shutil.rmtree(job_dir, ignore_errors = True)
    return records


def _synth_counters(targets: list) -> dict:
    rows = [row for target in targets if target.exists() for row in read_jsonl(path = target)]
    ok = sum(row["status"] == "ok" for row in rows)
    return {"ok": ok, "failed": len(rows) - ok}


def merge_synth(cfg: dict, layout: Layout) -> None:
    """把各块文件合并成 synth_results.jsonl，按 sample_id 排序。"""
    records = {}
    for model in cfg.get("tts_models", {}):
        records.update(load_chunk_dir(directory = layout.synth_dir(model = model)))
    write_jsonl_atomic(path = layout.synth, rows = [records[key] for key in sorted(records)])


# ---------------------------------------------------------------- 长度检查


def count_chars(text: str) -> int:
    return len(re.sub(pattern = r"\s+", repl = "", string = text))


def length_verdict(cfg: dict, task: dict, record: dict) -> str:
    """音频时长 / 文本长度超出该语言的合理范围时返回原因，否则返回空字符串。"""
    low, high = sec_per_char(cfg = cfg, lang = task["lang"])
    chars = max(1, count_chars(text = task["text"]))
    duration = record.get("duration_s")
    if not duration:
        return "length_zero_duration"
    ratio = duration / chars
    if ratio > high:
        return "length_too_long"
    if ratio < low:
        return "length_too_short"
    return ""


def length_stage(cfg: dict, tasks: list, layout: Layout, progress = None) -> None:
    """读取 synth_results.jsonl，给每个合成成功的样本判定时长比例，写出 length_check.jsonl。"""
    synth = {row["sample_id"]: row for row in load_checked(path = layout.synth, kind = "synth", hint = "先运行 tts 子命令")}
    ids = {task["sample_id"] for task in tasks}
    missing = ids - set(synth)
    if missing:
        raise InputError(f"synth_results.jsonl 缺少 {len(missing)} 个样本，先运行 tts 子命令完成合成")
    if progress is not None:
        progress.start(key = "length", stage = "length", total = len(tasks))
    rows = []
    for task in sorted(tasks, key = lambda row: row["sample_id"]):
        record = synth[task["sample_id"]]
        if record["status"] != "ok":
            reason = "tts_not_ok"
        else:
            reason = length_verdict(cfg = cfg, task = task, record = record)
        chars = max(1, count_chars(text = task["text"]))
        rows.append(
            {
                "sample_id": task["sample_id"],
                "lang": task["lang"],
                "duration_s": record["duration_s"],
                "chars": chars,
                "sec_per_char": round(record["duration_s"] / chars, 4) if record["duration_s"] else None,
                "ok": reason == "",
                "reason": reason,
            }
        )
    write_jsonl_atomic(path = layout.length, rows = rows)
    if progress is not None:
        progress.advance(
            key = "length",
            n = len(rows),
            counters = {
                "out_of_range": sum(row["reason"].startswith("length_") for row in rows),
                "tts_not_ok": sum(row["reason"] == "tts_not_ok" for row in rows),
            },
        )


# ---------------------------------------------------------------- ASR


def asr_assignments(cfg: dict, tasks: list, role: str) -> dict:
    """按 ASR 模型分组：role 为 primary 或 cross，返回 {asr 名: [task]}。"""
    key = "asr" if role == "primary" else "cross_asr"
    groups = {}
    for task in tasks:
        name = cfg.get("langs", {}).get(task["lang"], {}).get(key)
        if name:
            groups.setdefault(name, []).append(task)
    return groups


def merge_asr_raw(cfg: dict, layout: Layout, role: str) -> None:
    records = {}
    for name in cfg.get("asr_models", {}):
        records.update(load_chunk_dir(directory = layout.asr_dir(role = role, model = name)))
    write_jsonl_atomic(path = layout.asr_raw(role = role), rows = [records[key] for key in sorted(records)])


def asr_stage(
    cfg: dict,
    tasks: list,
    layout: Layout,
    work_dir: Path,
    budget: Budget,
    role: str,
    progress = None,
) -> None:
    """对通过长度检查的样本运行 ASR，写出原始识别结果 asr_<role>.jsonl（不比对）。

    交叉 ASR 只处理主 ASR 已经比对一致的样本，所以要先对主 ASR 的结果运行 compare 子命令。
    """
    synth = {row["sample_id"]: row for row in load_checked(path = layout.synth, kind = "synth", hint = "先运行 tts 子命令")}
    length = {row["sample_id"]: row for row in load_checked(path = layout.length, kind = "length", hint = "先运行 length 子命令")}
    primary_match = {}
    if role == "cross":
        results = load_checked(path = layout.asr_results, kind = "asr_results", hint = "先对主 ASR 运行 compare 子命令")
        primary_match = {row["sample_id"]: row["match"] for row in results if row["asr_role"] == "primary"}
    for name, group in sorted(asr_assignments(cfg = cfg, tasks = tasks, role = role).items()):
        backend = cfg["asr_models"][name]
        chunk_size = backend.get("chunk_size", cfg.get("asr_chunk_size", 256))
        chunks = split_chunks(items = sorted(group, key = lambda row: row["sample_id"]), size = chunk_size)
        targets = [
            layout.asr_dir(role = role, model = name) / chunk_name(sample_ids = [row["sample_id"] for row in chunk])
            for chunk in chunks
        ]
        resumed = sum(len(chunk) for chunk, target in zip(chunks, targets) if target.exists())
        key = f"asr_{role}:{name}"
        if progress is not None:
            progress.start(
                key = key,
                stage = f"asr_{role}",
                total = len(group),
                done = resumed,
                counters = _asr_counters(targets = targets),
            )
        for chunk, target in zip(chunks, targets):
            if target.exists():
                continue
            rows = []
            for task in chunk:
                sample_id = task["sample_id"]
                if sample_id not in length:
                    raise InputError(f"length_check.jsonl 缺少样本 {sample_id}，先运行 length 子命令")
                if not length[sample_id]["ok"]:
                    continue
                if role == "cross":
                    if sample_id not in primary_match:
                        raise InputError(f"主 ASR 的比对结果缺少样本 {sample_id}，先运行 asr --role primary 和 compare")
                    if not primary_match[sample_id]:
                        continue
                rows.append((task, synth[sample_id]))
            if rows:
                budget.consume()
            records = _asr_chunk(
                name = name,
                backend = backend,
                role = role,
                rows = rows,
                target = target,
                layout = layout,
                work_dir = work_dir,
                progress = progress,
            )
            if progress is not None:
                failed = sum(bool(rec["error"]) for rec in records)
                progress.advance(
                    key = key,
                    n = len(rows),
                    skipped = len(chunk) - len(rows),
                    audio_s = sum(record["duration_s"] or 0 for _, record in rows),
                    counters = {"asr_ok": len(records) - failed, "asr_failed": failed},
                )
    merge_asr_raw(cfg = cfg, layout = layout, role = role)


def _asr_counters(targets: list) -> dict:
    rows = [row for target in targets if target.exists() for row in read_jsonl(path = target)]
    failed = sum(bool(row["error"]) for row in rows)
    return {"asr_ok": len(rows) - failed, "asr_failed": failed}


def _asr_chunk(
    name: str,
    backend: dict,
    role: str,
    rows: list,
    target: Path,
    layout: Layout,
    work_dir: Path,
    progress = None,
) -> list:
    results = {}
    if rows:
        job_dir = layout.work / f"asr-{role}-{name}-{target.stem}"
        if job_dir.exists():
            shutil.rmtree(job_dir)
        job_dir.mkdir(parents = True)
        write_jsonl_atomic(
            path = job_dir / "audio.jsonl",
            rows = [
                {
                    "sample_id": task["sample_id"],
                    "audio_path": str((layout.out / record["audio_path"]).resolve()),
                    "lang": task["lang"],
                }
                for task, record in rows
            ],
        )
        run_wrapper(
            backend_name = name,
            backend = backend,
            work_dir = work_dir,
            args = [
                "--input", job_dir / "audio.jsonl",
                "--output", job_dir / "results.jsonl",
                "--batch-size", backend.get("batch_size", 16),
                *backend.get("args", []),
            ],
            log_path = layout.logs / f"asr-{name}.log",
            progress = progress,
        )
        results = {row["sample_id"]: row for row in read_jsonl(path = job_dir / "results.jsonl")}
        if not results:
            raise RunnerError(f"{name} 没有写出识别结果，日志 {layout.logs / f'asr-{name}.log'}")
        shutil.rmtree(job_dir, ignore_errors = True)
    records = []
    for task, _ in rows:
        result = results.get(task["sample_id"])
        error = None
        if result is None:
            error = "封装没有返回该样本"
        elif result.get("error"):
            error = str(result["error"])
        records.append(
            {
                "sample_id": task["sample_id"],
                "asr_model": (result or {}).get("asr_model") or name,
                "asr_role": role,
                "hypothesis": (result or {}).get("hypothesis", "") or "",
                "error": error,
            }
        )
    write_jsonl_atomic(path = target, rows = records)
    return records


# ---------------------------------------------------------------- 比对


def compare_stage(cfg: dict, tasks: list, layout: Layout, progress = None) -> None:
    """把 asr_primary.jsonl、asr_cross.jsonl 中尚未比对的样本与朗读稿比对，写入 asr_results.jsonl。"""
    by_id = {task["sample_id"]: task for task in tasks}
    done = {}
    if layout.asr_results.exists():
        for row in load_checked(path = layout.asr_results, kind = "asr_results"):
            done[(row["sample_id"], row["asr_role"])] = row
    todo = []
    total = 0
    for role in ("primary", "cross"):
        path = layout.asr_raw(role = role)
        if not path.exists():
            continue
        for row in load_checked(path = path, kind = "asr_raw", hint = f"先运行 asr --role {role}"):
            if row["sample_id"] in by_id:
                total += 1
                if (row["sample_id"], role) not in done:
                    todo.append(row)
    if progress is not None:
        progress.start(key = "compare", stage = "compare", total = total, done = total - len(todo))
    for start in range(0, len(todo), 500):
        began = time.monotonic()
        batch = todo[start:start + 500]
        for row in batch:
            done[(row["sample_id"], row["asr_role"])] = _compare_row(task = by_id[row["sample_id"]], row = row)
        if progress is not None:
            matched = sum(bool(done[(row["sample_id"], row["asr_role"])]["match"]) for row in batch)
            progress.advance(
                key = "compare",
                n = len(batch),
                busy_s = max(time.monotonic() - began, 1e-6),
                counters = {"match": matched, "mismatch": len(batch) - matched},
            )
    write_jsonl_atomic(path = layout.asr_results, rows = [done[key] for key in sorted(done)])


def _compare_row(task: dict, row: dict) -> dict:
    record = {
        "sample_id": row["sample_id"],
        "asr_model": row["asr_model"],
        "asr_role": row["asr_role"],
        "hypothesis": row["hypothesis"],
    }
    if row.get("error"):
        record.update(match = False, compare_detail = f"asr_error: {row['error']}")
    elif not str(row["hypothesis"]).strip():
        record.update(match = False, compare_detail = "asr_empty")
    else:
        try:
            outcome = compare(
                script = task["text"],
                hypothesis = row["hypothesis"],
                lang = task["lang"],
                original = task["original"],
            )
            record.update(match = bool(outcome.match), compare_detail = outcome.detail)
        except Exception as error:  # 比对本身出错时按失败记录，保留原因
            record.update(match = False, compare_detail = f"compare_error: {error}")
    return record
