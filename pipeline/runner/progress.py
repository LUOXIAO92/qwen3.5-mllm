"""进度报告：每个阶段按固定间隔和每个块完成时写一行日志，并持续更新输出目录下的 progress.json。

日志行写到标准错误和 ``<out-dir>/logs/progress.log``，格式例如::

    [progress] 2026-11-04T10:00:30 stage=synth key=omnivoice done=48/200 rate=0.82 samples/s audio=6.1 audio_s/wall_s eta=0:03:05

阶段（stage）有 synth（TTS 合成）、length（音频时长与文本长度的比例检查）、asr_primary、asr_cross（ASR 识别）
和 compare（朗读稿与识别结果比对）。rate 是本次进程内的完成速度（续跑时已完成的块不计入速度），
audio 是合成（或识别）的音频秒数除以墙钟秒数，eta 是按当前速度估计的剩余时间。
progress.json 记录每个阶段的完成数、速度、预计剩余时间和计数器，以及核对中的通过率。
"""

import json
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

from .io_utils import write_json_atomic


class Progress:
    def __init__(self, out_dir: Path, interval: float = 30.0, quiet: bool = False):
        self.out_dir = Path(out_dir)
        self.interval = interval
        self.quiet = quiet
        self.stages = {}
        self.final = None
        self._last_emit = 0.0
        self._lock = threading.RLock()
        self._started = time.monotonic()
        self.frozen = {}
        previous = self.out_dir / "progress.json"
        if previous.exists():
            try:
                old = json.loads(previous.read_text(encoding = "utf-8"))
                self.frozen = old.get("stages", {})
                self.final = old.get("final")
            except ValueError:
                self.frozen = {}

    def start(self, key: str, stage: str, total: int, done: int = 0, counters: dict | None = None) -> None:
        with self._lock:
            self.stages[key] = {
                "stage": stage,
                "total": total,
                "done": done,
                "resumed": done,
                "audio_s": 0.0,
                "busy_s": 0.0,
                "t0": time.monotonic(),
                "counters": dict(counters or {}),
            }
            self.emit(force = True)

    def advance(
        self,
        key: str,
        n: int,
        skipped: int = 0,
        audio_s: float = 0.0,
        busy_s: float | None = None,
        counters: dict | None = None,
    ) -> None:
        with self._lock:
            state = self.stages[key]
            state["done"] += n + skipped
            state["audio_s"] += audio_s
            state["processed"] = state.get("processed", 0) + n
            if busy_s is not None:
                state["busy_s"] += busy_s
            for name, value in (counters or {}).items():
                state["counters"][name] = state["counters"].get(name, 0) + value
            self.emit(force = True, key = key)

    def _numbers(self, state: dict) -> dict:
        elapsed = max(time.monotonic() - state["t0"], 1e-6)
        processed = state.get("processed", 0)
        wall = state["busy_s"] if state["busy_s"] > 0 else elapsed
        rate = processed / wall if processed else 0.0
        remaining = max(state["total"] - state["done"], 0)
        eta = remaining / rate if rate > 0 else None
        return {
            "stage": state["stage"],
            "done": state["done"],
            "total": state["total"],
            "resumed": state["resumed"],
            "samples_per_s": round(rate, 3),
            "audio_s_per_wall_s": round(state["audio_s"] / wall, 3) if state["audio_s"] else 0.0,
            "elapsed_s": round(elapsed, 1),
            "eta_s": round(eta, 1) if eta is not None else None,
            "counters": state["counters"],
        }

    def emit(self, force: bool = False, key: str | None = None) -> None:
        """写日志行与 progress.json。force 为假时按间隔限流（给心跳线程用）。"""
        with self._lock:
            now = time.monotonic()
            if not force and now - self._last_emit < self.interval:
                return
            self._last_emit = now
            numbers = {name: self._numbers(state = state) for name, state in self.stages.items()}
            merged = {**self.frozen, **numbers}
            stamp = datetime.now().isoformat(timespec = "seconds")
            lines = []
            for name, info in numbers.items():
                if key is not None and name != key:
                    continue
                if key is None and info["done"] >= info["total"]:
                    continue
                eta = str(timedelta(seconds = int(info["eta_s"]))) if info["eta_s"] is not None else "?"
                lines.append(
                    f"[progress] {stamp} stage={info['stage']} key={name} done={info['done']}/{info['total']} "
                    f"rate={info['samples_per_s']:.2f} samples/s audio={info['audio_s_per_wall_s']:.2f} audio_s/wall_s "
                    f"eta={eta}"
                )
            if lines:
                log = self.out_dir / "logs" / "progress.log"
                log.parent.mkdir(parents = True, exist_ok = True)
                with log.open(mode = "a", encoding = "utf-8") as handle:
                    handle.write("\n".join(lines) + "\n")
                if not self.quiet:
                    print("\n".join(lines), file = sys.stderr, flush = True)
            remaining = [info["eta_s"] for info in numbers.values() if info["done"] < info["total"]]
            write_json_atomic(
                path = self.out_dir / "progress.json",
                obj = {
                    "updated_at": stamp,
                    "elapsed_s": round(now - self._started, 1),
                    "overall": self._overall(stages = merged),
                    "stages": merged,
                    "eta_s_max_stage": max((eta for eta in remaining if eta is not None), default = None),
                    "final": self.final,
                },
            )

    def _overall(self, stages: dict) -> dict:
        """整体通过率：各阶段计数器的合计，交叉 ASR 只统计主 ASR 已一致的样本。"""
        sums = {}
        for state in stages.values():
            bucket = sums.setdefault(state["stage"], {})
            for name, value in state["counters"].items():
                bucket[name] = bucket.get(name, 0) + value

        def rate(stage: str):
            bucket = sums.get(stage, {})
            total = bucket.get("match", 0) + bucket.get("mismatch", 0)
            return round(bucket.get("match", 0) / total, 4) if total else None

        synth = sums.get("synth", {})
        compare = sums.get("compare", {})
        return {
            "synth_ok": synth.get("ok", 0),
            "synth_failed": synth.get("failed", 0),
            "length_out_of_range": sums.get("length", {}).get("out_of_range", 0),
            "compare_match": compare.get("match", 0),
            "compare_mismatch": compare.get("mismatch", 0),
            "compare_match_rate": rate("compare"),
        }

    def finish(self, stats: dict) -> None:
        with self._lock:
            self.final = {key: stats[key] for key in ("total", "pending", "decided", "pass", "rate")}
            self.final["fail_reasons"] = stats["fail_reasons"]
            self.emit(force = True, key = "__none__")

    def heartbeat(self):
        """返回上下文管理器：在等待封装期间，每隔 interval 秒刷新一次进度（速度、剩余时间随墙钟变化）。"""
        return _Heartbeat(progress = self)


class _Heartbeat:
    def __init__(self, progress: Progress):
        self.progress = progress
        self.stop = threading.Event()
        self.thread = threading.Thread(target = self._run, daemon = True)

    def _run(self) -> None:
        while not self.stop.wait(timeout = self.progress.interval):
            self.progress.emit(force = True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop.set()
        self.thread.join()
