"""训练进度：每隔固定步数打印一行进度，并维护持续更新的 progress.json。

progress.json 在开始、每次写日志行、dev 评估、保存检查点、停止、完成或出错时整体重写，内容是
当前状态（running、stopped、finished、failed）、步数与总步数、最近一个日志窗口的速度和损失、
已用时间、预计剩余时间、最近的 dev 损失和检查点。不读日志也能知道进度。
"""

import json
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path


def format_duration(seconds):
    """秒数写成 时:分:秒。"""
    if seconds is None:
        return None
    seconds = int(round(number = seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def now_text():
    return datetime.now(tz = timezone.utc).astimezone().isoformat(timespec = "seconds")


class Progress:
    """记录训练用时，估计剩余时间，写出 progress.json。

    已用时间包括续训之前各段的训练时间（由检查点带入）。剩余时间按最近若干个日志窗口的实际墙钟
    速度估计，其中包含 dev 评估和保存检查点花的时间。
    """

    def __init__(
        self,
        path,
        total_steps,
        start_step,
        previous_seconds = 0.0,
        window_count = 20,
    ):
        self.path = Path(path)
        self.total_steps = total_steps
        self.previous_seconds = previous_seconds
        self.segment_start = time.perf_counter()
        self.marks = deque([(start_step, self.segment_start)], maxlen = window_count + 1)
        self.state = {
            "status": "running",
            "step": start_step,
            "total_steps": total_steps,
            "segment_started_at": now_text(),
            "segment_start_step": start_step,
        }

    def elapsed(self):
        return self.previous_seconds + time.perf_counter() - self.segment_start

    def mark(self, step):
        self.marks.append((step, time.perf_counter()))

    def eta_seconds(self, step):
        first_step, first_time = self.marks[0]
        last_step, last_time = self.marks[-1]
        if last_step <= first_step:
            return None
        seconds_per_step = (last_time - first_time) / (last_step - first_step)
        return (self.total_steps - step) * seconds_per_step

    def write(self, step, **fields):
        """更新状态并整体重写 progress.json（先写临时文件再改名，读取方不会读到半个文件）。"""
        self.state.update(fields)
        eta = self.eta_seconds(step = step) if self.state["status"] == "running" else None
        finish_at = None
        if eta is not None:
            finish_at = datetime.now(tz = timezone.utc) + timedelta(seconds = eta)
            finish_at = finish_at.astimezone().isoformat(timespec = "seconds")
        elapsed = self.elapsed()
        self.state.update({
            "step": step,
            "percent": round(number = 100 * step / self.total_steps, ndigits = 2),
            "elapsed": format_duration(seconds = elapsed),
            "elapsed_s": round(number = elapsed),
            "eta": format_duration(seconds = eta),
            "eta_s": round(number = eta) if eta is not None else None,
            "eta_finish_at": finish_at,
            "updated_at": now_text(),
        })
        staging = self.path.with_name(name = self.path.name + ".tmp")
        staging.write_text(
            data = json.dumps(obj = self.state, ensure_ascii = False, indent = 2) + "\n",
            encoding = "utf-8",
        )
        staging.replace(target = self.path)

    def line(self, record):
        """日志窗口的一行文字。"""
        eta = format_duration(seconds = record["eta_s"]) if record["eta_s"] is not None else "-"
        return (
            f"[步 {record['step']}/{record['total_steps']} {100 * record['step'] / record['total_steps']:.1f}%]"
            f" {record['it_per_s']:.2f} it/s | {record['samples_per_s']:.1f} 样本/s"
            f" | {record['audio_h_per_h']:.0f} 音频h/h | loss {record['loss']:.4f}"
            f" | lr {record['lr']:.2e} | 显存 {record['mem_gb']:.1f} GB"
            f" | 已用 {format_duration(seconds = record['elapsed_s'])} | 剩余 {eta}"
        )
