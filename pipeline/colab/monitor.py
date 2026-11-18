"""训练监控：定时把训练进度、滚动窗口和 GPU 状态画成一张图，连同状态摘要写到输出目录（任务 C1）。

在 Colab 上训练时，训练由命令行会话在后台运行，浏览器连不上那台虚拟机。这个程序与训练同时在后台运行，
每隔 --interval 秒读一次训练输出目录（progress.json、train_log.jsonl、eval_log.jsonl）、训练日志、窗口程序日志和
nvidia-smi，写出 <输出目录>/monitor/dashboard.png 和 status.json。输出目录在 Google Drive 上时，可以在 Drive 里
直接打开这张图，或在任意 notebook 中挂载 Drive 后显示它（见 README）。训练结束（本程序启动后 progress.json
变为 finished、stopped 或 failed）后再写一次就退出。

只用 Python 标准库和 matplotlib，用 Colab 自带的 python3 运行，不需要训练环境。图中文字用英文：Colab 默认没有中文字体。

用法::

    python3 -m pipeline.colab.monitor --output-dir <训练输出目录> \\
        --train-log <训练日志> --window-log <窗口程序日志> [--interval 120]
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

# 与 pipeline/train/plot_curves.py 相同的配色
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
WINDOW_LINE = re.compile(r"^(\d\d):(\d\d):(\d\d) .*进度第 (\d+) 步，领先 (-?\d+) 步，本地 ([\d.]+) GB")
COPY_SPEED = re.compile(r"(\d+(?:\.\d+)?) MB/s")
COUNTED = ("[shard-fallback]", "[bad-audio]", "[bad-target]")


def read_jsonl(path):
    rows = []
    if path.exists():
        with open(file = path, mode = "r", encoding = "utf-8") as handle:
            for line in handle:
                try:
                    rows.append(json.loads(s = line))
                except json.JSONDecodeError:
                    pass
    return rows


def read_json(path):
    try:
        return json.loads(s = path.read_text(encoding = "utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def window_points(path):
    """窗口日志中带进度的行：(从第一行起的分钟数, 训练步数, 领先步数, 本地 GB) 和最近一次复制速度。"""
    points, speed, start, last, day = [], None, None, None, 0
    if not path or not Path(path).exists():
        return points, speed
    with open(file = path, mode = "r", encoding = "utf-8", errors = "replace") as handle:
        for line in handle:
            match = WINDOW_LINE.match(line)
            if match is None:
                continue
            seconds = int(match.group(1)) * 3600 + int(match.group(2)) * 60 + int(match.group(3)) + day
            if last is not None and seconds < last:
                day += 86400
                seconds += 86400
            last = seconds
            start = seconds if start is None else start
            points.append(((seconds - start) / 60, int(match.group(4)), int(match.group(5)), float(match.group(6))))
            copy = COPY_SPEED.search(line)
            if copy:
                speed = float(copy.group(1))
    return points, speed


def count_markers(path):
    counts = dict.fromkeys(COUNTED, 0)
    if path and Path(path).exists():
        with open(file = path, mode = "r", encoding = "utf-8", errors = "replace") as handle:
            for line in handle:
                for marker in COUNTED:
                    if marker in line:
                        counts[marker] += 1
    return counts


def gpu_sample():
    """(利用率 %, 已用显存 GB, 功率 W)；读不到时返回 None。"""
    try:
        text = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"],
            capture_output = True,
            text = True,
            timeout = 20,
            check = True,
        ).stdout.strip().splitlines()[0]
        util, memory, power = [float(value) for value in text.split(",")]
        return util, memory / 1024, power
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def style(axis, title):
    axis.set_facecolor(SURFACE)
    axis.grid(visible = True, axis = "y", color = GRID, linewidth = 0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors = TEXT_SECONDARY, labelsize = 8)
    axis.set_title(title, loc = "left", fontsize = 10, color = TEXT_PRIMARY)


def summarize(output_dir, train_log, window_log, gpu_history):
    progress = read_json(path = output_dir / "progress.json")
    train = read_jsonl(path = output_dir / "train_log.jsonl")
    evals = read_jsonl(path = output_dir / "eval_log.jsonl")
    window, copy_speed = window_points(path = window_log)
    recent = train[-10:]
    usage = shutil.disk_usage("/")
    return {
        "updated_utc": datetime.now(tz = timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "status": progress.get("status"),
        "step": progress.get("step"),
        "total_steps": progress.get("total_steps"),
        "eta": progress.get("eta"),
        "it_per_s": round(sum(row["it_per_s"] for row in recent) / len(recent), 3) if recent else None,
        "audio_h_per_h": round(sum(row.get("audio_h_per_h", 0) for row in recent) / len(recent), 1) if recent else None,
        "last_train_loss": train[-1]["loss"] if train else None,
        "last_eval": {"step": evals[-1]["step"], "loss": round(evals[-1]["overall"]["loss"], 4)} if evals else None,
        "window_lead_steps": window[-1][2] if window else None,
        "window_local_gb": window[-1][3] if window else None,
        "last_copy_mb_per_s": copy_speed,
        "markers": count_markers(path = train_log),
        "gpu": dict(zip(("util_pct", "mem_gb", "power_w"), gpu_history[-1][1:])) if gpu_history else None,
        "disk_free_gb": round(usage.free / 1e9, 1),
    }, train, evals, window


def draw(path, summary, train, evals, window, gpu_history):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (15, 8), facecolor = SURFACE)
    (loss_axis, speed_axis, text_axis), (lead_axis, util_axis, memory_axis) = axes
    if train:
        loss_axis.plot([row["step"] for row in train], [row["loss"] for row in train], color = SERIES[0], linewidth = 1, label = "train")
        speed_axis.plot([row["step"] for row in train], [row["it_per_s"] for row in train], color = SERIES[0], linewidth = 1)
    if evals:
        loss_axis.plot(
            [row["step"] for row in evals],
            [row["overall"]["loss"] for row in evals],
            color = SERIES[1],
            linewidth = 2,
            marker = "o",
            markersize = 5,
            label = "dev",
        )
    if train or evals:
        loss_axis.legend(loc = "upper right", frameon = False, fontsize = 8)
    if len(train) > 20:
        # 随机初始化的 projector 起步时损失很高，纵轴上限按去掉前 10% 记录后的最大值取
        values = [row["loss"] for row in train[len(train) // 10:]] + [row["overall"]["loss"] for row in evals[len(evals) // 10:]]
        loss_axis.set_ylim(top = max(values) * 1.15)
    style(axis = loss_axis, title = "loss (train / FLEURS dev) vs step")
    style(axis = speed_axis, title = "training speed (steps/s) vs step")
    if window:
        lead_axis.plot([point[0] for point in window], [point[2] for point in window], color = SERIES[2], linewidth = 1.5)
        lead_axis.axhline(y = 0, color = TEXT_SECONDARY, linewidth = 0.8)
    style(axis = lead_axis, title = "shard window lead (steps ahead) vs minutes")
    if gpu_history:
        minutes = [(sample[0] - gpu_history[0][0]) / 60 for sample in gpu_history]
        util_axis.plot(minutes, [sample[1] for sample in gpu_history], color = SERIES[3], linewidth = 1.5)
        memory_axis.plot(minutes, [sample[2] for sample in gpu_history], color = SERIES[5], linewidth = 1.5)
        util_axis.set_ylim(bottom = 0, top = 105)
    style(axis = util_axis, title = "GPU utilization (%) vs minutes")
    style(axis = memory_axis, title = "GPU memory used (GB) vs minutes")
    text_axis.axis("off")
    markers = summary["markers"]
    lines = [
        f"updated  {summary['updated_utc']} UTC",
        f"status   {summary['status']}",
        f"step     {summary['step']} / {summary['total_steps']}",
        f"ETA      {summary['eta']}",
        f"speed    {summary['it_per_s']} steps/s,  {summary['audio_h_per_h']} audio h/h",
        f"loss     train {summary['last_train_loss']},  dev {summary['last_eval']['loss'] if summary['last_eval'] else None} @ step {summary['last_eval']['step'] if summary['last_eval'] else None}",
        f"window   lead {summary['window_lead_steps']} steps,  local {summary['window_local_gb']} GB",
        f"copy     last {summary['last_copy_mb_per_s']} MB/s from Drive",
        f"logged   fallback {markers['[shard-fallback]']},  bad-audio {markers['[bad-audio]']},  bad-target {markers['[bad-target]']}",
        f"disk     {summary['disk_free_gb']} GB free",
    ]
    text_axis.text(x = 0, y = 1, s = "\n".join(lines), va = "top", family = "monospace", fontsize = 9, color = TEXT_PRIMARY, transform = text_axis.transAxes)
    figure.tight_layout()
    tmp = path.with_name(path.name + ".tmp.png")
    figure.savefig(tmp, dpi = 100, facecolor = SURFACE)
    plt.close(figure)
    os.replace(tmp, path)


def write_status(path, summary):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data = json.dumps(obj = summary, ensure_ascii = False, indent = 1) + "\n", encoding = "utf-8")
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description = "训练监控：定时写出进度图和状态摘要")
    parser.add_argument("--output-dir", required = True, type = Path, help = "训练输出目录（progress.json 等所在）")
    parser.add_argument("--train-log", default = None, help = "训练程序的日志文件，用来数 [shard-fallback] 等标记")
    parser.add_argument("--window-log", default = None, help = "shard_window.py run 的日志文件")
    parser.add_argument("--dest", default = None, type = Path, help = "写出目录，默认 <输出目录>/monitor")
    parser.add_argument("--interval", type = float, default = 120.0, help = "两次刷新之间的秒数")
    parser.add_argument("--once", action = "store_true", help = "只写一次就退出")
    args = parser.parse_args()
    dest = args.dest or args.output_dir / "monitor"
    dest.mkdir(parents = True, exist_ok = True)
    gpu_history = []
    started = time.time()
    progress_path = args.output_dir / "progress.json"
    while True:
        sample = gpu_sample()
        if sample is not None:
            gpu_history.append((time.time(),) + sample)
        summary, train, evals, window = summarize(
            output_dir = args.output_dir,
            train_log = args.train_log,
            window_log = args.window_log,
            gpu_history = gpu_history,
        )
        draw(
            path = dest / "dashboard.png",
            summary = summary,
            train = train,
            evals = evals,
            window = window,
            gpu_history = gpu_history,
        )
        write_status(path = dest / "status.json", summary = summary)
        # 续训开始前 progress.json 还是上一段的 stopped，只有在本程序启动之后写入的结束状态才算训练结束
        ended = summary["status"] in ("finished", "stopped", "failed") and progress_path.exists() and progress_path.stat().st_mtime > started
        if args.once or ended:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
