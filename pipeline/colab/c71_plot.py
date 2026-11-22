"""#71 C 组的损失图：把 projector 与六个全量微调运行的损失画成一张图（需要 matplotlib，用 Colab 自带的 python3 运行，不依赖 torch）。

读取 <运行根目录>/<运行名>/ 下的 train_log.jsonl（每 --log-every 步一行）和 eval_log.jsonl（每 --eval-every 步一行），
运行名与 c_runs.py 相同（v2b-2b-cpt77-projector、v2b-2b-cpt77-c0、…-c-mix）。四张小图：

1. projector 从随机初始化训练的训练损失（标出驱动程序选定的突降步，读 c71 的 status.json）；
2. 全量微调的 ASR 训练损失（加回放的运行中 loss 只含 ASR 部分，与 C0 可以直接比较；滑动平均），每个运行一条线；
3. FLEURS dev 损失，每个运行一条线；
4. 回放部分的每 token NLL（模型生成部分，按训练的损失范围；滑动平均），每个加回放的运行一条线。

全量微调三张图的纵轴不从 0 开始（去掉起步阶段后取最小到最大），运行之间的差别才看得出。

颜色按运行固定（C0 不出现在第 4 张图中时，其余运行的颜色也不变）。图中文字用英文：Colab 默认没有中文字体。

用法::

    python3 -m pipeline.colab.c71_plot --runs-dir $WORK_DIR/runs --status $WORK_DIR/c71/status.json \\
        --out $DRIVE_DIR/runs/c71-status/loss.png [--interval 300]

--interval 给出时每隔这么多秒重画一次，直到 status.json 的 phase 为 trained、done、done_with_failures 或 blocked。
"""

import argparse
import json
import time
from pathlib import Path

from pipeline.train.plot_curves import GRID, SERIES, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, read_jsonl, style

PREFIX = "v2b-2b-cpt77-"
RUNS = ["c0", "c-free_raw", "c-free_chat", "c-cont_raw", "c-cont_chat", "c-mix"]
FINAL_PHASES = ("trained", "done", "done_with_failures", "blocked")
# 训练与回放曲线的滑动平均点数（训练日志每 --log-every 步一行）
SMOOTH = 20


def series(rows: list, key: str) -> list:
    """[(步数, 值)]，同一步取最后一行，值为 None 的行跳过。"""
    values = {}
    for row in rows:
        if row.get(key) is not None:
            values[row["step"]] = row[key]
    return sorted(values.items())


def dev_series(rows: list) -> list:
    return sorted({row["step"]: row["overall"]["loss"] for row in rows}.items())


def top_after(points: list, start: int) -> float:
    """纵轴上限：去掉起步阶段（start 步之前）后的最大值，留 10% 余量。"""
    kept = [value for step, value in points if step >= start] or [value for _, value in points]
    return max(kept) * 1.1 if kept else 1.0


def limits_after(points: list, start: int) -> tuple:
    """损失曲线的纵轴范围：去掉起步阶段后取最小到最大，上下各留 8% 余量（不从 0 开始，运行之间的差别才看得出）。"""
    kept = [value for step, value in points if step >= start] or [value for _, value in points]
    low, high = min(kept), max(kept)
    margin = max(high - low, 1e-3) * 0.08
    return low - margin, high + margin


def smooth(points: list, window: int) -> list:
    """滑动平均（每点取它和前面 window - 1 个点的平均），训练日志每段的平均值仍有噪声，多条线叠在一起时看不清。"""
    out, total = [], 0.0
    for index, (step, value) in enumerate(points):
        total += value
        if index >= window:
            total -= points[index - window][1]
        out.append((step, total / min(index + 1, window)))
    return out


def draw(runs_dir: Path, status_path: Path, out: Path, tag: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    status = json.loads(status_path.read_text(encoding = "utf-8")) if status_path.exists() else {}
    figure, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (13, 8.5), facecolor = SURFACE)
    (projector_axis, train_axis), (dev_axis, replay_axis) = axes

    projector_rows = read_jsonl(path = runs_dir / (PREFIX + "projector" + tag) / "train_log.jsonl")
    projector = series(rows = projector_rows, key = "loss")
    if projector:
        projector_axis.plot(*zip(*projector), color = SERIES[0], linewidth = 1.2)
        projector_axis.set_ylim(bottom = 0, top = top_after(points = projector, start = 200))
        drop = status.get("drop_step")
        if drop:
            projector_axis.axvline(x = drop, color = TEXT_SECONDARY, linewidth = 1, linestyle = "--")
            projector_axis.annotate(
                text = f"loss drop at step {drop}",
                xy = (drop, projector_axis.get_ylim()[1] * 0.92),
                xytext = (6, 0),
                textcoords = "offset points",
                fontsize = 9,
                color = TEXT_PRIMARY,
            )
    style(axis = projector_axis, ylabel = "train loss")
    projector_axis.set_title("Projector from scratch (CPT format)", color = TEXT_PRIMARY, fontsize = 11, loc = "left")

    train_points, dev_points, replay_points = [], [], []
    for index, run in enumerate(RUNS):
        directory = runs_dir / (PREFIX + run + tag)
        color = SERIES[index % len(SERIES)]
        rows = read_jsonl(path = directory / "train_log.jsonl")
        train = smooth(points = series(rows = rows, key = "loss"), window = SMOOTH)
        dev = dev_series(rows = read_jsonl(path = directory / "eval_log.jsonl"))
        replay = smooth(points = series(rows = rows, key = "replay_nll"), window = SMOOTH)
        if train:
            train_axis.plot(*zip(*train), color = color, linewidth = 1.0, label = run)
            train_points += train
        if dev:
            dev_axis.plot(*zip(*dev), color = color, linewidth = 2, marker = "o", markersize = 4, label = run)
            dev_points += dev
        if replay:
            replay_axis.plot(*zip(*replay), color = color, linewidth = 1.0, label = run)
            replay_points += replay

    for axis, points, ylabel, title in (
        (train_axis, train_points, "ASR train loss", f"Full fine-tuning: ASR loss (replay excluded; {SMOOTH}-point moving average)"),
        (dev_axis, dev_points, "FLEURS dev loss", "Full fine-tuning: FLEURS dev loss"),
        (replay_axis, replay_points, "replay NLL per token", f"Full fine-tuning: replay NLL ({SMOOTH}-point moving average)"),
    ):
        style(axis = axis, ylabel = ylabel)
        axis.set_title(title, color = TEXT_PRIMARY, fontsize = 11, loc = "left")
        if points:
            low, high = limits_after(points = points, start = 300)
            axis.set_ylim(bottom = low, top = high)
            axis.legend(frameon = False, fontsize = 9, labelcolor = TEXT_PRIMARY, ncols = 2)
        else:
            axis.text(x = 0.5, y = 0.5, s = "no data yet", transform = axis.transAxes, ha = "center", color = TEXT_SECONDARY)
    for axis in axes.flat:
        axis.set_xlabel("step", color = TEXT_SECONDARY, fontsize = 10)

    phase = status.get("phase", "?")
    figure.suptitle(
        f"#71 C runs — phase: {phase} — {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
        color = TEXT_PRIMARY,
        fontsize = 12,
        x = 0.01,
        ha = "left",
    )
    figure.tight_layout()
    out.parent.mkdir(parents = True, exist_ok = True)
    temporary = out.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi = 120, facecolor = SURFACE)
    plt.close(figure)
    temporary.replace(out)


def main() -> None:
    parser = argparse.ArgumentParser(description = "#71 C 组的损失图")
    parser.add_argument("--runs-dir", required = True, type = Path, help = "各运行输出目录的上级目录（$WORK_DIR/runs）")
    parser.add_argument("--status", required = True, type = Path, help = "驱动程序的 status.json")
    parser.add_argument("--out", required = True, type = Path)
    parser.add_argument("--tag", default = "", help = "运行名后缀（C71_TAG）")
    parser.add_argument("--interval", type = int, default = None, help = "每隔多少秒重画一次；不给时只画一次")
    args = parser.parse_args()
    while True:
        try:
            draw(runs_dir = args.runs_dir, status_path = args.status, out = args.out, tag = args.tag)
        except Exception as error:  # 画图失败不影响训练，下次再试
            print(f"画图失败：{error!r}", flush = True)
        if args.interval is None:
            return
        status = json.loads(args.status.read_text(encoding = "utf-8")) if args.status.exists() else {}
        if status.get("phase") in FINAL_PHASES:
            draw(runs_dir = args.runs_dir, status_path = args.status, out = args.out, tag = args.tag)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
