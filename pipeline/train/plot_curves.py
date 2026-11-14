"""画训练损失与验证损失曲线（需要 matplotlib；在单独的画图环境中运行，不依赖 torch）。

读取训练输出目录里的 train_log.jsonl（每 --log-every 步一行：这段步数的平均训练损失、学习率等）和
eval_log.jsonl（每 --eval-every 步一行：FLEURS dev 上的总体损失与各语言损失）。

- 只给一个运行：上图把训练损失和验证损失画在同一个纵轴上（同为每 token 交叉熵），下图画学习率；
- 给多个运行（如 v2b 的 0.8B、2B、4B）：按指标分成训练损失、验证损失、学习率三张小图，每个运行一条线，
  颜色按运行固定。

用法::

    python -m pipeline.train.plot_curves --run v2a=<输出目录> --out <图片路径> [--title 标题]
    python -m pipeline.train.plot_curves --run 0.8B=<目录> --run 2B=<目录> --run 4B=<目录> --out <图片路径>
"""

import argparse
import json
from pathlib import Path

# 分类色按固定顺序分配给运行或指标（经色盲安全检查：相邻两色 CVD ΔE 24.7、对比度均不低于 3:1）
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    with open(file = path, mode = "r", encoding = "utf-8") as handle:
        return [json.loads(s = line) for line in handle if line.strip()]


def load_run(directory: Path) -> dict:
    """返回 {train: [(步数, 损失)], lr: [(步数, 学习率)], dev: [(步数, 损失)]}；同一步有多行时取最后一行。"""
    train, lr, dev = {}, {}, {}
    for row in read_jsonl(path = directory / "train_log.jsonl"):
        train[row["step"]] = row["loss"]
        lr[row["step"]] = row["lr"]
    for row in read_jsonl(path = directory / "eval_log.jsonl"):
        dev[row["step"]] = row["overall"]["loss"]
    return {name: sorted(values.items()) for name, values in (("train", train), ("lr", lr), ("dev", dev))}


def style(axis, ylabel: str) -> None:
    axis.set_facecolor(SURFACE)
    axis.grid(visible = True, axis = "y", color = GRID, linewidth = 0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors = TEXT_SECONDARY, labelsize = 9)
    axis.set_ylabel(ylabel, color = TEXT_SECONDARY, fontsize = 10)


def end_label(axis, points: list, text: str, color: str, offset: float = 0.0) -> None:
    """在曲线末端标注名称和末值（文字用正文墨色，旁边的线段颜色表明是哪条曲线）。"""
    step, value = points[-1]
    axis.annotate(
        text = f"{text} {value:.3f}",
        xy = (step, value),
        xytext = (6, offset),
        textcoords = "offset points",
        va = "center",
        fontsize = 9,
        color = TEXT_PRIMARY,
    )


def loss_top(series: list) -> float:
    """损失纵轴的上限：去掉开头 2% 的训练记录（随机初始化的 projector 起步时损失很高，会把其余部分压扁）后取最大值，再留 15% 余量。"""
    values = []
    for points in series:
        values += [value for _, value in points[len(points) // 50:]]
    return max(values) * 1.15


def clip_note(axis, series: list, top: float) -> None:
    """训练损失有超出纵轴的部分时，在左上角注明被截掉的步数范围和起始值。"""
    above = [(step, value) for points in series for step, value in points if value > top]
    if above:
        axis.text(
            x = 0.01,
            y = 0.98,
            s = f"纵轴截至 {top:.2f}：前 {max(step for step, _ in above):,} 步的训练损失（最高 {max(value for _, value in above):.2f}）超出范围",
            transform = axis.transAxes,
            va = "top",
            fontsize = 8.5,
            color = TEXT_SECONDARY,
        )


def plot(runs: dict, out: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    steps_fmt = FuncFormatter(lambda value, _: f"{value / 1000:.0f}k" if value else "0")
    single = len(runs) == 1
    rows = 2 if single else 3
    figure, axes = plt.subplots(
        nrows = rows,
        ncols = 1,
        sharex = True,
        figsize = (10, 7.5 if single else 9.5),
        gridspec_kw = {"height_ratios": [3, 1] if single else [2, 2, 1]},
        facecolor = SURFACE,
    )
    if single:
        name, data = next(iter(runs.items()))
        loss_axis, lr_axis = axes
        loss_axis.plot(*zip(*data["train"]), color = SERIES[0], linewidth = 1.2, label = "训练损失（每 100 步平均）")
        loss_axis.plot(*zip(*data["dev"]), color = SERIES[1], linewidth = 2, marker = "o", markersize = 4.5, label = "验证损失（FLEURS dev）")
        end_label(axis = loss_axis, points = data["train"], text = "训练", color = SERIES[0], offset = -8)
        end_label(axis = loss_axis, points = data["dev"], text = "验证", color = SERIES[1], offset = 8)
        style(axis = loss_axis, ylabel = "每 token 交叉熵")
        top = loss_top(series = [data["train"], data["dev"]])
        loss_axis.set_ylim(bottom = min(value for _, value in data["train"] + data["dev"]) * 0.9, top = top)
        clip_note(axis = loss_axis, series = [data["train"]], top = top)
        loss_axis.legend(loc = "upper right", frameon = False, fontsize = 9, labelcolor = TEXT_PRIMARY)
        lr_axis.plot(*zip(*data["lr"]), color = TEXT_SECONDARY, linewidth = 1.5)
        style(axis = lr_axis, ylabel = "学习率")
    else:
        train_axis, dev_axis, lr_axis = axes
        for index, (name, data) in enumerate(runs.items()):
            color = SERIES[index]
            if data["train"]:
                train_axis.plot(*zip(*data["train"]), color = color, linewidth = 1.2, label = name)
                end_label(axis = train_axis, points = data["train"], text = name, color = color)
            if data["dev"]:
                dev_axis.plot(*zip(*data["dev"]), color = color, linewidth = 2, marker = "o", markersize = 4.5, label = name)
                end_label(axis = dev_axis, points = data["dev"], text = name, color = color)
            if data["lr"]:
                lr_axis.plot(*zip(*data["lr"]), color = color, linewidth = 1.5, label = name)
        style(axis = train_axis, ylabel = "训练损失")
        trains = [data["train"] for data in runs.values() if data["train"]]
        if trains:
            top = loss_top(series = trains)
            train_axis.set_ylim(bottom = min(value for points in trains for _, value in points) * 0.9, top = top)
            clip_note(axis = train_axis, series = trains, top = top)
        style(axis = dev_axis, ylabel = "验证损失（FLEURS dev）")
        style(axis = lr_axis, ylabel = "学习率")
        train_axis.legend(loc = "upper right", frameon = False, fontsize = 9, labelcolor = TEXT_PRIMARY)
    lr_axis.xaxis.set_major_formatter(steps_fmt)
    lr_axis.set_xlabel("训练步数", color = TEXT_SECONDARY, fontsize = 10)
    figure.suptitle(title, x = 0.06, ha = "left", color = TEXT_PRIMARY, fontsize = 13)
    figure.tight_layout()
    out.parent.mkdir(parents = True, exist_ok = True)
    figure.savefig(out, dpi = 150, facecolor = SURFACE)


def main() -> None:
    parser = argparse.ArgumentParser(description = "画训练损失与验证损失曲线")
    parser.add_argument("--run", action = "append", required = True, help = "名称=训练输出目录，可重复")
    parser.add_argument("--out", required = True, type = Path)
    parser.add_argument("--title", default = "训练损失与验证损失")
    args = parser.parse_args()
    runs = {}
    for item in args.run:
        name, _, directory = item.partition("=")
        runs[name] = load_run(directory = Path(directory))
    plot(runs = runs, out = args.out, title = args.title)
    for name, data in runs.items():
        last_dev = data["dev"][-1] if data["dev"] else None
        print(f"{name}: 训练记录 {len(data['train'])} 行，验证 {len(data['dev'])} 次，最后的验证损失 {last_dev}")


if __name__ == "__main__":
    main()
