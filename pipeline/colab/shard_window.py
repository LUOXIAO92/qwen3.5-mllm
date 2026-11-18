"""在 Colab 本地磁盘上维护分片窗口：按训练进度把分片从 Google Drive 整个顺序复制下来，用过的删掉（任务 C1）。

分片的种类和步数范围见 pipeline/train/shards.py。本程序只做整文件的顺序复制（先写 .tmp，大小核对后改名），
不从 Drive 随机读取。常驻分片放在 --pool-dir，整个训练期间保留；顺序分片放在 --window-dir，按下面的优先级复制：

1. 第一优先：全部常驻分片，以及服务第 [检查点步数, 当前步数 + --ahead-steps) 步的顺序分片。默认 31,600 步，
   每步 16 条时约 50 万条样本、约 56 GB。
2. 第二优先：第一优先的都已在本地时，在 --disk-budget-gb（常驻与窗口合计）之内，按步数顺序继续预取后面的顺序分片。
   第一优先还有分片缺失（包括 Drive 上还没有）时不开始第二优先的复制。

当前步数读自 <输出目录>/progress.json，检查点步数读自 <输出目录>/latest（续训从这一步开始）。顺序分片只有在它的
最后一步早于「检查点步数」和「当前步数减 --margin-steps」两者时才删除：训练中途重启时从最近的检查点续训，这之后的
数据都还要用；数据读取进程会提前准备后面的批次，余量也覆盖进度文件的更新间隔。每次复制、删除都打一行日志，
其中「领先」是本地连续可用的数据比训练进度多出的步数。

    # 开始训练前：复制常驻分片和起点之后 --prepare-steps 步的顺序分片，复制完才退出
    python -m pipeline.colab.shard_window prepare --index <Drive 分片目录>/index.json --source <Drive 分片目录> \\
        --pool-dir <本地>/pool --window-dir <本地>/window --output-dir <训练输出目录>
    # 训练期间在后台运行，训练完成（progress.json 为 finished）后退出
    python -m pipeline.colab.shard_window run ...（参数同上，另有 --ahead-steps、--disk-budget-gb 等）
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from pipeline.train.shards import complete_path, read_header

GB = 1e9
COPY_CHUNK = 16 * 2 ** 20
# 准备阶段复制连续失败几次后放弃
PREPARE_ATTEMPTS = 3


def log(message):
    print(f"{time.strftime('%H:%M:%S')} {message}", flush = True)


def checkpoint_step(output_dir):
    """<输出目录>/latest 指向的检查点步数，没有时返回 None。"""
    try:
        name = (Path(output_dir) / "latest").read_text(encoding = "utf-8").strip()
        return int(name.rsplit(sep = "-", maxsplit = 1)[1])
    except (OSError, ValueError, IndexError):
        return None


def progress_state(output_dir):
    """progress.json 中的 (已完成步数, 状态)，读不到时为 (None, None)。"""
    try:
        progress = json.loads(s = (Path(output_dir) / "progress.json").read_text(encoding = "utf-8"))
        return int(progress["step"]), progress.get("status")
    except (OSError, ValueError, KeyError, TypeError):
        return None, None


class Window:
    def __init__(
        self,
        header,
        source,
        pool_dir,
        window_dir,
        ahead_steps,
        budget_bytes,
        margin_steps,
        keep_free_bytes,
    ):
        self.shards = header["shards"]
        self.source = Path(source)
        self.pool_dir = Path(pool_dir)
        self.window_dir = Path(window_dir)
        self.ahead_steps = ahead_steps
        self.budget_bytes = budget_bytes
        self.margin_steps = margin_steps
        self.keep_free_bytes = keep_free_bytes
        self.pool = [number for number, shard in enumerate(self.shards) if shard["kind"] == "pool"]
        self.stream = sorted(
            (number for number, shard in enumerate(self.shards) if shard["kind"] == "stream"),
            key = lambda number: self.shards[number]["first_step"],
        )
        self.unavailable = set()
        self.low_space = False
        for directory in (self.pool_dir, self.window_dir):
            directory.mkdir(parents = True, exist_ok = True)
            # 上次中断留下的半个文件
            for leftover in directory.glob(pattern = "*.tmp"):
                leftover.unlink()

    def directory(self, number):
        return self.pool_dir if self.shards[number]["kind"] == "pool" else self.window_dir

    def is_local(self, number):
        return complete_path(directory = self.directory(number = number), shard = self.shards[number]) is not None

    def local_bytes(self):
        return sum(shard["bytes"] for number, shard in enumerate(self.shards) if self.is_local(number = number))

    def steps_state(self, output_dir, start_step):
        """返回 (检查点步数, 当前步数, 状态)。没有检查点时检查点步数取 start_step。"""
        floor = checkpoint_step(output_dir = output_dir)
        floor = start_step if floor is None else floor
        step, status = progress_state(output_dir = output_dir)
        return floor, max(floor, step if step is not None else floor), status

    def lead(self, position):
        """从当前步数起本地连续可用的顺序分片覆盖到第几步，返回比当前步数多出的步数。"""
        reach = position
        for number in self.stream:
            shard = self.shards[number]
            if shard["last_step"] < position:
                continue
            if shard["first_step"] > reach or not self.is_local(number = number):
                break
            reach = shard["last_step"] + 1
        return reach - position

    def priority_one(self, floor, position):
        stream = [
            number for number in self.stream
            if self.shards[number]["last_step"] >= floor
            and self.shards[number]["first_step"] < position + self.ahead_steps
        ]
        return self.pool + stream

    def priority_two(self, position):
        return [number for number in self.stream if self.shards[number]["first_step"] >= position + self.ahead_steps]

    def describe(self, number):
        shard = self.shards[number]
        if shard["kind"] == "pool":
            return f"{shard['name']}（常驻）"
        return f"{shard['name']}（第 {shard['first_step']}–{shard['last_step']} 步）"

    def delete_behind(self, floor, position):
        """删除最后一步早于 min(检查点步数, 当前步数 - 余量) 的本地顺序分片。"""
        limit = min(floor, position - self.margin_steps)
        for number in self.stream:
            shard = self.shards[number]
            if shard["last_step"] >= limit:
                break
            path = self.window_dir / shard["name"]
            if path.exists():
                path.unlink()
                log(f"删除 {self.describe(number = number)}，进度第 {position} 步，检查点第 {floor} 步")

    def available(self, number):
        """Drive 上有完整文件时返回它的路径；没有时记一次日志，返回 None。"""
        origin = complete_path(directory = self.source, shard = self.shards[number])
        if origin is None and number not in self.unavailable:
            self.unavailable.add(number)
            log(f"{self.describe(number = number)} 在 {self.source} 中不存在或不完整，暂时跳过")
        elif origin is not None:
            self.unavailable.discard(number)
        return origin

    def copy(self, number, label, position):
        """复制一个分片；剩余空间不够或复制出错时返回 False。"""
        shard = self.shards[number]
        origin = self.available(number = number)
        if origin is None:
            return False
        target = self.directory(number = number) / shard["name"]
        if shutil.disk_usage(path = target.parent).free - shard["bytes"] < self.keep_free_bytes:
            if not self.low_space:
                log(f"本地剩余空间不足，暂不复制 {self.describe(number = number)}")
            self.low_space = True
            return False
        self.low_space = False
        tmp = target.with_name(name = target.name + ".tmp")
        start = time.perf_counter()
        try:
            with open(file = origin, mode = "rb") as reader, open(file = tmp, mode = "wb") as writer:
                shutil.copyfileobj(fsrc = reader, fdst = writer, length = COPY_CHUNK)
            if tmp.stat().st_size != shard["bytes"]:
                raise OSError(f"复制后大小 {tmp.stat().st_size}，索引中为 {shard['bytes']}")
            tmp.replace(target = target)
        except OSError as error:
            tmp.unlink(missing_ok = True)
            log(f"复制 {self.describe(number = number)} 出错：{error}")
            return False
        seconds = max(time.perf_counter() - start, 1e-9)
        log(
            f"复制 {self.describe(number = number)} [{label}] {shard['bytes'] / GB:.2f} GB {seconds:.1f} 秒 "
            f"{shard['bytes'] / 1e6 / seconds:.0f} MB/s | 进度第 {position} 步，领先 {self.lead(position = position)} 步，"
            f"本地 {self.local_bytes() / GB:.1f} GB"
        )
        return True

    def tick(self, floor, position, priority_two = True):
        """做一件事：删除用过的分片，再复制一个分片。复制了分片时返回 True；第一优先已全部在本地时 complete 为 True。"""
        self.delete_behind(floor = floor, position = position)
        missing = [
            number for number in self.priority_one(floor = floor, position = position)
            if not self.is_local(number = number)
        ]
        for number in missing:
            if self.copy(number = number, label = "第一优先", position = position):
                return True, False
            if self.low_space:
                return False, False
        if missing:
            return False, False
        if not priority_two:
            return False, True
        used = self.local_bytes()
        for number in self.priority_two(position = position):
            if self.is_local(number = number):
                continue
            if used + self.shards[number]["bytes"] > self.budget_bytes:
                break
            return self.copy(number = number, label = "第二优先", position = position), True
        return False, True

    def status_line(self, floor, position, status):
        stream_local = sum(1 for number in self.stream if self.is_local(number = number))
        pool_local = sum(1 for number in self.pool if self.is_local(number = number))
        return (
            f"进度第 {position} 步（{status or '未开始'}），检查点第 {floor} 步，领先 {self.lead(position = position)} 步，"
            f"本地 {self.local_bytes() / GB:.1f} GB（常驻 {pool_local}/{len(self.pool)}，顺序 {stream_local} 个）"
        )


def make_window(args):
    header = read_header(index_path = args.index)
    if header["order"] != "plan":
        raise SystemExit("窗口管理只用于按训练计划排列的索引（pack_shards index --order plan）")
    window = Window(
        header = header,
        source = args.source,
        pool_dir = args.pool_dir,
        window_dir = args.window_dir,
        ahead_steps = args.ahead_steps if args.command == "run" else args.prepare_steps,
        budget_bytes = args.disk_budget_gb * GB,
        margin_steps = args.margin_steps,
        keep_free_bytes = args.keep_free_gb * GB,
    )
    return window


def command_prepare(args):
    window = make_window(args = args)
    floor, position, status = window.steps_state(output_dir = args.output_dir, start_step = args.start_step)
    log(f"准备：{window.status_line(floor = floor, position = position, status = status)}")
    failures = 0
    while True:
        worked, complete = window.tick(floor = floor, position = floor, priority_two = False)
        if complete:
            break
        if worked:
            failures = 0
            continue
        # 复制出错或空间不足：稍等再试；Drive 上缺分片时重试也没有用
        missing = [
            number for number in window.priority_one(floor = floor, position = floor)
            if not window.is_local(number = number)
        ]
        failures += 1
        if failures >= PREPARE_ATTEMPTS or any(number in window.unavailable for number in missing):
            log(f"缺少 {len(missing)} 个分片，无法开始训练：{[window.describe(number = number) for number in missing[:5]]}")
            sys.exit(1)
        time.sleep(10)
    log(f"准备完成：{window.status_line(floor = floor, position = floor, status = status)}")


def command_run(args):
    window = make_window(args = args)
    last_report = 0.0
    while True:
        floor, position, status = window.steps_state(output_dir = args.output_dir, start_step = args.start_step)
        if status == "finished":
            log(f"训练已完成，退出。{window.status_line(floor = floor, position = position, status = status)}")
            return
        worked, _ = window.tick(floor = floor, position = position)
        if time.monotonic() - last_report >= args.report_seconds:
            log(window.status_line(floor = floor, position = position, status = status))
            last_report = time.monotonic()
        if not worked:
            time.sleep(args.interval)


def parse_args(argv = None):
    parser = argparse.ArgumentParser(description = "在本地维护分片窗口，按训练进度从 Google Drive 复制分片")
    commands = parser.add_subparsers(dest = "command", required = True)
    for name, text in (("prepare", "复制开始训练所需的分片后退出"), ("run", "训练期间持续维护窗口")):
        command = commands.add_parser(name = name, help = text)
        command.add_argument("--index", required = True, help = "分片索引 json")
        command.add_argument("--source", required = True, help = "Drive 上的分片目录")
        command.add_argument("--pool-dir", required = True, help = "本地常驻分片目录")
        command.add_argument("--window-dir", required = True, help = "本地顺序分片目录")
        command.add_argument("--output-dir", required = True, help = "训练输出目录（读 latest 和 progress.json）")
        command.add_argument(
            "--start-step",
            type = int,
            default = 0,
            help = "输出目录中还没有检查点时从第几步开始",
        )
        command.add_argument(
            "--margin-steps",
            type = int,
            default = 500,
            help = "删除顺序分片时在当前步数之前保留的余量",
        )
        command.add_argument(
            "--keep-free-gb",
            type = float,
            default = 5.0,
            help = "本地磁盘至少保留的剩余空间（GB）",
        )
        command.add_argument(
            "--disk-budget-gb",
            type = float,
            default = 150.0,
            help = "常驻与顺序分片在本地合计最多占用（GB），只限制第二优先的预取",
        )
    commands.choices["prepare"].add_argument(
        "--prepare-steps",
        type = int,
        default = 2000,
        help = "开始训练前复制起点之后多少步的顺序分片",
    )
    run = commands.choices["run"]
    run.add_argument(
        "--ahead-steps",
        type = int,
        default = 31600,
        help = "第一优先：当前步数之后多少步的顺序分片（31,600 步 × 16 条约 50 万条样本）",
    )
    run.add_argument(
        "--interval",
        type = float,
        default = 30.0,
        help = "没有要复制的分片时，隔多少秒再检查",
    )
    run.add_argument(
        "--report-seconds",
        type = float,
        default = 300.0,
        help = "隔多少秒打一行状态",
    )
    return parser.parse_args(args = argv)


def main(argv = None):
    args = parse_args(argv = argv)
    if args.command == "prepare":
        command_prepare(args = args)
    else:
        command_run(args = args)


if __name__ == "__main__":
    main()
