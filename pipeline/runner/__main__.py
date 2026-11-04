"""命令行入口，见 pipeline/runner/__init__.py 的说明。"""

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .io_utils import read_jsonl, write_json_atomic, write_jsonl_atomic
from .plan import build_tasks, in_shard, load_entries, load_voices
from .progress import Progress
from .report import write_report
from .stages import Budget, Layout, RunnerError, StopRun, asr_stage, synth_stage


def parse_shard(text: str) -> tuple:
    index, count = text.split("/")
    index, count = int(index), int(count)
    if not 0 <= index < count:
        raise argparse.ArgumentTypeError("分片写成 K/N，0 <= K < N")
    return index, count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "合成—核对流水线")
    parser.add_argument("command", choices = ["plan", "run", "report"], help = "plan 只生成任务；run 依次执行全部阶段；report 只重新汇总")
    parser.add_argument("--config", required = True, type = Path, help = "配置文件（YAML 或 JSON）")
    parser.add_argument("--work-dir", required = True, type = Path, help = "工作目录：含 envs/ 与 gpu.lock")
    parser.add_argument("--out-dir", required = True, type = Path, help = "输出目录：音频、状态、清单与统计都写在这里")
    parser.add_argument("--texts", nargs = "+", type = Path, help = "文本条目 JSONL，或含 <lang>.jsonl 的目录")
    parser.add_argument("--voices", type = Path, help = "音色库 voices.jsonl")
    parser.add_argument("--split", choices = ["train", "test"], default = "train", help = "train 只用非留出音色，test 只用留出音色")
    parser.add_argument("--langs", nargs = "+", default = None, help = "只处理这些语言")
    parser.add_argument("--limit-per-lang", type = int, default = None, help = "每种语言最多取多少条（按 entry_id 排序取前面的）")
    parser.add_argument("--fallback-original", action = "store_true", help = "条目没有 script 时用原文当朗读稿（集成测试用）")
    parser.add_argument("--shard", type = parse_shard, default = (0, 1), help = "只执行第 K 片，写成 K/N")
    parser.add_argument("--progress-interval", type = float, default = 30.0, help = "进度日志与 progress.json 的刷新间隔（秒）")
    parser.add_argument("--stop-after-calls", type = int, default = None, help = "封装调用达到该次数后终止（模拟中途终止）")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(path = args.config, work_dir = args.work_dir)
    layout = Layout(out_dir = args.out_dir)
    layout.out.mkdir(parents = True, exist_ok = True)

    if args.command in ("plan", "run"):
        if not args.texts or not args.voices:
            print("plan、run 需要 --texts 和 --voices", file = sys.stderr)
            return 2
        entries = load_entries(
            paths = args.texts,
            langs = args.langs,
            limit_per_lang = args.limit_per_lang,
            fallback_original = args.fallback_original,
        )
        voices = load_voices(path = args.voices)
        tasks, info = build_tasks(cfg = cfg, entries = entries, voices = voices, split = args.split)
        write_jsonl_atomic(path = layout.out / "tasks.jsonl", rows = tasks)
        write_json_atomic(path = layout.out / "config_snapshot.json", obj = cfg)
        print(json.dumps({"tasks": len(tasks), **info}, ensure_ascii = False), file = sys.stderr)

    if args.command == "run":
        tasks = [row for row in read_jsonl(path = layout.out / "tasks.jsonl") if in_shard(row["sample_id"], args.shard)]
        budget = Budget(calls = args.stop_after_calls)
        progress = Progress(out_dir = layout.out, interval = args.progress_interval)
        try:
            synth_stage(cfg = cfg, tasks = tasks, layout = layout, work_dir = args.work_dir, budget = budget, progress = progress)
            asr_stage(cfg = cfg, tasks = tasks, layout = layout, work_dir = args.work_dir, budget = budget, role = "primary", progress = progress)
            asr_stage(cfg = cfg, tasks = tasks, layout = layout, work_dir = args.work_dir, budget = budget, role = "cross", progress = progress)
        except StopRun as stop:
            print(f"已终止：{stop}", file = sys.stderr)
            return 3
        except RunnerError as error:
            print(f"失败：{error}", file = sys.stderr)
            return 1

    stats = write_report(cfg = cfg, layout = layout)
    if args.command == "run":
        progress.finish(stats = stats)
    print(json.dumps({key: stats[key] for key in ("total", "pending", "decided", "pass", "rate")}, ensure_ascii = False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
