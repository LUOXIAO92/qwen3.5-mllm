"""命令行入口，见 pipeline/runner/__init__.py 的说明。"""

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .io_utils import write_json_atomic, write_jsonl_atomic
from .plan import build_tasks, in_shard, load_entries, load_voices
from .progress import Progress
from .report import write_report
from .stages import (
    Budget,
    Layout,
    RunnerError,
    StopRun,
    asr_stage,
    compare_stage,
    length_stage,
    tts_stage,
)
from .validate import InputError, load_checked

COMMANDS = ["plan", "tts", "length", "asr", "compare", "manifest", "all"]


def parse_shard(text: str) -> tuple:
    index, count = text.split("/")
    index, count = int(index), int(count)
    if not 0 <= index < count:
        raise argparse.ArgumentTypeError("分片写成 K/N，0 <= K < N")
    return index, count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "合成—核对流水线，每个子命令只读取上一步写出的文件")
    parser.add_argument(
        "command",
        choices = COMMANDS,
        help = "plan 生成任务；tts 合成；length 时长检查；asr 识别（--role）；compare 比对；manifest 写训练清单与统计；all 依次执行全部",
    )
    parser.add_argument("--config", required = True, type = Path, help = "配置文件（YAML 或 JSON）")
    parser.add_argument("--work-dir", required = True, type = Path, help = "工作目录：含 envs/ 与 gpu.lock")
    parser.add_argument("--out-dir", required = True, type = Path, help = "输出目录：音频、各步的结果文件、状态与日志")
    parser.add_argument("--texts", nargs = "+", type = Path, help = "plan：文本条目 JSONL，或含 <lang>.jsonl 的目录")
    parser.add_argument("--scripts", nargs = "+", type = Path, default = None, help = "plan：朗读稿 JSONL（entry_id、script、tn_status）")
    parser.add_argument("--voices", type = Path, help = "plan：音色库 voices.jsonl")
    parser.add_argument("--split", choices = ["train", "test"], default = "train", help = "plan：train 只用非留出音色，test 只用留出音色")
    parser.add_argument("--langs", nargs = "+", default = None, help = "plan：只处理这些语言")
    parser.add_argument("--limit-per-lang", type = int, default = None, help = "plan：每种语言最多取多少条（按 entry_id 排序取前面的）")
    parser.add_argument("--fallback-original", action = "store_true", help = "plan：条目没有 script 时用原文当朗读稿（集成测试用）")
    parser.add_argument("--models", nargs = "+", default = None, help = "tts、asr：只处理这些模型（配置里的名字）；CPU 与 GPU 模型可在两个进程里同时跑同一输出目录")
    parser.add_argument("--role", choices = ["primary", "cross"], default = "primary", help = "asr：主 ASR 或交叉 ASR")
    parser.add_argument("--shard", type = parse_shard, default = (0, 1), help = "tts、asr：只执行第 K 片，写成 K/N")
    parser.add_argument("--progress-interval", type = float, default = 30.0, help = "进度日志与 progress.json 的刷新间隔（秒）")
    parser.add_argument("--stop-after-calls", type = int, default = None, help = "封装调用达到该次数后终止（模拟中途终止）")
    return parser.parse_args()


def run_plan(args: argparse.Namespace, cfg: dict, layout: Layout) -> None:
    if not args.texts or not args.voices:
        raise InputError("plan 需要 --texts 和 --voices")
    entries = load_entries(
        paths = args.texts,
        langs = args.langs,
        limit_per_lang = args.limit_per_lang,
        fallback_original = args.fallback_original,
        scripts = args.scripts,
    )
    voices = load_voices(path = args.voices)
    tasks, info = build_tasks(cfg = cfg, entries = entries, voices = voices, split = args.split)
    write_jsonl_atomic(path = layout.tasks, rows = tasks)
    write_json_atomic(path = layout.out / "config_snapshot.json", obj = cfg)
    print(json.dumps({"tasks": len(tasks), **info}, ensure_ascii = False), file = sys.stderr)


def main() -> int:
    args = parse_args()
    cfg = load_config(path = args.config, work_dir = args.work_dir)
    layout = Layout(out_dir = args.out_dir)
    layout.out.mkdir(parents = True, exist_ok = True)
    progress = Progress(out_dir = layout.out, interval = args.progress_interval)
    budget = Budget(calls = args.stop_after_calls)
    steps = [args.command] if args.command != "all" else ["plan", "tts", "length", "asr", "compare", "asr-cross", "compare", "manifest"]
    try:
        for step in steps:
            if step == "plan":
                run_plan(args = args, cfg = cfg, layout = layout)
                continue
            tasks = load_checked(path = layout.tasks, kind = "tasks", hint = "先运行 plan 子命令")
            shard_tasks = [row for row in tasks if in_shard(row["sample_id"], args.shard)]
            if step == "tts":
                tts_stage(cfg = cfg, tasks = shard_tasks, layout = layout, work_dir = args.work_dir, budget = budget, progress = progress, models = args.models)
            elif step == "length":
                length_stage(cfg = cfg, tasks = tasks, layout = layout, progress = progress)
            elif step in ("asr", "asr-cross"):
                role = "cross" if step == "asr-cross" else args.role
                asr_stage(
                    cfg = cfg,
                    tasks = shard_tasks,
                    layout = layout,
                    work_dir = args.work_dir,
                    budget = budget,
                    role = role,
                    progress = progress,
                    models = args.models,
                )
            elif step == "compare":
                compare_stage(cfg = cfg, tasks = tasks, layout = layout, progress = progress)
            elif step == "manifest":
                stats = write_report(cfg = cfg, layout = layout)
                progress.finish(stats = stats)
                print(json.dumps({key: stats[key] for key in ("total", "pending", "decided", "pass", "rate")}, ensure_ascii = False))
    except StopRun as stop:
        print(f"已终止：{stop}", file = sys.stderr)
        return 3
    except InputError as error:
        print(f"输入错误：{error}", file = sys.stderr)
        return 2
    except RunnerError as error:
        print(f"失败：{error}", file = sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
