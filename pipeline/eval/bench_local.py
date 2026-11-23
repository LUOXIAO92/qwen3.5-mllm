"""#71 文本基准的本机驱动程序：参考模型与 C 组六个运行的 MMLU-Redux、C-Eval、MMMLU、IFEval，最后是 GPQA Diamond（思考模式）。

用户 2026-11-24：看每个模型保留了多少 Qwen3.5-2B 原有的文本能力。各基准的数据、提示与打分见 pipeline/eval/text_bench.py
开头，IFEval 的打分见 pipeline/eval/ifeval/score.py。

模型：参考模型（ref，融合模型中原始的语言模型）和 c_runs.RUNS 的六个运行（<运行目录>/v2b-2b-cpt77-<运行>/
checkpoint-0030000/language_model.safetensors，按 safetensors 文件头核对已下载完整；纯文本评测不需要 projector）。

顺序：先对每个模型做四项简单基准（每个模型一个进程，模型只载入一次，约几分钟），参考模型在前；全部模型做完后，再对每个
模型做 GPQA（每个模型约 45 分钟，最费时，放在最后）。IFEval 生成之后，用 --py-ifeval（需要 langdetect 与 nltk 的
python，只用 CPU、不持有 GPU 锁）打分；没有给 --py-ifeval 或它缺少依赖时只做生成、跳过打分，并在 driver.log 中说明，
以后补上 --py-ifeval 重新运行即可补打分。GPQA 的数据（gated，需要在 Hugging Face 上同意条款后下载）不在时跳过 GPQA 并
说明。每做完一种项目（简单基准、GPQA）之前重新检查哪些运行已下载完整，中途下载完成的运行在后面补上。

续跑与失败：结果文件已存在的项目跳过（text_bench 按基准跳过），单项失败记下后接着做；重新运行驱动程序只重做缺少结果的
项目。GPU 的等待与锁与 c71_local 相同（使用它的 Driver）：每一项开始前持有 <工作目录>/gpu.lock，已用显存超过 --busy-mib
时等待，子进程在单独的进程组中运行、结束后整组结束，不留 GPU 进程；单项超过估计时间的 --timeout-factor 倍（至少 30 分钟）
视为超时。

输出（--out-dir，默认 <工作目录>/data/T71c/bench）：<模型标签>/<基准>.json（与 text_bench 的输出相同，IFEval 的打分为
ifeval.json）、各项日志 <模型标签>/<项目>.log、summary.md 与 summary.json（各基准的绝对分数、与参考模型之差、模型卡上的
分数，MMMLU 各语言，诊断量）、status.json 与 driver.log。summary 中 GPQA 只有汇总的比例，没有题目内容。

退出码：0 全部完成；1 有失败的项目；2 缺少基本输入；3 能做的都做完了，但有运行还没下载、GPQA 数据不在或 IFEval 没有打分；
143 / 130 收到 SIGTERM / SIGINT 后停下。

用法（在仓库根目录运行；--dry-run 只检查输入、列出计划和时间估计，不占用 GPU）：

    WORK_DIR=<工作目录> DATASET=<数据集目录> python -m pipeline.eval.bench_local --runs-root <运行目录> \\
        --py-ifeval <有 langdetect 与 nltk 的 python> --dry-run
    WORK_DIR=<工作目录> DATASET=<数据集目录> nohup setsid python -m pipeline.eval.bench_local --runs-root <运行目录> \\
        --py-ifeval <有 langdetect 与 nltk 的 python> >> <工作目录>/data/T71c/bench/launch.out 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from pipeline.colab import c_runs
from pipeline.eval import c71_local
from pipeline.eval.text_bench import CARD, DATA, LETTERS, MMMLU_FILES

REPO = Path(__file__).resolve().parents[2]
REF = "ref"
SIMPLE = ("mmlu_redux", "ceval", "mmmlu", "ifeval")
BENCH_CHOICES = SIMPLE + ("gpqa",)
LM_FILE = f"{c71_local.CHECKPOINT}/language_model.safetensors"
# RTX 5090 上的估计用时（分钟），不含从运行目录载入语言模型（c71_local.LM_LOAD_MINUTES）：
# - 载入融合模型约 1 分钟；
# - 三项选择题共约 137 万个提示 token，只做前向；IFEval 541 条贪心生成，按剩余显存约 2 批、每批最多 1,280 步；
# - GPQA：198 题、每题最多 16,384 个新 token，按剩余显存分成 3 批、每批 66 条（与思考检查相同；提示中位数约 218 个
#   token，与思考检查相近）。几乎每批都有样本写到上限，一批的用时由上限决定：参考模型的思考检查（2026-11-24 本机，
#   flash_attention_2，67 条一批）实测每批约 829 秒，比 c71_local.think_minutes 的估计（每批约 11 分钟）长。
LOAD_MINUTES = 1.0
SIMPLE_MINUTES = {"mmlu_redux": 1.0, "ceval": 0.5, "mmmlu": 1.5, "ifeval": 4.0}
GPQA_BATCHES = 3
THINK_BATCH_SECONDS = 829
IFEVAL_SCORE_TIMEOUT = 30 * 60


class Outputs:
    """几个结果文件合起来的完成标志（与 c71_local.Driver 中 marker 的用法相同：有 exists()）。"""

    def __init__(self, paths):
        self.paths = [Path(path) for path in paths]

    def exists(self):
        return all(path.exists() for path in self.paths)

    def __str__(self):
        return ", ".join(str(path) for path in self.paths)


def gpqa_minutes(max_new_tokens):
    """GPQA 一个模型的估计用时（分钟）：按实测的每批用时，与生成上限成正比。"""
    return LOAD_MINUTES + GPQA_BATCHES * THINK_BATCH_SECONDS / 60 * max_new_tokens / 16384


def bench_output(bench, label_dir):
    """一项基准做完后 text_bench 写出的标志文件（IFEval 为生成结果）。"""
    return Path(label_dir) / ("ifeval.responses.jsonl" if bench == "ifeval" else f"{bench}.json")


def ready_runs(runs_root, keys):
    """各运行的语言模型权重：{key: 路径}（只含下载完整的）与 {key: 缺少的原因}。"""
    ready = {}
    missing = {}
    for key in keys:
        path = Path(runs_root) / c_runs.run_name(key = key) / LM_FILE
        ok, reason = c71_local.safetensors_complete(path = path)
        if ok:
            ready[key] = path
        else:
            missing[key] = f"{path}（{reason}）"
    return ready, missing


def label_entries(paths, label, language_model, benches):
    """一个模型的项目：四项简单基准合成一项（simple），GPQA 单独一项（gpqa）。"""
    label_dir = paths["out_dir"] / label
    base = [
        str(paths["python_text"]), "-m", "pipeline.eval.text_bench",
    ]
    common = [
        "--model-dir", str(paths["model_dir"]),
        "--data-dir", str(paths["data_dir"]),
        "--out-dir", str(label_dir),
        "--label", label,
    ]
    if language_model is not None:
        common += ["--language-model", str(language_model)]
    load = LOAD_MINUTES + (0.0 if language_model is None else c71_local.LM_LOAD_MINUTES)
    entries = []
    simple = [bench for bench in SIMPLE if bench in benches]
    if simple:
        entries.append({
            "label": label,
            "item": "simple",
            "benches": simple,
            "command": base + simple + common,
            "marker": Outputs(paths = [bench_output(bench = bench, label_dir = label_dir) for bench in simple]),
            "minutes": load + sum(SIMPLE_MINUTES[bench] for bench in simple),
            "log": label_dir / "simple.log",
        })
    if "gpqa" in benches:
        entries.append({
            "label": label,
            "item": "gpqa",
            "benches": ["gpqa"],
            "command": base + ["gpqa"] + common + ["--gpqa-max-new-tokens", str(paths["gpqa_max_new"])],
            "marker": Outputs(paths = [bench_output(bench = "gpqa", label_dir = label_dir)]),
            "minutes": gpqa_minutes(max_new_tokens = paths["gpqa_max_new"]) + (load - LOAD_MINUTES),
            "log": label_dir / "gpqa.log",
        })
    return entries


def build_plan(paths, ready, keys, include_ref, benches):
    """全部项目的顺序表：先每个模型的简单基准（参考模型在前），再每个模型的 GPQA。"""
    models = ([(REF, None)] if include_ref else []) + [(key, ready[key]) for key in keys if key in ready]
    per_label = [
        label_entries(
            paths = paths,
            label = label,
            language_model = language_model,
            benches = benches,
        )
        for label, language_model in models
    ]
    plan = []
    for item in ("simple", "gpqa"):
        plan += [entry for entries in per_label for entry in entries if entry["item"] == item]
    return plan


def ifeval_python_problem(python):
    """--py-ifeval 能否打分：能时为 None，否则为原因。"""
    if python is None:
        return "没有给 --py-ifeval"
    if not Path(python).exists():
        return f"{python} 不存在"
    result = subprocess.run(
        args = [str(python), "-c", "import langdetect, nltk; nltk.data.find('tokenizers/punkt_tab')"],
        capture_output = True,
        text = True,
        cwd = REPO,
        check = False,
    )
    if result.returncode != 0:
        lines = result.stderr.strip().splitlines()
        return f"{python} 缺少 langdetect、nltk 或 punkt_tab：{lines[-1][:200] if lines else result.returncode}"
    return None


def score_ifeval(paths, label, log):
    """IFEval 打分（CPU，不持锁）。返回 "ok"、"skipped"（已有结果或还没有生成结果）或失败说明。"""
    label_dir = paths["out_dir"] / label
    responses = label_dir / "ifeval.responses.jsonl"
    output = label_dir / "ifeval.json"
    if output.exists() or not responses.exists():
        return "skipped"
    if paths.get("ifeval_problem"):
        return f"未打分（{paths['ifeval_problem']}）"
    command = [
        str(paths["python_ifeval"]), "-m", "pipeline.eval.ifeval.score",
        "--input", str(paths["data_dir"] / DATA["ifeval"]),
        "--responses", str(responses),
        "--output", str(output),
    ]
    with open(file = label_dir / "ifeval_score.log", mode = "a", encoding = "utf-8") as handle:
        handle.write(f"\n# {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(command)}\n")
        handle.flush()
        try:
            code = subprocess.run(
                args = command,
                cwd = REPO,
                stdin = subprocess.DEVNULL,
                stdout = handle,
                stderr = subprocess.STDOUT,
                timeout = IFEVAL_SCORE_TIMEOUT,
                check = False,
            ).returncode
        except subprocess.TimeoutExpired:
            return "timeout"
    result = "ok" if code == 0 and output.exists() else f"failed（退出码 {code}）"
    log(f"{label} IFEval 打分：{'完成' if result == 'ok' else result}，日志 {label_dir / 'ifeval_score.log'}")
    return result


# ---------------------------------------------------------------- 汇总


def load_results(out_dir, labels):
    """{标签: {基准: 汇总 JSON}}，IFEval 为打分结果 ifeval.json。"""
    results = {}
    for label in labels:
        found = {}
        for bench in BENCH_CHOICES:
            data = c71_local.load_json(path = Path(out_dir) / label / f"{bench}.json")
            if data is not None:
                found[bench] = data
        results[label] = found
    return results


def percent(value, reference = None):
    """百分数（一位小数），给出参考模型的值时括号中为与它之差（百分点）。"""
    if value is None:
        return ""
    text = f"{100 * value:.1f}"
    if reference is None:
        return text
    return f"{text}（{100 * (value - reference):+.1f}）"


# 主表的列：(列名, 基准, 取值的键, 模型卡上的分数)
MAIN_COLUMNS = (
    ("MMLU-Redux", "mmlu_redux", "accuracy", CARD["mmlu_redux"]),
    ("C-Eval（val）", "ceval", "accuracy", CARD["ceval"]),
    ("MMMLU（10 种语言）", "mmmlu", "accuracy", CARD["mmmlu"]),
    ("IFEval prompt strict", "ifeval", "prompt_strict", CARD["ifeval"]),
    ("IFEval inst strict", "ifeval", "inst_strict", None),
    ("IFEval prompt loose", "ifeval", "prompt_loose", None),
    ("IFEval inst loose", "ifeval", "inst_loose", None),
    ("GPQA Diamond（思考）", "gpqa", "accuracy", CARD["gpqa"]),
)


def value_of(results, label, bench, key):
    return (results.get(label, {}).get(bench) or {}).get(key)


def summary_lines(results, labels):
    """summary.md 的内容。"""
    lines = [
        "# #71 文本基准汇总（本机）",
        "",
        "分数为百分数，括号中为与参考模型（ref，原始语言模型）之差（百分点）。模型卡一行是 Qwen3.5-2B 模型卡上的分数，设置不同"
        "（见下方说明），只作大致参照。",
        "",
        "| 模型 | " + " | ".join(title for title, _, _, _ in MAIN_COLUMNS) + " |",
        "| --- |" + " ---: |" * len(MAIN_COLUMNS),
        "| 模型卡 | " + " | ".join("" if card is None else f"{card:.1f}" for _, _, _, card in MAIN_COLUMNS) + " |",
    ]
    for label in labels:
        cells = []
        for _, bench, key, _ in MAIN_COLUMNS:
            value = value_of(results = results, label = label, bench = bench, key = key)
            reference = None if label == REF else value_of(results = results, label = REF, bench = bench, key = key)
            cells.append(percent(value = value, reference = reference))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    langs = list(MMMLU_FILES)
    lines += [
        "",
        "## MMMLU 各语言",
        "",
        "| 模型 | " + " | ".join(langs) + " |",
        "| --- |" + " ---: |" * len(langs),
    ]
    for label in labels:
        groups = (results.get(label, {}).get("mmmlu") or {}).get("groups", {})
        reference = (results.get(REF, {}).get("mmmlu") or {}).get("groups", {}) if label != REF else {}
        cells = [
            percent(
                value = (groups.get(lang) or {}).get("accuracy"),
                reference = (reference.get(lang) or {}).get("accuracy") if reference else None,
            )
            for lang in langs
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 诊断",
        "",
        "选择题的字母概率和为 A–D 四个 token 在全词表中的概率之和的平均（低说明模型不按「只回答字母」作答）；IFEval 为回答的"
        "平均 token 数与写到上限（1,280）的比例；GPQA 为写完 </think> 的比例、写完但找不到答案的比例、写到上限的比例、循环的"
        "比例和推理 token 数的中位数。",
        "",
        "| 模型 | 字母概率和 MMLU-Redux | C-Eval | MMMLU | IFEval 平均 token | 到上限 | GPQA 写完 </think> | 无答案 | 到上限 | 循环 | 推理中位 |",
        "| --- |" + " ---: |" * 10,
    ]
    for label in labels:
        found = results.get(label, {})
        ifeval = (found.get("ifeval") or {}).get("generation", {})
        gpqa = found.get("gpqa") or {}
        reasoning = (gpqa.get("reasoning_tokens") or {}).get("median")
        cells = [
            *(
                "" if (found.get(bench) or {}).get("letter_mass_mean") is None else f"{found[bench]['letter_mass_mean']:.3f}"
                for bench in ("mmlu_redux", "ceval", "mmmlu")
            ),
            "" if ifeval.get("mean_new_tokens") is None else f"{ifeval['mean_new_tokens']:.0f}",
            percent(value = ifeval.get("capped")),
            percent(value = gpqa.get("closed")),
            percent(value = gpqa.get("no_answer")),
            percent(value = gpqa.get("capped")),
            percent(value = gpqa.get("loop")),
            "" if reasoning is None else f"{reasoning:.0f}",
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 说明",
        "",
        "- 选择题都是零样本、不思考（chat template，enable_thinking = False），不生成，比较助手回合第一个位置上 A–D 四个 token "
        "的对数概率。MMLU-Redux 排除数据集标为有错的 278 题，用其余 2,722 题；C-Eval 用验证集（测试集答案不公开，模型卡很可能"
        "用测试集）；MMMLU 只用与项目重合的 10 种语言、每种同一组 500 题（模型卡为全部 14 种语言）。模型卡的非思考分数是生成"
        "回答后判定的（温度 0.7、top_p 0.8、top_k 20、presence_penalty 1.5），与字母对数概率的分数不能直接比较。",
        "- IFEval 为贪心解码、最多 1,280 个新 token，用 google-research 的官方打分逻辑；模型卡的 61.2 对应 prompt strict。",
        "- GPQA Diamond 为思考模式，温度 1、不设 top-k / top-p（与思考检查相同；模型卡为 top_p 0.95、top_k 20、presence_penalty "
        "1.5），最多 16,384 个新 token（模型卡建议 32,768），每题采样一次，198 题的标准误约 3.5 个百分点；没有写完 </think> "
        "或找不到答案都算错。",
    ]
    return lines


def write_summary(paths, labels):
    results = load_results(out_dir = paths["out_dir"], labels = labels)
    present = [label for label in labels if results.get(label)]
    (paths["out_dir"] / "summary.md").write_text(data = "\n".join(summary_lines(results = results, labels = present)) + "\n", encoding = "utf-8")
    compact = {
        label: {
            bench: {key: value for key, value in data.items() if key not in ("tiers", "pred_share")}
            for bench, data in found.items()
        }
        for label, found in results.items()
        if found
    }
    data = {"labels": present, "card": CARD, "letters": list(LETTERS), "results": compact}
    (paths["out_dir"] / "summary.json").write_text(data = json.dumps(obj = data, ensure_ascii = False, indent = 1) + "\n", encoding = "utf-8")


# ---------------------------------------------------------------- 输入检查、演习与入口


def check_inputs(paths, benches):
    """基本输入的检查：[(是否通过, 说明, 是否必需)]。GPQA 数据与 IFEval 打分用的 python 不是必需的。"""
    checks = []
    model_dir = Path(paths["model_dir"])
    tokenizer_class = (c71_local.load_json(path = model_dir / "tokenizer_config.json") or {}).get("tokenizer_class")
    model_ok = (model_dir / "model.safetensors").exists() and tokenizer_class == c71_local.TOKENIZER_CLASS
    checks.append((model_ok, f"模型目录 {model_dir}：分词器 {tokenizer_class}（应为 {c71_local.TOKENIZER_CLASS}，#77）", True))
    for bench in benches:
        path = paths["data_dir"] / DATA[bench]
        checks.append((path.exists(), f"{bench} 数据 {path}", bench != "gpqa"))
    python = Path(paths["python_text"])
    ok = python.exists()
    detail = "不存在"
    if ok:
        result = subprocess.run(
            args = [str(python), "-c", "import torch, transformers, safetensors, pyarrow, fla"],
            capture_output = True,
            text = True,
            cwd = REPO,
            env = dict(os.environ, CUDA_VISIBLE_DEVICES = ""),
            check = False,
        )
        ok = result.returncode == 0
        detail = "import torch, transformers, safetensors, pyarrow, fla：" + ("通过" if ok else result.stderr.strip().splitlines()[-1][:200])
    checks.append((ok, f"python（text_bench）{python}：{detail}", True))
    if "ifeval" in benches:
        problem = paths.get("ifeval_problem")
        checks.append((problem is None, f"IFEval 打分用的 python {paths['python_ifeval']}：" + (problem or "langdetect、nltk、punkt_tab 齐全"), False))
    return checks


def dry_run(paths, keys, include_ref, benches):
    print("#71 文本基准的本机驱动程序：演习（只检查和列出计划，不占用 GPU）")
    for name in ("model_dir", "data_dir", "runs_root", "out_dir", "python_text", "python_ifeval"):
        print(f"  {name}: {paths[name]}")
    print("\n输入检查")
    checks = check_inputs(paths = paths, benches = benches)
    for ok, text, required in checks:
        print(f"  {'通过' if ok else ('缺少' if required else '跳过')}  {text}")
    ready, missing = ready_runs(runs_root = paths["runs_root"], keys = keys)
    print("\n运行（语言模型权重）")
    for key in keys:
        print(f"  {key:<12} {'齐全' if key in ready else '缺少 ' + missing[key]}")
    all_ready = {key: Path(paths["runs_root"]) / c_runs.run_name(key = key) / LM_FILE for key in keys}
    plan_now = build_plan(
        paths = paths,
        ready = ready,
        keys = keys,
        include_ref = include_ref,
        benches = benches,
    )
    plan_full = build_plan(
        paths = paths,
        ready = all_ready,
        keys = keys,
        include_ref = include_ref,
        benches = benches,
    )
    now_ids = {(entry["label"], entry["item"]) for entry in plan_now}
    print("\n计划（已有结果的项目跳过）")
    for entry in plan_full:
        if entry["marker"].exists():
            state = "已有结果，跳过"
        elif (entry["label"], entry["item"]) in now_ids:
            state = "运行"
        else:
            state = "等待下载"
        print(f"  {entry['item']:<7} {entry['label']:<12} {state:<10} 约 {entry['minutes']:.0f} 分钟")
    full = sum(entry["minutes"] for entry in plan_full if not entry["marker"].exists())
    per_model = {item: sum(entry["minutes"] for entry in plan_full if entry["item"] == item and entry["label"] == (keys[0] if keys else REF)) for item in ("simple", "gpqa")}
    print(
        f"\n时间估计（RTX 5090）：每个运行简单基准约 {per_model['simple']:.0f} 分钟、GPQA 约 {per_model['gpqa']:.0f} 分钟；"
        f"全部 {len([e for e in plan_full if not e['marker'].exists()])} 项约 {full / 60:.1f} 小时"
    )
    print("\n各项的命令（以第一个运行为例；参考模型不给 --language-model）")
    shown = set()
    for entry in plan_full:
        if entry["item"] not in shown and entry["label"] != REF:
            shown.add(entry["item"])
            print(f"  [{entry['item']}] " + " ".join(entry["command"]))
    return 0 if all(ok for ok, _, required in checks if required) else 2


def resolve_paths(args):
    work = Path(args.work_dir)
    out_dir = Path(args.out_dir) if args.out_dir else work / "data" / "T71c" / "bench"
    paths = {
        "work": work,
        "model_dir": Path(args.model_dir),
        "data_dir": Path(args.data_dir) if args.data_dir else Path(args.dataset or "") / "text",
        "runs_root": Path(args.runs_root) if args.runs_root else work / "data" / "T71c" / "runs",
        "out_dir": out_dir,
        "python_text": Path(args.py_text) if args.py_text else work / "envs" / "mllm-train" / "bin" / "python",
        "python_ifeval": Path(args.py_ifeval) if args.py_ifeval else None,
        "gpu_lock": Path(args.gpu_lock) if args.gpu_lock else work / "gpu.lock",
        "gpqa_max_new": args.gpqa_max_new_tokens,
    }
    paths["ifeval_problem"] = ifeval_python_problem(python = paths["python_ifeval"]) if "ifeval" in args.bench_list else None
    return paths


def parse_args(argv = None):
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", default = os.environ.get("WORK_DIR", str(REPO.parent / "qwen3.5-asr-work")), help = "工作目录，默认环境变量 WORK_DIR")
    parser.add_argument("--dataset", default = os.environ.get("DATASET"), help = "数据集目录，默认环境变量 DATASET（基准数据在其下的 text/）")
    parser.add_argument("--data-dir", default = None, help = "文本数据集目录，默认 <数据集目录>/text")
    parser.add_argument("--model-dir", default = os.environ.get("MODEL_DIR", str(REPO / "qwen3.5-2b-mllm")), help = "2B 融合模型目录（分词器已按 #77 修正）")
    parser.add_argument(
        "--runs-root",
        default = os.environ.get("C71_RUNS_ROOT"),
        help = "C 组运行目录的上级，默认环境变量 C71_RUNS_ROOT，再没有时为 <工作目录>/data/T71c/runs",
    )
    parser.add_argument("--out-dir", default = None, help = "输出目录，默认 <工作目录>/data/T71c/bench")
    parser.add_argument("--py-text", default = None, help = "text_bench 用的 python，默认 <工作目录>/envs/mllm-train/bin/python")
    parser.add_argument(
        "--py-ifeval",
        default = os.environ.get("IFEVAL_PYTHON"),
        help = "IFEval 打分用的 python（需要 langdetect、nltk ≥ 3.9.1 与 punkt_tab 数据），默认环境变量 IFEVAL_PYTHON",
    )
    parser.add_argument("--gpu-lock", default = None, help = "GPU 锁文件，默认 <工作目录>/gpu.lock")
    parser.add_argument("--runs", default = ",".join(c_runs.RUNS), help = "逗号分隔的运行，默认全部六个")
    parser.add_argument("--no-ref", action = "store_true", help = "不评测参考模型")
    parser.add_argument("--benches", default = ",".join(BENCH_CHOICES), help = "逗号分隔的基准，默认全部")
    parser.add_argument("--gpqa-max-new-tokens", type = int, default = 16384, help = "GPQA 的生成上限（各模型要用同一个）")
    parser.add_argument("--busy-mib", type = int, default = 3000, help = "已用显存超过它时等待（MiB）")
    parser.add_argument("--timeout-factor", type = float, default = 3.0, help = "单项超过估计用时的这么多倍（至少 30 分钟）时结束它")
    parser.add_argument("--dry-run", action = "store_true", help = "只检查输入、列出计划与时间估计")
    args = parser.parse_args(args = argv)
    args.keys = [key for key in args.runs.split(",") if key]
    unknown = [key for key in args.keys if key not in c_runs.RUNS]
    if unknown:
        parser.error(f"未知的运行：{unknown}，可选 {list(c_runs.RUNS)}")
    args.bench_list = [bench for bench in args.benches.split(",") if bench]
    unknown = [bench for bench in args.bench_list if bench not in BENCH_CHOICES]
    if unknown:
        parser.error(f"未知的基准：{unknown}，可选 {list(BENCH_CHOICES)}")
    if not args.dataset and not args.data_dir:
        parser.error("需要 --dataset（或环境变量 DATASET）或 --data-dir")
    return args


def main(argv = None):
    args = parse_args(argv = argv)
    paths = resolve_paths(args = args)
    keys = args.keys
    include_ref = not args.no_ref
    if args.dry_run:
        return dry_run(paths = paths, keys = keys, include_ref = include_ref, benches = args.bench_list)
    paths["out_dir"].mkdir(parents = True, exist_ok = True)
    driver = c71_local.Driver(paths = paths, busy_mib = args.busy_mib, timeout_factor = args.timeout_factor)

    def on_signal(signum, frame):
        raise c71_local.Stop(signum = signum)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    driver.set_status(pid = os.getpid(), started = time.strftime("%Y-%m-%d %H:%M:%S"), phase = "start", current = None, benches = args.bench_list)
    driver.log(f"开始（pid {os.getpid()}，基准 {args.bench_list}，运行 {keys}，参考模型 {'评测' if include_ref else '不评测'}）")
    checks = check_inputs(paths = paths, benches = args.bench_list)
    failed_inputs = [text for ok, text, required in checks if required and not ok]
    if failed_inputs:
        driver.log(f"缺少基本输入：{failed_inputs}")
        driver.set_status(phase = "error", error = failed_inputs)
        return 2
    notes = [text for ok, text, required in checks if not required and not ok]
    for text in notes:
        driver.log(f"跳过：{text}")
    benches = [bench for bench in args.bench_list if bench != "gpqa" or (paths["data_dir"] / DATA["gpqa"]).exists()]
    failures = []
    labels = ([REF] if include_ref else []) + keys
    try:
        for item in ("simple", "gpqa"):
            ready, missing = ready_runs(runs_root = paths["runs_root"], keys = keys)
            driver.set_status(runs_ready = list(ready), runs_missing = missing)
            plan = build_plan(
                paths = paths,
                ready = ready,
                keys = keys,
                include_ref = include_ref,
                benches = benches,
            )
            entries = [entry for entry in plan if entry["item"] == item]
            pending = [entry["label"] for entry in entries if not entry["marker"].exists()]
            driver.log(f"{item}：要做 {pending or '无'}；估计全部剩余 {driver.remaining_minutes(plan = plan) / 60:.1f} 小时")
            for entry in entries:
                result = driver.run_entry(entry = entry, plan = plan)
                if result not in ("ok", "skipped"):
                    failures.append(f"{entry['label']}:{entry['item']}")
                if item == "simple" and "ifeval" in entry["benches"]:
                    scored = score_ifeval(paths = paths, label = entry["label"], log = driver.log)
                    if scored not in ("ok", "skipped"):
                        driver.record(entry = dict(entry, item = "ifeval_score", log = paths["out_dir"] / entry["label"] / "ifeval_score.log"), result = scored)
                        if not scored.startswith("未打分"):
                            failures.append(f"{entry['label']}:ifeval_score")
                write_summary(paths = paths, labels = labels)
    except c71_local.Stop as stop:
        driver.kill_child()
        driver.release_lock()
        driver.log(f"停下：{stop}；重新运行驱动程序即可续跑")
        driver.set_status(phase = "stopped", current = None)
        return 128 + stop.signum
    except Exception as error:
        driver.kill_child()
        driver.release_lock()
        driver.log(f"出错：{error!r}")
        driver.set_status(phase = "error", error = repr(error), current = None)
        raise
    write_summary(paths = paths, labels = labels)
    ready, missing = ready_runs(runs_root = paths["runs_root"], keys = keys)
    summary = paths["out_dir"] / "summary.md"
    if failures:
        driver.log(f"结束，有失败的项目：{failures}（重新运行只重做这些）；汇总 {summary}")
        driver.set_status(phase = "done_with_failures", failed = failures, current = None)
        return 1
    unscored = "ifeval" in benches and paths.get("ifeval_problem") is not None
    if missing or notes or unscored:
        driver.log(f"能做的都做完了；还没下载的运行：{list(missing) or '无'}；跳过：{notes or '无'}；汇总 {summary}")
        driver.set_status(phase = "done_partial", current = None)
        return 3
    driver.log(f"全部完成；汇总 {summary}")
    driver.set_status(phase = "done", current = None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
