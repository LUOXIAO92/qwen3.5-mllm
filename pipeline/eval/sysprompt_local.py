"""#71 系统提示遵循评测的本机驱动程序：参考模型与 C 组 13 个运行，文本与语音两部分，全部用 vLLM 的 docker 镜像。

用户 2026-11-27：C 组的训练数据没有系统回合（ASR 为不套模板的续写格式，回放的对话组从 `<|im_start|>user` 开始），已有
评测也都把要求写在用户回合；这里看全量微调后的模型是否对系统消息变得不敏感。

每个模型（标签 ref 或运行名）依次做：

1. text：pipeline/eval/sysprompt_bench.py 的 generate（294 条中性问题各加一个 IFEval 约束，三种条件：A 无约束、B 约束在
   系统消息、C 约束接在用户消息之后），做完接着用 --py-ifeval（有 langdetect 与 nltk 的 python，只用 CPU、不持 GPU 锁）
   打分；
2. audio_system：pipeline/eval/instruct_probe.py 的严格版指令（--instruction-set strict），只做转写与翻译（英、中两个目标
   语言），指令放在系统消息中、用户回合只有音频（--instruction-placement system）；样本与已有的严格版探针相同（18 种语言，
   每种按种子 0 抽 30 条），批大小 16；
3. 参考模型另做 audio_user（同样的命令，指令在用户回合中音频之后）。参考模型的语音用 --ref-projector（默认 projector 只训练时
   第 2,700 步的检查点，C 组各运行全量微调的起点）加原始语言模型；它此前没有严格版探针的结果。--rerun-user 点名的运行也重做
   audio_user，用来核对与已有结果的一致程度。

各运行指令在用户回合的结果直接取已有的严格版探针 <--user-probe-root>/<运行>/probe_strict/summary.json（c71_local 用 vLLM 做的，
同样的样本与指令，另含时间段一项），不重做。

模型与导出：参考模型直接加载 --ref-lm（已导出的参考模型目录）；每个运行在第一项时把语言模型导出到 --vllm-export-root/<运行>
（约 3.8 GB，配置与分词器取自 --ref-lm），这个运行的各项共用，最后一项结束（或失败）后删除。任务目录为
--vllm-job-root/<标签>-<项目>，完成后删除。容器中的 HOME（编译缓存）默认与 c71_local 共用。

GPU：使用 c71_local.Driver：每一项开始前持有 <工作目录>/gpu.lock，已用显存超过 --busy-mib（默认 3,000 MiB，例如 ComfyUI）
时每 5 分钟再看一次，不结束别人的进程；子进程在单独的进程组中运行，结束后（包括超时、收到 SIGTERM / SIGINT）整组结束。
已有结果的项目跳过，单项失败记下后接着做。

输出（--out-dir，默认 <工作目录>/data/T71e/sysprompt）：<标签>/text/（responses、generation、scored、scores）、
<标签>/audio_system/、ref/audio_user/、各项日志 <标签>/<项目>.log、summary.md 与 summary.json（全部模型的表格）、
status.json、driver.log。

    WORK_DIR=<工作目录> python -m pipeline.eval.sysprompt_local --runs-root <运行目录> --py-ifeval <python> --dry-run
    WORK_DIR=<工作目录> nohup setsid <工作目录>/envs/mllm-eval/bin/python -m pipeline.eval.sysprompt_local \\
        --runs-root <运行目录> --py-ifeval <python> >> <输出目录>/launch.out 2>&1 < /dev/null &

只汇总（不占用 GPU）：--summary-only。
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
from pipeline.eval.sysprompt_bench import CONDITIONS, INSTRUCTION_TYPES, ITEMS_PATH, outcome_table, paired_difference, read_jsonl

REPO = Path(__file__).resolve().parents[2]
REF = "ref"
# 语音的两项共用的探针参数（与已有的严格版探针相同，只是不做时间段）。
PROBE_ARGS = ["--batch-size", "16", "--prompt-format", "chat", "--instruction-set", "strict", "--tasks", "transcribe", "translate"]
# RTX 5090 上的估计用时（分钟）：参考模型的文本（882 条，含启动容器与载入）实测 1.1；严格版探针 3,180 条实测 2.5，这里 2,100 条。
TEXT_MINUTES = 1.5
AUDIO_MINUTES = 2.5
# 汇总中语音的各列：(键, 任务, 指令编号, 指标)。
AUDIO_FIELDS = (
    ("transcribe_en", "transcribe", "transcribe_strict", "rate_basic"),
    ("transcribe_zh", "transcribe", "transcribe_strict_zh", "rate_basic"),
    ("transcribe_en_empty", "transcribe", "transcribe_strict", "empty_share"),
    ("transcribe_zh_empty", "transcribe", "transcribe_strict_zh", "empty_share"),
    ("translate_en_chrf", "translate", "translate_en_strict", "chrf"),
    ("translate_en_copy", "translate", "translate_en_strict", "verbatim_share"),
    ("translate_zh_chrf", "translate", "translate_zh_strict", "chrf"),
    ("translate_zh_copy", "translate", "translate_zh_strict", "verbatim_share"),
)


def load_json(path):
    try:
        return json.loads(s = Path(path).read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------- 计划


def label_entries(label, run, paths, rerun_user = False):
    """一个模型的评测项目：[{label, item, command, marker, minutes, log}]。run 为 c71_local.discover_runs 的一项，参考模型为 None。"""
    out = paths["out_dir"] / label
    checkpoint = None if run is None else run["dir"] / c71_local.CHECKPOINT
    lm_args = [] if checkpoint is None else ["--language-model", str(checkpoint / "language_model.safetensors")]
    projector = paths["ref_projector"] if checkpoint is None else checkpoint / "projector.safetensors"

    def vllm(item):
        return c71_local.vllm_arguments(paths = paths, label = label, run = run, item = item)

    entries = [{
        "item": "text",
        "command": [
            str(paths["python_text"]), "-m", "pipeline.eval.sysprompt_bench", "generate", "--model-dir", str(paths["model_dir"]),
            *lm_args, "--out-dir", str(out / "text"), "--label", label, *vllm(item = "text"),
        ],
        "marker": out / "text" / "responses.jsonl",
        "minutes": TEXT_MINUTES,
    }]
    placements = [("audio_user", "user")] if run is None or rerun_user else []
    placements.append(("audio_system", "system"))
    for item, placement in placements:
        entries.append({
            "item": item,
            "command": [
                str(paths["python_audio"]), "-m", "pipeline.eval.instruct_probe", "--model-dir", str(paths["model_dir"]),
                "--projector", str(projector), *lm_args, "--manifest-dir", str(paths["fleurs"]), "--output-dir", str(out / item),
                *PROBE_ARGS, "--instruction-placement", placement, *vllm(item = item),
            ],
            "marker": out / item / "summary.json",
            "minutes": AUDIO_MINUTES,
        })
    if run is not None:
        # 运行的语言模型在第一项时导出，后面的项目共用
        entries[0]["minutes"] += c71_local.VLLM_EXPORT_MINUTES + c71_local.LM_LOAD_MINUTES
    for entry in entries:
        entry["label"] = label
        entry["log"] = out / f"{entry['item']}.log"
    return entries


def build_plan(runs, keys, paths, include_ref = True, rerun_user = ()):
    """{标签: 项目列表}，参考模型在前，运行按 keys 的顺序（只含两个必需文件齐全的运行）。"""
    plan = {}
    if include_ref:
        plan[REF] = label_entries(label = REF, run = None, paths = paths)
    for key in keys:
        if runs.get(key, {}).get("ready"):
            plan[key] = label_entries(
                label = key,
                run = runs[key],
                paths = paths,
                rerun_user = key in rerun_user,
            )
    return plan


def score_text(paths, label, log):
    """用 --py-ifeval 给一个模型的文本回答打分（只用 CPU，不持 GPU 锁）；scores.json 比 responses.jsonl 新时跳过。"""
    out = paths["out_dir"] / label / "text"
    responses = out / "responses.jsonl"
    scores = out / "scores.json"
    if not responses.exists():
        return "no_responses"
    if scores.exists() and scores.stat().st_mtime >= responses.stat().st_mtime:
        return "skipped"
    with open(file = paths["out_dir"] / label / "text_score.log", mode = "a", encoding = "utf-8") as handle:
        code = subprocess.run(
            args = [str(paths["python_ifeval"]), "-m", "pipeline.eval.sysprompt_bench", "score", "--out-dir", str(out)],
            cwd = REPO,
            stdin = subprocess.DEVNULL,
            stdout = handle,
            stderr = subprocess.STDOUT,
            env = dict(os.environ, CUDA_VISIBLE_DEVICES = ""),
            check = False,
        ).returncode
    result = "ok" if code == 0 and scores.exists() else f"failed（退出码 {code}）"
    log(f"{label} 文本打分：{'完成' if result == 'ok' else result}")
    return result


# ---------------------------------------------------------------- 汇总


def audio_values(summary):
    """严格版探针 summary.json 中汇总用的各项（各语言平均，ALL）与条数：({键: 数值}, {键: 条数})。"""
    values, counts = {}, {}
    for key, task, instruction, metric in AUDIO_FIELDS:
        group = ((summary or {}).get(task) or {}).get(instruction) or {}
        values[key] = (group.get("ALL") or {}).get(metric)
        counts[key] = (group.get("ALL") or {}).get("n")
    return values, counts


def audio_sources(paths, label):
    """一个模型两种位置的语音结果：{"user": (summary, 路径), "system": ..., "user_rerun": ...}（没有的为 None）。"""
    out = paths["out_dir"] / label
    sources = {"system": out / "audio_system" / "summary.json"}
    if label == REF:
        sources["user"] = out / "audio_user" / "summary.json"
    else:
        sources["user"] = paths["user_probe_root"] / label / "probe_strict" / "summary.json"
        sources["user_rerun"] = out / "audio_user" / "summary.json"
    return {name: (load_json(path = path), path) for name, path in sources.items()}


def gap_difference(table, ref_table, ids):
    """(B − C)_模型 − (B − C)_参考模型（strict）：逐条算 (B − C) 之差（取值 −2 到 2）再平均，给出差与标准误。为负时，
    系统消息比用户消息少照做的部分比参考模型多，即对系统消息的敏感度比一般的指令遵循退化得更多。"""
    values = [
        (int(table["B"][item_id][0]) - int(table["C"][item_id][0])) - (int(ref_table["B"][item_id][0]) - int(ref_table["C"][item_id][0]))
        for item_id in ids
    ]
    n = len(values)
    if n < 2:
        return {"diff": None, "se": None, "n": n}
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    return {"diff": mean, "se": (variance / n) ** 0.5, "n": n}


def text_entry(paths, label, ref_table):
    """一个模型文本部分的汇总：scores.json 的主要数值，加上 B、C、B − C 与参考模型的成对差（strict）。"""
    scores = load_json(path = paths["out_dir"] / label / "text" / "scores.json")
    if scores is None:
        return None
    entry = {
        "conditions": scores["conditions"],
        "b_minus_c": scores["b_minus_c"],
        "generation": scores["generation"],
        "by_type": {kind: {condition: values["strict"] for condition, values in group.items()} for kind, group in scores["by_type"].items()},
    }
    scored = paths["out_dir"] / label / "text" / "scored.jsonl"
    if ref_table is not None and scored.exists():
        table = outcome_table(rows = read_jsonl(path = scored))
        ids = sorted(ref_table["B"])
        for condition in ("B", "C"):
            entry[f"{condition}_minus_ref"] = paired_difference(
                first = [table[condition][item_id][0] for item_id in ids],
                second = [ref_table[condition][item_id][0] for item_id in ids],
            )
        entry["gap_minus_ref"] = gap_difference(table = table, ref_table = ref_table, ids = ids)
    return entry


def percent(value, digits = 1):
    return "n/a" if value is None else f"{100 * value:.{digits}f}"


def number(value, digits = 1):
    return "n/a" if value is None else f"{value:.{digits}f}"


def signed(diff, se = None):
    if diff is None:
        return "n/a"
    text = f"{100 * diff:+.1f}"
    return text if se is None else f"{text} ±{100 * se:.1f}"


def collect(paths, labels):
    """全部模型的汇总数据（summary.json 的内容）。"""
    ref_scored = paths["out_dir"] / REF / "text" / "scored.jsonl"
    ref_table = outcome_table(rows = read_jsonl(path = ref_scored)) if ref_scored.exists() else None
    data = {"labels": [], "text": {}, "audio": {}}
    for label in labels:
        text = text_entry(paths = paths, label = label, ref_table = ref_table)
        sources = audio_sources(paths = paths, label = label)
        audio = {}
        for name, (summary, path) in sources.items():
            if summary is None:
                continue
            values, counts = audio_values(summary = summary)
            audio[name] = {"values": values, "n": counts, "source": str(path), "engine": (summary.get("config") or {}).get("engine")}
        if text is None and not audio:
            continue
        data["labels"].append(label)
        if text is not None:
            data["text"][label] = text
        if audio:
            data["audio"][label] = audio
    return data


def summary_lines(data, paths):
    """summary.md 的各行。"""
    labels = data["labels"]
    items = read_jsonl(path = paths["items"])
    lines = [
        "# #71 系统提示遵循评测（vLLM）",
        "",
        f"文本：{len(items)} 条中性问题各加一个 IFEval 约束（{len(INSTRUCTION_TYPES)} 类，见 pipeline/eval/sysprompt_bench.py），"
        "非思考模式贪心生成，最多 1,280 个新 token。A 为不提要求（碰巧满足的比例），B 为约束在系统消息、问题在用户消息，C 为没有"
        "系统消息、约束接在问题之后。数值为约束满足率（%），strict / loose 定义同 IFEval；B − C 与「比参考模型」都是同一组条目上的"
        "成对差（百分点）及其标准误。「B − C 比参考模型」为 (B − C) 减去参考模型的 (B − C)（strict，逐条成对）：为负时系统消息比"
        "用户消息少照做的部分比参考模型多，即系统消息的效力比一般的指令遵循退化得更多。",
        "",
        "| 模型 | A strict | A loose | B strict | B loose | C strict | C loose | B − C strict | B − C loose | B 比参考模型 | C 比参考模型 | B − C 比参考模型 | B 平均长度 |",
        "| --- |" + " ---: |" * 12,
    ]
    for label in labels:
        text = data["text"].get(label)
        if text is None:
            continue
        conditions = text["conditions"]
        gap = text["b_minus_c"]
        cells = [
            percent(value = conditions["A"]["strict"]),
            percent(value = conditions["A"]["loose"]),
            percent(value = conditions["B"]["strict"]),
            percent(value = conditions["B"]["loose"]),
            percent(value = conditions["C"]["strict"]),
            percent(value = conditions["C"]["loose"]),
            signed(diff = gap["strict"]["diff"], se = gap["strict"]["se"]),
            signed(diff = gap["loose"]["diff"], se = gap["loose"]["se"]),
            "—" if label == REF else signed(diff = (text.get("B_minus_ref") or {}).get("diff"), se = (text.get("B_minus_ref") or {}).get("se")),
            "—" if label == REF else signed(diff = (text.get("C_minus_ref") or {}).get("diff"), se = (text.get("C_minus_ref") or {}).get("se")),
            "—" if label == REF else signed(diff = (text.get("gap_minus_ref") or {}).get("diff"), se = (text.get("gap_minus_ref") or {}).get("se")),
            f"{text['generation']['B']['mean_new_tokens']:.0f}",
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    text_labels = [label for label in labels if label in data["text"]]
    if text_labels:
        lines += [
            "",
            "各类约束的满足率（strict，%），每格为 B / C（各 14 条）：",
            "",
            "| 约束 | " + " | ".join(text_labels) + " |",
            "| --- |" + " ---: |" * len(text_labels),
        ]
        for kind in INSTRUCTION_TYPES:
            cells = []
            for label in text_labels:
                group = data["text"][label]["by_type"].get(kind, {})
                cells.append(f"{percent(value = group.get('B'), digits = 0)} / {percent(value = group.get('C'), digits = 0)}")
            lines.append(f"| {kind} | " + " | ".join(cells) + " |")
    audio_labels = [label for label in labels if label in data["audio"]]
    if audio_labels:
        lines += [
            "",
            "语音：严格版指令（只输出答案），FLEURS test 18 种语言各 30 条，数值为各语言的平均。user 为指令在用户回合中音频之后（各运行取"
            "已有的 probe_strict 结果），system 为指令在系统消息、用户回合只有音频。参考模型的语音为原始语言模型加 projector 只训练时的"
            "检查点（全量微调的起点）。转写为基础规范化后的错误率（WER，中、日、韩、泰等按字符），翻译为 chrF 与照抄原文（输出几乎逐字"
            "是源语言转写）的比例。",
            "",
            "| 模型 | 转写·英文指令 user | system | 转写·中文指令 user | system | 译英 chrF user | system | 译英照抄 user | system | 译中 chrF user | system | 译中照抄 user | system |",
            "| --- |" + " ---: |" * 12,
        ]
        for label in audio_labels:
            audio = data["audio"][label]
            user = (audio.get("user") or {}).get("values") or {}
            system = (audio.get("system") or {}).get("values") or {}
            cells = []
            for key in ("transcribe_en", "transcribe_zh", "translate_en_chrf", "translate_en_copy", "translate_zh_chrf", "translate_zh_copy"):
                for values in (user, system):
                    value = values.get(key)
                    cells.append(number(value = value) if key.endswith("chrf") else percent(value = value))
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        reruns = [label for label in audio_labels if "user_rerun" in data["audio"][label]]
        if reruns:
            lines += [
                "",
                "重做的 user（与 system 同样只做转写和翻译、同样分批）与已有 probe_strict 的对照：",
                "",
                "| 模型 | 转写·英文指令 已有 / 重做 | 转写·中文指令 | 译英 chrF | 译中 chrF |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
            for label in reruns:
                old = data["audio"][label]["user"]["values"]
                new = data["audio"][label]["user_rerun"]["values"]
                lines.append(
                    f"| {label} | {percent(value = old.get('transcribe_en'))} / {percent(value = new.get('transcribe_en'))} | "
                    f"{percent(value = old.get('transcribe_zh'))} / {percent(value = new.get('transcribe_zh'))} | "
                    f"{number(value = old.get('translate_en_chrf'))} / {number(value = new.get('translate_en_chrf'))} | "
                    f"{number(value = old.get('translate_zh_chrf'))} / {number(value = new.get('translate_zh_chrf'))} |"
                )
    return lines


def write_summary(paths, labels, log = None):
    data = collect(paths = paths, labels = labels)
    data["items"] = str(paths["items"])
    data["conditions"] = list(CONDITIONS)
    paths["out_dir"].mkdir(parents = True, exist_ok = True)
    (paths["out_dir"] / "summary.json").write_text(data = json.dumps(obj = data, ensure_ascii = False, indent = 1) + "\n", encoding = "utf-8")
    (paths["out_dir"] / "summary.md").write_text(data = "\n".join(summary_lines(data = data, paths = paths)) + "\n", encoding = "utf-8")
    if log is not None:
        log(f"汇总 {paths['out_dir'] / 'summary.md'}（{len(data['labels'])} 个模型）")
    return data


# ---------------------------------------------------------------- 检查与运行


def check_inputs(paths, keys, runs):
    """基本输入的检查：[(是否通过, 说明)]。"""
    checks = []
    model_dir = Path(paths["model_dir"])
    checks.append(((model_dir / "model.safetensors").exists(), f"融合模型目录 {model_dir}"))
    checks.append((Path(paths["items"]).exists(), f"条目 {paths['items']}"))
    checks.append(((Path(paths["fleurs"]) / "en" / "test.jsonl").exists(), f"FLEURS 清单 {paths['fleurs']}"))
    checks.append((Path(paths["ref_projector"]).exists(), f"参考模型语音用的 projector {paths['ref_projector']}"))
    for role in ("text", "audio", "ifeval"):
        python = paths[f"python_{role}"]
        checks.append((python is not None and Path(python).exists(), f"python（{role}）{python}"))
    checks += c71_local.vllm_checks(paths = paths)
    missing_user = [key for key in keys if not (paths["user_probe_root"] / key / "probe_strict" / "summary.json").exists()]
    checks.append((not missing_user, f"已有的严格版探针 {paths['user_probe_root']}/<运行>/probe_strict" + (f"，缺少 {missing_user}" if missing_user else " 齐全")))
    not_ready = [key for key in keys if not runs[key]["ready"]]
    checks.append((not not_ready, f"运行目录 {paths['runs_root']}" + (f"，不齐全：{not_ready}" if not_ready else f"，{len(keys)} 个运行齐全")))
    return checks


def resolve_paths(args):
    work = Path(args.work_dir)
    out_dir = Path(args.out_dir) if args.out_dir else work / "data" / "T71e" / "sysprompt"
    return {
        "work": work,
        "out_dir": out_dir,
        "model_dir": Path(args.model_dir),
        "items": Path(args.items),
        "fleurs": Path(args.fleurs_dir) if args.fleurs_dir else work / "data" / "T02",
        "runs_root": Path(args.runs_root),
        "user_probe_root": Path(args.user_probe_root) if args.user_probe_root else work / "data" / "T71e" / "eval",
        "ref_projector": Path(args.ref_projector) if args.ref_projector else (
            work / "data" / "T71l" / "init" / "v2b-2b-cpt77-projector" / "checkpoint-0002700" / "projector.safetensors"
        ),
        "python_text": Path(args.py_text) if args.py_text else work / "envs" / "mllm-train" / "bin" / "python",
        "python_audio": Path(args.py_audio) if args.py_audio else work / "envs" / "mllm-eval" / "bin" / "python",
        "python_ifeval": Path(args.py_ifeval) if args.py_ifeval else None,
        "gpu_lock": Path(args.gpu_lock) if args.gpu_lock else work / "gpu.lock",
        "ref_lm": Path(args.ref_lm) if args.ref_lm else work / "data" / "T70" / "ref-lm",
        "vllm_image": args.vllm_image,
        "vllm_export_root": Path(args.vllm_export_root) if args.vllm_export_root else work / "tmp" / "T71e-sysprompt" / "vllm-lm",
        "vllm_job_root": Path(args.vllm_job_root) if args.vllm_job_root else work / "tmp" / "T71e-sysprompt" / "vllm-job",
        "vllm_home": Path(args.vllm_home) if args.vllm_home else work / "tmp" / "T71c" / "vllm-home",
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }


def parse_args(argv = None):
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", default = os.environ.get("WORK_DIR", str(REPO.parent / "qwen3.5-asr-work")), help = "工作目录，默认环境变量 WORK_DIR")
    parser.add_argument("--model-dir", default = os.environ.get("MODEL_DIR", str(REPO / "qwen3.5-2b-mllm")), help = "2B 融合模型目录")
    parser.add_argument("--runs-root", default = os.environ.get("C71_RUNS_ROOT"), help = "运行目录的上级，默认环境变量 C71_RUNS_ROOT")
    parser.add_argument("--runs", default = ",".join(c_runs.ALL_RUNS), help = "逗号分隔的运行，默认全部 13 个；空字符串时只做参考模型")
    parser.add_argument("--no-ref", action = "store_true", help = "不评测参考模型")
    parser.add_argument("--rerun-user", default = "", help = "逗号分隔的运行：另做指令在用户回合的语音探针，与已有结果核对")
    parser.add_argument("--out-dir", default = None, help = "输出目录，默认 <工作目录>/data/T71e/sysprompt")
    parser.add_argument("--items", default = str(ITEMS_PATH), help = "文本条目（sysprompt_bench build 的输出）")
    parser.add_argument("--fleurs-dir", default = None, help = "FLEURS 评测清单目录，默认 <工作目录>/data/T02")
    parser.add_argument("--user-probe-root", default = None, help = "已有严格版探针的上级目录（<运行>/probe_strict），默认 <工作目录>/data/T71e/eval")
    parser.add_argument("--ref-projector", default = None, help = "参考模型语音用的 projector，默认 projector 只训练时第 2,700 步的检查点")
    parser.add_argument("--py-text", default = None, help = "文本生成用的 python，默认 <工作目录>/envs/mllm-train/bin/python")
    parser.add_argument("--py-audio", default = None, help = "语音探针用的 python，默认 <工作目录>/envs/mllm-eval/bin/python")
    parser.add_argument("--py-ifeval", default = None, help = "文本打分用的 python（需要 langdetect 与 nltk）")
    parser.add_argument("--gpu-lock", default = None, help = "GPU 锁文件，默认 <工作目录>/gpu.lock")
    parser.add_argument("--ref-lm", default = None, help = "参考模型的语言模型目录，默认 <工作目录>/data/T70/ref-lm")
    parser.add_argument("--vllm-image", default = "vllm/vllm-openai:nightly-x86_64")
    parser.add_argument("--vllm-export-root", default = None, help = "各运行导出语言模型的上级目录，默认 <工作目录>/tmp/T71e-sysprompt/vllm-lm")
    parser.add_argument("--vllm-job-root", default = None, help = "任务目录的上级，默认 <工作目录>/tmp/T71e-sysprompt/vllm-job")
    parser.add_argument("--vllm-home", default = None, help = "容器中的 HOME（编译缓存），默认与 c71_local 共用 <工作目录>/tmp/T71c/vllm-home")
    parser.add_argument("--gpu-memory-utilization", type = float, default = 0.9)
    parser.add_argument("--busy-mib", type = int, default = 3000, help = "已用显存超过它时等待（MiB）")
    parser.add_argument("--timeout-factor", type = float, default = 3.0, help = "单项超过估计用时的这么多倍（至少 30 分钟）时结束它")
    parser.add_argument("--dry-run", action = "store_true", help = "只检查输入、列出计划")
    parser.add_argument("--summary-only", action = "store_true", help = "只根据已有结果写汇总")
    args = parser.parse_args(args = argv)
    args.keys = [key for key in args.runs.split(",") if key]
    unknown = [key for key in args.keys if key not in c_runs.ALL_RUNS]
    if unknown:
        parser.error(f"未知的运行：{unknown}，可选 {list(c_runs.ALL_RUNS)}")
    args.rerun = [key for key in args.rerun_user.split(",") if key]
    if args.keys and not args.runs_root:
        parser.error("需要 --runs-root（或环境变量 C71_RUNS_ROOT）")
    if not args.dry_run and not args.summary_only and not args.py_ifeval:
        parser.error("需要 --py-ifeval（文本打分）")
    return args


def main(argv = None):
    args = parse_args(argv = argv)
    paths = resolve_paths(args = args)
    keys = args.keys
    include_ref = not args.no_ref
    labels = ([REF] if include_ref else []) + keys
    if args.summary_only:
        write_summary(paths = paths, labels = labels, log = print)
        return 0
    runs = c71_local.discover_runs(runs_root = paths["runs_root"], keys = keys) if keys else {}
    plan = build_plan(
        runs = runs,
        keys = keys,
        paths = paths,
        include_ref = include_ref,
        rerun_user = args.rerun,
    )
    checks = check_inputs(paths = paths, keys = keys, runs = runs)
    if args.dry_run:
        print("#71 系统提示遵循评测：演习（只检查和列出计划，不占用 GPU）")
        for ok, text in checks:
            print(f"  {'通过' if ok else '缺少'}  {text}")
        total = 0.0
        for label, entries in plan.items():
            for entry in entries:
                state = "已有结果，跳过" if entry["marker"].exists() else "运行"
                total += 0.0 if entry["marker"].exists() else entry["minutes"]
                print(f"  {entry['item']:<13} {label:<28} {state:<8} 约 {entry['minutes']:.1f} 分钟")
        print(f"  共约 {total:.0f} 分钟")
        for label, entries in list(plan.items())[:2]:
            for entry in entries:
                print(f"  [{label} {entry['item']}] " + " ".join(entry["command"]))
        return 0 if all(ok for ok, _ in checks) else 2
    paths["out_dir"].mkdir(parents = True, exist_ok = True)
    driver = c71_local.Driver(paths = paths, busy_mib = args.busy_mib, timeout_factor = args.timeout_factor)

    def on_signal(signum, frame):
        raise c71_local.Stop(signum = signum)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    driver.set_status(pid = os.getpid(), started = time.strftime("%Y-%m-%d %H:%M:%S"), phase = "start", current = None)
    driver.log(f"开始（pid {os.getpid()}，模型 {list(plan)}，输出 {paths['out_dir']}）")
    failed_inputs = [text for ok, text in checks if not ok]
    if failed_inputs:
        driver.log(f"缺少基本输入：{failed_inputs}")
        driver.set_status(phase = "error", error = failed_inputs)
        return 2
    flat = [entry for entries in plan.values() for entry in entries]
    failures = []
    try:
        for label, entries in plan.items():
            for entry in entries:
                result = driver.run_entry(entry = entry, plan = flat)
                if result not in ("ok", "skipped"):
                    failures.append(f"{label}:{entry['item']}")
                if entry["item"] == "text" and result in ("ok", "skipped"):
                    scored = score_text(paths = paths, label = label, log = driver.log)
                    if scored not in ("ok", "skipped"):
                        failures.append(f"{label}:text_score")
            if label != REF:
                c71_local.remove_export(paths = paths, label = label, log = driver.log)
            write_summary(paths = paths, labels = labels, log = driver.log)
    except c71_local.Stop as stop:
        driver.kill_child()
        driver.release_lock()
        driver.log(f"停下：{stop}；重新运行即可续跑")
        driver.set_status(phase = "stopped", current = None)
        return 128 + stop.signum
    except Exception as error:
        driver.kill_child()
        driver.release_lock()
        driver.log(f"出错：{error!r}")
        driver.set_status(phase = "error", error = repr(error), current = None)
        raise
    write_summary(paths = paths, labels = labels, log = driver.log)
    if failures:
        driver.log(f"结束，有失败的项目：{failures}（重新运行只重做这些）")
        driver.set_status(phase = "done_with_failures", failed = failures, current = None)
        return 1
    driver.log("全部完成")
    driver.set_status(phase = "done", current = None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
