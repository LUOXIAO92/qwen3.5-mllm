"""#71 C 组第一轮的本机评测：六个全量微调运行和参考模型的全部评测在本机 RTX 5090（32 GB）上夜间进行。

用户 2026-11-23 决定：Colab 只训练（C71_TRAIN_ONLY=1），评测在本机做。评测项目、命令和设置一律取自 Colab 驱动程序
pipeline/colab/c_runs.py 的 eval_items（同一份代码），这里只换成本机的路径，另做下面几处本机的调整：

- python：读音频的两项（fleurs 即 run_eval，probe 即 instruct_probe）要 soundfile，用 --py-audio（默认
  <工作目录>/envs/mllm-eval）；纯文本的三项（lm_ppl、replay_gap、think_check）用 --py-text（默认 envs/mllm-train，有
  flash-linear-attention 与 mm_assistant_mask，#70 参考模型的 teacher 对数概率也是在这个环境里算的）。本机没有同时装
  这两样的环境；Colab 的训练环境两样都有。envs/mllm-eval 里线性注意力用 transformers 的 PyTorch 实现，数值与 fla
  内核有极小的差别，C 组各运行都在本机用同样的环境评测，彼此可以直接比较。
- 显存：think_check 一批 200 条生成到 16,384 个新 token 约需 74 GB，加 --fit-memory 后按剩余显存分批（5090 上约 3 批、
  每批 67 条，见 think_check.py 开头「显存」）。其余各项的批大小在 32 GB 内放得下，保持不变：fleurs 一批 48 条峰值约
  10 GiB（T54 实测），lm_ppl 一批 32 条最长的一批（菲律宾语）词表输出约 11 GB，replay_gap 每批 32,768 个 token（#70 的
  teacher 在本机用 65,536 个 token 一批）。所有子进程设 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True，减少反复
  拼接 KV 缓存造成的显存碎片。
- 顺序：默认按评测项目排列（--order item）：先对参考模型和全部运行做回放差距与困惑度（几分钟一项），再做 FLEURS、探针，
  最后做最费时的思考检查；一晚做不完时，已完成的部分在各运行之间可以直接比较。--order run 与 Colab 相同，一个运行做完
  全部项目再做下一个，最后做参考模型。

下载（用户从 Google Drive 下载，保持 Drive 上的目录结构）：每个运行 <名称> = v2b-2b-cpt77-<c0|c-free_raw|c-free_chat|
c-cont_raw|c-cont_chat|c-mix>，Drive 上的 runs/<名称>/ 对应本机的 <运行目录>/<名称>/（--runs-root，默认
<工作目录>/data/T71c/runs）：

- 必需：checkpoint-0030000/language_model.safetensors（约 3.76 GB）、checkpoint-0030000/projector.safetensors；
- 汇总用（缺少时表中留空）：checkpoint-0030000/projector.json、train_config.json（读 projector 的起点）、
  train_log.jsonl（训练损失、回放比例与梯度份额）、eval_log.jsonl（dev 损失）、replay_log.jsonl（C0 没有）。
  trainer_state.pt（约 22.6 GB）不下载。

驱动程序启动时检查哪些运行的两个必需文件齐全（按 safetensors 文件头核对文件长度，下载到一半的文件算缺少），只评测齐全
的运行；每做完一种评测项目重新检查一次，中途下载完成的运行在后面的项目中补上，前面的项目下次运行驱动程序时补做。

续跑与失败：结果文件（c_runs 的完成标志，例如 fleurs/summary.json）已存在的项目跳过。单项失败（非零退出、超时）记在
status.json 的 results 中，接着做下一项；重新运行驱动程序只重做缺少结果的项目。单项超过估计时间的 --timeout-factor 倍
（至少 30 分钟）视为超时，结束它的进程组。

GPU：每一项开始前持有 <工作目录>/gpu.lock（与 flock 命令同一种锁，fcntl.flock），子进程结束后释放；拿锁之前和之后都看
nvidia-smi 的已用显存，超过 --busy-mib（默认 3,000 MiB，例如 Windows 上的 ComfyUI 还开着）就等 5 分钟再看，不结束别人的
进程。子进程在单独的进程组中运行，结束后（包括超时、驱动程序收到 SIGTERM 或 SIGINT 时）整组结束，不留 GPU 进程。

输出（--out-dir，默认 <工作目录>/data/T71c）：
- eval/<标签>/：各项评测的结果与日志（<项目>.log），标签为 c0、c-free_raw、…、ref，目录结构与 Colab 的 eval/ 相同；
- summary/：summary.md、summary.json（与 c_runs.phase_summary 同样的两张表、report 的 FLEURS 表格，另加与 C0 比较的
  两张表）、report 的 tables.md 与 tables.json；
- status.json（阶段、正在做的项目、各项结果与用时、剩余时间的估计）、driver.log。

退出码：0 全部完成；1 有失败的项目；2 缺少基本输入（模型、FLEURS、回放数据、python 环境）；3 能做的都做完了，但有运行的
文件还没下载；143 / 130 收到 SIGTERM / SIGINT 后停下。

用法（在仓库根目录运行；--dry-run 只检查输入、列出计划和时间估计，不占用 GPU）：

    WORK_DIR=<工作目录> DATASET=<数据集目录> python -m pipeline.eval.c71_local --dry-run
    WORK_DIR=<工作目录> DATASET=<数据集目录> nohup setsid python -m pipeline.eval.c71_local \\
        >> <工作目录>/data/T71c/launch.out 2>&1 < /dev/null &

停下：kill <status.json 中的 pid>（驱动程序结束正在运行的评测进程组后退出），之后重新运行即可续跑。
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from pipeline.colab import c_runs
from pipeline.eval.lm_ppl import LANGS as FLEURS_LANGS
from pipeline.tokenizer_fix import TOKENIZER_CLASS

REPO = Path(__file__).resolve().parents[2]
REF = "ref"
BASE = "c0"
ITEM_ORDER = ("replay_gap", "lm_ppl", "fleurs", "probe", "think")
# 各项评测用的 python：audio 为 --py-audio（soundfile），text 为 --py-text（fla、mm_assistant_mask）。
PYTHON_OF_ITEM = {"fleurs": "audio", "probe": "audio", "lm_ppl": "text", "replay_gap": "text", "think": "text"}
# 本机在 c_runs 的命令之后追加的参数。
LOCAL_EXTRA = {"think": ["--fit-memory"]}
CHECKPOINT = f"checkpoint-{c_runs.FULL_STEPS:07d}"
REQUIRED_FILES = (f"{CHECKPOINT}/language_model.safetensors", f"{CHECKPOINT}/projector.safetensors")
OPTIONAL_FILES = (f"{CHECKPOINT}/projector.json", "train_config.json", "train_log.jsonl", "eval_log.jsonl", "replay_log.jsonl")
# 各运行每个文件的大约大小（字节），只用来报告下载量。
APPROX_BYTES = {"language_model.safetensors": 3.76e9, "projector.safetensors": 3.4e7}
BUSY_POLL_SECONDS = 300
LOCK_POLL_SECONDS = 30


def local_minutes(item, think_max_new):
    """本机 RTX 5090 上一项评测的估计用时（分钟），由 c_runs.ESTIMATE（Colab G4）换算：

    - fleurs：G4 估计 15；T54 在本机用同样的批大小（48）评测 2B 的 FLEURS test，生成实测 19.6–24.4 分钟，取 22。
    - probe：与 fleurs 同样是逐批贪心生成，按同样的比例换算（G4 15 → 22）。
    - lm_ppl：G4 估计 3，纯前向，本机相同。
    - replay_gap：#70 的参考模型前向在本机约 3 万 token/s，五组评测集约 620 万 token，加载模型，约 5（G4 估计 6）。
    - think：G4 估计 18（一批 200 条）；本机分 3 批、每批 67 条，各生成到上限，每步的固定开销按批数增加，KV 缓存与复制到
      全部查询头的临时张量的读写总量与条数成正比，估计 40（随生成上限按比例缩放）。
    """
    minutes = {"fleurs": 22, "probe": 22, "lm_ppl": 3, "replay_gap": 5}
    if item == "think":
        return 38 * think_max_new / 16384 + 2
    return minutes[item]


class Stop(Exception):
    """收到 SIGTERM 或 SIGINT。"""

    def __init__(self, signum):
        super().__init__(f"收到信号 {signum}")
        self.signum = signum


def safetensors_complete(path):
    """safetensors 文件是否完整：按文件头中各张量的数据区间算出的长度应等于文件长度。返回 (是否完整, 原因)。"""
    path = Path(path)
    if not path.exists():
        return False, "不存在"
    size = path.stat().st_size
    try:
        with open(file = path, mode = "rb") as handle:
            head = handle.read(8)
            if len(head) < 8:
                return False, f"只有 {size} 字节"
            header_len = int.from_bytes(bytes = head, byteorder = "little")
            if header_len <= 0 or 8 + header_len > size:
                return False, f"文件头长度 {header_len} 超出文件（{size} 字节）"
            header = json.loads(s = handle.read(header_len))
    except (OSError, ValueError) as error:
        return False, f"读不出文件头：{error}"
    end = max((value["data_offsets"][1] for key, value in header.items() if key != "__metadata__"), default = 0)
    expected = 8 + header_len + end
    if size != expected:
        return False, f"长度 {size} 字节，文件头记录应为 {expected} 字节（没有下载完？）"
    return True, ""


def discover_runs(runs_root, keys):
    """各运行的文件状况：{key: {"name", "dir", "ready", "missing", "optional_missing"}}。ready 为两个必需文件都完整。"""
    found = {}
    for key in keys:
        name = c_runs.run_name(key = key)
        run_dir = Path(runs_root) / name
        missing = []
        for relative in REQUIRED_FILES:
            ok, reason = safetensors_complete(path = run_dir / relative)
            if not ok:
                missing.append(f"{relative}（{reason}）")
        optional = [relative for relative in OPTIONAL_FILES if not (run_dir / relative).exists()]
        if key == BASE:
            optional = [relative for relative in optional if relative != "replay_log.jsonl"]
        found[key] = {
            "name": name,
            "dir": run_dir,
            "ready": not missing,
            "missing": missing,
            "optional_missing": optional,
        }
    return found


def localize(items, python_audio, python_text, think_max_new):
    """把 c_runs.eval_items 的项目换成本机的形式：[{item, command, marker, minutes}]。

    只换命令开头的 python 并追加 LOCAL_EXTRA，其余参数与 c_runs 相同。
    """
    pythons = {"audio": str(python_audio), "text": str(python_text)}
    localized = []
    for name, command, marker, _ in items:
        localized.append({
            "item": name,
            "command": [pythons[PYTHON_OF_ITEM[name]], *command[1:], *LOCAL_EXTRA.get(name, [])],
            "marker": Path(marker),
            "minutes": local_minutes(item = name, think_max_new = think_max_new),
        })
    return localized


def label_items(label, run, paths):
    """一个标签（运行或参考模型）的本机评测项目，按 c_runs.eval_items 生成。run 为 discover_runs 的一项，参考模型为 None。"""
    out = paths["eval_root"] / label
    common = {
        "python": paths["python_audio"],
        "model_dir": paths["model_dir"],
        "fleurs_dir": paths["fleurs"],
        "replay_dir": paths["replay"],
        "think_max_new": paths["think_max_new"],
    }
    if run is None:
        items = c_runs.eval_items(
            out = out,
            probe = False,
            fleurs = False,
            **common,
        )
    else:
        checkpoint = run["dir"] / CHECKPOINT
        items = c_runs.eval_items(
            out = out,
            projector = checkpoint / "projector.safetensors",
            language_model = checkpoint / "language_model.safetensors",
            **common,
        )
    localized = localize(
        items = items,
        python_audio = paths["python_audio"],
        python_text = paths["python_text"],
        think_max_new = paths["think_max_new"],
    )
    for entry in localized:
        entry["label"] = label
        entry["log"] = out / f"{entry['item']}.log"
    return localized


def build_plan(runs, keys, paths, order = "item", include_ref = True):
    """全部评测项目的顺序表。runs 为 discover_runs 的结果，只含 ready 的运行；keys 为运行的先后顺序。"""
    per_label = {}
    if include_ref:
        per_label[REF] = label_items(label = REF, run = None, paths = paths)
    for key in keys:
        if runs.get(key, {}).get("ready"):
            per_label[key] = label_items(label = key, run = runs[key], paths = paths)
    if order == "run":
        labels = [key for key in keys if key in per_label] + ([REF] if REF in per_label else [])
        return [entry for label in labels for entry in per_label[label]]
    labels = ([REF] if REF in per_label else []) + [key for key in keys if key in per_label]
    plan = []
    for item in ITEM_ORDER:
        for label in labels:
            plan += [entry for entry in per_label[label] if entry["item"] == item]
    return plan


def gpu_used_mib():
    """nvidia-smi 报告的已用显存（MiB）；没有 nvidia-smi 或读不出时为 None。"""
    try:
        result = subprocess.run(
            args = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output = True,
            text = True,
            timeout = 30,
            check = False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.split()[0])
    except (IndexError, ValueError):
        return None


def load_json(path):
    try:
        return json.loads(s = Path(path).read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------- 汇总


def with_delta(value, base, scale = 100.0, digits = 1, ratio = False):
    """数值加上与 C0 的比较：差（百分比按百分点）或比值。value 为 None 时为空。"""
    if value is None:
        return ""
    text = f"{scale * value:.{digits}f}"
    if base is None:
        return text
    if ratio:
        return f"{text}（{value / base:.2f}×）" if base else text
    return f"{text}（{scale * (value - base):+.{digits}f}）"


def dig(data, path):
    """按键路径 path 取值，路径中任何一层不存在时为 None。"""
    for key in path:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data


def column(title, path, scale = 100.0, digits = 1, ratio = False):
    """比较表的一列：列名、在一个运行的结果中的取值路径、显示的倍数与小数位、括号中给比值（真）还是差（假）。"""
    return {"title": title, "path": path, "scale": scale, "digits": digits, "ratio": ratio}


# 与 C0 比较的第一张表：识别与遗忘。fleurs 为 report 的 fleurs_mean 中这个运行的一项，其余见 c_runs.summary_tables。
RECOGNITION_COLUMNS = [
    column(title = "FLEURS 平均错误率 %", path = ("fleurs", "basic")),
    column(title = "失控 %", path = ("fleurs", "runaway")),
    column(
        title = "dev 损失",
        path = ("train", "dev_loss"),
        scale = 1.0,
        digits = 4,
    ),
    column(
        title = "回放差距（各组平均）",
        path = ("replay_gap", "mean_gap"),
        scale = 1.0,
        digits = 4,
    ),
] + [
    column(
        title = "每字困惑度 zh" if lang == "zh" else lang,
        path = ("lm_ppl", lang, "char_ppl"),
        scale = 1.0,
        digits = 2,
        ratio = True,
    )
    for lang in c_runs.PPL_LANGS
]
# 第二张表：思考检查与指令遵循探针（各语言平均）。
FOLLOW_COLUMNS = [
    column(title = "写完 </think> %", path = ("think", "closed")),
    column(title = "循环 %", path = ("think", "loop")),
    column(title = "到上限 %", path = ("think", "capped")),
    column(
        title = "推理中位（token）",
        path = ("think", "reasoning_tokens", "median"),
        scale = 1.0,
        digits = 0,
    ),
    column(title = "你听到了什么·逐字 英", path = ("probe", "what_heard", "heard_en", "verbatim_share")),
    column(title = "中", path = ("probe", "what_heard", "heard_zh", "verbatim_share")),
    column(title = "翻译·照抄原文 英", path = ("probe", "translate", "translate_en", "verbatim_share")),
    column(title = "中", path = ("probe", "translate", "translate_zh", "verbatim_share")),
    column(title = "翻译 chrF 英", path = ("probe", "translate", "translate_en", "chrf"), scale = 1.0),
    column(title = "中", path = ("probe", "translate", "translate_zh", "chrf"), scale = 1.0),
    column(title = "时间段·整段 英", path = ("probe", "timestamps", "segment_en", "whole_share")),
    column(title = "中", path = ("probe", "timestamps", "segment_zh", "whole_share")),
    column(title = "转写错误率 聊天", path = ("probe", "transcribe", "transcribe", "rate_basic")),
    column(title = "CPT", path = ("probe", "transcribe", "transcribe_cpt", "rate_basic")),
]


def compare_table(data, labels, columns, base = BASE):
    """一张与 base 比较的表：每行一个运行，每格为数值加括号中的差或比值（base 自己那一行只有数值）。"""
    lines = [
        "| 运行 | " + " | ".join(spec["title"] for spec in columns) + " |",
        "| --- |" + " ---: |" * len(columns),
    ]
    for label in labels:
        cells = []
        for spec in columns:
            value = dig(data = data.get(label), path = spec["path"])
            reference = None if label == base else dig(data = data.get(base), path = spec["path"])
            cells.append(with_delta(
                value = value,
                base = reference,
                scale = spec["scale"],
                digits = spec["digits"],
                ratio = spec["ratio"],
            ))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def comparison_lines(table, fleurs_mean, labels, base = BASE):
    """与 C0 比较的两张表：识别与遗忘；思考与指令遵循。table 为 c_runs.summary_tables 的第二个返回值，fleurs_mean 为
    report 的 tables.json 中的 fleurs_mean。"""
    data = {label: dict(table.get(label, {}), fleurs = (fleurs_mean or {}).get(label)) for label in labels}
    name = base.upper()
    lines = [
        f"## 与 {name} 的比较",
        "",
        f"括号中是与 {name} 之差（比例按百分点），困惑度括号中是与 {name} 之比。FLEURS 平均错误率为各语言基础分的简单平均"
        "（report 的「各语言平均」），失控为输出字符数超过参考 2 倍加 10 的句子比例。",
        "",
    ]
    lines += compare_table(
        data = data,
        labels = labels,
        columns = RECOGNITION_COLUMNS,
        base = base,
    )
    lines += [
        "",
        "思考检查与指令遵循探针（聊天格式提问，各语言平均）：「你听到了什么」与翻译两项为输出几乎逐字照抄原文转写的比例，"
        "时间段一项为把拼接音频整段转写出来的比例；转写错误率为基础分，聊天格式用训练时的指令提问，CPT 为只有音频标记的提示。",
        "",
    ]
    lines += compare_table(
        data = data,
        labels = labels,
        columns = FOLLOW_COLUMNS,
        base = base,
    )
    return lines


def projector_start(runs):
    """projector 的起点步数（训练配置 init_projector 中的 checkpoint-<步数>），读不出时为 None。"""
    for run in runs.values():
        config = load_json(path = run["dir"] / "train_config.json") or {}
        match = re.search(pattern = r"checkpoint-(\d+)", string = str(config.get("init_projector", "")))
        if match:
            return int(match.group(1)), config["init_projector"]
    return None


def write_summary(paths, runs, keys, with_report, log):
    """写 summary/：c_runs.summary_tables 的两张表、report 的 FLEURS 表格（with_report 时重新运行 report）、与 C0 的比较。"""
    eval_root = paths["eval_root"]
    summary_dir = paths["out_dir"] / "summary"
    summary_dir.mkdir(parents = True, exist_ok = True)
    labels = [key for key in keys if (eval_root / key).exists()] + ([REF] if (eval_root / REF).exists() else [])
    fleurs = [
        f"{key}={eval_root / key / 'fleurs'}"
        for key in keys
        if (eval_root / key / "fleurs" / "summary.json").exists() and (eval_root / key / "fleurs" / "hypotheses.jsonl").exists()
    ]
    if with_report and fleurs:
        with open(file = summary_dir / "report.log", mode = "a", encoding = "utf-8") as handle:
            code = subprocess.run(
                args = [str(paths["python_audio"]), "-m", "pipeline.eval.report", "--out-dir", str(summary_dir), "--fleurs", *fleurs],
                cwd = REPO,
                stdout = handle,
                stderr = subprocess.STDOUT,
                check = False,
            ).returncode
        if code != 0:
            log(f"report 失败（退出码 {code}），见 {summary_dir / 'report.log'}")
    train = {key: c_runs.train_stats(out = runs[key]["dir"]) for key in labels if key in runs}
    start = projector_start(runs = {key: runs[key] for key in keys if key in runs})
    lines = ["# #71 C 组第一轮汇总（本机评测）", ""]
    if start is not None:
        lines += [f"projector 起点：第 {start[0]} 步（训练配置的 init_projector：{start[1]}）", ""]
    body, table = c_runs.summary_tables(labels = labels, eval_root = eval_root, runs = train)
    lines += body
    tables = load_json(path = summary_dir / "tables.json") or {}
    lines += [""] + comparison_lines(table = table, fleurs_mean = tables.get("fleurs_mean"), labels = labels)
    if (summary_dir / "tables.md").exists():
        lines += ["", "## FLEURS test（report）", "", (summary_dir / "tables.md").read_text(encoding = "utf-8")]
    (summary_dir / "summary.md").write_text(data = "\n".join(lines) + "\n", encoding = "utf-8")
    (summary_dir / "summary.json").write_text(
        data = json.dumps(obj = {"labels": labels, "fleurs_mean": tables.get("fleurs_mean"), "runs": table}, ensure_ascii = False, indent = 1),
        encoding = "utf-8",
    )
    return labels


# ---------------------------------------------------------------- 运行


class Driver:
    """按计划依次运行评测，维护 status.json 与 driver.log。"""

    def __init__(self, paths, busy_mib, timeout_factor):
        self.paths = paths
        self.busy_mib = busy_mib
        self.timeout_factor = timeout_factor
        self.status_path = paths["out_dir"] / "status.json"
        self.log_path = paths["out_dir"] / "driver.log"
        self.child = None
        self.lock_handle = None
        self.measured = {}
        # 超时的下限（分钟）；测试中改小
        self.min_timeout_minutes = 30.0

    def log(self, message):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush = True)
        with open(file = self.log_path, mode = "a", encoding = "utf-8") as handle:
            handle.write(line + "\n")

    def read_status(self):
        return load_json(path = self.status_path) or {}

    def set_status(self, **fields):
        status = self.read_status()
        status.update(fields)
        status["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        temporary = self.status_path.with_name(self.status_path.name + ".tmp")
        text = json.dumps(
            obj = status,
            ensure_ascii = False,
            indent = 1,
            default = str,
        )
        temporary.write_text(data = text, encoding = "utf-8")
        os.replace(src = temporary, dst = self.status_path)

    def record(self, entry, result, minutes = None):
        status = self.read_status()
        results = status.get("results", {})
        results[f"{entry['label']}:{entry['item']}"] = {
            "status": result,
            "minutes": None if minutes is None else round(number = minutes, ndigits = 1),
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "log": str(entry["log"]),
        }
        self.set_status(results = results)

    def estimate_minutes(self, entry):
        """一项的估计用时：本次已经做完的同类项目的平均用时，没有时用 local_minutes。"""
        done = self.measured.get(entry["item"])
        return sum(done) / len(done) if done else entry["minutes"]

    def remaining_minutes(self, plan):
        return sum(self.estimate_minutes(entry = entry) for entry in plan if not entry["marker"].exists())

    def wait_gpu_free(self, entry):
        """已用显存超过 busy_mib 时每 5 分钟看一次，直到空出来。"""
        announced = False
        while True:
            used = gpu_used_mib()
            if used is None or used <= self.busy_mib:
                return
            if not announced:
                self.log(f"GPU 已用 {used} MiB（超过 {self.busy_mib} MiB），等其他程序释放后再做 {entry['label']} {entry['item']}")
                self.set_status(phase = "waiting_gpu", gpu_used_mib = used)
                announced = True
            time.sleep(BUSY_POLL_SECONDS)

    def acquire_lock(self, entry):
        """拿 gpu.lock；之后显存仍被锁外的程序占着时放开锁、等一会儿再来。"""
        lock_path = self.paths["gpu_lock"]
        announced = False
        while True:
            self.wait_gpu_free(entry = entry)
            handle = open(file = lock_path, mode = "a")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                if not announced:
                    self.log(f"gpu.lock 被其他任务持有，等待（{entry['label']} {entry['item']}）")
                    self.set_status(phase = "waiting_lock")
                    announced = True
                time.sleep(LOCK_POLL_SECONDS)
                continue
            used = gpu_used_mib()
            if used is not None and used > self.busy_mib:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
                continue
            self.lock_handle = handle
            return

    def release_lock(self):
        if self.lock_handle is not None:
            fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_UN)
            self.lock_handle.close()
            self.lock_handle = None

    def kill_child(self):
        """结束子进程所在的整个进程组（先 SIGTERM，30 秒后 SIGKILL）。"""
        if self.child is None:
            return
        try:
            os.killpg(self.child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.child.wait(timeout = 30)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(self.child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.child.wait()
        self.child = None

    def run_entry(self, entry, plan):
        """运行一项（结果已存在时跳过）。返回 "ok"、"skipped" 或失败说明。"""
        if entry["marker"].exists():
            return "skipped"
        entry["log"].parent.mkdir(parents = True, exist_ok = True)
        self.acquire_lock(entry = entry)
        timeout = max(self.min_timeout_minutes, self.timeout_factor * self.estimate_minutes(entry = entry))
        self.log(
            f"开始 {entry['label']} {entry['item']}（估计 {self.estimate_minutes(entry = entry):.0f} 分钟，超时 {timeout:.0f} 分钟），"
            f"日志 {entry['log']}"
        )
        self.set_status(
            phase = "running",
            current = {"label": entry["label"], "item": entry["item"], "started": time.strftime("%Y-%m-%d %H:%M:%S"), "log": str(entry["log"])},
            remaining_hours = round(number = self.remaining_minutes(plan = plan) / 60, ndigits = 1),
        )
        env = dict(os.environ)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env["PYTHONUNBUFFERED"] = "1"
        start = time.time()
        result = None
        try:
            with open(file = entry["log"], mode = "a", encoding = "utf-8") as handle:
                handle.write(f"\n# {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(entry['command'])}\n")
                handle.flush()
                self.child = subprocess.Popen(
                    args = entry["command"],
                    cwd = REPO,
                    stdin = subprocess.DEVNULL,
                    stdout = handle,
                    stderr = subprocess.STDOUT,
                    start_new_session = True,
                    env = env,
                )
                try:
                    code = self.child.wait(timeout = timeout * 60)
                except subprocess.TimeoutExpired:
                    result = f"timeout（{timeout:.0f} 分钟）"
                    code = None
            if result is None:
                result = "ok" if code == 0 and entry["marker"].exists() else f"failed（退出码 {code}）"
        finally:
            # 正常结束时进程组里通常已经没有进程；超时或收到信号时整组结束，不留 GPU 进程
            self.kill_child()
            self.release_lock()
        minutes = (time.time() - start) / 60
        if result == "ok":
            self.measured.setdefault(entry["item"], []).append(minutes)
        self.record(entry = entry, result = result, minutes = minutes)
        self.log(f"{entry['label']} {entry['item']}：{'完成' if result == 'ok' else result}，{minutes:.1f} 分钟")
        return result


def check_inputs(paths, check_python = True):
    """基本输入的检查：[(是否通过, 说明)]。"""
    checks = []
    model_dir = Path(paths["model_dir"])
    weights = model_dir / "model.safetensors"
    tokenizer_class = (load_json(path = model_dir / "tokenizer_config.json") or {}).get("tokenizer_class")
    model_ok = (model_dir / "config.json").exists() and weights.exists() and tokenizer_class == TOKENIZER_CLASS
    size = f"{weights.stat().st_size / 1e9:.2f} GB" if weights.exists() else "缺少"
    checks.append((
        model_ok,
        f"模型目录 {model_dir}：model.safetensors {size}，分词器 {tokenizer_class}（应为 {TOKENIZER_CLASS}，#77）",
    ))
    fleurs = Path(paths["fleurs"])
    rows = 0
    missing_audio = 0
    missing_langs = []
    for lang in FLEURS_LANGS:
        manifest = fleurs / lang / "test.jsonl"
        if not manifest.exists():
            missing_langs.append(lang)
            continue
        for line in manifest.read_text(encoding = "utf-8").splitlines():
            if line.strip():
                rows += 1
                if not (fleurs / json.loads(s = line)["audio_path"]).exists():
                    missing_audio += 1
    checks.append((
        not missing_langs and not missing_audio and rows > 0,
        f"FLEURS test {fleurs}：{len(FLEURS_LANGS) - len(missing_langs)} 种语言 {rows} 条"
        + (f"，缺少语言 {missing_langs}" if missing_langs else "")
        + (f"，缺少音频 {missing_audio} 个" if missing_audio else "，音频齐全"),
    ))
    replay = Path(paths["replay"])
    missing = [
        f"{group}/eval/{name}"
        for group in c_runs.REPLAY_GROUPS
        for name in ("samples.jsonl", "teacher/meta.json", "raw")
        if not (replay / group / "eval" / name).exists()
    ]
    checks.append((not missing, f"回放评测集 {replay}：五组 eval" + (f"，缺少 {missing}" if missing else " 齐全（samples、raw、teacher）")))
    if check_python:
        for role, modules in (("audio", "soundfile, torch, transformers, safetensors"), ("text", "fla, mm_assistant_mask, torch, transformers")):
            python = Path(paths[f"python_{role}"])
            ok = python.exists()
            detail = "不存在"
            if ok:
                result = subprocess.run(
                    args = [str(python), "-c", f"import {modules}"],
                    capture_output = True,
                    text = True,
                    cwd = REPO,
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES = ""),
                    check = False,
                )
                ok = result.returncode == 0
                detail = f"import {modules}：" + ("通过" if ok else result.stderr.strip().splitlines()[-1][:200])
            items = [item for item, kind in PYTHON_OF_ITEM.items() if kind == role]
            checks.append((ok, f"python（{'、'.join(items)}）{python}：{detail}"))
    lock = Path(paths["gpu_lock"])
    checks.append((lock.parent.exists(), f"GPU 锁 {lock}" + ("" if lock.exists() else "（还不存在，运行时创建）")))
    return checks


def dry_run(paths, keys, order, include_ref):
    """检查输入、列出计划和时间估计，不占用 GPU。返回退出码（基本输入不全时为 2）。"""
    print("#71 C 组第一轮的本机评测：演习（只检查和列出计划，不占用 GPU）")
    for name in ("model_dir", "fleurs", "replay", "runs_root", "out_dir", "python_audio", "python_text"):
        print(f"  {name}: {paths[name]}")
    print(f"  思考检查的生成上限：{paths['think_max_new']}")
    print("\n输入检查")
    checks = check_inputs(paths = paths)
    for ok, text in checks:
        print(f"  {'通过' if ok else '缺少'}  {text}")
    disk = shutil.disk_usage(path = paths["out_dir"] if paths["out_dir"].exists() else paths["out_dir"].parent)
    print(f"  信息  输出所在磁盘剩余 {disk.free / 1e9:.0f} GB")
    runs_root = Path(paths["runs_root"])
    root_disk = shutil.disk_usage(path = runs_root if runs_root.exists() else runs_root.parent if runs_root.parent.exists() else "/")
    print(f"  信息  运行目录所在磁盘剩余 {root_disk.free / 1e9:.0f} GB")
    used = gpu_used_mib()
    print(f"  信息  GPU 已用显存：{'读不出（没有 nvidia-smi？）' if used is None else f'{used} MiB'}")
    print("\n运行（下载）")
    runs = discover_runs(runs_root = paths["runs_root"], keys = keys)
    pending_bytes = 0.0
    for key in keys:
        run = runs[key]
        if run["ready"]:
            text = "齐全"
        else:
            text = "缺少 " + "；".join(run["missing"])
            pending_bytes += sum(APPROX_BYTES[Path(relative).name] for relative in REQUIRED_FILES if any(item.startswith(relative) for item in run["missing"]))
        if run["optional_missing"]:
            text += "；汇总用的文件缺少 " + "、".join(run["optional_missing"])
        print(f"  {key:<12} {run['dir']}：{text}")
    print(f"  还要下载的必需文件约 {pending_bytes / 1e9:.1f} GB")
    plan_now = build_plan(
        runs = runs,
        keys = keys,
        paths = paths,
        order = order,
        include_ref = include_ref,
    )
    all_ready = {key: dict(runs[key], ready = True) for key in keys}
    plan_full = build_plan(
        runs = all_ready,
        keys = keys,
        paths = paths,
        order = order,
        include_ref = include_ref,
    )
    print(f"\n计划（--order {order}；已有结果的项目跳过）")
    ready_ids = {(entry["label"], entry["item"]) for entry in plan_now}
    for entry in plan_full:
        if entry["marker"].exists():
            state = "已有结果，跳过"
        elif (entry["label"], entry["item"]) in ready_ids:
            state = "运行"
        else:
            state = "等待下载"
        print(f"  {entry['item']:<11} {entry['label']:<12} {state:<10} 约 {entry['minutes']:.0f} 分钟")
    now = sum(entry["minutes"] for entry in plan_now if not entry["marker"].exists())
    full = sum(entry["minutes"] for entry in plan_full if not entry["marker"].exists())
    per_run = sum(entry["minutes"] for entry in plan_full if entry["label"] == BASE)
    per_ref = sum(entry["minutes"] for entry in plan_full if entry["label"] == REF)
    print(f"\n时间估计（RTX 5090）：每个运行约 {per_run:.0f} 分钟，参考模型约 {per_ref:.0f} 分钟；"
          f"现在能做的 {len([e for e in plan_now if not e['marker'].exists()])} 项约 {now / 60:.1f} 小时，"
          f"全部下载后共 {len([e for e in plan_full if not e['marker'].exists()])} 项约 {full / 60:.1f} 小时")
    print("\n各项的命令（以第一个运行为例；参考模型不给 --projector、--language-model）")
    shown = set()
    for entry in plan_full:
        if entry["item"] not in shown and entry["label"] != REF:
            shown.add(entry["item"])
            print(f"  [{entry['item']}] " + " ".join(entry["command"]))
    return 0 if all(ok for ok, _ in checks) else 2


def resolve_paths(args):
    work = Path(args.work_dir)
    dataset = args.dataset
    out_dir = Path(args.out_dir) if args.out_dir else work / "data" / "T71c"
    return {
        "work": work,
        "model_dir": Path(args.model_dir),
        "fleurs": Path(args.fleurs_dir) if args.fleurs_dir else work / "data" / "T02",
        "replay": Path(args.replay_dir) if args.replay_dir else Path(dataset or "") / "text" / "qwen3.5-asr" / "replay-noloop",
        "runs_root": Path(args.runs_root) if args.runs_root else out_dir / "runs",
        "out_dir": out_dir,
        "eval_root": out_dir / "eval",
        "python_audio": Path(args.py_audio) if args.py_audio else work / "envs" / "mllm-eval" / "bin" / "python",
        "python_text": Path(args.py_text) if args.py_text else work / "envs" / "mllm-train" / "bin" / "python",
        "gpu_lock": Path(args.gpu_lock) if args.gpu_lock else work / "gpu.lock",
        "think_max_new": args.think_max_new,
    }


def parse_args(argv = None):
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", default = os.environ.get("WORK_DIR", str(REPO.parent / "qwen3.5-asr-work")), help = "工作目录，默认环境变量 WORK_DIR")
    parser.add_argument("--dataset", default = os.environ.get("DATASET"), help = "数据集目录，默认环境变量 DATASET（回放数据在其下）")
    parser.add_argument("--model-dir", default = os.environ.get("MODEL_DIR", str(REPO / "qwen3.5-2b-mllm")), help = "2B 融合模型目录（分词器已按 #77 修正）")
    parser.add_argument("--runs-root", default = None, help = "下载的运行目录的上级，默认 <输出目录>/runs")
    parser.add_argument("--out-dir", default = None, help = "输出目录，默认 <工作目录>/data/T71c")
    parser.add_argument("--fleurs-dir", default = None, help = "FLEURS 评测清单目录，默认 <工作目录>/data/T02")
    parser.add_argument("--replay-dir", default = None, help = "回放数据目录，默认 <数据集目录>/text/qwen3.5-asr/replay-noloop")
    parser.add_argument("--py-audio", default = None, help = "fleurs、probe 用的 python，默认 <工作目录>/envs/mllm-eval/bin/python")
    parser.add_argument("--py-text", default = None, help = "lm_ppl、replay_gap、think 用的 python，默认 <工作目录>/envs/mllm-train/bin/python")
    parser.add_argument("--gpu-lock", default = None, help = "GPU 锁文件，默认 <工作目录>/gpu.lock")
    parser.add_argument("--runs", default = ",".join(c_runs.RUNS), help = "逗号分隔的运行，默认全部六个")
    parser.add_argument("--no-ref", action = "store_true", help = "不评测参考模型")
    parser.add_argument(
        "--order",
        default = "item",
        choices = ("item", "run"),
        help = "item：按评测项目排列（默认）；run：与 Colab 相同",
    )
    parser.add_argument(
        "--think-max-new",
        type = int,
        default = c_runs.THINK_MAX_NEW,
        help = "思考检查的生成上限（各模型要用同一个）",
    )
    parser.add_argument(
        "--busy-mib",
        type = int,
        default = 3000,
        help = "已用显存超过它时等待（MiB）",
    )
    parser.add_argument(
        "--timeout-factor",
        type = float,
        default = 3.0,
        help = "单项超过估计用时的这么多倍（至少 30 分钟）时结束它",
    )
    parser.add_argument("--dry-run", action = "store_true", help = "只检查输入、列出计划与时间估计")
    args = parser.parse_args(args = argv)
    keys = [key for key in args.runs.split(",") if key]
    unknown = [key for key in keys if key not in c_runs.RUNS]
    if unknown:
        parser.error(f"未知的运行：{unknown}，可选 {list(c_runs.RUNS)}")
    if not args.dataset and not args.replay_dir:
        parser.error("需要 --dataset（或环境变量 DATASET）或 --replay-dir")
    args.keys = keys
    return args


def main(argv = None):
    args = parse_args(argv = argv)
    paths = resolve_paths(args = args)
    keys = args.keys
    include_ref = not args.no_ref
    if args.dry_run:
        return dry_run(
            paths = paths,
            keys = keys,
            order = args.order,
            include_ref = include_ref,
        )
    paths["out_dir"].mkdir(parents = True, exist_ok = True)
    driver = Driver(paths = paths, busy_mib = args.busy_mib, timeout_factor = args.timeout_factor)

    def on_signal(signum, frame):
        raise Stop(signum = signum)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    driver.set_status(
        pid = os.getpid(),
        started = time.strftime("%Y-%m-%d %H:%M:%S"),
        phase = "start",
        current = None,
        order = args.order,
        think_max_new = args.think_max_new,
    )
    driver.log(f"开始（pid {os.getpid()}，顺序 {args.order}，运行 {keys}，参考模型 {'评测' if include_ref else '不评测'}）")
    failed_inputs = [text for ok, text in check_inputs(paths = paths, check_python = True) if not ok]
    if failed_inputs:
        driver.log(f"缺少基本输入：{failed_inputs}")
        driver.set_status(phase = "error", error = failed_inputs)
        return 2
    failures = []

    def current_plan():
        """重新检查下载并生成计划，返回 (runs, plan)。"""
        runs = discover_runs(runs_root = paths["runs_root"], keys = keys)
        driver.set_status(
            runs_ready = [key for key in keys if runs[key]["ready"]],
            runs_missing = {key: runs[key]["missing"] for key in keys if not runs[key]["ready"]},
        )
        plan = build_plan(
            runs = runs,
            keys = keys,
            paths = paths,
            order = args.order,
            include_ref = include_ref,
        )
        return runs, plan

    def summarize(runs, with_report):
        write_summary(
            paths = paths,
            runs = runs,
            keys = keys,
            with_report = with_report,
            log = driver.log,
        )

    def run(entries, plan, runs):
        for entry in entries:
            result = driver.run_entry(entry = entry, plan = plan)
            if result not in ("ok", "skipped"):
                failures.append(f"{entry['label']}:{entry['item']}")
            if result == "ok":
                # FLEURS 的 report 要重新打分全部运行，只在 FLEURS 做完一项时重算
                summarize(runs = runs, with_report = entry["item"] == "fleurs")

    try:
        if args.order == "run":
            runs, plan = current_plan()
            driver.log(f"计划 {len(plan)} 项，估计剩余 {driver.remaining_minutes(plan = plan) / 60:.1f} 小时")
            run(entries = plan, plan = plan, runs = runs)
        else:
            for item in ITEM_ORDER:
                # 每种项目开始前重新检查下载，中途下载完成的运行在后面的项目中补上
                runs, plan = current_plan()
                entries = [entry for entry in plan if entry["item"] == item]
                pending = [entry["label"] for entry in entries if not entry["marker"].exists()]
                driver.log(f"{item}：要做 {pending or '无'}；估计全部剩余 {driver.remaining_minutes(plan = plan) / 60:.1f} 小时")
                run(entries = entries, plan = plan, runs = runs)
        runs = discover_runs(runs_root = paths["runs_root"], keys = keys)
        summarize(runs = runs, with_report = True)
    except Stop as stop:
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
    missing = [key for key in keys if not runs[key]["ready"]]
    summary = paths["out_dir"] / "summary" / "summary.md"
    if failures:
        driver.log(f"结束，有失败的项目：{failures}（重新运行只重做这些）；汇总 {summary}")
        driver.set_status(phase = "done_with_failures", failed = failures, current = None)
        return 1
    if missing:
        driver.log(f"能做的都做完了；还没下载的运行：{missing}（下载后重新运行）；汇总 {summary}")
        driver.set_status(phase = "done_partial", current = None)
        return 3
    driver.log(f"全部完成；汇总 {summary}")
    driver.set_status(phase = "done", current = None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
