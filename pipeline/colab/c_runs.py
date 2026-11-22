"""#71 C 组第一轮（纯 CPT 全量微调 + 自生成回放）在 Colab 上的无人值守流程。

用户 2026-11-23 的安排（2B，CPT 格式，AdamW，拼接批次，分词器已按 #77 修正）：

0. 检查输入（模型、分片索引、回放数据、FLEURS 评测清单），打印时间与算力的估计。
1. 冒烟测试（smoke）：回放的单元测试（在 GPU 上，含拼接内核下音频与文本同一行的检查）；C-mix（五组等量，ntp）训练
   100 步，第 1 步另加最长的回放样本测峰值显存（--replay-probe-longest），第 50 步停下再续训到第 100 步，然后按训练配置
   重新模拟抽样器核对续训后取到的样本（pipeline.train.replay verify-log）；C-mix 用 kl32（加尾部项）训练 20 步。任何一项
   失败就停下，长训练不开始。
2. projector：CPT 格式的 2B projector 从随机初始化训练（设置同 t71 的 crit：峰值 3e-3、预热 12,000 步、每 100 步保存），
   训练损失突降的那一步（突降规则同 t71_cpt_run.drop_step）的检查点作为全量微调的起点，随即停止。到第 15,000 步仍未突降
   视为阻塞点。
3. 六个运行依次全量微调 3 万步（语言模型 1e-5、projector 1e-4、预热 1,000 步、每 2,500 步 dev 损失与检查点）并评测：
   C0（不加回放，新的对照）、C-free_raw、C-free_chat、C-cont_raw、C-cont_chat（单一组）、C-mix（五组按 token 等量，含
   free_chat_think），回放比例 r = 0.5，ntp。ASR 数据与 B 组完全相同（同一份清单、同一个训练计划、同一批分片）。
   评测用第 30,000 步（最后）的检查点：
   - FLEURS test 18 种语言，CPT 格式提示（run_eval --prompt-format cpt），汇总时给出错误率与解码失控比例（report）；
   - 指令遵循探针（instruct_probe，聊天格式提问，另用 CPT 格式转写）；
   - 语言模型对 FLEURS 参考文本的每字困惑度（lm_ppl）；
   - 回放差距：五组回放评测集上的每 token NLL 减去参考模型的（replay_gap，前向 KL 估计）；
   - 思考检查：free_chat_think 评测集的 200 条提问开启思考生成（think_check，温度 1、上限见 C71_THINK_MAX_NEW）。
4. 参考模型（未训练的 2B 语言模型）：思考检查、回放差距（应接近 0，核对评测实现）、困惑度，各一次。
5. 汇总：report 的 FLEURS 表格加各运行的回放差距、困惑度、思考检查、探针要点，写到 Drive 的 runs/c71-summary/。

无人值守与续跑：
- 状态写在 $WORK_DIR/c71/status.json，日志 driver.log，两者随时复制到 Drive 的 runs/c71-status/。已完成的阶段记在
  status 的 done 中，重新运行驱动程序时跳过；换了新的 Colab 会话（本地盘清空）时从 Drive 取回状态。
- 训练输出在本地盘，后台每 10 分钟把检查点（projector、日志、回放日志，不含 trainer_state.pt 和
  language_model.safetensors）同步到 Drive 的 runs/<运行名>/，本地只保留最近一个检查点的 language_model.safetensors。
  训练结束或中途停下时，把最近检查点的 trainer_state.pt 复制到 Drive（Drive 上每个运行只留这一份，较早的删除），
  训练完成时再复制最后检查点的 language_model.safetensors。C71_STATE_SYNC_STEPS 大于 0 时，训练中每隔这么多步也把最新的
  trainer_state.pt 同步一次（约 26 GB，Drive 删除的旧文件会进回收站占空间，默认不做）。
- 会话时限：从驱动程序启动起算 C71_DEADLINE_HOURS（默认 23 小时）。训练用 --stop-after-minutes 在时限前保存并停下，
  评测开始前估计放不下就不开始；这时把状态写成 paused、同步到 Drive 后退出（退出码 75），下一个会话重新运行即可从
  Drive 上的 trainer_state.pt 续训。会话意外断开时，正在训练的运行从 Drive 上最近的 trainer_state.pt 续训，没有时
  把 Drive 上的这个运行目录改名为 <运行名>-aborted-<时间> 后从头开始（不删除）。
- 阻塞性错误（冒烟测试失败、训练非正常退出、找不到突降点）时同步并停下，状态为 blocked；单项评测失败只记下，继续。

环境变量（与 t71_cpt_run.py 相同）：REPO、WORK_DIR、DRIVE_DIR、DATASET、PY、INDEX、SHARDS_DRIVE、SHARDS_LOCAL；
回放数据须已由 setup.sh 解到 $DATASET/text/qwen3.5-asr/replay-noloop，FLEURS test 已由 fleurs_prepare 准备在
$WORK_DIR/data/T02。可选：C71_TAG（输出名后缀）、C71_RATIO（默认 0.5）、C71_REPLAY_NORM（asr 或 step，默认 asr，按 token 归一化）、
C71_RUNS（逗号分隔，只跑这些运行，默认全部）、
C71_SKIP_SMOKE=1、C71_THINK_MAX_NEW（默认 16384）、C71_DEADLINE_HOURS、C71_CU_PER_HOUR（默认 10，#65 实测的 G4 消耗）、
C71_STATE_SYNC_STEPS、C71_ESTIMATE_ONLY=1（只打印估计后退出）、C71_TRAIN_ONLY=1（只训练：跳过各运行的评测、参考模型的评测和
汇总，训练全部完成后状态为 trained；以后去掉它重新运行驱动程序，只补做评测）。

用法（Colab 单元格中，环境变量已设置，已运行 setup.sh、fleurs_prepare；见 pipeline/colab/README.md「#71 C 组」）：

    subprocess.Popen([os.environ["PY"], "-m", "pipeline.colab.c_runs"], cwd = REPO, start_new_session = True, ...)
"""

import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ENV = os.environ
REPO = Path(ENV.get("REPO", "."))
WORK = Path(ENV.get("WORK_DIR", "/content/work"))
DRIVE = Path(ENV.get("DRIVE_DIR", "/content/drive/MyDrive"))
DATASET = ENV.get("DATASET", "")
PY = ENV.get("PY", sys.executable)
MODEL = REPO / "qwen3.5-2b-mllm"
MANIFEST = f"{DATASET}/audio/qwen3.5-asr/v2/train/manifest-v2b.jsonl"
NOISE = f"{DATASET}/audio/qwen3.5-asr/nonspeech/v2/background.jsonl"
REPLAY_DIR = f"{DATASET}/text/qwen3.5-asr/replay-noloop"
FLEURS = WORK / "data" / "T02"
TAG = ENV.get("C71_TAG", "")
STAGE = WORK / ("c71" + TAG)
STATUS = STAGE / "status.json"
LOG = STAGE / "driver.log"
DRIVE_STATUS = DRIVE / "runs" / ("c71-status" + TAG)
EVAL_ROOT = WORK / "eval" / ("c71" + TAG)
RATIO = ENV.get("C71_RATIO", "0.5")
# 一步损失的分母（train.py --replay-norm）：step 是用户 2026-11-23 的决定；asr 见 pipeline/train/replay.py「损失归一化与实际权重」。
REPLAY_NORM = ENV.get("C71_REPLAY_NORM", "asr")
THINK_MAX_NEW = int(ENV.get("C71_THINK_MAX_NEW", "16384"))
DEADLINE_HOURS = float(ENV.get("C71_DEADLINE_HOURS", "23"))
CU_PER_HOUR = float(ENV.get("C71_CU_PER_HOUR", "10"))
STATE_SYNC_STEPS = int(ENV.get("C71_STATE_SYNC_STEPS", "0"))
# 只训练（用户 2026-11-23：先训练，评测以后在本机或其他平台另做）
TRAIN_ONLY = ENV.get("C71_TRAIN_ONLY", "0") == "1"
# 名字中的 cpt77：CPT 格式、#77 修正后的分词器，与 t71 的输出（旧分词器）区分开。
PROJECTOR_NAME = "v2b-2b-cpt77-projector" + TAG
PROJECTOR_SAVE_EVERY = 100
PROJECTOR_EVAL_EVERY = 2500
MAX_PROJECTOR_STEPS = 15000
FULL_STEPS = 30000
SAVE_EVERY = 2500
ALL_GROUPS = "free_raw=1,free_chat=1,cont_raw=1,cont_chat=1,free_chat_think=1"
RUNS = {
    "c0": None,
    "c-free_raw": "free_raw=1",
    "c-free_chat": "free_chat=1",
    "c-cont_raw": "cont_raw=1",
    "c-cont_chat": "cont_chat=1",
    "c-mix": ALL_GROUPS,
}
SELECTED = [key for key in ENV.get("C71_RUNS", ",".join(RUNS)).split(",") if key]
# 训练结束、停下后留给同步 trainer_state.pt（约 26 GB）和写状态的时间（分钟）。
RESERVE_MINUTES = 40
STARTED = time.time()

# 时间估计（分钟）。训练速度取 t71 crit（CPT 格式，拼接批次，G4）：每秒约 5.5 步、3 万步约 100 分钟，另加每 2,500 步的
# dev 损失和检查点；回放 r = 0.5 时语言模型一侧多出约 10–15% 的 token，另有长样本的纯文本微批，估计慢 15%。冒烟测试后按
# 实测速度重估。评测按本机与 G4 的已有用时估计。
ESTIMATE = {
    "shards_first_copy": 18,
    "smoke": 15,
    "projector": 12,
    "train_c0": 110,
    "train_replay": 127,
    "eval_fleurs": 15,
    "eval_probe": 15,
    "eval_lm_ppl": 3,
    "eval_replay_gap": 6,
    "eval_think": 16 * THINK_MAX_NEW / 16384 + 2,
}


def train_log(out):
    """训练日志 train_log.jsonl 的各行（正在写入的最后一行跳过）。与 t71_cpt_run.py 相同。"""
    path = Path(out) / "train_log.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding = "utf-8").splitlines():
        try:
            rows.append(json.loads(s = line))
        except json.JSONDecodeError:
            pass
    return rows


def drop_step(rows):
    """训练损失突降的步数：第 1,000 步以后，第一次低于第 500–1,500 步损失中位数的 0.6 倍；没有则返回 None。与 t71 相同。"""
    early = sorted(row["loss"] for row in rows if 500 <= row["step"] <= 1500)
    if len(early) < 3:
        return None
    plateau = early[len(early) // 2]
    for row in rows:
        if row["step"] >= 1000 and row["loss"] < 0.6 * plateau:
            return row["step"]
    return None


def checkpoints(out):
    """已经完整保存的检查点步数（有 projector.safetensors 的 checkpoint-<步数> 目录）。"""
    steps = []
    for path in Path(out).glob(pattern = "checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match and (path / "projector.safetensors").exists():
            steps.append(int(match.group(1)))
    return sorted(steps)


def trainer_pid(out):
    """训练程序本身的进程号（以虚拟环境的 python 开头，不匹配外层的 flock）。"""
    result = subprocess.run(args = ["pgrep", "-f", f"^{PY} -m pipeline.train.train .*{out}"], capture_output = True, text = True)
    pids = [int(pid) for pid in result.stdout.split()]
    return pids[0] if pids else None


class Pause(Exception):
    """会话时限前停下，下一个会话接着做。"""


class Blocked(Exception):
    """阻塞性错误：停下等协调者处理。"""


def log(message):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush = True)
    with open(file = LOG, mode = "a", encoding = "utf-8") as handle:
        handle.write(line + "\n")


def read_status():
    return json.loads(s = STATUS.read_text(encoding = "utf-8")) if STATUS.exists() else {}


def set_status(**fields):
    status = read_status()
    status.update(fields)
    status["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    STATUS.write_text(data = json.dumps(obj = status, ensure_ascii = False, indent = 1), encoding = "utf-8")
    push_status()


def mark_done(key, **fields):
    status = read_status()
    done = status.get("done", [])
    if key not in done:
        done.append(key)
    set_status(done = done, **fields)


def is_done(key):
    return key in read_status().get("done", [])


def push_status():
    DRIVE_STATUS.mkdir(parents = True, exist_ok = True)
    for path in (STATUS, LOG):
        if path.exists():
            shutil.copy2(src = path, dst = DRIVE_STATUS / path.name)


def restore_status():
    """新会话本地没有状态时，从 Drive 取回。"""
    STAGE.mkdir(parents = True, exist_ok = True)
    for name in ("status.json", "driver.log"):
        remote = DRIVE_STATUS / name
        if not (STAGE / name).exists() and remote.exists():
            shutil.copy2(src = remote, dst = STAGE / name)


def remaining_minutes():
    return DEADLINE_HOURS * 60 - (time.time() - STARTED) / 60


def need(minutes, what):
    """剩余时间放不下 minutes 分钟的 what 时停下（Pause）。"""
    left = remaining_minutes()
    if left < minutes + 10:
        raise Pause(f"会话剩余约 {left:.0f} 分钟，不够{what}（估计 {minutes:.0f} 分钟）")


def estimate(train_minutes = None):
    """各阶段的估计用时与算力，没有完成的阶段计入合计。"""
    train_c0 = ESTIMATE["train_c0"] if train_minutes is None else train_minutes[0]
    train_replay = ESTIMATE["train_replay"] if train_minutes is None else train_minutes[1]
    per_eval = sum(ESTIMATE[key] for key in ESTIMATE if key.startswith("eval_"))
    stages = {"smoke": ESTIMATE["shards_first_copy"] + ESTIMATE["smoke"], "projector": ESTIMATE["projector"]}
    for key in SELECTED:
        stages[f"train:{key}"] = train_c0 if RUNS[key] is None else train_replay
        if not TRAIN_ONLY:
            stages[f"eval:{key}"] = per_eval
    if not TRAIN_ONLY:
        stages["ref"] = ESTIMATE["eval_think"] + ESTIMATE["eval_replay_gap"] + ESTIMATE["eval_lm_ppl"]
    remaining = {key: value for key, value in stages.items() if not is_done(key)}
    hours = sum(remaining.values()) / 60
    return {
        "stages_minutes": {key: round(number = value) for key, value in stages.items()},
        "remaining_hours": round(number = hours, ndigits = 1),
        "remaining_compute_units": round(number = hours * CU_PER_HOUR),
        "cu_per_hour": CU_PER_HOUR,
    }


def background(command, log_file, env = None):
    handle = open(file = log_file, mode = "a")
    return subprocess.Popen(
        args = command,
        cwd = REPO,
        stdin = subprocess.DEVNULL,
        stdout = handle,
        stderr = subprocess.STDOUT,
        start_new_session = True,
        env = env,
    )


def run_logged(command, log_file, env = None):
    """前台运行，输出追加到 log_file，返回退出码。"""
    with open(file = log_file, mode = "a") as handle:
        return subprocess.run(
            args = command,
            cwd = REPO,
            stdout = handle,
            stderr = subprocess.STDOUT,
            env = env,
            check = False,
        ).returncode


def window(out, ahead_steps = 10000):
    """为 out 维护分片窗口（先停掉旧的窗口程序）。领先 1 万步、预算 80 GB，避免 t71 时窗口预取过远占满本地盘。"""
    subprocess.run(args = ["pkill", "-f", "pipeline.colab.shard_window run"], check = False)
    common = [
        "--index", ENV["INDEX"],
        "--source", ENV["SHARDS_DRIVE"],
        "--pool-dir", ENV["SHARDS_LOCAL"] + "/pool",
        "--window-dir", ENV["SHARDS_LOCAL"] + "/window",
        "--output-dir", str(out),
    ]
    code = run_logged(command = [PY, "-m", "pipeline.colab.shard_window", "prepare"] + common, log_file = WORK / "shard_window.log")
    if code != 0:
        raise Blocked(f"分片窗口准备失败（退出码 {code}），见 shard_window.log")
    return background(
        command = [PY, "-m", "pipeline.colab.shard_window", "run"] + common + [
            "--ahead-steps", str(ahead_steps),
            "--disk-budget-gb", "80",
        ],
        log_file = WORK / "shard_window.log",
    )


def train_command(out, extra):
    return [
        "pipeline/train/run.sh", MANIFEST, "original", str(out),
        "--model-dir", str(MODEL),
        "--prompt-format", "cpt",
        "--noise-manifest", NOISE,
        "--audio-shards", ENV["INDEX"],
        "--audio-shard-dirs", ENV["SHARDS_LOCAL"] + "/window:" + ENV["SHARDS_LOCAL"] + "/pool",
        "--audio-shard-fallback", ENV["SHARDS_DRIVE"],
        "--num-workers", "16",
        "--pack-sequences",
        "--rows-cache-dir", str(WORK / "rows" / "v2b-2b"),
    ] + extra


def replay_arguments(spec, loss = "ntp"):
    if spec is None:
        return []
    return [
        "--replay-dir", REPLAY_DIR,
        "--replay-groups", spec,
        "--replay-ratio", RATIO,
        "--replay-loss", loss,
        "--replay-norm", REPLAY_NORM,
    ]


def full_arguments(init):
    return [
        "--train-language-model", "--language-model-lr", "1e-5", "--lr", "1e-4", "--warmup-steps", "1000",
        "--max-steps", str(FULL_STEPS), "--eval-every", str(SAVE_EVERY), "--save-every", str(SAVE_EVERY),
        "--init-projector", str(init),
    ]


def progress(out):
    try:
        return json.loads(s = (Path(out) / "progress.json").read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return {}


def rsync(source, target, excludes = ()):
    Path(target).mkdir(parents = True, exist_ok = True)
    command = ["rsync", "-a"]
    for pattern in excludes:
        command += ["--exclude", pattern]
    return subprocess.run(args = command + [f"{source}/", f"{target}/"], check = False).returncode


# 定期同步时不传的文件：续训状态与语言模型权重只在停下、结束时复制；latest 只在 Drive 上有对应的续训状态时才写。
PERIODIC_EXCLUDES = ("trainer_state.pt", "language_model.safetensors", "dataset", "latest", "*.tmp")


def prune_local(out):
    """本地只留最近一个检查点的 language_model.safetensors（评测用最后一个检查点，较早的不再需要）。"""
    steps = checkpoints(out = out)
    for step in steps[:-1]:
        path = Path(out) / f"checkpoint-{step:07d}" / "language_model.safetensors"
        if path.exists():
            path.unlink()


def push_state(out, name, with_model = False):
    """把最近检查点的 trainer_state.pt（与 with_model 时的 language_model.safetensors）复制到 Drive，再写 Drive 上的
    latest，删掉 Drive 上较早检查点的 trainer_state.pt。"""
    out = Path(out)
    target = DRIVE / "runs" / name
    rsync(source = out, target = target, excludes = PERIODIC_EXCLUDES)
    if not (out / "latest").exists():
        return None
    latest = (out / "latest").read_text().strip()
    names = ["trainer_state.pt"] + (["language_model.safetensors"] if with_model else [])
    for file_name in names:
        local = out / latest / file_name
        if local.exists():
            remote = target / latest / file_name
            remote.parent.mkdir(parents = True, exist_ok = True)
            if not remote.exists() or remote.stat().st_size != local.stat().st_size:
                shutil.copy2(src = local, dst = str(remote) + ".tmp")
                os.replace(src = str(remote) + ".tmp", dst = remote)
    (target / "latest").write_text(data = latest + "\n")
    for state in target.glob(pattern = "checkpoint-*/trainer_state.pt"):
        if state.parent.name != latest:
            state.unlink()
    return latest


def sync_loop(out, name, stop):
    """训练期间每 10 分钟同步一次（不含大文件）并清理本地较早的语言模型权重；C71_STATE_SYNC_STEPS 大于 0 时按步数同步续训状态。"""
    last_state_step = 0
    while not stop.wait(timeout = 600):
        if rsync(source = out, target = DRIVE / "runs" / name, excludes = PERIODIC_EXCLUDES) == 0:
            prune_local(out = out)
        steps = checkpoints(out = out)
        if STATE_SYNC_STEPS > 0 and steps and steps[-1] >= last_state_step + STATE_SYNC_STEPS:
            push_state(out = out, name = name)
            last_state_step = steps[-1]


def restore_run(out, name):
    """新会话中本地没有这个运行的输出时，从 Drive 取回可以续训的部分。返回是否取回。"""
    out = Path(out)
    remote = DRIVE / "runs" / name
    if (out / "progress.json").exists() or not remote.exists():
        return False
    states = sorted(remote.glob(pattern = "checkpoint-*/trainer_state.pt"))
    if not states:
        aside = remote.with_name(f"{name}-aborted-{time.strftime('%Y%m%d-%H%M%S')}")
        remote.rename(target = aside)
        log(f"{name}：Drive 上没有可续训的 trainer_state.pt，原目录改名为 {aside.name}，从头开始")
        return False
    latest = states[-1].parent.name
    rsync(source = remote, target = out, excludes = ("language_model.safetensors",))
    (out / "latest").write_text(data = latest + "\n")
    log(f"{name}：从 Drive 取回，续训起点 {latest}")
    return True


def run_training(name, extra, stop_check = None):
    """训练一个运行（续训安全），返回 progress。stop_check(out) 返回真时向训练程序发 SIGTERM（projector 阶段用）。"""
    out = WORK / "runs" / name
    out.mkdir(parents = True, exist_ok = True)
    restore_run(out = out, name = name)
    if progress(out = out).get("status") == "finished":
        log(f"{name} 已训练完成")
        return progress(out = out)
    minutes = int(remaining_minutes() - RESERVE_MINUTES)
    if minutes < 30:
        raise Pause(f"会话剩余时间不够训练 {name}")
    window_proc = window(out = out)
    stop_sync = threading.Event()
    sync_thread = threading.Thread(target = sync_loop, kwargs = {"out": out, "name": name, "stop": stop_sync}, daemon = True)
    sync_thread.start()
    proc = background(
        command = train_command(out = out, extra = extra + ["--stop-after-minutes", str(minutes)]),
        log_file = WORK / f"{name}.log",
    )
    stopped_early = False
    while proc.poll() is None:
        time.sleep(30)
        if stop_check is not None and not stopped_early and stop_check(out):
            pid = trainer_pid(out = out)
            if pid:
                os.kill(pid, signal.SIGTERM)
            stopped_early = True
    stop_sync.set()
    sync_thread.join(timeout = 900)
    window_proc.terminate()
    state = progress(out = out)
    push_state(out = out, name = name, with_model = state.get("status") == "finished")
    shutil.copy2(src = WORK / f"{name}.log", dst = DRIVE / "runs" / name / f"{name}.log")
    state["exit_code"] = proc.returncode
    state["stopped_early"] = stopped_early
    return state


def train_stats(out):
    """训练日志的要点：最后的 dev 损失、平均速度、峰值显存、回放的实际比例。"""
    rows = train_log(out = out)
    if not rows:
        return {}
    speeds = [row["it_per_s"] for row in rows if row.get("it_per_s")]
    stats = {
        "last_step": rows[-1]["step"],
        "it_per_s": round(number = sum(speeds) / len(speeds), ndigits = 3) if speeds else None,
        "max_mem_gb": max(row.get("mem_gb", 0) for row in rows),
        "train_loss": rows[-1].get("loss"),
    }
    for key in ("replay_loss", "replay_ratio_total", "replay_grad_share"):
        if key in rows[-1]:
            stats[key] = rows[-1][key]
    eval_path = Path(out) / "eval_log.jsonl"
    if eval_path.exists():
        lines = [json.loads(s = line) for line in eval_path.read_text(encoding = "utf-8").splitlines() if line.strip()]
        if lines:
            stats["dev_loss"] = round(number = lines[-1]["overall"]["loss"], ndigits = 4)
            stats["dev_step"] = lines[-1]["step"]
    return stats


def has_replay_loss(rows):
    """日志中至少一行有有限的回放损失。回放按预算整条取样，某一段日志（例如 kl32 冒烟测试的第 11–20 步）可能正好没有取到回放，
    只看最后一行会误判。"""
    return any(row.get("replay_loss") is not None and math.isfinite(row["replay_loss"]) for row in rows)


def phase_smoke():
    log("冒烟测试：回放单元测试（GPU）")
    smoke = STAGE / "smoke"
    smoke.mkdir(parents = True, exist_ok = True)
    env = dict(ENV, REPLAY_DIR = REPLAY_DIR)
    code = run_logged(
        command = [PY, "-m", "unittest", "pipeline.train.tests.test_replay", "pipeline.eval.tests.test_replay_evals"],
        log_file = smoke / "unittest.log",
        env = env,
    )
    if code != 0:
        raise Blocked("回放单元测试失败，见 smoke/unittest.log")
    results = {}
    # 1. C-mix ntp：100 步，第 50 步停下再续训；第一步另加最长样本测峰值显存
    out = WORK / "smoke-c71" / "mix-ntp"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents = True)
    window_proc = window(out = out)
    base = train_command(out = out, extra = [
        "--train-language-model", "--language-model-lr", "1e-5", "--lr", "1e-4", "--warmup-steps", "10",
        "--max-steps", "100", "--eval-every", "0", "--save-every", "50", "--log-every", "10",
    ] + replay_arguments(spec = ALL_GROUPS))
    first = run_logged(command = base + ["--stop-at-step", "50", "--replay-probe-longest"], log_file = smoke / "mix-ntp.log")
    second = run_logged(command = base, log_file = smoke / "mix-ntp.log")
    rows = train_log(out = out)
    results["ntp"] = {
        "exit_codes": [first, second],
        "status": progress(out = out).get("status"),
        "probe_step_mem_gb": rows[0].get("mem_gb") if rows else None,
        **train_stats(out = out),
    }
    verify = subprocess.run(
        args = [PY, "-m", "pipeline.train.replay", "verify-log", str(out)],
        cwd = REPO,
        capture_output = True,
        text = True,
        check = False,
    )
    results["ntp"]["verify_log"] = verify.stdout.strip()[-500:]
    bad = (
        first != 75 or second != 0 or results["ntp"]["status"] != "finished" or verify.returncode != 0
        or not has_replay_loss(rows = rows)
    )
    # 2. C-mix kl32（加尾部项）：20 步
    out_kl = WORK / "smoke-c71" / "mix-kl"
    if out_kl.exists():
        shutil.rmtree(out_kl)
    out_kl.mkdir(parents = True)
    code = run_logged(command = train_command(out = out_kl, extra = [
        "--train-language-model", "--language-model-lr", "1e-5", "--lr", "1e-4", "--warmup-steps", "10",
        "--max-steps", "20", "--eval-every", "0", "--save-every", "20", "--log-every", "10", "--replay-kl-tail",
    ] + replay_arguments(spec = ALL_GROUPS, loss = "kl32")), log_file = smoke / "mix-kl.log")
    window_proc.terminate()
    rows_kl = train_log(out = out_kl)
    results["kl32"] = {"exit_code": code, "status": progress(out = out_kl).get("status"), **train_stats(out = out_kl)}
    replay_rows = [row for row in rows_kl if row.get("replay_loss") is not None]
    if replay_rows:
        results["kl32"]["replay_loss"] = replay_rows[-1]["replay_loss"]
        results["kl32"]["replay_nll"] = replay_rows[-1].get("replay_nll")
    bad = bad or code != 0 or not has_replay_loss(rows = rows_kl)
    for directory in (out, out_kl):
        rsync(source = directory, target = DRIVE_STATUS / "smoke" / directory.name, excludes = ("trainer_state.pt", "language_model.safetensors", "dataset"))
    rsync(source = smoke, target = DRIVE_STATUS / "smoke")
    log(f"冒烟测试结果：{json.dumps(obj = results, ensure_ascii = False)}")
    if bad:
        set_status(smoke = results)
        raise Blocked("冒烟测试没有通过，见 smoke/ 下的日志")
    # 按实测速度重估回放运行的训练用时：取最后一个日志窗口（第 91–100 步）的速度，另加 dev 损失与检查点约 15 分钟
    speed = rows[-1].get("it_per_s") if rows else None
    if speed:
        ESTIMATE["train_replay"] = FULL_STEPS / speed / 60 + 15
    for directory in (out, out_kl):
        shutil.rmtree(directory, ignore_errors = True)
    mark_done("smoke", smoke = results, smoke_it_per_s = speed)


def phase_projector():
    log("projector：CPT 格式，从随机初始化开始，取训练损失突降那一步的检查点")
    found = {}

    def stop_check(out):
        rows = train_log(out = out)
        if "step" not in found:
            step = drop_step(rows = rows)
            if step is not None:
                found["step"] = step
                log(f"训练损失在第 {step} 步突降")
        if "step" in found:
            return any(step >= found["step"] for step in checkpoints(out = out))
        return False

    out = WORK / "runs" / PROJECTOR_NAME
    if out.exists() and progress(out = out).get("status") != "finished":
        # 上次没跑完（例如会话中断）：projector 阶段很短，从头再来，旧的输出改名保留
        aside = out.with_name(f"{PROJECTOR_NAME}-aborted-{time.strftime('%Y%m%d-%H%M%S')}")
        out.rename(target = aside)
    remote = DRIVE / "runs" / PROJECTOR_NAME
    if remote.exists():
        remote.rename(target = remote.with_name(f"{PROJECTOR_NAME}-aborted-{time.strftime('%Y%m%d-%H%M%S')}"))
    need(minutes = ESTIMATE["projector"], what = "训练 projector")
    state = run_training(
        name = PROJECTOR_NAME,
        extra = [
            "--epochs", "1", "--warmup-steps", "12000", "--eval-every", str(PROJECTOR_EVAL_EVERY),
            "--save-every", str(PROJECTOR_SAVE_EVERY), "--stop-at-step", str(MAX_PROJECTOR_STEPS),
        ],
        stop_check = stop_check,
    )
    rows = train_log(out = out)
    if "step" not in found:
        found_step = drop_step(rows = rows)
        if found_step is None:
            raise Blocked(f"阻塞点：到第 {rows[-1]['step'] if rows else 0} 步训练损失仍未突降（退出码 {state.get('exit_code')}）")
        found["step"] = found_step
    after = [step for step in checkpoints(out = out) if step >= found["step"]]
    if not after:
        raise Blocked(f"突降在第 {found['step']} 步，但没有之后的检查点")
    chosen = after[0]
    init = DRIVE / "runs" / PROJECTOR_NAME / f"checkpoint-{chosen:07d}"
    if not (init / "projector.safetensors").exists():
        raise Blocked(f"Drive 上没有 {init}/projector.safetensors")
    mark_done(
        "projector",
        drop_step = found["step"],
        projector_checkpoint = chosen,
        projector_init = str(init),
    )
    log(f"projector 起点：第 {chosen} 步（突降在第 {found['step']} 步）")


def run_name(key):
    return f"v2b-2b-cpt77-{key}{TAG}"


def final_checkpoint(name):
    """最后一个检查点（第 30,000 步）的目录：本地有语言模型权重时用本地，否则用 Drive 上的。"""
    local = WORK / "runs" / name / f"checkpoint-{FULL_STEPS:07d}"
    if (local / "language_model.safetensors").exists():
        return local
    remote = DRIVE / "runs" / name / f"checkpoint-{FULL_STEPS:07d}"
    if (remote / "language_model.safetensors").exists():
        return remote
    raise Blocked(f"找不到 {name} 第 {FULL_STEPS} 步的 language_model.safetensors")


def phase_train(key):
    name = run_name(key = key)
    init = read_status()["projector_init"]
    log(f"训练 {key}（{name}）：从 {init} 全量微调 {FULL_STEPS} 步，回放 {RUNS[key] or '无'}")
    set_status(phase = f"train:{key}")
    state = run_training(name = name, extra = full_arguments(init = init) + replay_arguments(spec = RUNS[key]))
    stats = train_stats(out = WORK / "runs" / name)
    runs = read_status().get("runs", {})
    runs[key] = {"name": name, "status": state.get("status"), "exit_code": state.get("exit_code"), **stats}
    set_status(runs = runs)
    if state.get("status") == "finished":
        mark_done(f"train:{key}")
        log(f"{key} 训练完成：{json.dumps(obj = stats, ensure_ascii = False)}")
        return
    if state.get("status") == "stopped" and state.get("exit_code") == 75:
        raise Pause(f"{key} 在第 {state.get('step')} 步停下（时限），trainer_state 已同步到 Drive")
    raise Blocked(f"{key} 训练没有完成（{state.get('status')}，第 {state.get('step')} 步，退出码 {state.get('exit_code')}）")


def eval_items(out, projector = None, language_model = None, probe = True, fleurs = True):
    """一个模型的各项评测：[(名称, 命令, 完成标志文件, 估计分钟)]。都不给时评测参考模型（原始语言模型）。"""
    lm_args = ["--language-model", str(language_model)] if language_model is not None else []
    audio_args = (["--projector", str(projector)] if projector is not None else []) + lm_args
    items = []
    if fleurs:
        items.append(("fleurs", [
            PY, "-m", "pipeline.eval.run_eval", "--model-dir", str(MODEL), *audio_args, "--manifest-dir", str(FLEURS),
            "--batch-size", "48", "--prompt-format", "cpt", "--output-dir", str(out / "fleurs"),
        ], out / "fleurs" / "summary.json", ESTIMATE["eval_fleurs"]))
    if probe:
        items.append(("probe", [
            PY, "-m", "pipeline.eval.instruct_probe", "--model-dir", str(MODEL), *audio_args, "--manifest-dir", str(FLEURS),
            "--batch-size", "16", "--prompt-format", "chat", "cpt", "--output-dir", str(out / "probe"),
        ], out / "probe" / "summary.json", ESTIMATE["eval_probe"]))
    items += [
        ("lm_ppl", [
            PY, "-m", "pipeline.eval.lm_ppl", "--model-dir", str(MODEL), *lm_args, "--manifest-dir", str(FLEURS),
            "--output", str(out / "lm_ppl.json"),
        ], out / "lm_ppl.json", ESTIMATE["eval_lm_ppl"]),
        ("replay_gap", [
            PY, "-m", "pipeline.eval.replay_gap", "--model-dir", str(MODEL), *lm_args, "--replay-dir", REPLAY_DIR,
            "--output-dir", str(out / "replay_gap"),
        ], out / "replay_gap" / "summary.json", ESTIMATE["eval_replay_gap"]),
        ("think", [
            PY, "-m", "pipeline.eval.think_check", "--model-dir", str(MODEL), *lm_args, "--replay-dir", REPLAY_DIR,
            "--max-new-tokens", str(THINK_MAX_NEW), "--output-dir", str(out / "think"),
        ], out / "think" / "summary.json", ESTIMATE["eval_think"]),
    ]
    return items


def run_evals(label, out, remote, items):
    """依次运行评测，已有结果（本地或 Drive）的跳过；单项失败只记下，返回失败的项目。每项结束后同步到 Drive。"""
    out.mkdir(parents = True, exist_ok = True)
    if remote.exists():
        rsync(source = remote, target = out)
    results = read_status().get("evals", {})
    failed = []
    for name, command, marker, minutes in items:
        if marker.exists():
            results[f"{label}:{name}"] = "ok"
            continue
        need(minutes = minutes, what = f"评测 {label} {name}")
        log(f"评测 {label}：{name}")
        start = time.time()
        code = run_logged(command = command, log_file = out / f"{name}.log")
        ok = code == 0 and marker.exists()
        results[f"{label}:{name}"] = "ok" if ok else f"failed ({code})"
        log(f"评测 {label} {name}：{'完成' if ok else f'失败（退出码 {code}）'}，{(time.time() - start) / 60:.1f} 分钟")
        if not ok:
            failed.append(name)
        rsync(source = out, target = remote)
        set_status(evals = results)
    return failed


def phase_eval(key):
    name = run_name(key = key)
    checkpoint = final_checkpoint(name = name)
    set_status(phase = f"eval:{key}")
    failed = run_evals(
        label = key,
        out = EVAL_ROOT / key,
        remote = DRIVE / "runs" / name / "eval",
        items = eval_items(
            out = EVAL_ROOT / key,
            projector = checkpoint / "projector.safetensors",
            language_model = checkpoint / "language_model.safetensors",
        ),
    )
    # 有失败的项目时不记为完成：继续后面的运行，下次运行驱动程序时只重做失败的项目
    if not failed:
        mark_done(f"eval:{key}")


def phase_reference():
    set_status(phase = "ref")
    out = EVAL_ROOT / "ref"
    failed = run_evals(
        label = "ref",
        out = out,
        remote = DRIVE / "runs" / ("c71-ref" + TAG) / "eval",
        items = eval_items(out = out, probe = False, fleurs = False),
    )
    if not failed:
        mark_done("ref")


def load_json(path):
    try:
        return json.loads(s = Path(path).read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None


def phase_summary():
    """report 的 FLEURS 表格，加各运行的回放差距、困惑度、思考检查、探针要点。"""
    summary_dir = EVAL_ROOT / "summary"
    summary_dir.mkdir(parents = True, exist_ok = True)
    labels = SELECTED + ["ref"]
    for label in labels:
        remote = DRIVE / "runs" / (run_name(key = label) if label != "ref" else "c71-ref" + TAG) / "eval"
        if remote.exists():
            rsync(source = remote, target = EVAL_ROOT / label, excludes = ("generations.ids.i32",))
    fleurs = [f"{key}={EVAL_ROOT / key / 'fleurs'}" for key in SELECTED if (EVAL_ROOT / key / "fleurs" / "hypotheses.jsonl").exists()]
    if fleurs:
        run_logged(
            command = [PY, "-m", "pipeline.eval.report", "--out-dir", str(summary_dir), "--fleurs", *fleurs],
            log_file = summary_dir / "report.log",
        )
    status = read_status()
    lines = ["# #71 C 组第一轮汇总", "", f"projector 起点：第 {status.get('projector_checkpoint')} 步（突降在第 {status.get('drop_step')} 步）", ""]
    lines += ["| 运行 | dev 损失 | 回放比例 | 回放梯度份额 | 差距 free_raw | free_chat | cont_raw | cont_chat | free_chat_think | 各组平均 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    table = {}
    for label in labels:
        run = status.get("runs", {}).get(label, {})
        gap = load_json(path = EVAL_ROOT / label / "replay_gap" / "summary.json") or {}
        groups = gap.get("groups", {})
        cells = [f"{groups[g]['gap']:.4f}" if g in groups else "" for g in ("free_raw", "free_chat", "cont_raw", "cont_chat", "free_chat_think")]
        lines.append(
            f"| {label} | {run.get('dev_loss', '')} | {run.get('replay_ratio_total', '')} | {run.get('replay_grad_share', '')} | "
            + " | ".join(cells) + f" | {gap.get('mean_gap', '')} |"
        )
        table[label] = {"train": run, "replay_gap": gap, "lm_ppl": load_json(path = EVAL_ROOT / label / "lm_ppl.json"),
                        "think": load_json(path = EVAL_ROOT / label / "think" / "summary.json")}
        probe = load_json(path = EVAL_ROOT / label / "probe" / "summary.json")
        if probe:
            table[label]["probe"] = {
                task: {instruction: value["ALL"] for instruction, value in instructions.items()}
                for task, instructions in probe.items()
            }
    lines += ["", "| 运行 | 写完 </think> | 循环 | 到上限 | 推理中位 | 回答中位 | 每字困惑度 zh / ja / ko / th / en |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for label in labels:
        think = table[label]["think"] or {}
        ppl = table[label]["lm_ppl"] or {}
        ppl_cells = " / ".join(str(ppl.get(lang, {}).get("char_ppl", "")) for lang in ("zh", "ja", "ko", "th", "en"))
        reasoning = (think.get("reasoning_tokens") or {}).get("median", "")
        answer = (think.get("answer_tokens") or {}).get("median", "")
        lines.append(f"| {label} | {think.get('closed', '')} | {think.get('loop', '')} | {think.get('capped', '')} | {reasoning} | {answer} | {ppl_cells} |")
    if (summary_dir / "tables.md").exists():
        lines += ["", "## FLEURS test（report）", "", (summary_dir / "tables.md").read_text(encoding = "utf-8")]
    (summary_dir / "summary.md").write_text(data = "\n".join(lines) + "\n", encoding = "utf-8")
    (summary_dir / "summary.json").write_text(data = json.dumps(obj = table, ensure_ascii = False, indent = 1), encoding = "utf-8")
    rsync(source = summary_dir, target = DRIVE / "runs" / ("c71-summary" + TAG))
    mark_done("summary")


def check_inputs():
    missing = [
        str(path) for path in (
            MODEL / "config.json",
            Path(MANIFEST),
            Path(NOISE),
            Path(ENV["INDEX"]),
            FLEURS / "zh" / "test.jsonl",
            Path(REPLAY_DIR) / "templates.json",
        )
        # 只训练时不用 FLEURS test（评测才用）
        if not path.exists() and not (TRAIN_ONLY and path == FLEURS / "zh" / "test.jsonl")
    ]
    for group in ("free_raw", "free_chat", "cont_raw", "cont_chat", "free_chat_think"):
        for split in ("train", "eval"):
            for name in ("samples.jsonl", "teacher/meta.json"):
                path = Path(REPLAY_DIR) / group / split / name
                if not path.exists():
                    missing.append(str(path))
    if missing:
        raise Blocked(f"缺少输入：{missing[:8]}")


def main():
    restore_status()
    disk = shutil.disk_usage("/")
    set_status(
        pid = os.getpid(),
        session_started = time.strftime("%Y-%m-%d %H:%M:%S"),
        phase = "start",
        disk_free_gb = round(number = disk.free / 1e9),
    )
    plan = estimate()
    log(f"计划：运行 {SELECTED}，回放比例 {RATIO}；估计剩余 {plan['remaining_hours']} 小时、约 {plan['remaining_compute_units']} 个计算单元"
        f"（按每小时 {CU_PER_HOUR}）；各阶段分钟数 {plan['stages_minutes']}")
    set_status(estimate = plan)
    if ENV.get("C71_ESTIMATE_ONLY") == "1":
        return 0
    try:
        check_inputs()
        if not is_done("smoke") and ENV.get("C71_SKIP_SMOKE") != "1":
            set_status(phase = "smoke")
            need(minutes = ESTIMATE["shards_first_copy"] + ESTIMATE["smoke"], what = "冒烟测试")
            phase_smoke()
            speed = read_status().get("smoke_it_per_s")
            plan = estimate()
            log(f"冒烟测试通过（C-mix 第 91–100 步每秒 {speed} 步）；重估剩余 {plan['remaining_hours']} 小时、约 "
                f"{plan['remaining_compute_units']} 个计算单元")
            set_status(estimate = plan)
        if not is_done("projector"):
            set_status(phase = "projector")
            phase_projector()
        for key in SELECTED:
            if not is_done(f"train:{key}"):
                phase_train(key = key)
            if not TRAIN_ONLY and not is_done(f"eval:{key}"):
                phase_eval(key = key)
        if TRAIN_ONLY:
            set_status(phase = "trained")
            log("训练全部完成（只训练模式，C71_TRAIN_ONLY=1）；评测另行进行：去掉 C71_TRAIN_ONLY 重新运行驱动程序只补做评测")
            return 0
        if not is_done("ref"):
            phase_reference()
        phase_summary()
        unfinished = [key for key in [f"eval:{key}" for key in SELECTED] + ["ref"] if not is_done(key)]
        if unfinished:
            set_status(phase = "done_with_failures", unfinished = unfinished)
            log(f"训练全部完成，评测中有失败的项目：{unfinished}（见 status.json 的 evals；修好后重新运行驱动程序只重做这些）")
            return 1
        set_status(phase = "done")
        log("全部完成")
        return 0
    except Pause as reason:
        log(f"暂停：{reason}；下一个会话重新运行驱动程序即可接着做")
        set_status(phase = "paused", pause_reason = str(reason))
        return 75
    except Blocked as reason:
        log(f"阻塞：{reason}")
        set_status(phase = "blocked", error = str(reason))
        return 1
    except Exception as error:  # 任何意外都记下后停止，等协调者处理
        log(f"出错：{error!r}")
        set_status(phase = "error", error = repr(error))
        return 1
    finally:
        subprocess.run(args = ["pkill", "-f", "pipeline.colab.shard_window run"], check = False)
        push_status()


if __name__ == "__main__":
    sys.exit(main())
