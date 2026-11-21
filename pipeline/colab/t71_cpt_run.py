"""#71 B 组（纯 CPT）在 Colab 上的无人值守流程：CPT 格式的 projector → 临界点之后转全量微调 3 万步 → 评测。

用户 2026-11-22 的安排：
1. 2B projector 用 CPT 格式（--prompt-format cpt）从随机初始化训练，设置同 #61（run.sh 默认的峰值 3e-3，预热 12,000 步，
   一轮的日程）。训练损失突降后，取突降之后的第一个检查点（每 2,500 步一个）作为起点，随即停止这一阶段。到第 15,000 步
   仍未突降视为阻塞点：保存后停止，不进入下一阶段。
2. 从该起点全量微调 3 万步，设置与 #66 相同（语言模型 1e-5、projector 1e-4、预热 1,000 步、同一段数据、默认精度），
   另开拼接批次加速，格式仍为 CPT。
3. 评测：B 组成品的 FLEURS test（CPT 格式）与语言模型每字困惑度；指令遵循探针（instruct_probe）分别跑 B 组的 projector
   起点、B 组成品、A 组的 projector 起点（Drive 上的 v2b-2b 第 5,000 步）、A 组成品（#66 第 30,000 步）和原始模型的级联参照。
4. 每个阶段的输出同步到 Drive 的 runs/<名称>/；任何一步失败就停在那里，状态写进 status.json。

输出目录在本地磁盘，后台每 10 分钟把检查点（不含 trainer_state.pt）、日志同步到 Drive；每个阶段结束时再把最后的
trainer_state.pt 复制过去。

用法（Colab 单元格中，已设置 DRIVE_DIR、DATASET、WORK_DIR、REPO、SHARDS_DRIVE、SHARDS_LOCAL、INDEX、PY，并已运行
setup.sh、fleurs_prepare 与 shard_window prepare）：

    subprocess.Popen([os.environ["PY"], "-m", "pipeline.colab.t71_cpt_run"], cwd = REPO, start_new_session = True, ...)
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
REPO = Path(ENV["REPO"])
WORK = Path(ENV["WORK_DIR"])
DRIVE = Path(ENV["DRIVE_DIR"])
DATASET = ENV["DATASET"]
PY = ENV["PY"]
MODEL = REPO / "qwen3.5-2b-mllm"
MANIFEST = f"{DATASET}/audio/qwen3.5-asr/v2/train/manifest-v2b.jsonl"
NOISE = f"{DATASET}/audio/qwen3.5-asr/nonspeech/v2/background.jsonl"
FLEURS = WORK / "data" / "T02"
# 同一流程可以带后缀再跑一次（T71_TAG），例如用户 2026-11-22 要求的「从临界点本身出发」：projector 每 100 步保存，
# 取训练损失突降那一步的检查点（T71_PICK=at），只给本次的模型跑探针（T71_BASELINES=0，A 组与级联参照已在第一次跑过）。
TAG = ENV.get("T71_TAG", "")
PROJECTOR_NAME = "v2b-2b-cpt-projector" + TAG
FULL_NAME = "v2b-2b-cpt-full" + TAG
STAGE = WORK / ("t71" + TAG)
PROJECTOR_OUT = WORK / "runs" / PROJECTOR_NAME
FULL_OUT = WORK / "runs" / FULL_NAME
EVAL_OUT = WORK / "eval" / ("t71" + TAG)
A_PROJECTOR = DRIVE / "upload" / "init" / "v2b-2b-checkpoint-0005000"
A_FULL = DRIVE / "runs" / "v2b-2b-full" / "checkpoint-0030000"
MAX_PROJECTOR_STEPS = int(ENV.get("T71_MAX_PROJECTOR_STEPS", "15000"))
PROJECTOR_SAVE_EVERY = int(ENV.get("T71_PROJECTOR_SAVE_EVERY", "2500"))
PROJECTOR_EVAL_EVERY = int(ENV.get("T71_PROJECTOR_EVAL_EVERY", "2500"))
# after：突降之后至少 500 步的第一个检查点（第一次的规则）；at：突降那一步（或之后最近）的检查点
PICK = ENV.get("T71_PICK", "after")
BASELINES = ENV.get("T71_BASELINES", "1") == "1"
SAVE_EVERY = 2500
STATUS = STAGE / "status.json"
LOG = STAGE / "driver.log"


def log(message):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush = True)
    with open(file = LOG, mode = "a", encoding = "utf-8") as handle:
        handle.write(line + "\n")


def set_status(**fields):
    status = json.loads(STATUS.read_text(encoding = "utf-8")) if STATUS.exists() else {}
    status.update(fields)
    status["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    STATUS.write_text(json.dumps(status, ensure_ascii = False, indent = 1), encoding = "utf-8")


def sync(out, name, with_state = False):
    """把输出目录同步到 Drive 的 runs/<name>/；with_state 时连同最近检查点的 trainer_state.pt。"""
    target = DRIVE / "runs" / name
    target.mkdir(parents = True, exist_ok = True)
    subprocess.run(["rsync", "-a", "--exclude", "trainer_state.pt", "--exclude", "dataset", f"{out}/", f"{target}/"], check = False)
    for log_file in (WORK / f"{name}.log", WORK / "shard_window.log"):
        if log_file.exists():
            shutil.copy2(src = log_file, dst = target / log_file.name)
    if with_state and (out / "latest").exists():
        latest = (out / "latest").read_text().strip()
        state = out / latest / "trainer_state.pt"
        if state.exists():
            shutil.copy2(src = state, dst = target / latest / "trainer_state.pt")


def background(command, log_file):
    handle = open(file = log_file, mode = "a")
    return subprocess.Popen(
        args = command,
        cwd = REPO,
        stdin = subprocess.DEVNULL,
        stdout = handle,
        stderr = subprocess.STDOUT,
        start_new_session = True,
    )


def window(out):
    """为 out 维护分片窗口（先停掉旧的窗口程序）。"""
    subprocess.run(["pkill", "-f", "pipeline.colab.shard_window run"], check = False)
    subprocess.run(
        args = [PY, "-m", "pipeline.colab.shard_window", "prepare", "--index", ENV["INDEX"], "--source", ENV["SHARDS_DRIVE"],
                "--pool-dir", ENV["SHARDS_LOCAL"] + "/pool", "--window-dir", ENV["SHARDS_LOCAL"] + "/window", "--output-dir", str(out)],
        cwd = REPO,
        check = True,
        stdout = subprocess.DEVNULL,
    )
    return background(
        command = [PY, "-m", "pipeline.colab.shard_window", "run", "--index", ENV["INDEX"], "--source", ENV["SHARDS_DRIVE"],
                   "--pool-dir", ENV["SHARDS_LOCAL"] + "/pool", "--window-dir", ENV["SHARDS_LOCAL"] + "/window",
                   "--output-dir", str(out), "--ahead-steps", "31600", "--disk-budget-gb", "90"],
        log_file = WORK / "shard_window.log",
    )


def sync_and_prune(out, name, stop):
    """每 10 分钟同步到 Drive，同步后删掉本地较早检查点的 language_model.safetensors（只留最近一个），给本地磁盘留余量。"""
    while not stop.wait(timeout = 600):
        result = subprocess.run(["rsync", "-a", "--exclude", "trainer_state.pt", "--exclude", "dataset", f"{out}/",
                                 f"{DRIVE}/runs/{name}/"], check = False)
        if result.returncode != 0:
            continue
        steps = checkpoints(out = out)
        for step in steps[:-1]:
            local = out / f"checkpoint-{step:07d}" / "language_model.safetensors"
            remote = DRIVE / "runs" / name / f"checkpoint-{step:07d}" / "language_model.safetensors"
            if local.exists() and remote.exists() and remote.stat().st_size == local.stat().st_size:
                local.unlink()


def on_drive(path, name):
    """本地文件被清理时改用 Drive 上同步过去的副本。"""
    if path.exists():
        return path
    return DRIVE / "runs" / name / path.parent.name / path.name


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
    ] + extra


def train_log(out):
    path = out / "train_log.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding = "utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:  # 正在写入的最后一行
            pass
    return rows


def drop_step(rows):
    """训练损失突降的步数：第 1,000 步以后，第一次低于第 500–1,500 步损失中位数的 0.6 倍；没有则返回 None。"""
    early = sorted(row["loss"] for row in rows if 500 <= row["step"] <= 1500)
    if len(early) < 3:
        return None
    plateau = early[len(early) // 2]
    for row in rows:
        if row["step"] >= 1000 and row["loss"] < 0.6 * plateau:
            return row["step"]
    return None


def checkpoints(out):
    """已经完整保存的检查点步数（有 projector.safetensors 且没有 .tmp 后缀）。"""
    steps = []
    for path in out.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match and (path / "projector.safetensors").exists():
            steps.append(int(match.group(1)))
    return sorted(steps)


def trainer_pid(out):
    # 只匹配训练程序本身（以虚拟环境的 python 开头），不要匹配外层的 flock
    result = subprocess.run(["pgrep", "-f", f"^{PY} -m pipeline.train.train .*{out}"], capture_output = True, text = True)
    pids = [int(pid) for pid in result.stdout.split()]
    return pids[0] if pids else None


def phase_projector():
    log("阶段 1：CPT 格式的 projector，从随机初始化开始")
    set_status(phase = "projector")
    PROJECTOR_OUT.mkdir(parents = True, exist_ok = True)
    window_proc = window(out = PROJECTOR_OUT)
    proc = background(
        command = train_command(out = PROJECTOR_OUT, extra = [
            "--epochs", "1", "--warmup-steps", "12000", "--eval-every", str(PROJECTOR_EVAL_EVERY),
            "--save-every", str(PROJECTOR_SAVE_EVERY),
            "--rows-cache-dir", str(WORK / "rows" / "v2b-2b"), "--stop-at-step", str(MAX_PROJECTOR_STEPS),
        ]),
        log_file = WORK / f"{PROJECTOR_NAME}.log",
    )
    chosen = None
    found = None
    while proc.poll() is None:
        time.sleep(30)
        rows = train_log(out = PROJECTOR_OUT)
        if found is None:
            found = drop_step(rows = rows)
            if found is not None:
                log(f"训练损失在第 {found} 步突降")
                set_status(drop_step = found)
        if found is not None and chosen is None:
            after = [step for step in checkpoints(out = PROJECTOR_OUT) if step >= found + (500 if PICK == "after" else 0)]
            if after:
                chosen = after[0]
                log(f"突降之后的第一个检查点是第 {chosen} 步，停止阶段 1")
                pid = trainer_pid(out = PROJECTOR_OUT)
                if pid:
                    os.kill(pid, signal.SIGTERM)
    proc.wait()
    window_proc.terminate()
    sync(out = PROJECTOR_OUT, name = PROJECTOR_NAME, with_state = True)
    rows = train_log(out = PROJECTOR_OUT)
    set_status(projector_last_step = rows[-1]["step"] if rows else 0, projector_exit = proc.returncode)
    if chosen is None:
        if found is not None:
            after = [step for step in checkpoints(out = PROJECTOR_OUT) if step >= found]
            chosen = after[0] if after else None
        if chosen is None:
            log(f"阻塞点：到第 {rows[-1]['step'] if rows else 0} 步训练损失仍未突降（退出码 {proc.returncode}），停止")
            set_status(phase = "blocked_no_drop")
            return None
    set_status(projector_checkpoint = chosen)
    return PROJECTOR_OUT / f"checkpoint-{chosen:07d}"


def phase_full(init):
    log(f"阶段 2：从 {init.name} 全量微调 3 万步（CPT 格式）")
    set_status(phase = "full")
    FULL_OUT.mkdir(parents = True, exist_ok = True)
    window_proc = window(out = FULL_OUT)
    stop_sync = threading.Event()
    sync_thread = threading.Thread(target = sync_and_prune, kwargs = {"out": FULL_OUT, "name": FULL_NAME, "stop": stop_sync},
                                   daemon = True)
    sync_thread.start()
    proc = background(
        command = train_command(out = FULL_OUT, extra = [
            "--train-language-model", "--language-model-lr", "1e-5", "--lr", "1e-4", "--warmup-steps", "1000",
            "--max-steps", "30000", "--eval-every", str(SAVE_EVERY), "--save-every", str(SAVE_EVERY),
            "--init-projector", str(init), "--rows-cache-dir", str(WORK / "rows" / "v2b-2b"), "--stop-after-minutes", "1260",
        ]),
        log_file = WORK / f"{FULL_NAME}.log",
    )
    proc.wait()
    stop_sync.set()
    sync_thread.join(timeout = 900)
    window_proc.terminate()
    sync(out = FULL_OUT, name = FULL_NAME, with_state = True)
    progress = json.loads((FULL_OUT / "progress.json").read_text()) if (FULL_OUT / "progress.json").exists() else {}
    set_status(full_exit = proc.returncode, full_status = progress.get("status"), full_step = progress.get("step"))
    if progress.get("status") != "finished":
        log(f"阶段 2 没有完成（{progress.get('status')}，第 {progress.get('step')} 步，退出码 {proc.returncode}），停止")
        set_status(phase = "blocked_full")
        return None
    best = subprocess.run([PY, "-m", "pipeline.train.best_checkpoint", str(FULL_OUT)], cwd = REPO, capture_output = True,
                          text = True).stdout.strip()
    log(f"阶段 2 完成，dev 损失最低的检查点 {best}")
    set_status(full_best = best)
    return Path(best)


def run_step(name, command):
    log(f"评测 {name}")
    result = subprocess.run(command, cwd = REPO, capture_output = True, text = True)
    (EVAL_OUT / f"{name}.log").write_text(result.stdout[-20000:] + result.stderr[-20000:], encoding = "utf-8")
    if result.returncode != 0:
        log(f"评测 {name} 失败（退出码 {result.returncode}）")
    return result.returncode == 0


def phase_eval(projector_init, full_best):
    log("阶段 3：评测")
    set_status(phase = "eval")
    EVAL_OUT.mkdir(parents = True, exist_ok = True)
    probe = [PY, "-m", "pipeline.eval.instruct_probe", "--model-dir", str(MODEL), "--manifest-dir", str(FLEURS), "--batch-size", "16"]
    results = {}
    full_lm = on_drive(path = full_best / "language_model.safetensors", name = FULL_NAME)
    results["fleurs_cpt"] = run_step(name = "fleurs-b-full", command = [
        PY, "-m", "pipeline.eval.run_eval", "--model-dir", str(MODEL), "--projector", str(full_best / "projector.safetensors"),
        "--language-model", str(full_lm), "--manifest-dir", str(FLEURS), "--batch-size", "48",
        "--prompt-format", "cpt", "--output-dir", str(EVAL_OUT / "fleurs-b-full")])
    targets = {
        "probe-b-projector": ["--projector", str(projector_init / "projector.safetensors"), "--prompt-format", "chat", "cpt"],
        "probe-b-full": ["--projector", str(full_best / "projector.safetensors"), "--language-model", str(full_lm),
                         "--prompt-format", "chat", "cpt"],
    }
    if BASELINES:
        targets["probe-a-projector"] = ["--projector", str(A_PROJECTOR / "projector.safetensors")]
        targets["probe-a-full"] = ["--projector", str(A_FULL / "projector.safetensors"), "--language-model",
                                   str(A_FULL / "language_model.safetensors")]
        targets["probe-cascade"] = ["--cascade"]
    for name, extra in targets.items():
        results[name] = run_step(name = name, command = probe + extra + ["--output-dir", str(EVAL_OUT / name)])
        set_status(eval_results = results)
    results["lm_ppl"] = run_step(name = "lm-ppl", command = [PY, "-c", LM_PPL, str(FLEURS), str(MODEL), str(EVAL_OUT / "lm_ppl_b_full.json"),
                                                             str(full_lm)])
    set_status(eval_results = results)
    target = DRIVE / "runs" / FULL_NAME / "eval"
    target.mkdir(parents = True, exist_ok = True)
    subprocess.run(["rsync", "-a", f"{EVAL_OUT}/", f"{target}/"], check = False)


LM_PPL = '''
import json, math, sys
from pathlib import Path
import torch
from pipeline.eval.run_eval import load_model
LANGS = ["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it", "id", "ms", "th", "vi", "fil", "ar", "tr", "fa"]
fleurs, model_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
language_model = Path(sys.argv[4]) if len(sys.argv) > 4 else None
model, processor = load_model(model_dir = model_dir, projector_path = None, language_model_path = language_model)
tokenizer = processor.tokenizer
tokenizer.padding_side = "right"
device = next(model.parameters()).device
results = {}
for lang in LANGS:
    texts = [json.loads(line)["raw_transcription"] for line in open(fleurs / lang / "test.jsonl", encoding = "utf-8")]
    nll = chars = 0.0
    for start in range(0, len(texts), 32):
        batch = texts[start:start + 32]
        enc = tokenizer(batch, return_tensors = "pt", padding = True, add_special_tokens = False).to(device)
        with torch.no_grad():
            logits = model(input_ids = enc["input_ids"], attention_mask = enc["attention_mask"]).logits.float()
        logp = torch.log_softmax(logits[:, :-1], dim = -1)
        target = enc["input_ids"][:, 1:]
        mask = enc["attention_mask"][:, 1:].bool()
        nll += float((-logp.gather(dim = -1, index = target.unsqueeze(-1)).squeeze(-1))[mask].sum())
        chars += sum(len("".join(t.split())) for t in batch)
    results[lang] = {"char_ppl": round(math.exp(nll / chars), 3), "nats_per_char": round(nll / chars, 4)}
out.write_text(json.dumps(results, ensure_ascii = False, indent = 1), encoding = "utf-8")
'''


def main():
    STAGE.mkdir(parents = True, exist_ok = True)
    set_status(phase = "start", pid = os.getpid())
    try:
        projector_init = phase_projector()
        if projector_init is None:
            return 1
        full_best = phase_full(init = projector_init)
        if full_best is None:
            return 1
        phase_eval(projector_init = projector_init, full_best = full_best)
        set_status(phase = "done")
        log("全部完成")
        return 0
    except Exception as error:  # 任何意外都记下后停止，等协调者处理
        log(f"出错：{error!r}")
        set_status(phase = "error", error = repr(error))
        return 1
    finally:
        subprocess.run(["pkill", "-f", "pipeline.colab.shard_window run"], check = False)
        status_dir = DRIVE / "runs" / ("t71-status" + TAG)
        status_dir.mkdir(parents = True, exist_ok = True)
        for path in (STATUS, LOG):
            if path.exists():
                shutil.copy2(src = path, dst = status_dir / path.name)


if __name__ == "__main__":
    sys.exit(main())
