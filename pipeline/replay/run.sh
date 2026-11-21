#!/usr/bin/env bash
# 自生成回放文本（#70）的完整流程。用法：
#
#   WORK_DIR=<工作目录> DATASET=<数据集目录> MLLM=<阶段 1 的 2B 融合模型目录> ORIGINAL=<原始 Qwen3.5-2B 目录> \
#       bash pipeline/replay/run.sh [步骤 ...]
#
# 步骤依次为 export prefixes prompts generate finalize langid nll report，不给参数时全部执行。
# 每一步都可以单独重跑；generate 每次持有 GPU 锁约 TIME_BUDGET 秒加收尾，没有完成就释放锁再续跑。
# 每次占用 GPU 之前检查 nvidia-smi：有其他进程占着显存（例如 ComfyUI）就等待，不结束别人的进程。
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="${WORK_DIR:-$REPO_DIR/../qwen3.5-asr-work}"
DATASET="${DATASET:?需要设置 DATASET}"
MLLM="${MLLM:?需要设置 MLLM（阶段 1 的 2B 融合模型目录）}"
ORIGINAL="${ORIGINAL:?需要设置 ORIGINAL（原始 Qwen3.5-2B 目录）}"
REPLAY="${REPLAY:-$DATASET/text/qwen3.5-asr/replay}"
OUT="${OUT:-$WORK_DIR/data/T70}"
REF="$OUT/ref-lm"
EXCLUSION="${EXCLUSION:-$WORK_DIR/data/T02/exclusion.jsonl}"
LANGID_MODEL="${LANGID_MODEL:-$WORK_DIR/tmp/T03/glotlid/model_v3.bin}"
PLAN="${PLAN:-free_raw:eval,free_chat:eval,cont_raw:eval,cont_chat:eval,free_raw:train,free_chat:train,cont_raw:train,cont_chat:train}"
TARGET="${TARGET:-eval=1000000,train=12000000}"
TIME_BUDGET="${TIME_BUDGET:-720}"
FREE_RAW_START="${FREE_RAW_START:-newline}"
EVAL_ENV="$WORK_DIR/envs/mllm-eval/bin/python"
TRAIN_ENV="$WORK_DIR/envs/mllm-train/bin/python"
TEXT_ENV="$WORK_DIR/envs/t03/bin/python"
GPU_LOCK="$WORK_DIR/gpu.lock"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
# FlashInfer 的采样内核需要现场编译（要 ninja），温度 1、不设 top-k/top-p 时用不到，关掉
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "$OUT/logs" "$REPLAY"
cd "$REPO_DIR"

gpu_busy() {
    [ -n "$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader)" ]
}

# 在 GPU 锁内运行命令；拿到锁后若发现 GPU 仍被占用（锁外的进程），放弃本次，返回 75
run_on_gpu() {
    while gpu_busy; do
        echo "$(date '+%F %T') GPU 上有其他进程，5 分钟后再看"
        sleep 300
    done
    flock "$GPU_LOCK" bash -c 'if [ -n "$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader)" ]; then exit 75; fi; exec "$@"' _ "$@"
}

step_export() {
    "$EVAL_ENV" -m pipeline.replay.export_lm --mllm "$MLLM" --original "$ORIGINAL" --out "$REF"
}

step_prefixes() {
    "$TEXT_ENV" -m pipeline.replay.prefixes --raw-root "$DATASET/text" --exclusion "$EXCLUSION" \
        --out-dir "$REPLAY/prefixes" --eval-per-lang 250 --train-per-lang 3000 --jobs 10
}

step_prompts() {
    # free_raw 的第一个 token 要用参考模型前向一次（first_token.py），短暂占用 GPU
    run_on_gpu "$EVAL_ENV" -m pipeline.replay.prompts --model "$REF" --replay-dir "$REPLAY" --free-raw-start "$FREE_RAW_START"
}

step_generate() {
    local status
    while true; do
        set +e
        run_on_gpu "$EVAL_ENV" -m pipeline.replay.generate --model "$REF" --replay-dir "$REPLAY" \
            --plan "$PLAN" --target "$TARGET" --time-budget "$TIME_BUDGET" >> "$OUT/logs/generate.log" 2>&1
        status=$?
        set -e
        case "$status" in
            0) echo "$(date '+%F %T') 生成全部完成"; return 0 ;;
            3) echo "$(date '+%F %T') 本次时间用完，释放 GPU 锁后续跑"; sleep 20 ;;
            75) echo "$(date '+%F %T') 拿到锁时 GPU 被占用，稍后重试"; sleep 300 ;;
            *) echo "$(date '+%F %T') 生成出错（退出码 $status），见 $OUT/logs/generate.log"; return "$status" ;;
        esac
    done
}

step_finalize() {
    "$EVAL_ENV" -m pipeline.replay.finalize --model "$REF" --replay-dir "$REPLAY"
}

step_langid() {
    "$TEXT_ENV" -m pipeline.replay.langid --replay-dir "$REPLAY" --langid-model "$LANGID_MODEL"
}

step_nll() {
    run_on_gpu "$TRAIN_ENV" -m pipeline.replay.nll --mllm "$MLLM" --replay-dir "$REPLAY" --split eval
}

step_report() {
    "$EVAL_ENV" -m pipeline.replay.report --replay-dir "$REPLAY" --out-dir "$OUT"
    cp "$OUT/summary.json" "$OUT/summary.md" "$REPLAY/"
}

STEPS=("$@")
if [ "${#STEPS[@]}" -eq 0 ]; then
    STEPS=(export prefixes prompts generate finalize langid nll report)
fi
for step in "${STEPS[@]}"; do
    echo "$(date '+%F %T') ===== $step"
    "step_$step"
done
