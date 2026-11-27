#!/usr/bin/env bash
# 自生成回放文本（#70）的完整流程。用法：
#
#   WORK_DIR=<工作目录> DATASET=<数据集目录> MLLM=<阶段 1 的 2B 融合模型目录> ORIGINAL=<原始 Qwen3.5-2B 目录> \
#       bash pipeline/replay/run.sh [步骤 ...]
#
# 筛选模式（默认，FILTER=1，写入 $DATASET/text/qwen3.5-asr/replay）的步骤依次为
# export prefixes prompts generate finalize langid nll report。
# 不筛选模式（FILTER=0，写入 $DATASET/text/qwen3.5-asr/replay-unfiltered）的步骤依次为
# export prompts generate finalize langid report：沿用筛选目录的文档开头和自由生成两组的提示，续写两组的提示用修正后的
# 分词器重新写出；生成时不检查循环、不保存 top-20，每条样本生成到结束符或上限，循环只作为元数据。
# 丢弃循环模式（FILTER=noloop，写入 $DATASET/text/qwen3.5-asr/replay-noloop）的步骤依次为
# import think_prompts generate finalize langid estimate teacher verify report：import 把不筛选目录中没有循环的样本复制
# 过来、有循环的只记元数据；think_prompts 写出 free_chat_think（与 free_chat 相同，助手轮按思考模式打开）的提示；generate 发现循环就中止并整条丢弃，其余样本原样保留，目标按保留的 token 数计；estimate 统计保留的
# token 数与磁盘需求，teacher 再用参考模型前向一遍，保存保留样本每个生成位置的 top-32 分布。
# 带系统提示的两组 free_sys_chat、free_sys_chat_think（#71，模型先自己写系统提示，再写用户消息和回答）不在默认计划中：
# 先用 sys_prompts 一步写出提示（条数由 SYS_COUNT 给出），再加上 PLAN=...,free_sys_chat:train,free_sys_chat_think:train 和对应的 TARGET。
# 第五组 prompt_chat（数据集中的真人提问，参考模型写回答）的提示可以用 user_prompts 一步写出，但暂不生成（用户
# 2026-11-22 决定），默认计划不含这一组；需要时加上 PLAN=...,prompt_chat:eval@instruct,prompt_chat:train@instruct 和对应的 TARGET。
# 不给参数时执行当前模式的全部步骤。每一步都可以单独重跑；generate 与 teacher 每次持有 GPU 锁约 TIME_BUDGET 秒，
# 没有完成就释放锁再续跑。每次占用 GPU 之前检查 nvidia-smi：有其他进程占着显存（例如 ComfyUI）就等待，不结束别人的进程。
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="${WORK_DIR:-$REPO_DIR/../qwen3.5-asr-work}"
DATASET="${DATASET:?需要设置 DATASET}"
MLLM="${MLLM:?需要设置 MLLM（阶段 1 的 2B 融合模型目录）}"
ORIGINAL="${ORIGINAL:?需要设置 ORIGINAL（原始 Qwen3.5-2B 目录）}"
FILTER="${FILTER:-1}"
FILTERED_REPLAY="$DATASET/text/qwen3.5-asr/replay"
UNFILTERED_REPLAY="$DATASET/text/qwen3.5-asr/replay-unfiltered"
PLAN_DEFAULT="free_raw:eval,free_chat:eval,cont_raw:eval,cont_chat:eval,free_raw:train,free_chat:train,cont_raw:train,cont_chat:train"
if [ "$FILTER" = 1 ]; then
    REPLAY="${REPLAY:-$FILTERED_REPLAY}"
    OUT="${OUT:-$WORK_DIR/data/T70}"
    TARGET="${TARGET:-eval=1000000,train=12000000}"
    PRIOR_MEAN="${PRIOR_MEAN:-1000}"
    MAX_INFLIGHT="${MAX_INFLIGHT:-0}"
    GEN_FLAGS=()
    DEFAULT_STEPS=(export prefixes prompts generate finalize langid nll report)
elif [ "$FILTER" = noloop ]; then
    REPLAY="${REPLAY:-$DATASET/text/qwen3.5-asr/replay-noloop}"
    OUT="${OUT:-$WORK_DIR/data/T70/noloop}"
    # 无模板两组的训练用目标降为 1,000 万保留 token（用户 2026-11-22）
    TARGET="${TARGET:-eval=1000000,train=30000000,free_raw:train=10000000,cont_raw:train=10000000,free_chat_think:train=10000000}"
    # 每条投放的提示平均保留的 token 数（有循环的样本按 0 计），由不筛选目录估计
    PRIOR_MEAN="${PRIOR_MEAN:-free_raw=700,free_chat=750,cont_raw=800,cont_chat=400,free_chat_think=1500,prompt_chat=400,free_sys_chat=750,free_sys_chat_think=1500}"
    MAX_INFLIGHT="${MAX_INFLIGHT:-0}"
    # free_chat_think 的进度只计写出 </think> 的样本（没有写出的照样保留，是否使用由训练决定）
    GEN_FLAGS=(--drop-loops --topk 0 --count-closed-think)
    PLAN_DEFAULT="cont_raw:train,cont_chat:train,free_raw:eval,free_chat:eval,cont_raw:eval,cont_chat:eval,free_chat:train,free_raw:train"
    PLAN_DEFAULT="$PLAN_DEFAULT,free_chat_think:eval,free_chat_think:train"
    DEFAULT_STEPS=(import think_prompts generate finalize langid estimate teacher verify report)
else
    REPLAY="${REPLAY:-$UNFILTERED_REPLAY}"
    OUT="${OUT:-$WORK_DIR/data/T70/unfiltered}"
    TARGET="${TARGET:-eval=1000000,train=30000000}"
    # 每条样本的平均长度：由筛选目录估计（出现循环的样本按 16,384 计），在途样本都要写完，先验过低会多投放
    PRIOR_MEAN="${PRIOR_MEAN:-free_raw=10000,free_chat=850,cont_raw=8000,cont_chat=1200}"
    # 无模板两组的样本大多写到 16k 上限：在途 144 条时 KV 缓存平均约用七成，256 条会超出容量、反复抢占重算
    MAX_INFLIGHT="${MAX_INFLIGHT:-free_raw=144,cont_raw=144,free_chat=384,cont_chat=384}"
    GEN_FLAGS=(--no-filter --topk 0)
    DEFAULT_STEPS=(export prompts generate finalize langid report)
fi
# 文档开头（prefixes 的输出）与自由生成两组提示的来源：两种模式共用筛选目录中的
PREFIXES="${PREFIXES:-$FILTERED_REPLAY/prefixes}"
REF="${REF:-$WORK_DIR/data/T70/ref-lm}"
EXCLUSION="${EXCLUSION:-$WORK_DIR/data/T02/exclusion.jsonl}"
LANGID_MODEL="${LANGID_MODEL:-$WORK_DIR/tmp/T03/glotlid/model_v3.bin}"
PLAN="${PLAN:-$PLAN_DEFAULT}"
TIME_BUDGET="${TIME_BUDGET:-720}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
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

# 反复在 GPU 上运行一个可续跑的命令（退出码 0 完成，3 时间用完待续跑），直到完成
run_resumable() {
    local log="$1"
    shift
    local status
    while true; do
        set +e
        run_on_gpu "$@" >> "$log" 2>&1
        status=$?
        set -e
        case "$status" in
            0) echo "$(date '+%F %T') 全部完成"; return 0 ;;
            3) echo "$(date '+%F %T') 本次时间用完，释放 GPU 锁后续跑"; sleep 20 ;;
            75) echo "$(date '+%F %T') 拿到锁时 GPU 被占用，稍后重试"; sleep 300 ;;
            *) echo "$(date '+%F %T') 出错（退出码 $status），见 $log"; return "$status" ;;
        esac
    done
}

step_export() {
    if [ -f "$REF/model.safetensors" ]; then
        echo "参考模型已导出：$REF"
        return 0
    fi
    "$EVAL_ENV" -m pipeline.replay.export_lm --mllm "$MLLM" --original "$ORIGINAL" --out "$REF"
}

step_prefixes() {
    "$TEXT_ENV" -m pipeline.replay.prefixes --raw-root "$DATASET/text" --exclusion "$EXCLUSION" \
        --out-dir "$PREFIXES" --eval-per-lang 250 --train-per-lang 3000 --jobs 10
}

step_prompts() {
    if [ "$FILTER" = 1 ]; then
        # free_raw 的第一个 token 要用参考模型前向一次（first_token.py），只有这一组占用 GPU
        run_on_gpu "$EVAL_ENV" -m pipeline.replay.prompts --model "$REF" --replay-dir "$REPLAY" --free-raw-start "$FREE_RAW_START" --groups free_raw
        "$EVAL_ENV" -m pipeline.replay.prompts --groups free_chat cont_raw cont_chat --model "$REF" --replay-dir "$REPLAY" \
            --free-raw-start "$FREE_RAW_START" --prefixes-dir "$PREFIXES"
        return 0
    fi
    # 自由生成两组的提示只含换行与聊天模板的特殊 token，不受分词器问题（#77）影响，直接沿用筛选目录的文件
    local group split
    cp "$FILTERED_REPLAY/templates.json" "$REPLAY/templates.json"
    for group in free_raw free_chat; do
        for split in eval train; do
            mkdir -p "$REPLAY/$group/$split"
            cp "$FILTERED_REPLAY/$group/$split/prompts.jsonl" "$REPLAY/$group/$split/prompts.jsonl"
        done
    done
    "$EVAL_ENV" -m pipeline.replay.prompts --groups cont_raw cont_chat --model "$REF" --replay-dir "$REPLAY" --prefixes-dir "$PREFIXES"
}

step_think_prompts() {
    if [ ! -f "$REPLAY/free_chat_think/train/prompts.jsonl" ]; then
        "$EVAL_ENV" -m pipeline.replay.prompts --groups free_chat_think --model "$REF" --replay-dir "$REPLAY"
    fi
}

step_sys_prompts() {
    # 带系统提示的两组的提示，不在默认步骤中；SYS_COUNT 为每个划分的条数（生成在达到 token 目标时停止）
    if [ ! -f "$REPLAY/free_sys_chat/train/prompts.jsonl" ]; then
        "$EVAL_ENV" -m pipeline.replay.prompts --groups free_sys_chat free_sys_chat_think --model "$REF" --replay-dir "$REPLAY" \
            --free-count "${SYS_COUNT:-eval=8000,train=60000}"
    fi
}

step_user_prompts() {
    # 只写出 prompt_chat 的提示，不在默认步骤中
    if [ ! -f "$REPLAY/prompt_chat/train/prompts.jsonl" ]; then
        "$EVAL_ENV" -m pipeline.replay.user_prompts --model "$REF" --raw-root "$DATASET/text" --replay-dir "$REPLAY"
    fi
}

step_import() {
    "$EVAL_ENV" -m pipeline.replay.noloop --source "$UNFILTERED_REPLAY" --out "$REPLAY"
}

step_generate() {
    run_resumable "$OUT/logs/generate.log" "$EVAL_ENV" -m pipeline.replay.generate --model "$REF" --replay-dir "$REPLAY" \
        --plan "$PLAN" --target "$TARGET" --time-budget "$TIME_BUDGET" --max-num-seqs "$MAX_NUM_SEQS" --prior-mean "$PRIOR_MEAN" \
        --max-inflight "$MAX_INFLIGHT" "${GEN_FLAGS[@]}"
}

step_finalize() {
    local flags=()
    [ "$FILTER" = 0 ] && flags=(--no-filter)
    [ "$FILTER" = noloop ] && flags=(--drop-loops)
    "$EVAL_ENV" -m pipeline.replay.finalize --model "$REF" --replay-dir "$REPLAY" "${flags[@]}"
}

step_langid() {
    "$TEXT_ENV" -m pipeline.replay.langid --replay-dir "$REPLAY" --langid-model "$LANGID_MODEL"
}

step_nll() {
    run_on_gpu "$TRAIN_ENV" -m pipeline.replay.nll --mllm "$MLLM" --replay-dir "$REPLAY" --split eval
}

step_estimate() {
    "$TRAIN_ENV" -m pipeline.replay.teacher --mllm "$MLLM" --replay-dir "$REPLAY" --plan "$PLAN" --estimate | tee "$OUT/estimate.txt"
}

step_teacher() {
    # TEACHER_EXTRA 可以给出 --exclude-unclosed-think（free_chat_think 中没有写出 </think> 的样本不计算）
    # shellcheck disable=SC2086
    run_resumable "$OUT/logs/teacher.log" "$TRAIN_ENV" -m pipeline.replay.teacher --mllm "$MLLM" --replay-dir "$REPLAY" \
        --plan "$PLAN" --time-budget "$TIME_BUDGET" ${TEACHER_EXTRA:-}
}

step_verify() {
    # shellcheck disable=SC2086
    run_on_gpu "$TRAIN_ENV" -m pipeline.replay.teacher --mllm "$MLLM" --replay-dir "$REPLAY" --plan "$PLAN" --verify 4 ${TEACHER_EXTRA:-}
}

step_report() {
    "$EVAL_ENV" -m pipeline.replay.report --replay-dir "$REPLAY" --out-dir "$OUT"
    cp "$OUT/summary.json" "$OUT/summary.md" "$REPLAY/"
}

STEPS=("$@")
if [ "${#STEPS[@]}" -eq 0 ]; then
    STEPS=("${DEFAULT_STEPS[@]}")
fi
for step in "${STEPS[@]}"; do
    echo "$(date '+%F %T') ===== $step"
    "step_$step"
done
