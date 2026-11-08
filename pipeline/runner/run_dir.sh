#!/usr/bin/env bash
# 对一个输出目录逐步运行合成—核对流水线：plan → tts → length → 主 ASR → compare → 交叉 ASR → compare → manifest。
# tts 阶段开两个进程：Supertonic-3 在 CPU 上合成，其余模型在 GPU 上合成，两者同时进行。
# 每一步都可以续跑：已完成的块直接跳过；输出目录已有不同的任务计划时 plan 会报错，不会覆盖已合成的音频。
#
# 用法（在任意目录运行）：
#   pipeline/runner/run_dir.sh <名称> <train|test> <条目 JSONL（含 script）> <输出目录>
# 环境变量：
#   WORK_DIR  工作目录（含 envs/、gpu.lock、data/T07/voices.jsonl），默认是仓库旁边的 qwen3.5-asr-work
#   CONFIG    合成配置，默认 pipeline/runner/config_v1.yaml
# 日志写到 $WORK_DIR/logs/run_<名称>.log，CPU 进程的日志另写到 .cpu 文件。
set -euo pipefail
name="${1:?名称}"
split="${2:?train 或 test}"
entries="${3:?条目 JSONL}"
out="${4:?输出目录}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work="${WORK_DIR:-$repo/../qwen3.5-asr-work}"
config="${CONFIG:-$repo/pipeline/runner/config_v1.yaml}"
py="$work/envs/runner/bin/python"
log="$work/logs/run_${name}.log"
mkdir -p "$work/logs"
cd "$repo"
C=(--config "$config" --work-dir "$work" --out-dir "$out")
step() {
  local label="$1"; shift
  local t0
  t0=$(date +%s)
  echo "=== $(date '+%F %T') 开始 $label" | tee -a "$log"
  "$py" -m pipeline.runner "$@" "${C[@]}" >> "$log" 2>&1
  echo "=== $(date '+%F %T') 完成 $label，用时 $(( $(date +%s) - t0 )) 秒" | tee -a "$log"
}
step plan plan --texts "$entries" --scripts "$entries" --voices "$work/data/T07/voices.jsonl" --split "$split"
t0=$(date +%s)
echo "=== $(date '+%F %T') 开始 tts（GPU 与 CPU 并行）" | tee -a "$log"
"$py" -m pipeline.runner tts --models supertonic3 "${C[@]}" >> "$log.cpu" 2>&1 &
cpu_pid=$!
"$py" -m pipeline.runner tts --models omnivoice qwen3_tts voxcpm2 indextts25 vieneu "${C[@]}" >> "$log" 2>&1
wait "$cpu_pid"
echo "=== $(date '+%F %T') 完成 tts，用时 $(( $(date +%s) - t0 )) 秒" | tee -a "$log"
step length length
step asr-primary asr --role primary
step compare-1 compare
step asr-cross asr --role cross
step compare-2 compare
step manifest manifest
echo "=== 全部完成" | tee -a "$log"
