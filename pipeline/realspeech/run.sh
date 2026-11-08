#!/usr/bin/env bash
# 导入并核对一个真人语音数据集的一种语言：select → materialize → length → 主 ASR → compare → 交叉 ASR → compare → manifest。
# 每一步都可以重跑：materialize 复用已转出的音频，ASR 复用已完成的块。
#
# 用法：pipeline/realspeech/run.sh <名称> <配置> <输出目录> <select 的参数 ...>
#   配置：pipeline/realspeech/config_real.yaml，转写来自 Qwen3-ASR 时用 config_real_qwen3text.yaml
#   select 的参数：--source、--lang、--input、--target 等，见 pipeline/realspeech/select.py
# 例：pipeline/realspeech/run.sh emilia-de pipeline/realspeech/config_real.yaml $DATASET/audio/qwen3.5-asr/real/emilia/de \
#       --source emilia --lang de --input $DATASET/audio/amphion/Emilia-Dataset/Emilia*/DE/*.tar --target transcript
# 环境变量：
#   WORK_DIR  工作目录（含 envs/、gpu.lock、FLEURS 排除清单、GlotLID 模型），默认是仓库旁边的 qwen3.5-asr-work
#   GLOTLID   GlotLID 模型文件，默认 $WORK_DIR/tmp/T03/glotlid/model_v3.bin
#   RETRANSCRIBE  设为 1 时，主 ASR 之后用它的识别结果替换数据集的转写（adopt_asr.py），由交叉 ASR 核对；
#                 用于转写本身由模型生成、质量不可靠的数据集（如 Emilia 原始转写）
#   SPLIT_MAX_SECONDS  大于 0 时把长音频在停顿处切成不超过这么长的段（只能与 RETRANSCRIBE=1 一起用）
set -euo pipefail
name="${1:?名称}"
config="${2:?配置}"
out="${3:?输出目录}"
shift 3
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work="${WORK_DIR:-$repo/../qwen3.5-asr-work}"
glotlid="${GLOTLID:-$work/tmp/T03/glotlid/model_v3.bin}"
log="$work/logs/real_${name}.log"
mkdir -p "$work/logs" "$out"
cd "$repo"
step() {
  local label="$1"; shift
  local t0
  t0=$(date +%s)
  echo "=== $(date '+%F %T') 开始 $label" | tee -a "$log"
  "$@" >> "$log" 2>&1
  echo "=== $(date '+%F %T') 完成 $label，用时 $(( $(date +%s) - t0 )) 秒" | tee -a "$log"
}
C=(--config "$config" --work-dir "$work" --out-dir "$out")
runner=("$work/envs/runner/bin/python" -m pipeline.runner)
if [ ! -f "$out/selected.jsonl" ]; then
  step select "$work/envs/t03/bin/python" -m pipeline.realspeech.select --out-dir "$out" \
    --glotlid "$glotlid" --exclusion "$work/data/T02/exclusion.jsonl" "$@"
fi
step materialize "$work/envs/mllm-eval/bin/python" -m pipeline.realspeech.materialize --out-dir "$out" --split-max-seconds "${SPLIT_MAX_SECONDS:-0}"
step length "${runner[@]}" length "${C[@]}"
step asr-primary "${runner[@]}" asr --role primary "${C[@]}"
if [ "${RETRANSCRIBE:-0}" = 1 ]; then
  step adopt-asr "$work/envs/runner/bin/python" -m pipeline.realspeech.adopt_asr --out-dir "$out"
fi
step compare-1 "${runner[@]}" compare "${C[@]}"
step asr-cross "${runner[@]}" asr --role cross "${C[@]}"
step compare-2 "${runner[@]}" compare "${C[@]}"
step manifest "${runner[@]}" manifest "${C[@]}"
echo "=== 全部完成" | tee -a "$log"
