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
#   VIRTUAL_SPLIT_SHARE  大于 0 时做虚拟切分：长于 15 秒的音频中随机抽出这个比例，在停顿处求切点写入
#                        virtual_split.jsonl，不另存片段，读取时在内存中截取（只能与 RETRANSCRIBE=1 一起用）
#   HUMAN_VERIFIED  设为 1 时转写视为已人工核验（如 Common Voice 的已验证录音）：select 加 --human-verified，
#                   跳过主 ASR、交叉 ASR 和比对，时长检查通过即进入清单
#   PUNCTUATE_LANGS  逗号分隔的语言（如 fa）：manifest 之后用 LLM 给这些语言没有标点的训练目标还原标点
#                    （pipeline/realspeech/punctuate.py），再重新生成 manifest；需要一个兼容 OpenAI 接口的服务，
#                    地址 PUNCTUATE_ENDPOINT（默认 http://127.0.0.1:8081/v1），并发数 PUNCTUATE_CONCURRENCY（默认 8）
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
verified=()
if [ "${HUMAN_VERIFIED:-0}" = 1 ]; then
  verified=(--human-verified)
fi
if [ ! -f "$out/selected.jsonl" ]; then
  step select "$work/envs/t03/bin/python" -m pipeline.realspeech.select --out-dir "$out" \
    --glotlid "$glotlid" --exclusion "$work/data/T02/exclusion.jsonl" "${verified[@]}" "$@"
fi
step materialize "$work/envs/mllm-eval/bin/python" -m pipeline.realspeech.materialize --out-dir "$out" --split-max-seconds "${SPLIT_MAX_SECONDS:-0}" \
  --virtual-split-share "${VIRTUAL_SPLIT_SHARE:-0}"
if [ ! -s "$out/tasks.jsonl" ]; then
  echo "=== materialize 没有转出任何音频，见 $out/materialize_errors.json" | tee -a "$log"
  exit 1
fi
step length "${runner[@]}" length "${C[@]}"
if [ "${HUMAN_VERIFIED:-0}" != 1 ]; then
  step asr-primary "${runner[@]}" asr --role primary "${C[@]}"
  if [ "${RETRANSCRIBE:-0}" = 1 ]; then
    step adopt-asr "$work/envs/runner/bin/python" -m pipeline.realspeech.adopt_asr --out-dir "$out"
  fi
  step compare-1 "${runner[@]}" compare "${C[@]}"
  step asr-cross "${runner[@]}" asr --role cross "${C[@]}"
  step compare-2 "${runner[@]}" compare "${C[@]}"
fi
step manifest "${runner[@]}" manifest "${C[@]}"
if [ -n "${PUNCTUATE_LANGS:-}" ]; then
  step punctuate "$work/envs/runner/bin/python" -m pipeline.realspeech.punctuate run --out-dir "$out" \
    --langs "$PUNCTUATE_LANGS" --endpoint "${PUNCTUATE_ENDPOINT:-http://127.0.0.1:8081/v1}" \
    --concurrency "${PUNCTUATE_CONCURRENCY:-8}"
  step manifest-punctuated "${runner[@]}" manifest "${C[@]}"
fi
echo "=== 全部完成" | tee -a "$log"
