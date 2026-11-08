#!/usr/bin/env bash
# 用一次训练中 dev 损失最低的检查点评测一个评测清单目录，结果写到 <结果目录>（hypotheses.jsonl、summary.json 等）。
# 结果目录已有 summary.json 时直接跳过。
#
# 用法：pipeline/eval/run.sh <训练输出目录> <评测清单目录> <结果目录>
#   评测清单目录含 <语言>/test.jsonl：FLEURS 测试集用 pipeline.eval.fleurs_prepare 生成，
#   合成测试集用 pipeline.eval.synth_manifest 从合成输出目录转换。
# 环境变量：
#   WORK_DIR  工作目录（含 envs/mllm-eval、envs/mllm-train、gpu.lock），默认是仓库旁边的 qwen3.5-asr-work
set -uo pipefail
run="${1:?训练输出目录}"
manifest="${2:?评测清单目录}"
out="${3:?结果目录}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work="${WORK_DIR:-$repo/../qwen3.5-asr-work}"
[ -f "$out/summary.json" ] && { echo "$out 已完成"; exit 0; }
cd "$repo"
best=$("$work/envs/mllm-train/bin/python" -m pipeline.train.best_checkpoint "$run") || exit 1
echo "选用 $best"
flock "$work/gpu.lock" "$work/envs/mllm-eval/bin/python" -m pipeline.eval.run_eval \
  --model-dir "$repo/qwen3.5-0.8b-mllm" \
  --projector "$best/projector.safetensors" \
  --manifest-dir "$manifest" \
  --batch-size 48 \
  --output-dir "$out"
