#!/usr/bin/env bash
# 为每种语言的最终原文条目生成朗读稿：读 $DATASET/text/qwen3.5-asr/source-text/final/<语言>.jsonl，
# 写 $DATASET/text/qwen3.5-asr/scripts-v1/<语言>.jsonl（每行补上 script、tn_status、tn_note）。
#
# 用法：pipeline/tn/run_all.sh [语言 ...]，不给语言时处理全部 18 种
# 环境变量：
#   WORK_DIR  工作目录（含 envs/tn），默认是仓库旁边的 qwen3.5-asr-work
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work="${WORK_DIR:-$repo/../qwen3.5-asr-work}"
src="${DATASET:?需要设置 DATASET}/text/qwen3.5-asr/source-text/final"
dst="$DATASET/text/qwen3.5-asr/scripts-v1"
langs=("$@")
[ ${#langs[@]} -gt 0 ] || langs=(zh en ja ko de fr ru pt es it id ms th vi fil ar tr fa)
mkdir -p "$dst"
cd "$repo"
for lang in "${langs[@]}"; do
  if [ -f "$src/$lang.jsonl" ]; then
    "$work/envs/tn/bin/python" -m pipeline.tn.run --input "$src/$lang.jsonl" --output "$dst/$lang.jsonl"
  else
    echo "$lang：没有最终原文条目，跳过"
  fi
done
