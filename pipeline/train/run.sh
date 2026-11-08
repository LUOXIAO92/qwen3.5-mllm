#!/usr/bin/env bash
# 用 v1 的超参数训练音频 projector。可以随时中断：训练程序收到 SIGTERM 时保存检查点并以 75 退出，
# 重新运行同一条命令就从最近的检查点续训；训练已完成（progress.json 为 finished）时直接退出。
#
# 用法：pipeline/train/run.sh <训练清单 JSONL> <script|original> <输出目录> [其他 train.py 参数 ...]
#   第二个参数是训练目标：script 让模型学着输出朗读稿，original 让模型学着输出原文。
#   其他参数会追加在默认参数之后，同名参数以后面的为准，例如 --max-steps 20000 --warmup-steps 1000。
# 环境变量：
#   WORK_DIR  工作目录（含 envs/mllm-train、gpu.lock、data/T10/fleurs_dev.jsonl），默认是仓库旁边的 qwen3.5-asr-work
set -uo pipefail
manifest="${1:?训练清单}"
target="${2:?script 或 original}"
out="${3:?输出目录}"
shift 3
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work="${WORK_DIR:-$repo/../qwen3.5-asr-work}"
mkdir -p "$out"
if grep -q '"status": "finished"' "$out/progress.json" 2>/dev/null; then
  echo "$out 已训练完成"
  exit 0
fi
# 长音频批次的显存碎片会让占用逼近 32 GB，WSL 下超出部分借用内存会变得很慢
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$repo"
# 每步 16 条、每批「条数 × 最长时长」不超过 360 秒；峰值学习率 3e-3，前 500 步线性预热，之后余弦降到峰值的 0.1 倍；
# 训练 2 轮；每 500 步在 FLEURS dev（每种语言 50 句）上算一次损失并保存检查点。
flock "$work/gpu.lock" "$work/envs/mllm-train/bin/python" -m pipeline.train.train \
  --model-dir "$repo/qwen3.5-0.8b-mllm" \
  --train-manifest "$manifest" \
  --target "$target" \
  --output-dir "$out" \
  --batch-size 16 \
  --grad-accum 1 \
  --max-duration-s 30 \
  --max-batch-seconds 360 \
  --epochs 2 \
  --lr 3e-3 \
  --warmup-steps 500 \
  --min-lr-ratio 0.1 \
  --eval-manifest "$work/data/T10/fleurs_dev.jsonl" \
  --eval-max-per-lang 50 \
  --eval-every 500 \
  --save-every 500 \
  --log-every 100 \
  --resume \
  "$@"
