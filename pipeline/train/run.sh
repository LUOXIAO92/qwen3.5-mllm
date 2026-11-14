#!/usr/bin/env bash
# 用 v1 的超参数训练音频 projector。可以随时中断：训练程序收到 SIGTERM 时保存检查点并以 75 退出，
# 重新运行同一条命令就从最近的检查点续训；训练已完成（progress.json 为 finished）时直接退出。
#
# 用法：pipeline/train/run.sh <训练清单 JSONL> <script|original> <输出目录> [其他 train.py 参数 ...]
#   第二个参数是训练目标：script 让模型学着输出朗读稿，original 让模型学着输出原文。
#   其他参数会追加在默认参数之后，同名参数以后面的为准，例如 --max-steps 20000 --warmup-steps 1000。
#   实验 v2 的混音加 --noise-manifest <背景声列表 background.jsonl>；空目标样本默认占每轮样本的 3%
#   （--empty-target-share），混音比例默认 30%（--noise-prob），见 train.py 的说明。
#   2B、4B 主干加 --model-dir <仓库>/qwen3.5-2b-mllm 或 qwen3.5-4b-mllm，其余设置相同。
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
# CUDA 缓存分配器（与 moss-image-trainer 的默认设置相同）：expandable_segments 让 PyTorch 按需扩展、归还显存段，
# 长短不一的批次不会留下大片碎片；garbage_collection_threshold:0.9 在保留的显存超过可用量的 90% 时，先回收
# 缓存中空闲的块再向驱动申请，避免占用逼近 32 GB（WSL 下超出部分会借用内存，速度骤降）。分配器在 CUDA 初始化时
# 读取这个变量，所以在启动 Python 之前设置；环境中已经设置了 PYTORCH_CUDA_ALLOC_CONF 时以环境为准。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.9}"
cd "$repo"
# 每步 16 条、每批「条数 × 最长时长」不超过 360 秒；峰值学习率 3e-3，前 500 步线性预热，之后余弦降到峰值的 0.1 倍；
# 训练 2 轮；每 500 步在 FLEURS dev（每种语言 50 句）上算一次损失并保存检查点。
# 显存与速度（任务 T59，实测见 pipeline/train/__init__.py「显存与速度选项」）：音频塔的窗口注意力一层调用一次
# （varlen），位置编号、音频分块和计损失的位置在数据读取进程中预先算好，不改变批次与损失，0.8B 上比原实现快约 25%，
# 2B 快约 28%。PyTorch 最多使用 94% 的显存：超出时训练程序逐步增加做梯度检查点的层（再不够时改为流式加载冻结权重），
# 重试同一批，训练结果不变。4B 在这个批次上需要梯度检查点，建议另加 --checkpoint-layers 24（实测最快），免得每段
# 续训都先超出一次显存。torch.compile（--compile all）在 0.8B 上再快约 12%，但 4B 上编译后的梯度不对，不要使用。
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
  --audio-attention varlen \
  --precompute-metadata \
  --memory-fraction 0.94 \
  --resume \
  "$@"
