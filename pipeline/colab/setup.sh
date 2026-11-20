#!/usr/bin/env bash
# 在 Colab 上准备训练环境与训练要读的小文件（任务 C1）。每一步做完会在 $WORK_DIR/.setup/ 下留一个标记，
# 重复运行时跳过已完成的步骤；换了新的 Colab 会话（本地磁盘清空）就全部重做。用法见 pipeline/colab/README.md。
#
# 环境变量：
#   DRIVE_DIR       必填。Google Drive 上本项目的目录，含 upload/（本机上传的文件）和 wheels/（安装包缓存）
#   DATASET         必填。与本机相同的 $DATASET 路径：训练清单和背景声列表要放回与本机完全相同的路径，
#                   train.py 续训时按路径字符串核对它们
#   WORK_DIR        工作目录，默认 /content/work：虚拟环境 envs/mllm-train、gpu.lock、dev 清单 data/T10 都在这里
#   MODEL_NAME      融合模型目录名，默认 qwen3.5-4b-mllm，从 $DRIVE_DIR/upload/ 复制到仓库根目录下
#   TRAIN_MANIFEST  训练清单路径，默认 $DATASET/audio/qwen3.5-asr/v2/train/manifest-v2b.jsonl；
#                   从 $DRIVE_DIR/upload/<文件名>.gz 解压
#   NOISE_TAR       背景声 tar（bundle.py，--root $DATASET），默认 $DRIVE_DIR/upload/noise-v2.tar
#   EVAL_TAR        dev 清单 tar（bundle.py，--root <本机工作目录>），默认 $DRIVE_DIR/upload/fleurs-dev.tar
#   TORCH_INDEX     PyTorch 的包索引，默认 https://download.pytorch.org/whl/cu130
set -euo pipefail
: "${DRIVE_DIR:?请设置 DRIVE_DIR（Google Drive 上本项目的目录）}"
: "${DATASET:?请设置 DATASET（与本机相同的路径）}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work="${WORK_DIR:-/content/work}"
upload="$DRIVE_DIR/upload"
wheels="$DRIVE_DIR/wheels"
model="${MODEL_NAME:-qwen3.5-4b-mllm}"
manifest="${TRAIN_MANIFEST:-$DATASET/audio/qwen3.5-asr/v2/train/manifest-v2b.jsonl}"
noise_tar="${NOISE_TAR:-$upload/noise-v2.tar}"
eval_tar="${EVAL_TAR:-$upload/fleurs-dev.tar}"
torch_index="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
venv="$work/envs/mllm-train"
requirements="$repo/pipeline/colab/requirements-train.txt"
marks="$work/.setup"
mkdir -p "$work" "$marks"

step() { echo "== $(date '+%H:%M:%S') $*"; }
done_before() { [ -f "$marks/$1" ]; }
mark() { touch "$marks/$1"; }

# 1. torchcodec 解码音频要用系统的 FFmpeg 共享库
if ! ldconfig -p | grep -q 'libavcodec\.so'; then
  step "安装 FFmpeg"
  apt-get update -q && apt-get install -y -q ffmpeg
fi

# 2. 虚拟环境。安装包第一次从网上下载并缓存到 Drive 上的 wheels/（约 4 GB），以后的会话直接从缓存安装。
#    mm-assistant-mask 来自 git，缓存时构建成 wheel，安装时按包名从缓存中取。
if ! done_before venv; then
  step "建立虚拟环境 $venv"
  command -v uv >/dev/null || python3 -m pip install -q uv
  rm -rf "$venv"
  uv venv --seed --python "$(command -v python3)" "$venv"
  mkdir -p "$wheels"
  if [ ! -f "$wheels/.complete" ]; then
    step "下载安装包到 $wheels"
    "$venv/bin/python" -m pip wheel --wheel-dir "$wheels" --find-links "$wheels" \
      --extra-index-url "$torch_index" -r "$requirements"
    touch "$wheels/.complete"
  fi
  sed 's|^mm-assistant-mask @ .*|mm-assistant-mask|' "$requirements" > "$work/requirements-offline.txt"
  step "从缓存安装"
  "$venv/bin/python" -m pip install --no-index --find-links "$wheels" -r "$work/requirements-offline.txt"
  mark venv
fi

# 2b. 拼接批次（--pack-sequences）用的 flash-attn 和 causal-conv1d：PyPI 上没有与 torch 2.13.0+cu130、Python 3.13
#     对应的安装包，事先放在 Drive 的 wheels/packing/（来源与做法见 README「拼接批次用的安装包」）。目录不存在时跳过，
#     不用 --pack-sequences 的训练不受影响；用了而缺包时，训练程序开始前就会停止。
packing_wheels="$wheels/packing"
if ! done_before packing && compgen -G "$packing_wheels/*.whl" > /dev/null; then
  step "安装拼接批次用的 flash-attn、causal-conv1d"
  "$venv/bin/python" -m pip install --no-index --no-deps "$packing_wheels"/*.whl
  "$venv/bin/python" -c "import torch, causal_conv1d_cuda, flash_attn_2_cuda"
  mark packing
fi

# 3. 训练清单放回与本机相同的路径
if ! done_before manifest; then
  step "解压训练清单到 $manifest"
  mkdir -p "$(dirname "$manifest")"
  gunzip -c "$upload/$(basename "$manifest").gz" > "$manifest.tmp"
  mv "$manifest.tmp" "$manifest"
  mark manifest
fi

# 4. 背景声：列表和音频解到 $DATASET 下与本机相同的位置，混音时随机读取，放在本地磁盘上
if ! done_before noise; then
  step "解开背景声 $noise_tar"
  tar -C "$DATASET" -xf "$noise_tar"
  mark noise
fi

# 5. dev 清单与音频：解到 $WORK_DIR 下，run.sh 默认读 $WORK_DIR/data/T10/fleurs_dev.jsonl
if ! done_before eval; then
  step "解开 dev 清单 $eval_tar"
  tar -C "$work" -xf "$eval_tar"
  mark eval
fi

# 6. 融合模型（不在仓库中）复制到仓库根目录
if ! done_before model; then
  step "复制模型 $model"
  rm -rf "$repo/$model.tmp"
  cp -r "$upload/$model" "$repo/$model.tmp"
  rm -rf "$repo/$model"
  mv "$repo/$model.tmp" "$repo/$model"
  mark model
fi

step "完成"
"$venv/bin/python" - <<'PY'
import sys

import torch
import torchcodec
import transformers

print(f"Python {sys.version.split()[0]}，torch {torch.__version__}，torchcodec {torchcodec.__version__}，transformers {transformers.__version__}")
if torch.cuda.is_available():
    properties = torch.cuda.get_device_properties(0)
    print(f"GPU {properties.name}，{properties.total_memory / 2 ** 30:.0f} GB")
else:
    print("没有可用的 GPU")
PY
df -h / | tail -1
