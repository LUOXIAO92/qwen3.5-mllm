"""训练音频 projector：冻结音频塔和语言模型，只更新 model.audio_projector；也可以连同语言模型一起训练（全量微调）。

用法：

    python -m pipeline.train.train \\
        --model-dir <阶段 1 模型目录> \\
        --train-manifest <训练清单.jsonl> \\
        --target script \\
        --output-dir <输出目录> \\
        --batch-size 32 --grad-accum 2 --lr 1e-3 --warmup-steps 200 --epochs 2

实验 v2 增加的三项（任务 T1）：

- 空目标样本：清单中训练目标为空字符串的样本（无人声片段），助手回合只有 <|im_end|> 计算损失。
  --empty-target-share（默认 0.03）设定它们在每轮样本中的期望占比，由抽样权重实现，见 sampling.py。
- 加权抽样：--sample-weight 字段:取值=权重，可重复，例如 --sample-weight lang:fa=2，见 sampling.py。
- 实时混音：--noise-manifest 给出背景声列表时，语音样本按 --noise-prob（默认 0.3）的比例、在
  [--noise-snr-min, --noise-snr-max]（默认 0–20 dB）内均匀随机的信噪比混入背景声；--noise-category-weight
  类别=权重 调整各类背景声的抽中概率。混音在数据读取进程中完成，见 mixing.py。

抽样和混音的随机数都只由 --seed、轮次和样本在本轮中的位置决定，断点续训与不中断的训练得到相同的批次。

断点续训：加 --resume，从 <输出目录>/latest 指向的检查点继续。--stop-after-minutes 到时保存并退出，
--stop-at-step 训练到指定步数时保存并退出（学习率日程仍按 --epochs 或 --max-steps 的完整长度计算，
用于只跑完整训练的前一段、以后再接着训），收到 SIGTERM 或 Ctrl-C 时也在当前一步结束后保存并退出，便于分段占用 GPU。退出码 0 表示已训练到
最后一步，退出码 75 表示中途停止、可以用 --resume 继续。

进度：每 --log-every 步打印一行并追加到 <输出目录>/train_log.jsonl，<输出目录>/progress.json 随时反映
当前状态、速度和预计剩余时间，格式见 pipeline/train/__init__.py。

在 Google Colab 上训练（任务 C1，步骤见 pipeline/colab/README.md）时另有几项，它们不改变批次，不在 RESUME_KEYS 中，
本机开始的训练可以加上它们在 Colab 上续训：

- --audio-shards <索引 json>：训练音频从分片读取（pipeline/colab/pack_shards.py 按训练计划生成，读取见 shards.py）。
  按训练计划排列的索引记录了生成它的计划（PLAN_KEYS 各项和按读取顺序算出的摘要），与本次训练不一致时不开始训练；
  按路径排列的索引与计划无关，不做这项核对。
  --audio-shard-dirs <目录>[:<目录>...] 是本地查找分片的目录和顺序（窗口目录、常驻目录），
  --audio-shard-fallback <目录> 是本地没有时的备用目录（Google Drive 上的分片），用到时记 [shard-fallback]。
- --rows-cache-dir：清单行内存映射文件（data.PackedRows，v2b 约 1.4 GB）所在目录，默认 <输出目录>/dataset。
  输出目录放在 Google Drive 上时，用它把这个文件放到本地磁盘。

全量微调（任务 #66）：

- --init-projector <检查点目录或 projector.safetensors>：新开始的训练从已有的 projector 出发（例如只训练 projector
  的训练越过损失突降之后的检查点）。续训时不起作用，projector 取自续训的检查点。只记在训练配置中，不参与续训核对。
- --train-language-model：语言模型的全部参数（词嵌入兼输出层、各解码层、最后的归一化）也参与训练，音频塔和视觉塔
  仍冻结。语言模型的参数仍以 bfloat16 计算，默认由优化器更新它们的 float32 主权重（master.py）。它改变训练内容，在
  RESUME_KEYS 中；以前的训练配置中没有这一项，按默认值（不训练）核对，可以照常续训。
- --language-model-master-weights、--language-model-grad-dtype、--language-model-rounding、
  --language-model-optimizer-states：语言模型参数组的精度（有没有 float32 主权重、梯度精度、直接更新 bfloat16 参数时
  的舍入、AdamW 状态的格式），默认值是以前的做法，见 precision.py。它们改变优化器状态的格式，在 RESUME_KEYS 中；
  以前的训练配置中没有这几项，按默认值核对。
- --language-model-lr：语言模型参数组的峰值学习率，projector 仍用 --lr，两组按同一条预热加余弦的曲线变化。
  与 --lr 一样不在续训核对的配置中；续训时两组的峰值学习率都沿用检查点中学习率调度器保存的数值，命令行给出的
  数值与它不同时打印提示。
- 检查点每次多存 language_model.safetensors（bfloat16，评测用）；主权重（没有主权重时是语言模型参数）与 AdamW 状态
  在 trainer_state.pt 中，与 projector 的续训状态一样只在最近的检查点中保留（--keep-trainer-states）。格式见 checkpoint.py。

速度选项 --loss-impl（交叉熵的实现）和 --pack-sequences（每批样本拼成一行、去掉填充）见 optimize.py（任务 #66）。
它们不改变每步的样本和损失的定义，不在 RESUME_KEYS 中，续训时可以改变。
"""

import argparse
import json
import math
import os
import random
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from pipeline.train.checkpoint import (
    STATE_FILE,
    latest_checkpoint,
    read_projector,
    save_checkpoint,
)
from pipeline.train.data import (
    TARGETS,
    TrainCollator,
    TranscriptionDataset,
    read_manifest,
    summarize_rows,
)
from pipeline.train import precision
from pipeline.train.master import DirectWeights, MasterWeights
from pipeline.train.mixing import BackgroundMixer, parse_category_weights
from pipeline.train.shards import attach_shards, parse_dirs, plan_digest, read_header
from pipeline.train.optimize import add_arguments as add_optimize_arguments
from pipeline.train.optimize import apply as apply_optimizations
from pipeline.train.optimize import options_from_args
from pipeline.train.progress import Progress
from pipeline.train.sampling import (
    EpochPlan,
    expected_composition,
    parse_weight_rules,
    sample_weights,
)
from pipeline.train.model import (
    PACKED_ATTENTION,
    PROJECTOR_PREFIX,
    check_packing_kernels,
    frozen_fingerprint,
    language_model_parameters,
    load_model,
    load_projector,
    move_batch,
    projector_parameters,
    token_losses,
)

EXIT_STOPPED = 75
# 只为取得各选项的默认值而解析一次参数时，填入必填项的占位值。
REQUIRED_PLACEHOLDERS = ["--model-dir", "", "--train-manifest", "", "--target", TARGETS[0], "--output-dir", ""]
# 续训时必须与原训练一致的配置：它们决定数据顺序、批次内容和损失，train_language_model 决定训练哪些参数，
# precision.KEYS 决定语言模型参数组的更新方式和优化器状态的格式。
# 音频从哪里读（--audio-shards、--audio-shard-dirs、--audio-shard-fallback）、清单行缓存放在哪里（--rows-cache-dir）、
# 读取进程数、模型和输出目录都不改变批次，不在其中；--init-projector 只影响新开始的训练，也不在其中。
RESUME_KEYS = (
    "train_manifest",
    "audio_root",
    "target",
    "langs",
    "max_duration_s",
    "batch_size",
    "grad_accum",
    "group_batches",
    "max_batch_seconds",
    "seed",
    "sample_weight",
    "empty_target_share",
    "noise_manifest",
    "noise_audio_root",
    "noise_prob",
    "noise_snr_min",
    "noise_snr_max",
    "noise_category_weight",
    "train_language_model",
    # 全量微调的精度（precision.py）：决定优化器更新哪一份权重、状态以什么格式保存。
    *precision.KEYS,
)
# 决定训练计划（每一步读哪些样本）的配置，即 RESUME_KEYS 中混音和训练范围以外的各项。音频分片按这个计划排列。
PLAN_KEYS = (
    "train_manifest",
    "audio_root",
    "target",
    "langs",
    "max_duration_s",
    "batch_size",
    "grad_accum",
    "group_batches",
    "max_batch_seconds",
    "seed",
    "sample_weight",
    "empty_target_share",
)
# 不写入训练配置的选项：它们只控制这一段运行何时停止。
RUN_ONLY_KEYS = ("resume", "stop_after_minutes", "stop_at_step")
# 数据读取进程的启动方式。Python 3.14 在 Linux 上默认 forkserver，本机的训练和测量都是这样运行的；Colab 的
# Python 3.13 默认 fork，读取进程会继承主进程中几百万行清单字典，垃圾回收一碰就逐页复制，内存成倍增长。
# 显式指定 forkserver，两边行为一致，读取进程只收到 PackedRows 等对象序列化后的少量状态。
WORKER_START_METHOD = "forkserver"


def parse_args(argv = None):
    parser = argparse.ArgumentParser(description = "训练音频 projector")
    parser.add_argument("--model-dir", required = True, help = "阶段 1 完整模型目录")
    parser.add_argument("--train-manifest", required = True, help = "训练清单 JSONL")
    parser.add_argument("--audio-root", default = None, help = "清单中相对 audio_path 的基准目录，默认是清单所在目录")
    parser.add_argument(
        "--audio-shards",
        default = None,
        help = "音频分片索引 index.json（pipeline/colab/pack_shards.py 生成）；给出时训练音频从分片读取",
    )
    parser.add_argument(
        "--audio-shard-dirs",
        default = None,
        help = "查找分片的目录，多个用冒号分隔，按顺序查找（例如先本地副本再 Google Drive），默认是索引所在目录",
    )
    parser.add_argument(
        "--audio-shard-fallback",
        default = None,
        help = "本地目录中没有某个分片时到这里读取（例如 Google Drive 上的分片目录），用到时记 [shard-fallback]",
    )
    parser.add_argument(
        "--rows-cache-dir",
        default = None,
        help = "清单行内存映射文件所在目录，默认是 <输出目录>/dataset；输出目录在网络盘上时指向本地磁盘",
    )
    parser.add_argument(
        "--target",
        required = True,
        choices = TARGETS,
        help = "训练目标：script 朗读稿，original 原文",
    )
    parser.add_argument("--output-dir", required = True, help = "检查点与日志的输出目录")
    parser.add_argument("--langs", default = None, help = "只用这些语言，逗号分隔")
    parser.add_argument(
        "--max-duration-s",
        type = float,
        default = None,
        help = "丢弃更长的样本",
    )
    # 默认 40（用户 2026-11-15 决定）：v2b 清单中超过每秒 40 字的只有 136 条（0.14 小时），抽查都是对齐错位、
    # 录音被截短或 ASR 重复输出，最长的会让一批超出显存。读取时跳过，组批计划不变，所以不列入 RESUME_KEYS，
    # 正在进行的训练可以带着它续训。设为 0 不跳过。
    parser.add_argument(
        "--max-target-chars-per-second",
        type = float,
        default = 40.0,
        help = "训练目标字数除以音频时长超过它的样本读取时跳过；0 表示不跳过",
    )
    parser.add_argument(
        "--batch-size",
        type = int,
        default = 32,
        help = "每个微批的样本数",
    )
    parser.add_argument(
        "--grad-accum",
        type = int,
        default = 1,
        help = "梯度累积的微批数",
    )
    parser.add_argument(
        "--max-batch-seconds",
        type = float,
        default = None,
        help = "每批「条数 × 批内最长时长」的上限（秒），用来控制显存",
    )
    parser.add_argument(
        "--group-batches",
        type = int,
        default = 64,
        help = "按时长分组时每组包含的批次数",
    )
    parser.add_argument(
        "--epochs",
        type = float,
        default = None,
        help = "训练轮数，可以是小数",
    )
    parser.add_argument(
        "--max-steps",
        type = int,
        default = None,
        help = "优化步数，给出时优先于 --epochs",
    )
    parser.add_argument(
        "--lr",
        type = float,
        default = 1e-3,
        help = "峰值学习率（训练语言模型时是 projector 参数组的峰值学习率）",
    )
    parser.add_argument(
        "--init-projector",
        default = None,
        help = "新开始的训练从这个检查点目录（或 projector.safetensors）载入 projector；续训时不起作用",
    )
    parser.add_argument(
        "--train-language-model",
        action = "store_true",
        help = "语言模型的全部参数也参与训练（bfloat16 计算，精度见 --language-model-* 各项），音频塔和视觉塔仍冻结；续训时必须与原训练一致",
    )
    parser.add_argument(
        "--language-model-lr",
        type = float,
        default = None,
        help = (
            "语言模型参数组的峰值学习率，与 --train-language-model 一起给出，续训时也要再给出；"
            "与 --lr 一样，续训沿用检查点中保存的峰值学习率，命令行的数值不同时只打印提示"
        ),
    )
    parser.add_argument(
        "--min-lr-ratio",
        type = float,
        default = 0.1,
        help = "余弦退火结束时学习率与峰值之比",
    )
    parser.add_argument(
        "--warmup-steps",
        type = int,
        default = 100,
        help = "线性预热步数",
    )
    parser.add_argument("--weight-decay", type = float, default = 0.0)
    parser.add_argument("--max-grad-norm", type = float, default = 1.0)
    parser.add_argument("--seed", type = int, default = 1234)
    parser.add_argument(
        "--num-workers",
        type = int,
        default = 6,
        help = "数据读取进程数",
    )
    parser.add_argument("--log-every", type = int, default = 10)
    parser.add_argument("--save-every", type = int, default = 500)
    parser.add_argument(
        "--keep-trainer-states",
        type = int,
        default = 1,
        help = "保留续训状态的检查点个数，0 表示全部保留",
    )
    parser.add_argument("--eval-manifest", default = None, help = "dev 清单 JSONL，用于定期计算 dev 损失")
    parser.add_argument("--eval-audio-root", default = None)
    parser.add_argument("--eval-every", type = int, default = 500)
    parser.add_argument(
        "--eval-max-per-lang",
        type = int,
        default = 50,
        help = "dev 每种语言最多用多少条",
    )
    parser.add_argument("--eval-batch-size", type = int, default = 32)
    parser.add_argument(
        "--sample-weight",
        action = "append",
        default = [],
        help = "抽样权重规则「字段:取值=权重」，可重复，命中多条时相乘，例如 lang:fa=2",
    )
    parser.add_argument(
        "--empty-target-share",
        type = float,
        default = 0.03,
        help = "空目标样本在每轮样本中的期望占比，清单中没有空目标样本时不起作用",
    )
    parser.add_argument("--noise-manifest", default = None, help = "背景声列表 JSONL，给出时对语音样本实时混音")
    parser.add_argument("--noise-audio-root", default = None, help = "背景声相对 audio_path 的基准目录，默认是列表所在目录")
    parser.add_argument(
        "--noise-prob",
        type = float,
        default = 0.3,
        help = "语音样本中混音的比例",
    )
    parser.add_argument(
        "--noise-snr-min",
        type = float,
        default = 0.0,
        help = "混音信噪比下限（dB）",
    )
    parser.add_argument(
        "--noise-snr-max",
        type = float,
        default = 20.0,
        help = "混音信噪比上限（dB）",
    )
    parser.add_argument(
        "--noise-category-weight",
        action = "append",
        default = [],
        help = "背景声类别权重「类别=权重」，可重复，例如 music=0.5",
    )
    parser.add_argument("--resume", action = "store_true", help = "从输出目录中最近的检查点继续")
    parser.add_argument(
        "--stop-after-minutes",
        type = float,
        default = None,
        help = "训练这么多分钟后保存并退出",
    )
    parser.add_argument(
        "--stop-at-step",
        type = int,
        default = None,
        help = "训练到这一步时保存并退出；学习率日程不变，之后可以用 --resume 接着训",
    )
    parser.add_argument("--no-frozen-check", action = "store_true", help = "跳过冻结参数前后一致的检查")
    parser.add_argument("--deterministic", action = "store_true", help = "使用确定性算法，同样的配置逐位复现")
    precision.add_arguments(parser = parser)
    add_optimize_arguments(parser = parser)
    return parser.parse_args(args = argv)


def run_config(args):
    """写入 train_config.json 和检查点的训练配置：全部选项，去掉只控制本段运行何时停止的几项。"""
    return {key: value for key, value in vars(args).items() if key not in RUN_ONLY_KEYS}


def resume_mismatches(saved, config):
    """返回续训时与原训练不一致的 RESUME_KEYS。旧检查点的配置中没有后来新增的选项，按选项的默认值比较。"""
    defaults = vars(parse_args(argv = REQUIRED_PLACEHOLDERS))
    return [key for key in RESUME_KEYS if saved.get(key, defaults[key]) != config.get(key)]


def plan_mismatches(saved, config):
    """返回分片索引记录的训练计划配置 saved 与本次训练配置不一致的 PLAN_KEYS。"""
    return [key for key in PLAN_KEYS if saved.get(key) != config.get(key)]


def training_rows(args):
    """读取并筛选训练清单。训练和分片打包共用，两边的行号一致。"""
    return read_manifest(
        path = args.train_manifest,
        audio_root = args.audio_root,
        langs = set(args.langs.split(sep = ",")) if args.langs else None,
        max_duration_s = args.max_duration_s,
    )


def training_weights(args, rows, empty):
    """每条样本的抽样权重（sampling.py）。训练和分片打包共用。"""
    return sample_weights(
        rows = rows,
        empty = empty,
        rules = parse_weight_rules(specs = args.sample_weight),
        empty_share = args.empty_target_share,
    )


def make_batch_stream(
    args,
    rows,
    weights,
    dataset = None,
    collator = None,
):
    """训练用的批次流；分片打包只用它的训练计划 plan，不给 dataset 和 collator。"""
    return BatchStream(
        dataset = dataset,
        collator = collator,
        batch_size = args.batch_size,
        group_batches = args.group_batches,
        max_batch_seconds = args.max_batch_seconds,
        seed = args.seed,
        num_workers = args.num_workers,
        durations = [float(row["duration_s"]) for row in rows],
        weights = weights,
    )


def check_shard_plan(
    header,
    stream,
    rows,
    args,
    start_step,
    total_steps,
):
    """核对分片按本次训练的计划排列：按读取顺序的摘要必须相同，否则停止。

    索引只覆盖部分步数（速度测试用）或本次训练会超出索引的轮次时给出提示：范围外的样本读不出，按 [bad-audio] 跳过。
    """
    plan = header["plan"]
    digest = plan_digest(
        plan = stream.plan,
        rows = rows,
        epoch = plan["epoch"],
        grad_accum = args.grad_accum,
    )
    if digest != plan["digest"]:
        raise SystemExit(f"分片索引 {header['index_path']} 的读取顺序与本次训练不同，需要按本次训练的配置重新生成索引。")
    first, stop = plan["steps"]
    end = min(total_steps, args.stop_at_step) if args.stop_at_step is not None else total_steps
    if start_step < first or end > stop:
        print(
            f"[分片] 索引只覆盖第 {first}–{stop - 1} 步（下标从 0 起），本次训练第 {start_step}–{end - 1} 步；"
            "范围外的样本读不出，按 [bad-audio] 跳过。",
            flush = True,
        )


def worker_context(num_workers):
    """DataLoader 的 multiprocessing_context：有读取进程时用 WORKER_START_METHOD，没有时只能不指定。"""
    return WORKER_START_METHOD if num_workers > 0 else None


def lr_lambda_factory(warmup_steps, total_steps, min_lr_ratio):
    """线性预热到峰值，再余弦退火到 min_lr_ratio 倍。"""
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return lr_lambda


class WorkerInit:
    """数据读取进程的初始化。

    Ctrl-C 只由主进程处理，读取进程忽略，避免读取进程先退出导致无法保存。Python 3.14 默认用
    forkserver 启动读取进程，它们的父进程是 forkserver 而不是训练主进程，主进程被 kill -9 后
    PyTorch 自带的检查发现不了，所以另起一个线程，主进程不在时自行退出。
    """

    def __init__(self, main_pid):
        self.main_pid = main_pid

    def __call__(self, worker_id):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        threading.Thread(target = self._watch, daemon = True).start()

    def _watch(self):
        while True:
            time.sleep(5)
            try:
                os.kill(self.main_pid, 0)
            except ProcessLookupError:
                os._exit(0)


class EpochBatchSampler:
    """从给定位置开始，一轮接一轮地给出批次（SampleKey 列表），不停止。"""

    def __init__(self, stream, epoch, offset):
        self.stream = stream
        self.epoch = epoch
        self.offset = offset

    def __iter__(self):
        epoch, offset = self.epoch, self.offset
        plan = self.stream.plan
        while True:
            occurrences, batches = plan.position_batches(epoch = epoch)
            for batch in batches[offset:]:
                yield plan.keys(occurrences = occurrences, epoch = epoch, batch = batch)
            epoch += 1
            offset = 0


class BatchStream:
    """按轮次依次给出微批，并记录训练已经用掉的位置：第几轮、本轮已取出几个微批。

    读取进程会提前准备后面的批次，位置只在批次交给训练循环时前进，因此保存的位置
    恰好是已经参与训练的批次数。
    """

    def __init__(
        self,
        dataset,
        collator,
        batch_size,
        group_batches,
        max_batch_seconds,
        seed,
        num_workers,
        durations,
        weights = None,
    ):
        """durations 为各条样本的时长（秒），与 dataset 的下标一一对应，只在主进程中用于组批。"""
        self.dataset = dataset
        self.collator = collator
        self.plan = EpochPlan(
            durations = durations,
            weights = weights,
            batch_size = batch_size,
            group_batches = group_batches,
            max_batch_seconds = max_batch_seconds,
            seed = seed,
        )
        self.num_workers = num_workers
        self.epoch = 0
        self.offset = 0
        # 限制每批音频秒数时，各轮的批次数会略有不同，按轮记下。
        self._epoch_lengths = {}
        self.batches_per_epoch = self.epoch_length(epoch = 0)
        self._iterator = None

    def epoch_length(self, epoch):
        if epoch not in self._epoch_lengths:
            self._epoch_lengths[epoch] = len(self.plan.position_batches(epoch = epoch)[1])
        return self._epoch_lengths[epoch]

    def _batches(self):
        loader = torch.utils.data.DataLoader(
            dataset = self.dataset,
            batch_sampler = EpochBatchSampler(stream = self, epoch = self.epoch, offset = self.offset),
            collate_fn = self.collator,
            num_workers = self.num_workers,
            pin_memory = True,
            prefetch_factor = 4 if self.num_workers > 0 else None,
            worker_init_fn = WorkerInit(main_pid = os.getpid()),
            multiprocessing_context = worker_context(num_workers = self.num_workers),
        )
        for batch in loader:
            self.offset += 1
            if self.offset == self.epoch_length(epoch = self.epoch):
                self.epoch += 1
                self.offset = 0
            yield batch

    def next(self):
        if self._iterator is None:
            self._iterator = self._batches()
        return next(self._iterator)


def evaluate(
    model,
    loader,
    device,
    feature_dtype,
    loss_impl = "chunked",
):
    """在 dev 清单上计算目标文本的平均交叉熵和 token 准确率，按语言汇总。"""
    totals = {}
    with torch.no_grad():
        for batch in loader:
            inputs = move_batch(batch = batch, device = device, feature_dtype = feature_dtype)
            losses, correct, rows = token_losses(model = model, inputs = inputs, loss_impl = loss_impl)
            for row_index, lang in enumerate(batch["langs"]):
                in_row = rows == row_index
                entry = totals.setdefault(lang, [0.0, 0, 0])
                entry[0] += float(losses[in_row].sum())
                entry[1] += int(in_row.sum())
                entry[2] += int(correct[in_row].sum())
    per_lang = {
        lang: {"loss": entry[0] / entry[1], "token_acc": entry[2] / entry[1], "tokens": entry[1]}
        for lang, entry in sorted(totals.items())
    }
    all_tokens = sum(entry[1] for entry in totals.values())
    overall = {
        "loss": sum(entry[0] for entry in totals.values()) / all_tokens,
        "token_acc": sum(entry[2] for entry in totals.values()) / all_tokens,
        "tokens": all_tokens,
        # 各语言损失的平均，避免样本多的语言主导检查点选择。
        "lang_mean_loss": sum(item["loss"] for item in per_lang.values()) / len(per_lang),
    }
    return {"overall": overall, "per_lang": per_lang}


def sample_eval_rows(rows, max_per_lang, seed):
    by_lang = {}
    for row in rows:
        by_lang.setdefault(row["lang"], []).append(row)
    selected = []
    generator = random.Random(x = seed)
    for lang in sorted(by_lang):
        lang_rows = sorted(by_lang[lang], key = lambda row: row["sample_id"])
        generator.shuffle(x = lang_rows)
        selected.extend(lang_rows[:max_per_lang])
    return selected


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    random.setstate(state = state["python"])
    np.random.set_state(state = state["numpy"])
    torch.set_rng_state(new_state = state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(new_states = state["cuda"])


def append_jsonl(path, record):
    with open(file = path, mode = "a", encoding = "utf-8") as handle:
        handle.write(json.dumps(obj = record, ensure_ascii = False) + "\n")


def check_language_model_args(args):
    """--train-language-model 与 --language-model-lr 必须同时给出；精度选项的组合必须合法（precision.validate）。"""
    if args.train_language_model and args.language_model_lr is None:
        raise SystemExit("--train-language-model 需要同时给出语言模型的峰值学习率 --language-model-lr。")
    if args.language_model_lr is not None and not args.train_language_model:
        raise SystemExit("--language-model-lr 只在 --train-language-model 时使用。")
    try:
        precision.validate(options = precision.options_from_args(args = args), train_language_model = args.train_language_model)
    except ValueError as error:
        raise SystemExit(str(error)) from error


def make_optimizer(args, params, language_weights):
    """AdamW：只训练 projector 时一个参数组；训练语言模型时 projector 与语言模型各一组（--lr、--language-model-lr），
    实现与语言模型组的精度见 precision.make_optimizer。
    """
    if language_weights is None:
        return torch.optim.AdamW(
            params = params,
            lr = args.lr,
            betas = (0.9, 0.999),
            weight_decay = args.weight_decay,
        )
    return precision.make_optimizer(
        projector_params = params,
        language_model_params = language_weights.optimizer_params(),
        lr = args.lr,
        language_model_lr = args.language_model_lr,
        weight_decay = args.weight_decay,
        options = precision.options_from_args(args = args),
    )


def main(argv = None):
    args = parse_args(argv = argv)
    check_language_model_args(args = args)
    os.environ.setdefault(key = "TOKENIZERS_PARALLELISM", value = "false")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if args.deterministic:
        # 默认的注意力反传（memory-efficient SDPA）用原子加，梯度有微小的随机差异；
        # 打开确定性算法后，同一配置的训练逐位相同，可用来核对断点续训。
        os.environ.setdefault(key = "CUBLAS_WORKSPACE_CONFIG", value = ":4096:8")
        torch.use_deterministic_algorithms(mode = True)
    device = "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents = True, exist_ok = True)
    config = run_config(args = args)
    category_weights = parse_category_weights(specs = args.noise_category_weight)

    rows = training_rows(args = args)
    if not rows:
        raise SystemExit("训练清单中没有可用的样本。")
    shards = None
    shard_header = None
    if args.audio_shards:
        shard_header = read_header(index_path = args.audio_shards)
        # 按训练计划排列的索引（本地滚动窗口用）必须与本次训练的计划一致；按路径排列的索引与计划无关。
        if shard_header["order"] == "plan":
            changed = plan_mismatches(saved = shard_header["plan"]["params"], config = config)
            if changed:
                raise SystemExit(f"分片索引 {args.audio_shards} 按另一套训练计划生成，不一致的配置：{changed}")
        # 主进程读入索引，给每行加上分片位置后就丢掉索引；读取进程只收到分片名单和查找目录。
        shards, shard_summary = attach_shards(
            rows = rows,
            header = shard_header,
            dirs = parse_dirs(spec = args.audio_shard_dirs),
            fallback = args.audio_shard_fallback,
        )
        print(json.dumps(obj = {"audio_shards": shard_summary}, ensure_ascii = False), flush = True)
        if shard_header["order"] == "path" and shard_summary["rows_in_index"] < len(rows):
            print(f"[分片] 清单中有 {len(rows) - shard_summary['rows_in_index']} 行的音频不在索引中，读到时按 [bad-audio] 跳过。", flush = True)

    resume_dir = latest_checkpoint(output_dir = output_dir) if args.resume else None
    trainer_state = None
    if resume_dir is not None:
        trainer_state = torch.load(f = resume_dir / STATE_FILE, weights_only = False)
        changed = resume_mismatches(saved = trainer_state["config"], config = config)
        if changed:
            raise SystemExit(f"续训的配置与原训练不一致：{changed}")
    elif args.resume:
        print("没有找到可续训的检查点，从头开始。")

    random.seed(a = args.seed)
    np.random.seed(seed = args.seed)
    torch.manual_seed(seed = args.seed)

    model, processor = load_model(
        model_dir = args.model_dir,
        device = device,
        attn_implementation = PACKED_ATTENTION if args.pack_sequences else "sdpa",
        train_language_model = args.train_language_model,
    )
    if args.init_projector and trainer_state is None:
        load_projector(model = model, state = read_projector(path = args.init_projector))
        print(f"projector 从 {args.init_projector} 载入。", flush = True)
    elif args.init_projector:
        print("续训：--init-projector 不起作用，projector 取自续训的检查点。", flush = True)
    feature_dtype = model.model.audio_tower.dtype
    optimized = apply_optimizations(
        model = model,
        options = options_from_args(args = args),
        device = device,
        train_language_model = args.train_language_model,
    )
    if args.pack_sequences:
        # 拼接批次时三处都必须是按样本边界分段的内核，退回参考实现时各条样本互相串扰而不报错。
        print(json.dumps(obj = {"packing_kernels": check_packing_kernels(model = model)}, ensure_ascii = False), flush = True)
    params = projector_parameters(model = model)
    language_named = language_model_parameters(model = model) if args.train_language_model else []
    language_names = [name for name, _ in language_named]
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    expected = [name for name, _ in model.named_parameters() if name.startswith(PROJECTOR_PREFIX)] + language_names
    if not trainable or set(trainable) != set(expected):
        scope = "projector 和语言模型" if args.train_language_model else "projector"
        raise RuntimeError(f"可训练参数应当只有 {scope}：{sorted(set(trainable) ^ set(expected))[:10]}")
    # 训练语言模型时它的参数仍以 bfloat16 计算，优化器更新 float32 主权重或直接更新这些参数（master.py、
    # precision.py）。续训时用检查点中保存的那份权重，并写回模型参数。
    language_weights = None
    if args.train_language_model:
        weights_class = MasterWeights if args.language_model_master_weights == "fp32" else DirectWeights
        saved_weights = trainer_state.pop(weights_class.STATE_KEY, None) if trainer_state is not None else None
        if trainer_state is not None and saved_weights is None:
            raise SystemExit(f"{resume_dir / STATE_FILE} 中没有 {weights_class.STATE_KEY}（语言模型的权重），无法续训。")
        language_weights = weights_class(
            named_parameters = language_named,
            state = saved_weights,
            grad_dtype = precision.DTYPES[args.language_model_grad_dtype],
        )
        del saved_weights
    # 裁剪梯度的对象：projector 参数和优化器更新的语言模型张量（梯度在它们的 .grad 上）。
    trainable_tensors = params + (language_weights.optimizer_params() if language_weights is not None else [])

    mixer = None
    if args.noise_manifest:
        mixer = BackgroundMixer(
            manifest = args.noise_manifest,
            prob = args.noise_prob,
            snr_min = args.noise_snr_min,
            snr_max = args.noise_snr_max,
            category_weights = category_weights,
            audio_root = args.noise_audio_root,
        )
    # 清单行改存为内存映射文件，各读取进程共享一份，不再各自复制几百万个字典（v2b 时每个读取进程约 13 GB）
    rows_cache_dir = Path(args.rows_cache_dir) if args.rows_cache_dir else output_dir / "dataset"
    dataset = TranscriptionDataset(
        rows = rows,
        target = args.target,
        mixer = mixer,
        seed = args.seed,
        storage = rows_cache_dir / "rows.jsonl.bin",
        max_chars_per_second = args.max_target_chars_per_second or None,
        shards = shards,
    )
    weights = training_weights(args = args, rows = rows, empty = dataset.empty)
    if weights.sum() == 0:
        raise SystemExit("全部样本的抽样权重都是 0。")
    audio_config = model.config.audio_config
    collator = TrainCollator(
        processor = processor,
        audio_windows = (audio_config.n_window, audio_config.n_window_infer) if args.precompute_metadata else None,
        pack = args.pack_sequences,
    )
    stream = make_batch_stream(
        args = args,
        rows = rows,
        weights = weights,
        dataset = dataset,
        collator = collator,
    )
    batches_per_epoch = stream.batches_per_epoch
    steps_per_epoch = batches_per_epoch / args.grad_accum
    if args.max_steps is not None:
        total_steps = args.max_steps
    elif args.epochs is not None:
        total_steps = math.ceil(args.epochs * steps_per_epoch)
    else:
        raise SystemExit("请给出 --epochs 或 --max-steps。")
    config["total_steps"] = total_steps

    optimizer = make_optimizer(args = args, params = params, language_weights = language_weights)
    # 同一条曲线作用于每个参数组：各组的学习率都是本组峰值乘以这个系数。
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer = optimizer,
        lr_lambda = lr_lambda_factory(
            warmup_steps = args.warmup_steps,
            total_steps = total_steps,
            min_lr_ratio = args.min_lr_ratio,
        ),
    )

    step = 0
    if trainer_state is not None:
        load_projector(model = model, state = read_projector(path = resume_dir))
        optimizer.load_state_dict(state_dict = trainer_state["optimizer"])
        scheduler.load_state_dict(state_dict = trainer_state["scheduler"])
        restore_rng_state(state = trainer_state["rng"])
        step = trainer_state["step"]
        stream.epoch = trainer_state["epoch"]
        stream.offset = trainer_state["offset"]
        print(f"从 {resume_dir.name} 续训：第 {step} 步，第 {stream.epoch} 轮第 {stream.offset} 个微批。")
        # 学习率调度器的状态带着各组的峰值学习率（base_lrs），续训按它继续，命令行的 --lr 等不起作用。
        requested = [args.lr] + ([args.language_model_lr] if language_weights is not None else [])
        if [float(value) for value in scheduler.base_lrs] != requested:
            print(f"续训沿用检查点中的峰值学习率 {scheduler.base_lrs}，命令行给出的 {requested} 不起作用。", flush = True)
    else:
        (output_dir / "train_config.json").write_text(
            data = json.dumps(obj = config, ensure_ascii = False, indent = 2) + "\n",
            encoding = "utf-8",
        )

    if shard_header is not None and shard_header["order"] == "plan":
        check_shard_plan(
            header = shard_header,
            stream = stream,
            rows = rows,
            args = args,
            start_step = step,
            total_steps = total_steps,
        )

    eval_loader = None
    if args.eval_manifest:
        eval_rows = sample_eval_rows(
            rows = read_manifest(path = args.eval_manifest, audio_root = args.eval_audio_root),
            max_per_lang = args.eval_max_per_lang,
            seed = args.seed,
        )
        eval_rows.sort(key = lambda row: row["duration_s"])
        eval_batches = [
            list(range(start, min(start + args.eval_batch_size, len(eval_rows))))
            for start in range(0, len(eval_rows), args.eval_batch_size)
        ]
        eval_loader = torch.utils.data.DataLoader(
            dataset = TranscriptionDataset(rows = eval_rows, target = args.target),
            batch_sampler = eval_batches,
            collate_fn = collator,
            num_workers = min(args.num_workers, 4),
            pin_memory = True,
            worker_init_fn = WorkerInit(main_pid = os.getpid()),
            persistent_workers = args.num_workers > 0,
            multiprocessing_context = worker_context(num_workers = min(args.num_workers, 4)),
        )

    print(json.dumps(obj = {
        "samples": len(rows),
        "hours": round(number = sum(row["duration_s"] for row in rows) / 3600, ndigits = 2),
        "per_lang": summarize_rows(rows = rows),
        # 按抽样权重算出的每轮期望样本数、占比和音频小时数，空目标样本单列为 empty。
        "per_epoch": expected_composition(rows = rows, weights = weights, empty = dataset.empty),
        "sample_weight_rules": args.sample_weight,
        "noise": None if mixer is None else {
            "clips": len(mixer),
            "prob": args.noise_prob,
            "snr_db": [args.noise_snr_min, args.noise_snr_max],
            "categories": mixer.category_shares(),
        },
        "batches_per_epoch": batches_per_epoch,
        "steps_per_epoch": round(number = steps_per_epoch, ndigits = 1),
        "total_steps": total_steps,
        "start_step": step,
        "optimize": optimized.describe(),
        "language_model_precision": (
            precision.describe(
                options = precision.options_from_args(args = args),
                parameters = sum(parameter.numel() for _, parameter in language_named),
            )
            if args.train_language_model else None
        ),
    }, ensure_ascii = False))

    before = None
    if not args.no_frozen_check:
        before = frozen_fingerprint(model = model, trainable = language_names)

    stop = {"reason": None}

    def request_stop(signum, frame):
        stop["reason"] = signal.Signals(signum).name
        print(f"收到 {stop['reason']}，当前一步结束后保存并退出。", flush = True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    log_path = output_dir / "train_log.jsonl"
    eval_path = output_dir / "eval_log.jsonl"
    window = {
        "loss": 0.0,
        "tokens": 0,
        "correct": 0,
        "samples": 0,
        "empty": 0,
        "empty_loss": 0.0,
        "empty_correct": 0,
        "mixed": 0,
        "audio": 0.0,
        "steps": 0,
        "grad_norm": 0.0,
    }
    window_start = time.perf_counter()
    run_start = time.perf_counter()
    progress = Progress(
        path = output_dir / "progress.json",
        total_steps = total_steps,
        start_step = step,
        previous_seconds = trainer_state.get("wall_seconds", 0.0) if trainer_state else 0.0,
    )
    progress.write(
        step = step,
        status = "running",
        target = args.target,
        output_dir = str(output_dir),
    )
    max_mem_gb = 0.0
    last_eval = None
    last_train_loss = None
    frozen_check = None
    skipped_steps = 0
    checked_grads = False
    torch.cuda.reset_peak_memory_stats()

    def current_info():
        return {
            "target": args.target,
            "epoch": round(number = stream.epoch + stream.offset / batches_per_epoch, ndigits = 4),
            "train_loss": last_train_loss,
            "eval": last_eval,
            "frozen_check": frozen_check,
            "config": config,
            "saved_at": datetime.now(tz = timezone.utc).isoformat(timespec = "seconds"),
        }

    def save(step):
        state = {
            "step": step,
            "epoch": stream.epoch,
            "offset": stream.offset,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": rng_state(),
            "config": config,
            "wall_seconds": progress.elapsed(),
        }
        if language_weights is not None:
            state[language_weights.STATE_KEY] = language_weights.state_dict()
        directory = save_checkpoint(
            output_dir = output_dir,
            step = step,
            model = model,
            info = current_info(),
            trainer_state = state,
            keep_trainer_states = args.keep_trainer_states,
            language_model = language_weights.model_state() if language_weights is not None else None,
        )
        print(f"已保存 {directory}", flush = True)
        progress.write(step = step, last_checkpoint = directory.name)
        return directory

    def run_eval(step):
        nonlocal last_eval
        if eval_loader is None:
            return
        start = time.perf_counter()
        result = evaluate(
            model = model,
            loader = eval_loader,
            device = device,
            feature_dtype = feature_dtype,
            loss_impl = args.loss_impl,
        )
        last_eval = {"step": step, **result}
        append_jsonl(path = eval_path, record = last_eval)
        print(json.dumps(obj = {
            "eval_step": step,
            "loss": round(number = result["overall"]["loss"], ndigits = 4),
            "lang_mean_loss": round(number = result["overall"]["lang_mean_loss"], ndigits = 4),
            "token_acc": round(number = result["overall"]["token_acc"], ndigits = 4),
            "seconds": round(number = time.perf_counter() - start, ndigits = 1),
        }), flush = True)
        progress.write(
            step = step,
            last_eval = {
                "step": step,
                "loss": round(number = result["overall"]["loss"], ndigits = 4),
                "lang_mean_loss": round(number = result["overall"]["lang_mean_loss"], ndigits = 4),
                "token_acc": round(number = result["overall"]["token_acc"], ndigits = 4),
            },
        )

    def forward_backward(micro_batches, n_tokens):
        """一步内各微批的前向与反传，梯度累积在 projector（和语言模型的主权重）上；返回这一步的损失与正确数统计。"""
        loss_total = torch.zeros(size = (), device = device)
        correct_total = torch.zeros(size = (), device = device, dtype = torch.long)
        empty_loss_total = torch.zeros(size = (), device = device)
        empty_correct_total = torch.zeros(size = (), device = device, dtype = torch.long)
        for batch in micro_batches:
            inputs = move_batch(batch = batch, device = device, feature_dtype = feature_dtype)
            with optimized.forward_context():
                losses, correct, rows = token_losses(model = model, inputs = inputs, loss_impl = args.loss_impl)
            loss_sum = losses.sum()
            (loss_sum / n_tokens).backward()
            loss_total += loss_sum.detach()
            correct_total += correct.sum()
            # 空目标样本只有一个计损失的 token（<|im_end|>），单独统计它的损失和预测正确率。
            # 用乘法和逻辑与代替按布尔掩码取元素，不让 CPU 等 GPU 算出元素个数。
            on_empty = batch["empty_rows"].to(device = device)[rows]
            empty_loss_total += (losses.detach() * on_empty).sum()
            empty_correct_total += (correct & on_empty).sum()
        return float(loss_total), int(correct_total), empty_loss_total, empty_correct_total

    try:
        last_saved = step
        while step < total_steps:
            micro_batches = [stream.next() for _ in range(args.grad_accum)]
            n_tokens = sum(int((batch["labels"][:, 1:] != -100).sum()) for batch in micro_batches)
            while True:
                try:
                    totals = forward_backward(micro_batches = micro_batches, n_tokens = n_tokens)
                except torch.OutOfMemoryError:
                    totals = None
                if totals is not None:
                    break
                # 显存不足：丢掉这一步已累积的梯度，加大梯度检查点或流式加载后重试同一批，训练结果不变。
                # 在 except 之外处理，这时异常连同失败那一步的中间结果已经释放。
                optimizer.zero_grad(set_to_none = True)
                if language_weights is not None:
                    language_weights.zero_model_grads()
                torch.cuda.empty_cache()
                change = optimized.degrade()
                if change is None:
                    raise RuntimeError("显存不足：梯度检查点和全部层的流式加载都已打开")
                config["optimize_degraded"] = optimized.describe()
                print(f"第 {step + 1} 步显存不足，改为 {change}，重试同一批。", flush = True)
            step_loss, step_correct, empty_loss_total, empty_correct_total = totals

            if not checked_grads:
                trainable_names = set(expected)
                leaked = [
                    name for name, parameter in model.named_parameters()
                    if name not in trainable_names and parameter.grad is not None
                ]
                if leaked:
                    raise RuntimeError(f"冻结参数出现了梯度：{leaked[:5]}")
                missing = language_weights.without_grad() if language_weights is not None else []
                if missing:
                    raise RuntimeError(f"语言模型的这些参数没有得到梯度：{missing[:5]}")
                checked_grads = True

            grad_norm = torch.nn.utils.clip_grad_norm_(parameters = trainable_tensors, max_norm = args.max_grad_norm)
            if not math.isfinite(step_loss) or not torch.isfinite(input = grad_norm):
                optimizer.zero_grad(set_to_none = True)
                skipped_steps += 1
                ids = [sample_id for batch in micro_batches for sample_id in batch["sample_ids"]]
                print(f"第 {step + 1} 步损失或梯度不是有限值，跳过。样本：{ids[:8]}", flush = True)
                if skipped_steps >= 10:
                    raise RuntimeError("连续多步出现非有限值，停止训练。")
                continue
            skipped_steps = 0
            optimizer.step()
            if language_weights is not None:
                language_weights.copy_to_model()
            scheduler.step()
            optimizer.zero_grad(set_to_none = True)
            step += 1

            window["loss"] += step_loss
            window["tokens"] += n_tokens
            window["correct"] += step_correct
            window["samples"] += sum(len(batch["sample_ids"]) for batch in micro_batches)
            window["empty"] += sum(batch["n_empty"] for batch in micro_batches)
            window["empty_loss"] += float(empty_loss_total)
            window["empty_correct"] += int(empty_correct_total)
            window["mixed"] += sum(batch["n_mixed"] for batch in micro_batches)
            window["audio"] += sum(batch["audio_seconds"] for batch in micro_batches)
            window["steps"] += 1
            window["grad_norm"] += float(grad_norm)

            if step % args.log_every == 0 or step == total_steps:
                # 速度按训练本身的时间计算（不含 dev 评估和保存）；剩余时间按含这些开销的墙钟时间估计。
                train_seconds = time.perf_counter() - window_start
                last_train_loss = window["loss"] / window["tokens"]
                mem_gb = torch.cuda.max_memory_allocated() / 2 ** 30
                reserved_gb = torch.cuda.max_memory_reserved() / 2 ** 30
                max_mem_gb = max(max_mem_gb, mem_gb)
                progress.mark(step = step)
                eta = progress.eta_seconds(step = step)
                record = {
                    "step": step,
                    "total_steps": total_steps,
                    "epoch": round(number = stream.epoch + stream.offset / batches_per_epoch, ndigits = 4),
                    "loss": round(number = last_train_loss, ndigits = 5),
                    "token_acc": round(number = window["correct"] / window["tokens"], ndigits = 4),
                    # 这一段中空目标样本占全部样本、混音样本占语音样本的实际比例。
                    "empty_frac": round(number = window["empty"] / window["samples"], ndigits = 4),
                    # 空目标样本上 <|im_end|> 的平均损失，以及模型最可能输出的 token 就是 <|im_end|> 的比例。
                    "empty_loss": (
                        round(number = window["empty_loss"] / window["empty"], ndigits = 5)
                        if window["empty"] else None
                    ),
                    "empty_acc": (
                        round(number = window["empty_correct"] / window["empty"], ndigits = 4)
                        if window["empty"] else None
                    ),
                    "mixed_frac": round(
                        number = window["mixed"] / max(1, window["samples"] - window["empty"]),
                        ndigits = 4,
                    ),
                    # torchao 的优化器中学习率是张量，统一转成数值。
                    "lr": float(scheduler.get_last_lr()[0]),
                    "grad_norm": round(number = window["grad_norm"] / window["steps"], ndigits = 4),
                    "it_per_s": round(number = window["steps"] / train_seconds, ndigits = 3),
                    "samples_per_s": round(number = window["samples"] / train_seconds, ndigits = 2),
                    # 每秒处理的音频秒数，也就是每墙钟小时处理的音频小时数。
                    "audio_h_per_h": round(number = window["audio"] / train_seconds, ndigits = 1),
                    "mem_gb": round(number = mem_gb, ndigits = 2),
                    # PyTorch 缓存分配器向驱动申请并保留的显存峰值（含已释放、留作复用的部分），更接近显卡上的实际占用。
                    "reserved_gb": round(number = reserved_gb, ndigits = 2),
                    "elapsed_s": round(number = progress.elapsed()),
                    "eta_s": round(number = eta) if eta is not None else None,
                }
                if language_weights is not None:
                    # 语言模型参数组的学习率；lr 是 projector 参数组的。
                    record["lm_lr"] = float(scheduler.get_last_lr()[1])
                append_jsonl(path = log_path, record = record)
                print(progress.line(record = record), flush = True)
                progress.write(
                    step = step,
                    status = "running",
                    window = {
                        key: value for key, value in record.items()
                        if key not in ("step", "total_steps", "elapsed_s", "eta_s")
                    },
                    max_mem_gb = round(number = max_mem_gb, ndigits = 2),
                )
                torch.cuda.reset_peak_memory_stats()
                window = {key: 0 for key in window}
                window_start = time.perf_counter()

            pause_start = time.perf_counter()
            if args.eval_every and step % args.eval_every == 0:
                run_eval(step = step)
            if args.save_every and step % args.save_every == 0:
                save(step = step)
                last_saved = step
            # 吞吐只统计训练本身，扣除 dev 评估和保存检查点的时间。
            window_start += time.perf_counter() - pause_start

            timed_out = (
                args.stop_after_minutes is not None
                and time.perf_counter() - run_start > args.stop_after_minutes * 60
            )
            reached = args.stop_at_step is not None and step >= args.stop_at_step
            if (stop["reason"] or timed_out or reached) and step < total_steps:
                if last_saved != step:
                    save(step = step)
                reason = stop["reason"] or ("time limit" if timed_out else "stop step")
                label = stop["reason"] or ("达到时限" if timed_out else "达到指定步数")
                print(f"在第 {step} 步停止（{label}），用 --resume 继续。", flush = True)
                progress.write(step = step, status = "stopped", stop_reason = reason)
                return EXIT_STOPPED

        if eval_loader is not None and (last_eval is None or last_eval["step"] != step):
            run_eval(step = step)
        if before is not None:
            after = frozen_fingerprint(model = model)
            changed = sorted(name for name in before if before[name] != after.get(name))
            frozen_check = {"tensors": len(before), "changed": changed, "unchanged": not changed}
            (output_dir / "frozen_check.json").write_text(
                data = json.dumps(obj = frozen_check, ensure_ascii = False, indent = 2) + "\n",
                encoding = "utf-8",
            )
            print(f"冻结参数核对：{len(before)} 个张量，变化 {len(changed)} 个。", flush = True)
        if last_saved != step or frozen_check is not None:
            save(step = step)
        progress.write(step = step, status = "finished")
        return 0
    except BaseException as error:
        progress.write(step = step, status = "failed", error = repr(error)[:500])
        raise


if __name__ == "__main__":
    sys.exit(main())
