"""自生成回放（任务 #71 C 组；回放数据见 #70 与 pipeline/README.md 第 9 节）。

全量微调时，在 ASR 批次之外混入参考模型自己采样的文本，减轻语言模型的遗忘。用户 2026-11-23 的决定：

- 损失只算模型生成的部分。无模板两组（free_raw、cont_raw）是全部生成 token（cont_raw 的人写文档开头只作上下文）；
  带模板的组只算助手轮：生成的助手内容和结尾的 <|im_end|>。free_chat、free_chat_think 中模型自己写的用户轮只作上下文，
  不计损失；free_chat_think 的助手轮包括推理和 </think>（<think>\\n 属于模板）。按回放数据的存放格式，就是 role 为
  text 或 assistant 的生成片段（LOSS_ROLES）计损失，role 为 user 的生成片段和直接给定的 token（提示、模板）不计。
- 样本整条使用，从不截断。训练时直接用保存的 token id（提示片段的 ids 加生成片段的 [offset, offset + keep)），不重新分词。
- 比例 r = 回放损失 token 数 / ASR 转写损失 token 数，按长期平均控制（ReplaySampler）。ASR 批次与不加回放时完全相同
  （同一批分片、同一读取顺序、同样的步数），回放额外加在同一步里。
- 一步内全部损失 token（ASR 加回放）一起取平均（--replay-norm step），r 同时决定回放的量和权重。分开记录 ASR 与回放的
  损失，回放再按组记录。
- 回放损失 --replay-loss：ntp（默认，对保存的采样 token 做下一个 token 预测的交叉熵）或 kl32（对参考模型 top-32 的有偏
  截断前向 KL，见 model.LinearTopKL）。
- --replay-groups free_raw=1,free_chat=1,... 按 token 数混合各组；C-mix 是五组等量。
- 续训时恢复抽样器的状态，续训之后取到的样本与不中断时相同。

抽样（ReplaySampler）
====================

每一步先数出这一步 ASR 批次中计损失的 token 数 A（与损失归一化用的是同一个数），预算 budget 加上 r × A，然后只要预算
大于 0 就整条取下一个回放样本，预算减去它的损失 token 数。预算可以变成负数（一条长样本之后接下来几步不再取样），所以
从开始到任意一步，回放损失 token 的总数与 r 倍 ASR 损失 token 总数之差不超过一条样本的长度。取哪一组：已取 token 数除以
该组权重最小的组（并列时按 --replay-groups 中的顺序），各组的 token 份额因此按权重收敛。每组内按 (seed, 组, 轮次) 打乱的
固定顺序取样，取完一遍重新打乱再取。抽样器的状态（预算、各组已取 token 数、轮次、位置）随检查点保存在
trainer_state.pt 的 replay 项中，只由这些状态和每步的 A 决定，不用其他随机数。

长样本与拼接
============

回放只支持拼接批次（--pack-sequences）。取到的样本按整条长度（上下文加生成）分两类：

- 不超过 --replay-inline-max-tokens（默认 2,048）的短样本接在这一步 ASR 微批拼成的那一行后面（append_samples），音频样本
  和文本样本在同一行里，各自从位置 0 编号，cu_seq_lens_q 和 seq_idx 标出边界，与 data.pack_batch 的拼接规则相同；
  有多个 ASR 微批（--grad-accum）时接在当前最短的那一个后面。
- 更长的样本另成纯文本的微批（text_batch），属于同一个优化步（梯度累积），每个微批的总长不超过
  --replay-micro-batch-tokens（默认 16,640，回放数据中最长的样本 16,639 个 token），超过上限的单条样本自己成一个微批。
  这样一条几千到 16k token 的样本与 ASR 的一行分开前向、反传，一步的峰值显存取两者中较大的一个。

一步之内有多次反传，所以回放与「语言模型参数在主机内存中更新」（--offload-optimizer、--lm-blocks-to-swap，要求
--grad-accum 1）不能同时使用。

损失归一化与实际权重
====================

--replay-norm step（用户的决定）：一步的损失 = 这一步全部损失 token 的损失之和 / 这一步的损失 token 总数。回放按整条样本
加入，样本比一步的 ASR token 长得多时（free_raw 平均约 1,860 个损失 token，一步的 ASR 约几百个），回放集中在少数步里，
这些步里每个回放 token 的权重被稀释，整个训练中回放梯度所占的份额低于 r / (1 + r)，样本越长的组越低。训练日志中的
replay_grad_share 是实际份额（每步梯度总量中回放所占部分的平均），与 r / (1 + r) 对照。

--replay-norm asr：分母改为 (1 + r) × 这一步的 ASR 损失 token 数，每个 token 的权重不随这一步取到多少回放而变，回放的
份额在长期平均上等于 r / (1 + r)；代价是取到长样本的那一步梯度总量变大（梯度裁剪会限制它）。

按回放数据的实际长度模拟 3 万步（r = 0.5，每步 ASR 约 330 个损失 token，v2b 清单抽样估计）：step 时回放在梯度中的实际
份额为 free_raw 5.5%、free_chat 9.9%、cont_raw 6.9%、cont_chat 19.9%、只用 free_chat_think 4.3%、五组等量 9.1%（目标
33.3%），只有 6%–38% 的步取到回放，1%–3% 的步有长样本的纯文本微批；asr 时各组都约 33%，但取到 16k 样本的那一步回放的
梯度总量可达 ASR 部分的约 120 倍。两种归一化下实际比例（回放损失 token / ASR 损失 token）都是 0.500。

日志
====

<输出目录>/replay_log.jsonl 每次取样一行：步数、这一步的 ASR 损失 token 数、取到的样本（组、sample_id、损失 token 数、
总长、放在哪里）和取样后的预算。续训时截掉续训起点之后的行。verify_log 用日志中各步的 ASR token 数重新模拟抽样器，
核对取到的样本与日志一致（Colab 冒烟测试用来检查续训）。
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from pipeline.replay.store import ShardReader
from pipeline.replay.teacher import SUFFIXES as TEACHER_SUFFIXES
from pipeline.replay.teacher import teacher_paths
from pipeline.train.data import IGNORE_INDEX

REPLAY_LOSSES = ("ntp", "kl32")
REPLAY_NORMS = ("step", "asr")
REPLAY_SPLITS = ("train", "eval")
# 损失来源的编号：0 是 ASR，回放各组从 1 开始。日志按它分组统计，取样顺序的随机数也用它区分各组。
ASR_SOURCE = 0
GROUP_CODES = {
    "free_raw": 1,
    "free_chat": 2,
    "cont_raw": 3,
    "cont_chat": 4,
    "free_chat_think": 5,
}
SOURCE_COUNT = 1 + len(GROUP_CODES)
# 计损失的生成片段：无模板组的 text，带模板组的 assistant。user（模型自己写的用户轮）只作上下文。
LOSS_ROLES = ("text", "assistant")
TOP_K = 32
# 每组样本顺序的随机数流编号，与 data.py 的 STREAM_* 区分开。
STREAM_ORDER = 71
LOG_FILE = "replay_log.jsonl"
# 回放样本从不截断，纯文本微批的默认上限覆盖最长的一条（16,639 个 token）。
DEFAULT_MICRO_BATCH_TOKENS = 16640
DEFAULT_INLINE_MAX_TOKENS = 2048
THINK_END_ID = 248069


def parse_groups(spec):
    """「组=权重,组=权重」→ {组: 权重}，保持给出的顺序；权重为 0 的组不参与。"""
    weights = {}
    for item in spec.split(sep = ","):
        item = item.strip()
        if not item:
            continue
        name, _, value = item.partition("=")
        name = name.strip()
        if name not in GROUP_CODES:
            raise ValueError(f"未知的回放组：{name}，可选 {tuple(GROUP_CODES)}")
        if name in weights:
            raise ValueError(f"回放组 {name} 重复")
        weight = float(value) if value.strip() else 1.0
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"回放组 {name} 的权重应为非负数：{value}")
        weights[name] = weight
    weights = {name: weight for name, weight in weights.items() if weight > 0}
    if not weights:
        raise ValueError(f"--replay-groups 中没有权重大于 0 的组：{spec}")
    return weights


def piece_keep(piece):
    """生成片段保留的 token 数（samples.jsonl 中的 keep，没有时为整段长度）。"""
    return int(piece.get("keep", piece["length"]))


def count_tokens(pieces):
    """(样本总长, 计损失的 token 数)。"""
    total = 0
    loss = 0
    for piece in pieces:
        if "gen" in piece:
            keep = piece_keep(piece = piece)
            total += keep
            if piece["gen"] in LOSS_ROLES:
                loss += keep
        else:
            total += len(piece["ids"])
    return total, loss


class ReplayGroup:
    """一组一个划分的回放样本：samples.jsonl 中 keep 为真、有计损失 token 的行。

    只在内存中保留每行在文件中的位置、shard 编号、总长和损失 token 数，用到时再读出这一行（行中有生成文本，较大）。
    token 从 raw/ 的 shard 读，参考模型的 top-32 从 teacher/ 读（kl32 和回放差距评测用），都是内存映射。
    """

    def __init__(self, root, group, split = "train"):
        if group not in GROUP_CODES:
            raise ValueError(f"未知的回放组：{group}")
        self.group = group
        self.code = GROUP_CODES[group]
        self.directory = Path(root) / group / split
        self.samples_path = self.directory / "samples.jsonl"
        if not self.samples_path.exists():
            raise FileNotFoundError(f"没有找到回放样本 {self.samples_path}")
        offsets, shards, totals, losses = [], [], [], []
        position = 0
        with open(file = self.samples_path, mode = "rb") as handle:
            for line in handle:
                start = position
                position += len(line)
                if not line.strip():
                    continue
                record = json.loads(s = line)
                if not record.get("keep"):
                    continue
                total, loss = count_tokens(pieces = record["pieces"])
                if loss == 0:
                    continue
                offsets.append(start)
                shards.append(int(record["shard"]))
                totals.append(total)
                losses.append(loss)
        if not offsets:
            raise ValueError(f"{self.samples_path} 中没有可用的回放样本")
        self.offsets = np.asarray(a = offsets, dtype = np.int64)
        self.shards = np.asarray(a = shards, dtype = np.int32)
        self.totals = np.asarray(a = totals, dtype = np.int64)
        self.losses = np.asarray(a = losses, dtype = np.int64)
        self.raw = ShardReader(raw_dir = self.directory / "raw")
        self._teacher = {}
        self._handle = None

    def __len__(self):
        return len(self.offsets)

    def record(self, index):
        """samples.jsonl 中第 index 个可用样本的完整一行。"""
        if self._handle is None:
            self._handle = open(file = self.samples_path, mode = "rb")
        self._handle.seek(int(self.offsets[index]))
        return json.loads(s = self._handle.readline())

    def teacher_arrays(self, shard):
        """teacher/ 中 shard 的 top_ids、top_lp、lse、lp 四个数组（内存映射），与 raw shard 的生成 token 逐行对应。"""
        if shard not in self._teacher:
            arrays = {}
            for name, path in teacher_paths(teacher_dir = self.directory / "teacher", number = shard).items():
                _, dtype, width = TEACHER_SUFFIXES[name]
                if not path.exists():
                    raise FileNotFoundError(f"没有参考模型的 top-32 文件 {path}（kl32 和回放差距评测需要 teacher/）")
                data = np.memmap(filename = path, dtype = dtype, mode = "r")
                arrays[name] = data.reshape(-1, width) if width > 1 else data
            self._teacher[shard] = arrays
        return self._teacher[shard]

    def sample(self, index, teacher = False, reference = False):
        """读出一条样本。

        返回字典：group、code、sample_id、ids（int64，整条 token）、labels（int64，计损失的位置是 token 本身，其余为 -100）、
        n_tokens、n_loss、record（samples.jsonl 的整行）。teacher 为真时加 teacher_ids（int64，[n_loss, 32]）和
        teacher_lp（float32，[n_loss, 32]，参考模型的对数概率）；reference 为真时加 ref_lp（float32，[n_loss]，参考模型对
        实际 token 的对数概率，取 teacher/ 的 lp，没有 teacher/ 时取 raw/ 中采样时记录的 lp）。各数组的行与计损失的位置
        按顺序一一对应。
        """
        record = self.record(index = index)
        shard = int(record["shard"])
        ids, labels = [], []
        top_ids, top_lp, ref_lp = [], [], []
        for piece in record["pieces"]:
            if "gen" not in piece:
                part = np.asarray(a = piece["ids"], dtype = np.int64)
                ids.append(part)
                labels.append(np.full(shape = part.shape, fill_value = IGNORE_INDEX, dtype = np.int64))
                continue
            keep = piece_keep(piece = piece)
            part = np.asarray(a = self.raw.piece(number = shard, piece = piece)[:keep], dtype = np.int64)
            if len(part) != keep:
                raise ValueError(f"{record['sample_id']} 的生成片段只读出 {len(part)} 个 token，应为 {keep}")
            ids.append(part)
            if piece["gen"] not in LOSS_ROLES:
                labels.append(np.full(shape = part.shape, fill_value = IGNORE_INDEX, dtype = np.int64))
                continue
            labels.append(part)
            rows = slice(int(piece["offset"]), int(piece["offset"]) + keep)
            if teacher or (reference and (self.directory / "teacher").exists()):
                arrays = self.teacher_arrays(shard = shard)
                if teacher:
                    top_ids.append(np.asarray(a = arrays["top_ids"][rows], dtype = np.int64))
                    top_lp.append(np.asarray(a = arrays["top_lp"][rows], dtype = np.float32))
                if reference:
                    ref_lp.append(np.asarray(a = arrays["lp"][rows], dtype = np.float32))
            elif reference:
                ref_lp.append(np.asarray(a = self.raw.piece(number = shard, piece = piece, name = "lp")[:keep], dtype = np.float32))
        ids = np.concatenate(ids)
        labels = np.concatenate(labels)
        if labels[0] != IGNORE_INDEX:
            raise ValueError(f"{record['sample_id']} 的第一个 token 计损失，前面没有上下文")
        sample = {
            "group": self.group,
            "code": self.code,
            "sample_id": record["sample_id"],
            "ids": ids,
            "labels": labels,
            "n_tokens": len(ids),
            "n_loss": int((labels != IGNORE_INDEX).sum()),
            "record": record,
        }
        if teacher:
            sample["teacher_ids"] = np.concatenate(top_ids)
            sample["teacher_lp"] = np.concatenate(top_lp)
            if len(sample["teacher_ids"]) != sample["n_loss"]:
                raise ValueError(f"{record['sample_id']} 的 top-32 行数与计损失的 token 数不一致")
        if reference:
            sample["ref_lp"] = np.concatenate(ref_lp)
        return sample


def load_groups(root, weights, split = "train"):
    """按 --replay-groups 的顺序读入各组的样本索引。"""
    return {name: ReplayGroup(root = root, group = name, split = split) for name in weights}


class ReplaySampler:
    """按比例 r 取回放样本（见模块说明「抽样」）。状态只有几个数，state_dict 可以直接存进 trainer_state.pt。"""

    def __init__(
        self,
        groups,
        weights,
        ratio,
        seed,
    ):
        """groups 为 {组名: ReplayGroup 或任何有 losses（各样本损失 token 数）和 code 的对象}，weights 同 parse_groups。"""
        if not ratio > 0:
            raise ValueError(f"回放比例应大于 0：{ratio}")
        self.names = list(weights)
        self.weights = dict(weights)
        self.losses = {name: np.asarray(a = groups[name].losses, dtype = np.int64) for name in self.names}
        self.codes = {name: GROUP_CODES[name] for name in self.names}
        self.ratio = float(ratio)
        self.seed = int(seed)
        self._orders = {}
        self.state = {
            "budget": 0.0,
            "asr_tokens": 0,
            "replay_tokens": 0,
            "samples": 0,
            "tokens": {name: 0 for name in self.names},
            "epoch": {name: 0 for name in self.names},
            "position": {name: 0 for name in self.names},
        }

    def order(self, name, epoch):
        """组 name 第 epoch 遍的样本顺序，只由 (seed, 组, 轮次) 决定。"""
        key = (name, epoch)
        if key not in self._orders:
            rng = np.random.default_rng(seed = [self.seed, STREAM_ORDER, self.codes[name], epoch])
            self._orders = {k: v for k, v in self._orders.items() if k[0] != name}
            self._orders[key] = rng.permutation(len(self.losses[name]))
        return self._orders[key]

    def next_group(self):
        """已取 token 数除以权重最小的组；并列时取 --replay-groups 中靠前的。"""
        state = self.state
        return min(self.names, key = lambda name: (state["tokens"][name] / self.weights[name], self.names.index(name)))

    def take(self, asr_tokens):
        """这一步的 ASR 损失 token 数为 asr_tokens 时取哪些样本，返回 [(组名, 样本下标)]，并更新状态。"""
        state = self.state
        state["asr_tokens"] += int(asr_tokens)
        state["budget"] += self.ratio * int(asr_tokens)
        picked = []
        while state["budget"] > 0:
            name = self.next_group()
            epoch, position = state["epoch"][name], state["position"][name]
            index = int(self.order(name = name, epoch = epoch)[position])
            position += 1
            if position == len(self.losses[name]):
                epoch += 1
                position = 0
            state["epoch"][name], state["position"][name] = epoch, position
            loss = int(self.losses[name][index])
            state["tokens"][name] += loss
            state["replay_tokens"] += loss
            state["samples"] += 1
            state["budget"] -= loss
            picked.append((name, index))
        return picked

    def state_dict(self):
        return json.loads(s = json.dumps(obj = self.state))

    def load_state_dict(self, state):
        if set(state["tokens"]) != set(self.names):
            raise ValueError(f"续训状态中的回放组 {sorted(state['tokens'])} 与本次 {sorted(self.names)} 不一致")
        self.state = json.loads(s = json.dumps(obj = state))


def route(samples, inline_max_tokens, micro_batch_tokens):
    """把这一步取到的样本分成 (接在 ASR 行后面的短样本, [纯文本微批的样本列表])。

    长样本按取样顺序依次放进第一个放得下的纯文本微批（总长不超过 micro_batch_tokens），放不下就新开一个；
    单条超过上限的样本自己成一个微批，不截断。
    """
    inline = [sample for sample in samples if sample["n_tokens"] <= inline_max_tokens]
    batches = []
    sizes = []
    for sample in samples:
        if sample["n_tokens"] <= inline_max_tokens:
            continue
        for number, size in enumerate(sizes):
            if size + sample["n_tokens"] <= micro_batch_tokens:
                batches[number].append(sample)
                sizes[number] += sample["n_tokens"]
                break
        else:
            batches.append([sample])
            sizes.append(sample["n_tokens"])
    return inline, batches


def _sample_rows(samples, offset, first_row):
    """拼接用的各列：样本依次从 offset 开始放，行号从 first_row 开始。"""
    columns = {
        "ids": [],
        "labels": [],
        "positions": [],
        "seq_idx": [],
        "loss_index": [],
        "loss_targets": [],
        "loss_rows": [],
        "loss_source": [],
        "teacher_ids": [],
        "teacher_lp": [],
        "ends": [],
    }
    for number, sample in enumerate(samples):
        ids = torch.from_numpy(sample["ids"]).long()
        labels = torch.from_numpy(sample["labels"]).long()
        length = len(ids)
        row = first_row + number
        targets = (labels != IGNORE_INDEX).nonzero(as_tuple = True)[0]
        columns["ids"].append(ids)
        columns["labels"].append(labels)
        columns["positions"].append(torch.arange(length))
        columns["seq_idx"].append(torch.full(size = (length,), fill_value = row, dtype = torch.int32))
        # 第 t 个 token 由第 t - 1 个位置预测
        columns["loss_index"].append(offset + targets - 1)
        columns["loss_targets"].append(ids[targets])
        columns["loss_rows"].append(torch.full(size = (len(targets),), fill_value = row, dtype = torch.long))
        columns["loss_source"].append(torch.full(size = (len(targets),), fill_value = sample["code"], dtype = torch.long))
        if "teacher_ids" in sample:
            columns["teacher_ids"].append(torch.from_numpy(sample["teacher_ids"]).long())
            columns["teacher_lp"].append(torch.from_numpy(sample["teacher_lp"]).float())
        offset += length
        columns["ends"].append(offset)
    return columns


def _cat(tensors, dtype):
    return torch.cat(tensors = tensors) if tensors else torch.zeros(size = (0,), dtype = dtype)


def source_tokens(loss_source):
    """各来源计损失的 token 数（长度 SOURCE_COUNT 的列表）。"""
    return np.bincount(loss_source.numpy(), minlength = SOURCE_COUNT).tolist()


def append_samples(batch, samples, kl = False):
    """把回放样本接在一个拼接好的 ASR 微批（data.pack_batch 的结果）后面，返回新的批次。

    各条文本样本像 pack_batch 中的一条样本一样：位置从 0 编号，cu_seq_lens_q、seq_idx 标出边界，loss_index 是预测下一个
    计损失 token 的位置（这一行中的下标）。不接样本时也补上 loss_source（全为 ASR），训练循环按来源统计损失。
    kl 为真时补上 kl_start（ASR 计损失位置的个数，之后的位置都属于回放）和对应回放位置的 teacher_ids、teacher_lp。
    """
    if "cu_seq_lens_q" not in batch:
        raise ValueError("回放样本只能接在拼接的批次后面（--pack-sequences）")
    batch = dict(batch)
    asr_positions = batch["loss_index"].shape[0]
    sources = torch.full(size = (asr_positions,), fill_value = ASR_SOURCE, dtype = torch.long)
    if not samples:
        batch["loss_source"] = sources
        batch["source_tokens"] = source_tokens(loss_source = sources)
        batch["replay_ids"] = []
        return batch
    length = batch["input_ids"].shape[1]
    starts = batch["cu_seq_lens_q"]
    first_row = len(starts) - 1
    columns = _sample_rows(samples = samples, offset = length, first_row = first_row)
    ids = torch.cat(tensors = columns["ids"])
    added = len(ids)
    batch["input_ids"] = torch.cat(tensors = [batch["input_ids"], ids[None].to(dtype = batch["input_ids"].dtype)], dim = 1)
    batch["attention_mask"] = torch.ones(size = (1, length + added), dtype = batch["attention_mask"].dtype)
    batch["mm_token_type_ids"] = torch.cat(
        tensors = [batch["mm_token_type_ids"], torch.zeros(size = (1, added), dtype = batch["mm_token_type_ids"].dtype)],
        dim = 1,
    )
    batch["labels"] = torch.cat(tensors = [batch["labels"], torch.cat(tensors = columns["labels"])[None]], dim = 1)
    positions = torch.cat(tensors = columns["positions"])
    batch["position_ids"] = torch.cat(
        tensors = [batch["position_ids"], positions[None, None].expand(size = (3, -1, -1))],
        dim = 2,
    ).contiguous()
    batch["audio_mask"] = torch.cat(tensors = [batch["audio_mask"], torch.zeros(size = (1, added), dtype = torch.bool)], dim = 1)
    batch["loss_index"] = torch.cat(tensors = [batch["loss_index"], torch.cat(tensors = columns["loss_index"])])
    batch["loss_targets"] = torch.cat(tensors = [batch["loss_targets"], torch.cat(tensors = columns["loss_targets"])])
    batch["loss_rows"] = torch.cat(tensors = [batch["loss_rows"], torch.cat(tensors = columns["loss_rows"])])
    batch["loss_source"] = torch.cat(tensors = [sources, torch.cat(tensors = columns["loss_source"])])
    batch["cu_seq_lens_q"] = torch.cat(tensors = [starts, torch.tensor(data = columns["ends"], dtype = starts.dtype)])
    batch["max_length_q"] = max(int(batch["max_length_q"]), max(sample["n_tokens"] for sample in samples))
    batch["seq_idx"] = torch.cat(tensors = [batch["seq_idx"], torch.cat(tensors = columns["seq_idx"])[None]], dim = 1)
    batch["empty_rows"] = torch.cat(tensors = [batch["empty_rows"], torch.zeros(size = (len(samples),), dtype = torch.bool)])
    batch["source_tokens"] = source_tokens(loss_source = batch["loss_source"])
    batch["replay_ids"] = [sample["sample_id"] for sample in samples]
    if kl:
        batch["kl_start"] = asr_positions
        batch["teacher_ids"] = torch.cat(tensors = columns["teacher_ids"])
        batch["teacher_lp"] = torch.cat(tensors = columns["teacher_lp"])
    return batch


def text_batch(samples, kl = False):
    """只有回放样本的拼接微批（长样本用），键与 append_samples 的结果相同，只是没有音频的各项。"""
    columns = _sample_rows(samples = samples, offset = 0, first_row = 0)
    ids = torch.cat(tensors = columns["ids"])
    total = len(ids)
    batch = {
        "input_ids": ids[None],
        "attention_mask": torch.ones(size = (1, total), dtype = torch.long),
        "mm_token_type_ids": torch.zeros(size = (1, total), dtype = torch.long),
        "labels": torch.cat(tensors = columns["labels"])[None],
        "position_ids": torch.cat(tensors = columns["positions"])[None, None].expand(size = (3, -1, -1)).contiguous(),
        "loss_index": torch.cat(tensors = columns["loss_index"]),
        "loss_targets": torch.cat(tensors = columns["loss_targets"]),
        "loss_rows": torch.cat(tensors = columns["loss_rows"]),
        "loss_source": torch.cat(tensors = columns["loss_source"]),
        "cu_seq_lens_q": torch.tensor(data = [0] + columns["ends"], dtype = torch.int32),
        "max_length_q": max(sample["n_tokens"] for sample in samples),
        "seq_idx": torch.cat(tensors = columns["seq_idx"])[None],
        "sample_ids": [],
        "langs": [],
        "audio_seconds": 0.0,
        "n_empty": 0,
        "empty_rows": torch.zeros(size = (len(samples),), dtype = torch.bool),
        "n_mixed": 0,
        "replay_ids": [sample["sample_id"] for sample in samples],
    }
    batch["source_tokens"] = source_tokens(loss_source = batch["loss_source"])
    if kl:
        batch["kl_start"] = 0
        batch["teacher_ids"] = torch.cat(tensors = columns["teacher_ids"])
        batch["teacher_lp"] = torch.cat(tensors = columns["teacher_lp"])
    return batch


class Replay:
    """训练循环用的回放：读入各组、按比例取样、把样本放进这一步的微批，并记录日志。"""

    def __init__(
        self,
        root,
        weights,
        ratio,
        seed,
        loss = "ntp",
        split = "train",
        norm = "step",
        inline_max_tokens = DEFAULT_INLINE_MAX_TOKENS,
        micro_batch_tokens = DEFAULT_MICRO_BATCH_TOKENS,
        log_path = None,
    ):
        if loss not in REPLAY_LOSSES:
            raise ValueError(f"未知的回放损失：{loss}，可选 {REPLAY_LOSSES}")
        if norm not in REPLAY_NORMS:
            raise ValueError(f"未知的回放归一化：{norm}，可选 {REPLAY_NORMS}")
        self.weights = weights
        self.groups = load_groups(root = root, weights = weights, split = split)
        self.sampler = ReplaySampler(
            groups = self.groups,
            weights = weights,
            ratio = ratio,
            seed = seed,
        )
        self.ratio = float(ratio)
        self.kl = loss == "kl32"
        self.norm = norm
        self.inline_max_tokens = int(inline_max_tokens)
        self.micro_batch_tokens = int(micro_batch_tokens)
        self.log_path = Path(log_path) if log_path is not None else None

    def describe(self):
        """开始训练时打印的概况：各组样本数、损失 token 总数、按比例估计的每步 token 数。"""
        return {
            "ratio": self.ratio,
            "loss": "kl32" if self.kl else "ntp",
            "norm": self.norm,
            "inline_max_tokens": self.inline_max_tokens,
            "micro_batch_tokens": self.micro_batch_tokens,
            "groups": {
                name: {
                    "weight": self.weights[name],
                    "samples": len(group),
                    "loss_tokens": int(group.losses.sum()),
                    "tokens": int(group.totals.sum()),
                    "longest": int(group.totals.max()),
                    "over_inline": int((group.totals > self.inline_max_tokens).sum()),
                }
                for name, group in self.groups.items()
            },
        }

    def read(self, picked):
        return [self.groups[name].sample(index = index, teacher = self.kl) for name, index in picked]

    def longest(self):
        """各组中整条最长的样本（--replay-probe-longest 用来在冒烟测试中测峰值显存）。"""
        name = max(self.groups, key = lambda name: int(self.groups[name].totals.max()))
        index = int(np.argmax(a = self.groups[name].totals))
        return self.read(picked = [(name, index)])[0]

    def extend(self, micro_batches, asr_tokens, step, probe = False):
        """取这一步的回放样本并放进微批，返回 (新的微批列表, 这一步的回放信息)。

        asr_tokens 是这一步 ASR 微批中计损失的 token 数。probe 为真时另加一个只含最长样本的纯文本微批（不经过抽样器）。
        """
        picked = self.sampler.take(asr_tokens = asr_tokens)
        samples = self.read(picked = picked)
        inline, long_batches = route(
            samples = samples,
            inline_max_tokens = self.inline_max_tokens,
            micro_batch_tokens = self.micro_batch_tokens,
        )
        groups = [[] for _ in micro_batches]
        sizes = [int(batch["input_ids"].shape[1]) for batch in micro_batches]
        for sample in inline:
            number = int(np.argmin(a = sizes))
            groups[number].append(sample)
            sizes[number] += sample["n_tokens"]
        batches = [
            append_samples(batch = batch, samples = members, kl = self.kl)
            for batch, members in zip(micro_batches, groups)
        ]
        batches += [text_batch(samples = members, kl = self.kl) for members in long_batches]
        probed = None
        if probe:
            probed = self.longest()
            batches.append(text_batch(samples = [probed], kl = self.kl))
        replay_tokens = sum(sample["n_loss"] for sample in samples) + (probed["n_loss"] if probed else 0)
        info = {
            "step": step,
            "asr_tokens": int(asr_tokens),
            "replay_tokens": replay_tokens,
            "samples": len(samples),
            "long_batches": len(long_batches) + (1 if probed else 0),
            "context_tokens": sum(sample["n_tokens"] - sample["n_loss"] for sample in samples),
        }
        if self.log_path is not None:
            long_ids = {sample["sample_id"] for members in long_batches for sample in members}
            record = {
                "step": step,
                "asr_tokens": int(asr_tokens),
                "samples": [
                    [sample["group"], sample["sample_id"], sample["n_loss"], sample["n_tokens"],
                     "long" if sample["sample_id"] in long_ids else "inline"]
                    for sample in samples
                ],
                "budget": round(number = self.sampler.state["budget"], ndigits = 6),
            }
            if probed is not None:
                record["probe"] = [probed["group"], probed["sample_id"], probed["n_loss"], probed["n_tokens"]]
            with open(file = self.log_path, mode = "a", encoding = "utf-8") as handle:
                handle.write(json.dumps(obj = record, ensure_ascii = False) + "\n")
        return batches, info

    def normalizer(self, asr_tokens, replay_tokens):
        """这一步损失之和的分母（--replay-norm）：step 为全部损失 token 数，asr 为 (1 + r) × ASR 损失 token 数。"""
        if self.norm == "asr":
            return (1 + self.ratio) * asr_tokens
        return asr_tokens + replay_tokens

    def state_dict(self):
        return self.sampler.state_dict()

    def load_state_dict(self, state):
        self.sampler.load_state_dict(state = state)

    def truncate_log(self, step):
        """续训时去掉日志中续训起点之后的行（那些步会重新训练、重新记录）。"""
        if self.log_path is None or not self.log_path.exists():
            return
        lines = self.log_path.read_text(encoding = "utf-8").splitlines()
        kept = []
        for line in lines:
            try:
                if json.loads(s = line)["step"] <= step:
                    kept.append(line)
            except (ValueError, KeyError):
                break
        self.log_path.write_text(data = "".join(line + "\n" for line in kept), encoding = "utf-8")


def verify_log(output_dir):
    """按训练配置重建抽样器，用 replay_log.jsonl 中各步的 ASR token 数重新取样，核对与日志记录的样本相同。

    返回 {"steps": 核对的取样次数, "samples": 样本数, "mismatch": 第一处不同（没有时为 None）}。续训或中途停下再续训的运行
    也应完全一致；冒烟测试在续训之后调用它。
    """
    output_dir = Path(output_dir)
    config = json.loads(s = (output_dir / "train_config.json").read_text(encoding = "utf-8"))
    weights = parse_groups(spec = config["replay_groups"])
    groups = load_groups(root = config["replay_dir"], weights = weights, split = config.get("replay_split", "train"))
    sampler = ReplaySampler(
        groups = groups,
        weights = weights,
        ratio = config["replay_ratio"],
        seed = config["seed"],
    )
    steps = 0
    count = 0
    with open(file = output_dir / LOG_FILE, encoding = "utf-8") as handle:
        for line in handle:
            record = json.loads(s = line)
            picked = sampler.take(asr_tokens = record["asr_tokens"])
            want = [(name, groups[name].record(index = index)["sample_id"]) for name, index in picked]
            got = [(item[0], item[1]) for item in record["samples"]]
            steps += 1
            count += len(got)
            if want != got:
                return {"steps": steps, "samples": count, "mismatch": {"step": record["step"], "expected": want, "logged": got}}
    return {"steps": steps, "samples": count, "mismatch": None}


def main(argv = None):
    parser = argparse.ArgumentParser(description = "回放日志的核对：按训练配置重新模拟抽样器，与 replay_log.jsonl 比较")
    commands = parser.add_subparsers(dest = "command", required = True)
    verify = commands.add_parser(name = "verify-log", help = "核对一次训练的回放日志")
    verify.add_argument("output_dir", help = "训练输出目录（含 train_config.json 与 replay_log.jsonl）")
    args = parser.parse_args(args = argv)
    result = verify_log(output_dir = args.output_dir)
    print(json.dumps(obj = result, ensure_ascii = False), flush = True)
    return 0 if result["mismatch"] is None and result["steps"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
