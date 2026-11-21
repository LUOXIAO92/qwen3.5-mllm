"""训练清单读取、目标文本选择与批次构造。

一条样本是一段完整的对话：用户消息为音频加固定指令，助手回答为目标文本。整段对话用阶段 1 模型自带的
chat template 渲染，再交给 processor 做特征提取和音频标记展开，得到最终的 input_ids。标签由
mm_assistant_mask（https://github.com/LUOXIAO92/MultimodalAssistantMask）在最终 input_ids 上按 token
匹配助手回合生成：助手回合开头和空的思考块不计损失，目标文本和结尾的 <|im_end|> 计算损失，其余位置填 -100。
transformers 的 return_assistant_tokens_mask 按展开音频标记之前的字符位置计算，展开后会错位或全为 0，
所以不用它。模型内部再把标签左移一位，用第 t 个位置的输出预测第 t + 1 个 token。

训练目标为空字符串的样本（空目标样本，即无人声片段）照常渲染：助手回合只有空的思考块和 <|im_end|>，
按同样的规则只有 <|im_end|> 计算损失，模型学着听到无人声的音频时直接结束输出。

以上是默认的聊天格式（prompt_format = "chat"）。纯续写预训练格式（prompt_format = "cpt"，任务 #71）不套 chat
template，没有系统、用户、助手回合：一条样本是一段音频的标记（与聊天格式中的音频标记相同，
<|audio_start|><|audio_pad|><|audio_end|>，由 processor 展开成按音频长度计的 <|audio_pad|>），后面直接接目标文本，
以纯文本的结束标记 <|endoftext|> 结尾。标签在最终的 input_ids 上按位置确定：<|audio_end|> 之后的有效 token
（目标文本和 <|endoftext|>）计算损失，其余位置填 -100。空目标样本只有 <|endoftext|> 计算损失。

训练时数据集按 sampling.SampleKey 取样本，给出背景声混音器时对语音样本实时混音（mixing.py），
空目标样本不混音；按整数下标取样本时（dev 评估、标签核对）不混音。

音频可以从分片读取（shards.py，Colab 训练用）：给出 ShardReader 时，数据集按清单行的 audio_shard =
(分片编号, 数据起点, 长度) 读出这个文件的原始字节，再从字节解码，结果与直接解码 audio_path 指向的文件相同。
"""

import json
import mmap
import random
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
from mm_assistant_mask import AssistantMaskSpec, build_assistant_labels

# 训练时数据集的下标：清单行号、轮次、在本轮样本序列中的位置（见 sampling.py）。
SampleKey = namedtuple(typename = "SampleKey", field_names = ("index", "epoch", "position"))
# 随机数流的编号：每轮的抽样次数和混音各用一条，互不影响。
STREAM_OCCURRENCES = 1
STREAM_MIXING = 2
IGNORE_INDEX = -100
SAMPLING_RATE = 16000
INSTRUCTION = "Transcribe the audio."
TARGETS = ("script", "original")
# 助手回合在 token 序列中的边界。add_generation_prompt = True 生成的提示以「助手回合开头 + 空的思考块」结尾，
# 这两段属于提示，不计损失；从目标文本开始到 <|im_end|> 为止计算损失，<|im_end|> 之后的换行不计。
ASSISTANT_MASK_SPEC = AssistantMaskSpec(
    assistant_header = "<|im_start|>assistant\n",
    assistant_end = "<|im_end|>",
    generation_stop = "<|im_end|>",
    excluded_assistant_prefixes = ("<think>\n\n</think>\n\n",),
)
# 训练样本的格式：chat 聊天格式（音频 + 固定指令 → 助手回答），cpt 纯续写格式（音频标记后直接接目标文本）。
PROMPT_FORMATS = ("chat", "cpt")
# 各格式中目标文本之后的结束标记，计算损失。cpt 用分词器的纯文本结束标记（tokenizer_config 中的 pad_token，
# generation_config 的 eos_token_id 248044）；chat 的助手回合以 <|im_end|> 结束。
END_TOKENS = {"chat": "<|im_end|>", "cpt": "<|endoftext|>"}


def build_messages(audio_path):
    """按 CONVENTIONS 的对话格式构造用户消息：一段音频加一句固定指令，不给语言提示。"""
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": str(audio_path)},
                {"type": "text", "text": INSTRUCTION},
            ],
        },
    ]


def audio_markup(processor):
    """一段音频的标记 <|audio_start|><|audio_pad|><|audio_end|>，与 chat template 中音频的写法相同。"""
    return processor.audio_start_token + processor.audio_token + processor.audio_end_token


def build_prompt(processor, audio_path, prompt_format = "chat"):
    """推理时的提示。chat：展开 chat template，结尾是助手回合开头和空的思考块 <think>\\n\\n</think>\\n\\n；
    cpt：只有音频标记。
    """
    if prompt_format == "cpt":
        return audio_markup(processor = processor)
    if prompt_format != "chat":
        raise ValueError(f"未知的样本格式：{prompt_format}，可选 {PROMPT_FORMATS}")
    return processor.apply_chat_template(
        conversation = build_messages(audio_path = audio_path),
        tokenize = False,
        add_generation_prompt = True,
        enable_thinking = False,
    )


def build_training_text(processor, audio_path, target, prompt_format = "chat"):
    """渲染带目标文本的完整样本，前半段与 build_prompt 的结果逐字相同。

    chat：带助手回答的完整对话，后面接目标文本、<|im_end|> 和换行。cpt：音频标记、目标文本、<|endoftext|>。
    """
    if prompt_format == "cpt":
        return audio_markup(processor = processor) + target + END_TOKENS["cpt"]
    if prompt_format != "chat":
        raise ValueError(f"未知的样本格式：{prompt_format}，可选 {PROMPT_FORMATS}")
    conversation = build_messages(audio_path = audio_path) + [
        {"role": "assistant", "content": [{"type": "text", "text": target}]},
    ]
    return processor.apply_chat_template(
        conversation = conversation,
        tokenize = False,
        add_generation_prompt = False,
        enable_thinking = False,
    )


def build_cpt_labels(input_ids, attention_mask, audio_end_id, end_id):
    """cpt 格式的标签：每行唯一的 <|audio_end|> 之后的有效 token（目标文本和结尾的 <|endoftext|>）计算损失。

    只看最终的 input_ids（processor 展开音频标记并分词之后）和 attention_mask，与填充在哪一侧无关；填充 token
    与 <|endoftext|> 是同一个 id，靠 attention_mask 区分。每行必须恰好有一个 <|audio_end|>，最后一个有效 token
    必须是结束标记，否则报错。
    """
    valid = attention_mask.bool()
    ends = (input_ids == audio_end_id) & valid
    counts = ends.sum(dim = -1)
    if bool((counts != 1).any()):
        raise ValueError(f"cpt 样本每行应恰好有一个音频结束标记，实际为 {counts.tolist()}")
    # 某个位置之前（不含该位置）出现过 <|audio_end|>，即在它之后。
    after = (ends.long().cumsum(dim = -1) - ends.long()) > 0
    supervised = after & valid
    lengths = valid.long().sum(dim = -1)
    last = valid.long().cumsum(dim = -1).eq(lengths[:, None]) & valid
    if bool((input_ids[last] != end_id).any()) or bool((supervised.sum(dim = -1) == 0).any()):
        raise ValueError("cpt 样本的最后一个有效 token 应为结束标记，且在音频结束标记之后")
    labels = torch.full_like(input = input_ids, fill_value = IGNORE_INDEX)
    labels[supervised] = input_ids[supervised]
    return labels


def to_simplified(text):
    """把中文繁体转成简体，与朗读稿使用同一个词级转换（pipeline.tn.core，OpenCC），保证两种训练目标的字形一致。"""
    from pipeline.tn.core import to_simplified as tn_to_simplified

    return tn_to_simplified(text = text)


def target_text(row, target):
    """取一条样本的训练目标：script 用朗读稿，original 用原文（中文转为简体）。"""
    if target == "script":
        text = row["script"]
    elif target == "original":
        text = row["original"]
        if row["lang"] == "zh":
            text = to_simplified(text = text)
    else:
        raise ValueError(f"未知的训练目标：{target}，可选 {TARGETS}")
    return text.strip()


def training_targets(rows, target):
    """返回 (各行的训练目标, 各行是否为空目标)。TranscriptionDataset 和分片打包（按训练计划排列音频）共用。"""
    targets = [target_text(row = row, target = target) for row in rows]
    return targets, [text == "" for text in targets]


def read_manifest(
    path,
    audio_root = None,
    langs = None,
    max_duration_s = None,
):
    """读取训练清单（JSONL）。

    相对的 audio_path 以 audio_root 为基准解析，未给出时以清单所在目录为基准。
    langs 和 max_duration_s 用来筛选样本。
    """
    path = Path(path)
    root = Path(audio_root) if audio_root is not None else path.parent
    rows = []
    with open(file = path, mode = "r", encoding = "utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(s = line)
            if langs is not None and row["lang"] not in langs:
                continue
            if max_duration_s is not None and row["duration_s"] > max_duration_s:
                continue
            audio_path = Path(row["audio_path"])
            if not audio_path.is_absolute():
                audio_path = root / audio_path
            row["audio_path"] = str(audio_path)
            rows.append(row)
    return rows


def load_audio(
    path,
    start_s = None,
    stop_s = None,
    data = None,
):
    """读取音频，统一为 16 kHz 单声道 float32 波形。给出 start_s、stop_s 时只解码这一段（秒）。

    data 给出时（音频文件的原始字节，例如从分片中读出）从这些字节解码，这时 path 只用来标识这段音频。
    """
    from torchcodec.decoders import AudioDecoder

    decoder = AudioDecoder(
        source = str(path) if data is None else data,
        sample_rate = SAMPLING_RATE,
        num_channels = 1,
    )
    if start_s is None and stop_s is None:
        samples = decoder.get_all_samples()
    else:
        samples = decoder.get_samples_played_in_range(
            start_seconds = start_s or 0.0,
            stop_seconds = stop_s,
        )
    return samples.data.squeeze(dim = 0).numpy()


class PackedRows:
    """把清单行中读取样本要用的几项编码进一个文件，用内存映射读取。

    DataLoader 的每个读取进程都会收到一份数据集对象。清单行原本是 Python 字典，489 万行（v2b）时每个读取进程
    约 13 GB，6 个进程合计近 80 GB。这里把每行的样本编号、语言、音频路径、时长、起点、分片位置和训练目标写成
    一行 JSON，依次存进 path，各行的起止位置存进 path 旁的 .offsets.npy；传给读取进程的只有这两个文件的路径，
    各进程用 mmap 打开同一个文件，共享操作系统的页缓存，用到哪一行才解码哪一行。取出的行与原来的字典在这几项上
    相同（分片位置 audio_shard 取出时是列表 [分片编号, 起点, 长度]，不从分片读取时为 None）。
    """

    FIELDS = ("sample_id", "lang", "audio_path", "duration_s", "offset_s", "audio_shard")

    def __init__(self, rows, targets, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents = True, exist_ok = True)
        offsets = np.zeros(shape = (len(rows) + 1,), dtype = np.int64)
        tmp = self.path.with_name(self.path.name + ".tmp")
        position = 0
        with open(file = tmp, mode = "wb") as handle:
            for index, (row, target) in enumerate(zip(rows, targets)):
                record = [row.get(field) for field in self.FIELDS] + [target]
                data = json.dumps(obj = record, ensure_ascii = False).encode(encoding = "utf-8")
                handle.write(data)
                position += len(data)
                offsets[index + 1] = position
        np.save(file = self.offsets_path, arr = offsets)
        tmp.replace(self.path)
        self._open()

    @property
    def offsets_path(self):
        return self.path.with_name(self.path.name + ".offsets.npy")

    def _open(self):
        self._offsets = np.load(file = self.offsets_path, mmap_mode = "r")
        self._handle = open(file = self.path, mode = "rb")
        size = self.path.stat().st_size
        self._map = mmap.mmap(self._handle.fileno(), 0, access = mmap.ACCESS_READ) if size else b""

    def __getstate__(self):
        # 只传文件路径，读取进程自己打开内存映射
        return {"path": str(self.path)}

    def __setstate__(self, state):
        self.path = Path(state["path"])
        self._open()

    def __len__(self):
        return len(self._offsets) - 1

    def record(self, index):
        """返回 (行字典, 训练目标)。"""
        start, end = int(self._offsets[index]), int(self._offsets[index + 1])
        values = json.loads(s = self._map[start:end])
        return dict(zip(self.FIELDS, values[:-1])), values[-1]

    def __getitem__(self, index):
        return self.record(index = index)[0]


class PackedTargets:
    """PackedRows 中训练目标的只读视图，按下标取第几条样本的目标。"""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows.record(index = index)[1]


class TranscriptionDataset(torch.utils.data.Dataset):
    """每条样本给出波形和目标文本；特征提取和分词放在批次构造中统一完成。

    下标可以是整数（不混音），也可以是 sampling.SampleKey(index, epoch, position)：给出 mixer 时，
    按 (seed, epoch, position) 生成随机数，由 mixer 决定这条语音样本是否混音以及怎样混音。
    """

    def __init__(
        self,
        rows,
        target,
        mixer = None,
        seed = 0,
        storage = None,
        max_chars_per_second = None,
        shards = None,
    ):
        """storage 给出文件路径时，清单行改存为 PackedRows（大清单用，避免每个读取进程复制全部字典）。

        shards 是 shards.ShardReader 时，按清单行的 audio_shard 从分片读出音频字节再解码（shards.attach_shards
        已给每行加上 audio_shard）。

        max_chars_per_second 给出时，训练目标的字数除以音频时长超过它的样本读取时跳过（与读不出的音频同样处理，
        组批计划不变）。这样的目标说不完：对齐错位、录音被截短或 ASR 重复输出，最长的一条 6.8 秒配 5,477 字，
        会让一批的序列长到超出显存。
        """
        self.max_chars_per_second = max_chars_per_second
        self.target = target
        targets, self.empty = training_targets(rows = rows, target = target)
        if storage is None:
            self.rows = rows
            self.targets = targets
        else:
            self.rows = PackedRows(rows = rows, targets = targets, path = storage)
            self.targets = PackedTargets(rows = self.rows)
        self.mixer = mixer
        self.seed = seed
        self.shards = shards

    def __len__(self):
        return len(self.rows)

    def read_waveform(self, row):
        """读取一行清单的音频；有 offset_s 的虚拟切分片段只解码整段文件中的这一段。"""
        data = None
        if self.shards is not None:
            data = self.shards.read(location = row.get("audio_shard"))
        if row.get("offset_s") is not None:
            return load_audio(
                path = row["audio_path"],
                start_s = row["offset_s"],
                stop_s = row["offset_s"] + row["duration_s"],
                data = data,
            )
        return load_audio(path = row["audio_path"], data = data)

    def __getitem__(self, key):
        keyed = isinstance(key, SampleKey)
        index = key.index if keyed else int(key)
        if isinstance(self.rows, PackedRows):
            row, target = self.rows.record(index = index)
        else:
            row, target = self.rows[index], self.targets[index]
        if self.max_chars_per_second is not None and len(target) > self.max_chars_per_second * float(row["duration_s"]):
            rate = len(target) / max(float(row["duration_s"]), 1e-3)
            print(f"[bad-target] {row['sample_id']}: {len(target)} 字 / {row['duration_s']} 秒 = {rate:.0f} 字每秒", file = sys.stderr, flush = True)
            return None
        try:
            waveform = self.read_waveform(row = row)
        except Exception as error:
            # 几百万个文件里个别损坏（例如转换时机器意外关机）：记下路径，这条样本交给 TrainCollator 丢掉。
            # 同一个文件每次都读不出，续训与不中断的运行仍然一致。
            print(f"[bad-audio] {row['audio_path']}: {type(error).__name__}: {str(error)[:200]}", file = sys.stderr, flush = True)
            return None
        mix = None
        if self.mixer is not None and keyed and not self.empty[index]:
            rng = np.random.default_rng(seed = [self.seed, key.epoch, STREAM_MIXING, key.position])
            waveform, mix = self.mixer(waveform = waveform, rng = rng)
        return {
            "sample_id": row["sample_id"],
            "lang": row["lang"],
            "audio_path": row["audio_path"],
            "duration_s": float(row["duration_s"]),
            "waveform": waveform,
            "target": target,
            "empty": self.empty[index],
            "mix": mix,
        }


def batch_metadata(batch, audio_token_id, n_window, n_window_infer):
    """在 CPU 上预先算出前向要用的位置与索引，训练时不必在 GPU 上逐条样本计算（任务 T59，--precompute-metadata）。

    原实现在前向中逐条样本计算这些量，每条都要把 GPU 上的数取回 CPU，CPU 因此反复等 GPU，批越大越慢。
    这里调用模型代码中同样的函数（音频塔的分块、窗口边界），或按同样的规则计算（位置编号、计损失的位置），
    结果与原实现逐位相同：

    - position_ids：语言模型的位置编号 (3, 批大小, 长度)。输入中只有文字和音频，三路位置相同，都是有效 token
      在本条中的序号，填充位置为 0（与 get_rope_index 对音频、文字输入的结果相同）。
    - audio_mask：input_ids 中音频标记的位置。
    - audio_padded_feature、audio_chunk_lengths、audio_valid_indices、audio_cu_seqlens、audio_max_seqlen：
      音频塔把梅尔特征切成 100 帧的块、卷积后按窗口做注意力，这些是 transformers 的 Qwen3OmniMoeAudioEncoder
      本来在前向开头算出的分块与窗口信息；它的 forward 接受预先算好的同名参数。
    - loss_index、loss_targets、loss_rows：计损失的位置在 (批大小 × 长度) 中的平展下标（第 t 个位置预测
      第 t + 1 个 token）、目标 token 和所在的样本行号。
    """
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        chunk_and_pad_features,
        get_audio_cu_seqlens,
        get_valid_indices,
    )

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    positions = (attention_mask.long().cumsum(dim = -1) - 1).clamp(min = 0) * attention_mask.long()
    feature_mask = batch["feature_attention_mask"].bool()
    feature_lens = feature_mask.sum(dim = -1)
    packed = batch["input_features"].transpose(dim0 = 1, dim1 = 2)[feature_mask].transpose(dim0 = 0, dim1 = 1)
    padded_feature, chunk_lengths = chunk_and_pad_features(
        input_features = packed,
        feature_lens = feature_lens,
        n_window = n_window,
    )
    cu_seqlens = get_audio_cu_seqlens(
        chunk_lengths = chunk_lengths,
        feature_lens = feature_lens,
        n_window_infer = n_window_infer,
        n_window = n_window,
    )
    labels = batch["labels"]
    length = labels.shape[1]
    shifted = labels[:, 1:]
    rows, columns = (shifted != IGNORE_INDEX).nonzero(as_tuple = True)
    return {
        "position_ids": positions[None].expand(size = (3, -1, -1)).contiguous(),
        "audio_mask": input_ids == audio_token_id,
        "audio_padded_feature": padded_feature,
        "audio_chunk_lengths": chunk_lengths,
        "audio_valid_indices": get_valid_indices(chunk_lengths = chunk_lengths, n_window = n_window),
        "audio_cu_seqlens": cu_seqlens,
        "audio_max_seqlen": int((cu_seqlens[1:] - cu_seqlens[:-1]).max()),
        "loss_index": rows * length + columns,
        "loss_targets": shifted[rows, columns],
        "loss_rows": rows,
    }


def pack_batch(batch):
    """把一批样本首尾相接拼成一行，去掉填充（任务 #66，--pack-sequences）。batch 是 batch_metadata 补充过的批次。

    样本、顺序和音频都不变，只是语言模型一侧不再算填充位置：v2b 的批次中填充约占语言模型 token 的 9%。做法与
    transformers 的 DataCollatorWithFlattening(return_flash_attn_kwargs = True, return_seq_idx = True) 相同：

    - input_ids、mm_token_type_ids、labels、audio_mask 只保留各条的有效 token，按样本顺序接成形状 (1, 总长) 的一行；
      attention_mask 全为 1。每条样本第一个 token 的标签改为 -100（本来就是提示，不计损失），于是没有位置去预测
      下一条样本的开头。
    - position_ids：每条样本从 0 重新编号，形状 (3, 1, 总长)。输入中只有文字和音频，MRoPE 的三路位置相同。
    - cu_seq_lens_q：各条样本在这一行中的起点，最后一项是总长度（int32，FlashAttention 和 fla 的 cu_seqlens）；
      max_length_q：最长一条的长度（Python 整数）；seq_idx：每个 token 属于第几条样本，形状 (1, 总长)（int32，
      causal_conv1d 的 seq_idx）。
    - loss_index、loss_targets：计损失的位置在这一行中的下标和目标 token；loss_rows 仍是所在样本的序号。各条样本
      计损失的 token 与填充的批次一一对应、顺序相同。

    音频塔的分块与窗口信息（audio_*）与填充无关，保持不变。
    """
    mask = batch["attention_mask"].bool()
    lengths = mask.sum(dim = -1)
    count = lengths.shape[0]
    total = int(lengths.sum())
    starts = torch.zeros(size = (count + 1,), dtype = torch.int32)
    starts[1:] = lengths.cumsum(dim = 0)
    seq_idx = torch.repeat_interleave(torch.arange(count, dtype = torch.int32), repeats = lengths)
    positions = torch.arange(total) - starts[:-1].long()[seq_idx.long()]
    labels = batch["labels"][mask].clone()
    labels[starts[:-1].long()] = IGNORE_INDEX
    shifted = labels[1:]
    columns = (shifted != IGNORE_INDEX).nonzero(as_tuple = True)[0]
    batch.update({
        "input_ids": batch["input_ids"][mask][None],
        "attention_mask": torch.ones(size = (1, total), dtype = batch["attention_mask"].dtype),
        "mm_token_type_ids": batch["mm_token_type_ids"][mask][None],
        "labels": labels[None],
        "position_ids": positions[None, None].expand(size = (3, -1, -1)).contiguous(),
        "audio_mask": batch["audio_mask"][mask][None],
        "loss_index": columns,
        "loss_targets": shifted[columns],
        "loss_rows": seq_idx[columns].long(),
        "cu_seq_lens_q": starts,
        "max_length_q": int(lengths.max()),
        "seq_idx": seq_idx[None],
    })
    return batch


class TrainCollator:
    """把若干条样本拼成一个批次。

    每条样本渲染成带助手回答的完整对话，整批交给 processor 做特征提取、音频标记展开和分词（与评测推理
    相同的调用），再在最终的 input_ids 上用 mm_assistant_mask 生成标签。prompt_format = "cpt" 时样本是音频标记
    后直接接目标文本和 <|endoftext|>，标签由 build_cpt_labels 在最终的 input_ids 上生成。给出 audio_windows（音频塔的
    n_window、n_window_infer）时，再用 batch_metadata 预先算出前向要用的位置与索引；pack 为真时（需要
    audio_windows）再用 pack_batch 把整批拼成一行、去掉填充。
    """

    def __init__(
        self,
        processor,
        audio_windows = None,
        pack = False,
        prompt_format = "chat",
    ):
        if pack and audio_windows is None:
            raise ValueError("拼接批次（pack）需要预先算好的元数据（audio_windows，即 --precompute-metadata）")
        if prompt_format not in PROMPT_FORMATS:
            raise ValueError(f"未知的样本格式：{prompt_format}，可选 {PROMPT_FORMATS}")
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.audio_windows = audio_windows
        self.pack = pack
        self.prompt_format = prompt_format

    def __call__(self, items):
        # 读不出音频的样本（TranscriptionDataset 返回 None）不进批次，这一批少几条照常训练
        items = [item for item in items if item is not None]
        if not items:
            raise RuntimeError("这一批的音频全部读不出，检查日志中的 [bad-audio] 路径")
        texts = [
            build_training_text(
                processor = self.processor,
                audio_path = item["audio_path"],
                target = item["target"],
                prompt_format = self.prompt_format,
            )
            for item in items
        ]
        inputs = self.processor(
            text = texts,
            audio = [item["waveform"] for item in items],
            sampling_rate = SAMPLING_RATE,
            padding = True,
            return_tensors = "pt",
        )
        if self.prompt_format == "cpt":
            labels = build_cpt_labels(
                input_ids = inputs["input_ids"],
                attention_mask = inputs["attention_mask"],
                audio_end_id = self.tokenizer.convert_tokens_to_ids(tokens = self.processor.audio_end_token),
                end_id = self.tokenizer.convert_tokens_to_ids(tokens = END_TOKENS["cpt"]),
            )
        else:
            labels = build_assistant_labels(
                batch = inputs,
                tokenizer = self.tokenizer,
                spec = ASSISTANT_MASK_SPEC,
            )
        batch = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "mm_token_type_ids": inputs["mm_token_type_ids"],
            "labels": labels,
            "input_features": inputs["input_features"],
            "feature_attention_mask": inputs["feature_attention_mask"],
            "sample_ids": [item["sample_id"] for item in items],
            "langs": [item["lang"] for item in items],
            "audio_seconds": sum(item["duration_s"] for item in items),
            # 本批的空目标样本数和混音样本数，训练日志按它们统计实际比例。
            "n_empty": sum(bool(item.get("empty")) for item in items),
            "empty_rows": torch.tensor(data = [bool(item.get("empty")) for item in items], dtype = torch.bool),
            "n_mixed": sum(item.get("mix") is not None for item in items),
        }
        if self.audio_windows is not None:
            n_window, n_window_infer = self.audio_windows
            batch.update(batch_metadata(
                batch = batch,
                audio_token_id = self.tokenizer.convert_tokens_to_ids(tokens = self.processor.audio_token),
                n_window = n_window,
                n_window_infer = n_window_infer,
            ))
            if self.pack:
                batch = pack_batch(batch = batch)
        return batch


def length_grouped_batches(
    durations,
    batch_size,
    seed,
    epoch,
    group_batches = 64,
    max_batch_seconds = None,
):
    """按时长分组成批，返回这一轮的批次列表（每个批次是样本下标列表）。

    每轮先按 (seed, epoch) 打乱全部样本，每 group_batches 个批次的样本按时长排序后依次装批，
    最后再打乱批次顺序。同一批内时长相近，填充少；批次之间仍是随机顺序。
    一批最多 batch_size 条；给出 max_batch_seconds 时，还要求「条数 × 批内最长时长」（填充后的
    音频总秒数，大致正比于显存）不超过它，长音频的批次因此条数更少。
    结果只由参数决定，断点续训时可以重新生成同一轮的批次再跳过已训练的部分。
    """
    generator = random.Random(x = f"{seed}-{epoch}")
    indices = list(range(len(durations)))
    generator.shuffle(x = indices)
    group_size = batch_size * group_batches
    batches = []
    for start in range(0, len(indices), group_size):
        group = sorted(indices[start:start + group_size], key = lambda index: durations[index])
        batch = []
        for index in group:
            # 组内按时长升序，新加入的样本就是批内最长的。
            too_long = (
                max_batch_seconds is not None
                and (len(batch) + 1) * durations[index] > max_batch_seconds
            )
            if batch and (len(batch) == batch_size or too_long):
                batches.append(batch)
                batch = []
            batch.append(index)
        if batch:
            batches.append(batch)
    generator.shuffle(x = batches)
    return batches


def summarize_rows(rows):
    """按语言统计样本数和时长，打印清单概况用。"""
    summary = {}
    for row in rows:
        entry = summary.setdefault(row["lang"], {"samples": 0, "hours": 0.0})
        entry["samples"] += 1
        entry["hours"] += row["duration_s"] / 3600
    return {
        lang: {"samples": entry["samples"], "hours": round(number = entry["hours"], ndigits = 3)}
        for lang, entry in sorted(summary.items())
    }
