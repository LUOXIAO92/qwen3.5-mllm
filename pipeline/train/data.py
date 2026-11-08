"""训练清单读取、目标文本选择与批次构造。

一条样本是一段完整的对话：用户消息为音频加固定指令，助手回答为目标文本。整段对话用阶段 1 模型自带的
chat template 渲染，再交给 processor 做特征提取和音频标记展开，得到最终的 input_ids。标签由
mm_assistant_mask（https://github.com/LUOXIAO92/MultimodalAssistantMask）在最终 input_ids 上按 token
匹配助手回合生成：助手回合开头和空的思考块不计损失，目标文本和结尾的 <|im_end|> 计算损失，其余位置填 -100。
transformers 的 return_assistant_tokens_mask 按展开音频标记之前的字符位置计算，展开后会错位或全为 0，
所以不用它。模型内部再把标签左移一位，用第 t 个位置的输出预测第 t + 1 个 token。
"""

import json
import random
from pathlib import Path

import torch
from mm_assistant_mask import AssistantMaskSpec, build_assistant_labels

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


def build_prompt(processor, audio_path):
    """展开 chat template，结尾是助手回合开头和空的思考块 <think>\\n\\n</think>\\n\\n。"""
    return processor.apply_chat_template(
        conversation = build_messages(audio_path = audio_path),
        tokenize = False,
        add_generation_prompt = True,
        enable_thinking = False,
    )


def build_training_text(processor, audio_path, target):
    """渲染带助手回答的完整对话。前半段与 build_prompt 的结果逐字相同，后面接目标文本、<|im_end|> 和换行。"""
    conversation = build_messages(audio_path = audio_path) + [
        {"role": "assistant", "content": [{"type": "text", "text": target}]},
    ]
    return processor.apply_chat_template(
        conversation = conversation,
        tokenize = False,
        add_generation_prompt = False,
        enable_thinking = False,
    )


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


def load_audio(path):
    """读取音频，统一为 16 kHz 单声道 float32 波形。"""
    from torchcodec.decoders import AudioDecoder

    decoder = AudioDecoder(
        source = str(path),
        sample_rate = SAMPLING_RATE,
        num_channels = 1,
    )
    samples = decoder.get_all_samples()
    return samples.data.squeeze(dim = 0).numpy()


class TranscriptionDataset(torch.utils.data.Dataset):
    """每条样本给出波形和目标文本；特征提取和分词放在批次构造中统一完成。"""

    def __init__(self, rows, target):
        self.rows = rows
        self.target = target
        self.targets = [target_text(row = row, target = target) for row in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {
            "sample_id": row["sample_id"],
            "lang": row["lang"],
            "audio_path": row["audio_path"],
            "duration_s": float(row["duration_s"]),
            "waveform": load_audio(path = row["audio_path"]),
            "target": self.targets[index],
        }


class TrainCollator:
    """把若干条样本拼成一个批次。

    每条样本渲染成带助手回答的完整对话，整批交给 processor 做特征提取、音频标记展开和分词（与评测推理
    相同的调用），再在最终的 input_ids 上用 mm_assistant_mask 生成标签。
    """

    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer

    def __call__(self, items):
        texts = [
            build_training_text(
                processor = self.processor,
                audio_path = item["audio_path"],
                target = item["target"],
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
        labels = build_assistant_labels(
            batch = inputs,
            tokenizer = self.tokenizer,
            spec = ASSISTANT_MASK_SPEC,
        )
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "mm_token_type_ids": inputs["mm_token_type_ids"],
            "labels": labels,
            "input_features": inputs["input_features"],
            "feature_attention_mask": inputs["feature_attention_mask"],
            "sample_ids": [item["sample_id"] for item in items],
            "langs": [item["lang"] for item in items],
            "audio_seconds": sum(item["duration_s"] for item in items),
        }


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
