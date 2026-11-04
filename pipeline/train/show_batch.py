"""打印一两条训练样本的标记序列与标签掩码，检查音频标记数量和损失位置。

用法：

    python -m pipeline.train.show_batch \\
        --model-dir <阶段 1 模型目录> --manifest <训练清单.jsonl> --target original --count 2 [--run-model]

对每条样本打印：特征帧数、按音频塔规则算出的音频特征数、序列中 <|audio_pad|> 的个数；连续的
<|audio_pad|> 合并成一项显示，其余 token 逐个列出，带 [loss] 的是计算损失的位置。
加 --run-model 时在 GPU 上跑一次音频塔和 projector，核对实际输出的特征数，并计算一次损失。
"""

import argparse
import json

import torch
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    _get_feat_extract_output_lengths,
)

from pipeline.train.data import IGNORE_INDEX, TrainCollator, TranscriptionDataset, read_manifest


def describe_tokens(tokenizer, input_ids, labels, audio_token_id):
    """把一条序列写成可读的形式：连续音频标记合并，计损失的 token 前加 [loss]。"""
    parts = []
    index = 0
    length = len(input_ids)
    while index < length:
        token_id = int(input_ids[index])
        if token_id == audio_token_id:
            end = index
            while end < length and int(input_ids[end]) == audio_token_id:
                end += 1
            parts.append(f"<|audio_pad|>×{end - index}")
            index = end
            continue
        text = tokenizer.convert_ids_to_tokens(ids = token_id)
        text = tokenizer.convert_tokens_to_string(tokens = [text])
        marker = "[loss]" if int(labels[index]) != IGNORE_INDEX else ""
        parts.append(f"{marker}{json.dumps(obj = text, ensure_ascii = False)}")
        index += 1
    return " ".join(parts)


def main(argv = None):
    parser = argparse.ArgumentParser(description = "检查音频标记数与标签掩码")
    parser.add_argument("--model-dir", required = True)
    parser.add_argument("--manifest", required = True)
    parser.add_argument("--audio-root", default = None)
    parser.add_argument("--target", default = "script", choices = ("script", "original"))
    parser.add_argument("--count", type = int, default = 2)
    parser.add_argument("--sample-ids", default = None, help = "指定样本，逗号分隔")
    parser.add_argument("--run-model", action = "store_true")
    args = parser.parse_args(args = argv)

    rows = read_manifest(path = args.manifest, audio_root = args.audio_root)
    if args.sample_ids:
        wanted = args.sample_ids.split(sep = ",")
        rows = [row for row in rows if row["sample_id"] in wanted]
    rows = rows[:args.count]

    if args.run_model:
        from pipeline.train.model import load_model

        model, processor = load_model(model_dir = args.model_dir)
    else:
        from transformers import AutoProcessor

        model = None
        processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path = args.model_dir,
            trust_remote_code = True,
        )
    tokenizer = processor.tokenizer
    audio_token_id = tokenizer.convert_tokens_to_ids(tokens = processor.audio_token)
    dataset = TranscriptionDataset(rows = rows, target = args.target)
    collator = TrainCollator(processor = processor)
    batch = collator(items = [dataset[index] for index in range(len(dataset))])

    feature_lens = batch["feature_attention_mask"].sum(dim = -1)
    expected_audio = _get_feat_extract_output_lengths(
        input_lengths = feature_lens,
        n_window = processor.audio_n_window,
    )
    print(f"批次形状：input_ids {tuple(batch['input_ids'].shape)}，input_features {tuple(batch['input_features'].shape)}")
    for row_index, sample_id in enumerate(batch["sample_ids"]):
        input_ids = batch["input_ids"][row_index]
        labels = batch["labels"][row_index]
        valid = batch["attention_mask"][row_index].bool()
        audio_count = int((input_ids == audio_token_id).sum())
        loss_ids = labels[labels != IGNORE_INDEX].tolist()
        first_loss = int((labels != IGNORE_INDEX).nonzero()[0])
        print(f"\n== {sample_id}（{batch['langs'][row_index]}，{rows[row_index]['duration_s']} 秒）")
        print(f"特征帧数 {int(feature_lens[row_index])}，音频塔规则给出的特征数 {int(expected_audio[row_index])}，"
              f"序列中的 <|audio_pad|> {audio_count} 个，有效长度 {int(valid.sum())}，填充 {int((~valid).sum())}")
        print(f"目标文本：{dataset.targets[row_index]}")
        print(f"计损失的 token {len(loss_ids)} 个，解码为：{json.dumps(obj = tokenizer.decode(token_ids = loss_ids), ensure_ascii = False)}")
        before = tokenizer.decode(token_ids = input_ids[first_loss - 6:first_loss].tolist())
        print(f"第一个计损失的 token 之前的 6 个 token：{json.dumps(obj = before, ensure_ascii = False)}")
        print("序列：" + describe_tokens(
            tokenizer = tokenizer,
            input_ids = input_ids[valid],
            labels = labels[valid],
            audio_token_id = audio_token_id,
        ))

    if model is not None:
        from pipeline.train.model import move_batch, target_loss

        inputs = move_batch(batch = batch, device = "cuda", feature_dtype = model.model.audio_tower.dtype)
        with torch.no_grad():
            audio_features = model.model.get_audio_features(
                input_features = inputs["input_features"],
                feature_attention_mask = inputs["feature_attention_mask"],
            )
            loss_sum, n_tokens, n_correct = target_loss(model = model, inputs = inputs)
        print(f"\n音频塔与 projector 输出 {tuple(audio_features.shape)}，"
              f"序列中音频标记共 {int((batch['input_ids'] == audio_token_id).sum())} 个")
        print(f"未训练时目标文本上的平均损失 {float(loss_sum) / int(n_tokens):.3f}，计损失 token {int(n_tokens)} 个，"
              f"标签中非 -100 的位置 {int((batch['labels'] != IGNORE_INDEX).sum())} 个")


if __name__ == "__main__":
    main()
