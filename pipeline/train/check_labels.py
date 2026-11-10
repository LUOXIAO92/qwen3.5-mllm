"""核对训练批次的标签（实验 v1，任务 T10；空目标样本，实验 v2，任务 T1）。

对训练清单中每种语言抽出的样本，用 TrainCollator 构造批次，逐条检查两件事：

1. 计算损失的 token 解码后，正好等于目标文本加 <|im_end|>，没有混入提示、思考块或换行。
   空目标样本（训练目标为空字符串，清单中语言为 zxx 的无人声片段）只有一个计算损失的 token，就是 <|im_end|>。
2. 第一个计算损失的 token 之前的序列，与评测推理时的提示（build_prompt，add_generation_prompt = True）
   经同一个 processor 展开后的 token 逐个相同，保证训练与推理看到的输入一致。

最后打印一条空目标样本助手回合附近的 token 和标签，带 [loss] 的是计算损失的位置。

更换 chat template、processor 或 mm_assistant_mask 版本后运行一次。用法（在仓库根目录运行）::

    python -m pipeline.train.check_labels --model-dir qwen3.5-0.8b-mllm --manifest <训练清单> --target original

全部通过时退出码为 0，否则打印出错的样本并以 1 退出。
"""

import argparse
import random
import sys
from pathlib import Path

from pipeline.train.data import (
    IGNORE_INDEX,
    SAMPLING_RATE,
    TARGETS,
    TrainCollator,
    TranscriptionDataset,
    build_prompt,
    read_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description = "核对训练批次的标签")
    parser.add_argument("--model-dir", required = True, type = Path)
    parser.add_argument("--manifest", required = True, type = Path)
    parser.add_argument("--target", choices = TARGETS, default = "original")
    parser.add_argument("--per-lang", type = int, default = 8, help = "每种语言抽多少条")
    parser.add_argument("--batch-size", type = int, default = 8)
    parser.add_argument("--seed", type = int, default = 0)
    args = parser.parse_args()

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path = str(args.model_dir),
        trust_remote_code = True,
    )
    rows = read_manifest(path = args.manifest)
    by_lang = {}
    for row in rows:
        by_lang.setdefault(row["lang"], []).append(row)
    rng = random.Random(x = args.seed)
    picked = []
    for lang in sorted(by_lang):
        picked += rng.sample(population = by_lang[lang], k = min(args.per_lang, len(by_lang[lang])))

    dataset = TranscriptionDataset(rows = picked, target = args.target)
    collator = TrainCollator(processor = processor)
    im_end_id = processor.tokenizer.convert_tokens_to_ids(tokens = "<|im_end|>")
    failures = 0
    empty_checked = 0
    empty_example = None
    for start in range(0, len(dataset), args.batch_size):
        items = [dataset[index] for index in range(start, min(start + args.batch_size, len(dataset)))]
        batch = collator(items)
        for row_index, item in enumerate(items):
            input_ids = batch["input_ids"][row_index]
            labels = batch["labels"][row_index]
            supervised = (labels != IGNORE_INDEX).nonzero().view(-1)
            decoded = processor.tokenizer.decode(input_ids[supervised].tolist(), skip_special_tokens = False)
            expected = item["target"] + "<|im_end|>"
            prompt = processor(
                text = [build_prompt(processor = processor, audio_path = item["audio_path"])],
                audio = [item["waveform"]],
                sampling_rate = SAMPLING_RATE,
                return_tensors = "pt",
            )["input_ids"][0]
            first = int(supervised[0]) if len(supervised) else 0
            prefix_ok = input_ids[:first].tolist() == prompt.tolist()
            empty_ok = True
            if item["target"] == "":
                empty_checked += 1
                empty_ok = input_ids[supervised].tolist() == [im_end_id]
                if empty_example is None:
                    empty_example = (item["sample_id"], input_ids, labels, batch["attention_mask"][row_index])
            if decoded != expected or not prefix_ok or not empty_ok:
                failures += 1
                print(f"不一致 {item['sample_id']}：计损失的内容 {decoded!r}，应为 {expected!r}，提示部分一致：{prefix_ok}")
    if empty_example is not None:
        sample_id, input_ids, labels, attention_mask = empty_example
        valid = attention_mask.bool()
        input_ids, labels = input_ids[valid], labels[valid]
        tail = []
        for position in range(len(input_ids) - 12, len(input_ids)):
            text = processor.tokenizer.decode(token_ids = [int(input_ids[position])], skip_special_tokens = False)
            marker = "[loss]" if int(labels[position]) != IGNORE_INDEX else ""
            tail.append(f"{marker}{text!r}")
        print(f"空目标样本 {sample_id} 序列末尾 12 个 token：" + " ".join(tail))
    print(f"核对 {len(dataset)} 条（{len(by_lang)} 种语言，训练目标 {args.target}），"
          f"其中空目标 {empty_checked} 条，不一致 {failures} 条")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
