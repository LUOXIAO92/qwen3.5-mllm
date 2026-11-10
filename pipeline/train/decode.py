"""用训练好的 projector 对清单做贪心解码，统计与训练目标逐字一致的比例。

用法：

    python -m pipeline.train.decode \\
        --model-dir <阶段 1 模型目录> --projector <检查点目录或 projector.safetensors> \\
        --manifest <清单.jsonl> --target original --output <结果.jsonl>

输入与训练相同：同一个对话格式，冻结部分 bfloat16、projector float32。生成从空思考块之后开始，
遇到 <|im_end|> 停止。不给 --projector 时使用模型目录中自带的 projector。
这个脚本用于检查训练是否生效（例如过拟合测试）；正式评测用 pipeline/eval。
"""

import argparse
import json
import time

import torch

from pipeline.train.checkpoint import read_projector
from pipeline.train.data import SAMPLING_RATE, TranscriptionDataset, build_prompt, read_manifest
from pipeline.train.model import load_model, load_projector


def main(argv = None):
    parser = argparse.ArgumentParser(description = "贪心解码并与训练目标比较")
    parser.add_argument("--model-dir", required = True)
    parser.add_argument("--projector", default = None)
    parser.add_argument("--manifest", required = True)
    parser.add_argument("--audio-root", default = None)
    parser.add_argument("--target", required = True, choices = ("script", "original"))
    parser.add_argument("--output", default = None)
    parser.add_argument("--batch-size", type = int, default = 8)
    parser.add_argument("--max-new-tokens", type = int, default = 256)
    args = parser.parse_args(args = argv)

    model, processor = load_model(model_dir = args.model_dir)
    if args.projector:
        load_projector(model = model, state = read_projector(path = args.projector))
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    stop_ids = [
        tokenizer.convert_tokens_to_ids(tokens = "<|im_end|>"),
        tokenizer.convert_tokens_to_ids(tokens = "<|endoftext|>"),
    ]

    rows = read_manifest(path = args.manifest, audio_root = args.audio_root)
    dataset = TranscriptionDataset(rows = rows, target = args.target)
    results = []
    start = time.perf_counter()
    for begin in range(0, len(dataset), args.batch_size):
        items = [dataset[index] for index in range(begin, min(begin + args.batch_size, len(dataset)))]
        inputs = processor(
            text = [build_prompt(processor = processor, audio_path = item["audio_path"]) for item in items],
            audio = [item["waveform"] for item in items],
            sampling_rate = SAMPLING_RATE,
            padding = True,
            return_tensors = "pt",
        ).to(device = "cuda")
        inputs["input_features"] = inputs["input_features"].to(dtype = model.model.audio_tower.dtype)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens = args.max_new_tokens,
                do_sample = False,
                eos_token_id = stop_ids,
                pad_token_id = tokenizer.pad_token_id,
            )
        new_ids = generated[:, inputs["input_ids"].shape[1]:]
        hypotheses = tokenizer.batch_decode(sequences = new_ids, skip_special_tokens = True)
        for item, hypothesis in zip(items, hypotheses):
            hypothesis = hypothesis.strip()
            # 分词器会对文本做 NFC 规范化，训练时模型实际看到的目标是分词再解码后的文本。
            target_ids = tokenizer(text = item["target"], add_special_tokens = False)["input_ids"]
            tokenized_target = tokenizer.decode(token_ids = target_ids)
            results.append({
                "sample_id": item["sample_id"],
                "lang": item["lang"],
                "target": item["target"],
                "hypothesis": hypothesis,
                "exact": hypothesis == item["target"],
                "exact_tokenized": hypothesis == tokenized_target,
            })
    elapsed = time.perf_counter() - start

    exact = sum(result["exact"] for result in results)
    exact_tokenized = sum(result["exact_tokenized"] for result in results)
    for result in results:
        if not result["exact"]:
            note = "（与分词后的目标一致）" if result["exact_tokenized"] else ""
            print(f"[不一致{note}] {result['sample_id']}\n  目标：{result['target']}\n  输出：{result['hypothesis']}")
    print(f"逐字一致 {exact}/{len(results)}（{exact / len(results):.1%}），"
          f"与分词后的目标一致 {exact_tokenized}/{len(results)}（{exact_tokenized / len(results):.1%}），"
          f"用时 {elapsed:.1f} 秒")
    # 空目标样本（无人声片段）中输出了非空文字的比例，即无人声误输出率；语音样本中输出为空的比例。
    empty = [result for result in results if result["target"] == ""]
    speech = [result for result in results if result["target"] != ""]
    if empty:
        wrong = sum(result["hypothesis"] != "" for result in empty)
        print(f"空目标 {len(empty)} 条，输出非空文字 {wrong} 条，无人声误输出率 {wrong / len(empty):.1%}")
    if speech:
        silent = sum(result["hypothesis"] == "" for result in speech)
        print(f"语音 {len(speech)} 条，输出为空 {silent} 条（{silent / len(speech):.1%}）")
    if args.output:
        with open(file = args.output, mode = "w", encoding = "utf-8") as handle:
            for result in results:
                handle.write(json.dumps(obj = result, ensure_ascii = False) + "\n")


if __name__ == "__main__":
    main()
