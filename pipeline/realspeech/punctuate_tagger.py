"""用逐词分类的小模型给没有标点的训练目标还原标点与大小写（在 envs/mllm-eval 中运行，需要 torch 与 transformers）。

模型为 Horizon-Labs/multilingual-punctuation-truecase（Apache-2.0，ModernBERT，88 种语言，含 fa、ms、tl、vi）：对每个词
预测一个「标点|大小写」标签，标点为不加、句号、逗号、问号、连字符、冒号之一，大小写为全小写、首字母大写、全大写之一。
FLEURS dev 上与 LLM（pipeline/realspeech/punctuate.py）相比，加的标点精确率相当或略高、召回率低，大小写还原相当，
速度快两个数量级，所以波斯语、马来语、菲律宾语改用它（用户 2026-11-11 决定）。

模型的输出与 LLM 输出一样交给 punctuate.transfer：只取标点和大小写，词的写法取自原文本；结果写入同一个
punctuation.jsonl（model 字段记模型名），manifest 步骤套用状态为 ok 或 aligned 的结果。连字符不放回（相当于连词）。
只认同一模型的缓存：其他方法（如 LLM）处理过的样本会重新处理，manifest 取最后一行，保证同一目录只用一种方法。

用法::

    python -m pipeline.realspeech.punctuate_tagger run --out-dir <输出目录> --langs fa --model <模型目录或仓库名>
    python -m pipeline.realspeech.punctuate_tagger eval --input fleurs_dev.jsonl --lang ms --model <...> --output eval.json
"""

import argparse
import json
import time
from pathlib import Path

from pipeline.realspeech.punctuate import (
    LANG_MARKS,
    RECASE,
    fold,
    load_cache,
    needs_punctuation,
    score,
    strip_marks,
    transfer,
)

DEFAULT_MODEL = "Horizon-Labs/multilingual-punctuation-truecase-base"
SKIPPED_MARKS = {"0", "-"}


def apply_labels(words: list, labels: list) -> str:
    """把每个词的「标点|大小写」标签作用到词上，返回带标点的文本；连字符和「不加」都不放标点。"""
    pieces = []
    for word, label in zip(words, labels):
        mark, case = label.split("|")
        if case == "Cap":
            word = word[:1].upper() + word[1:]
        elif case == "UPPER":
            word = word.upper()
        pieces.append(word + ("" if mark in SKIPPED_MARKS else mark))
    return " ".join(pieces)


class Tagger:
    def __init__(self, model: str, device: str, batch_size: int):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = model)
        self.model = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path = model)
        self.model = self.model.to(device = device).eval()
        self.device = device
        self.batch_size = batch_size
        self.id2label = self.model.config.id2label

    def restore(self, texts: list) -> list:
        """模型按小写训练：先转小写、按空白分词，每个词取第一个子词的标签。按长度排序组批，结果按原顺序返回。"""
        order = sorted(range(len(texts)), key = lambda index: len(texts[index]))
        outputs = [""] * len(texts)
        for start in range(0, len(order), self.batch_size):
            indices = order[start:start + self.batch_size]
            batch = [texts[index].lower().split() for index in indices]
            encoded = self.tokenizer(
                batch,
                is_split_into_words = True,
                return_tensors = "pt",
                padding = True,
                truncation = True,
                max_length = 512,
            )
            with self.torch.no_grad():
                predicted = self.model(**{key: value.to(device = self.device) for key, value in encoded.items()}).logits.argmax(dim = -1).cpu()
            for row, (index, words) in enumerate(zip(indices, batch)):
                first = {}
                for position, word in enumerate(encoded.word_ids(batch_index = row)):
                    if word is not None and word not in first:
                        first[word] = int(predicted[row, position])
                labels = [self.id2label[first.get(word, 0)] for word in range(len(words))]
                outputs[index] = apply_labels(words = words, labels = labels)
        return outputs


def run(args: argparse.Namespace) -> None:
    langs = set(args.langs.split(","))
    unknown = langs - set(LANG_MARKS)
    if unknown:
        raise SystemExit(f"不支持的语言：{sorted(unknown)}")
    cache_path = args.out_dir / "punctuation.jsonl"
    cache = load_cache(path = cache_path)
    todo = []
    with open(file = args.out_dir / "manifest.jsonl", mode = "r", encoding = "utf-8") as handle:
        for line in handle:
            row = json.loads(s = line)
            if row["lang"] not in langs:
                continue
            text = row.get("unpunctuated_target", row["original"])
            if not needs_punctuation(text = text, lang = row["lang"]):
                continue
            cached = cache.get(row["sample_id"])
            if cached and cached["input"] == text and cached.get("model") == args.model:
                continue
            todo.append((row["sample_id"], row["lang"], text))
    print(f"[punctuate_tagger] {len(todo)} 条待处理", flush = True)
    tagger = Tagger(model = args.model, device = args.device, batch_size = args.batch_size)
    counts = {}
    started = time.time()
    with open(file = cache_path, mode = "a", encoding = "utf-8") as out:
        for start in range(0, len(todo), args.chunk):
            chunk = todo[start:start + args.chunk]
            outputs = tagger.restore(texts = [text for _, _, text in chunk])
            for (key, lang, text), tagged in zip(chunk, outputs):
                result = transfer(source = text, llm_output = tagged, lang = lang)
                row = {"sample_id": key, "lang": lang, "input": text, "output": result["output"], "status": result["status"],
                       "reason": result["reason"], "llm_output": tagged, "request": "tagger", "model": args.model}
                out.write(json.dumps(obj = row, ensure_ascii = False) + "\n")
                counts[result["status"]] = counts.get(result["status"], 0) + 1
            out.flush()
            done = start + len(chunk)
            print(f"[punctuate_tagger] {done}/{len(todo)} 条，{done / (time.time() - started):.1f} 句/秒，{counts}", flush = True)
    print(f"[punctuate_tagger] 完成，用时 {time.time() - started:.0f} 秒，{counts}", flush = True)


def evaluate(args: argparse.Namespace) -> None:
    """与 punctuate eval 相同：参考文本去掉标点（还原大小写的语言再转小写）作输入，还原后与参考比较。"""
    references = {}
    with open(file = args.input, mode = "r", encoding = "utf-8") as handle:
        for line in handle:
            row = json.loads(s = line)
            if row["lang"] == args.lang:
                references[row["sample_id"]] = row["original"]
    plain = fold if args.lang in RECASE else (lambda text: text)
    keys = list(references)
    inputs = [plain(strip_marks(text = references[key], lang = args.lang)) for key in keys]
    tagger = Tagger(model = args.model, device = args.device, batch_size = args.batch_size)
    started = time.time()
    tagged = tagger.restore(texts = inputs)
    elapsed = time.time() - started
    results = [transfer(source = text, llm_output = output, lang = args.lang) for text, output in zip(inputs, tagged)]
    predictions = [result["output"] or text for text, result in zip(inputs, results)]
    status = {}
    for result in results:
        status[result["status"]] = status.get(result["status"], 0) + 1
    report = {
        "lang": args.lang,
        "model": args.model,
        "sentences": len(keys),
        "status": status,
        "seconds": round(elapsed, 1),
        "tagger": score(references = [references[key] for key in keys], predictions = predictions, lang = args.lang),
        "examples": [
            {"reference": references[key], "input": text, "output": prediction}
            for key, text, prediction in list(zip(keys, inputs, predictions))[:20]
        ],
    }
    args.output.parent.mkdir(parents = True, exist_ok = True)
    args.output.write_text(data = json.dumps(obj = report, ensure_ascii = False, indent = 1), encoding = "utf-8")
    print(json.dumps(obj = {key: report[key] for key in ("lang", "sentences", "status", "seconds")}, ensure_ascii = False))


def main() -> None:
    parser = argparse.ArgumentParser(description = "用逐词分类模型还原标点与大小写")
    sub = parser.add_subparsers(dest = "command", required = True)
    for name in ("run", "eval"):
        command = sub.add_parser(name)
        command.add_argument("--model", default = DEFAULT_MODEL, help = "模型目录或 Hugging Face 仓库名")
        command.add_argument("--device", default = "cuda")
        command.add_argument("--batch-size", type = int, default = 64)
        if name == "run":
            command.add_argument("--out-dir", required = True, type = Path)
            command.add_argument("--langs", required = True, help = "逗号分隔的语言代码")
            command.add_argument("--chunk", type = int, default = 4096, help = "每处理这么多条写一次缓存")
        else:
            command.add_argument("--input", required = True, type = Path)
            command.add_argument("--lang", required = True)
            command.add_argument("--output", required = True, type = Path)
    args = parser.parse_args()
    if args.command == "run":
        run(args = args)
    else:
        evaluate(args = args)


if __name__ == "__main__":
    main()
