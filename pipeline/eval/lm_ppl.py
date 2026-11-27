"""语言模型对 FLEURS 参考文本的每字困惑度（任务 #66、#71 用来看全量微调后的遗忘）。

    flock <gpu.lock> python -m pipeline.eval.lm_ppl --model-dir <融合模型目录> \\
        [--language-model <检查点>/language_model.safetensors] \\
        --manifest-dir <含 <lang>/test.jsonl 的 FLEURS 评测清单目录> --output <结果 json>

与 #71 第一轮 Colab 驱动（t71_cpt_run.py）中的计算相同，结果可以直接比较：每种语言取 test.jsonl 的 raw_transcription，
32 条一批右填充、不加特殊标记，模型只看文字（不给音频），第一个 token 之后的每个 token 的负对数似然相加，除以去掉
空白后的字符数，困惑度为它的指数。不给 --language-model 时是原始语言模型。

vLLM 引擎（--engine vllm，任务 #71，见 pipeline/eval/vllm_host.py）：每条文字单独作为提示（与右填充的批中各条的 token 相同），
只生成 1 个 token，prompt_logprobs = 0 返回每个位置实际 token 的对数概率（float32 的 log_softmax），第一个 token 之后的
负对数似然相加，其余与上面相同。结果文件另有 engine 与 vllm 两项。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

LANGS = ["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it", "id", "ms", "th", "vi", "fil", "ar", "tr", "fa"]


def main(argv = None):
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required = True, type = Path)
    parser.add_argument("--language-model", default = None, type = Path)
    parser.add_argument("--manifest-dir", required = True, type = Path)
    parser.add_argument("--langs", nargs = "+", default = LANGS)
    parser.add_argument("--batch-size", type = int, default = 32)
    parser.add_argument("--output", required = True, type = Path)
    from pipeline.eval import vllm_host

    vllm_host.add_arguments(parser = parser)
    args = parser.parse_args(args = argv)
    vllm_host.check_arguments(
        parser = parser,
        args = args,
        default_job_dir = args.output.parent / f"{args.output.stem}.vllm-job",
        language_model = args.language_model,
    )
    if args.engine == "vllm":
        return main_vllm(args = args)

    import torch

    from pipeline.eval.run_eval import load_model

    model, processor = load_model(model_dir = args.model_dir, projector_path = None, language_model_path = args.language_model)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "right"
    device = next(model.parameters()).device
    results = {}
    for lang in args.langs:
        with open(file = args.manifest_dir / lang / "test.jsonl", encoding = "utf-8") as handle:
            texts = [json.loads(s = line)["raw_transcription"] for line in handle if line.strip()]
        nll = 0.0
        chars = 0
        for start in range(0, len(texts), args.batch_size):
            batch = texts[start: start + args.batch_size]
            encoded = tokenizer(
                batch,
                return_tensors = "pt",
                padding = True,
                add_special_tokens = False,
            ).to(device = device)
            with torch.no_grad():
                logits = model(input_ids = encoded["input_ids"], attention_mask = encoded["attention_mask"]).logits.float()
            log_probs = torch.log_softmax(input = logits[:, :-1], dim = -1)
            target = encoded["input_ids"][:, 1:]
            mask = encoded["attention_mask"][:, 1:].bool()
            token_nll = -log_probs.gather(dim = -1, index = target.unsqueeze(dim = -1)).squeeze(dim = -1)
            nll += float(token_nll[mask].sum())
            chars += sum(len("".join(text.split())) for text in batch)
        results[lang] = {"char_ppl": round(number = math.exp(nll / chars), ndigits = 3), "nats_per_char": round(number = nll / chars, ndigits = 4)}
        print(f"{lang}: 每字困惑度 {results[lang]['char_ppl']}", flush = True)
    results["engine"] = "hf"
    args.output.parent.mkdir(parents = True, exist_ok = True)
    args.output.write_text(data = json.dumps(obj = results, ensure_ascii = False, indent = 1), encoding = "utf-8")
    return 0


def main_vllm(args):
    """vLLM 引擎：全部语言的文字一起交给 vLLM（一个容器），按语言汇总。"""
    import time

    from pipeline.eval import vllm_host
    from pipeline.eval.run_eval import load_processor
    from pipeline.eval.think_check import STOP_IDS

    def log(message):
        print(f"{time.strftime('%H:%M:%S')} {message}", flush = True)

    tokenizer = load_processor(model_dir = args.model_dir).tokenizer
    requests = []
    owners = []
    chars = {}
    for lang in args.langs:
        with open(file = args.manifest_dir / lang / "test.jsonl", encoding = "utf-8") as handle:
            texts = [json.loads(s = line)["raw_transcription"] for line in handle if line.strip()]
        chars[lang] = sum(len("".join(text.split())) for text in texts)
        for text in texts:
            ids = tokenizer(text, add_special_tokens = False)["input_ids"]
            owners.append(lang)
            requests.append({"index": len(requests), "seed": 0, "prompt_token_ids": ids, "max_tokens": 1, "prompt_logprobs": 0})
    settings = vllm_host.job_settings(
        args = args,
        max_model_len = max(len(request["prompt_token_ids"]) for request in requests) + 1,
        max_tokens = 1,
        temperature = 0.0,
        stop_token_ids = list(STOP_IDS),
    )
    output_results, record = vllm_host.run_job(
        args = args,
        model_dir = args.model_dir,
        language_model = args.language_model,
        requests = requests,
        settings = settings,
        kind = "ppl",
        label = args.output.parent.name,
        log = log,
    )
    nll = {lang: 0.0 for lang in args.langs}
    for request, lang in zip(requests, owners):
        values = output_results[request["index"]]["prompt_logprobs"]
        nll[lang] -= sum(value for value in values[1:])
    results = {}
    for lang in args.langs:
        results[lang] = {"char_ppl": round(number = math.exp(nll[lang] / chars[lang]), ndigits = 3), "nats_per_char": round(number = nll[lang] / chars[lang], ndigits = 4)}
        print(f"{lang}: 每字困惑度 {results[lang]['char_ppl']}", flush = True)
    results["engine"] = "vllm"
    results["vllm"] = record
    args.output.parent.mkdir(parents = True, exist_ok = True)
    args.output.write_text(data = json.dumps(obj = results, ensure_ascii = False, indent = 1), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
