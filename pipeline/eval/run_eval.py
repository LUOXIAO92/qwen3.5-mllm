"""评测推理脚本（实验 v1，任务 T15）：批量转写评测清单并打分。

用法（在仓库根目录运行，推理需要 GPU，请经 GPU 锁排队）：

    flock <gpu.lock> python -m pipeline.eval.run_eval \
        --model-dir <完整模型目录> \
        --projector <projector.safetensors> \
        --manifest-dir <含 <lang>/test.jsonl 的目录> \
        --langs zh en ja \
        --limit 20 \
        --batch-size 8 \
        --output-dir <输出目录>

输入
====

- 评测清单 `<manifest-dir>/<lang>/test.jsonl`：每行含 sample_id、audio_path、duration_s、raw_transcription。
  audio_path 相对 `--audio-root`（默认等于 `--manifest-dir`），参考文本取 raw_transcription。
- `--model-dir`：带 remote code 的完整模型目录，用 `trust_remote_code = True` 加载，bf16 推理。
  目录里可以是阶段 1 的模型（projector 未训练），也可以是已经合入 projector 的模型。
- `--projector`（可选）：一个 safetensors 文件，只含完整模型 state_dict 中以 `model.audio_projector.`
  开头的键，键名与完整模型一致。加载前检查文件里的键恰好等于模型的全部 projector 参数，
  不缺也不多，形状一致。不给时直接使用 `--model-dir` 里的 projector。
- `--language-model`（可选）：全量微调的检查点中的 `language_model.safetensors`，含语言模型的全部参数（以
  `model.language_model.` 开头，与输出层共用的词嵌入只有一份），在加载模型和 projector 之后载入。同样检查键恰好
  等于模型的全部语言模型参数、形状一致；载入后输出层仍与词嵌入共用同一个权重。

输入格式
========

与训练一致（pipeline/CONVENTIONS.md「训练与评测的对话格式」）：用户消息依次是一段音频和
`Transcribe the audio.`，用模型自带 chat template 展开，`add_generation_prompt = True`、
`enable_thinking = False`，助手回合以空思考块开头。批量生成用左填充、贪心解码，
`max_new_tokens` 按该批最长音频的时长估算，防止未训练的模型无限输出。

`--prompt-format cpt`（任务 #71）用于纯续写格式训练的模型：提示只有一段音频的标记
`<|audio_start|><|audio_pad|><|audio_end|>`（由 processor 展开），不套 chat template，模型接着写出转写，
在 `<|endoftext|>` 停止（`<|im_end|>` 同样作为停止标记）。输出的清理与聊天格式相同。

输出（写入 `--output-dir`）
===========================

- `hypotheses.jsonl`：逐条结果，含 sample_id、lang、reference、hypothesis、duration_s、n_new_tokens，
  以及打分器给出的 basic、numeric 两种逐条分数。
- `summary.json` 与 `summary.md`：按语言的结果表，基础分和数词统一分都列出。
- 无人声评测集（语言代码 zxx，没有参考文本）不计 WER/CER：hypotheses.jsonl 保留 category 字段，
  summary.json 的 `nonspeech` 项给出「无人声误输出率」（输出去掉空白后非空的片段比例）及按 category 的分布。
  转换方法：`python -m pipeline.eval.synth_manifest <无人声评测集目录> <评测清单目录>`，再 `--langs zxx`。
- `perf.json`：每个批量的生成耗时、音频总时长、实时倍率、生成 token 数和峰值显存。

vLLM 引擎（--engine vllm，任务 #71）
===================================

用户 2026-11-27 决定本机评测改用 vLLM 的 docker 镜像（见 pipeline/eval/vllm_host.py）。提示、分批、每批的 max_new_tokens、
贪心解码、结束符、输出的清理与打分都与上面相同，只把计算分成两步：宿主机上 pipeline/eval/audio_embeds.py 按同样的批用
融合模型的音频塔与 projector 算出音频向量（就是 forward 中 masked_scatter 填进去的向量），容器中的 vLLM 以混合输入（音频
占位符的位置用这些向量，其余位置用 token id）贪心生成，每条的 max_tokens 是它所在批的 max_new_tokens。语言模型由 vLLM
加载（参考模型 --vllm-model，运行先导出，见 vllm_host）。位置编码：融合模型把音频 token 当作文字按顺序排位置（左填充的批
中每条都从 0 开始），vLLM 对没有图像、视频的请求同样按顺序给位置，两者相同。summary.json 另有 engine（vllm）与 vllm
（设置与用时），perf.json 为 vLLM 的记录。--first-logits K 时两种引擎都在 hypotheses.jsonl 中记下第一步生成时 logits 最大
的 K 个 token 与它们的 logits（核对两种引擎用）。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

from pipeline.eval.scoring import LANGS, METRIC_BY_LANG, NONSPEECH_LANG, false_output_summary, score_corpus
from pipeline.tokenizer_fix import check_tokenizer

PROMPT_TEXT = "Transcribe the audio."
# 样本格式，与 pipeline.train.data.PROMPT_FORMATS 相同。这里不导入训练模块，评测环境不需要训练的依赖。
PROMPT_FORMATS = ("chat", "cpt")
# 生成的停止标记：聊天格式的助手回合以 <|im_end|> 结束，纯续写格式以 <|endoftext|> 结束，两者都停止。
STOP_TOKENS = ["<|im_end|>", "<|endoftext|>"]
SAMPLING_RATE = 16000
PROJECTOR_PREFIX = "model.audio_projector."
# 与 pipeline.train.model.LANGUAGE_MODEL_PREFIXES 相同。这里不导入训练模块，评测环境不需要训练的依赖。
LANGUAGE_MODEL_PREFIXES = ("model.language_model.", "lm_head.")
_THINK_BLOCK = re.compile(pattern = r"<think>.*?</think>", flags = re.DOTALL)


def build_messages(audio_path: str, instruction: str = PROMPT_TEXT) -> list:
    """按 CONVENTIONS 构造一条对话：音频 + 指令（默认是训练用的固定指令）。"""
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": instruction},
            ],
        },
    ]


def audio_markup(processor) -> str:
    """一段音频的标记 <|audio_start|><|audio_pad|><|audio_end|>，与 chat template 中音频的写法相同。"""
    return processor.audio_start_token + processor.audio_token + processor.audio_end_token


def build_prompt(
    processor,
    audio_path: str,
    prompt_format: str = "chat",
    instruction: str = PROMPT_TEXT,
) -> str:
    """推理时的提示。chat：展开 chat template（关闭思考，结尾是助手回合开头和空的思考块）；cpt：只有音频标记。"""
    if prompt_format == "cpt":
        return audio_markup(processor = processor)
    if prompt_format != "chat":
        raise ValueError(f"未知的样本格式：{prompt_format}，可选 {PROMPT_FORMATS}")
    return processor.apply_chat_template(
        conversation = build_messages(audio_path = audio_path, instruction = instruction),
        tokenize = False,
        add_generation_prompt = True,
        enable_thinking = False,
    )


def load_waveform(path: str):
    """读取 16 kHz 音频为 float32 单声道波形。"""
    import soundfile as sf

    wave, sample_rate = sf.read(file = path, dtype = "float32")
    if sample_rate != SAMPLING_RATE:
        raise ValueError(f"{path} 采样率为 {sample_rate}，需要 {SAMPLING_RATE}")
    if wave.ndim > 1:
        wave = wave.mean(axis = 1)
    return wave


def clean_output(text: str) -> str:
    """去掉生成中残留的思考块、孤立的思考标记和特殊标记。"""
    text = _THINK_BLOCK.sub(repl = "", string = text)
    text = text.replace("<think>", "").replace("</think>", "")
    text = re.sub(pattern = r"<\|[^|>]*\|>", repl = "", string = text)
    return text.strip()


def read_manifest(manifest_dir: Path, lang: str, limit: int | None) -> list:
    """读取一种语言的 test 清单，取前 limit 条。"""
    path = manifest_dir / lang / "test.jsonl"
    rows = []
    with open(file = path, mode = "r", encoding = "utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(s = line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_projector(model, projector_path: Path) -> None:
    """加载 projector 检查点：键必须恰好覆盖模型的全部 projector 参数。"""
    from safetensors.torch import load_file

    state = load_file(filename = str(projector_path), device = "cpu")
    expected = {
        key: value
        for key, value in model.state_dict().items()
        if key.startswith(PROJECTOR_PREFIX)
    }
    missing = sorted(set(expected) - set(state))
    extra = sorted(set(state) - set(expected))
    if missing or extra:
        raise ValueError(
            f"projector 检查点的键与模型不一致。缺少：{missing}；多余：{extra}",
        )
    for key, value in expected.items():
        if tuple(state[key].shape) != tuple(value.shape):
            raise ValueError(
                f"{key} 形状不一致：检查点 {tuple(state[key].shape)}，模型 {tuple(value.shape)}",
            )
    result = model.load_state_dict(state_dict = state, strict = False)
    # strict = False 只是因为文件只含 projector；此处确认没有意外的多余键
    if result.unexpected_keys:
        raise ValueError(f"出现多余的键：{result.unexpected_keys}")


def load_language_model(model, language_model_path: Path) -> None:
    """加载全量微调后的语言模型权重：键必须恰好覆盖模型的全部语言模型参数，形状一致。

    named_parameters 中与输出层共用的词嵌入只有 model.language_model.embed_tokens.weight 一个名字。
    """
    from safetensors.torch import load_file

    state = load_file(filename = str(language_model_path), device = "cpu")
    expected = {
        name: parameter.shape
        for name, parameter in model.named_parameters()
        if name.startswith(LANGUAGE_MODEL_PREFIXES)
    }
    missing = sorted(set(expected) - set(state))
    extra = sorted(set(state) - set(expected))
    if missing or extra:
        raise ValueError(
            f"语言模型权重的键与模型不一致。缺少：{missing[:5]}；多余：{extra[:5]}",
        )
    for name, shape in expected.items():
        if tuple(state[name].shape) != tuple(shape):
            raise ValueError(
                f"{name} 形状不一致：文件 {tuple(state[name].shape)}，模型 {tuple(shape)}",
            )
    embedding = model.get_input_embeddings().weight
    tied = model.lm_head.weight.data_ptr() == embedding.data_ptr()
    result = model.load_state_dict(state_dict = state, strict = False)
    if result.unexpected_keys:
        raise ValueError(f"出现多余的键：{result.unexpected_keys}")
    # load_state_dict 原地复制，共用的权重仍然共用；确认输出层随词嵌入一起更新了。
    if tied and model.lm_head.weight.data_ptr() != model.get_input_embeddings().weight.data_ptr():
        raise ValueError("载入后输出层不再与词嵌入共用权重")


def load_model(
    model_dir: Path,
    projector_path: Path | None,
    language_model_path: Path | None = None,
    device: str = "cuda",
    attn_implementation: str | dict | None = None,
):
    """载入融合模型（可选换上 projector 与全量微调的语言模型）。attn_implementation 不给时用 transformers 的默认（sdpa）；
    也可以是按子配置给出的字典，例如思考检查的 think_check.FLASH_ATTENTION。"""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path = str(model_dir),
        trust_remote_code = True,
    )
    # 分词规则必须与原始 Qwen3.5 相同（#77），提示中的文字（例如级联参照中的转写）才按语言模型预训练时的切法分词
    check_tokenizer(tokenizer = processor, source = model_dir)
    processor.tokenizer.padding_side = "left"
    extra = {} if attn_implementation is None else {"attn_implementation": attn_implementation}
    model = AutoModelForImageTextToText.from_pretrained(
        pretrained_model_name_or_path = str(model_dir),
        trust_remote_code = True,
        dtype = torch.bfloat16,
        **extra,
    )
    if projector_path is not None:
        load_projector(model = model, projector_path = projector_path)
    if language_model_path is not None:
        load_language_model(model = model, language_model_path = language_model_path)
    model.to(device = device)
    model.eval()
    return model, processor


def build_inputs(processor, audio_paths: list, waveforms: list, prompt_format: str = "chat"):
    """按样本格式构造提示并处理音频，返回 (展开前的文本列表, 模型输入)。"""
    prompts = [
        build_prompt(processor = processor, audio_path = path, prompt_format = prompt_format)
        for path in audio_paths
    ]
    inputs = processor(
        text = prompts,
        audio = waveforms,
        sampling_rate = SAMPLING_RATE,
        padding = True,
        return_tensors = "pt",
    )
    return prompts, inputs


def generate(
    model,
    processor,
    inputs,
    max_new_tokens: int,
    device: str = "cuda",
    first_logits: int = 0,
    extras: list | None = None,
) -> tuple:
    """贪心解码一个批量（左填充），返回 (清理后的输出, n_new_tokens 列表, 生成耗时秒, 峰值显存字节)。

    在 CPU 上运行时（只用来核对流程）峰值显存记为 0。first_logits 大于 0 时，把每条第一步 logits 最大的 first_logits 个
    [token id, logit] 依次追加到 extras 中。
    """
    import torch

    cuda = device == "cuda"
    inputs = inputs.to(device = device)
    if "input_features" in inputs:
        inputs["input_features"] = inputs["input_features"].to(dtype = torch.bfloat16)
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    extra = {"return_dict_in_generate": True, "output_logits": True} if first_logits > 0 else {}
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            do_sample = False,
            use_cache = True,
            pad_token_id = processor.tokenizer.pad_token_id,
            # 阶段 1 的 generation_config 只在 <|endoftext|> 停止；助手回合以 <|im_end|> 结束，两者都作为停止标记
            eos_token_id = processor.tokenizer.convert_tokens_to_ids(tokens = STOP_TOKENS),
            **extra,
        )
    if first_logits > 0:
        values, ids = torch.topk(input = generated.logits[0].float(), k = first_logits, dim = -1)
        for row_ids, row_values in zip(ids.cpu().tolist(), values.cpu().tolist()):
            extras.append([[int(token), float(value)] for token, value in zip(row_ids, row_values)])
        generated = generated.sequences
    if cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() if cuda else 0

    new_ids = generated[:, inputs["input_ids"].shape[1]:]
    pad_id = processor.tokenizer.pad_token_id
    n_new = [int((row != pad_id).sum()) for row in new_ids]
    texts = processor.batch_decode(sequences = new_ids, skip_special_tokens = True)
    return [clean_output(text = t) for t in texts], n_new, elapsed, peak


def batch_max_new_tokens(samples: list, max_new_tokens_cfg: dict) -> int:
    """一个批量的 max_new_tokens：按批内最长音频的时长（采样点数 ÷ 16000）估算。"""
    longest = max(samples) / SAMPLING_RATE
    return int(math.ceil(longest * max_new_tokens_cfg["tokens_per_second"]) + max_new_tokens_cfg["extra"])


def transcribe_batch(
    model,
    processor,
    rows: list,
    audio_root: Path,
    max_new_tokens_cfg: dict,
    prompt_format: str = "chat",
    first_logits: int = 0,
    extras: list | None = None,
) -> tuple:
    """转写一个批量，返回 (hypotheses, n_new_tokens 列表, 生成耗时秒, 峰值显存字节, max_new_tokens)。"""
    audio_paths = [str(audio_root / row["audio_path"]) for row in rows]
    waveforms = [load_waveform(path = path) for path in audio_paths]
    max_new_tokens = batch_max_new_tokens(samples = [len(wave) for wave in waveforms], max_new_tokens_cfg = max_new_tokens_cfg)
    _, inputs = build_inputs(
        processor = processor,
        audio_paths = audio_paths,
        waveforms = waveforms,
        prompt_format = prompt_format,
    )
    texts, n_new, elapsed, peak = generate(
        model = model,
        processor = processor,
        inputs = inputs,
        max_new_tokens = max_new_tokens,
        first_logits = first_logits,
        extras = extras,
    )
    return texts, n_new, elapsed, peak, max_new_tokens


def format_markdown(summary: dict) -> str:
    """按语言的结果表（错误率为百分数）。"""

    def pct(v):
        return "n/a" if v is None else f"{v * 100:.2f}"

    lines = [
        "| 语言 | 指标 | 条数 | 参考长度 | 基础分 % | 数词统一分 % | S | D | I |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lang, v in summary.items():
        basic, numeric = v["basic"], v["numeric"]
        lines.append(
            f"| {lang} | {basic['metric']} | {basic['n']} | {numeric.get('ref_len', '')} "
            f"| {pct(basic['rate'])} | {pct(numeric['rate'])} "
            f"| {numeric.get('sub', '')} | {numeric.get('del', '')} | {numeric.get('ins', '')} |",
        )
    return "\n".join(lines) + "\n"


def format_nonspeech_markdown(summary: dict) -> str:
    """无人声误输出率表：总体一行，再按 category 逐行。"""
    lines = ["| 类别 | 片段数 | 误输出数 | 误输出率 % |", "| --- | ---: | ---: | ---: |"]
    rows = [("全部", summary)] + list(summary["by_category"].items())
    for name, v in rows:
        rate = "n/a" if v["rate"] is None else f"{v['rate'] * 100:.1f}"
        lines.append(f"| {name} | {v['n']} | {v['false_outputs']} | {rate} |")
    return "\n".join(lines) + "\n"


def parse_args(argv: list | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "批量转写评测清单并按语言打分")
    parser.add_argument("--model-dir", required = True, type = Path, help = "完整模型目录（remote code）")
    parser.add_argument("--projector", default = None, type = Path, help = "projector 检查点（safetensors，可选）")
    parser.add_argument(
        "--language-model",
        default = None,
        type = Path,
        help = "全量微调检查点中的 language_model.safetensors（可选），在模型和 projector 之后载入",
    )
    parser.add_argument("--manifest-dir", required = True, type = Path, help = "含 <lang>/test.jsonl 的目录")
    parser.add_argument("--audio-root", default = None, type = Path, help = "audio_path 的根目录，默认等于 --manifest-dir")
    parser.add_argument("--langs", nargs = "+", default = LANGS, help = "要评测的语言代码，默认全部 18 种")
    parser.add_argument("--limit", default = None, type = int, help = "每种语言最多评测多少条，默认全部")
    parser.add_argument("--batch-size", default = 8, type = int, help = "批量大小")
    parser.add_argument("--tokens-per-second", default = 12.0, type = float, help = "max_new_tokens 估算：每秒音频的 token 上限")
    parser.add_argument("--extra-tokens", default = 32, type = int, help = "max_new_tokens 估算：额外余量")
    parser.add_argument("--output-dir", required = True, type = Path, help = "输出目录")
    parser.add_argument("--show-inputs", default = 0, type = int, help = "每种语言打印前几条样本展开前的输入文本")
    parser.add_argument(
        "--prompt-format",
        default = "chat",
        choices = PROMPT_FORMATS,
        help = "提示格式：chat 聊天格式（音频 + 固定指令），cpt 只有音频标记（纯续写格式训练的模型），与训练时相同",
    )
    parser.add_argument("--first-logits", default = 0, type = int, help = "记下第一步 logits 最大的这么多个 token（核对两种引擎用）")
    from pipeline.eval import vllm_host

    vllm_host.add_arguments(parser = parser)
    args = parser.parse_args(argv)
    vllm_host.check_arguments(
        parser = parser,
        args = args,
        default_job_dir = args.output_dir / "vllm-job",
        language_model = args.language_model,
    )
    return args


def load_processor(model_dir: Path):
    """只载入 processor（检查 #77 的分词规则），vLLM 引擎时构造提示和解码用。"""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path = str(model_dir), trust_remote_code = True)
    check_tokenizer(tokenizer = processor, source = model_dir)
    processor.tokenizer.padding_side = "left"
    return processor


def vllm_generate_batches(args, batches: list, processor, log, kind: str) -> tuple:
    """vLLM 引擎：batches 为 [(max_new_tokens, [audio_embeds 的 item（不含 index、batch）])]，按 transformers 版的批算音频
    向量、贪心生成。返回 (按批、批内顺序排列的 [(清理后的输出, n_new_tokens, 生成的 token id, 首步 logits 或 None)],
    vLLM 的记录)。args 需要 model_dir、projector、language_model、output_dir、first_logits 与 vllm_host 的参数。
    """
    from pipeline.eval import vllm_host
    from pipeline.eval.audio_embeds import run_encoder

    tokenizer = processor.tokenizer
    stop_ids = vllm_host.stop_ids_of(tokenizer = tokenizer)
    first_logits = getattr(args, "first_logits", 0)
    items = []
    budgets = {}
    for number, (max_new_tokens, group) in enumerate(batches):
        for item in group:
            index = len(items)
            items.append({"index": index, "batch": number, **item})
            budgets[index] = int(max_new_tokens)

    def prepare(job_dir):
        encoded = run_encoder(
            model_dir = args.model_dir,
            projector = args.projector,
            items = items,
            job_dir = job_dir,
            log = log,
        )
        requests = []
        for item in items:
            row = encoded[item["index"]]
            request = {
                "index": item["index"],
                "seed": 0,
                "prompt_token_ids": row["prompt_token_ids"],
                "embeds": {"file": row["file"], "key": row["key"]},
                "max_tokens": budgets[item["index"]],
            }
            if first_logits > 0:
                request["logprobs"] = first_logits
            requests.append(request)
        settings = vllm_host.job_settings(
            args = args,
            max_model_len = max(len(request["prompt_token_ids"]) + request["max_tokens"] for request in requests),
            max_tokens = 1,
            temperature = 0.0,
            stop_token_ids = stop_ids,
            enable_prompt_embeds = True,
            embeds_token_id = int(tokenizer.convert_tokens_to_ids(processor.audio_token)),
            logprob_steps = 1,
            **({"max_logprobs": first_logits, "logprobs_mode": "raw_logits"} if first_logits > 0 else {}),
        )
        return requests, settings

    results, record = vllm_host.run_job(
        args = args,
        model_dir = args.model_dir,
        language_model = args.language_model,
        requests = None,
        settings = None,
        kind = kind,
        label = Path(args.output_dir).parent.name,
        log = log,
        prepare = prepare,
    )
    pad_id = tokenizer.pad_token_id
    outputs = []
    for item in items:
        row = results[item["index"]]
        ids = [int(token) for token in row["token_ids"]]
        text = clean_output(text = tokenizer.decode(token_ids = ids, skip_special_tokens = True))
        first = row["logprobs"][0] if first_logits > 0 and row.get("logprobs") else None
        outputs.append((text, sum(1 for token in ids if token != pad_id), ids, first))
    return outputs, record


def transcribe_vllm(args, audio_root: Path, log) -> tuple:
    """vLLM 引擎的转写：分批（每种语言按时长排序后每 --batch-size 条一批）与每批的 max_new_tokens 都与 transformers 版相同。
    返回 (results, perf)，results 的各行与 transformers 版相同。"""
    import soundfile as sf

    processor = load_processor(model_dir = args.model_dir)
    token_cfg = {"tokens_per_second": args.tokens_per_second, "extra": args.extra_tokens}
    batches = []
    batch_rows = []
    for lang in args.langs:
        rows = read_manifest(manifest_dir = args.manifest_dir, lang = lang, limit = args.limit)
        rows = sorted(rows, key = lambda r: r["duration_s"])
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start: start + args.batch_size]
            paths = [str(audio_root / row["audio_path"]) for row in batch]
            samples = [sf.info(file = path).frames for path in paths]
            max_new = batch_max_new_tokens(samples = samples, max_new_tokens_cfg = token_cfg)
            group = [
                {
                    "prompt": build_prompt(processor = processor, audio_path = path, prompt_format = args.prompt_format),
                    "audio_paths": [str(Path(path).resolve())],
                }
                for path in paths
            ]
            batches.append((max_new, group))
            batch_rows.append((lang, batch, max_new))
    outputs, record = vllm_generate_batches(
        args = args,
        batches = batches,
        processor = processor,
        log = log,
        kind = "fleurs",
    )
    results = []
    position = 0
    batch_perf = []
    for lang, batch, max_new in batch_rows:
        for row in batch:
            text, count, _, first = outputs[position]
            position += 1
            result = {
                "sample_id": row["sample_id"],
                "lang": lang,
                "reference": row["raw_transcription"],
                "hypothesis": text,
                "duration_s": row["duration_s"],
                "n_new_tokens": count,
                **({"category": row["category"]} if "category" in row else {}),
            }
            if first is not None:
                result["first_logits"] = first
            results.append(result)
        batch_perf.append({"lang": lang, "batch_size": len(batch), "max_new_tokens": max_new})
    return results, {"engine": "vllm", "vllm": record, "batches": batch_perf}


def main(argv: list | None = None) -> int:
    args = parse_args(argv = argv)
    for lang in args.langs:
        if lang not in METRIC_BY_LANG and lang != NONSPEECH_LANG:
            raise SystemExit(f"不支持的语言代码：{lang}")
    audio_root = args.audio_root if args.audio_root is not None else args.manifest_dir
    args.output_dir.mkdir(parents = True, exist_ok = True)

    def log(message):
        print(f"{time.strftime('%H:%M:%S')} {message}", flush = True)

    if args.engine == "vllm":
        results, perf = transcribe_vllm(args = args, audio_root = audio_root, log = log)
    else:
        results, perf = transcribe_hf(args = args, audio_root = audio_root)
    return write_results(args = args, results = results, perf = perf)


def transcribe_hf(args, audio_root: Path) -> tuple:
    """transformers 版的转写：返回 (results, perf)。"""
    model, processor = load_model(
        model_dir = args.model_dir,
        projector_path = args.projector,
        language_model_path = args.language_model,
    )
    token_cfg = {"tokens_per_second": args.tokens_per_second, "extra": args.extra_tokens}

    results = []
    perf = []
    for lang in args.langs:
        rows = read_manifest(manifest_dir = args.manifest_dir, lang = lang, limit = args.limit)
        # 按时长排序，同一批的音频长度接近，填充少
        rows = sorted(rows, key = lambda r: r["duration_s"])
        if args.show_inputs:
            prompts = [
                build_prompt(
                    processor = processor,
                    audio_path = str(audio_root / r["audio_path"]),
                    prompt_format = args.prompt_format,
                )
                for r in rows[: args.show_inputs]
            ]
            for r, prompt in zip(rows, prompts):
                print(f"[{lang}] {r['sample_id']} 展开前的输入文本：\n{prompt}\n", flush = True)
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start: start + args.batch_size]
            extras = []
            hyps, n_new, elapsed, peak, max_new = transcribe_batch(
                model = model,
                processor = processor,
                rows = batch,
                audio_root = audio_root,
                max_new_tokens_cfg = token_cfg,
                prompt_format = args.prompt_format,
                first_logits = args.first_logits,
                extras = extras,
            )
            audio_seconds = sum(r["duration_s"] for r in batch)
            perf.append(
                {
                    "lang": lang,
                    "batch_size": len(batch),
                    "audio_seconds": round(audio_seconds, 2),
                    "gen_seconds": round(elapsed, 3),
                    "realtime_factor": round(audio_seconds / elapsed, 2),
                    "new_tokens": sum(n_new),
                    "tokens_per_second": round(sum(n_new) / elapsed, 1),
                    "max_new_tokens": max_new,
                    "peak_mem_gib": round(peak / 2 ** 30, 2),
                },
            )
            for number, (row, hyp, n) in enumerate(zip(batch, hyps, n_new)):
                results.append(
                    {
                        "sample_id": row["sample_id"],
                        "lang": lang,
                        "reference": row["raw_transcription"],
                        "hypothesis": hyp,
                        "duration_s": row["duration_s"],
                        "n_new_tokens": n,
                        **({"category": row["category"]} if "category" in row else {}),
                        **({"first_logits": extras[number]} if extras else {}),
                    },
                )
            print(f"[{lang}] {start + len(batch)}/{len(rows)} 条，{elapsed:.1f}s，峰值显存 {peak / 2 ** 30:.2f} GiB", flush = True)
    return results, perf


def write_results(args, results: list, perf) -> int:
    """打分，写 hypotheses.jsonl、summary.json、summary.md 与 perf.json（两种引擎相同）。"""
    speech = [row for row in results if row["lang"] != NONSPEECH_LANG]
    nonspeech = [row for row in results if row["lang"] == NONSPEECH_LANG]
    per_sample, summary = score_corpus(rows = speech)
    with open(file = args.output_dir / "hypotheses.jsonl", mode = "w", encoding = "utf-8") as f:
        for row, scored in zip(speech, per_sample):
            record = dict(row)
            record["basic"] = scored["basic"]
            record["numeric"] = scored["numeric"]
            f.write(json.dumps(obj = record, ensure_ascii = False) + "\n")
        for row in nonspeech:
            f.write(json.dumps(obj = row, ensure_ascii = False) + "\n")
    nonspeech_summary = false_output_summary(rows = nonspeech) if nonspeech else None
    table = format_markdown(summary = dict(summary)) if speech else ""
    if nonspeech_summary is not None:
        summary["nonspeech"] = nonspeech_summary
    if isinstance(perf, dict) and perf.get("engine") == "vllm":
        summary["engine"] = "vllm"
        summary["vllm"] = perf["vllm"]
    (args.output_dir / "summary.json").write_text(
        data = json.dumps(obj = summary, ensure_ascii = False, indent = 2),
        encoding = "utf-8",
    )
    if nonspeech_summary is not None:
        table += ("\n" if speech else "") + format_nonspeech_markdown(summary = nonspeech_summary)
    (args.output_dir / "summary.md").write_text(data = table, encoding = "utf-8")
    (args.output_dir / "perf.json").write_text(
        data = json.dumps(obj = perf, ensure_ascii = False, indent = 2),
        encoding = "utf-8",
    )
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
