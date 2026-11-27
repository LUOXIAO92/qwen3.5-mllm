"""语音指令遵循探针（任务 #72）：同一批 FLEURS 音频换不同指令提问，看模型照做还是一律转写。

用法（在仓库根目录运行，推理需要 GPU，请经 GPU 锁排队）：

    flock <gpu.lock> python -m pipeline.eval.instruct_probe \\
        --model-dir <完整模型目录> \\
        --projector <projector.safetensors> \\
        [--language-model <language_model.safetensors>] \\
        --manifest-dir <含 <lang>/test.jsonl 的 FLEURS 评测清单目录> \\
        --output-dir <输出目录> \\
        [--langs zh en ja] [--per-lang 30] [--prompt-format chat cpt] [--cascade]

模型的载入与 run_eval.py 相同（--model-dir、--projector、--language-model 的含义和检查都一样），提问用聊天格式：
用户回合依次是音频和指令，用模型自带 chat template 展开，enable_thinking = False，批量贪心解码，左填充，在
<|im_end|> 或 <|endoftext|> 处停止，max_new_tokens 按该批最长音频的时长估算（--tokens-per-second、--extra-tokens）。

样本
====

每种语言从 `<manifest-dir>/<lang>/test.jsonl`（pipeline.eval.fleurs_prepare 生成，audio_path 相对 --audio-root，
默认等于 --manifest-dir）中按 --seed 固定抽 --per-lang 条（默认 30），四项任务都用这同一批音频：

1. transcribe：`Transcribe the audio.`（训练用的指令）和改写过的 `Please write down exactly what is said.`，
   按 scoring.py 计算各语言的 WER/CER（基础分与数词统一分）。
2. what_heard：`What did you hear?` 和 `你听到了什么？`，保存输出，统计其中「几乎逐字的转写」所占比例。
3. translate：`Translate the speech into English.` 和 `把这段话翻译成中文。`，用 FLEURS 中同一句话（fleurs_id 相同）
   的英文、中文转写作参考译文计算 chrF；源语言与目标语言相同时不问。另统计输出直接转写了原文、没有翻译的比例。
4. timestamps：同一种语言的 2–3 条音频在内存中首尾相接，中间插入 --gap-seconds（默认 0.5 秒）的静音，总长不超过
   --concat-max-seconds（默认 30 秒）。各段的起止时间按实际采样点数算出，问其中一段：
   `What is said between X and Y seconds?` 和 `第 X 秒到第 Y 秒之间说了什么？`（X、Y 保留一位小数）。
   按这一段的转写计算 WER/CER，另统计输出把整段拼接音频都转写出来的比例。

「几乎逐字的转写」：输出与参考转写在基础规范化后的字符错误率（CER，去掉空白按字符计算，所有语言都按字符）
低于 --verbatim-threshold（默认 0.3）。每条输出都记下它与源语言转写的这个字符错误率和是否逐字（verbatim），
用来看模型是否不管指令一律转写。

--prompt-format 可以给出 chat、cpt 或两者（默认 chat）。cpt 只用于第 1 项：提示只有音频标记（纯续写格式训练的
模型自己的格式，与 run_eval.py --prompt-format cpt 相同），记为指令 transcribe_cpt。纯续写格式训练的模型同时用
chat 与 cpt 探测，就能比较两种格式。

--cascade：级联参照。不用音频，把参考转写放进用户回合代替音频（CASCADE_TEMPLATE），问同样的指令，看原始的语言模型
怎样照做；不给 --projector 和 --language-model 时就是原始语言模型。时间段一项需要音频，级联时跳过。

输出（写入 --output-dir）
=========================

- `generations.jsonl`：逐条生成结果，边生成边追加；中断后用同样的参数再运行，已完成的条目跳过。
  `probe_config.json` 记下模型、检查点和级联与否，同一个输出目录换了这些时停止，避免混入另一个模型的结果。
- `items.jsonl`：逐条结果与分数：任务、指令、语言、样本、参考、输出、生成 token 数，以及各项分数。
- `summary.json` 与 `summary.md`：按任务、指令和语言汇总，ALL 是各语言数值的无权重平均。

vLLM 引擎（--engine vllm，任务 #71）：与 run_eval.py 的 vLLM 引擎相同（宿主机按同样的批算音频向量，容器中的 vLLM 贪心生成，
每条的 max_tokens 为它所在批的 max_new_tokens；见 run_eval.py 开头与 pipeline/eval/vllm_host.py）。请求、分批、打分与输出
文件都不变；probe_config.json 另记 engine，同一个输出目录不混用两种引擎的结果；summary.json 的 config 中记下 engine 与
vLLM 的设置。级联参照（--cascade）只用 transformers。

指令的位置（--instruction-placement，任务 #71 的系统提示遵循评测，见 pipeline/eval/sysprompt_local.py）：user（默认）为上面
的做法，指令在用户回合中音频之后；system 时指令是系统消息，用户回合只有音频（chat template 展开为
`<|im_start|>system\\n指令<|im_end|>\\n<|im_start|>user\\n<音频><|im_end|>\\n<|im_start|>assistant\\n` 加空的思考块）。只影响
聊天格式；样本、指令文字、分批与打分都不变，两种位置的结果可以逐条比较。probe_config.json 另记 instruction_placement
（只在 system 时记，已有的 user 结果目录不受影响），summary.json 的 config 中都记下。级联参照只用 user。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

from pipeline.eval.scoring import LANGS, METRIC_BY_LANG, align, normalize_basic, score_corpus, tokenize

TASKS = ("transcribe", "what_heard", "translate", "timestamps")
INSTRUCTIONS = {
    "transcribe": [
        {"id": "transcribe", "text": "Transcribe the audio."},
        {"id": "transcribe_paraphrase", "text": "Please write down exactly what is said."},
    ],
    "what_heard": [
        {"id": "heard_en", "text": "What did you hear?"},
        {"id": "heard_zh", "text": "你听到了什么？"},
    ],
    "translate": [
        {"id": "translate_en", "text": "Translate the speech into English.", "target_lang": "en"},
        {"id": "translate_zh", "text": "把这段话翻译成中文。", "target_lang": "zh"},
    ],
    "timestamps": [
        {"id": "segment_en", "text": "What is said between {start:.1f} and {end:.1f} seconds?"},
        {"id": "segment_zh", "text": "第 {start:.1f} 秒到第 {end:.1f} 秒之间说了什么？"},
    ],
}
# 严格版指令（用户 2026-11-24）：要求只输出答案、不加任何其他文字，排除「加开场白、Markdown、概括」这类与任务无关的输出。
# 「你听到了什么」本来就是开放式问题，不在严格版中。id 与默认版不同，结果可以并列比较。
STRICT_INSTRUCTIONS = {
    "transcribe": [
        {
            "id": "transcribe_strict",
            "text": "Transcribe the speech in this audio verbatim, in the language it is spoken. "
                    "Output only the transcription, with no other text.",
        },
        {"id": "transcribe_strict_zh", "text": "逐字转写这段音频，使用说话所用的语言。只输出转写内容，不要输出任何其他文字。"},
    ],
    "translate": [
        {
            "id": "translate_en_strict",
            "text": "Translate the speech in this audio into English. Output only the English translation, with no other text.",
            "target_lang": "en",
        },
        {"id": "translate_zh_strict", "text": "把这段音频里说的话翻译成中文。只输出中文译文，不要输出任何其他文字。", "target_lang": "zh"},
    ],
    "timestamps": [
        {
            "id": "segment_en_strict",
            "text": "Transcribe verbatim only the words spoken between {start:.1f} and {end:.1f} seconds of this audio. "
                    "Output only those words, with no other text.",
        },
        {"id": "segment_zh_strict", "text": "逐字转写这段音频第 {start:.1f} 秒到第 {end:.1f} 秒之间说的话。只输出这部分内容，不要输出任何其他文字。"},
    ],
}
INSTRUCTION_SETS = {"default": INSTRUCTIONS, "strict": STRICT_INSTRUCTIONS}
# cpt 格式只有音频标记，没有指令。
CPT_INSTRUCTION = {"id": "transcribe_cpt", "text": None}
# 聊天格式中指令的位置：user 为用户回合中音频之后，system 为系统消息（用户回合只有音频）。
PLACEMENTS = ("user", "system")
# 级联参照中代替音频的文字：参考转写放在指令之前，与音频在指令之前的位置相同。
CASCADE_TEMPLATE = "Audio transcript:\n{transcript}\n\n{instruction}"
SAMPLING_RATE = 16000


# ---------------------------------------------------------------- 样本与请求


def read_rows(manifest_dir: Path, lang: str) -> list:
    """读取一种语言的全部 test 清单行。"""
    rows = []
    with open(file = Path(manifest_dir) / lang / "test.jsonl", mode = "r", encoding = "utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(s = line))
    return rows


def select_rows(rows: list, count: int, seed: int, lang: str) -> list:
    """按 (seed, 语言) 固定抽 count 条：先按 sample_id 排序再打乱，结果与清单的行序无关。"""
    ordered = sorted(rows, key = lambda row: row["sample_id"])
    random.Random(x = f"{seed}-{lang}").shuffle(x = ordered)
    return ordered[:count]


def parallel_references(manifest_dir: Path, target_langs) -> dict:
    """{目标语言: {fleurs_id: 转写}}：FLEURS 同一句话（fleurs_id 相同）在各语言中是互译的句子，用作参考译文。

    同一个 fleurs_id 有几位说话人的录音，转写相同，取第一条。目标语言的清单不存在时这一语言为空。
    """
    references = {}
    for lang in target_langs:
        table = {}
        if (Path(manifest_dir) / lang / "test.jsonl").exists():
            for row in read_rows(manifest_dir = manifest_dir, lang = lang):
                table.setdefault(str(row["fleurs_id"]), row["raw_transcription"])
        references[lang] = table
    return references


def segment_bounds(lengths: list, gap: int, sampling_rate: int = SAMPLING_RATE) -> list:
    """各段音频（采样点数 lengths）之间插入 gap 个采样点的静音首尾相接后，每段的 (起点秒, 终点秒)。"""
    bounds = []
    position = 0
    for index, length in enumerate(lengths):
        if index:
            position += gap
        bounds.append((position / sampling_rate, (position + length) / sampling_rate))
        position += length
    return bounds


def concatenate(waveforms: list, gap: int):
    """把几段波形首尾相接，中间插入 gap 个采样点的静音，返回 (波形, 各段 (起点秒, 终点秒))。"""
    import numpy as np

    pieces = []
    for index, wave in enumerate(waveforms):
        if index:
            pieces.append(np.zeros(shape = (gap,), dtype = np.float32))
        pieces.append(np.asarray(a = wave, dtype = np.float32))
    bounds = segment_bounds(lengths = [len(wave) for wave in waveforms], gap = gap)
    return np.concatenate(pieces), bounds


def pick_concat_parts(
    rows: list,
    count: int,
    seed: int,
    lang: str,
    gap_s: float,
    max_total_s: float,
) -> list:
    """为时间段任务挑出 count 组音频，每组 2 或 3 条、fleurs_id 互不相同，总长（含静音）不超过 max_total_s。

    返回 [(这一组的清单行列表, 问第几段)]。每组先随机定条数 k，再从时长不超过 (max_total_s − 静音) / k 的音频中
    随机抽；这样的音频不足时改为 2 条。结果只由参数决定。
    """
    ordered = sorted(rows, key = lambda row: row["sample_id"])
    rng = random.Random(x = f"{seed}-{lang}-concat")
    groups = []
    for _ in range(count):
        parts = None
        for size in (rng.choice(seq = (2, 3)), 2):
            cap = (max_total_s - gap_s * (size - 1)) / size
            candidates = [row for row in ordered if row["duration_s"] <= cap]
            if len({str(row["fleurs_id"]) for row in candidates}) < size:
                continue
            while True:
                parts = rng.sample(population = candidates, k = size)
                if len({str(row["fleurs_id"]) for row in parts}) == size:
                    break
            break
        if parts is None:
            raise ValueError(f"{lang} 中没有足够短的音频拼成不超过 {max_total_s} 秒的一组")
        groups.append((parts, rng.randrange(start = 0, stop = len(parts))))
    return groups


def request_key(request: dict) -> str:
    """一条请求的唯一标识，续跑时用来跳过已完成的条目。"""
    return "|".join([request["task"], request["instruction_id"], request["prompt_format"], request["item_id"]])


def build_requests(
    manifests: dict,
    tasks,
    prompt_formats,
    per_lang: int,
    seed: int,
    cascade: bool,
    references: dict,
    audio_frames,
    gap_s: float = 0.5,
    max_total_s: float = 30.0,
) -> list:
    """构造全部请求（不读音频）。manifests 为 {语言: 全部清单行}，audio_frames(row) 给出一条音频的采样点数
    （时间段任务按它算出各段的准确起止时间）。
    """
    requests = []
    gap = int(round(gap_s * SAMPLING_RATE))
    for lang, rows in manifests.items():
        picked = select_rows(
            rows = rows,
            count = per_lang,
            seed = seed,
            lang = lang,
        )
        for task in tasks:
            if task == "timestamps":
                continue
            instructions = []
            if "chat" in prompt_formats:
                instructions += [(item, "chat") for item in INSTRUCTIONS[task]]
            if task == "transcribe" and "cpt" in prompt_formats:
                instructions.append((CPT_INSTRUCTION, "cpt"))
            for instruction, prompt_format in instructions:
                target_lang = instruction.get("target_lang")
                if target_lang == lang:
                    continue
                for row in picked:
                    request = {
                        "task": task,
                        "instruction_id": instruction["id"],
                        "instruction": instruction["text"],
                        "prompt_format": prompt_format,
                        "mode": "cascade" if cascade else "audio",
                        "lang": lang,
                        "item_id": row["sample_id"],
                        "sample_ids": [row["sample_id"]],
                        "fleurs_ids": [str(row["fleurs_id"])],
                        "audio_paths": [row["audio_path"]],
                        "duration_s": float(row["duration_s"]),
                        "reference": row["raw_transcription"],
                    }
                    if target_lang is not None:
                        request["target_lang"] = target_lang
                        request["translation_reference"] = references.get(target_lang, {}).get(str(row["fleurs_id"]))
                    requests.append(request)
        if "timestamps" in tasks and not cascade and "chat" in prompt_formats:
            groups = pick_concat_parts(
                rows = rows,
                count = per_lang,
                seed = seed,
                lang = lang,
                gap_s = gap_s,
                max_total_s = max_total_s,
            )
            for number, (parts, target) in enumerate(groups):
                bounds = segment_bounds(lengths = [audio_frames(row) for row in parts], gap = gap)
                start, end = bounds[target]
                for instruction in INSTRUCTIONS["timestamps"]:
                    requests.append({
                        "task": "timestamps",
                        "instruction_id": instruction["id"],
                        "instruction": instruction["text"].format(start = start, end = end),
                        "prompt_format": "chat",
                        "mode": "audio",
                        "lang": lang,
                        "item_id": f"{lang}-concat-{number}",
                        "sample_ids": [row["sample_id"] for row in parts],
                        "fleurs_ids": [str(row["fleurs_id"]) for row in parts],
                        "audio_paths": [row["audio_path"] for row in parts],
                        "duration_s": bounds[-1][1],
                        "gap_s": gap_s,
                        "segment_index": target,
                        "segment_bounds": [list(bound) for bound in bounds],
                        "reference": parts[target]["raw_transcription"],
                        "whole_reference": " ".join(row["raw_transcription"] for row in parts),
                    })
    return requests


# ---------------------------------------------------------------- 打分


def char_error_rate(reference: str, hypothesis: str, lang: str):
    """基础规范化后去掉空白、按字符计算的错误率；参考为空时为 None。所有语言都按字符计算。"""
    ref = tokenize(text = normalize_basic(text = reference, lang = lang), metric = "cer")
    hyp = tokenize(text = normalize_basic(text = hypothesis, lang = lang), metric = "cer")
    if not ref:
        return None
    errors = sum(1 for op in align(ref = ref, hyp = hyp) if op[0] != "ok")
    return errors / len(ref)


def char_ngrams(text: str, order: int) -> Counter:
    """去掉空白后的字符 order 元组计数（chrF 不计空白）。"""
    chars = "".join(text.split())
    return Counter(chars[index:index + order] for index in range(len(chars) - order + 1))


def chrf_statistics(hypothesis: str, reference: str, char_order: int = 6) -> list:
    """每个阶数的 [输出的 n 元组数, 参考的 n 元组数, 匹配数]，语料级 chrF 把各句的统计量相加。"""
    stats = []
    for order in range(1, char_order + 1):
        hyp = char_ngrams(text = hypothesis, order = order)
        ref = char_ngrams(text = reference, order = order)
        stats.append([sum(hyp.values()), sum(ref.values()), sum((hyp & ref).values())])
    return stats


def chrf_score(stats: list, beta: float = 2.0) -> float:
    """chrF（0–100）：输出和参考都有 n 元组的各阶分别算精确率、召回率，取平均后算 F_β。

    与 sacrebleu 默认的 chrF（字符 6 元、β = 2、不计空白、区分大小写）的计算方法相同。
    """
    precisions, recalls = [], []
    for n_hyp, n_ref, n_match in stats:
        if n_hyp > 0 and n_ref > 0:
            precisions.append(n_match / n_hyp)
            recalls.append(n_match / n_ref)
    if not precisions:
        return 0.0
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    if precision + recall == 0:
        return 0.0
    factor = beta ** 2
    return 100 * (1 + factor) * precision * recall / (factor * precision + recall)


def corpus_chrf(pairs: list) -> float | None:
    """[(输出, 参考), ...] 的语料级 chrF；空列表为 None。"""
    if not pairs:
        return None
    total = None
    for hypothesis, reference in pairs:
        stats = chrf_statistics(hypothesis = hypothesis, reference = reference)
        total = stats if total is None else [[a + b for a, b in zip(x, y)] for x, y in zip(total, stats)]
    return chrf_score(stats = total)


def segment_outcome(segment_rate, whole_rate, threshold: float) -> str:
    """时间段任务一条输出的类别：whole（整段拼接音频的转写）、segment（问的这一段）或 other。"""
    if whole_rate is not None and whole_rate < threshold and (segment_rate is None or whole_rate < segment_rate):
        return "whole"
    if segment_rate is not None and segment_rate < threshold:
        return "segment"
    return "other"


def score_item(item: dict, threshold: float) -> dict:
    """给一条结果加上分数（原地修改并返回）。"""
    lang, output = item["lang"], item["output"]
    item["empty"] = not "".join(output.split())
    item["cer_vs_reference"] = char_error_rate(reference = item["reference"], hypothesis = output, lang = lang)
    item["verbatim"] = item["cer_vs_reference"] is not None and item["cer_vs_reference"] < threshold
    if item["task"] == "translate":
        reference = item.get("translation_reference")
        item["chrf"] = None
        if reference is not None:
            item["chrf"] = chrf_score(stats = chrf_statistics(hypothesis = output, reference = reference))
    if item["task"] == "timestamps":
        item["cer_vs_whole"] = char_error_rate(reference = item["whole_reference"], hypothesis = output, lang = lang)
        item["outcome"] = segment_outcome(
            segment_rate = item["cer_vs_reference"],
            whole_rate = item["cer_vs_whole"],
            threshold = threshold,
        )
    return item


def share(items: list, key: str, value = True):
    return (sum(1 for item in items if item.get(key) == value) / len(items)) if items else None


def summarize_group(task: str, items: list) -> dict:
    """一项任务、一条指令的结果按语言汇总，ALL 为各语言数值的无权重平均。"""
    by_lang = {}
    for item in items:
        by_lang.setdefault(item["lang"], []).append(item)
    per_lang = {}
    if task in ("transcribe", "timestamps"):
        rows = [
            {
                "sample_id": item["item_id"],
                "lang": item["lang"],
                "reference": item["reference"],
                "hypothesis": item["output"],
            }
            for item in items
        ]
        _, corpus = score_corpus(rows = rows)
    for lang in [code for code in LANGS if code in by_lang]:
        group = by_lang[lang]
        entry = {
            "n": len(group),
            "verbatim_share": share(items = group, key = "verbatim"),
            "empty_share": share(items = group, key = "empty"),
        }
        if task in ("transcribe", "timestamps"):
            entry["metric"] = METRIC_BY_LANG[lang]
            entry["rate_basic"] = corpus[lang]["basic"]["rate"]
            entry["rate_numeric"] = corpus[lang]["numeric"]["rate"]
        if task == "what_heard":
            entry["descriptive_share"] = share(
                items = [item for item in group if not item["empty"]],
                key = "verbatim",
                value = False,
            )
        if task == "translate":
            pairs = [
                (item["output"], item["translation_reference"])
                for item in group
                if item.get("translation_reference") is not None
            ]
            entry["n_with_reference"] = len(pairs)
            entry["chrf"] = corpus_chrf(pairs = pairs)
        if task == "timestamps":
            for outcome in ("segment", "whole", "other"):
                entry[f"{outcome}_share"] = share(items = group, key = "outcome", value = outcome)
        per_lang[lang] = entry
    overall = {"n": sum(entry["n"] for entry in per_lang.values())}
    keys = sorted({key for entry in per_lang.values() for key, value in entry.items() if isinstance(value, float)})
    for key in keys:
        values = [entry[key] for entry in per_lang.values() if isinstance(entry.get(key), float)]
        overall[key] = sum(values) / len(values) if values else None
    return {"per_lang": per_lang, "ALL": overall}


def summarize(items: list) -> dict:
    """{任务: {指令: {"per_lang": {...}, "ALL": {...}}}}。"""
    groups = {}
    for item in items:
        groups.setdefault(item["task"], {}).setdefault(item["instruction_id"], []).append(item)
    return {
        task: {instruction: summarize_group(task = task, items = group) for instruction, group in instructions.items()}
        for task, instructions in groups.items()
    }


# 每项任务在 summary.md 中列出的数值：(键, 列名, 是否为比例)。
MARKDOWN_COLUMNS = {
    "transcribe": [("rate_basic", "错误率 %", True), ("verbatim_share", "逐字 %", True)],
    "what_heard": [("verbatim_share", "逐字转写 %", True), ("descriptive_share", "描述性回答 %", True)],
    "translate": [("chrf", "chrF", False), ("verbatim_share", "原文转写 %", True)],
    "timestamps": [("rate_basic", "该段错误率 %", True), ("segment_share", "该段 %", True), ("whole_share", "整段 %", True)],
}


def format_markdown(summary: dict) -> str:
    """每项任务一张表：行是语言（最后一行 ALL），列是各指令的主要数值。"""

    def cell(value, percent):
        if value is None:
            return "n/a"
        return f"{value * 100:.1f}" if percent else f"{value:.1f}"

    parts = []
    for task in [name for name in TASKS if name in summary]:
        instructions = summary[task]
        header = ["语言"]
        for instruction in instructions:
            header += [f"{instruction} {title}" for _, title, _ in MARKDOWN_COLUMNS[task]]
        lines = [f"## {task}", "", "| " + " | ".join(header) + " |", "| --- |" + " ---: |" * (len(header) - 1)]
        langs = [code for code in LANGS if any(code in group["per_lang"] for group in instructions.values())]
        for lang in langs + ["ALL"]:
            cells = [lang]
            for group in instructions.values():
                entry = group["ALL"] if lang == "ALL" else group["per_lang"].get(lang, {})
                cells += [cell(value = entry.get(key), percent = percent) for key, _, percent in MARKDOWN_COLUMNS[task]]
            lines.append("| " + " | ".join(cells) + " |")
        parts.append("\n".join(lines) + "\n")
    return "\n".join(parts)


# ---------------------------------------------------------------- 推理


def cascade_prompt(processor, transcript: str, instruction: str) -> str:
    """级联参照的提示：参考转写代替音频，放在指令之前，只有文字。"""
    text = CASCADE_TEMPLATE.format(transcript = transcript, instruction = instruction)
    return processor.apply_chat_template(
        conversation = [{"role": "user", "content": [{"type": "text", "text": text}]}],
        tokenize = False,
        add_generation_prompt = True,
        enable_thinking = False,
    )


def system_messages(audio_path: str, instruction: str) -> list:
    """指令作为系统消息、用户回合只有音频的对话（--instruction-placement system）。"""
    return [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": [{"type": "audio", "audio": audio_path}]},
    ]


def probe_prompt(processor, request: dict, audio_path: str, placement: str = "user") -> str:
    """一条音频请求的提示：cpt 只有音频标记；聊天格式的指令按 placement 放在用户回合中音频之后（user）或系统消息中（system）。"""
    from pipeline.eval.run_eval import build_prompt

    if placement not in PLACEMENTS:
        raise ValueError(f"未知的指令位置：{placement}，可选 {PLACEMENTS}")
    if request["prompt_format"] == "chat" and placement == "system":
        return processor.apply_chat_template(
            conversation = system_messages(audio_path = audio_path, instruction = request["instruction"]),
            tokenize = False,
            add_generation_prompt = True,
            enable_thinking = False,
        )
    return build_prompt(
        processor = processor,
        audio_path = audio_path,
        prompt_format = request["prompt_format"],
        instruction = request["instruction"],
    )


def batch_inputs(processor, requests: list, audio_root: Path, placement: str = "user"):
    """一批请求的模型输入：音频请求读入（时间段任务按 gap_s 拼接）后与提示一起交给 processor；级联请求只有文字。"""
    from pipeline.eval.run_eval import load_waveform

    if requests[0]["mode"] == "cascade":
        prompts = [
            cascade_prompt(processor = processor, transcript = request["reference"], instruction = request["instruction"])
            for request in requests
        ]
        return processor(text = prompts, padding = True, return_tensors = "pt")
    prompts, waveforms = [], []
    for request in requests:
        paths = [str(Path(audio_root) / path) for path in request["audio_paths"]]
        waves = [load_waveform(path = path) for path in paths]
        if len(waves) == 1:
            waveforms.append(waves[0])
        else:
            wave, bounds = concatenate(waveforms = waves, gap = int(round(request["gap_s"] * SAMPLING_RATE)))
            if [list(bound) for bound in bounds] != request["segment_bounds"]:
                raise ValueError(f"{request['item_id']} 读入的音频长度与文件头记录的长度不一致")
            waveforms.append(wave)
        prompts.append(probe_prompt(
            processor = processor,
            request = request,
            audio_path = paths[0],
            placement = placement,
        ))
    return processor(
        text = prompts,
        audio = waveforms,
        sampling_rate = SAMPLING_RATE,
        padding = True,
        return_tensors = "pt",
    )


def audio_frames_reader(audio_root: Path):
    """返回按清单行读取音频采样点数的函数（只读文件头）。"""
    import soundfile as sf

    def frames(row):
        info = sf.info(file = str(Path(audio_root) / row["audio_path"]))
        if info.samplerate != SAMPLING_RATE:
            raise ValueError(f"{row['audio_path']} 采样率为 {info.samplerate}，需要 {SAMPLING_RATE}")
        return info.frames

    return frames


def read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    with open(file = path, mode = "r", encoding = "utf-8") as handle:
        return [json.loads(s = line) for line in handle if line.strip()]


def parse_args(argv: list | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "语音指令遵循探针：同一批音频换不同指令，看模型照做还是一律转写")
    parser.add_argument("--model-dir", required = True, type = Path, help = "完整模型目录（remote code）")
    parser.add_argument("--projector", default = None, type = Path, help = "projector 检查点（safetensors，可选）")
    parser.add_argument("--language-model", default = None, type = Path, help = "全量微调检查点中的 language_model.safetensors（可选）")
    parser.add_argument("--manifest-dir", required = True, type = Path, help = "含 <lang>/test.jsonl 的 FLEURS 评测清单目录")
    parser.add_argument("--audio-root", default = None, type = Path, help = "audio_path 的根目录，默认等于 --manifest-dir")
    parser.add_argument("--output-dir", required = True, type = Path, help = "输出目录")
    parser.add_argument("--langs", nargs = "+", default = LANGS, help = "语言代码，默认全部 18 种")
    parser.add_argument("--tasks", nargs = "+", default = list(TASKS), choices = TASKS, help = "要做的任务，默认全部")
    parser.add_argument(
        "--instruction-set",
        default = "default",
        choices = sorted(INSTRUCTION_SETS),
        help = "指令：default 为原来的简短指令；strict 要求只输出答案（只有 transcribe、translate、timestamps）",
    )
    parser.add_argument(
        "--prompt-format",
        nargs = "+",
        default = ["chat"],
        choices = ("chat", "cpt"),
        help = "chat：聊天格式提问（全部任务）；cpt：只有音频标记的转写（只用于 transcribe）；可以两者都给",
    )
    parser.add_argument("--cascade", action = "store_true", help = "级联参照：参考转写代替音频，只用语言模型回答")
    parser.add_argument(
        "--instruction-placement",
        default = "user",
        choices = PLACEMENTS,
        help = "聊天格式中指令的位置：user 为用户回合中音频之后（默认）；system 为系统消息，用户回合只有音频",
    )
    parser.add_argument("--per-lang", default = 30, type = int, help = "每种语言抽多少条音频（时间段任务也是这么多组）")
    parser.add_argument("--seed", default = 0, type = int, help = "抽样的随机数种子")
    parser.add_argument("--gap-seconds", default = 0.5, type = float, help = "时间段任务中各段之间的静音（秒）")
    parser.add_argument("--concat-max-seconds", default = 30.0, type = float, help = "时间段任务中拼接后音频的最长时长（秒）")
    parser.add_argument("--verbatim-threshold", default = 0.3, type = float, help = "字符错误率低于它算作逐字转写")
    parser.add_argument("--batch-size", default = 16, type = int, help = "批量大小")
    parser.add_argument("--tokens-per-second", default = 12.0, type = float, help = "max_new_tokens 估算：每秒音频的 token 上限")
    parser.add_argument("--extra-tokens", default = 64, type = int, help = "max_new_tokens 估算：额外余量")
    parser.add_argument("--device", default = "cuda", help = "推理设备，默认 cuda（cpu 只用来核对流程）")
    from pipeline.eval import vllm_host

    vllm_host.add_arguments(parser = parser)
    args = parser.parse_args(argv)
    vllm_host.check_arguments(
        parser = parser,
        args = args,
        default_job_dir = args.output_dir / "vllm-job",
        language_model = args.language_model,
    )
    if args.engine == "vllm" and args.cascade:
        parser.error("--cascade 只用 transformers（--engine hf）")
    if args.cascade and args.instruction_placement != "user":
        parser.error("--cascade 只用 --instruction-placement user")
    return args


def generate_vllm(args, pending: list, audio_root: Path) -> tuple:
    """vLLM 引擎：pending 已按 (时长, 请求标识) 排好，每 --batch-size 条一批，max_new_tokens 按批内最长的时长估算（与
    transformers 版相同）。返回 ([(输出, n_new_tokens)]，vLLM 的记录)。"""
    import time

    from pipeline.eval.run_eval import load_processor, vllm_generate_batches

    def log(message):
        print(f"{time.strftime('%H:%M:%S')} {message}", flush = True)

    processor = load_processor(model_dir = args.model_dir)
    batches = []
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        longest = max(request["duration_s"] for request in batch)
        max_new_tokens = int(math.ceil(longest * args.tokens_per_second) + args.extra_tokens)
        group = []
        for request in batch:
            paths = [str((Path(audio_root) / path).resolve()) for path in request["audio_paths"]]
            group.append({
                "prompt": probe_prompt(
                    processor = processor,
                    request = request,
                    audio_path = paths[0],
                    placement = args.instruction_placement,
                ),
                "audio_paths": paths,
                "gap_s": request.get("gap_s"),
                "segment_bounds": request.get("segment_bounds"),
            })
        batches.append((max_new_tokens, group))
    outputs, record = vllm_generate_batches(
        args = args,
        batches = batches,
        processor = processor,
        log = log,
        kind = "probe",
    )
    return [(text, count) for text, count, _, _ in outputs], record


def main(argv: list | None = None) -> int:
    args = parse_args(argv = argv)
    global INSTRUCTIONS
    INSTRUCTIONS = INSTRUCTION_SETS[args.instruction_set]
    missing = [task for task in args.tasks if task not in INSTRUCTIONS]
    if missing:
        raise SystemExit(f"指令集 {args.instruction_set} 没有这些任务：{missing}")
    for lang in args.langs:
        if lang not in METRIC_BY_LANG:
            raise SystemExit(f"不支持的语言代码：{lang}")
    if args.cascade and "cpt" in args.prompt_format:
        raise SystemExit("--cascade 只用聊天格式，不能与 --prompt-format cpt 一起使用")
    audio_root = args.audio_root if args.audio_root is not None else args.manifest_dir
    args.output_dir.mkdir(parents = True, exist_ok = True)

    manifests = {lang: read_rows(manifest_dir = args.manifest_dir, lang = lang) for lang in args.langs}
    references = parallel_references(
        manifest_dir = args.manifest_dir,
        target_langs = [item["target_lang"] for item in INSTRUCTIONS["translate"]],
    )
    requests = build_requests(
        manifests = manifests,
        tasks = args.tasks,
        prompt_formats = args.prompt_format,
        per_lang = args.per_lang,
        seed = args.seed,
        cascade = args.cascade,
        references = references,
        audio_frames = audio_frames_reader(audio_root = audio_root),
        gap_s = args.gap_seconds,
        max_total_s = args.concat_max_seconds,
    )
    generations_path = args.output_dir / "generations.jsonl"
    identity = {
        "model_dir": str(args.model_dir),
        "projector": None if args.projector is None else str(args.projector),
        "language_model": None if args.language_model is None else str(args.language_model),
        "cascade": args.cascade,
    }
    if args.engine == "vllm":
        identity["engine"] = "vllm"
    if args.instruction_placement != "user":
        identity["instruction_placement"] = args.instruction_placement
    identity_path = args.output_dir / "probe_config.json"
    if identity_path.exists():
        saved = json.loads(s = identity_path.read_text(encoding = "utf-8"))
        if saved != identity:
            raise SystemExit(f"{args.output_dir} 中已有另一个模型的结果（{saved}），换一个输出目录")
    else:
        identity_path.write_text(data = json.dumps(obj = identity, ensure_ascii = False, indent = 2), encoding = "utf-8")
    done = {record["key"]: record for record in read_jsonl(path = generations_path)}
    pending = [request for request in requests if request_key(request = request) not in done]
    print(f"共 {len(requests)} 条请求，已完成 {len(requests) - len(pending)} 条", flush = True)

    # vLLM 的设置与用时（写进 summary.json 的 config）；下面逐条结果的变量用别的名字，不覆盖它
    vllm_record = None
    if pending and args.engine == "vllm":
        pending.sort(key = lambda request: (request["duration_s"], request_key(request = request)))
        outputs, vllm_record = generate_vllm(args = args, pending = pending, audio_root = audio_root)
        with open(file = generations_path, mode = "a", encoding = "utf-8") as handle:
            for request, (output, count) in zip(pending, outputs):
                entry = {"key": request_key(request = request), "output": output, "n_new_tokens": count}
                done[entry["key"]] = entry
                handle.write(json.dumps(obj = entry, ensure_ascii = False) + "\n")
        pending = []
    if pending:
        from pipeline.eval.run_eval import generate, load_model

        model, processor = load_model(
            model_dir = args.model_dir,
            projector_path = args.projector,
            language_model_path = args.language_model,
            device = args.device,
        )
        # 按时长排序，同一批的音频长度接近，填充少
        pending.sort(key = lambda request: (request["duration_s"], request_key(request = request)))
        with open(file = generations_path, mode = "a", encoding = "utf-8") as handle:
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start:start + args.batch_size]
                longest = max(request["duration_s"] for request in batch)
                max_new_tokens = int(math.ceil(longest * args.tokens_per_second) + args.extra_tokens)
                outputs, n_new, elapsed, peak = generate(
                    model = model,
                    processor = processor,
                    inputs = batch_inputs(
                        processor = processor,
                        requests = batch,
                        audio_root = audio_root,
                        placement = args.instruction_placement,
                    ),
                    max_new_tokens = max_new_tokens,
                    device = args.device,
                )
                for request, output, count in zip(batch, outputs, n_new):
                    entry = {"key": request_key(request = request), "output": output, "n_new_tokens": count}
                    done[entry["key"]] = entry
                    handle.write(json.dumps(obj = entry, ensure_ascii = False) + "\n")
                handle.flush()
                print(
                    f"{start + len(batch)}/{len(pending)} 条，{elapsed:.1f}s，max_new_tokens {max_new_tokens}，"
                    f"峰值显存 {peak / 2 ** 30:.2f} GiB",
                    flush = True,
                )

    items = []
    for request in requests:
        generated = done[request_key(request = request)]
        item = dict(request)
        item["output"] = generated["output"]
        item["n_new_tokens"] = generated["n_new_tokens"]
        items.append(score_item(item = item, threshold = args.verbatim_threshold))
    with open(file = args.output_dir / "items.jsonl", mode = "w", encoding = "utf-8") as handle:
        for item in items:
            handle.write(json.dumps(obj = item, ensure_ascii = False) + "\n")
    summary = {
        "config": {
            **identity,
            "prompt_format": args.prompt_format,
            "per_lang": args.per_lang,
            "seed": args.seed,
            "verbatim_threshold": args.verbatim_threshold,
            "engine": args.engine,
            "instruction_placement": args.instruction_placement,
            **({"vllm": vllm_record} if vllm_record is not None else {}),
        },
        **summarize(items = items),
    }
    (args.output_dir / "summary.json").write_text(
        data = json.dumps(obj = summary, ensure_ascii = False, indent = 2),
        encoding = "utf-8",
    )
    table = format_markdown(summary = {key: value for key, value in summary.items() if key != "config"})
    (args.output_dir / "summary.md").write_text(data = table, encoding = "utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
