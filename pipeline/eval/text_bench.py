"""文本基准（任务 #71）：看全量微调后的语言模型还保留多少 Qwen3.5-2B 原有的文本能力。

    flock <gpu.lock> python -m pipeline.eval.text_bench mmlu_redux ceval mmmlu ifeval gpqa \\
        --model-dir <融合模型目录> [--language-model <检查点>/language_model.safetensors] \\
        --data-dir $DATASET/text --out-dir <输出目录>/<模型标签>

一次可以做几项基准，模型只载入一次（GPQA 用 FlashAttention-2 重新载入）；输出目录中已有结果的基准跳过。不给
--language-model 时评测参考模型（融合模型中原始的语言模型）。只用语言模型做纯文本的前向和生成，不需要 projector。
数据（--data-dir 下，见 datasets.md「文本基准（#71）」）：

| 基准 | 数据 | 题数 | 模式与打分 |
| --- | --- | ---: | --- |
| mmlu_redux | edinburgh-dawg/mmlu-redux（30 个学科各 100 题） | 2,722 | 不思考，字母对数概率 |
| ceval | ceval/ceval-exam 的 val 划分（52 个学科） | 1,346 | 不思考，字母对数概率 |
| mmmlu | openai/MMMLU 中与项目重合的 10 种语言，每种固定抽 500 题 | 5,000 | 不思考，字母对数概率 |
| ifeval | google/IFEval | 541 | 不思考，贪心生成，官方打分（pipeline/eval/ifeval） |
| gpqa | Idavidrein/gpqa 的 gpqa_diamond.csv | 198 | 思考，温度 1 采样，解析答案字母 |

提示一律用模型自带的 chat template（只有一条用户消息，没有系统消息），add_generation_prompt = True。

选择题（mmlu_redux、ceval、mmmlu）：enable_thinking = False，提示结尾是助手回合开头和空的思考块
`<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`。用户消息为题目、四个选项（`A. …` 到 `D. …`）和「只回答字母」的要求
（英文模板 EN_TEMPLATE 用于 mmlu_redux 与 mmmlu，题目前写明学科；中文模板 ZH_TEMPLATE 用于 ceval）。不生成：取助手回合
第一个位置上全词表的对数概率，比较 A、B、C、D 四个 token（各是一个 token），最大的作为模型的答案（零样本）。另记下四个
字母在全词表中的概率之和（letter_mass），它很低时说明模型不按「只回答字母」作答，准确率要谨慎看待。前向只用语言模型
（与 replay_gap 相同）：按长度排序、凑成不超过 --max-batch-tokens 个 token 的批，右侧填充、不用注意力掩码（因果注意力和
线性注意力都只看前文），只在每条最后一个位置上算词表投影（float32）。

- mmlu_redux：MMLU-Redux 第一版的 3,000 题中，数据集把 error_type 标为 ok 以外（题目不清楚、选项不清楚、没有正确答案、
  多个正确答案、标准答案错误、需要专家判断）的 278 题排除，主指标是其余 2,722 题的准确率；全部 3,000 题按原标准答案的
  准确率另记在 all 中。模型卡（非思考 69.2）没有说明题目范围。
- ceval：测试集的答案不公开，用验证集（1,346 题）；模型卡的 65.2 很可能是测试集上的结果，两者不能直接比较。主指标是
  按题的准确率，另给 52 个学科准确率的平均（macro_accuracy）。
- mmmlu：10 种语言的 MMLU 测试集译本（ar、de、es、fr、id、it、ja、ko、pt、zh；文件按行对应同一道题），用 MMMLU_SEED
  从 14,042 行中抽同一组 500 个行号，各语言用同样的题，标准答案取各语言文件自己的（少数行的译本调整了选项顺序）。
  主指标是 10 种语言准确率的平均；模型卡的 56.9 是全部 14 种语言的结果，只能大致参照。

IFEval：enable_thinking = False，贪心解码，最多 --ifeval-max-new-tokens（默认 1,280）个新 token，在 <|im_end|> 或
<|endoftext|> 停止；左填充分批（按剩余显存估算一批放得下几条，与思考检查的 --fit-memory 相同），全注意力层用 sdpa。回答
去掉思考块与特殊标记后写入 ifeval.responses.jsonl（key、prompt、response、生成长度、是否写到上限），统计写入
ifeval.generation.json。打分由 pipeline.eval.ifeval.score 在另一个 python 中进行（需要 langdetect 与 nltk），驱动程序
bench_local.py 会接着做；结果为 ifeval.json（prompt 级与 instruction 级，strict 与 loose；模型卡的 61.2 对应 prompt 级
strict）。

GPQA Diamond（思考模式）：数据的使用条款要求不在网上公开题目，所以结果文件只有题目的 Record ID、选项顺序、答案字母与
对错，日志不打印题目和模型的回答；生成的 token id 只存在本机的 gpqa.ids.i32 中。每题四个选项（正确答案与三个错误答案）
按 GPQA_OPTION_SEED 与 Record ID 固定打乱（items 中的 permutation 为显示在 A–D 位置上的原选项编号，0 是正确答案）。
用户消息为题目、四个选项和模型卡对选择题建议的格式要求（在 answer 字段中只写字母）；enable_thinking = True，提示结尾是
`<|im_start|>assistant\\n<think>\\n`。生成设置与思考检查（think_check）相同：温度 1，不设 top-k、top-p（模型卡用
top_p 0.95、top_k 20、presence_penalty 1.5），每批开始前按 --gpqa-seed 加批序号固定随机数，每题只采样一次；最多
--gpqa-max-new-tokens（默认 16,384，与思考检查相同；模型卡建议大多数提问用 32,768）个新 token；全注意力层用
FlashAttention-2（think_check.FLASH_ATTENTION），载入或生成出错时整项改用 sdpa，批大小按剩余显存估算（与思考检查的
--fit-memory 相同，RTX 5090 上 3 批、每批 66 题），显存不足时对半拆开重试；题目按提示长度排序后分批，减少左填充。答案只在 </think> 之后找：`"answer": "X"`（键可以不带引号）与
`\\boxed{X}` 两种写法中位置最靠后的一个。没有写完 </think>（unclosed）或者找不到答案（no_answer）都算错，单独报告。
主指标是 198 题的准确率（模型卡思考模式 51.6）；每题只采样一次，198 题的标准误约 3.5 个百分点。GPQA 也可以用 vLLM 的连续
批处理生成（pipeline/eval/gpqa_vllm.py）：题目、提示、答案解析、指标和输出文件都调用这里的函数（gpqa_rows、gpqa_prompts、
write_gpqa），只有生成和随机数的分配不同。

输出（--out-dir）：每项基准 <基准>.json（汇总）与 <基准>.items.jsonl（逐题：编号、标准答案、模型答案、四个字母的对数概率
或 GPQA 的状态），IFEval 另有 responses 与 generation 两个文件，GPQA 另有 gpqa.ids.i32。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
import traceback
from pathlib import Path

BENCHES = ("mmlu_redux", "ceval", "mmmlu", "ifeval", "gpqa")
MC_BENCHES = ("mmlu_redux", "ceval", "mmmlu")
LETTERS = ("A", "B", "C", "D")
# 各基准在 --data-dir（$DATASET/text）下的位置。
DATA = {
    "mmlu_redux": "edinburgh-dawg/mmlu-redux",
    "ceval": "ceval/ceval-exam",
    "mmmlu": "openai/MMMLU/test",
    "ifeval": "google/IFEval/ifeval_input_data.jsonl",
    "gpqa": "Idavidrein/gpqa/gpqa_diamond.csv",
}
# MMMLU 中与项目 18 种语言重合的语言及其文件名。
MMMLU_FILES = {
    "ar": "AR-XY",
    "de": "DE-DE",
    "es": "ES-LA",
    "fr": "FR-FR",
    "id": "ID-ID",
    "it": "IT-IT",
    "ja": "JA-JP",
    "ko": "KO-KR",
    "pt": "PT-BR",
    "zh": "ZH-CN",
}
MMMLU_PER_LANG = 500
MMMLU_SEED = 71
GPQA_OPTION_SEED = 71
# Qwen3.5-2B 模型卡（weights/Qwen3.5-2B/README.md）上的分数，只作参照：前四项为非思考模式，GPQA 为思考模式。
CARD = {"mmlu_redux": 69.2, "ceval": 65.2, "mmmlu": 56.9, "ifeval": 61.2, "gpqa": 51.6}

EN_TEMPLATE = (
    "The following is a multiple choice question{about}.\n\n{question}\n\n"
    "A. {A}\nB. {B}\nC. {C}\nD. {D}\n\n"
    "Answer with only the letter (A, B, C, or D) of the correct option."
)
ZH_TEMPLATE = (
    "以下是一道单项选择题，请选出正确答案。\n\n{question}\n\n"
    "A. {A}\nB. {B}\nC. {C}\nD. {D}\n\n"
    "只回答正确选项的字母（A、B、C 或 D）。"
)
GPQA_TEMPLATE = (
    "{question}\n\n"
    "A. {A}\nB. {B}\nC. {C}\nD. {D}\n\n"
    "Please show your choice in the `answer` field with only the choice letter, e.g., `\"answer\": \"C\"`."
)
# GPQA 的答案：`"answer": "C"`（键的引号可有可无，冒号可以是全角）或 \boxed{C}（可带 \text{}），取 </think> 之后最靠后的一个。
ANSWER_FIELD = re.compile(pattern = r"""(?:["']?(?i:answer)["']?)\s*[:：]\s*["'(\[（]?\s*([ABCD])\b""")
BOXED = re.compile(pattern = r"\\boxed\{\s*(?:\\text(?:bf)?\{\s*)?\(?([ABCD])\)?")


# ---------------------------------------------------------------- 数据


def spread(rows: list, limit: int | None) -> list:
    """limit 为 None 时原样返回；否则在 rows 中等距取 limit 条（冒烟测试用，各学科都取到一些）。"""
    if limit is None or limit >= len(rows):
        return list(rows)
    return [rows[math.floor(index * len(rows) / limit)] for index in range(limit)]


def load_mmlu_redux(root: Path) -> list:
    """MMLU-Redux 全部 3,000 题：[{id, group（学科）, question, options, gold, error_type}]，按学科名、题号排列。"""
    import pyarrow as pa

    rows = []
    subjects = sorted(path.parent.name for path in root.glob("*/data-00000-of-00001.arrow"))
    for subject in subjects:
        with pa.memory_map(str(root / subject / "data-00000-of-00001.arrow")) as source:
            table = pa.ipc.open_stream(source).read_all()
        for index, row in enumerate(table.to_pylist()):
            if len(row["choices"]) != 4:
                raise ValueError(f"{subject} 第 {index} 题有 {len(row['choices'])} 个选项")
            rows.append({
                "id": f"{subject}/{index}",
                "group": subject,
                "subject": subject,
                "question": row["question"],
                "options": list(row["choices"]),
                "gold": LETTERS[int(row["answer"])],
                "error_type": row["error_type"],
            })
    return rows


def load_ceval(root: Path) -> list:
    """C-Eval 验证集全部 1,346 题：[{id, group, question, options, gold}]。"""
    import pyarrow.parquet as pq

    rows = []
    for path in sorted(root.glob("*/val-00000-of-00001.parquet")):
        subject = path.parent.name
        for row in pq.read_table(source = path).to_pylist():
            rows.append({
                "id": f"{subject}/{row['id']}",
                "group": subject,
                "question": row["question"],
                "options": [row[letter] for letter in LETTERS],
                "gold": row["answer"].strip(),
            })
    return rows


def read_csv(path: Path) -> list:
    """按行读 CSV（全部当作字符串，"None"、"NA" 之类不转成空值）。"""
    with open(file = path, encoding = "utf-8", newline = "") as handle:
        return list(csv.DictReader(handle))


def mmmlu_indices(count: int, per_lang: int = MMMLU_PER_LANG, seed: int = MMMLU_SEED) -> list:
    """MMMLU 抽题的行号（升序），各语言相同。"""
    return sorted(random.Random(seed).sample(population = range(count), k = per_lang))


def load_mmmlu(root: Path, langs: list, per_lang: int = MMMLU_PER_LANG, seed: int = MMMLU_SEED) -> list:
    """MMMLU 各语言同一组行号的题：[{id（语言/行号）, group（语言）, subject, question, options, gold}]。"""
    tables = {lang: read_csv(path = root / f"mmlu_{MMMLU_FILES[lang]}.csv") for lang in langs}
    counts = {len(table) for table in tables.values()}
    if len(counts) != 1:
        raise ValueError(f"MMMLU 各语言的行数不同：{ {lang: len(table) for lang, table in tables.items()} }")
    indices = mmmlu_indices(count = counts.pop(), per_lang = per_lang, seed = seed)
    rows = []
    for lang, table in tables.items():
        for index in indices:
            row = table[index]
            rows.append({
                "id": f"{lang}/{index}",
                "group": lang,
                "subject": row["Subject"],
                "question": row["Question"],
                "options": [row[letter] for letter in LETTERS],
                "gold": row["Answer"].strip(),
            })
    return rows


def gpqa_permutation(record_id: str, seed: int = GPQA_OPTION_SEED) -> list:
    """GPQA 一题的选项顺序：显示在 A–D 位置上的原选项编号（0 为正确答案，1–3 为三个错误答案）。"""
    return random.Random(f"{seed}:{record_id}").sample(population = range(4), k = 4)


def load_gpqa(path: Path, seed: int = GPQA_OPTION_SEED) -> list:
    """GPQA Diamond 全部题目：[{id（Record ID）, question, options（已打乱）, permutation, gold}]。"""
    rows = []
    for row in read_csv(path = path):
        original = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        permutation = gpqa_permutation(record_id = row["Record ID"], seed = seed)
        rows.append({
            "id": row["Record ID"],
            "question": row["Question"].strip(),
            "options": [original[index].strip() for index in permutation],
            "permutation": permutation,
            "gold": LETTERS[permutation.index(0)],
        })
    return rows


# ---------------------------------------------------------------- 提示与打分


def mc_text(row: dict, template: str) -> str:
    """一道选择题的用户消息。英文模板在有学科名（subject）时写成「… question about <学科>.」，中文模板不写学科。"""
    about = f" about {row['subject'].replace('_', ' ')}" if row.get("subject") else ""
    fields = {letter: str(option).strip() for letter, option in zip(LETTERS, row["options"])}
    return template.format(about = about, question = str(row["question"]).strip(), **fields)


def gpqa_text(row: dict) -> str:
    """GPQA 一题的用户消息（选项已按 permutation 打乱）。"""
    return GPQA_TEMPLATE.format(question = row["question"], **dict(zip(LETTERS, row["options"])))


def chat_ids(tokenizer, text: str, thinking: bool) -> list:
    """用户消息按 chat template 展开（加助手回合开头）后的 token id。"""
    prompt = tokenizer.apply_chat_template(
        conversation = [{"role": "user", "content": text}],
        tokenize = False,
        add_generation_prompt = True,
        enable_thinking = thinking,
    )
    return tokenizer(text = prompt, add_special_tokens = False)["input_ids"]


def letter_ids(tokenizer) -> list:
    """A–D 的 token id；每个字母必须正好是一个 token。"""
    ids = [tokenizer.encode(text = letter, add_special_tokens = False) for letter in LETTERS]
    if any(len(item) != 1 for item in ids):
        raise ValueError(f"字母不是单个 token：{ids}")
    return [item[0] for item in ids]


def letter_logprobs(model, prompts: list, letters: list, max_batch_tokens: int) -> list:
    """每条提示最后一个位置上 A–D 的全词表对数概率（float32），按 prompts 的顺序。"""
    import torch

    from pipeline.eval.replay_gap import PAD_ID, batches_by_tokens

    device = next(model.parameters()).device
    letter_index = torch.tensor(data = letters, device = device)
    out = [None] * len(prompts)
    for indices in batches_by_tokens(lengths = [len(ids) for ids in prompts], max_batch_tokens = max_batch_tokens):
        longest = max(len(prompts[index]) for index in indices)
        input_ids = torch.full(size = (len(indices), longest), fill_value = PAD_ID, dtype = torch.long)
        for row, index in enumerate(indices):
            input_ids[row, : len(prompts[index])] = torch.tensor(data = prompts[index])
        last = torch.tensor(data = [len(prompts[index]) - 1 for index in indices], device = device)
        with torch.inference_mode():
            hidden = model.model.language_model(input_ids = input_ids.to(device = device), use_cache = False).last_hidden_state
            picked = hidden[torch.arange(len(indices), device = device), last]
            logits = torch.nn.functional.linear(input = picked, weight = model.lm_head.weight).float()
            chosen = torch.log_softmax(input = logits, dim = -1)[:, letter_index].cpu()
        for row, index in enumerate(indices):
            out[index] = [float(value) for value in chosen[row]]
    return out


def mc_summary(rows: list, logprobs: list) -> tuple:
    """选择题的逐题结果与总体统计 (items, stats)：准确率、四个字母的概率和的平均、答案分布、按组（学科或语言）的准确率。"""
    items = []
    for row, values in zip(rows, logprobs):
        pred = LETTERS[max(range(4), key = lambda index: values[index])]
        item = {
            "id": row["id"],
            "group": row["group"],
            "gold": row["gold"],
            "pred": pred,
            "correct": pred == row["gold"],
            "lp": [round(number = value, ndigits = 4) for value in values],
        }
        if "error_type" in row:
            item["error_type"] = row["error_type"]
        items.append(item)
    return items, mc_stats(items = items)


def mc_stats(items: list) -> dict:
    if not items:
        return {"n": 0, "accuracy": None}
    groups = {}
    for item in items:
        groups.setdefault(item["group"], []).append(item["correct"])
    return {
        "n": len(items),
        "accuracy": round(number = sum(item["correct"] for item in items) / len(items), ndigits = 4),
        "letter_mass_mean": round(number = sum(sum(math.exp(value) for value in item["lp"]) for item in items) / len(items), ndigits = 4),
        "pred_share": {letter: round(number = sum(item["pred"] == letter for item in items) / len(items), ndigits = 4) for letter in LETTERS},
        "groups": {name: {"n": len(values), "accuracy": round(number = sum(values) / len(values), ndigits = 4)} for name, values in sorted(groups.items())},
    }


def run_mc(bench: str, model, tokenizer, data_dir: Path, limit: int | None, max_batch_tokens: int, log) -> tuple:
    """一项选择题基准，返回 (汇总, 逐题结果)。"""
    root = data_dir / DATA[bench]
    if bench == "mmlu_redux":
        rows = spread(rows = load_mmlu_redux(root = root), limit = limit)
        template = EN_TEMPLATE
    elif bench == "ceval":
        rows = spread(rows = load_ceval(root = root), limit = limit)
        template = ZH_TEMPLATE
    else:
        langs = list(MMMLU_FILES)
        rows = load_mmmlu(root = root, langs = langs)
        if limit is not None:
            rows = [row for lang in langs for row in spread(rows = [r for r in rows if r["group"] == lang], limit = limit)]
        template = EN_TEMPLATE
    start = time.perf_counter()
    prompts = [chat_ids(tokenizer = tokenizer, text = mc_text(row = row, template = template), thinking = False) for row in rows]
    logprobs = letter_logprobs(
        model = model,
        prompts = prompts,
        letters = letter_ids(tokenizer = tokenizer),
        max_batch_tokens = max_batch_tokens,
    )
    items, stats = mc_summary(rows = rows, logprobs = logprobs)
    summary = {"bench": bench, "mode": "non-thinking, letter logits", "prompt_tokens": sum(len(ids) for ids in prompts)}
    if bench == "mmlu_redux":
        kept = [item for item in items if item["error_type"] == "ok"]
        excluded = {}
        for item in items:
            if item["error_type"] != "ok":
                excluded[item["error_type"]] = excluded.get(item["error_type"], 0) + 1
        summary.update(mc_stats(items = kept))
        summary["all"] = {key: stats[key] for key in ("n", "accuracy", "letter_mass_mean")}
        summary["excluded"] = dict(sorted(excluded.items()))
    elif bench == "ceval":
        summary.update(stats)
        summary["macro_accuracy"] = round(number = sum(group["accuracy"] for group in stats["groups"].values()) / len(stats["groups"]), ndigits = 4)
        summary["split"] = "val"
    else:
        summary.update(stats)
        summary["micro_accuracy"] = summary["accuracy"]
        summary["accuracy"] = round(number = sum(group["accuracy"] for group in stats["groups"].values()) / len(stats["groups"]), ndigits = 4)
        summary["per_lang_sample"] = {"per_lang": MMMLU_PER_LANG if limit is None else limit, "seed": MMMLU_SEED}
    summary["seconds"] = round(number = time.perf_counter() - start, ndigits = 1)
    summary["card"] = CARD[bench]
    log(f"{bench}：{summary['n']} 题，准确率 {summary['accuracy']:.4f}，字母概率和平均 {summary['letter_mass_mean']:.3f}，{summary['seconds']} 秒")
    return summary, items


# ---------------------------------------------------------------- 生成


def greedy_batch(model, prompts, max_new_tokens, temperature, seed):
    """左填充贪心生成一批（temperature、seed 不用，只为与 think_check.generate_batch 的参数相同），返回每条新生成的 token
    （到第一个结束符为止，含结束符）。"""
    import numpy as np
    import torch

    from pipeline.eval.think_check import PAD_ID, STOP_IDS, left_padded

    device = next(model.parameters()).device
    input_ids, attention_mask, longest = left_padded(prompts = prompts)
    with torch.inference_mode():
        output = model.generate(
            input_ids = input_ids.to(device = device),
            attention_mask = attention_mask.to(device = device),
            mm_token_type_ids = torch.zeros_like(input = input_ids).to(device = device),
            max_new_tokens = max_new_tokens,
            do_sample = False,
            use_cache = True,
            pad_token_id = PAD_ID,
            eos_token_id = list(STOP_IDS),
        )
    results = []
    for row in output[:, longest:].cpu().numpy():
        stops = np.nonzero(np.isin(element = row, test_elements = STOP_IDS))[0]
        results.append(row[: stops[0] + 1] if len(stops) else row)
    return results


def fitted_batch_size(model, prompts: list, max_new_tokens: int, attention: str, requested: int, reserve_gb: float, log) -> int:
    """按剩余显存估算一批放得下几条（与 think_check 的 --fit-memory 相同）。"""
    import torch

    from pipeline.eval.think_check import fit_batch_size, per_sample_bytes

    free_bytes, _ = torch.cuda.mem_get_info()
    sample_bytes = per_sample_bytes(
        config = model.config.text_config,
        total_tokens = max(len(ids) for _, ids in prompts) + max_new_tokens,
        attn_implementation = attention,
    )
    size = fit_batch_size(
        requested = requested,
        count = len(prompts),
        sample_bytes = sample_bytes,
        free_bytes = free_bytes,
        reserve_bytes = int(reserve_gb * 2 ** 30),
    )
    log(f"剩余显存 {free_bytes / 2 ** 30:.1f} GiB，{attention} 每条约 {sample_bytes / 2 ** 30:.3f} GiB，批大小 {size}")
    return size


def run_ifeval(model, tokenizer, data_dir: Path, out_dir: Path, args, log) -> None:
    """IFEval 的贪心生成，写 ifeval.responses.jsonl 与 ifeval.generation.json（打分见 pipeline.eval.ifeval.score）。"""
    from pipeline.eval.run_eval import clean_output
    from pipeline.eval.think_check import STOP_IDS, generate_all

    with open(file = data_dir / DATA["ifeval"], encoding = "utf-8") as handle:
        rows = [json.loads(s = line) for line in handle if line.strip()]
    if args.ifeval_limit is not None:
        rows = rows[: args.ifeval_limit]
    prompts = [(row["key"], chat_ids(tokenizer = tokenizer, text = row["prompt"], thinking = False)) for row in rows]
    batch_size = fitted_batch_size(
        model = model,
        prompts = prompts,
        max_new_tokens = args.ifeval_max_new_tokens,
        attention = "sdpa",
        requested = len(prompts),
        reserve_gb = args.memory_reserve_gb,
        log = log,
    )
    start = time.perf_counter()
    outputs, batches = generate_all(
        model = model,
        prompts = prompts,
        batch_size = batch_size,
        max_new_tokens = args.ifeval_max_new_tokens,
        temperature = 0.0,
        seed = 0,
        generate = greedy_batch,
        log = log,
    )
    seconds = round(number = time.perf_counter() - start, ndigits = 1)
    lines = []
    capped = 0
    for row, ids in zip(rows, outputs):
        ended = len(ids) > 0 and int(ids[-1]) in STOP_IDS
        is_capped = not ended and len(ids) >= args.ifeval_max_new_tokens
        capped += is_capped
        text = clean_output(text = tokenizer.decode(token_ids = [int(token) for token in ids], skip_special_tokens = True))
        lines.append(json.dumps(
            obj = {
                "key": row["key"],
                "prompt": row["prompt"],
                "response": text,
                "n_new_tokens": int(len(ids)),
                "capped": bool(is_capped),
            },
            ensure_ascii = False,
        ))
    generation = {
        "n": len(rows),
        "decoding": "greedy",
        "max_new_tokens": args.ifeval_max_new_tokens,
        "attn_implementation": "sdpa",
        "batch_size": batch_size,
        "batches": batches,
        "seconds": seconds,
        "capped_count": int(capped),
    }
    (out_dir / "ifeval.generation.json").write_text(data = json.dumps(obj = generation, indent = 1) + "\n", encoding = "utf-8")
    write_atomic(path = out_dir / "ifeval.responses.jsonl", text = "\n".join(lines) + "\n")
    log(f"ifeval：{len(rows)} 条，写到上限 {capped} 条，{seconds} 秒")


def parse_choice(text: str) -> str | None:
    """</think> 之后的回答中最靠后的一个答案字母（`"answer": "X"` 或 \\boxed{X}），没有时为 None。"""
    found = [(match.start(), match.group(1)) for pattern in (ANSWER_FIELD, BOXED) for match in pattern.finditer(text)]
    return max(found)[1] if found else None


def gpqa_rows(data_dir: Path, limit: int | None) -> list:
    """GPQA Diamond 的题目（选项已打乱），limit 不为空时只取前这么多题（按 gpqa_diamond.csv 的顺序）。"""
    rows = load_gpqa(path = data_dir / DATA["gpqa"])
    return rows if limit is None else rows[:limit]


def gpqa_prompts(tokenizer, rows: list) -> list:
    """每题的提示：[(Record ID, 提示 token id)]，chat template 展开、enable_thinking = True。"""
    return [(row["id"], chat_ids(tokenizer = tokenizer, text = gpqa_text(row = row), thinking = True)) for row in rows]


def write_gpqa(rows: list, prompts: list, outputs: list, tokenizer, max_new_tokens: int, out_dir: Path, settings: dict) -> dict:
    """解析每题的答案、统计指标，写 gpqa.ids.i32、gpqa.items.jsonl 与 gpqa.json（最后写，作为做完的标志），返回汇总。

    outputs 与 rows、prompts 按同样的顺序，是每题新生成的 token（到第一个结束符为止，含结束符）；settings 是生成设置
    （温度、种子、引擎与批的划分等），按给出的顺序写在汇总的答案分布之后。
    """
    import numpy as np

    from pipeline.eval import think_check

    items = []
    generated = []
    offset = 0
    for row, (_, prompt), ids in zip(rows, prompts, outputs):
        stats = think_check.analyze(ids = ids, max_new_tokens = max_new_tokens)
        pred = None
        if stats["closed"]:
            ends = np.nonzero(np.asarray(a = ids) == think_check.THINK_END_ID)[0]
            answer = tokenizer.decode(token_ids = [int(token) for token in ids[int(ends[0]) + 1:]], skip_special_tokens = True)
            pred = parse_choice(text = answer)
        status = "unclosed" if not stats["closed"] else ("no_answer" if pred is None else "answered")
        items.append({
            "id": row["id"],
            "permutation": row["permutation"],
            "gold": row["gold"],
            "pred": pred,
            "correct": pred == row["gold"],
            "status": status,
            "prompt_tokens": len(prompt),
            "offset": offset,
            **stats,
        })
        offset += len(ids)
        generated.append(np.asarray(a = ids, dtype = np.int32))
    np.concatenate(generated).astype(np.int32).tofile(out_dir / "gpqa.ids.i32")
    write_items(path = out_dir / "gpqa.items.jsonl", items = items)
    count = len(items)
    answered = [item for item in items if item["status"] == "answered"]
    summary = {
        "bench": "gpqa",
        "mode": "thinking, sampled",
        "n": count,
        "accuracy": round(number = sum(item["correct"] for item in items) / count, ndigits = 4),
        "answered": round(number = len(answered) / count, ndigits = 4),
        "accuracy_answered": round(number = sum(item["correct"] for item in answered) / len(answered), ndigits = 4) if answered else None,
        "unclosed": round(number = sum(item["status"] == "unclosed" for item in items) / count, ndigits = 4),
        "no_answer": round(number = sum(item["status"] == "no_answer" for item in items) / count, ndigits = 4),
        "pred_share": {letter: round(number = sum(item["pred"] == letter for item in items) / count, ndigits = 4) for letter in LETTERS},
        **settings,
        "card": CARD["gpqa"],
        **think_check.summarize(rows = items, max_new_tokens = max_new_tokens),
    }
    write_atomic(path = out_dir / "gpqa.json", text = json.dumps(obj = summary, ensure_ascii = False, indent = 1) + "\n")
    return summary


def gpqa_log_line(summary: dict, engine: str) -> str:
    return (
        f"gpqa：{summary['n']} 题，准确率 {summary['accuracy']:.4f}，没写完 </think> {summary['unclosed']:.3f}，"
        f"找不到答案 {summary['no_answer']:.3f}，{summary['seconds']} 秒（{engine}）"
    )


def run_gpqa(args, data_dir: Path, out_dir: Path, log) -> None:
    """GPQA Diamond（思考模式），写 gpqa.json、gpqa.items.jsonl 与 gpqa.ids.i32（见开头）。"""
    import torch

    from pipeline.eval import think_check

    rows = gpqa_rows(data_dir = data_dir, limit = args.gpqa_limit)

    def load(attention):
        return think_check.load_for_attention(
            model_dir = args.model_dir,
            language_model = args.language_model,
            attention = attention,
        )

    tokenizer = tokenizer_of(model_dir = args.model_dir)
    prompts = gpqa_prompts(tokenizer = tokenizer, rows = rows)

    def choose_batch_size(model, attention):
        if not torch.cuda.is_available():
            return args.gpqa_batch_size
        return fitted_batch_size(
            model = model,
            prompts = prompts,
            max_new_tokens = args.gpqa_max_new_tokens,
            attention = attention,
            requested = args.gpqa_batch_size,
            reserve_gb = args.memory_reserve_gb,
            log = log,
        )

    # 提示长度相差很大（中位数约 218 个 token，最长约 2,800），按长度排序后再分批，同一批左填充得少；结果按原顺序写出
    order = sorted(range(len(prompts)), key = lambda index: (len(prompts[index][1]), index))
    result = think_check.think_generate(
        load = load,
        prompts = [prompts[index] for index in order],
        requested = args.gpqa_attention,
        choose_batch_size = choose_batch_size,
        max_new_tokens = args.gpqa_max_new_tokens,
        temperature = args.gpqa_temperature,
        seed = args.gpqa_seed,
        log = log,
    )
    outputs = [None] * len(prompts)
    for position, index in enumerate(order):
        outputs[index] = result["outputs"][position]
    settings = {
        "temperature": args.gpqa_temperature,
        "seed": args.gpqa_seed,
        "option_seed": GPQA_OPTION_SEED,
        "attn_implementation_requested": args.gpqa_attention,
        "attn_implementation_used": result["used"],
        "attn_fallback_reason": result["reason"],
        "batch_size": result["batch_size"],
        "batches": result["batches"],
        "seconds": result["seconds"],
    }
    summary = write_gpqa(
        rows = rows,
        prompts = prompts,
        outputs = outputs,
        tokenizer = tokenizer,
        max_new_tokens = args.gpqa_max_new_tokens,
        out_dir = out_dir,
        settings = settings,
    )
    log(gpqa_log_line(summary = summary, engine = result["used"]))


# ---------------------------------------------------------------- 输出与入口


def write_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(data = text, encoding = "utf-8")
    temporary.replace(target = path)


def write_items(path: Path, items: list) -> None:
    write_atomic(path = path, text = "".join(json.dumps(obj = item, ensure_ascii = False) + "\n" for item in items))


def marker(bench: str, out_dir: Path) -> Path:
    """一项基准做完的标志文件：IFEval 为生成结果（打分在别处做），其余为汇总 JSON。"""
    return out_dir / ("ifeval.responses.jsonl" if bench == "ifeval" else f"{bench}.json")


def tokenizer_of(model_dir: Path):
    """只载入分词器（检查 #77 的分词规则），用于构造提示和解码。"""
    from transformers import AutoProcessor

    from pipeline.tokenizer_fix import check_tokenizer

    processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path = str(model_dir), trust_remote_code = True)
    check_tokenizer(tokenizer = processor, source = model_dir)
    return processor.tokenizer


def parse_args(argv: list | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("benches", nargs = "+", choices = BENCHES, help = "要做的基准，按给出的顺序做")
    parser.add_argument("--model-dir", required = True, type = Path, help = "融合模型目录")
    parser.add_argument(
        "--language-model",
        default = None,
        type = Path,
        help = "全量微调检查点中的 language_model.safetensors；不给时是参考模型",
    )
    parser.add_argument("--data-dir", required = True, type = Path, help = "文本数据集目录（$DATASET/text）")
    parser.add_argument("--out-dir", required = True, type = Path, help = "这个模型的输出目录")
    parser.add_argument("--label", default = None, help = "模型标签（写进汇总）")
    parser.add_argument("--limit", default = None, type = int, help = "选择题每项（MMMLU 每种语言）只做等距抽出的这么多题，冒烟测试用")
    parser.add_argument("--max-batch-tokens", default = 32768, type = int, help = "选择题前向每批填充后的 token 上限")
    parser.add_argument("--ifeval-limit", default = None, type = int, help = "IFEval 只做前这么多条")
    parser.add_argument("--ifeval-max-new-tokens", default = 1280, type = int)
    parser.add_argument("--gpqa-limit", default = None, type = int, help = "GPQA 只做前这么多题")
    parser.add_argument("--gpqa-max-new-tokens", default = 16384, type = int)
    parser.add_argument("--gpqa-batch-size", default = 198, type = int, help = "GPQA 每批最多条数（再按剩余显存减小）")
    parser.add_argument("--gpqa-temperature", default = 1.0, type = float)
    parser.add_argument("--gpqa-seed", default = 1234, type = int)
    parser.add_argument(
        "--gpqa-attention",
        default = "flash_attention_2",
        choices = ("sdpa", "flash_attention_2"),
        help = "GPQA 时语言模型全注意力层的实现，出错时自动改用 sdpa",
    )
    parser.add_argument("--memory-reserve-gb", default = 2.0, type = float, help = "按剩余显存估算批大小时留出的显存（GiB）")
    parser.add_argument("--force", action = "store_true", help = "已有结果也重做")
    return parser.parse_args(args = argv)


def main(argv: list | None = None) -> int:
    args = parse_args(argv = argv)
    args.out_dir.mkdir(parents = True, exist_ok = True)

    def log(message):
        print(f"{time.strftime('%H:%M:%S')} {message}", flush = True)

    missing = [bench for bench in args.benches if not (args.data_dir / DATA[bench]).exists()]
    if missing:
        log("缺少数据：" + "，".join(f"{bench}（{args.data_dir / DATA[bench]}）" for bench in missing))
        return 2
    pending = [bench for bench in dict.fromkeys(args.benches) if args.force or not marker(bench = bench, out_dir = args.out_dir).exists()]
    log(f"模型 {args.label or args.language_model or args.model_dir}；要做 {pending or '无'}")
    failed = []
    model = None
    tokenizer = None
    for bench in pending:
        start = time.perf_counter()
        try:
            if bench == "gpqa":
                if model is not None:
                    from pipeline.eval.think_check import release_memory

                    model = None
                    release_memory()
                run_gpqa(args = args, data_dir = args.data_dir, out_dir = args.out_dir, log = log)
                continue
            if model is None:
                from pipeline.eval.run_eval import load_model

                model, processor = load_model(
                    model_dir = args.model_dir,
                    projector_path = None,
                    language_model_path = args.language_model,
                )
                tokenizer = processor.tokenizer
                log(f"载入模型 {time.perf_counter() - start:.0f} 秒")
            if bench == "ifeval":
                run_ifeval(
                    model = model,
                    tokenizer = tokenizer,
                    data_dir = args.data_dir,
                    out_dir = args.out_dir,
                    args = args,
                    log = log,
                )
                continue
            summary, items = run_mc(
                bench = bench,
                model = model,
                tokenizer = tokenizer,
                data_dir = args.data_dir,
                limit = args.limit,
                max_batch_tokens = args.max_batch_tokens,
                log = log,
            )
            summary["model"] = args.label or str(args.language_model or args.model_dir)
            write_items(path = args.out_dir / f"{bench}.items.jsonl", items = items)
            write_atomic(path = marker(bench = bench, out_dir = args.out_dir), text = json.dumps(obj = summary, ensure_ascii = False, indent = 1) + "\n")
        except Exception:
            failed.append(bench)
            log(f"{bench} 出错：\n{traceback.format_exc()}")
    if failed:
        log(f"出错的基准：{failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
