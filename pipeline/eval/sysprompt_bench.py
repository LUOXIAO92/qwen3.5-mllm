"""系统提示遵循评测（任务 #71）：全量微调后的语言模型是否还听从系统消息中的要求。

C 组各运行的训练数据都没有系统回合：ASR 数据是不套模板的续写格式，回放的对话组从 `<|im_start|>user` 开始；已有的评测
（IFEval、指令探针、选择题、GPQA）也都把要求写在用户回合。这里把同一条要求分别放在系统消息和用户消息中，比较模型照做的
比例。

    <py3.12> -m pipeline.eval.sysprompt_bench build
    python -m pipeline.eval.sysprompt_bench generate --model-dir <融合模型目录> [--language-model <检查点>/language_model.safetensors] \\
        --out-dir <输出目录> --label <模型标签> [vLLM 引擎的参数，见 pipeline/eval/vllm_host.py]
    <py3.12> -m pipeline.eval.sysprompt_bench score --out-dir <输出目录>

条目
====

sysprompt/tasks.jsonl 是本仓库编写的 100 个中性的英文开放问题（日常科学、自然、生活常识等），只问内容，不提格式、长度、
语言和语气方面的要求，与下面的约束都不冲突。约束取自 IFEval 的指令注册表（pipeline/eval/ifeval/instructions_registry.py，
官方代码）：约束原文由注册表中这一类的 build_description 写出，打分用同一个类的 check_following，经官方 evaluation_lib 的
strict 与 loose 判定。不用其他外部指令数据。

注册表的 25 类中用 21 类（INSTRUCTION_TYPES），各 14 条，共 294 条。排除的 4 类见 EXCLUDED_TYPES：两类与开放问题不相称
（数字母、只许回答 yes/no/maybe），另两类在参考模型的冒烟检查（2026-11-27）中无法反映模型是否照做：高亮（参考模型不提要求时
回答本来就用 Markdown 加粗，碰巧满足的比例为 100%）和禁用词（禁用问题的主题词，参考模型在系统消息与用户消息中都几乎做不到，
0% 与 15%；按 IFEval 的做法随机抽词又几乎总是碰巧满足）。build 按固定种子（SEED）把 21 类排成一个循环，问题打乱后第 k 条
分给第 k mod 100 个问题（每个问题 2–3 条，类别互不相同，build 时检查），写出 sysprompt/items.jsonl（提交在仓库中，评测只
读它）。

约束的参数都明确给出（constraint_kwargs），由每条自己的种子从下面的范围中抽取，取值让「不提要求时碰巧满足」的比例尽量低：
关键词从注册表的 WORD_LIST 中取至少 5 个字母、不在 EXCLUDED_WORDS 中的词（官方的检查不加词边界，短词会在别的词中匹配）；
关键词频率只用 at least（2–4 次）；回答语言为项目 18 种语言中注册表也支持的 12 种（英语除外）；句数 less than 2–4 或
at least 12–20；词数 less than 20–50 或 at least 350–500；段落数 2–5；第 n 段的首词（n 不超过段落数，注册表自己的抽法
可能抽到不存在的段）；占位符 1–4 个；项目符号 2–5 个；分节 2–5 节；全大写词至少 8–15 个；其余类别（结尾短语、附言标记、
分节的写法）从注册表自己的选项中抽。

三种条件
========

同一个 chat template（模型自带），enable_thinking = False，贪心解码，最多 MAX_NEW_TOKENS（1,280，与 IFEval 相同）个新
token，在 <|im_end|> 或 <|endoftext|> 处停止：

- A：没有约束，用户消息只有问题；
- B：约束是系统消息，问题是用户消息；
- C：没有系统消息，用户消息为问题、一个空行、约束（与 IFEval 把要求写在请求中的做法相同）。

指标：B、C 的约束满足率（strict 与 loose，定义同 IFEval；每条只有一个约束，prompt 级与 instruction 级相同）。A 是对照，
即不提要求时回答碰巧满足约束的比例。主要看各模型 B 与参考模型 B 之差，以及 B − C（系统消息比写在用户消息中少照做多少）。
同一组条目在三种条件、各模型之间成对比较，scores.json 记下成对的差的标准误和不一致的条数（B 照做而 C 没有、C 照做而 B
没有）。

输出（--out-dir）
=================

- generate：responses.jsonl（每条 item_id、condition、instruction_id、回答、生成 token 数、是否写到上限、结束原因）与
  generation.json（引擎、解码设置、items.jsonl 的 sha256、vLLM 的设置与用时）。responses.jsonl 已存在时跳过（--force 重做）。
- score：scored.jsonl（每条 item_id、condition、instruction_id、strict、loose，不含回答原文）与 scores.json（三种条件的满足率、
  B − C、按类别与大类的满足率、回答长度）。打分需要 langdetect 与 nltk（≥ 3.9.1，带 punkt_tab），与 IFEval 的打分相同，在
  另一个 python 中只用 CPU；langdetect 固定 DetectorFactory.seed = 0。

生成只用 vLLM 的 docker 镜像（用户 2026-11-27 决定本机评测一律用 vLLM）：提示在宿主机上展开成 token id，容器中的
pipeline/eval/vllm_generate.py 生成，模型来源（参考模型 --vllm-model，运行 --export-dir 加 --config-dir）与其他评测相同。
各模型一起运行由 pipeline/eval/sysprompt_local.py 完成。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "sysprompt"
TASKS_PATH = DATA_DIR / "tasks.jsonl"
ITEMS_PATH = DATA_DIR / "items.jsonl"
SEED = 71
PER_TYPE = 14
CONDITIONS = ("A", "B", "C")
CONDITION_TEXT = {"A": "无约束", "B": "约束在系统消息", "C": "约束在用户消息末尾"}
# 用到的 21 类约束（注册表 INSTRUCTION_DICT 的键）。
INSTRUCTION_TYPES = (
    "keywords:existence",
    "keywords:frequency",
    "language:response_language",
    "length_constraints:number_sentences",
    "length_constraints:number_paragraphs",
    "length_constraints:number_words",
    "length_constraints:nth_paragraph_first_word",
    "detectable_content:number_placeholders",
    "detectable_content:postscript",
    "detectable_format:number_bullet_lists",
    "detectable_format:multiple_sections",
    "detectable_format:json_format",
    "detectable_format:title",
    "combination:two_responses",
    "combination:repeat_prompt",
    "startend:end_checker",
    "startend:quotation",
    "change_case:english_capital",
    "change_case:english_lowercase",
    "change_case:capital_word_frequency",
    "punctuation:no_comma",
)
EXCLUDED_TYPES = {
    "keywords:letter_frequency": "数某个字母出现几次，衡量字符级的计数能力，与是否听从系统消息无关",
    "detectable_format:constrained_response": "只许回答 My answer is yes/no/maybe，与开放问题冲突",
    "detectable_format:number_highlighted_sections": "参考模型不提要求时回答本来就用 Markdown 加粗，碰巧满足的比例为 100%",
    "keywords:forbidden_words": "禁用主题词时参考模型在两种位置都几乎做不到（0%、15%），随机抽词又几乎总是碰巧满足",
}
# 关键词：注册表 WORD_LIST 中至少这么多个字母的词（官方的关键词检查不加词边界，短词会在别的词中匹配）。
KEYWORD_MIN_LENGTH = 5
# 不用作关键词的词：与中性问题不相称的话题。
EXCLUDED_WORDS = frozenset({
    "abuse", "alcohol", "anger", "anxiety", "attack", "blood", "breast", "cancer", "cigarette", "crash", "death", "debt",
    "depression", "devil", "disaster", "drunk", "funeral", "girlfriend", "boyfriend", "horror", "injury", "nasty", "panic",
    "penalty", "police", "shoot", "steal", "stupid", "suspect", "virus", "witness",
})
MAX_NEW_TOKENS = 1280
# 条件 C 中问题与约束之间的分隔。
USER_JOIN = "\n\n"


def read_jsonl(path: Path) -> list:
    with open(file = path, encoding = "utf-8") as handle:
        return [json.loads(s = line) for line in handle if line.strip()]


def write_atomic(path: Path, text: str) -> None:
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    temporary.write_text(data = text, encoding = "utf-8")
    temporary.replace(target = path)


def write_jsonl(path: Path, rows: list) -> None:
    write_atomic(path = path, text = "".join(json.dumps(obj = row, ensure_ascii = False) + "\n" for row in rows))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- 条目（build，需要 langdetect 与 nltk）


def keyword_pool() -> list:
    """关键词、首词的候选：WORD_LIST 中至少 KEYWORD_MIN_LENGTH 个字母、不在 EXCLUDED_WORDS 中的词（去重后保持原顺序）。"""
    from pipeline.eval.ifeval import instructions_util

    words = [word for word in instructions_util.WORD_LIST if word.isalpha() and len(word) >= KEYWORD_MIN_LENGTH]
    return list(dict.fromkeys(word for word in words if word not in EXCLUDED_WORDS))


def response_languages() -> list:
    """回答语言约束的候选：项目的 18 种语言中注册表也支持的（英语除外）。"""
    from pipeline.eval.ifeval import instructions_util
    from pipeline.eval.scoring import LANGS

    return [lang for lang in LANGS if lang in instructions_util.LANGUAGE_CODES and lang != "en"]


def constraint_kwargs(instruction_id: str, task: dict, rng: random.Random, words: list, languages: list) -> dict:
    """一条约束交给 build_description 的参数（全部明确给出，见开头「条目」）。"""
    from pipeline.eval.ifeval import instructions

    relations = instructions._COMPARISON_RELATION
    if instruction_id == "keywords:existence":
        return {"keywords": rng.sample(population = words, k = 2)}
    if instruction_id == "keywords:frequency":
        return {"keyword": rng.choice(seq = words), "frequency": rng.randint(a = 2, b = 4), "relation": "at least"}
    if instruction_id == "language:response_language":
        return {"language": rng.choice(seq = languages)}
    if instruction_id == "length_constraints:number_sentences":
        relation = rng.choice(seq = relations)
        count = rng.randint(a = 2, b = 4) if relation == "less than" else rng.randint(a = 12, b = 20)
        return {"num_sentences": count, "relation": relation}
    if instruction_id == "length_constraints:number_paragraphs":
        return {"num_paragraphs": rng.randint(a = 2, b = 5)}
    if instruction_id == "length_constraints:number_words":
        relation = rng.choice(seq = relations)
        count = rng.randint(a = 20, b = 50) if relation == "less than" else rng.randint(a = 350, b = 500)
        return {"num_words": count, "relation": relation}
    if instruction_id == "length_constraints:nth_paragraph_first_word":
        paragraphs = rng.randint(a = 2, b = 5)
        return {
            "num_paragraphs": paragraphs,
            "nth_paragraph": rng.randint(a = 1, b = paragraphs),
            "first_word": rng.choice(seq = words),
        }
    if instruction_id == "detectable_content:number_placeholders":
        return {"num_placeholders": rng.randint(a = 1, b = 4)}
    if instruction_id == "detectable_content:postscript":
        return {"postscript_marker": rng.choice(seq = instructions._POSTSCRIPT_MARKER)}
    if instruction_id == "detectable_format:number_bullet_lists":
        return {"num_bullets": rng.randint(a = 2, b = 5)}
    if instruction_id == "detectable_format:multiple_sections":
        return {"section_spliter": rng.choice(seq = instructions._SECTION_SPLITER), "num_sections": rng.randint(a = 2, b = 5)}
    if instruction_id == "combination:repeat_prompt":
        return {"prompt_to_repeat": task["task"]}
    if instruction_id == "startend:end_checker":
        return {"end_phrase": rng.choice(seq = instructions._ENDING_OPTIONS)}
    if instruction_id == "change_case:capital_word_frequency":
        return {"capital_frequency": rng.randint(a = 8, b = 15), "capital_relation": "at least"}
    if instruction_id in INSTRUCTION_TYPES:
        # json_format、title、two_responses、quotation、english_capital、english_lowercase、no_comma 没有参数
        return {}
    raise ValueError(f"没有用到的约束类别：{instruction_id}")


def describe(instruction_id: str, kwargs: dict) -> tuple:
    """用注册表的类写出约束原文：(原文, 注册表确认后的参数)。参数都已给出，结果不含随机性。"""
    from pipeline.eval.ifeval import instructions_registry

    instruction = instructions_registry.INSTRUCTION_DICT[instruction_id](instruction_id)
    text = instruction.build_description(**kwargs)
    return text, instruction.get_instruction_args() or {}


def build_items(tasks: list, seed: int = SEED, per_type: int = PER_TYPE) -> list:
    """由问题列表生成条目（见开头「条目」）：第 k 条的类别为循环中的第 k mod 类别数 个，问题为打乱后的第 k mod 问题数 个；
    每类 per_type 条，同一个问题分到的类别互不相同（不满足时报错）。"""
    rng = random.Random(x = f"{seed}-assign")
    order = list(INSTRUCTION_TYPES)
    rng.shuffle(x = order)
    shuffled = list(tasks)
    rng.shuffle(x = shuffled)
    count = per_type * len(order)
    kinds = {}
    for number in range(count):
        kinds.setdefault(number % len(shuffled), []).append(order[number % len(order)])
    repeated = [shuffled[index]["task_id"] for index, values in kinds.items() if len(values) != len(set(values))]
    if repeated:
        raise ValueError(f"这些问题分到了重复的约束类别：{repeated[:5]}")
    words = keyword_pool()
    languages = response_languages()
    items = []
    for number in range(count):
        instruction_id = order[number % len(order)]
        task = shuffled[number % len(shuffled)]
        item_id = f"s{number + 1:03d}"
        kwargs = constraint_kwargs(
            instruction_id = instruction_id,
            task = task,
            rng = random.Random(x = f"{seed}-{item_id}"),
            words = words,
            languages = languages,
        )
        constraint, resolved = describe(instruction_id = instruction_id, kwargs = kwargs)
        items.append({
            "item_id": item_id,
            "task_id": task["task_id"],
            "task": task["task"],
            "instruction_id": instruction_id,
            "kwargs": resolved,
            "constraint": constraint,
        })
    return items


def run_build(args) -> int:
    tasks = read_jsonl(path = args.tasks)
    items = build_items(tasks = tasks, seed = args.seed, per_type = args.per_type)
    write_jsonl(path = args.items, rows = items)
    counts = {}
    for item in items:
        counts[item["instruction_id"]] = counts.get(item["instruction_id"], 0) + 1
    print(f"{len(items)} 条（{len(counts)} 类，每类 {sorted(set(counts.values()))} 条，{len(tasks)} 个问题）→ {args.items}")
    print(f"sha256 {file_sha256(path = args.items)}")
    return 0


# ---------------------------------------------------------------- 提示与生成


def conversation(item: dict, condition: str) -> list:
    """一条条目在一种条件下的对话（见开头「三种条件」）。"""
    if condition == "A":
        return [{"role": "user", "content": item["task"]}]
    if condition == "B":
        return [{"role": "system", "content": item["constraint"]}, {"role": "user", "content": item["task"]}]
    if condition == "C":
        return [{"role": "user", "content": item["task"] + USER_JOIN + item["constraint"]}]
    raise ValueError(f"未知的条件：{condition}")


def prompt_ids(tokenizer, messages: list) -> list:
    """对话按 chat template 展开（加助手回合开头，不思考）后的 token id。"""
    text = tokenizer.apply_chat_template(
        conversation = messages,
        tokenize = False,
        add_generation_prompt = True,
        enable_thinking = False,
    )
    return tokenizer(text = text, add_special_tokens = False)["input_ids"]


def requests_of(tokenizer, items: list, max_new_tokens: int) -> list:
    """全部请求（条目 × 条件），index 从 0 起，按条目、条件排列。"""
    requests = []
    for item in items:
        for condition in CONDITIONS:
            requests.append({
                "index": len(requests),
                "seed": 0,
                "prompt_token_ids": [int(token) for token in prompt_ids(tokenizer = tokenizer, messages = conversation(item = item, condition = condition))],
                "max_tokens": int(max_new_tokens),
                "item_id": item["item_id"],
                "condition": condition,
                "instruction_id": item["instruction_id"],
            })
    return requests


def response_rows(requests: list, results: dict, tokenizer, max_new_tokens: int) -> list:
    """生成结果解码、清理（与 IFEval 相同：去掉思考块与特殊标记）后的 responses.jsonl 各行。"""
    from pipeline.eval.run_eval import clean_output
    from pipeline.eval.think_check import STOP_IDS

    rows = []
    for request in requests:
        result = results[request["index"]]
        ids = [int(token) for token in result["token_ids"]]
        ended = len(ids) > 0 and ids[-1] in STOP_IDS
        rows.append({
            "item_id": request["item_id"],
            "condition": request["condition"],
            "instruction_id": request["instruction_id"],
            "response": clean_output(text = tokenizer.decode(token_ids = ids, skip_special_tokens = True)),
            "n_new_tokens": len(ids),
            "capped": bool(not ended and len(ids) >= max_new_tokens),
            "finish_reason": result.get("finish_reason"),
        })
    return rows


def run_generate(args) -> int:
    from pipeline.eval import vllm_host
    from pipeline.eval.text_bench import tokenizer_of

    def log(message):
        print(f"{time.strftime('%H:%M:%S')} {message}", flush = True)

    args.out_dir.mkdir(parents = True, exist_ok = True)
    responses_path = args.out_dir / "responses.jsonl"
    if responses_path.exists() and not args.force:
        log(f"{responses_path} 已存在，跳过（--force 重做）")
        return 0
    items = read_jsonl(path = args.items)
    tokenizer = tokenizer_of(model_dir = args.model_dir)
    stop_ids = vllm_host.stop_ids_of(tokenizer = tokenizer)
    requests = requests_of(tokenizer = tokenizer, items = items, max_new_tokens = args.max_new_tokens)
    log(f"模型 {args.label or args.language_model or args.model_dir}：{len(items)} 条 × {len(CONDITIONS)} 种条件 = {len(requests)} 条请求")
    settings = vllm_host.job_settings(
        args = args,
        max_model_len = max(len(request["prompt_token_ids"]) for request in requests) + args.max_new_tokens,
        max_tokens = args.max_new_tokens,
        temperature = 0.0,
        stop_token_ids = stop_ids,
    )
    start = time.perf_counter()
    results, record = vllm_host.run_job(
        args = args,
        model_dir = args.model_dir,
        language_model = args.language_model,
        requests = [{key: request[key] for key in ("index", "seed", "prompt_token_ids", "max_tokens")} for request in requests],
        settings = settings,
        kind = "sysprompt",
        label = args.label,
        log = log,
    )
    rows = response_rows(
        requests = requests,
        results = results,
        tokenizer = tokenizer,
        max_new_tokens = args.max_new_tokens,
    )
    generation = {
        "label": args.label,
        "model_dir": str(args.model_dir),
        "language_model": None if args.language_model is None else str(args.language_model),
        "items": str(args.items),
        "items_sha256": file_sha256(path = args.items),
        "n_items": len(items),
        "conditions": {condition: CONDITION_TEXT[condition] for condition in CONDITIONS},
        "user_join": USER_JOIN,
        "decoding": "greedy",
        "enable_thinking": False,
        "max_new_tokens": args.max_new_tokens,
        "engine": "vllm",
        "seconds": round(number = time.perf_counter() - start, ndigits = 1),
        "vllm": record,
    }
    write_atomic(path = args.out_dir / "generation.json", text = json.dumps(obj = generation, ensure_ascii = False, indent = 1) + "\n")
    write_jsonl(path = responses_path, rows = rows)
    capped = sum(row["capped"] for row in rows)
    log(f"写出 {len(rows)} 条回答（写到上限 {capped} 条），{generation['seconds']} 秒")
    return 0


# ---------------------------------------------------------------- 打分（需要 langdetect 与 nltk）


def judge(item: dict, response: str) -> tuple:
    """一条回答按条目的约束判定：(strict, loose)，用官方 evaluation_lib（与 IFEval 打分相同）。"""
    from pipeline.eval.ifeval import evaluation_lib

    key = item["item_id"]
    example = evaluation_lib.InputExample(
        key = key,
        instruction_id_list = [item["instruction_id"]],
        prompt = key,
        kwargs = [item["kwargs"]],
    )
    mapping = {key: response}
    strict = evaluation_lib.test_instruction_following_strict(inp = example, prompt_to_response = mapping)
    loose = evaluation_lib.test_instruction_following_loose(inp = example, prompt_to_response = mapping)
    return bool(strict.follow_all_instructions), bool(loose.follow_all_instructions)


def score_rows(items: list, responses: list) -> list:
    """逐条判定：[{item_id, condition, instruction_id, strict, loose}]，langdetect 固定种子 0。"""
    from langdetect import DetectorFactory

    DetectorFactory.seed = 0
    by_id = {item["item_id"]: item for item in items}
    rows = []
    for response in responses:
        item = by_id[response["item_id"]]
        strict, loose = judge(item = item, response = response["response"])
        rows.append({
            "item_id": response["item_id"],
            "condition": response["condition"],
            "instruction_id": item["instruction_id"],
            "strict": strict,
            "loose": loose,
        })
    return rows


def rate(values: list):
    return sum(values) / len(values) if values else None


def paired_difference(first: list, second: list) -> dict:
    """同一组条目上两组 0/1 结果之差（first − second）：差、标准误、first 照做而 second 没有的条数与相反的条数。"""
    pairs = list(zip(first, second))
    n = len(pairs)
    if n == 0:
        return {"diff": None, "se": None, "first_only": 0, "second_only": 0, "n": 0}
    first_only = sum(1 for a, b in pairs if a and not b)
    second_only = sum(1 for a, b in pairs if b and not a)
    p10, p01 = first_only / n, second_only / n
    variance = max(p10 + p01 - (p10 - p01) ** 2, 0.0)
    return {
        "diff": p10 - p01,
        "se": math.sqrt(variance / n),
        "first_only": first_only,
        "second_only": second_only,
        "n": n,
    }


def outcome_table(rows: list) -> dict:
    """{条件: {item_id: (strict, loose)}}。"""
    table = {condition: {} for condition in CONDITIONS}
    for row in rows:
        table[row["condition"]][row["item_id"]] = (bool(row["strict"]), bool(row["loose"]))
    return table


def summarize(items: list, rows: list, responses: list) -> dict:
    """一个模型的 scores.json。"""
    table = outcome_table(rows = rows)
    ids = [item["item_id"] for item in items]
    missing = [(condition, item_id) for condition in CONDITIONS for item_id in ids if item_id not in table[condition]]
    if missing:
        raise ValueError(f"缺少 {len(missing)} 条判定：{missing[:5]}")
    conditions = {}
    for condition in CONDITIONS:
        strict = [table[condition][item_id][0] for item_id in ids]
        loose = [table[condition][item_id][1] for item_id in ids]
        conditions[condition] = {"n": len(ids), "strict": rate(values = strict), "loose": rate(values = loose)}
    gap = {}
    for position, mode in enumerate(("strict", "loose")):
        gap[mode] = paired_difference(
            first = [table["B"][item_id][position] for item_id in ids],
            second = [table["C"][item_id][position] for item_id in ids],
        )
    by_type = {}
    by_tier0 = {}
    for item in items:
        for condition in CONDITIONS:
            strict, loose = table[condition][item["item_id"]]
            for bucket, key in ((by_type, item["instruction_id"]), (by_tier0, item["instruction_id"].split(":")[0])):
                entry = bucket.setdefault(key, {}).setdefault(condition, {"n": 0, "strict": 0, "loose": 0})
                entry["n"] += 1
                entry["strict"] += int(strict)
                entry["loose"] += int(loose)
    for bucket in (by_type, by_tier0):
        for key, entry in bucket.items():
            for condition, counts in entry.items():
                entry[condition] = {
                    "n": counts["n"],
                    "strict": counts["strict"] / counts["n"],
                    "loose": counts["loose"] / counts["n"],
                }
    generation = {}
    for condition in CONDITIONS:
        group = [row for row in responses if row["condition"] == condition]
        generation[condition] = {
            "mean_new_tokens": round(number = sum(row["n_new_tokens"] for row in group) / len(group), ndigits = 1) if group else None,
            "capped": rate(values = [bool(row.get("capped")) for row in group]),
            "empty": rate(values = [not row["response"].strip() for row in group]),
        }
    return {
        "n_items": len(ids),
        "conditions": conditions,
        "b_minus_c": gap,
        "by_type": {key: by_type[key] for key in sorted(by_type)},
        "by_tier0": {key: by_tier0[key] for key in sorted(by_tier0)},
        "generation": generation,
        "checker": "google-research instruction_following_eval（pipeline/eval/ifeval，strict 与 loose）",
        "langdetect_seed": 0,
    }


def run_score(args) -> int:
    from pipeline.eval.ifeval.score import check_dependencies

    check_dependencies()
    items = read_jsonl(path = args.items)
    responses = read_jsonl(path = args.out_dir / "responses.jsonl")
    generation_path = args.out_dir / "generation.json"
    generation = json.loads(s = generation_path.read_text(encoding = "utf-8")) if generation_path.exists() else {}
    sha = file_sha256(path = args.items)
    if generation.get("items_sha256") not in (None, sha):
        raise SystemExit(f"{args.out_dir} 的回答来自另一份 items.jsonl（sha256 {generation['items_sha256']}，现在是 {sha}）")
    rows = score_rows(items = items, responses = responses)
    summary = summarize(items = items, rows = rows, responses = responses)
    summary = {"label": generation.get("label"), "items_sha256": sha, **summary}
    write_jsonl(path = args.out_dir / "scored.jsonl", rows = rows)
    write_atomic(path = args.out_dir / "scores.json", text = json.dumps(obj = summary, ensure_ascii = False, indent = 1) + "\n")
    parts = "，".join(
        f"{condition} strict {summary['conditions'][condition]['strict']:.3f} loose {summary['conditions'][condition]['loose']:.3f}"
        for condition in CONDITIONS
    )
    print(f"{summary['label'] or args.out_dir}：{parts}；B − C strict {summary['b_minus_c']['strict']['diff']:+.3f}", flush = True)
    return 0


# ---------------------------------------------------------------- 命令行


def parse_args(argv: list | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest = "command", required = True)
    build = commands.add_parser("build", help = "由 tasks.jsonl 生成 items.jsonl（需要 langdetect 与 nltk）")
    build.add_argument("--tasks", default = TASKS_PATH, type = Path)
    build.add_argument("--items", default = ITEMS_PATH, type = Path)
    build.add_argument("--seed", default = SEED, type = int)
    build.add_argument("--per-type", default = PER_TYPE, type = int)
    generate = commands.add_parser("generate", help = "用 vLLM 生成三种条件下的回答")
    generate.add_argument("--model-dir", required = True, type = Path, help = "融合模型目录（取分词器与 chat template）")
    generate.add_argument("--language-model", default = None, type = Path, help = "全量微调检查点中的 language_model.safetensors；不给时是参考模型")
    generate.add_argument("--out-dir", required = True, type = Path, help = "这个模型的输出目录")
    generate.add_argument("--label", default = None, help = "模型标签（写进结果、容器名）")
    generate.add_argument("--items", default = ITEMS_PATH, type = Path)
    generate.add_argument("--max-new-tokens", default = MAX_NEW_TOKENS, type = int)
    generate.add_argument("--force", action = "store_true", help = "已有回答也重做")
    from pipeline.eval import vllm_host

    vllm_host.add_arguments(parser = generate)
    generate.set_defaults(engine = "vllm")
    score = commands.add_parser("score", help = "打分（需要 langdetect 与 nltk）")
    score.add_argument("--out-dir", required = True, type = Path)
    score.add_argument("--items", default = ITEMS_PATH, type = Path)
    args = parser.parse_args(args = argv)
    if args.command == "generate":
        if args.engine != "vllm":
            parser.error("generate 只用 vLLM（--engine vllm）")
        vllm_host.check_arguments(
            parser = parser,
            args = args,
            default_job_dir = args.out_dir / "vllm-job",
            language_model = args.language_model,
        )
    return args


def main(argv: list | None = None) -> int:
    args = parse_args(argv = argv)
    if args.command == "build":
        return run_build(args = args)
    if args.command == "generate":
        return run_generate(args = args)
    return run_score(args = args)


if __name__ == "__main__":
    sys.exit(main())
