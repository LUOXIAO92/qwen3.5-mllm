"""为四组回放文本写出提示清单 `<回放目录>/<组>/<划分>/prompts.jsonl`，并把各组的起始 token 记在 `<回放目录>/templates.json`。

    python -m pipeline.replay.prompts --model <参考模型目录> --replay-dir <回放目录> --free-count eval=8000,train=60000

四组的开头：

| 组 | 提示 |
| --- | --- |
| free_raw | 一个换行 `\\n`（模拟 BOS）；`--free-raw-start random` 时改为从普通词表中均匀抽取的一个 token。第一个生成 token 由 `first_token` 抽取后写进提示清单的 `first` 字段 |
| free_chat | 聊天模板的用户轮开头 `<|im_start|>user\\n`。模型写完用户消息（以 `<|im_end|>` 结束）后，补上助手轮开头和空的思考块，再让模型写回答 |
| free_chat_think | 与 free_chat 相同，助手轮按思考模式打开（`enable_thinking = True` 的生成提示，补上的是 `<|im_start|>assistant\\n<think>\\n`），模型先写思考、`</think>`，再写回答 |
| cont_raw | 文档开头（64–256 token），不加模板 |
| cont_chat | 用户轮内容为续写指令加空行加同一段文档开头，助手轮开头和空的思考块之后由模型续写 |
| free_sys_chat | 聊天模板的系统轮开头 `<|im_start|>system\\n`。模型写完系统提示（以 `<|im_end|>` 结束）后补上 `\\n<|im_start|>user\\n`，模型写用户消息，之后与 free_chat 相同 |
| free_sys_chat_think | 与 free_sys_chat 相同，助手轮按思考模式打开（与 free_chat_think 相同） |

free_raw 默认用换行开头。试运行中两种开头各采样 600 条：换行开头 58% 的样本出现循环，前 20 万 token 中不同 4-gram
的比例为 0.66，开头 8 个 token 与其他样本相同的占 2.5%，一半像代码；随机 token 开头分别为 51%、0.61、0%，四分之一像代码。
两者的重复程度相近，随机开头没有更少重复，所以保留换行开头。

聊天模板与 ASR 训练相同：`pipeline/train/data.py` 的 `build_training_text` 用 `apply_chat_template`，
`add_generation_prompt = False`、`enable_thinking = False`，没有系统消息，渲染结果是
`<|im_start|>user\\n{用户}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n{助手}<|im_end|>\\n`。
这里用同一个函数渲染带占位符的对话，再按占位符切出模板的各段，所以模板写法不在代码里重复。
训练时只对助手内容和结尾的 `<|im_end|>` 计算损失，free_chat 的用户消息也是模型生成的，是否计算损失由训练决定。

带系统提示的两组（#71）：模型在系统轮里自己写系统提示，程序补上的各段取自 chat template 对
「系统消息 + 用户消息」加生成提示的渲染（`enable_thinking` 为假或真），见 `system_chat_segments`。拼出的结果是
`<|im_start|>system\\n{系统}<|im_end|>\\n<|im_start|>user\\n{用户}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n{助手}<|im_end|>`
（思考模式下助手轮开头为 `<|im_start|>assistant\\n<think>\\n`，模型接着写思考、`</think>` 和回答），与 chat template 渲染完整的
三轮对话逐个 token 相同（`pipeline/replay/tests/test_system_prompts.py`）。系统提示和用户消息只作上下文，训练时只对助手
内容和结尾的 `<|im_end|>` 计算损失。

`--meta-per-lang eval=20,train=703` 时（用户 2026-11-28 决定），带系统提示的组改用元指令（`meta_prompts.py`）：每个划分每种
语言给出这么多条提示，每条提示另有 `meta_ids`，即「用户轮为元指令、助手轮按 `enable_thinking = False` 打开」的渲染；模型在
这个上下文中写出的回答就是系统提示。`prompt_ids` 仍是 `<|im_start|>system\n`，保存的样本从这里开始，不含元指令这一轮。
每行另记 `meta_lang`、`meta_kind`（措辞）、`meta_domain`（主题）、`meta_rule`（是否要求可检查的规则）、`meta_version` 和
`meta_text`。保留的样本不足时，`--meta-append train --meta-append-counts ja=80,ko=60` 在已有的提示清单后面按这些条数补上
提示（编号接着往后排，不改动已有的行），generate 再次运行时会接着生成新加的提示。

文档开头的长度：每篇文档先在 [64, 256] 中均匀抽一个目标 token 数，再在不超过目标的位置中选最靠后的句末或行末
（没有时退到词边界），切下的文本去掉结尾空白后重新分词。至少 64 token 才能定下语言、话题和文体；
至多 256 token 让人写的前缀只占样本的一小部分，样本的主体是参考模型自己写的续写。

分词器（#77）：加载参考模型目录的分词器后用 `pipeline.tokenizer_fix.check_tokenizer` 检查切法与原始 Qwen3.5 的
`tokenizer.json` 相同。#70 的筛选回放目录 `replay/` 的续写提示是修正之前写出的（泰语、阿拉伯语、波斯语的部分文档开头
切法不同），不筛选的回放目录 `replay-unfiltered/` 用修正后的分词器重新写出。自由生成两组的提示只有换行和聊天模板的
特殊 token，修正前后相同。
"""

import argparse
import json
import random
import unicodedata
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from pipeline.replay.first_token import first_token_logprobs, sample_first
from pipeline.replay.meta_prompts import DOMAINS, KINDS, LANGS, META_VERSION, assign, meta_text
from pipeline.replay.groups import (  # noqa: F401（其他模块从这里导入）
    ALL_GROUPS,
    CHAT_GROUPS,
    GROUPS,
    SELF_SYSTEM_GROUPS,
    SELF_USER_GROUPS,
    SPLITS,
    THINK_GROUPS,
)
from pipeline.tokenizer_fix import check_tokenizer

CONTINUE_INSTRUCTION = "Continue writing the following text."
MIN_PREFIX_TOKENS = 64
MAX_PREFIX_TOKENS = 256
_SYSTEM = "\x01SYSTEM\x01"
_USER = "\x01USER\x01"
_ASSISTANT = "\x01ASSISTANT\x01"
_TERMINALS = set("。！？!?.…;؟؛।")


def render_training(tokenizer, user, assistant):
    """按 ASR 训练的设置渲染一轮对话（与 build_training_text 相同，只是内容为纯文本）。"""
    conversation = [
        {"role": "user", "content": [{"type": "text", "text": user}]},
        {"role": "assistant", "content": [{"type": "text", "text": assistant}]},
    ]
    return tokenizer.apply_chat_template(
        conversation = conversation,
        tokenize = False,
        add_generation_prompt = False,
        enable_thinking = False,
    )


def render_prompt(tokenizer, user):
    """用户轮加生成提示，与训练评测时的提示相同。"""
    conversation = [{"role": "user", "content": [{"type": "text", "text": user}]}]
    return tokenizer.apply_chat_template(
        conversation = conversation,
        tokenize = False,
        add_generation_prompt = True,
        enable_thinking = False,
    )


def thinking_between(tokenizer):
    """思考模式下用户消息的 <|im_end|> 之后补上的文本：按 enable_thinking = True 渲染生成提示，取用户内容之后的部分。"""
    conversation = [{"role": "user", "content": [{"type": "text", "text": _USER}]}]
    text = tokenizer.apply_chat_template(
        conversation = conversation,
        tokenize = False,
        add_generation_prompt = True,
        enable_thinking = True,
    )
    return text.split(_USER)[1].removeprefix("<|im_end|>")


def chat_segments(tokenizer):
    """按占位符切出模板：用户轮开头、用户与助手之间、助手之后。"""
    text = render_training(tokenizer = tokenizer, user = _USER, assistant = _ASSISTANT)
    head, rest = text.split(_USER)
    middle, tail = rest.split(_ASSISTANT)
    prompt = render_prompt(tokenizer = tokenizer, user = _USER)
    assert prompt == head + _USER + middle, "加生成提示的渲染与训练渲染的前半段不一致"
    return {"user_start": head, "between": middle, "end": tail}


def system_conversation(system, user, assistant = None, reasoning = None):
    """系统消息加用户消息（以及可选的助手回答、思考）的对话，内容为纯文本。"""
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [{"type": "text", "text": user}]},
    ]
    if assistant is not None:
        message = {"role": "assistant", "content": [{"type": "text", "text": assistant}]}
        if reasoning is not None:
            message["reasoning_content"] = reasoning
        conversation.append(message)
    return conversation


def system_chat_segments(tokenizer, thinking):
    """带系统消息的模板各段：系统轮开头、系统提示的 <|im_end|> 之后补上的文本、用户消息的 <|im_end|> 之后补上的文本、助手之后。

    先按占位符渲染「系统 + 用户」加生成提示（enable_thinking = thinking），切出前三段；再渲染加上助手回答的完整对话
    （add_generation_prompt = False，思考模式下助手回答带一段思考），确认完整对话就是生成提示后面接模型写出的内容：
    非思考模式为「回答」，思考模式为「思考 + \\n</think>\\n\\n + 回答」，最后是结尾的 <|im_end|>\\n。
    """
    prompt = tokenizer.apply_chat_template(
        conversation = system_conversation(system = _SYSTEM, user = _USER),
        tokenize = False,
        add_generation_prompt = True,
        enable_thinking = thinking,
    )
    head, rest = prompt.split(_SYSTEM)
    system_tail, user_tail = rest.split(_USER)
    assert system_tail.startswith("<|im_end|>") and user_tail.startswith("<|im_end|>"), "系统轮或用户轮不以 <|im_end|> 结束"
    reasoning = "\x01REASONING\x01" if thinking else None
    full = tokenizer.apply_chat_template(
        conversation = system_conversation(
            system = _SYSTEM,
            user = _USER,
            assistant = _ASSISTANT,
            reasoning = reasoning,
        ),
        tokenize = False,
        add_generation_prompt = False,
        enable_thinking = thinking,
    )
    written = f"{reasoning}\n</think>\n\n{_ASSISTANT}" if thinking else _ASSISTANT
    assert full.startswith(prompt + written), "完整对话的渲染与「生成提示 + 模型写出的内容」不一致"
    return {
        "system_start": head,
        "system_between": system_tail.removeprefix("<|im_end|>"),
        "user_between": user_tail.removeprefix("<|im_end|>"),
        "end": full.removeprefix(prompt + written),
    }


def encode(tokenizer, text):
    return tokenizer(text, add_special_tokens = False)["input_ids"]


def regular_token_ids(tokenizer):
    """可作 free_raw 随机开头的 token：普通词表（不含特殊 token），解码后不是控制字符，也不是半个 UTF-8 字符。"""
    special = set(tokenizer.all_special_ids) | set(tokenizer.added_tokens_decoder)
    ids = []
    for token_id in range(len(tokenizer)):
        if token_id in special:
            continue
        text = tokenizer.decode([token_id])
        if not text or "�" in text:
            continue
        if all(unicodedata.category(char) == "Cc" and char not in "\n\t" for char in text):
            continue
        ids.append(token_id)
    return ids


def cut_prefix(tokenizer, text, target):
    """在不超过 target 个 token 的位置切出文档开头；不足 MIN_PREFIX_TOKENS 时返回 None。"""
    encoding = tokenizer(text, add_special_tokens = False, return_offsets_mapping = True)
    offsets = encoding["offset_mapping"]
    n = len(offsets)
    target = min(target, n - 16)
    if target < MIN_PREFIX_TOKENS:
        return None
    sentence, word, any_boundary = None, None, None
    for k in range(MIN_PREFIX_TOKENS, target + 1):
        end = offsets[k - 1][1]
        if k < n and offsets[k][0] < end:
            continue  # 一个字符被拆成几个字节 token，不能从中间切
        any_boundary = end
        after = text[end: end + 1]
        if text[end - 1] in _TERMINALS or after == "\n":
            sentence = end
        elif after.isspace():
            word = end
    end = sentence or word or any_boundary
    if end is None:
        return None
    return text[:end].rstrip()


def meta_rows(tokenizer, start_ids, per_lang, seed):
    """按元指令写系统提示的提示清单：每种语言 per_lang 条（meta_prompts.assign 交错排列），meta_ids 为元指令一轮的渲染。"""
    rows = []
    cache = {}
    for item in assign(per_lang = per_lang, seed = seed):
        text = meta_text(
            lang = item["meta_lang"],
            kind = item["meta_kind"],
            domain = item["meta_domain"],
            rule = item["meta_rule"],
        )
        if text not in cache:
            cache[text] = encode(tokenizer = tokenizer, text = render_prompt(tokenizer = tokenizer, user = text))
        rows.append(
            {
                "prompt_ids": start_ids,
                "meta_ids": cache[text],
                **item,
                "meta_version": META_VERSION,
                "meta_text": text,
            }
        )
    return rows


def append_meta_rows(tokenizer, path, counts, seed):
    """在已有的提示清单后面按语言补上元指令提示，编号接着往后排；先写临时文件再替换，正在运行的 generate 下次运行时读到。"""
    with open(path, encoding = "utf-8") as handle:
        existing = [json.loads(line) for line in handle]
    start_ids = existing[0]["prompt_ids"]
    rows = meta_rows(tokenizer = tokenizer, start_ids = start_ids, per_lang = counts, seed = f"{seed}-{path.parent.parent.name}-{path.parent.name}-append-{len(existing)}")
    for number, row in enumerate(rows):
        row["prompt_index"] = len(existing) + number
        row["appended"] = True
    tmp = path.with_suffix(".jsonl.tmp")
    write_jsonl(path = tmp, rows = existing + rows)
    tmp.replace(path)
    print(f"{path}: 原有 {len(existing)} 条，补上 {len(rows)} 条", flush = True)


def write_jsonl(path, rows):
    path.parent.mkdir(parents = True, exist_ok = True)
    with open(path, "w", encoding = "utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii = False) + "\n")


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required = True, help = "参考模型目录（取分词器与 chat template）")
    parser.add_argument("--replay-dir", required = True, help = "回放目录；续写组默认读取其中的 prefixes/<划分>.jsonl")
    parser.add_argument("--prefixes-dir", default = None, help = "文档开头目录（prefixes 的输出），默认为 <回放目录>/prefixes")
    parser.add_argument("--groups", nargs = "+", default = list(GROUPS), help = "free_chat_think、free_sys_chat、free_sys_chat_think 只用于丢弃循环的回放目录，需要显式给出")
    parser.add_argument("--free-count", default = "eval=8000,train=60000", help = "自由生成两组每个划分的提示条数（生成在达到 token 目标时停止）")
    parser.add_argument("--free-raw-start", choices = ["newline", "random"], default = "newline")
    parser.add_argument(
        "--meta-per-lang",
        default = None,
        help = "带系统提示的组改用元指令写系统提示（meta_prompts.py），每个划分每种语言的提示条数，例如 eval=20,train=703；"
        "给出时这两组不用 --free-count",
    )
    parser.add_argument("--meta-append", choices = list(SPLITS), default = None, help = "在这个划分已有的提示清单后面补提示（只改这一个划分）")
    parser.add_argument("--meta-append-counts", default = None, help = "补提示的条数，按语言给出，例如 ja=80,ko=60")
    parser.add_argument("--seed", type = int, default = 0)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # 文档开头按原始 Qwen3.5 的规则分词（#77）
    check_tokenizer(tokenizer = tokenizer, source = args.model)
    replay_dir = Path(args.replay_dir)
    if args.meta_append:
        counts = {key: int(value) for key, value in (item.split("=") for item in args.meta_append_counts.split(","))}
        for group in args.groups:
            assert group in SELF_SYSTEM_GROUPS, "--meta-append 只用于带系统提示的组"
            append_meta_rows(
                tokenizer = tokenizer,
                path = replay_dir / group / args.meta_append / "prompts.jsonl",
                counts = counts,
                seed = args.seed,
            )
        return
    prefixes_dir = Path(args.prefixes_dir) if args.prefixes_dir else replay_dir / "prefixes"
    segments = chat_segments(tokenizer = tokenizer)
    free_count = {key: int(value) for key, value in (item.split("=") for item in args.free_count.split(","))}
    meta_per_lang = None
    if args.meta_per_lang:
        meta_per_lang = {key: int(value) for key, value in (item.split("=") for item in args.meta_per_lang.split(","))}
    templates_path = replay_dir / "templates.json"
    cuts = {}
    templates = json.loads(templates_path.read_text(encoding = "utf-8")) if templates_path.exists() else {}

    for group in args.groups:
        for split in SPLITS:
            rng = random.Random(f"{args.seed}-{group}-{split}")
            rows = []
            if group == "free_raw":
                if args.free_raw_start == "newline":
                    starts = [encode(tokenizer = tokenizer, text = "\n")] * free_count[split]
                else:
                    pool = regular_token_ids(tokenizer = tokenizer)
                    starts = [[rng.choice(pool)] for _ in range(free_count[split])]
                # 单 token 提示的第一个生成 token 在这里抽取（原因见 first_token.py）
                distinct = sorted({ids[0] for ids in starts})
                table = dict(zip(distinct, first_token_logprobs(model_dir = args.model, start_ids = distinct))) if distinct else {}
                sampler = np.random.default_rng(seed = [args.seed, SPLITS.index(split)])
                rows = [{"prompt_ids": ids, "first": sample_first(logprobs = table[ids[0]], rng = sampler)} for ids in starts]
                templates[group] = {"start": args.free_raw_start, "start_text": "\n" if args.free_raw_start == "newline" else None, "template": None}
            elif group in SELF_USER_GROUPS:
                start_ids = encode(tokenizer = tokenizer, text = segments["user_start"])
                rows = [{"prompt_ids": start_ids} for _ in range(free_count[split])]
                if group == "free_chat":
                    between = segments["between"].removeprefix("<|im_end|>")
                    template = "apply_chat_template(add_generation_prompt = False, enable_thinking = False)，无系统消息"
                else:
                    between = thinking_between(tokenizer = tokenizer)
                    template = "apply_chat_template(add_generation_prompt = True, enable_thinking = True) 的助手轮开头，模型自己写思考与 </think>，无系统消息"
                templates[group] = {
                    "start_text": segments["user_start"],
                    "start_ids": start_ids,
                    # 用户消息以生成的 <|im_end|> 结束，其后补上的 token
                    "between_text": between,
                    "between_ids": encode(tokenizer = tokenizer, text = between),
                    "end_text": segments["end"],
                    "template": template,
                }
            elif group in SELF_SYSTEM_GROUPS:
                thinking = group in THINK_GROUPS
                system_parts = system_chat_segments(tokenizer = tokenizer, thinking = thinking)
                start_ids = encode(tokenizer = tokenizer, text = system_parts["system_start"])
                if meta_per_lang is None:
                    rows = [{"prompt_ids": start_ids} for _ in range(free_count[split])]
                else:
                    rows = meta_rows(
                        tokenizer = tokenizer,
                        start_ids = start_ids,
                        per_lang = meta_per_lang[split],
                        seed = f"{args.seed}-{group}-{split}-meta",
                    )
                templates[group] = {
                    "start_text": system_parts["system_start"],
                    "start_ids": start_ids,
                    # 系统提示以生成的 <|im_end|> 结束，其后补上的 token（用户轮开头）
                    "system_between_text": system_parts["system_between"],
                    "system_between_ids": encode(tokenizer = tokenizer, text = system_parts["system_between"]),
                    # 用户消息以生成的 <|im_end|> 结束，其后补上的 token（助手轮开头）
                    "between_text": system_parts["user_between"],
                    "between_ids": encode(tokenizer = tokenizer, text = system_parts["user_between"]),
                    "end_text": system_parts["end"],
                    "template": f"apply_chat_template(add_generation_prompt = True, enable_thinking = {thinking})，系统消息与用户消息都由模型自己写",
                }
                if meta_per_lang is not None:
                    templates[group]["meta"] = {
                        "version": META_VERSION,
                        "template": "系统提示由模型按元指令写出：用户轮为元指令，apply_chat_template(add_generation_prompt = True, "
                        "enable_thinking = False)，模型的回答就是系统提示；元指令这一轮不进入保存的样本",
                        "langs": list(LANGS),
                        "kinds": list(KINDS),
                        "domains": list(DOMAINS),
                    }
            else:
                with open(prefixes_dir / f"{split}.jsonl", encoding = "utf-8") as handle:
                    prefixes = [json.loads(line) for line in handle]
                skipped = 0
                for prefix in prefixes:
                    target = random.Random(f"{args.seed}-{prefix['prefix_id']}").randint(MIN_PREFIX_TOKENS, MAX_PREFIX_TOKENS)
                    # cont_raw 与 cont_chat 用同一段开头，切分结果只算一次
                    key = (split, prefix["prefix_id"])
                    if key not in cuts:
                        cuts[key] = cut_prefix(tokenizer = tokenizer, text = prefix["text"], target = target)
                    prefix_text = cuts[key]
                    if prefix_text is None:
                        skipped += 1
                        continue
                    if group == "cont_raw":
                        prompt_ids = encode(tokenizer = tokenizer, text = prefix_text)
                    else:
                        user = f"{CONTINUE_INSTRUCTION}\n\n{prefix_text}"
                        prompt_ids = encode(tokenizer = tokenizer, text = render_prompt(tokenizer = tokenizer, user = user))
                    meta = {key: prefix[key] for key in ("prefix_id", "lang", "source", "source_license", "source_ref", "doc_id")}
                    rows.append(
                        {
                            "prompt_ids": prompt_ids,
                            **meta,
                            "prefix_text": prefix_text,
                            "prefix_tokens": len(encode(tokenizer = tokenizer, text = prefix_text)),
                            "target_tokens": target,
                        }
                    )
                print(f"{group}/{split}: 文档 {len(prefixes)} 篇，开头不足 {MIN_PREFIX_TOKENS} token 跳过 {skipped} 篇", flush = True)
                if group == "cont_raw":
                    templates[group] = {"start_text": "文档开头（64–256 token，句末或行末切分）", "template": None}
                else:
                    templates[group] = {
                        "start_text": segments["user_start"],
                        "user_content": f"{CONTINUE_INSTRUCTION}\\n\\n<文档开头>",
                        "prompt_suffix_text": segments["between"],
                        "end_text": segments["end"],
                        "template": "apply_chat_template(add_generation_prompt = True, enable_thinking = False)，无系统消息",
                    }
            for number, row in enumerate(rows):
                row["prompt_index"] = number
            write_jsonl(path = replay_dir / group / split / "prompts.jsonl", rows = rows)
            print(f"{group}/{split}: 提示 {len(rows)} 条", flush = True)
    tmp = templates_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(templates, ensure_ascii = False, indent = 2) + "\n", encoding = "utf-8")
    tmp.replace(templates_path)


if __name__ == "__main__":
    main()
