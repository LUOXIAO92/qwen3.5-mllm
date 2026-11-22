"""为四组回放文本写出提示清单 `<回放目录>/<组>/<划分>/prompts.jsonl`，并把各组的起始 token 记在 `<回放目录>/templates.json`。

    python -m pipeline.replay.prompts --model <参考模型目录> --replay-dir <回放目录> --free-count eval=8000,train=60000

四组的开头：

| 组 | 提示 |
| --- | --- |
| free_raw | 一个换行 `\\n`（模拟 BOS）；`--free-raw-start random` 时改为从普通词表中均匀抽取的一个 token。第一个生成 token 由 `first_token` 抽取后写进提示清单的 `first` 字段 |
| free_chat | 聊天模板的用户轮开头 `<|im_start|>user\\n`。模型写完用户消息（以 `<|im_end|>` 结束）后，补上助手轮开头和空的思考块，再让模型写回答 |
| cont_raw | 文档开头（64–256 token），不加模板 |
| cont_chat | 用户轮内容为续写指令加空行加同一段文档开头，助手轮开头和空的思考块之后由模型续写 |

free_raw 默认用换行开头。试运行中两种开头各采样 600 条：换行开头 58% 的样本出现循环，前 20 万 token 中不同 4-gram
的比例为 0.66，开头 8 个 token 与其他样本相同的占 2.5%，一半像代码；随机 token 开头分别为 51%、0.61、0%，四分之一像代码。
两者的重复程度相近，随机开头没有更少重复，所以保留换行开头。

聊天模板与 ASR 训练相同：`pipeline/train/data.py` 的 `build_training_text` 用 `apply_chat_template`，
`add_generation_prompt = False`、`enable_thinking = False`，没有系统消息，渲染结果是
`<|im_start|>user\\n{用户}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n{助手}<|im_end|>\\n`。
这里用同一个函数渲染带占位符的对话，再按占位符切出模板的各段，所以模板写法不在代码里重复。
训练时只对助手内容和结尾的 `<|im_end|>` 计算损失，free_chat 的用户消息也是模型生成的，是否计算损失由训练决定。

文档开头的长度：每篇文档先在 [64, 256] 中均匀抽一个目标 token 数，再在不超过目标的位置中选最靠后的句末或行末
（没有时退到词边界），切下的文本去掉结尾空白后重新分词。至少 64 token 才能定下语言、话题和文体；
至多 256 token 让人写的前缀只占样本的一小部分，样本的主体是参考模型自己写的续写。
"""

import argparse
import json
import random
import unicodedata
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from pipeline.replay.first_token import first_token_logprobs, sample_first
from pipeline.tokenizer_fix import check_tokenizer

GROUPS = ("free_raw", "free_chat", "cont_raw", "cont_chat")
SPLITS = ("eval", "train")
CONTINUE_INSTRUCTION = "Continue writing the following text."
MIN_PREFIX_TOKENS = 64
MAX_PREFIX_TOKENS = 256
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


def chat_segments(tokenizer):
    """按占位符切出模板：用户轮开头、用户与助手之间、助手之后。"""
    text = render_training(tokenizer = tokenizer, user = _USER, assistant = _ASSISTANT)
    head, rest = text.split(_USER)
    middle, tail = rest.split(_ASSISTANT)
    prompt = render_prompt(tokenizer = tokenizer, user = _USER)
    assert prompt == head + _USER + middle, "加生成提示的渲染与训练渲染的前半段不一致"
    return {"user_start": head, "between": middle, "end": tail}


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


def write_jsonl(path, rows):
    path.parent.mkdir(parents = True, exist_ok = True)
    with open(path, "w", encoding = "utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii = False) + "\n")


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required = True, help = "参考模型目录（取分词器与 chat template）")
    parser.add_argument("--replay-dir", required = True, help = "回放目录；续写组读取其中的 prefixes/<划分>.jsonl")
    parser.add_argument("--groups", nargs = "+", default = list(GROUPS))
    parser.add_argument("--free-count", default = "eval=8000,train=60000", help = "自由生成两组每个划分的提示条数（生成在达到 token 目标时停止）")
    parser.add_argument("--free-raw-start", choices = ["newline", "random"], default = "newline")
    parser.add_argument("--seed", type = int, default = 0)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # 文档开头按原始 Qwen3.5 的规则分词（#77）
    check_tokenizer(tokenizer = tokenizer, source = args.model)
    replay_dir = Path(args.replay_dir)
    segments = chat_segments(tokenizer = tokenizer)
    free_count = {key: int(value) for key, value in (item.split("=") for item in args.free_count.split(","))}
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
            elif group == "free_chat":
                start_ids = encode(tokenizer = tokenizer, text = segments["user_start"])
                rows = [{"prompt_ids": start_ids} for _ in range(free_count[split])]
                templates[group] = {
                    "start_text": segments["user_start"],
                    "start_ids": start_ids,
                    # 用户消息以生成的 <|im_end|> 结束，其后补上的 token
                    "between_text": segments["between"].removeprefix("<|im_end|>"),
                    "between_ids": encode(tokenizer = tokenizer, text = segments["between"].removeprefix("<|im_end|>")),
                    "end_text": segments["end"],
                    "template": "apply_chat_template(add_generation_prompt = False, enable_thinking = False)，无系统消息",
                }
            else:
                with open(replay_dir / "prefixes" / f"{split}.jsonl", encoding = "utf-8") as handle:
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
    templates_path.write_text(json.dumps(templates, ensure_ascii = False, indent = 2) + "\n", encoding = "utf-8")


if __name__ == "__main__":
    main()
