"""第五组回放文本 prompt_chat 的提示：指令数据集中真人写的用户提问，参考模型写回答。

    python -m pipeline.replay.user_prompts --model <参考模型目录> --raw-root <$DATASET/text> --replay-dir <回放目录>

在 `envs/mllm-eval` 中运行（transformers 与 pyarrow，只用 CPU）。来源见 `datasets.md`：

- `CohereLabs/aya_dataset`（Apache-2.0，人写的指令与回答）：只用 `inputs` 字段作用户消息；`test` 划分作评测用，`train`
  划分作训练用。语言按 `language_code` 对应到项目的 18 种语言：阿拉伯语的标准语与各地方言（arb、ary、arz、ars、acq、ajp
  等）都算 `ar`，简体与繁体中文（zho）都算 `zh`，标准马来语 zsm 算 `ms`，Filipino 与 Tagalog 算 `fil`，伊朗波斯语 pes 与
  达里语 prs 算 `fa`；其他语言不用。
- `OpenAssistant/oasst2`（Apache-2.0）：每棵对话树只用第一条用户消息（`parent_id` 为空、`role` 为 prompter、
  `review_result` 为真、没有删除）；`validation` 划分作评测用，`train` 划分作训练用。语言取 `lang` 的主标签（pt-BR 算 `pt`）。

处理：提示文本原样使用（不改写、不按内容筛选）；按规范化文本（NFKC、忽略大小写、合并空白）去重，评测用出现过的提示不进
训练用；用参考模型的分词器（#77 修正后的切法）计 token 数，少于 3 个 token 的丢弃，多于 2,048 个 token 的丢弃并报告数量。
每个划分用固定种子打乱；某种语言超过该划分提示数的 25% 时，把它截到 25%（其余语言全部保留），再打乱一次写出。生成时按
文件顺序投放，任意前若干条的语言分布都与整个文件相近。

对话格式与 free_chat、cont_chat 的助手部分相同：`prompts.render_prompt` 渲染「用户消息 + 助手轮开头 + 空的思考块」，
没有系统消息，参考模型从这里写回答。输出 `<回放目录>/prompt_chat/<划分>/prompts.jsonl` 与 `prompts_stats.json`，这些提示的
`part` 为 `instruct`。

工具调用与编程的上下文（`--agent <目录>`，`sources_agent` 的输出，每行 `id`、`source`、`kind`、`messages`、`tools`、
`lang`）追加在同一个提示清单的后面，`part` 为 `agent`：用 chat template 渲染到最后一条用户或工具消息为止
（`apply_chat_template(messages, tools = tools, add_generation_prompt = True, enable_thinking = False)`），参考模型只写
下一轮助手回答（可以含工具调用），回答之前的全部内容都是提示。渲染后超过 12,288 个 token 的上下文丢弃并报告数量。
生成时用 `generate --plan prompt_chat:train@instruct,prompt_chat:train@agent` 按部分分别计目标。
"""

import argparse
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from pipeline.replay.prompts import chat_segments, encode, render_prompt
from pipeline.tokenizer_fix import check_tokenizer

GROUP = "prompt_chat"
LICENSE = "Apache-2.0"
AYA = "CohereLabs/aya_dataset"
OASST = "OpenAssistant/oasst2"
AYA_CODES = {
    "zho": "zh",
    "eng": "en",
    "jpn": "ja",
    "kor": "ko",
    "deu": "de",
    "fra": "fr",
    "rus": "ru",
    "por": "pt",
    "spa": "es",
    "ita": "it",
    "ind": "id",
    "zsm": "ms",
    "msa": "ms",
    "tha": "th",
    "vie": "vi",
    "fil": "fil",
    "tgl": "fil",
    "arb": "ar",
    "ary": "ar",
    "arz": "ar",
    "ars": "ar",
    "acq": "ar",
    "ajp": "ar",
    "apc": "ar",
    "acm": "ar",
    "aeb": "ar",
    "ayl": "ar",
    "tur": "tr",
    "pes": "fa",
    "prs": "fa",
}
OASST_CODES = {code: code for code in ("zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it", "id", "ms", "th", "vi", "ar", "tr", "fa")} | {"tl": "fil", "fil": "fil"}
MIN_TOKENS = 3
MAX_TOKENS = 2048
MAX_LANG_SHARE = 0.25
MAX_CONTEXT_TOKENS = 12288
_SPACES = re.compile(r"\s+")


def normalize(text):
    return _SPACES.sub(" ", unicodedata.normalize("NFKC", text)).strip().casefold()


def read_aya(raw_root, split):
    import pyarrow.parquet as pq

    path = Path(raw_root) / AYA / "data" / f"{split}-00000-of-00001.parquet"
    table = pq.read_table(path, columns = ["inputs", "language", "language_code"])
    for number, row in enumerate(table.to_pylist()):
        lang = AYA_CODES.get(row["language_code"])
        if lang is not None:
            yield {"lang": lang, "source": AYA, "source_lang": f"{row['language']} ({row['language_code']})", "source_ref": f"{split}:{number}", "text": row["inputs"] or ""}


def read_oasst(raw_root, split):
    import pyarrow.parquet as pq

    path = next((Path(raw_root) / OASST / "data").glob(f"{split}-*.parquet"))
    columns = ["message_id", "parent_id", "role", "lang", "review_result", "deleted", "text"]
    for row in pq.read_table(path, columns = columns).to_pylist():
        if row["parent_id"] is not None or row["role"] != "prompter" or row["review_result"] is not True or row["deleted"]:
            continue
        lang = OASST_CODES.get((row["lang"] or "").split("-")[0])
        if lang is not None:
            yield {"lang": lang, "source": OASST, "source_lang": row["lang"], "source_ref": row["message_id"], "text": row["text"] or ""}


def cap_languages(rows, share, rng):
    """某种语言超过 share 时截到 share：找最大的 N，使各语言取 min(数量, share × N) 之和等于 N。"""
    by_lang = {}
    for row in rows:
        by_lang.setdefault(row["lang"], []).append(row)
    counts = {lang: len(items) for lang, items in by_lang.items()}
    total = sum(counts.values())
    while True:
        cap = int(share * total)
        new_total = sum(min(count, cap) for count in counts.values())
        if new_total >= total:
            break
        total = new_total
    kept = []
    for lang, items in by_lang.items():
        kept.extend(items[:cap] if len(items) > cap else items)
    rng.shuffle(kept)
    return kept, cap


def append_agent(tokenizer, agent_dir, replay_dir, seed):
    """把工具调用与编程上下文追加到 prompt_chat 的提示清单（part 为 agent），编号接在已有提示之后。"""
    stats = {}
    for split in ("eval", "train"):
        source = Path(agent_dir) / f"{split}.jsonl"
        target = Path(replay_dir) / GROUP / split / "prompts.jsonl"
        with open(target, encoding = "utf-8") as handle:
            existing = [json.loads(line) for line in handle]
        if any(row.get("part") == "agent" for row in existing):
            print(f"{GROUP}/{split}: 已经追加过 agent 部分，跳过", flush = True)
            continue
        with open(source, encoding = "utf-8") as handle:
            contexts = [json.loads(line) for line in handle]
        random.Random(f"{seed}-{GROUP}-{split}-agent").shuffle(contexts)
        counter = Counter()
        rows = []
        for context in contexts:
            text = tokenizer.apply_chat_template(
                conversation = context["messages"],
                tools = context.get("tools") or None,
                tokenize = False,
                add_generation_prompt = True,
                enable_thinking = False,
            )
            prompt_ids = encode(tokenizer = tokenizer, text = text)
            if len(prompt_ids) > MAX_CONTEXT_TOKENS:
                counter["too_long"] += 1
                continue
            rows.append(
                {
                    "prompt_ids": prompt_ids,
                    "lang": context.get("lang"),
                    "source": context["source"],
                    "kind": context.get("kind"),
                    "source_ref": context["id"],
                    "context_tokens": len(prompt_ids),
                    "part": "agent",
                    "prompt_index": len(existing) + len(rows),
                }
            )
        with open(target, "a", encoding = "utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii = False) + "\n")
        stats[split] = {
            "contexts": len(rows),
            "dropped": dict(counter),
            "kind": dict(Counter(row["kind"] for row in rows)),
            "source": dict(Counter(row["source"] for row in rows)),
            "lang": dict(Counter(row["lang"] for row in rows)),
            "context_tokens_mean": round(sum(row["context_tokens"] for row in rows) / max(len(rows), 1), 1),
        }
        print(f"{GROUP}/{split} agent: {json.dumps(stats[split], ensure_ascii = False)}", flush = True)
    path = Path(replay_dir) / GROUP / "prompts_stats.json"
    all_stats = json.loads(path.read_text(encoding = "utf-8")) if path.exists() else {}
    all_stats["agent"] = stats
    path.write_text(json.dumps(all_stats, ensure_ascii = False, indent = 2) + "\n", encoding = "utf-8")


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required = True, help = "参考模型目录（取分词器与 chat template）")
    parser.add_argument("--raw-root", default = None, help = "下载的原始语料根目录，例如 $DATASET/text")
    parser.add_argument("--replay-dir", required = True)
    parser.add_argument("--agent", default = None, help = "工具调用与编程上下文目录（含 eval.jsonl、train.jsonl）；给出时只做追加")
    parser.add_argument("--seed", type = int, default = 0)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    check_tokenizer(tokenizer = tokenizer, source = args.model)
    replay_dir = Path(args.replay_dir)
    if args.agent:
        append_agent(
            tokenizer = tokenizer,
            agent_dir = args.agent,
            replay_dir = replay_dir,
            seed = args.seed,
        )
        return
    if not args.raw_root:
        parser.error("需要 --raw-root")
    sources = {
        "eval": [read_aya(raw_root = args.raw_root, split = "test"), read_oasst(raw_root = args.raw_root, split = "validation")],
        "train": [read_aya(raw_root = args.raw_root, split = "train"), read_oasst(raw_root = args.raw_root, split = "train")],
    }
    seen = set()
    stats = {}
    for split in ("eval", "train"):
        counter = Counter()
        rows = []
        for reader in sources[split]:
            for row in reader:
                key = normalize(text = row["text"])
                if not key:
                    counter["empty"] += 1
                    continue
                if key in seen:
                    counter["duplicate"] += 1
                    continue
                seen.add(key)
                n_tokens = len(encode(tokenizer = tokenizer, text = row["text"]))
                if n_tokens < MIN_TOKENS:
                    counter["too_short"] += 1
                    continue
                if n_tokens > MAX_TOKENS:
                    counter["too_long"] += 1
                    continue
                rows.append(dict(row, user_tokens = n_tokens))
        rng = random.Random(f"{args.seed}-{GROUP}-{split}")
        rng.shuffle(rows)
        before = Counter(row["lang"] for row in rows)
        rows, cap = cap_languages(rows = rows, share = MAX_LANG_SHARE, rng = rng)
        out_rows = []
        for number, row in enumerate(rows):
            prompt_ids = encode(tokenizer = tokenizer, text = render_prompt(tokenizer = tokenizer, user = row["text"]))
            out_rows.append(
                {
                    "prompt_ids": prompt_ids,
                    "lang": row["lang"],
                    "source": row["source"],
                    "source_license": LICENSE,
                    "source_lang": row["source_lang"],
                    "source_ref": row["source_ref"],
                    "user_text": row["text"],
                    "user_tokens": row["user_tokens"],
                    "part": "instruct",
                    "prompt_index": number,
                }
            )
        path = replay_dir / GROUP / split / "prompts.jsonl"
        path.parent.mkdir(parents = True, exist_ok = True)
        with open(path, "w", encoding = "utf-8") as handle:
            for row in out_rows:
                handle.write(json.dumps(row, ensure_ascii = False) + "\n")
        stats[split] = {
            "prompts": len(out_rows),
            "dropped": dict(counter),
            "lang_cap": cap,
            "lang_before_cap": dict(sorted(before.items())),
            "lang": dict(sorted(Counter(row["lang"] for row in out_rows).items())),
            "source": dict(Counter(row["source"] for row in out_rows)),
            "lang_source": dict(sorted(Counter(f"{row['lang']}/{row['source'].split('/')[0]}" for row in out_rows).items())),
            "user_tokens_mean": round(sum(row["user_tokens"] for row in out_rows) / max(len(out_rows), 1), 1),
        }
        print(f"{GROUP}/{split}: {json.dumps(stats[split], ensure_ascii = False)}", flush = True)
    (replay_dir / GROUP / "prompts_stats.json").write_text(json.dumps(stats, ensure_ascii = False, indent = 2) + "\n", encoding = "utf-8")
    templates_path = replay_dir / "templates.json"
    templates = json.loads(templates_path.read_text(encoding = "utf-8")) if templates_path.exists() else {}
    segments = chat_segments(tokenizer = tokenizer)
    templates[GROUP] = {
        "start_text": segments["user_start"],
        "user_content": "<数据集中的用户提问>",
        "prompt_suffix_text": segments["between"],
        "end_text": segments["end"],
        "template": "apply_chat_template(add_generation_prompt = True, enable_thinking = False)，无系统消息",
        "sources": [AYA, OASST],
    }
    tmp = templates_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(templates, ensure_ascii = False, indent = 2) + "\n", encoding = "utf-8")
    tmp.replace(templates_path)


if __name__ == "__main__":
    main()
