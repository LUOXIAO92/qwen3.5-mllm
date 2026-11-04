"""抽取候选条目：来源读取 → 清理 → 过滤 → 去重 → FLEURS 排除 → 分档抽样 → 语言检测。

命令行入口：

    python -m pipeline.text.extract --source tatoeba --lang all \
        --raw-dir <Tatoeba 导出目录> --exclusion <exclusion.jsonl> \
        --langid-model <GlotLID model_v3.bin> --out-dir <输出目录>

输出 `<out-dir>/<source>/<lang>.jsonl`（每行一个候选）和 `<lang>.stats.json`（各步丢弃数）。
网页类来源只需要提供一个产出 (文本, 定位信息) 的生成器，再调用 `extract_candidates`。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from urllib.parse import urlparse

from .dedup import Deduper
from .exclusion import is_excluded, load_blocklist, load_exclusion
from .filters import FilterOptions, check_text, default_options
from .langs import LANGS, NO_SPACE_LANGS, WIKI_PARAGRAPH_LANGS
from .langid import LangIdentifier
from .lengths import BINS, estimate_duration, length_bin
from .normalize import clean_text
from .spans import build_spans, clean_runs

DEFAULT_CAPS = {"short": 12000, "medium": 14000, "long": 10000}
DEFAULT_BIN_WEIGHTS = {"short": 0.25, "medium": 0.35, "long": 0.4}


class BinReservoir:
    """每个长度档一个蓄水池抽样：容量为 `cap`，每条进入的概率均等。"""

    def __init__(self, *, caps: dict[str, int], rng: random.Random) -> None:
        self.caps = caps
        self.rng = rng
        self.items: dict[str, list] = {name: [] for name in BINS}
        self.seen: Counter = Counter()

    def add(self, bin_name: str, item) -> None:
        self.seen[bin_name] += 1
        pool = self.items[bin_name]
        if len(pool) < self.caps[bin_name]:
            pool.append(item)
            return
        slot = self.rng.randrange(self.seen[bin_name])
        if slot < self.caps[bin_name]:
            pool[slot] = item


def extract_candidates(
    records: Iterable[tuple[str, str]],
    *,
    lang: str,
    exclusion: dict[str, set[str]],
    identifier: LangIdentifier | None,
    caps: dict[str, int] | None = None,
    seed: int = 0,
    options: FilterOptions | None = None,
    paragraphs: bool = False,
    bin_weights: dict[str, float] | None = None,
    max_per_prefix: int = 0,
    prefix_len: int = 12,
    blocklist: frozenset[str] = frozenset(),
    min_prob: float = 0.5,
    extra_labels: frozenset[str] = frozenset(),
    stats: Counter | None = None,
    max_per_doc: int = 0,
    max_per_domain: int = 0,
) -> list[dict]:
    """过滤并抽样一批 (文本, 定位信息)，返回候选字典列表。

    `paragraphs = True` 时每条记录是一个段落，先切句、剔除不干净的句子，再拼成目标长度的片段。
    `identifier` 为 None 时跳过语言检测（单元测试用）。
    网页来源的定位信息是 `<URL>#<段落序号>`：`max_per_doc`、`max_per_domain` 限制同一文档、同一域名通过规则的条数
    （0 为不限），防止个别站点或文档占满样本。
    """
    stats = stats if stats is not None else Counter()
    rng = random.Random(seed)
    options = options or default_options(lang)
    sentence_options = FilterOptions(
        require_terminal = options.require_terminal,
        min_words = 1,
        min_chars = 25 if (lang == "th" and options.web) else 1,  # 泰语网页按 crfcut 切出的片段常是短语，太短的丢弃
        max_chars = options.max_chars,
        max_digit_ratio = options.max_digit_ratio,
        max_symbols = options.max_symbols,
        web = options.web,
    )
    deduper = Deduper(lang = lang, max_per_prefix = max_per_prefix, prefix_len = prefix_len)
    reservoir = BinReservoir(caps = caps or DEFAULT_CAPS, rng = rng)
    per_doc: Counter = Counter()
    per_domain: Counter = Counter()

    def sentence_ok(sentence: str) -> bool:
        return check_text(sentence, lang = lang, options = sentence_options) is None

    for raw, ref in records:
        stats["records"] += 1
        if ref in blocklist:
            stats["drop_blocklist"] += 1
            continue
        text = clean_text(raw, lang = lang)
        if paragraphs:
            runs = clean_runs(text, lang = lang, sentence_ok = sentence_ok)
            pieces = build_spans(runs, lang = lang, rng = rng, bin_weights = bin_weights or DEFAULT_BIN_WEIGHTS)
            if not pieces:
                stats["drop_no_clean_sentence"] += 1
            pieces = [(piece, f"{ref}@{index}") for index, piece in enumerate(pieces)]
        else:
            pieces = [(text, ref)]
        for piece, piece_ref in pieces:
            reason = check_text(piece, lang = lang, options = options)
            if reason:
                stats[f"drop_{reason}"] += 1
                continue
            duration = estimate_duration(piece, lang = lang)
            bin_name = length_bin(duration)
            if bin_name is None:
                stats["drop_duration"] += 1
                continue
            if not deduper.add(piece):
                stats["drop_duplicate"] += 1
                continue
            if is_excluded(piece, lang = lang, exclusion = exclusion):
                stats["drop_fleurs"] += 1
                continue
            doc_key = ref.rsplit("#", 1)[0]
            domain_key = urlparse(doc_key).netloc or doc_key
            if max_per_doc and per_doc[doc_key] >= max_per_doc:
                stats["drop_doc_cap"] += 1
                continue
            if max_per_domain and per_domain[domain_key] >= max_per_domain:
                stats["drop_domain_cap"] += 1
                continue
            per_doc[doc_key] += 1
            per_domain[domain_key] += 1
            reservoir.add(bin_name, (piece, piece_ref, bin_name, duration))
            stats["passed_rules"] += 1

    candidates = []
    for bin_name in BINS:
        stats[f"sampled_{bin_name}"] = len(reservoir.items[bin_name])
        for piece, piece_ref, _, duration in reservoir.items[bin_name]:
            if identifier is not None and not identifier.is_lang(
                piece,
                lang = lang,
                min_prob = min_prob,
                extra_labels = extra_labels,
            ):
                stats["drop_langid"] += 1
                continue
            candidates.append(
                {
                    "lang": lang,
                    "text": piece,
                    "ref": piece_ref,
                    "length_bin": bin_name,
                    "est_duration_s": round(duration, 1),
                }
            )
            stats[f"kept_{bin_name}"] += 1
    rng.shuffle(candidates)
    return candidates


def build_reader(args: argparse.Namespace, *, lang: str) -> tuple[Callable[[], Iterable[tuple[str, str]]], bool, int]:
    """按 `--source` 返回 (读取函数, 是否段落类, 同开头上限)。"""
    if args.source == "common_voice":
        from .sources.common_voice import read_common_voice

        return lambda: read_common_voice(repo_dir = args.raw_dir, lang = lang), False, 1
    if args.source == "tatoeba":
        from .sources.tatoeba import read_tatoeba

        return lambda: read_tatoeba(raw_dir = args.raw_dir, lang = lang, seed = args.seed), False, 1
    if args.source == "wikipedia":
        from .sources.wikipedia_paragraphs import read_wikipedia_paragraphs

        config = f"{WIKI_PARAGRAPH_LANGS[lang]}-{args.wiki_version}"
        return (
            lambda: read_wikipedia_paragraphs(
                config = config,
                n_shards = args.wiki_shards,
                row_groups_per_shard = args.wiki_row_groups,
                seed = args.seed,
            ),
            True,
            3,
        )
    raise ValueError(f"未知来源：{args.source}")


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required = True, choices = ["common_voice", "tatoeba", "wikipedia"])
    parser.add_argument("--lang", nargs = "+", default = ["all"], help = "语言代码，或 all")
    parser.add_argument("--raw-dir", default = None, help = "来源原始数据目录（common_voice 为仓库根目录）")
    parser.add_argument("--exclusion", required = True, help = "FLEURS 排除清单 JSONL")
    parser.add_argument("--langid-model", required = True, help = "GlotLID model_v3.bin")
    parser.add_argument("--out-dir", required = True)
    parser.add_argument("--cap-short", type = int, default = DEFAULT_CAPS["short"])
    parser.add_argument("--cap-medium", type = int, default = DEFAULT_CAPS["medium"])
    parser.add_argument("--cap-long", type = int, default = DEFAULT_CAPS["long"])
    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument("--min-prob", type = float, default = 0.5)
    parser.add_argument("--extra-labels", nargs = "*", default = [], help = "额外接受的 GlotLID 标签，如 ind_Latn（仅用于来源自带语言标注可靠的场合）")
    parser.add_argument("--wiki-version", default = "20260607-v1.2.1")
    parser.add_argument("--wiki-shards", type = int, default = 3)
    parser.add_argument("--wiki-row-groups", type = int, default = 1)
    args = parser.parse_args()

    langs = LANGS if args.lang == ["all"] else args.lang
    exclusion = load_exclusion(args.exclusion)
    blocklist = frozenset(load_blocklist())
    identifier = LangIdentifier(model_path = args.langid_model)
    caps = {"short": args.cap_short, "medium": args.cap_medium, "long": args.cap_long}
    out_root = Path(args.out_dir) / args.source
    out_root.mkdir(parents = True, exist_ok = True)

    for lang in langs:
        if args.source == "wikipedia" and lang not in WIKI_PARAGRAPH_LANGS:
            print(f"[{lang}] 该来源不覆盖，跳过", flush = True)
            continue
        reader, paragraphs, max_per_prefix = build_reader(args, lang = lang)
        options = default_options(lang)
        if args.source == "common_voice" and lang == "th":
            # 泰语 Common Voice 没有句末标点，12–20 字的多是标题、地名和短语；抽样看 20 字以上才多为完整句。
            options = dataclasses.replace(options, min_chars = 20)
        stats: Counter = Counter()
        candidates = extract_candidates(
            reader(),
            lang = lang,
            exclusion = exclusion,
            identifier = identifier,
            caps = caps,
            seed = args.seed,
            paragraphs = paragraphs,
            options = options,
            max_per_prefix = max_per_prefix,
            prefix_len = 12 if paragraphs else (14 if lang in NO_SPACE_LANGS else 28),
            blocklist = blocklist,
            min_prob = args.min_prob,
            extra_labels = frozenset(args.extra_labels),
            stats = stats,
        )
        with open(out_root / f"{lang}.jsonl", "w", encoding = "utf-8") as handle:
            for row in candidates:
                handle.write(json.dumps(row, ensure_ascii = False) + "\n")
        with open(out_root / f"{lang}.stats.json", "w", encoding = "utf-8") as handle:
            json.dump(dict(stats), handle, ensure_ascii = False, indent = 2, sort_keys = True)
        print(f"[{lang}] {args.source}: 候选 {len(candidates)}，统计 {dict(stats)}", flush = True)


if __name__ == "__main__":
    main()
