"""把各来源的候选合并成每种语言的条目文件：跨来源去重、再次排除、按长度档配额抽样、编号、导出审阅样本。

命令行入口：

    python -m pipeline.text.build --cand-dir <extract 输出目录> --exclusion <exclusion.jsonl> \
        --out-dir <data/T03/part1>

条目字段见 `pipeline/CONVENTIONS.md` 的「文本条目」。朗读稿、规范化状态等 T04 的字段不写。
网页类来源（T03b）把自己的候选放进同一个 `--cand-dir` 下的新子目录，并在 `SOURCE_INFO` 登记即可复用。
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .dedup import Deduper
from .exclusion import is_excluded, load_blocklist, load_exclusion
from .filters import check_text
from .langs import LANGS
from .lengths import BINS
from .sources import common_voice, tatoeba, web, wikipedia_paragraphs

# 候选目录名 -> (datasets.md 中的来源名, entry_id 里的短名, 许可)
SOURCE_INFO = {
    "common_voice": (common_voice.SOURCE_NAME, common_voice.SOURCE_SLUG, common_voice.SOURCE_LICENSE),
    "tatoeba": (tatoeba.SOURCE_NAME, tatoeba.SOURCE_SLUG, tatoeba.SOURCE_LICENSE),
    "wikipedia": (
        wikipedia_paragraphs.SOURCE_NAME,
        wikipedia_paragraphs.SOURCE_SLUG,
        wikipedia_paragraphs.SOURCE_LICENSE,
    ),
    # 网页与新闻来源（T03b）：HPLT 按体裁分三个候选目录，共用来源名和 entry_id 短名。
    **{
        f"hplt_{genre}": (web.HPLT_NAME, web.HPLT_SLUG, web.HPLT_LICENSE)
        for genre in ("na", "op", "in")
    },
    "fw2hq": (web.FW2_NAME, web.FW2_SLUG, web.FW2_LICENSE),
    "fwedu": (web.FWEDU_NAME, web.FWEDU_SLUG, web.FWEDU_LICENSE),
    "wikinews": (web.WIKINEWS_NAME, web.WIKINEWS_SLUG, web.WIKINEWS_LICENSE),
}
# 个别语言的许可与来源默认值不同。
LICENSE_BY_LANG = {("wikinews", lang): name for lang, name in web.WIKINEWS_LICENSE_BY_LANG.items()}

DEFAULT_QUOTAS = {"short": 10000, "medium": 12000, "long": 8000}
DEFAULT_MAX_TOTAL = 30000


def read_candidates(path: Path) -> list[dict]:
    with open(path, encoding = "utf-8") as handle:
        return [json.loads(line) for line in handle]


def interleave(groups: list[list[dict]], *, limit: int) -> list[dict]:
    """各来源轮流取，直到取满 `limit` 或全部取完；小来源不会被大来源淹没。"""
    picked: list[dict] = []
    cursors = [0] * len(groups)
    while len(picked) < limit:
        progressed = False
        for index, group in enumerate(groups):
            if cursors[index] < len(group) and len(picked) < limit:
                picked.append(group[cursors[index]])
                cursors[index] += 1
                progressed = True
        if not progressed:
            break
    return picked


def select_entries(
    by_source: dict[str, list[dict]],
    *,
    quotas: dict[str, int],
    max_total: int,
    rng: random.Random,
) -> list[dict]:
    """按长度档配额选条目；配额没用满时，用各档剩余候选补到 `max_total`。"""
    pools: dict[str, dict[str, list[dict]]] = {name: {} for name in BINS}
    for source, rows in by_source.items():
        for bin_name in BINS:
            pool = [row for row in rows if row["length_bin"] == bin_name]
            rng.shuffle(pool)
            pools[bin_name][source] = pool

    chosen: list[dict] = []
    leftovers: dict[str, list[list[dict]]] = {}
    for bin_name in BINS:
        groups = [pools[bin_name][source] for source in sorted(pools[bin_name])]
        taken = interleave(groups, limit = min(quotas[bin_name], max_total - len(chosen)))
        chosen.extend(taken)
        taken_ids = {id(row) for row in taken}
        leftovers[bin_name] = [[row for row in group if id(row) not in taken_ids] for group in groups]

    for bin_name in ("medium", "short", "long"):
        room = max_total - len(chosen)
        if room <= 0:
            break
        chosen.extend(interleave(leftovers[bin_name], limit = room))
    return chosen


def to_entry(row: dict, *, source_dir: str, index: int) -> dict:
    name, slug, license_name = SOURCE_INFO[source_dir]
    license_name = LICENSE_BY_LANG.get((source_dir, row["lang"]), license_name)
    return {
        "entry_id": f"{row['lang']}-{slug}-{index:06d}",
        "lang": row["lang"],
        "source": name,
        "source_license": license_name,
        "source_ref": row["ref"],
        "original": row["text"],
        "n_chars": len(row["text"]),
        "length_bin": row["length_bin"],
        "est_duration_s": row["est_duration_s"],
    }


def build_language(
    lang: str,
    *,
    cand_dir: Path,
    sources: list[str],
    exclusion: dict[str, set[str]],
    quotas: dict[str, int],
    max_total: int,
    seed: int,
) -> list[dict]:
    """合并一种语言的各来源候选，返回已编号的条目列表。"""
    rng = random.Random(f"{seed}-{lang}")
    blocklist = load_blocklist()
    deduper = Deduper(lang = lang)
    by_source: dict[str, list[dict]] = {}
    for source in sources:
        path = cand_dir / source / f"{lang}.jsonl"
        if not path.exists():
            continue
        rows = []
        for row in read_candidates(path):
            if row["ref"] in blocklist:
                continue
            if check_text(row["text"], lang = lang) is not None:
                continue  # 过滤规则可能在抽取之后又补充过，合并时再检查一遍
            if is_excluded(row["text"], lang = lang, exclusion = exclusion):
                continue
            if deduper.add(row["text"]):
                row["_source"] = source
                rows.append(row)
        by_source[source] = rows

    chosen = select_entries(by_source, quotas = quotas, max_total = max_total, rng = rng)
    rng.shuffle(chosen)
    counters: Counter = Counter()
    entries = []
    for row in chosen:
        slug = SOURCE_INFO[row["_source"]][1]
        counters[slug] += 1
        entries.append(to_entry(row, source_dir = row["_source"], index = counters[slug]))
    return entries


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding = "utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii = False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cand-dir", required = True)
    parser.add_argument("--exclusion", required = True)
    parser.add_argument("--out-dir", required = True)
    parser.add_argument("--lang", nargs = "+", default = ["all"])
    parser.add_argument("--sources", nargs = "+", default = list(SOURCE_INFO))
    parser.add_argument("--quota-short", type = int, default = DEFAULT_QUOTAS["short"])
    parser.add_argument("--quota-medium", type = int, default = DEFAULT_QUOTAS["medium"])
    parser.add_argument("--quota-long", type = int, default = DEFAULT_QUOTAS["long"])
    parser.add_argument("--max-total", type = int, default = DEFAULT_MAX_TOTAL)
    parser.add_argument("--review-size", type = int, default = 30)
    parser.add_argument("--seed", type = int, default = 0)
    args = parser.parse_args()

    langs = LANGS if args.lang == ["all"] else args.lang
    exclusion = load_exclusion(args.exclusion)
    out_dir = Path(args.out_dir)
    (out_dir / "review").mkdir(parents = True, exist_ok = True)
    quotas = {"short": args.quota_short, "medium": args.quota_medium, "long": args.quota_long}

    summary: dict = {}
    for lang in langs:
        entries = build_language(
            lang,
            cand_dir = Path(args.cand_dir),
            sources = args.sources,
            exclusion = exclusion,
            quotas = quotas,
            max_total = args.max_total,
            seed = args.seed,
        )
        write_jsonl(out_dir / f"{lang}.jsonl", entries)
        review = random.Random(f"review-{args.seed}-{lang}").sample(entries, k = min(args.review_size, len(entries)))
        write_jsonl(out_dir / "review" / f"{lang}.jsonl", review)

        table: dict = defaultdict(lambda: defaultdict(int))
        hours = 0.0
        for entry in entries:
            table[entry["source"]][entry["length_bin"]] += 1
            hours += entry["est_duration_s"] / 3600
        summary[lang] = {
            "total": len(entries),
            "est_hours": round(hours, 1),
            "by_source_bin": {source: dict(bins) for source, bins in table.items()},
        }
        print(f"[{lang}] {len(entries)} 条，约 {hours:.1f} 小时 {summary[lang]['by_source_bin']}", flush = True)

    with open(out_dir / "summary.json", "w", encoding = "utf-8") as handle:
        json.dump(summary, handle, ensure_ascii = False, indent = 2)


if __name__ == "__main__":
    main()
