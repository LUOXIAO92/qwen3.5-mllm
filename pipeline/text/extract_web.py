"""从网页与新闻来源抽取候选条目（T03b）。

    python -m pipeline.text.extract_web --source hplt --lang all --raw-root <$DATASET/text> \
        --exclusion <exclusion.jsonl> --langid-model <GlotLID model_v3.bin> --out-dir <候选目录> --jobs 6

输出 `<out-dir>/<候选目录名>/<lang>.jsonl` 和 `<lang>.stats.json`，格式与 `extract` 一致，`build` / `merge` 直接读取。
候选目录名：HPLT 按体裁分三个（`hplt_na` 叙事与新闻、`hplt_op` 观点、`hplt_in` 说明），
其余为 `fw2hq`、`fwedu`、`wikinews`。合并时每个目录是一路来源，轮流取样，体裁因此大致均衡。

网页文本先在文档级按体裁标签、机翻概率、聚类大小、语种得分粗筛，再按段落切句，句子逐条过 `filters.check_text`
（`web = True`，含 `web_rules` 的样板、社交、电话、价格等规则）；每个域名和每篇文档限制条数。
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

from .exclusion import load_blocklist, load_exclusion
from .extract import extract_candidates
from .fetch_web import FINEWEB2_CONFIGS, HPLT_CODES, WIKINEWS_LANGS
from .filters import default_options
from .langid import LangIdentifier
from .langs import GLOTLID_LABELS, LANGS
from .sources import web

# 网页来源偏重中长档：短句已由 Common Voice 和 Tatoeba 覆盖。
WEB_BIN_WEIGHTS = {"short": 0.15, "medium": 0.4, "long": 0.45}
WEB_CAPS = {"short": 4000, "medium": 10000, "long": 8000}


def candidate_dirs(source: str) -> list[str]:
    if source == "hplt":
        return [f"hplt_{genre.lower()}" for genre in web.HPLT_GENRES]
    return {"fineweb2hq": ["fw2hq"], "fineweb-edu": ["fwedu"], "wikinews": ["wikinews"]}[source]


def make_reader(*, source: str, directory: str, lang: str, raw_root: Path):
    """返回 (读取函数, 是否可用)。"""
    if source == "hplt":
        code = HPLT_CODES[lang]
        files = web.list_files(raw_root / "HPLT" / "HPLT3.0" / code / "*.jsonl.zst")
        genre = directory.split("_", 1)[1]
        glotlid = code  # HPLT 自己的语言代码（中文为 cmn_Hans，波斯语为 pes_Arab）
        return (lambda: web.read_hplt(files = files, lang = lang, genre = genre, glotlid_code = glotlid)), bool(files)
    if source == "fineweb2hq":
        if lang not in FINEWEB2_CONFIGS:
            return None, False
        files = web.list_files(raw_root / "epfml" / "FineWeb2-HQ-Plus" / FINEWEB2_CONFIGS[lang] / "subset-*.parquet")
        return (lambda: web.read_parquet_docs(files = files, lang = lang)), bool(files)
    if source == "fineweb-edu":
        if lang != "en":
            return None, False
        files = web.list_files(raw_root / "HuggingFaceFW" / "fineweb-edu" / "sample-10BT" / "subset-*.parquet")
        return (lambda: web.read_parquet_docs(files = files, lang = lang, min_language_score = 0.95, min_int_score = 3)), bool(files)
    if source == "wikinews":
        if lang not in WIKINEWS_LANGS:
            return None, False
        path = raw_root / "wikimedia" / "wikinews" / lang / f"{lang}wikinews-latest-pages-articles.xml.bz2"
        return (lambda: web.read_wikinews(path = str(path), lang = lang)), path.exists()
    raise ValueError(source)


def run_one(job: dict) -> str:
    """一个 (语言, 候选目录) 任务：读取、过滤、抽样、语言检测，写出候选文件。"""
    lang, source, directory = job["lang"], job["source"], job["directory"]
    reader, available = make_reader(source = source, directory = directory, lang = lang, raw_root = Path(job["raw_root"]))
    if not available:
        return f"[{lang}] {directory}: 没有原始数据，跳过"
    identifier = LangIdentifier(model_path = job["langid_model"])
    exclusion = load_exclusion(job["exclusion"])
    blocklist = frozenset(load_blocklist())
    options = dataclasses.replace(default_options(lang), web = True)
    stats: Counter = Counter()
    records = reader()
    if job["max_records"]:
        records = itertools.islice(records, job["max_records"])
    candidates = extract_candidates(
        records,
        lang = lang,
        exclusion = exclusion,
        identifier = identifier,
        caps = WEB_CAPS,
        seed = job["seed"],
        paragraphs = True,
        options = options,
        bin_weights = WEB_BIN_WEIGHTS,
        max_per_prefix = 2,
        prefix_len = 14 if lang in {"zh", "ja", "th"} else 24,
        blocklist = blocklist,
        min_prob = 0.5,
        stats = stats,
        max_per_doc = job["max_per_doc"],
        max_per_domain = job["max_per_domain"],
    )
    out_root = Path(job["out_dir"]) / directory
    out_root.mkdir(parents = True, exist_ok = True)
    with open(out_root / f"{lang}.jsonl", "w", encoding = "utf-8") as handle:
        for row in candidates:
            handle.write(json.dumps(row, ensure_ascii = False) + "\n")
    with open(out_root / f"{lang}.stats.json", "w", encoding = "utf-8") as handle:
        json.dump(dict(stats), handle, ensure_ascii = False, indent = 2, sort_keys = True)
    bins = Counter(row["length_bin"] for row in candidates)
    return f"[{lang}] {directory}: 候选 {len(candidates)} {dict(bins)}，记录 {stats['records']}"


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required = True, choices = ["hplt", "fineweb2hq", "fineweb-edu", "wikinews"])
    parser.add_argument("--lang", nargs = "+", default = ["all"])
    parser.add_argument("--raw-root", required = True, help = "原始数据的文本根目录，例如 $DATASET/text")
    parser.add_argument("--exclusion", required = True)
    parser.add_argument("--langid-model", required = True)
    parser.add_argument("--out-dir", required = True)
    parser.add_argument("--jobs", type = int, default = 4)
    parser.add_argument("--max-per-doc", type = int, default = 4)
    parser.add_argument("--max-per-domain", type = int, default = 40)
    parser.add_argument("--max-records", type = int, default = 0, help = "每路来源最多读取的段落数（泰语切句较慢时用），0 为不限")
    parser.add_argument("--seed", type = int, default = 0)
    args = parser.parse_args()

    langs = LANGS if args.lang == ["all"] else args.lang
    jobs = [
        {
            "lang": lang,
            "directory": directory,
            "source": args.source,
            "raw_root": args.raw_root,
            "exclusion": args.exclusion,
            "langid_model": args.langid_model,
            "out_dir": args.out_dir,
            "seed": args.seed,
            "max_per_doc": args.max_per_doc,
            "max_per_domain": args.max_per_domain,
            "max_records": args.max_records,
        }
        for lang in langs
        for directory in candidate_dirs(args.source)
    ]
    with get_context("spawn").Pool(processes = min(args.jobs, len(jobs))) as pool:
        for message in pool.imap_unordered(run_one, jobs):
            print(message, flush = True)


if __name__ == "__main__":
    main()
