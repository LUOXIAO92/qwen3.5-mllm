"""从为 TTS 下载的纯文本语料中抽取文档开头，作为续写两组（cont_raw、cont_chat）的提示。

    python -m pipeline.replay.prefixes --raw-root <$DATASET/text> --exclusion <exclusion.jsonl> \
        --out-dir <回放目录>/prefixes --eval-per-lang 150 --train-per-lang 2000

在 `envs/t03` 中运行（需要 pyarrow、mwparserfromhell 和命令行工具 zstd）。来源与语言的对应见 `datasets.md`：
HPLT 3.0 覆盖全部 18 种语言，FineWeb2-HQ-Plus 覆盖英语以外的 17 种，英语改用 fineweb-edu，Wikinews 有 14 种语言。
每种语言的文档数相同；同一语言内各来源轮流取，份额相同，某个来源不够时由同语言的其他来源补足。

每篇文档只保留开头（默认 4000 字符，按行截断），送进模型前再由 `prompts` 按 token 数切出 64–256 token 的
前缀。文档级粗筛沿用 `pipeline.text.sources.web` 的规则：HPLT 去掉机翻（MT）和推广（IP）体裁、聚类过大和语种
不符的文档，FineWeb 系列按语种得分、聚类大小和教育分筛选；网址或开头命中赌博、外汇等关键词的文档丢弃。
开头文本中任一整句与 FLEURS 句子（`exclusion.jsonl`）相同的文档整篇丢弃。

输出 `<out-dir>/eval.jsonl` 和 `<out-dir>/train.jsonl`，两者的文档互不重叠；每行一篇文档：
`prefix_id`、`lang`、`source`、`source_license`、`source_ref`（网址或 Wikinews 标题）、`doc_id`、`split`、`text`。
文件内按「语言轮流、同语言内来源轮流」排列，取任意前若干行，语言和来源都大致均衡。
"""

import argparse
import bz2
import json
import random
import re
import subprocess
from collections import Counter
from multiprocessing import get_context
from pathlib import Path
from urllib.parse import urlsplit

from pipeline.text.exclusion import is_excluded, load_exclusion
from pipeline.text.fetch_web import FINEWEB2_CONFIGS, HPLT_CODES, WIKINEWS_LANGS
from pipeline.text.langs import LANGS, NO_SPACE_LANGS
from pipeline.text.sources import web
from pipeline.text.web_entry_rules import _SPAM_RE
from pipeline.text.web_rules import url_is_spam

# HPLT 的体裁：去掉机翻（MT）和推广（IP），其余都可以作为续写的开头
HPLT_GENRES = tuple(name for name in web.HPLT_TOP_REGISTERS if name not in {"MT", "IP"})
# 机翻腔偏多的语言收紧机翻概率上限（与 T03b 抽取一致）
STRICT_MT_LANGS = {"fil", "vi", "th"}
MAX_CHARS = 4000
# 开头至少的字符数：不用空格分词的语言和韩语 150 字，其余 300 字
MIN_CHARS_DENSE = 150
MIN_CHARS = 300
_BLANK_LINES = re.compile(r"\n{3,}")
_SPACES = re.compile(r"[ \t]+\n")


def doc_beginning(text, max_chars = MAX_CHARS):
    """文档开头：统一换行，去掉行尾空白和多余空行，按行截断到 max_chars 字符以内。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _SPACES.sub("\n", text)
    text = _BLANK_LINES.sub("\n\n", text).strip()
    if len(text) <= max_chars:
        return text
    cut = text.rfind("\n", 0, max_chars)
    return text[:cut].rstrip() if cut > max_chars // 2 else text[:max_chars]


def min_chars(lang):
    return MIN_CHARS_DENSE if lang in NO_SPACE_LANGS or lang == "ko" else MIN_CHARS


def round_robin(iterators):
    """轮流从各迭代器取一项，直到全部取完；让抽样覆盖每个文件，而不只是第一个文件的开头。"""
    iterators = list(iterators)
    while iterators:
        alive = []
        for iterator in iterators:
            try:
                yield next(iterator)
            except StopIteration:
                continue
            alive.append(iterator)
        iterators = alive


def read_hplt_file(path, code, max_mt):
    process = subprocess.Popen(["zstd", "-dc", path], stdout = subprocess.PIPE, stderr = subprocess.DEVNULL)
    for raw in process.stdout:
        try:
            doc = json.loads(raw)
        except ValueError:
            continue  # 截断处的最后一行
        if web.hplt_doc_ok(doc, genres = HPLT_GENRES, max_mt = max_mt, glotlid_code = code) is None:
            continue
        url = doc.get("u") or ""
        yield str(doc.get("id") or url), url, doc.get("text") or ""
    process.stdout.close()
    process.wait()


def read_hplt(raw_root, lang):
    """逐篇产出 (doc_id, 网址, 全文)，在该语言的几个 shard 之间轮流读取。"""
    code = HPLT_CODES[lang]
    max_mt = 0.05 if lang in STRICT_MT_LANGS else 0.2
    files = web.list_files(Path(raw_root) / "HPLT" / "HPLT3.0" / code / "*.jsonl.zst")
    yield from round_robin(iterators = [read_hplt_file(path = path, code = code, max_mt = max_mt) for path in files])


def read_parquet(files, min_language_score, max_cluster = 50, min_int_score = 0):
    yield from round_robin(
        iterators = [
            read_parquet_file(
                path = path,
                min_language_score = min_language_score,
                max_cluster = max_cluster,
                min_int_score = min_int_score,
            )
            for path in files
        ]
    )


def read_parquet_file(path, min_language_score, max_cluster, min_int_score):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    names = set(parquet.schema_arrow.names)
    columns = [name for name in ("text", "id", "url", "language_score", "minhash_cluster_size", "int_score") if name in names]
    for group in range(parquet.metadata.num_row_groups):
        for doc in parquet.read_row_group(group, columns = columns).to_pylist():
            if (doc.get("language_score") or 0.0) < min_language_score:
                continue
            if "minhash_cluster_size" in names and (doc.get("minhash_cluster_size") or 1) > max_cluster:
                continue
            if "int_score" in names and (doc.get("int_score") or 0) < min_int_score:
                continue
            url = doc.get("url") or ""
            yield str(doc.get("id") or url), url, doc.get("text") or ""


def read_wikinews(raw_root, lang):
    """逐篇读取 Wikinews dump 的主命名空间文章，产出 (页面编号, 标题, 正文)。"""
    import xml.etree.ElementTree as ET

    path = Path(raw_root) / "wikimedia" / "wikinews" / lang / f"{lang}wikinews-latest-pages-articles.xml.bz2"
    with bz2.open(path, "rb") as handle:
        title, namespace, page_id, redirect = "", "", "", False
        for _, element in ET.iterparse(handle, events = ("end",)):
            tag = element.tag.rsplit("}", 1)[-1]
            if tag == "title":
                title = element.text or ""
            elif tag == "ns":
                namespace = element.text or ""
            elif tag == "id" and not page_id:
                page_id = element.text or ""
            elif tag == "redirect":
                redirect = True
            elif tag == "page":
                text = ""
                for child in element.iter():
                    if child.tag.rsplit("}", 1)[-1] == "text":
                        text = child.text or ""
                if namespace == "0" and not redirect and len(text) >= 200:
                    yield page_id, f"wikinews:{lang}:{title}", web.wikitext_to_text(text)
                title, namespace, page_id, redirect = "", "", "", False
                element.clear()


# 来源短名 -> (来源名, 许可)
SOURCES = {
    "hplt": (web.HPLT_NAME, web.HPLT_LICENSE),
    "fw2hq": (web.FW2_NAME, web.FW2_LICENSE),
    "fwedu": (web.FWEDU_NAME, web.FWEDU_LICENSE),
    "wikinews": (web.WIKINEWS_NAME, web.WIKINEWS_LICENSE),
}


def sources_for(lang, raw_root):
    """该语言可用的来源短名及读取函数。"""
    raw_root = Path(raw_root)
    available = []
    if web.list_files(raw_root / "HPLT" / "HPLT3.0" / HPLT_CODES[lang] / "*.jsonl.zst"):
        available.append(("hplt", lambda: read_hplt(raw_root = raw_root, lang = lang)))
    if lang in FINEWEB2_CONFIGS:
        files = web.list_files(raw_root / "epfml" / "FineWeb2-HQ-Plus" / FINEWEB2_CONFIGS[lang] / "subset-*.parquet")
        if files:
            available.append(("fw2hq", lambda: read_parquet(files = files, min_language_score = 0.9)))
    if lang == "en":
        files = web.list_files(raw_root / "HuggingFaceFW" / "fineweb-edu" / "sample-10BT" / "subset-*.parquet")
        if files:
            available.append(("fwedu", lambda: read_parquet(files = files, min_language_score = 0.95, min_int_score = 3)))
    if lang in WIKINEWS_LANGS:
        path = raw_root / "wikimedia" / "wikinews" / lang / f"{lang}wikinews-latest-pages-articles.xml.bz2"
        if path.exists():
            available.append(("wikinews", lambda: read_wikinews(raw_root = raw_root, lang = lang)))
    return available


def collect(job):
    """一种语言的一个来源：扫描至多 scan 篇合格文档，随机取 need 篇。"""
    lang, slug, need, seed = job["lang"], job["slug"], job["need"], job["seed"]
    reader = dict(sources_for(lang = lang, raw_root = job["raw_root"]))[slug]
    exclusion = load_exclusion(job["exclusion"])
    spam = _SPAM_RE.get(lang)
    stats = Counter()
    candidates = []
    seen = set()
    per_domain = Counter()
    for doc_id, ref, text in reader():
        stats["docs"] += 1
        if url_is_spam(ref):
            stats["spam_url"] += 1
            continue
        domain = urlsplit(ref).netloc.lower() if "://" in ref else ""
        if domain and per_domain[domain] >= job["max_per_domain"]:
            stats["domain_cap"] += 1
            continue
        beginning = doc_beginning(text = text)
        if len(beginning) < min_chars(lang = lang):
            stats["too_short"] += 1
            continue
        if "�" in beginning:
            stats["encoding"] += 1
            continue
        if spam is not None and spam.search(beginning):
            stats["spam_text"] += 1
            continue
        key = beginning[:200]
        if key in seen:
            stats["duplicate"] += 1
            continue
        seen.add(key)
        if is_excluded(beginning, lang = lang, exclusion = exclusion):
            stats["fleurs"] += 1
            continue
        per_domain[domain] += 1
        candidates.append({"doc_id": doc_id, "source_ref": ref, "text": beginning})
        if len(candidates) >= job["scan"]:
            break
    stats["candidates"] = len(candidates)
    random.Random(f"{seed}-{lang}-{slug}").shuffle(candidates)
    return lang, slug, candidates[:need], dict(stats)


def interleave(lists):
    """轮流从各列表取一个，直到全部取完。"""
    out = []
    position = 0
    while any(position < len(items) for items in lists):
        for items in lists:
            if position < len(items):
                out.append(items[position])
        position += 1
    return out


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", required = True, help = "原始语料的文本根目录，例如 $DATASET/text")
    parser.add_argument("--exclusion", required = True, help = "FLEURS 排除清单 exclusion.jsonl")
    parser.add_argument("--out-dir", required = True)
    parser.add_argument("--eval-per-lang", type = int, default = 150)
    parser.add_argument("--train-per-lang", type = int, default = 2000)
    parser.add_argument("--scan-factor", type = float, default = 4.0, help = "每个来源扫描的合格文档数是所需数量的倍数，从中随机抽取")
    parser.add_argument("--max-per-domain", type = int, default = 3, help = "每个网站最多取的文档数")
    parser.add_argument("--lang", nargs = "+", default = LANGS)
    parser.add_argument("--jobs", type = int, default = 8)
    parser.add_argument("--seed", type = int, default = 0)
    args = parser.parse_args()

    per_lang = args.eval_per_lang + args.train_per_lang
    jobs = []
    for lang in args.lang:
        slugs = [slug for slug, _ in sources_for(lang = lang, raw_root = args.raw_root)]
        # 每个来源多取一些，某个来源不够时由同语言的其他来源补足
        need = per_lang
        for slug in slugs:
            jobs.append(
                {
                    "lang": lang,
                    "slug": slug,
                    "need": need,
                    "scan": int(need * args.scan_factor),
                    "seed": args.seed,
                    "max_per_domain": args.max_per_domain,
                    "raw_root": args.raw_root,
                    "exclusion": args.exclusion,
                }
            )
    found = {}
    stats = {}
    with get_context("spawn").Pool(processes = min(args.jobs, len(jobs))) as pool:
        for lang, slug, docs, job_stats in pool.imap_unordered(collect, jobs):
            found[(lang, slug)] = docs
            stats[f"{lang}/{slug}"] = job_stats
            print(f"[{lang}] {slug}: 取 {len(docs)} 篇，统计 {job_stats}", flush = True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents = True, exist_ok = True)
    by_split = {"eval": [], "train": []}
    for lang in args.lang:
        slugs = [slug for slug, _ in sources_for(lang = lang, raw_root = args.raw_root)]
        # 来源轮流排成一列，前 eval-per-lang 篇作评测用，其后作训练用
        pools = [[dict(doc, slug = slug) for doc in found.get((lang, slug), [])] for slug in slugs]
        ordered = interleave(lists = pools)[:per_lang]
        if len(ordered) < per_lang:
            print(f"[{lang}] 只有 {len(ordered)} 篇，少于所需的 {per_lang} 篇", flush = True)
        for number, doc in enumerate(ordered):
            split = "eval" if number < args.eval_per_lang else "train"
            name, license_ = SOURCES[doc["slug"]]
            if doc["slug"] == "wikinews":
                license_ = web.WIKINEWS_LICENSE_BY_LANG.get(lang, license_)
            by_split[split].append(
                {
                    "prefix_id": f"{lang}-{doc['slug']}-{number:05d}",
                    "lang": lang,
                    "source": name,
                    "source_license": license_,
                    "source_ref": doc["source_ref"],
                    "doc_id": doc["doc_id"],
                    "split": split,
                    "text": doc["text"],
                }
            )
    for split, rows in by_split.items():
        per_language = [[row for row in rows if row["lang"] == lang] for lang in args.lang]
        with open(out_dir / f"{split}.jsonl", "w", encoding = "utf-8") as handle:
            for row in interleave(lists = per_language):
                handle.write(json.dumps(row, ensure_ascii = False) + "\n")
        print(f"{split}: {len(rows)} 篇，{dict(Counter(row['lang'] for row in rows))}", flush = True)
    with open(out_dir / "stats.json", "w", encoding = "utf-8") as handle:
        json.dump(stats, handle, ensure_ascii = False, indent = 2, sort_keys = True)


if __name__ == "__main__":
    main()
