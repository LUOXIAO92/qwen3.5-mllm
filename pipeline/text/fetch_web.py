"""下载网页与新闻来源的原始数据，放进数据集目录（不进系统盘缓存）。

    python -m pipeline.text.fetch_web hplt --dest <数据集根目录> --lang all
    python -m pipeline.text.fetch_web fineweb2hq --dest <数据集根目录> --lang all
    python -m pipeline.text.fetch_web fineweb-edu --dest <数据集根目录>
    python -m pipeline.text.fetch_web wikinews --dest <数据集根目录> --lang all

落盘位置（`<dest>` 下，与已有数据集的布局一致）：

- `HPLT/HPLT3.0/<语言代码>/<档位>_<序号>.jsonl.zst`：每个 shard 只下载开头的若干 MB（shard 本身有
  十几到几十 GB），文件末尾是被截断的 zstd 流，`zstd -dc` 可以读到截断处。
- `epfml/FineWeb2-HQ-Plus/<config>/subset-<序号>.parquet`：从若干 parquet 文件里随机取行组，只保留文本和筛选字段，
  不下载体积很大的 `embeddings` 列。
- `HuggingFaceFW/fineweb-edu/sample-10BT/subset-<序号>.parquet`：同上。
- `wikimedia/wikinews/<语言>/<语言>wikinews-latest-pages-articles.xml.bz2`：官方 dump 整个文件。
"""

from __future__ import annotations

import argparse
import random
import urllib.request
from pathlib import Path

from .langs import LANGS

HPLT_BASE = "https://data.hplt-project.org/three/sorted"
# 语言 -> HPLT 3.0 的语言代码
HPLT_CODES = {
    "zh": "cmn_Hans", "en": "eng_Latn", "ja": "jpn_Jpan", "ko": "kor_Hang", "de": "deu_Latn",
    "fr": "fra_Latn", "ru": "rus_Cyrl", "pt": "por_Latn", "es": "spa_Latn", "it": "ita_Latn",
    "id": "ind_Latn", "ms": "zsm_Latn", "th": "tha_Thai", "vi": "vie_Latn", "fil": "fil_Latn",
    "ar": "arb_Arab", "tr": "tur_Latn", "fa": "pes_Arab",
}
# 语言 -> FineWeb2-HQ-Plus 的 config（没有英语）
FINEWEB2_CONFIGS = {
    "zh": "cmn_Hani", "ja": "jpn_Jpan", "ko": "kor_Hang", "de": "deu_Latn", "fr": "fra_Latn",
    "ru": "rus_Cyrl", "pt": "por_Latn", "es": "spa_Latn", "it": "ita_Latn", "id": "ind_Latn",
    "ms": "zsm_Latn", "th": "tha_Thai", "vi": "vie_Latn", "fil": "fil_Latn", "ar": "arb_Arab",
    "tr": "tur_Latn", "fa": "fas_Arab",
}
FINEWEB2_COLUMNS = ["text", "id", "url", "language_score", "minhash_cluster_size", "quality_score"]
FINEWEB_EDU_COLUMNS = ["text", "id", "url", "language_score", "score", "int_score", "token_count"]
# Wikinews 有 dump 的语言（英语也有，用于补英语新闻体裁）
WIKINEWS_LANGS = ["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it", "ar", "fa", "tr", "th"]
# 默认的 HPLT 下载计划：(shard, 最多下载的 MB)；shard 不存在时顺延到下一个
DEFAULT_HPLT_PLAN = "10_1:300,9_1:250,8_1:250"


def parse_plan(plan: str) -> list[tuple[str, int]]:
    items = []
    for part in plan.split(","):
        shard, megabytes = part.split(":")
        items.append((shard, int(megabytes)))
    return items


def http_get_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout = 60) as response:
        return response.read().decode("utf-8")


def download_prefix(url: str, path: Path, *, max_bytes: int) -> int:
    """下载 url 开头的 max_bytes 字节（文件更小就下载全部），返回写入的字节数。已存在则跳过。"""
    if path.exists() and path.stat().st_size > 0:
        return path.stat().st_size
    path.parent.mkdir(parents = True, exist_ok = True)
    request = urllib.request.Request(url, headers = {"Range": f"bytes=0-{max_bytes - 1}"})
    temp = path.with_suffix(path.suffix + ".tmp")
    written = 0
    with urllib.request.urlopen(request, timeout = 120) as response, open(temp, "wb") as handle:
        while written < max_bytes:
            chunk = response.read(min(1 << 20, max_bytes - written))
            if not chunk:
                break
            handle.write(chunk)
            written += len(chunk)
    temp.rename(path)
    return written


def fetch_hplt(*, dest: Path, lang: str, plan: str) -> None:
    code = HPLT_CODES[lang]
    shards = {line.rsplit("/", 1)[-1].removesuffix(".jsonl.zst") for line in http_get_text(f"{HPLT_BASE}/{code}.map").split()}
    out_dir = dest / "HPLT" / "HPLT3.0" / code
    for shard, megabytes in parse_plan(plan):
        if shard not in shards:
            print(f"[{lang}] HPLT 没有 {shard}，跳过", flush = True)
            continue
        written = download_prefix(
            f"{HPLT_BASE}/{code}/{shard}.jsonl.zst",
            out_dir / f"{shard}.jsonl.zst",
            max_bytes = megabytes * 1024 * 1024,
        )
        print(f"[{lang}] HPLT {code}/{shard}: {written / 1e6:.0f} MB", flush = True)


def fetch_parquet_subset(
    *,
    repo_path: str,
    out_dir: Path,
    columns: list[str],
    n_files: int,
    row_groups_per_file: int,
    seed: int,
    file_glob: str = "*.parquet",
) -> None:
    """在 HF 数据集目录 `repo_path` 下均匀选 n_files 个 parquet，各随机取若干行组，只存 `columns` 列。"""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    out_dir.mkdir(parents = True, exist_ok = True)
    fs = HfFileSystem()
    files = sorted(fs.glob(f"{repo_path}/{file_glob}"))
    if not files:
        raise FileNotFoundError(repo_path)
    rng = random.Random(seed)
    picks = sorted({round(i * (len(files) - 1) / max(n_files - 1, 1)) for i in range(min(n_files, len(files)))})
    for number, pick in enumerate(picks):
        target = out_dir / f"subset-{number:03d}.parquet"
        if target.exists():
            continue
        with fs.open(files[pick]) as handle:
            parquet = pq.ParquetFile(handle)
            groups = sorted(rng.sample(range(parquet.metadata.num_row_groups), k = min(row_groups_per_file, parquet.metadata.num_row_groups)))
            tables = [parquet.read_row_group(group, columns = columns) for group in groups]
        pq.write_table(pa.concat_tables(tables), target, compression = "zstd")
        print(f"{files[pick]} 行组 {groups} -> {target}", flush = True)


def fetch_wikinews(*, dest: Path, lang: str) -> None:
    out_dir = dest / "wikimedia" / "wikinews" / lang
    name = f"{lang}wikinews-latest-pages-articles.xml.bz2"
    target = out_dir / name
    if target.exists() and target.stat().st_size > 0:
        return
    out_dir.mkdir(parents = True, exist_ok = True)
    temp = target.with_suffix(".tmp")
    request = urllib.request.Request(
        f"https://dumps.wikimedia.org/{lang}wikinews/latest/{name}",
        headers = {"User-Agent": "qwen3.5-asr-text-fetch/1.0 (research; contact via GitHub issues)"},
    )
    with urllib.request.urlopen(request, timeout = 300) as response, open(temp, "wb") as handle:
        while chunk := response.read(1 << 20):
            handle.write(chunk)
    temp.rename(target)
    print(f"[{lang}] Wikinews {target.stat().st_size / 1e6:.0f} MB", flush = True)


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", choices = ["hplt", "fineweb2hq", "fineweb-edu", "wikinews"])
    parser.add_argument("--dest", required = True, help = "数据集的文本根目录，例如 $DATASET/text")
    parser.add_argument("--lang", nargs = "+", default = ["all"])
    parser.add_argument("--plan", default = DEFAULT_HPLT_PLAN, help = "HPLT：shard:MB 列表")
    parser.add_argument("--n-files", type = int, default = 8)
    parser.add_argument("--row-groups", type = int, default = 5)
    parser.add_argument("--seed", type = int, default = 0)
    args = parser.parse_args()

    dest = Path(args.dest)
    langs = LANGS if args.lang == ["all"] else args.lang
    if args.source == "hplt":
        for lang in langs:
            fetch_hplt(dest = dest, lang = lang, plan = args.plan)
    elif args.source == "fineweb2hq":
        for lang in langs:
            if lang not in FINEWEB2_CONFIGS:
                continue
            config = FINEWEB2_CONFIGS[lang]
            fetch_parquet_subset(
                repo_path = f"datasets/epfml/FineWeb2-HQ-Plus/{config}",
                out_dir = dest / "epfml" / "FineWeb2-HQ-Plus" / config,
                columns = FINEWEB2_COLUMNS,
                n_files = args.n_files,
                row_groups_per_file = args.row_groups,
                seed = args.seed,
            )
    elif args.source == "fineweb-edu":
        fetch_parquet_subset(
            repo_path = "datasets/HuggingFaceFW/fineweb-edu/sample/10BT",
            out_dir = dest / "HuggingFaceFW" / "fineweb-edu" / "sample-10BT",
            columns = FINEWEB_EDU_COLUMNS,
            n_files = args.n_files,
            row_groups_per_file = args.row_groups,
            seed = args.seed,
        )
    else:
        for lang in langs:
            if lang in WIKINEWS_LANGS:
                fetch_wikinews(dest = dest, lang = lang)


if __name__ == "__main__":
    main()
