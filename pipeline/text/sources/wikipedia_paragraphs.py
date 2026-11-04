"""singletongue/wikipedia-paragraphs：用 pyarrow 经 HTTP 直接读远端 parquet 的部分行组。

每个 config（如 `kowiki-20260607-v1.2.1`）由若干 `train-*.parquet` 组成，一行是一个页面，
`paragraph_texts` 是该页的段落列表。这里只读需要的列，只取 `p` 标签的段落，
排除参考文献等章节、消歧义页和热度过低的页面。数据不落盘。
"""

from __future__ import annotations

import random
from collections.abc import Iterator

SOURCE_NAME = "singletongue/wikipedia-paragraphs"
SOURCE_SLUG = "wikipara"
SOURCE_LICENSE = "CC-BY-SA-4.0"
REPO = "datasets/singletongue/wikipedia-paragraphs"
COLUMNS = ["id", "page_type", "paragraph_texts", "paragraph_sections", "paragraph_html_tags", "templates", "num_inlinks"]

# 不适合朗读的章节名（小写比较）：参考文献、外部链接、年表、作品列表等。
_SKIP_SECTIONS = {
    "references", "external links", "see also", "further reading", "notes", "sources", "bibliography",
    "footnotes", "filmography", "discography", "works", "publications",
    "参考文献", "外部链接", "外部連結", "参考资料", "参考資料", "注释", "注釋", "相关条目", "相關條目",
    "外部リンク", "脚注", "関連項目", "参考文献・出典", "出典", "注釈", "著作", "作品",
    "각주", "외부 링크", "같이 보기", "참고 문헌", "바깥 고리", "관련 항목",
    "weblinks", "einzelnachweise", "literatur", "siehe auch", "anmerkungen", "fußnoten",
    "liens externes", "notes et références", "voir aussi", "bibliographie", "références",
    "примечания", "ссылки", "литература", "см. также", "источники",
    "ligações externas", "referências", "ver também", "notas", "bibliografia",
    "enlaces externos", "referencias", "véase también", "notas y referencias",
    "collegamenti esterni", "note", "voci correlate", "bibliografia",
    "وصلات خارجية", "مراجع", "انظر أيضا", "هوامش", "المصادر", "انظر أيضاً",
}
_DISAMBIG_MARKS = (
    "disambig", "曖昧", "消歧义", "消歧義", "동음이의", "homonymie", "desambig", "begriffsklärung",
    "неоднозначность", "значения", "disambigua", "توضيح",
)


def is_disambiguation(templates: list[str]) -> bool:
    """页面使用的模板里含消歧义标记。"""
    lowered = [name.lower() for name in templates]
    return any(mark in name for name in lowered for mark in _DISAMBIG_MARKS)


def iter_paragraphs(row: dict, *, min_inlinks: int = 3) -> Iterator[tuple[str, str]]:
    """一个页面的可用段落：(段落文本, `<page id>#<段落序号>`)。"""
    if row["page_type"] != "article" or (row["num_inlinks"] or 0) < min_inlinks:
        return
    if is_disambiguation(row["templates"] or []):
        return
    for index, (text, section, tag) in enumerate(
        zip(row["paragraph_texts"], row["paragraph_sections"], row["paragraph_html_tags"])
    ):
        if tag != "p" or section.lower().strip() in _SKIP_SECTIONS:
            continue
        yield text, f"{row['id']}#{index}"


def read_wikipedia_paragraphs(
    *,
    config: str,
    n_shards: int = 3,
    row_groups_per_shard: int = 1,
    min_inlinks: int = 3,
    seed: int = 0,
) -> Iterator[tuple[str, str]]:
    """从 config 的若干 shard 各读几个随机行组，逐段产出 (段落, 定位信息)。

    shard 沿文件序号均匀选取，行组随机选取，让样本分散在不同的页面区间。
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    files = sorted(fs.glob(f"{REPO}/{config}/train-*.parquet"))
    if not files:
        raise FileNotFoundError(f"config 不存在或没有 parquet 文件：{config}")
    rng = random.Random(seed)
    picks = sorted({round(i * (len(files) - 1) / max(n_shards - 1, 1)) for i in range(min(n_shards, len(files)))})
    for pick in picks:
        with fs.open(files[pick]) as handle:
            parquet = pq.ParquetFile(handle)
            groups = rng.sample(range(parquet.metadata.num_row_groups), k = min(row_groups_per_shard, parquet.metadata.num_row_groups))
            for group in groups:
                table = parquet.read_row_group(group, columns = COLUMNS)
                for row in table.to_pylist():
                    yield from iter_paragraphs(row, min_inlinks = min_inlinks)
