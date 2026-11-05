"""网页与新闻来源的读取器：HPLT 3.0、FineWeb2-HQ-Plus、fineweb-edu、Wikinews。

每个读取器逐段产出 `(段落, 定位信息)`，定位信息形如 `<URL 或页面标题>#<段落序号>`，
抽取时再按 URL 的域名和文档限制每个站点、每篇文档的条数。文档级的粗筛（体裁标签、机翻概率、
聚类大小、语种得分）在这里做；句子级的清洗和过滤交给 `extract.extract_candidates`。
"""

from __future__ import annotations

import bz2
import glob
import json
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path

from ..web_rules import url_is_spam

# ---------------------------------------------------------------- 来源名与许可
HPLT_NAME, HPLT_SLUG, HPLT_LICENSE = "HPLT 3.0", "hplt", "CC0-1.0（HPLT 包装，原文版权归原作者）"
FW2_NAME, FW2_SLUG, FW2_LICENSE = "epfml/FineWeb2-HQ-Plus", "fw2hq", "ODC-By-1.0（原文版权归原作者）"
FWEDU_NAME, FWEDU_SLUG, FWEDU_LICENSE = "HuggingFaceFW/fineweb-edu", "fwedu", "ODC-By-1.0（原文版权归原作者）"
WIKINEWS_NAME, WIKINEWS_SLUG = "Wikinews", "wikinews"
WIKINEWS_LICENSE = "CC-BY-4.0"
WIKINEWS_LICENSE_BY_LANG = {"de": "CC-BY-2.5", "fr": "CC-BY-SA-4.0"}

# HPLT 的体裁标签（顶层）：NA 叙事（含新闻、叙事博客），IN 信息说明，OP 观点；
# 其余 MT 机翻、LY 歌词诗歌、SP 口语转写、ID 互动讨论、HI 操作指南、IP 推广都不适合直接朗读。
HPLT_TOP_REGISTERS = ("MT", "LY", "SP", "ID", "NA", "HI", "IN", "OP", "IP")
HPLT_GENRES = ("NA", "OP", "IN")


# ---------------------------------------------------------------- 文本切段
_TERMINALS = "。！？.!?…"


def text_to_paragraphs(text: str, *, lang: str) -> list[str]:
    """按换行把文档文本拆成段落。

    日语博客常常一句一行：相邻的短行（都在 100 字以内）合并成一个段落，让后面的切句和拼段能连起来；
    没有句末标点的行（标题、导航）合并后会在切句时被丢掉，不会混进正文。
    """
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    if lang != "ja":
        return lines
    paragraphs: list[str] = []
    buffer: list[str] = []
    for line in lines:
        if len(line) <= 100:
            buffer.append(line)
            continue
        if buffer:
            paragraphs.append("".join(buffer))
            buffer = []
        paragraphs.append(line)
    if buffer:
        paragraphs.append("".join(buffer))
    return paragraphs


# ---------------------------------------------------------------- HPLT 3.0
def hplt_main_register(registers: dict) -> tuple[str, float]:
    """顶层体裁里概率最高的标签及其概率。"""
    best = max(HPLT_TOP_REGISTERS, key = lambda name: registers.get(name, 0.0))
    return best, registers.get(best, 0.0)


def hplt_doc_ok(
    doc: dict,
    *,
    genres: tuple[str, ...],
    max_mt: float = 0.2,
    max_cluster: int = 20,
    min_lang_prob: float = 0.8,
    min_target_fraction: float = 0.9,
    glotlid_code: str = "",
) -> str | None:
    """文档级粗筛，返回体裁（小写）；不合格返回 None。"""
    registers = doc.get("web-register") or {}
    main, _ = hplt_main_register(registers)
    if main not in genres or registers.get("MT", 0.0) > max_mt:
        return None
    if doc.get("filter", "keep") != "keep" or doc.get("pii"):
        return None
    if (doc.get("cluster_size") or 1) > max_cluster:
        return None
    langs, probs = doc.get("lang") or [""], doc.get("prob") or [0.0]
    if glotlid_code and (langs[0] != glotlid_code or probs[0] < min_lang_prob):
        return None
    segments = doc.get("seg_langs") or []
    # 波斯语：文档语种标 pes_Arab，段落语种标 fas_Arab，两个代码都算目标语言。
    accepted = {glotlid_code, glotlid_code.replace("pes_", "fas_")}
    if glotlid_code and segments and sum(code in accepted for code in segments) / len(segments) < min_target_fraction:
        return None
    return main.lower()


def read_hplt(
    *,
    files: list[str],
    lang: str,
    genre: str,
    glotlid_code: str,
    max_chars: int = 40000,
    max_mt: float = 0.2,
) -> Iterator[tuple[str, str]]:
    """读取若干 `.jsonl.zst`（可以是被截断的前缀），只产出指定体裁（`na`、`op`、`in`）的段落。"""
    for path in files:
        process = subprocess.Popen(["zstd", "-dc", path], stdout = subprocess.PIPE, stderr = subprocess.DEVNULL)
        assert process.stdout is not None
        for raw in process.stdout:
            try:
                doc = json.loads(raw)
            except ValueError:
                continue  # 截断处的最后一行
            text = doc.get("text") or ""
            if len(text) > max_chars:
                continue
            if hplt_doc_ok(doc, genres = (genre.upper(),), glotlid_code = glotlid_code, max_mt = max_mt) is None:
                continue
            url = doc.get("u") or doc.get("id", "")
            if url_is_spam(url):
                continue
            for number, paragraph in enumerate(text_to_paragraphs(text, lang = lang)):
                yield paragraph, f"{url}#{number}"
        process.stdout.close()
        process.wait()


# ---------------------------------------------------------------- FineWeb 系列（parquet 子集）
def read_parquet_docs(
    *,
    files: list[str],
    lang: str,
    min_language_score: float = 0.9,
    max_cluster: int = 50,
    min_int_score: int = 0,
    max_chars: int = 40000,
) -> Iterator[tuple[str, str]]:
    """读取 `fetch_web` 保存的 parquet 子集。FineWeb2-HQ-Plus 有 `minhash_cluster_size`，fineweb-edu 有 `int_score`。"""
    import pyarrow.parquet as pq

    for path in files:
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        for group in range(parquet.metadata.num_row_groups):
            for doc in parquet.read_row_group(group).to_pylist():
                if (doc.get("language_score") or 0.0) < min_language_score:
                    continue
                if "minhash_cluster_size" in names and (doc.get("minhash_cluster_size") or 1) > max_cluster:
                    continue
                if "int_score" in names and (doc.get("int_score") or 0) < min_int_score:
                    continue
                text = doc["text"] or ""
                if len(text) > max_chars:
                    continue
                url = doc.get("url") or doc.get("id", "")
                if url_is_spam(url):
                    continue
                for number, paragraph in enumerate(text_to_paragraphs(text, lang = lang)):
                    yield paragraph, f"{url}#{number}"


# ---------------------------------------------------------------- Wikinews
# 正文之后的章节：消息来源、相关新闻、外部链接、注释。遇到就截断。
_TAIL_HEADINGS = (
    "sources?|related news|related stories|external links|references|notes|see also|"
    "消息[來来]源|[來来]源|相[關关]新[聞闻]|外部[連链]接|参考资料|參考資料|註釋|注释|"
    "sources|articles connexes|voir aussi|liens externes|notes|"
    "quellen|einzelnachweise|weblinks|siehe auch|anmerkungen|"
    "источники|примечания|ссылки|см\\. также|связанные новости|"
    "fontes?|notícias relacionadas|ligações externas|"
    "fuentes?|noticias relacionadas|enlaces externos|"
    "fonti|notizie correlate|collegamenti esterni|"
    "出典|関連ニュース|外部リンク|関連項目|"
    "출처|관련 뉴스|바깥 고리|각주|"
    "المصادر|مصادر|أخبار ذات صلة|وصلات خارجية|"
    "منابع|منبع|خبرهای مرتبط|پیوند به بیرون|"
    "kaynaklar?|i̇lgili haberler|dış bağlantılar|"
    "แหล่งที่มา|ข่าวที่เกี่ยวข้อง"
)
_TAIL_RE = re.compile(rf"^=+\s*(?:{_TAIL_HEADINGS})\s*=+\s*$", re.IGNORECASE | re.MULTILINE)
_NAMESPACE_LINK = re.compile(
    r"^(category|file|image|media|catégorie|fichier|kategorie|datei|категория|файл|categoría|archivo|categoria|ficheiro|"
    r"arquivo|ファイル|カテゴリ|分类|分類|文件|檔案|파일|분류|تصنيف|ملف|رده|پرونده|kategori|dosya|หมวดหมู่|ไฟล์)\s*:",
    re.IGNORECASE,
)


# Wikinews 用模板写内部链接：{{w|页面名|显示文字}}（en、ko）、{{v|…}}（tr）、{{wikipedia|…}}（zh）等。
# strip_code 会把模板整个丢掉，句子里就缺了实体名；这里先换成显示文字（有第二个参数用第二个，否则用第一个）。
_LINK_TEMPLATES = {
    "w", "wikipedia", "wp", "v", "vikipedi", "wikinews", "위키뉴스", "ویکی‌پدیا", "ویکی پدیا", "ویکی\u200cپدیا",
    "lang", "ill", "iw", "w2", "wl",
}


def expand_link_templates(code) -> None:
    """把链接模板就地换成显示文字，其余模板保持原样（由 strip_code 删除）。"""
    for template in code.filter_templates(recursive = False):
        name = str(template.name).strip().lower()
        if name not in _LINK_TEMPLATES:
            continue
        positional = [str(param.value).strip() for param in template.params if not str(param.name).strip().lower().startswith(("lang", "wd")) and str(param.name).strip().isdigit()]
        if name in {"lang", "ill", "iw"}:
            text = positional[1] if len(positional) > 1 else ""
        else:
            text = positional[1] if len(positional) > 1 else (positional[0] if positional else "")
        code.replace(template, text)


def wikitext_to_text(wikitext: str) -> str:
    """去掉模板、链接标记、分类和文件，只留正文；遇到「消息来源」等章节就截断。"""
    import mwparserfromhell

    match = _TAIL_RE.search(wikitext)
    if match:
        wikitext = wikitext[: match.start()]
    code = mwparserfromhell.parse(wikitext)
    expand_link_templates(code)
    for link in code.filter_wikilinks():
        if _NAMESPACE_LINK.match(str(link.title).strip()):
            try:
                code.remove(link)
            except ValueError:
                pass
    text = code.strip_code(normalize = True, collapse = True)
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith(("=", "*", "#", ":", ";", "|", "!")):
            continue  # 标题行、列表、表格
        lines.append(line)
    return "\n".join(lines)


def read_wikinews(*, path: str, lang: str, min_chars: int = 200) -> Iterator[tuple[str, str]]:
    """逐篇读取 Wikinews dump 的主命名空间文章，产出段落；定位信息是 `wikinews:<语言>:<标题>#<段落序号>`。"""
    import xml.etree.ElementTree as ET

    with bz2.open(path, "rb") as handle:
        title = ""
        namespace = ""
        redirect = False
        for event, element in ET.iterparse(handle, events = ("end",)):
            tag = element.tag.rsplit("}", 1)[-1]
            if tag == "title":
                title = element.text or ""
            elif tag == "ns":
                namespace = element.text or ""
            elif tag == "redirect":
                redirect = True
            elif tag == "page":
                text = ""
                for child in element.iter():
                    if child.tag.rsplit("}", 1)[-1] == "text":
                        text = child.text or ""
                if namespace == "0" and not redirect and len(text) >= min_chars:
                    plain = wikitext_to_text(text)
                    for number, paragraph in enumerate(text_to_paragraphs(plain, lang = lang)):
                        yield paragraph, f"wikinews:{lang}:{title}#{number}"
                title, namespace, redirect = "", "", False
                element.clear()


def list_files(pattern: str) -> list[str]:
    return sorted(glob.glob(str(Path(pattern))))


# ---------------------------------------------------------------- 过滤选项
WEB_SOURCE_NAMES = {HPLT_NAME, FW2_NAME, FWEDU_NAME, WIKINEWS_NAME}
SPAM_SOURCE_NAMES = {HPLT_NAME, FW2_NAME, FWEDU_NAME}


def entry_options(lang: str, source_name: str):
    """条目所属来源对应的过滤选项：网页来源开启句级和整条规则，HPLT、FineWeb 系列再开启垃圾关键词；part1 来源返回默认值。"""
    import dataclasses

    from ..filters import default_options

    options = default_options(lang)
    if source_name in WEB_SOURCE_NAMES:
        options = dataclasses.replace(options, web = True, entry = True, spam = source_name in SPAM_SOURCE_NAMES)
    return options
