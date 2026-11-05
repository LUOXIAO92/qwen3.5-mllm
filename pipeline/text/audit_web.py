"""网页来源条目的问题正则统计，用于 T03b 返工前后对照（审阅报告第一部分的模式）。

    python -m pipeline.text.audit_web --before <旧 final 目录> --after <新 final 目录>

每项统计「网页来源条目（HPLT、FineWeb2-HQ-Plus、fineweb-edu、Wikinews）中命中的条数 / 该语言网页条目总数」。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

WEB = {"HPLT 3.0", "epfml/FineWeb2-HQ-Plus", "HuggingFaceFW/fineweb-edu", "Wikinews"}
LATIN_SHORT = 30

# (名称, 语言列表或 None=全部, 来源限定或 None, 判定函数)
CHECKS = [
    ("单句问句标题（以 ? 结尾、拉丁语 <60 字符、中日韩泰 <40 字符、单句）", None, None, lambda t: bool(re.search(r"[?？]$", t)) and len(t) < (40 if re.search(r"[぀-鿿가-힣฀-๿]", t) else 60) and len(re.findall(r"[.!?。！？]\s", t)) == 0),
    ("过短残句（拉丁语 <30 字符或 <6 词）", ["en", "de", "fr", "es", "it", "pt", "ru", "id", "ms", "vi", "fil", "tr", "ar", "fa"], None, lambda t: len(t) < LATIN_SHORT or len(t.split()) < 6),
    ("过短残句（zh/ja <10，ko <12，th <20）", ["zh", "ja", "ko", "th"], None, lambda t: len(t) < 10),
    ("小标题前缀 ^X: ", None, None, lambda t: bool(re.match(r"^[^.!?:：。！？]{3,40}(?<!\d)[:：]\s", t))),
    ("幻灯编号 ^\\d+/\\d+ ", None, None, lambda t: bool(re.match(r"^\d+/\d+\s", t))),
    ("署名、日期线前缀", None, None, lambda t: bool(re.match(r"^[A-Za-z]{3,15}\s?[–—]\s?[A-Z]|^[A-Za-z]{3,15}\s-\s[A-Z]|^بقلم\s|^\w{3} \d{1,2}, \d{4}\s?[·•]", t))),
    ("行首节号 ^\\d{1,3}\\s?[A-Z]", None, None, lambda t: bool(re.match(r"^\d{1,3}\s?[A-ZÀ-Ý]", t))),
    ("ms 以 Ybhg./Datuk/Dr./Hj. 结尾", ["ms"], None, lambda t: bool(re.search(r"(?<!\w)(Ybhg|Datuk|Dato|Dr|Hj|Hjh|Prof|Tn|Pn|Puan|Tuan)\.$", t))),
    ("id 以 Rp. 结尾", ["id"], None, lambda t: bool(re.search(r"(?<!\w)Rp\.$", t))),
    ("链接缺口：行首逗号、' ، '", None, "Wikinews", lambda t: bool(re.match(r"^[,，،、]", t)) or " ، " in t),
    ("链接缺口 en：the/of/at + have/has/is/was", ["en"], None, lambda t: bool(re.search(r"\b(the|of|at|by|to)\s+(have|has|is|was|said)\b", t, re.I))),
    ("链接缺口 ko：行首助词", ["ko"], None, lambda t: bool(re.match(r"^(는|은|가|를|을)\s", t))),
    ("tr 问号代替撇号", ["tr"], None, lambda t: bool(re.search(r"[A-Za-zÇĞİÖŞÜçğıöşü]\?[a-zçğıöşü]", t))),
    ("id/en ' s 编码", ["id", "en", "ms"], None, lambda t: bool(re.search(r"\s'\s?[a-z]\b|\w\s'\s(s|t|re|ll|ve|d|m)\b", t))),
    ("替换字符 �", None, None, lambda t: "�" in t),
    ("赌博 spam：id judi|togel|slot gacor", ["id"], None, lambda t: bool(re.search(r"judi|togel|slot gacor", t, re.I))),
    ("赌博 spam：vi nhà cái|cá cược", ["vi"], None, lambda t: bool(re.search(r"nhà cái|cá cược", t, re.I))),
    ("spam：tr bahis|casino", ["tr"], None, lambda t: bool(re.search(r"bahis|casino", t, re.I))),
    ("spam：zh 厂家|彩票|赌场", ["zh"], None, lambda t: bool(re.search(r"厂家|彩票|赌场", t))),
    ("spam：ru бинарн", ["ru"], None, lambda t: "бинарн" in t.lower()),
    ("spam：pt/es forex|apuestas", ["pt", "es"], None, lambda t: bool(re.search(r"forex|opções binárias|apuestas", t, re.I))),
    ("spam：ko 카지노|대출", ["ko"], None, lambda t: bool(re.search(r"카지노|대출", t))),
]


def load(directory: Path, lang: str) -> list[dict]:
    path = directory / f"{lang}.jsonl"
    with open(path, encoding = "utf-8") as handle:
        return [row for row in map(json.loads, handle) if row["source"] in WEB]


def count(rows: list[dict], *, source_filter: str | None, check) -> int:
    return sum(1 for row in rows if (source_filter is None or row["source"] == source_filter) and check(row["original"]))


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--before", required = True)
    parser.add_argument("--after", required = True)
    args = parser.parse_args()
    langs = ["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it", "id", "ms", "th", "vi", "fil", "ar", "tr", "fa"]
    before = {lang: load(Path(args.before), lang) for lang in langs}
    after = {lang: load(Path(args.after), lang) for lang in langs}
    for name, selected, source_filter, check in CHECKS:
        parts = []
        total_before = total_after = 0
        for lang in selected or langs:
            hit_before = count(before[lang], source_filter = source_filter, check = check)
            hit_after = count(after[lang], source_filter = source_filter, check = check)
            total_before += hit_before
            total_after += hit_after
            if hit_before or hit_after:
                parts.append(f"{lang} {hit_before}->{hit_after}")
        print(f"{name}：合计 {total_before} -> {total_after}；{', '.join(parts)}")


if __name__ == "__main__":
    main()
