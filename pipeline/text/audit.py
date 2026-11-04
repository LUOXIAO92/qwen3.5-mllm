"""审阅报告里的问题正则，在两份条目目录上各统计一遍命中数，用于修复前后对照。

    python -m pipeline.text.audit --before <旧目录> --after <新目录>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# (名称, 语言, 来源名前缀或 None, 正则)
CHECKS = [
    ("de 序数/月份切句：1-2 位数字加句点结尾", "de", "singletongue", r"\b\d{1,2}\.$"),
    ("de 以月份名开头", "de", "singletongue", r"^(Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\b"),
    ("de 罗马数字加句点结尾", "de", "singletongue", r"\b[IVX]{2,}\.$"),
    ("en 缩写结尾 U.S./St./Mt./Jr./Dr./No./Inc.", "en", "singletongue", r"\b(U\.S|St|Mt|Jr|Dr|No|Inc)\.$"),
    ("ru 以 св. 结尾", "ru", "singletongue", r"св\.$"),
    ("de 以 geb./Chr./Mio./z. B. 结尾", "de", "singletongue", r"(\b(geb|Chr|Mio)|z\. ?B)\.$"),
    ("fr 引用残留 ,, ,. (,", "fr", None, r",\s*,|,\.|\(\s*[,;]"),
    ("fr CV 功能词重复", "fr", "Common", r"\b(de|du|des|la|le|les|à|un|une)\s+\1\b"),
    ("fr CV 省音词结尾", "fr", "Common", r"[ld]['’]\.$"),
    ("it CV 省音词结尾", "it", "Common", r"[ld]['’]\.$"),
    ("ko 汉字", "ko", None, r"[一-鿿]"),
    ("ko 假名", "ko", None, r"[぀-ヿ]"),
    ("ko 外语括注 (일본어: …)", "ko", None, r"\([^)]*[가-힣]+어\s*:"),
    ("ko 依赖后文的句尾", "ko", None, r"(다음과 같다|다음과 같습니다|참조하라|가리킨다)\.$"),
    ("th CV 连词开头", "th", "Common", r"^(แต่|และ|ดังนั้น|ส่วน|หรือ)"),
    ("th CV 含 ' - ' 或冒号", "th", "Common", r"\s[-–—]\s|:"),
    ("th CV 20 字以下", "th", "Common", r"^.{0,19}$"),
    ("Tatoeba ' / ' 诗行 en", "en", "Tatoeba", r"\s/\s"),
    ("Tatoeba ' / ' 诗行 fr", "fr", "Tatoeba", r"\s/\s"),
    ("Tatoeba ' / ' 诗行 pt", "pt", "Tatoeba", r"\s/\s"),
    ("Tatoeba ' / ' 诗行 fa", "fa", "Tatoeba", r"\s/\s"),
    ("Tatoeba ' / ' 诗行 de", "de", "Tatoeba", r"\s/\s"),
    ("Tatoeba ' / ' 诗行 fil", "fil", "Tatoeba", r"\s/\s"),
    ("pt CV 问号后接小写", "pt", "Common", r"\?\s+[a-zà-ú]"),
    ("pt 全部问号后接小写", "pt", None, r"\?\s+[a-zà-ú]"),
    ("ar 行末 ..", "ar", None, r"\.\.$"),
    ("ar ' / ' 双历法", "ar", None, r"\s/\s"),
    ("ar 列表引言 هذه قائمة/صفحة", "ar", None, r"^هذه (قائمة|صفحة)"),
    ("ja 以下の通り。结尾", "ja", None, r"(次の通り|以下の通り)。$"),
    ("ru 象棋记谱（拉丁与西里尔相连）", "ru", None, r"[а-яё][a-z]|[a-z][а-яё]"),
    ("ru W/L 统计", "ru", None, r"\b[WLDwld]/[WLDwld]\b"),
    ("括号组 >=2：fr wiki", "fr", "singletongue", r"(\([^()]*\)[^()]*){2,}"),
    ("括号组 >=2：ko wiki", "ko", "singletongue", r"(\([^()]*\)[^()]*){2,}"),
    ("括号组 >=2：ja wiki", "ja", "singletongue", r"([（(][^()（）]*[)）][^()（）]*){2,}"),
    ("括号组 >=2：ru wiki", "ru", "singletongue", r"(\([^()]*\)[^()]*){2,}"),
    ("fa 错位 ZWNJ（词首尾、连续、紧邻空格）", "fa", None, "^‌|‌$|‌\\s|\\s‌|‌{2}"),
    ("fa 阿拉伯字母 ي/ك", "fa", None, "[يك]"),
]


def count_hits(directory: Path) -> dict[str, tuple[int, int]]:
    cache: dict[str, list[dict]] = {}
    result = {}
    for name, lang, source, pattern in CHECKS:
        if lang not in cache:
            path = directory / f"{lang}.jsonl"
            cache[lang] = [json.loads(line) for line in open(path, encoding = "utf-8")] if path.exists() else []
        compiled = re.compile(pattern)
        pool = [row for row in cache[lang] if source is None or row["source"].startswith(source)]
        result[name] = (sum(bool(compiled.search(row["original"])) for row in pool), len(pool))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--before", required = True)
    parser.add_argument("--after", required = True)
    args = parser.parse_args()
    before = count_hits(Path(args.before))
    after = count_hits(Path(args.after))
    print("| 检查项 | 修复前命中 / 范围 | 修复后命中 / 范围 |")
    print("|---|---|---|")
    for name, _, _, _ in CHECKS:
        print(f"| {name} | {before[name][0]} / {before[name][1]} | {after[name][0]} / {after[name][1]} |")


if __name__ == "__main__":
    main()
