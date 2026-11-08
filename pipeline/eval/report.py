"""汇总多次评测的结果，写出报告用的表格（实验 v1，任务 T16）。

每次评测由 pipeline.eval.run_eval 写出 hypotheses.jsonl（逐句的参考、转写和分数）。本脚本读取若干次评测，输出：

- FLEURS（真人录音）：每种语言的错误率，分「基础规范化」和「数词统一」两种；非拉丁文字的语言另给「去掉括注」
  后的数词统一错误率，括注指参考文本中括号内的拉丁字母原文（如「辛特拉 (Sintra)」），录音里通常不读出来。
- 解码失控：输出字符数超过参考的 2 倍再加 10 的句子（贪心解码陷入重复）所占比例，以及去掉这些句子后的错误率。
- 合成测试集：按语言、TTS 模型、合成方式（内置音色或克隆）、是否跨语言克隆分组，给出整句完全正确的比例和
  逐句错误率的平均；两份文本相同、音色不同的测试集，另比较两边都通过核对的同一批原文。

错误率都是「错误的词（或字）数 ÷ 参考的词（或字）数」，多出的词也计入，所以可能超过 100%。
「平均」指先算每种语言的错误率，再对各语言取简单平均。

用法（在仓库根目录运行）::

    python -m pipeline.eval.report --out-dir <报告目录> \\
        --fleurs v1a=<结果目录> v1b=<结果目录> \\
        --synth v1a:holdout=<结果目录>@<评测清单目录> v1a:trainvoice=<结果目录>@<评测清单目录> ...

写出 <报告目录>/tables.md 与 tables.json。
"""

import argparse
import json
import re
from pathlib import Path

from pipeline.eval.scoring import LANGS, METRIC_BY_LANG, score_pair

NON_LATIN = {"zh", "ja", "ko", "th", "ar", "fa", "ru"}
GLOSS = re.compile(r"\s*[（(][A-Za-z][^()（）]*[)）]")


def load_hypotheses(result_dir: Path) -> list:
    with open(file = Path(result_dir) / "hypotheses.jsonl", mode = "r", encoding = "utf-8") as handle:
        return [json.loads(s = line) for line in handle if line.strip()]


def load_meta(manifest_dir: Path) -> dict:
    meta = {}
    for path in Path(manifest_dir).glob(pattern = "*/test.jsonl"):
        with open(file = path, mode = "r", encoding = "utf-8") as handle:
            for line in handle:
                row = json.loads(s = line)
                meta[row["sample_id"]] = row
    return meta


def pooled(rows: list, variant: str):
    ref_len = sum(row[variant]["ref_len"] for row in rows)
    return sum(row[variant]["errors"] for row in rows) / ref_len if ref_len else None


def runaway(row: dict) -> bool:
    return len(row["hypothesis"]) > 2 * len(row["reference"]) + 10


def pct(value) -> str:
    return "" if value is None else f"{100 * value:.1f}"


def mean(values: list):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def fleurs_tables(runs: dict, out: dict) -> list:
    names = list(runs)
    data = {name: load_hypotheses(result_dir = path) for name, path in runs.items()}
    out["fleurs"] = {}
    lines = ["## FLEURS 测试集（真人录音）", ""]
    lines.append("| 语言 | 指标 | 句数 | " + " | ".join(f"{n} 基础 | {n} 数词统一 | {n} 去括注" for n in names) + " |")
    lines.append("| --- | --- | ---: | " + " | ".join("---: | ---: | ---:" for _ in names) + " |")
    for lang in LANGS:
        cells = []
        count = 0
        for name in names:
            rows = [row for row in data[name] if row["lang"] == lang]
            if not rows:
                cells += ["", "", ""]
                continue
            count = len(rows)
            stripped = None
            if lang in NON_LATIN:
                rescored = [
                    score_pair(reference = GLOSS.sub("", row["reference"]), hypothesis = row["hypothesis"], lang = lang)
                    for row in rows
                ]
                stripped = pooled(rows = rescored, variant = "numeric")
            kept = [row for row in rows if not runaway(row = row)]
            out["fleurs"].setdefault(lang, {})[name] = {
                "n": count,
                "basic": pooled(rows = rows, variant = "basic"),
                "numeric": pooled(rows = rows, variant = "numeric"),
                "stripped": stripped,
                "runaway": 1 - len(kept) / count,
                "numeric_no_runaway": pooled(rows = kept, variant = "numeric"),
            }
            item = out["fleurs"][lang][name]
            cells += [pct(item["basic"]), pct(item["numeric"]), pct(stripped)]
        lines.append(f"| {lang} | {METRIC_BY_LANG[lang]} | {count} | " + " | ".join(cells) + " |")
    out["fleurs_mean"] = {}
    for name in names:
        items = [out["fleurs"][lang][name] for lang in out["fleurs"] if name in out["fleurs"][lang]]
        rows = data[name]
        out["fleurs_mean"][name] = {
            "basic": mean(values = [item["basic"] for item in items]),
            "numeric": mean(values = [item["numeric"] for item in items]),
            "stripped": mean(values = [item["stripped"] if item["stripped"] is not None else item["numeric"] for item in items]),
            "runaway": sum(1 for row in rows if runaway(row = row)) / len(rows),
            "numeric_no_runaway": mean(values = [item["numeric_no_runaway"] for item in items]),
        }
    cells = []
    for name in names:
        item = out["fleurs_mean"][name]
        cells += [pct(item["basic"]), pct(item["numeric"]), pct(item["stripped"])]
    lines.append("| 各语言平均 |  |  | " + " | ".join(cells) + " |")
    lines += ["", "### 解码失控", ""]
    lines.append("失控：输出字符数超过参考的 2 倍再加 10。「其余」：去掉失控句子后的数词统一错误率。")
    lines.append("")
    lines.append("| 语言 | " + " | ".join(f"{n} 失控 % | {n} 其余 %" for n in names) + " |")
    lines.append("| --- | " + " | ".join("---: | ---:" for _ in names) + " |")
    for lang in out["fleurs"]:
        cells = []
        for name in names:
            item = out["fleurs"][lang].get(name)
            cells += ["", ""] if item is None else [pct(item["runaway"]), pct(item["numeric_no_runaway"])]
        lines.append(f"| {lang} | " + " | ".join(cells) + " |")
    cells = []
    for name in names:
        item = out["fleurs_mean"][name]
        cells += [pct(item["runaway"]), pct(item["numeric_no_runaway"])]
    lines.append("| 全部句子 / 各语言平均 | " + " | ".join(cells) + " |")
    return lines + [""]


def group_stats(rows: list) -> dict:
    if not rows:
        return {"n": 0, "exact": None, "mean_rate": None}
    rates = [row["numeric"]["rate"] for row in rows if row["numeric"]["rate"] is not None]
    return {
        "n": len(rows),
        "exact": sum(1 for row in rows if row["numeric"]["errors"] == 0) / len(rows),
        "mean_rate": mean(values = rates),
    }


def synth_tables(runs: dict, out: dict) -> list:
    """runs 的键为 (模型名, 测试集名)，值为 (结果目录, 评测清单目录)。"""
    cols = list(runs)
    data = {key: load_hypotheses(result_dir = result) for key, (result, _) in runs.items()}
    meta = {key: load_meta(manifest_dir = manifest) for key, (_, manifest) in runs.items()}
    out["synth"] = {}
    lines = ["## 合成测试集", ""]
    lines.append("每格为「整句完全正确 % / 逐句错误率的平均 %（句数）」，错误率按数词统一计算。")
    lines.append("")

    def add_table(title: str, rows_by_col: dict) -> None:
        keys = sorted({key for col in cols for key in rows_by_col[col]})
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"| 分组 | " + " | ".join(f"{name} {test_set}" for name, test_set in cols) + " |")
        lines.append("| --- | " + " | ".join("---:" for _ in cols) + " |")
        for key in keys:
            cells = []
            for col in cols:
                stats = group_stats(rows = rows_by_col[col].get(key, []))
                out["synth"].setdefault(title, {}).setdefault(str(key), {})[f"{col[0]}-{col[1]}"] = stats
                cells.append("" if not stats["n"] else f"{pct(stats['exact'])} / {pct(stats['mean_rate'])}（{stats['n']}）")
            lines.append(f"| {key} | " + " | ".join(cells) + " |")
        lines.append("")

    def grouped(key_fn, keep = None) -> dict:
        result = {}
        for col in cols:
            groups = {}
            for row in data[col]:
                info = meta[col].get(row["sample_id"], {})
                if keep is not None and not keep(info):
                    continue
                groups.setdefault(key_fn(info, row), []).append(row)
            result[col] = groups
        return result

    test_sets = sorted({test_set for _, test_set in cols})
    if len(test_sets) == 2:
        # 两份测试集各自只保留核对通过的句子，这里只看两边都通过的同一批原文
        common = set.intersection(*[
            {info["entry_id"] for col in cols if col[1] == test_set for info in meta[col].values()}
            for test_set in test_sets
        ])
        add_table(
            title = f"两份测试集都通过核对的同一批原文（{len(common)} 条）",
            rows_by_col = grouped(key_fn = lambda info, row: info.get("mode", "?"), keep = lambda info: info.get("entry_id") in common),
        )
    add_table(title = "全部", rows_by_col = grouped(key_fn = lambda info, row: "全部"))
    add_table(title = "按语言", rows_by_col = grouped(key_fn = lambda info, row: row["lang"]))
    add_table(title = "按 TTS 模型", rows_by_col = grouped(key_fn = lambda info, row: info.get("tts_model", "?")))
    add_table(title = "按合成方式（builtin 内置音色，clone 克隆）", rows_by_col = grouped(key_fn = lambda info, row: info.get("mode", "?")))
    add_table(
        title = "按参考录音与目标语言是否相同",
        rows_by_col = grouped(key_fn = lambda info, row: {True: "跨语言", False: "同语言"}.get(info.get("cross_lingual"), "?")),
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description = "汇总评测结果")
    parser.add_argument("--out-dir", required = True, type = Path)
    parser.add_argument("--fleurs", nargs = "*", default = [], help = "名称=结果目录")
    parser.add_argument("--synth", nargs = "*", default = [], help = "名称:测试集=结果目录@评测清单目录")
    args = parser.parse_args()
    fleurs = dict(item.split("=", 1) for item in args.fleurs)
    synth = {}
    for item in args.synth:
        key, value = item.split("=", 1)
        name, test_set = key.split(":", 1)
        result, manifest = value.split("@", 1)
        synth[(name, test_set)] = (result, manifest)
    out = {}
    lines = ["# 评测汇总", ""]
    if fleurs:
        lines += fleurs_tables(runs = fleurs, out = out)
    if synth:
        lines += synth_tables(runs = synth, out = out)
    args.out_dir.mkdir(parents = True, exist_ok = True)
    (args.out_dir / "tables.md").write_text(data = "\n".join(lines) + "\n", encoding = "utf-8")
    (args.out_dir / "tables.json").write_text(data = json.dumps(obj = out, ensure_ascii = False, indent = 2), encoding = "utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
