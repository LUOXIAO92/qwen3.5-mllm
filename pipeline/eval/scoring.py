"""按语言计算 WER / CER 的评测打分器（实验 v1，任务 T09）。

用法
====

    from pipeline.eval.scoring import score_pair, score_corpus

    r = score_pair(reference = "他有二十五本书", hypothesis = "他有25本书", lang = "zh")
    r["basic"]["errors"]      # 只做基础规范化：「二十五」与「25」不同，计 3 个错误
    r["numeric"]["errors"]    # 数词统一后：0

命令行（在仓库根目录运行）：

    python -m pipeline.eval.scoring --input hyp.jsonl --output per_sample.jsonl --summary summary.json

输入 JSONL 每行含 sample_id、lang、reference、hypothesis。标准输出打印按语言汇总的表，
--output 写出逐条结果（含逐条对齐），--summary 写出汇总 JSON。

每个语言的两种分数
==================

- basic：只做基础规范化。大小写、Unicode（NFKC）、零宽字符、数字字形、中文繁简、日语假名、
  阿拉伯语/波斯语字母变体等，与 pipeline.compare.normalize.normalize_text 完全相同；再删除标点和
  符号。数字照原样保留，所以「二十五」与「25」会算错。
- numeric：在 basic 的基础上，用 pipeline.compare.normalize_for_scoring 把数词（读法和阿拉伯数字）
  统一成数值。两种分数的差距就是「只是写法不同的数词」造成的错误。

两种分数在同一批 (reference, hypothesis) 上计算，报告里可以并列给出。

WER 还是 CER
============

规则：没有词间空格的文字按字符计算 CER，其余语言按空格分词计算 WER。

| 语言 | 指标 | 依据 |
| --- | --- | --- |
| zh | CER | Qwen3-ASR 技术报告（arXiv 2601.21337）「Evaluation Metrics」：CER 用于 Mandarin Chinese、Cantonese、Korean 等 character-based languages |
| ko | CER | 同上，报告明确把韩语列入 CER |
| ja | CER | 日语没有词间空格，属于报告所说的 character-based languages；报告未逐语言列出，此项按其规则推断 |
| th | CER | 泰语没有词间空格，同上，按其规则推断 |
| en de fr ru pt es it id ms vi fil ar tr fa | WER | 报告称词间有分隔的语言（English、German、French 等）用 WER；越南语以空格分隔音节，按音节计 |

Hugging Face Open ASR Leaderboard（arXiv 2510.06961）的多语种赛道只含 de fr it es pt，统一用 WER，
文本规范化用 BasicMultilingualTextNormalizer（小写、删除标点和符号，保留变音符号）。它没有覆盖中日韩泰，
这几种语言以 Qwen3-ASR 的规则为准。本模块的基础规范化与它一致：小写、删标点、保留变音符号。
Qwen3-ASR 报告没有写明文本规范化的细节。Open ASR Leaderboard 的英文数词规范化用 Whisper 的
EnglishTextNormalizer，本模块的 numeric 分数对应这一步，并覆盖全部 18 种语言。

两处约定：
- 韩语虽然有空格，按 Qwen3-ASR 的做法用 CER，计算时去掉全部空白。
- 字符按 Unicode 码位切分（不按字素簇），泰语的声调符号、元音符号各算一个字符。

语料级结果
==========

语料级错误率 = 编辑距离总和 / 参考长度总和，与逐条错误率取平均的结果不同。「ALL」行是各语言错误率的
宏平均（无权重平均，Qwen3-ASR 同样这样汇总）；WER 与 CER 混合平均只用于粗看。
参考长度为 0 的样本没有逐条错误率（rate 为 None），其插入错误计入语料级总和。

只依赖标准库，需要 Python 3.10 及以上。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict

from pipeline.compare import normalize_for_scoring
from pipeline.compare.normalize import normalize_text

LANGS = ["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it", "id", "ms", "th", "vi", "fil", "ar", "tr", "fa"]

# 没有词间空格的语言，以及按 Qwen3-ASR 惯例使用 CER 的韩语，按字符计算
_CER_LANGS = {"zh", "ja", "ko", "th"}
METRIC_BY_LANG = {lang: ("cer" if lang in _CER_LANGS else "wer") for lang in LANGS}

_APOSTROPHES = "'’ʼ‘`´"
_VARIANTS = ("basic", "numeric")


def normalize_basic(text: str, lang: str) -> str:
    """基础规范化：字符层规范化 + 删除标点和符号，词之间以单个空格分隔。

    撇号直接删除（don't → dont，与 compare 的切分一致），数字之间的小数点和千分位逗号保留，
    百分号保留，其余标点和符号都当作空格。
    """
    s = normalize_text(text = text, lang = lang)
    out = []
    n = len(s)
    for i, ch in enumerate(s):
        if ch in _APOSTROPHES:
            continue
        if unicodedata.category(ch)[0] in "LMN":
            out.append(ch)
        elif ch in ".," and 0 < i < n - 1 and s[i - 1].isdigit() and s[i + 1].isdigit():
            out.append(ch)
        elif ch == "%":
            out.append(ch)
        else:
            out.append(" ")
    return re.sub(pattern = r"\s+", repl = " ", string = "".join(out)).strip()


def normalize_numeric(text: str, lang: str) -> str:
    """数词统一后的规范化文本，直接使用 compare 模块的 normalize_for_scoring。"""
    return normalize_for_scoring(text = text, lang = lang)


def tokenize(text: str, metric: str) -> list:
    """WER 按空格分词；CER 去掉全部空白后按字符切分。"""
    if metric == "wer":
        return text.split()
    return list("".join(text.split()))


def align(ref: list, hyp: list) -> list:
    """最小编辑距离对齐，返回 [(op, ref_token, hyp_token), ...]。

    op 取值 ok（一致）、sub（替换）、del（参考有、识别没有）、ins（识别多出）。
    回溯时优先选一致或替换，其次删除，最后插入，使结果确定。
    """
    n, m = len(ref), len(hyp)
    prev = list(range(m + 1))
    rows = [prev]
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(prev[j - 1] + cost, prev[j] + 1, cur[j - 1] + 1)
        rows.append(cur)
        prev = cur
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        d = rows[i][j]
        if i > 0 and j > 0 and rows[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1) == d:
            ops.append(("ok" if ref[i - 1] == hyp[j - 1] else "sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and rows[i - 1][j] + 1 == d:
            ops.append(("del", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, hyp[j - 1]))
            j -= 1
    ops.reverse()
    return ops


def _score_tokens(ref_text: str, hyp_text: str, metric: str) -> dict:
    ref_tokens = tokenize(text = ref_text, metric = metric)
    hyp_tokens = tokenize(text = hyp_text, metric = metric)
    ops = align(ref = ref_tokens, hyp = hyp_tokens)
    sub = sum(1 for o in ops if o[0] == "sub")
    dele = sum(1 for o in ops if o[0] == "del")
    ins = sum(1 for o in ops if o[0] == "ins")
    ref_len = len(ref_tokens)
    errors = sub + dele + ins
    return {
        "ref_text": ref_text,
        "hyp_text": hyp_text,
        "ref_len": ref_len,
        "sub": sub,
        "del": dele,
        "ins": ins,
        "errors": errors,
        "rate": (errors / ref_len) if ref_len else None,
        # 只记录有差异的位置（op, 参考, 识别），一致的部分省略
        "diff": [list(o) for o in ops if o[0] != "ok"],
    }


def score_pair(reference: str, hypothesis: str, lang: str) -> dict:
    """计算一条样本的两种分数，返回 {"lang", "metric", "basic": {...}, "numeric": {...}}。"""
    if lang not in METRIC_BY_LANG:
        raise ValueError(f"不支持的语言代码：{lang}")
    metric = METRIC_BY_LANG[lang]
    return {
        "lang": lang,
        "metric": metric,
        "basic": _score_tokens(
            ref_text = normalize_basic(text = reference, lang = lang),
            hyp_text = normalize_basic(text = hypothesis, lang = lang),
            metric = metric,
        ),
        "numeric": _score_tokens(
            ref_text = normalize_numeric(text = reference, lang = lang),
            hyp_text = normalize_numeric(text = hypothesis, lang = lang),
            metric = metric,
        ),
    }


def score_corpus(rows: list) -> tuple:
    """rows 为含 sample_id、lang、reference、hypothesis 的字典列表。

    返回 (per_sample, summary)。summary[lang][variant] 含 metric、n、ref_len、sub、del、ins、errors、rate；
    summary["ALL"][variant] 是各语言 rate 的宏平均。
    """
    per_sample = []
    totals = defaultdict(
        lambda: {v: {"n": 0, "ref_len": 0, "sub": 0, "del": 0, "ins": 0, "errors": 0} for v in _VARIANTS}
    )
    for row in rows:
        res = score_pair(
            reference = row["reference"],
            hypothesis = row["hypothesis"],
            lang = row["lang"],
        )
        res["sample_id"] = row["sample_id"]
        per_sample.append(res)
        for variant in _VARIANTS:
            t = totals[row["lang"]][variant]
            t["n"] += 1
            for key in ("ref_len", "sub", "del", "ins", "errors"):
                t[key] += res[variant][key]
    summary = {}
    for lang in [x for x in LANGS if x in totals]:
        summary[lang] = {}
        for variant in _VARIANTS:
            t = dict(totals[lang][variant])
            t["metric"] = METRIC_BY_LANG[lang]
            t["rate"] = (t["errors"] / t["ref_len"]) if t["ref_len"] else None
            summary[lang][variant] = t
    if summary:
        langs = list(summary)
        summary["ALL"] = {}
        for variant in _VARIANTS:
            rates = [summary[x][variant]["rate"] for x in langs if summary[x][variant]["rate"] is not None]
            summary["ALL"][variant] = {
                "metric": "macro",
                "n": sum(summary[x][variant]["n"] for x in langs),
                "rate": (sum(rates) / len(rates)) if rates else None,
            }
    return per_sample, summary


def format_table(summary: dict) -> str:
    """把汇总结果排成文本表，错误率以百分数显示；S/D/I 取 numeric 分数。"""

    def pct(v):
        return "n/a" if v is None else f"{v * 100:.2f}"

    lines = [f"{'lang':<5}{'metric':<7}{'n':>7}{'ref_len':>9}{'basic%':>9}{'numeric%':>10}{'S':>7}{'D':>7}{'I':>7}"]
    for lang, v in summary.items():
        b, nm = v["basic"], v["numeric"]
        lines.append(
            f"{lang:<5}{b['metric']:<7}{b['n']:>7}{nm.get('ref_len', ''):>9}{pct(b['rate']):>9}{pct(nm['rate']):>10}"
            f"{nm.get('sub', ''):>7}{nm.get('del', ''):>7}{nm.get('ins', ''):>7}"
        )
    return "\n".join(lines)


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description = "按语言计算 WER/CER，输出基础规范化与数词统一两种分数")
    parser.add_argument("--input", required = True, help = "含 sample_id、lang、reference、hypothesis 的 JSONL")
    parser.add_argument("--output", default = None, help = "逐条结果 JSONL（含对齐）")
    parser.add_argument("--summary", default = None, help = "按语言汇总的 JSON")
    args = parser.parse_args(argv)

    rows = []
    with open(args.input, encoding = "utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    per_sample, summary = score_corpus(rows = rows)
    if args.output:
        with open(args.output, "w", encoding = "utf-8") as f:
            for r in per_sample:
                f.write(json.dumps(r, ensure_ascii = False) + "\n")
    if args.summary:
        with open(args.summary, "w", encoding = "utf-8") as f:
            json.dump(summary, f, ensure_ascii = False, indent = 2)
    print(format_table(summary = summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
