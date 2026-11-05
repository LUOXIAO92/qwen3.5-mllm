"""朗读稿生成的入口：清理原文、检查需要改写的片段、决定 unchanged / normalized / dropped。"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import NamedTuple

from ..compare import compare
from .detect import describe, find_issues
from .langs import get_rules
from .rules import apply_rules

LANGS = ("zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it", "id", "ms", "th", "vi", "fil", "ar", "tr", "fa")

# 删除的不可见字符：零宽空格、BOM、词连接符、软连字符、方向控制符。
# 零宽不连字 U+200C 在波斯语中分隔词的部件，零宽连接符 U+200D 用于组合字形，都保留。
_INVISIBLE = [0x200B, 0xFEFF, 0x2060, 0x00AD, 0x200E, 0x200F, 0x061C]
_INVISIBLE += list(range(0x202A, 0x202F)) + list(range(0x2066, 0x206A))
_REMOVE_INVISIBLE = dict.fromkeys(_INVISIBLE, None)
_SPACES = re.compile(pattern = r"\s+")


class TnResult(NamedTuple):
    script: str | None
    status: str
    note: str


@lru_cache(maxsize = 1)
def _opencc() -> tuple:
    """OpenCC（Apache-2.0）的两个转换器：先把香港异体字转成标准繁体，再按台湾用字和词组转成简体。

    词组级转换能区分「乾隆」与「乾净」、「著名」与「看著」，单字表做不到。
    """
    try:
        import opencc
    except ImportError as exc:
        raise RuntimeError("中文繁简转换需要 OpenCC（Apache-2.0）：uv pip install opencc==1.4.2") from exc
    # 不启用 tofu risk 词典：它把少数字转成扩展区的简化字（䭔 → 𫗰），TTS 的词表多半没有这些字
    hk2t = opencc.OpenCC(config = "hk2t", include_tofu_risk_dictionaries = False)
    tw2s = opencc.OpenCC(config = "tw2s", include_tofu_risk_dictionaries = False)
    return hk2t, tw2s


# tw2s 把表示「写作」的「著」也改成了「着」（所著、显著、原著、著称），这里改回来
_ZHU_FIXES = re.compile(
    pattern = r"(?:显|卓|昭|原|名|巨|编|译|专|遗|论|拙|土|合)着|着(?:称|述|录)|所着(?![重眼想手陆迷急火凉实落色意])"
    r"|(?<![一-鿿])着有|着书立说",
)


def to_simplified(text: str) -> str:
    hk2t, tw2s = _opencc()
    text = tw2s.convert(text = hk2t.convert(text = text))
    return _ZHU_FIXES.sub(repl = lambda m: m.group().replace("着", "著"), string = text)


def clean(text: str, lang: str) -> str:
    """NFC、删除不可见字符、空白合并为一个空格；中文转为简体。"""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_REMOVE_INVISIBLE)
    text = _SPACES.sub(repl = " ", string = text).strip()
    if lang == "zh":
        text = to_simplified(text = text)
    return text


def _unique(items: list) -> list:
    seen = []
    for it in items:
        if it not in seen:
            seen.append(it)
    return seen


def _merge_reasons(reasons: list) -> list:
    """同一原因的多处片段合并：「暂不处理「5000」」「暂不处理「2021」」→「暂不处理「5000」「2021」」。"""
    grouped = {}
    for reason in reasons:
        head, sep, tail = reason.partition("「")
        grouped.setdefault(head, [])
        piece = sep + tail
        if piece not in grouped[head]:
            grouped[head].append(piece)
    return [head + "".join(pieces) for head, pieces in grouped.items()]


def _inside(it, spans: list) -> bool:
    return any(a <= it.start and it.end <= b for a, b in spans)


def normalize_entry(original: str, lang: str, mode: str = "basic") -> TnResult:
    """为一条原文生成朗读稿。

    没有需要改写的片段时，朗读稿就是清理后的原文（unchanged）。mode 为 passthrough 时，有数字、符号、
    缩写、罗马数字或网址的条目一律丢弃；为 basic 时先按该语言的规则把数字改写成读法，全部改写成功、
    改写后不再有需要改写的片段、每处读法与原写法经 pipeline.compare 核对数值一致，才标 normalized，
    否则丢弃。tn_note 写明改写或丢弃的原因。
    """
    if lang not in LANGS:
        raise ValueError(f"不支持的语言代码：{lang!r}")
    if mode not in ("passthrough", "basic"):
        raise ValueError(f"mode 只能是 passthrough 或 basic：{mode!r}")
    text = clean(text = original, lang = lang)
    if not text:
        return TnResult(script = None, status = "dropped", note = "原文为空")
    issues = find_issues(text = text, lang = lang)
    if not issues:
        note = "转为简体" if lang == "zh" and text != clean(text = original, lang = "") else ""
        return TnResult(script = text, status = "unchanged", note = note)
    if mode == "passthrough":
        return TnResult(script = None, status = "dropped", note = describe(issues = issues))

    rewrite = apply_rules(text = text, lang = lang, rules = get_rules(lang = lang))
    # 原文中需要改写的片段（数字、符号、缩写等）必须全部落在被替换的片段里；
    # 落在 Skip 片段里的已由规则说明原因，其余（例如「No. 47」中的 No.）另行列出
    skipped = [span for _, span in rewrite.skips]
    unexplained = [it for it in issues if not _inside(it = it, spans = rewrite.replaced + skipped)]
    if rewrite.skips or unexplained:
        reasons = _merge_reasons(reasons = [reason for reason, _ in rewrite.skips])
        if unexplained:
            reasons.append(describe(issues = unexplained))
        return TnResult(script = None, status = "dropped", note = "；".join(reasons))
    left = find_issues(text = rewrite.text, lang = lang)
    if left:
        return TnResult(script = None, status = "dropped", note = "改写后仍" + describe(issues = left))
    failed = []
    for spoken, written in rewrite.checks:
        if not compare(script = spoken, hypothesis = written, lang = lang).match:
            failed.append(f"「{spoken}」≠「{written}」")
    if failed:
        return TnResult(script = None, status = "dropped", note = "读法核对不一致：" + "、".join(failed))
    labels = _unique(items = [f"{name}「{surface}」" for name, surface in rewrite.kinds])
    return TnResult(script = rewrite.text, status = "normalized", note = "改写：" + "、".join(labels))
