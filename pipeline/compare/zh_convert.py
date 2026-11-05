"""中文繁简转换：与朗读稿生成（pipeline.tn.core.to_simplified）相同，用 OpenCC 1.4.2 的词组级转换。

先用 hk2t 把香港异体字转成标准繁体，再用 tw2s 按台湾用字和词组转成简体，最后把 tw2s 改成「着」的
「写作」义的「著」改回来（显著、原著、所著、著称）。词组级转换能区分「看著」（看着）与「著名」、
「乾隆」与「乾净」，ASR 输出繁体时与朗读稿的字形一致。normalize.py 在这之后再查一遍单字表，因为朗读稿
已经转过一次，再转一次不一定不变（原文「徵調」转成「徵调」，「徵调」再转是「征调」）。OpenCC（Apache-2.0）是可选依赖：
没有安装时 normalize.py 退回 data/zh_t2s.txt 的单字表，发出一次 RuntimeWarning，不一致的差异说明后注明
MISSING_NOTE。opencc-python-reimplemented 也提供 opencc 模块，但没有 hk2t、tw2s 的词组配置，按缺失处理。
"""

from __future__ import annotations

import re
import warnings
from functools import lru_cache

MISSING_NOTE = "中文按单字表转简体：没有安装 opencc 1.4.2"

# 与 pipeline.tn.core._ZHU_FIXES 相同（tests/test_ext.py 检查两者一致）
_ZHU_FIXES = re.compile(
    pattern = r"(?:显|卓|昭|原|名|巨|编|译|专|遗|论|拙|土|合)着|着(?:称|述|录)|所着(?![重眼想手陆迷急火凉实落色意])"
    r"|(?<![一-鿿])着有|着书立说",
)

_warned = False


@lru_cache(maxsize = 1)
def _converters() -> tuple | None:
    try:
        import opencc

        # 不启用 tofu risk 词典：它把少数字转成扩展区的简化字（䭔 → 𫗰）
        hk2t = opencc.OpenCC(config = "hk2t", include_tofu_risk_dictionaries = False)
        tw2s = opencc.OpenCC(config = "tw2s", include_tofu_risk_dictionaries = False)
    except Exception:  # 未安装，或装的是没有这些配置与参数的其他实现
        return None
    return hk2t, tw2s


def available() -> bool:
    """OpenCC 可用时返回 True；不可用时第一次调用发出 RuntimeWarning。"""
    global _warned
    if _converters() is not None:
        return True
    if not _warned:
        _warned = True
        warnings.warn(
            message = MISSING_NOTE + "（uv pip install opencc==1.4.2），繁体识别结果可能因字形不同判为不一致",
            category = RuntimeWarning,
            stacklevel = 2,
        )
    return False


def to_simplified(text: str) -> str:
    """词组级转为简体。调用前先用 available() 确认 OpenCC 可用。"""
    hk2t, tw2s = _converters()
    text = tw2s.convert(text = hk2t.convert(text = text))
    return _ZHU_FIXES.sub(repl = lambda m: m.group().replace("着", "著"), string = text)
