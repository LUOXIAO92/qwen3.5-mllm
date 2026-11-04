"""评测打分（实验 v1，任务 T09）。入口见 scoring.py。"""

from .scoring import LANGS, METRIC_BY_LANG, score_corpus, score_pair

__all__ = ["LANGS", "METRIC_BY_LANG", "score_corpus", "score_pair"]
