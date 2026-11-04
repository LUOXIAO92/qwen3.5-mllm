"""语言检测：用 GlotLID v3（fastText，Apache-2.0）在 CPU 上判断文本是不是目标语言。

模型文件由命令行参数传入，不写进仓库。依赖 `fasttext-wheel`；fastText 的
`predict` 在 numpy 2 下会报错，所以这里直接调用底层的 `f.predict`。
"""

from __future__ import annotations

from collections.abc import Iterable

from .langs import GLOTLID_LABELS


class LangIdentifier:
    """GlotLID 的薄封装：`predict` 返回每条文本的 (标签, 概率)。"""

    def __init__(self, *, model_path: str) -> None:
        import fasttext

        fasttext.FastText.eprint = lambda *args, **kwargs: None  # 关闭加载时的警告输出
        self._model = fasttext.load_model(model_path)

    def predict(self, text: str) -> tuple[str, float]:
        """返回最可能的标签（如 `eng_Latn`）和概率。"""
        result = self._model.f.predict(text.replace("\n", " ") + "\n", 1, 0.0, "strict")
        if not result:
            return "", 0.0
        probability, label = result[0]
        return label.replace("__label__", ""), float(probability)

    def is_lang(
        self,
        text: str,
        *,
        lang: str,
        min_prob: float = 0.5,
        extra_labels: frozenset[str] = frozenset(),
    ) -> bool:
        """最可能的标签属于目标语言，且概率不低于 `min_prob`。

        `extra_labels` 给来源自带语言标注可靠的场合放宽（例如 Tatoeba 的马来语
        和印尼语在 GlotLID 里难以区分）。
        """
        label, probability = self.predict(text)
        return (label in GLOTLID_LABELS[lang] or label in extra_labels) and probability >= min_prob

    def filter(self, texts: Iterable[str], *, lang: str, min_prob: float = 0.5) -> list[bool]:
        """批量判断，返回与输入等长的布尔列表。"""
        return [self.is_lang(text, lang = lang, min_prob = min_prob) for text in texts]
