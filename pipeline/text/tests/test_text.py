"""pipeline/text 的单元测试：过滤规则、规范化与排除、切句、分档、去重、整体抽取。

运行：`python -m pytest pipeline/text/tests`。真实的 GlotLID 测试需要设置环境变量
`GLOTLID_MODEL` 指向 model_v3.bin，没有设置时跳过。
"""

from __future__ import annotations

import os
import random

import pytest

from pipeline.text.dedup import Deduper, dedup
from pipeline.text.exclusion import is_excluded
from pipeline.text.extract import extract_candidates
from pipeline.text.filters import check_text
from pipeline.text.lengths import estimate_duration, length_bin
from pipeline.text.normalize import clean_text, norm_key
from pipeline.text.sources.tatoeba import has_common_name
from pipeline.text.spans import build_spans, clean_runs
from pipeline.text.split import split_sentences

# ---------------------------------------------------------------- 过滤规则


GOOD = {
    "en": "The old library on the corner opens at nine every morning.",
    "de": "Die alte Bibliothek an der Ecke öffnet jeden Morgen um neun Uhr.",
    "fr": "La vieille bibliothèque du coin ouvre tous les matins à neuf heures.",
    "ru": "Старая библиотека на углу открывается каждое утро в девять часов.",
    "zh": "街角那家老图书馆每天早上九点开门。",
    "ja": "角にある古い図書館は毎朝九時に開きます。",
    "ko": "모퉁이에 있는 오래된 도서관은 매일 아침 아홉 시에 문을 엽니다.",
    "th": "ห้องสมุดเก่าตรงหัวมุมเปิดทุกเช้าตอนเก้าโมง",
    "ar": "تفتح المكتبة القديمة عند الزاوية كل صباح في الساعة التاسعة.",
    "fa": "کتابخانه قدیمی سر گوشه هر روز صبح ساعت نه باز می شود.",
}


@pytest.mark.parametrize("lang", sorted(GOOD))
def test_good_sentences_pass(lang):
    assert check_text(GOOD[lang], lang = lang) is None


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("Visit https://example.org/page for more details about it.", "url"),
        ("See www.example.com for more details about the library.", "url"),
        ("Write to someone@mail.org for more details about the library.", "email"),
        ("Contact email@example.com for the details of this library.", "url"),
        ("def main(): return value for the whole library program.", "code"),
        ("The function foo() { return bar; } is called by the library.", "code"),
        ("<div class=\"x\">The library opens at nine every morning.</div>", "code"),
        ("The value is set with x == y and the library opens at nine.", "code"),
        ("import numpy as np then the library opens at nine every morning.", "code"),
        ("The library opens at nine every morning and closes at five;", "list_intro"),
        ("The library offers the following services:", "list_intro"),
        ("以下為各獎項結果：", "list_intro"),
        ("- The library opens at nine every morning of the week.", "markup"),
        ("1. The library opens at nine every morning of the week.", "markup"),
        ("# The library opens at nine every morning of the week.", "markup"),
        ("The **library** opens at nine every morning of the week.", "markup"),
        ("| name | time | opens at nine every morning of the week.", "markup"),
        ("The library opens at nine every morning of the week [1].", "bracket_note"),
        ("The library (French: , German: ) opens at nine every morning.", "template_hole"),
        ("史特拉斯堡座堂（法語：，德語：）是一座著名的大教堂。", "template_hole"),
        ("The library () opens at nine every morning of the week.", "template_hole"),
    ],
)
def test_unreadable_rules(text, reason):
    lang = "zh" if any("一" <= ch <= "鿿" for ch in text) else "en"
    got = check_text(text, lang = lang)
    assert got is not None, text
    if reason not in {"url", "email", "code"}:
        assert got == reason, (text, got)


def test_url_and_code_always_rejected():
    for text in [
        "Visit https://example.org/page for more details about it.",
        "Write to someone@mail.org for more details about the library.",
        "def main(): return value for the whole library program.",
    ]:
        assert check_text(text, lang = "en") in {"url", "email", "code"}


def test_heading_and_fragment_rejected():
    assert check_text("The History of the Old Library", lang = "en") == "no_terminal"
    assert check_text("Bibliothek", lang = "de") == "too_short"
    assert check_text("the library opens at nine every morning.", lang = "en") == "lowercase_start"
    assert check_text("THE LIBRARY OPENS AT NINE EVERY MORNING.", lang = "en") == "all_caps"


def test_list_like_rejected():
    assert check_text("Apples, pears, plums, cherries, grapes, figs, dates, limes.", lang = "en") == "list_like"
    assert check_text("苹果、梨、李子、樱桃、葡萄、无花果、枣子。", lang = "zh") == "list_like"


def test_non_target_language_rejected_by_script():
    assert check_text("Старая библиотека на углу открывается каждое утро.", lang = "en") == "script"
    assert check_text("The old library on the corner opens every morning.", lang = "ru") == "script"
    assert check_text("The old library on the corner opens every morning.", lang = "zh") == "script"
    assert check_text("街角那家老图书馆每天早上九点开门。", lang = "ja") == "no_kana"
    assert check_text("这是一个很长的中文句子，不是日语。", lang = "ja") == "no_kana"


def test_gambling_and_digits():
    assert check_text("欢迎来到澳门娱乐城，注册送彩金立即下注。", lang = "zh") == "gambling"
    assert check_text("Call 123456789012 now to hear the library hours.", lang = "en") == "long_number"
    assert check_text("1990 2001 2010 2020 3000 4000 5000 6000 70.", lang = "en") == "digits"


def test_unbalanced_and_repetition():
    assert check_text("The library (opens at nine every morning.", lang = "en") == "unbalanced"
    assert check_text("The library opens at nine nine nine nine every morning.", lang = "en") == "repetition"


def test_empty_quotes_rejected():
    assert check_text("蓬塔穆松城市公共交通（商业名称为“”）是当地的城市公交系统。", lang = "zh") == "template_hole"
    assert check_text('He said "Yes" and then "No" to the whole library plan.', lang = "en") is None


def test_thai_does_not_need_terminal():
    assert check_text(GOOD["th"], lang = "th") is None


# ---------------------------------------------------------------- 规范化、排除


def test_clean_text_removes_invisible_and_russian_accent():
    assert clean_text("a​b­c  d\n", lang = "en") == "abc d"
    assert clean_text("Мо́сква", lang = "ru") == "Москва"
    assert "‌" in clean_text("می‌شود", lang = "fa")  # ZWNJ 保留


def test_norm_key_ignores_case_punct_space_and_script_variant():
    assert norm_key("Hello, World!", lang = "en") == norm_key("hello world", lang = "en")
    assert norm_key("要了解圣殿骑士，就必须了解。", lang = "zh") == norm_key("要 了 解 圣 殿 骑 士 就 必 须 了 解", lang = "zh")
    assert norm_key("要了解聖殿騎士。", lang = "zh") == norm_key("要了解圣殿骑士", lang = "zh")
    assert norm_key("ＡＢＣ１２３", lang = "en") == "abc123"


def test_exclusion_matches_whole_and_sentence():
    exclusion = {"en": {norm_key("The cat sat on the mat.", lang = "en")}}
    assert is_excluded("the cat sat on the mat", lang = "en", exclusion = exclusion)
    paragraph = "Dogs bark loudly. The cat sat on the mat! Birds sing."
    assert is_excluded(paragraph, lang = "en", exclusion = exclusion)
    assert not is_excluded("The cat sat on the rug.", lang = "en", exclusion = exclusion)
    assert not is_excluded("The cat sat on the mat.", lang = "de", exclusion = exclusion)


# ---------------------------------------------------------------- 切句、分档、去重


def test_split_sentences():
    assert split_sentences("Dr. Smith went home. He slept.", lang = "en") == ["Dr. Smith went home.", "He slept."]
    assert split_sentences("J. R. R. Tolkien wrote it. Yes!", lang = "en") == ["J. R. R. Tolkien wrote it.", "Yes!"]
    assert split_sentences("今天天气很好。我们去公园吧！好的。", lang = "zh") == ["今天天气很好。", "我们去公园吧！", "好的。"]
    assert split_sentences("ทดสอบข้อความ ภาษาไทย", lang = "th") == ["ทดสอบข้อความ ภาษาไทย"]


def test_length_bins():
    assert length_bin(0.5) is None
    assert length_bin(3.0) == "short"
    assert length_bin(5.0) == "short"
    assert length_bin(5.1) == "medium"
    assert length_bin(15.0) == "medium"
    assert length_bin(15.1) == "long"
    assert length_bin(30.0) == "long"
    assert length_bin(30.1) is None
    assert estimate_duration("a" * 135, lang = "en") == pytest.approx(10.0, rel = 0.01)


def test_dedup():
    assert list(dedup(["Hello, world.", "hello world", "Hello there."], lang = "en")) == ["Hello, world.", "Hello there."]
    deduper = Deduper(lang = "en", max_per_prefix = 1, prefix_len = 5)
    assert deduper.add("Alpha one")
    assert not deduper.add("Alpha two")


def test_has_common_name():
    assert has_common_name("Tom is here.")
    assert has_common_name("トムはここにいます。")
    assert not has_common_name("Tomorrow is here.")


# ---------------------------------------------------------------- 段落拼片段、整体抽取


def test_spans_are_whole_sentences_and_break_at_bad_sentence():
    paragraph = (
        "The old library opens at nine every morning. It closes at five in the evening. "
        "See www.example.com for details. The reading room is quiet and warm."
    )
    runs = clean_runs(
        paragraph,
        lang = "en",
        sentence_ok = lambda s: check_text(s, lang = "en") is None,
    )
    assert len(runs) == 2
    assert all("www" not in s for run in runs for s in run)
    spans = build_spans(runs, lang = "en", rng = random.Random(0), bin_weights = {"short": 1.0})
    assert spans
    for span in spans:
        assert span.endswith(".")


class _FakeIdentifier:
    """只认英文字符的假检测器，用来测试检测环节的接线。"""

    def is_lang(self, text, *, lang, min_prob = 0.5, extra_labels = frozenset()):
        return "bibliothek" not in text.lower()


def test_extract_candidates_end_to_end():
    records = [
        ("The old library on the corner opens at nine every morning.", "a:1"),
        ("The old library on the corner opens at nine every morning!", "a:2"),  # 键重复
        ("Visit https://example.org/page for more details about it.", "a:3"),
        ("The cat sat on the mat.", "a:4"),  # 在排除清单中
        ("Die Bibliothek öffnet jeden Morgen um neun Uhr in der Stadt.", "a:5"),  # 被语言检测拒绝
        ("The History of the Old Library", "a:6"),
        ("Many people walk to the old library on the corner every single morning before work.", "a:7"),
    ]
    from collections import Counter

    counter = Counter()
    exclusion = {"en": {norm_key("The cat sat on the mat.", lang = "en")}}
    candidates = extract_candidates(
        records,
        lang = "en",
        exclusion = exclusion,
        identifier = _FakeIdentifier(),
        caps = {"short": 10, "medium": 10, "long": 10},
        stats = counter,
    )
    texts = {row["ref"]: row for row in candidates}
    assert set(texts) == {"a:1", "a:7"}
    assert counter["drop_duplicate"] == 1
    assert counter["drop_url"] == 1
    assert counter["drop_fleurs"] == 1
    assert counter["drop_langid"] == 1
    assert counter["drop_no_terminal"] == 1
    assert texts["a:1"]["length_bin"] in {"short", "medium"}


@pytest.mark.skipif("GLOTLID_MODEL" not in os.environ, reason = "需要 GLOTLID_MODEL 指向 model_v3.bin")
def test_glotlid_real_model():
    from pipeline.text.langid import LangIdentifier

    identifier = LangIdentifier(model_path = os.environ["GLOTLID_MODEL"])
    assert identifier.is_lang(GOOD["en"], lang = "en")
    assert identifier.is_lang(GOOD["de"], lang = "de")
    assert not identifier.is_lang(GOOD["de"], lang = "en")
    assert not identifier.is_lang(GOOD["ru"], lang = "en")
    assert not identifier.is_lang(GOOD["zh"], lang = "ja")
