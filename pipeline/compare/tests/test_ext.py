"""比对增强（T21）的单元测试：python -m unittest pipeline.compare.tests.test_ext"""

from __future__ import annotations

import unittest
import warnings
from unittest import mock

from pipeline.compare import compare, ja_reading, normalize_for_scoring, zh_convert
from pipeline.compare.tests.cases_ext import EXT_NEGATIVE, EXT_POSITIVE

READING = ja_reading.available()
OPENCC = zh_convert._converters() is not None


def _run(test: unittest.TestCase, cases: dict, expect: bool) -> None:
    for group, rows in cases.items():
        if group == "ja_reading" and not READING:
            continue
        for lang, script, hyp, original, note in rows:
            with test.subTest(msg = f"{group} {lang} {note}: {script} | {hyp}"):
                result = compare(
                    script = script,
                    hypothesis = hyp,
                    lang = lang,
                    original = original,
                )
                test.assertEqual(result.match, expect, msg = result.detail)
                if not expect:
                    test.assertTrue(result.detail)


class ExtCaseTests(unittest.TestCase):
    def test_case_counts(self):
        for group in EXT_POSITIVE:
            with self.subTest(group = group):
                self.assertGreaterEqual(len(EXT_POSITIVE[group]), 8)
                self.assertGreaterEqual(len(EXT_NEGATIVE[group]), 8)

    def test_positive_cases_match(self):
        _run(test = self, cases = EXT_POSITIVE, expect = True)

    def test_negative_cases_rejected(self):
        _run(test = self, cases = EXT_NEGATIVE, expect = False)


@unittest.skipUnless(READING, "需要 fugashi 和 unidic-lite")
class ReadingTests(unittest.TestCase):
    def test_reading(self):
        reading = ja_reading.to_reading
        self.assertTrue(reading(text = "4月1日に言った").startswith("4月1日"))
        self.assertEqual(reading(text = "四月一日"), "四月一日")
        self.assertEqual(reading(text = "ノヴゴロド"), "ノブゴロド")
        self.assertEqual(reading(text = "言って"), reading(text = "いって"))
        self.assertEqual(reading(text = "交響曲"), reading(text = "公共局"))
        self.assertEqual(reading(text = "麗奈"), reading(text = "レーナ"))
        # 只由假名写成的词按原样读，不被形态分析改成别的词形
        self.assertNotEqual(reading(text = "論争になった"), reading(text = "論争になた"))
        self.assertNotEqual(reading(text = "ありがとう"), reading(text = "ありがと"))

    def test_mismatch_detail_uses_reading(self):
        result = compare(script = "今日は雨です", hypothesis = "今日は亀です", lang = "ja")
        self.assertFalse(result.match)
        self.assertIn("按读音", result.detail)


class FallbackTests(unittest.TestCase):
    def test_missing_dependency_falls_back_to_writing(self):
        no_tagger = mock.patch.object(target = ja_reading, attribute = "_tagger", return_value = None)
        not_warned = mock.patch.object(target = ja_reading, attribute = "_warned", new = False)
        with no_tagger, not_warned:
            with warnings.catch_warnings(record = True) as caught:
                warnings.simplefilter("always")
                differ = compare(script = "私に言って", hypothesis = "私にいって", lang = "ja")
                same = compare(script = "鉛筆を三本買った。", hypothesis = "鉛筆を3本買った。", lang = "ja")
        self.assertFalse(differ.match)
        self.assertIn(ja_reading.MISSING_NOTE, differ.detail)
        self.assertTrue(same.match)
        self.assertTrue(any(ja_reading.MISSING_NOTE in str(w.message) for w in caught))


@unittest.skipUnless(OPENCC, "需要 opencc 1.4.2")
class ZhConvertTests(unittest.TestCase):
    def test_word_level_conversion(self):
        # 单字表把「著」「瞭」「衹」留作原字，词组级转换按词选字
        positive = [
            ("他看着我，又执着又沉着。", "他看著我，又執著又沉著。"),
            ("乾隆年间的著名学者都了解此事。", "乾隆年間的著名學者都瞭解此事。"),
            ("飞机着陆以后只剩三人。", "飛機著陸以後衹剩3人。"),
            ("这部原著显著不同。", "這部原著顯著不同。"),
        ]
        negative = [
            ("他看着我。", "他看到我。"),
            ("著名学者", "看着学者"),
            ("飞机着陆以后只剩三人。", "飛機著陸以後衹剩4人。"),
        ]
        for script, hyp in positive:
            with self.subTest(script = script, hyp = hyp):
                result = compare(script = script, hypothesis = hyp, lang = "zh")
                self.assertTrue(result.match, msg = result.detail)
        for script, hyp in negative:
            with self.subTest(script = script, hyp = hyp):
                self.assertFalse(compare(script = script, hypothesis = hyp, lang = "zh").match)

    def test_same_as_tn(self):
        try:
            from pipeline.tn import core as tn_core
        except ImportError as exc:
            self.skipTest(reason = f"无法导入 pipeline.tn：{exc}")
        self.assertEqual(zh_convert._ZHU_FIXES.pattern, tn_core._ZHU_FIXES.pattern)
        for text in ["他看著我，所著之書顯著", "乾隆年間的乾淨綫路", "著名作家穿著西裝", "裡面有隻貓"]:
            with self.subTest(text = text):
                self.assertEqual(zh_convert.to_simplified(text = text), tn_core.to_simplified(text = text))


class ZhFallbackTests(unittest.TestCase):
    def test_missing_opencc_falls_back_to_table(self):
        no_opencc = mock.patch.object(target = zh_convert, attribute = "_converters", return_value = None)
        not_warned = mock.patch.object(target = zh_convert, attribute = "_warned", new = False)
        with no_opencc, not_warned:
            with warnings.catch_warnings(record = True) as caught:
                warnings.simplefilter("always")
                differ = compare(script = "他看着我。", hypothesis = "他看著我。", lang = "zh")
                same = compare(script = "他买了三本书。", hypothesis = "他買了3本書。", lang = "zh")
        self.assertFalse(differ.match)
        self.assertIn(zh_convert.MISSING_NOTE, differ.detail)
        self.assertTrue(same.match)
        self.assertTrue(any(zh_convert.MISSING_NOTE in str(w.message) for w in caught))


class ScoringUnchangedTests(unittest.TestCase):
    def test_scoring_ignores_units_and_times(self):
        # 评测打分的规范化不受比对增强影响
        cases = [
            ("en", "It costs $5 at 5:30.", "it costs 5 at 5 30"),
            ("zh", "三点十五分", "3.十5分"),
            ("de", "São Paulo", "são paulo"),
        ]
        for lang, text, expected in cases:
            with self.subTest(lang = lang, text = text):
                self.assertEqual(normalize_for_scoring(text = text, lang = lang), expected)


if __name__ == "__main__":
    unittest.main()
