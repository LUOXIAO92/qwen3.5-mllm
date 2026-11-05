"""比对增强（T21）的单元测试：python -m unittest pipeline.compare.tests.test_ext"""

from __future__ import annotations

import unittest
import warnings
from unittest import mock

from pipeline.compare import compare, ja_reading, normalize_for_scoring
from pipeline.compare.tests.cases_ext import EXT_NEGATIVE, EXT_POSITIVE

READING = ja_reading.available()


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
