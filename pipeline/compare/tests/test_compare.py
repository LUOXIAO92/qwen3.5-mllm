"""比对模块的单元测试：python -m unittest pipeline.compare.tests.test_compare"""

from __future__ import annotations

import unittest

from pipeline.compare import LANGS, CompareResult, compare, normalize_for_scoring
from pipeline.compare.langs import get_spec
from pipeline.compare.parse import Num, analyze
from pipeline.compare.tests.cases import NEGATIVE, POSITIVE


def numbers(text: str, lang: str) -> list:
    """文本中的数值记号：(值, 是否只能是序数)。"""
    items = analyze(text = text, spec = get_spec(lang = lang))[0]
    return [(it.value, "card" not in it.kinds) for it in items if isinstance(it, Num)]


class CaseTests(unittest.TestCase):
    def test_case_counts(self):
        for lang in LANGS:
            with self.subTest(lang = lang):
                self.assertGreaterEqual(len(POSITIVE[lang]), 20)
                self.assertGreaterEqual(len(NEGATIVE[lang]), 20)

    def test_positive_cases_match(self):
        for lang in LANGS:
            for script, hyp, original, note in POSITIVE[lang]:
                with self.subTest(msg = f"{lang} {note}: {script} | {hyp}"):
                    result = compare(
                        script = script,
                        hypothesis = hyp,
                        lang = lang,
                        original = original,
                    )
                    self.assertTrue(result.match, msg = result.detail)

    def test_negative_cases_rejected(self):
        for lang in LANGS:
            for script, hyp, original, note in NEGATIVE[lang]:
                with self.subTest(msg = f"{lang} {note}: {script} | {hyp}"):
                    result = compare(
                        script = script,
                        hypothesis = hyp,
                        lang = lang,
                        original = original,
                    )
                    self.assertFalse(result.match)
                    self.assertTrue(result.detail)


class ApiTests(unittest.TestCase):
    def test_result_shape(self):
        result = compare(script = "I have three cats.", hypothesis = "I have 3 cats", lang = "en")
        self.assertIsInstance(result, CompareResult)
        match, detail = result
        self.assertTrue(match)
        self.assertEqual(detail, "")

    def test_detail_names_difference(self):
        result = compare(script = "I have three cats.", hypothesis = "I have 4 cats.", lang = "en")
        self.assertIn("three", result.detail)
        self.assertIn("4", result.detail)

    def test_original_check(self):
        ok = compare(
            script = "I have three cats.",
            hypothesis = "I have 3 cats.",
            lang = "en",
            original = "I have three cats.",
        )
        self.assertTrue(ok.match)
        bad = compare(
            script = "two thousand twenty seven",
            hypothesis = "2027",
            lang = "en",
            original = "In 2026",
        )
        self.assertFalse(bad.match)
        self.assertIn("2027", bad.detail)

    def test_original_only_checks_digits(self):
        # 识别结果写成读法时不做原文核对
        result = compare(
            script = "two thousand twenty seven",
            hypothesis = "two thousand twenty seven",
            lang = "en",
            original = "In 2026",
        )
        self.assertTrue(result.match)

    def test_empty_text(self):
        self.assertTrue(compare(script = "", hypothesis = "", lang = "en").match)
        self.assertFalse(compare(script = "hello", hypothesis = "", lang = "en").match)

    def test_unknown_language(self):
        with self.assertRaises(ValueError):
            compare(script = "a", hypothesis = "a", lang = "xx")


class NumeralTests(unittest.TestCase):
    def test_values(self):
        cases = [
            ("en", "nineteen hundred eighty-four", [(1984, False)]),
            ("en", "twenty-one hundred", [(2100, False)]),
            ("de", "zweitausendsechsundzwanzigsten", [(2026, True)]),
            ("de", "neunzehnhundertvierundachtzig", [(1984, False)]),
            ("fr", "quatre-vingt-dix-sept", [(97, False)]),
            ("es", "vigésimo primero", [(21, True)]),
            ("it", "millenovecentottantaquattro", [(1984, False)]),
            ("ru", "двухсот тридцати пяти", [(235, False)]),
            ("ru", "две тысячи двадцать шестого", [(2026, True)]),
            ("ar", "ألف وتسعمائة وأربعة وثمانون", [(1984, False)]),
            ("fa", "هزار و چهارصد و دو", [(1402, False)]),
            ("tr", "bin dokuz yüz seksen dördüncü", [(1984, True)]),
            ("vi", "hai nghìn không trăm hai mươi sáu", [(2026, False)]),
            ("th", "สองพันห้าร้อยหกสิบเจ็ด", [(2567, False)]),
            ("zh", "两千零二十六", [(2026, False)]),
            ("ja", "五千万", [(50000000, False)]),
            ("ko", "스물다섯", [(25, False)]),
            ("id", "seribu sembilan ratus", [(1900, False)]),
            ("fil", "isang libo siyam na raan walumpu't apat", [(1984, False)]),
        ]
        for lang, text, expected in cases:
            with self.subTest(lang = lang, text = text):
                self.assertEqual(numbers(text = text, lang = lang), expected)

    def test_segmented_readings_stay_separate(self):
        self.assertEqual(numbers(text = "twenty twenty-six", lang = "en"), [(20, False), (26, False)])
        self.assertEqual(numbers(text = "二零二六", lang = "zh"), [(2, False), (0, False), (2, False), (6, False)])
        self.assertEqual(numbers(text = "hai mươi năm", lang = "vi"), [(20, False), (5, False)])


class ScoringTests(unittest.TestCase):
    def test_normalize_for_scoring(self):
        cases = [
            (
                "en",
                "In nineteen eighty-four, 105 people paid three point one four dollars!",
                "in 19 84 105 people paid 3.14 dollars",
            ),
            ("de", "Ein Plus von drei Komma fünf Prozent.", "1 plus von 3.5 %"),
            ("ru", "Рост на три целых пять десятых процента", "рост на 3.5 %"),
            ("zh", "二零二六年我花了两千零二十六元。", "2026年我花了2026元"),
            ("zh", "增长了百分之三点五", "增长了%3.5"),
            ("ar", "لديَّ خَمْسَةُ كُتُبٍ", "لدي 5 كتب"),
            ("fa", "من ۲۱ کتاب دارم", "من 21 کتاب دارم"),
            ("ja", "ありがとう、三点五", "アリガトウ3.5"),
        ]
        for lang, text, expected in cases:
            with self.subTest(lang = lang, text = text):
                self.assertEqual(normalize_for_scoring(text = text, lang = lang), expected)

    def test_scoring_equal_for_equivalent_forms(self):
        pairs = [
            ("en", "I have three cats.", "i have 3 cats"),
            ("ru", "в пяти домах", "В 5 домах"),
            ("th", "ปีสองพันห้าร้อยหกสิบเจ็ด", "ปี ๒๕๖๗"),
        ]
        for lang, a, b in pairs:
            with self.subTest(lang = lang):
                self.assertEqual(
                    normalize_for_scoring(text = a, lang = lang),
                    normalize_for_scoring(text = b, lang = lang),
                )


if __name__ == "__main__":
    unittest.main()
