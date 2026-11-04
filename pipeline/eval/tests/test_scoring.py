"""打分器测试，用手工构造的样例验证。运行：python -m unittest pipeline.eval.tests.test_scoring"""

import json
import os
import tempfile
import unittest

from pipeline.eval.scoring import LANGS, METRIC_BY_LANG, align, main, normalize_basic, score_corpus, score_pair


def counts(res: dict, variant: str = "basic") -> tuple:
    r = res[variant]
    return (r["sub"], r["del"], r["ins"], r["ref_len"])


class TestAlign(unittest.TestCase):
    def test_ops(self):
        ops = align(ref = list("abc"), hyp = list("axc"))
        self.assertEqual([o[0] for o in ops], ["ok", "sub", "ok"])
        ops = align(ref = ["a", "b", "c"], hyp = ["a", "c"])
        self.assertEqual([o for o in ops if o[0] != "ok"], [("del", "b", None)])
        ops = align(ref = ["a", "c"], hyp = ["a", "b", "c"])
        self.assertEqual([o for o in ops if o[0] != "ok"], [("ins", None, "b")])


class TestWer(unittest.TestCase):
    def test_identical(self):
        res = score_pair(reference = "The cat sat.", hypothesis = "the cat sat", lang = "en")
        self.assertEqual(counts(res), (0, 0, 0, 3))
        self.assertEqual(res["basic"]["rate"], 0.0)

    def test_substitution(self):
        res = score_pair(reference = "the cat sat", hypothesis = "the dog sat", lang = "en")
        self.assertEqual(counts(res), (1, 0, 0, 3))
        self.assertAlmostEqual(res["basic"]["rate"], 1 / 3)

    def test_deletion(self):
        res = score_pair(reference = "the cat sat down", hypothesis = "the cat down", lang = "en")
        self.assertEqual(counts(res), (0, 1, 0, 4))

    def test_insertion(self):
        res = score_pair(reference = "the cat", hypothesis = "the big cat", lang = "en")
        self.assertEqual(counts(res), (0, 0, 1, 2))

    def test_empty_reference(self):
        res = score_pair(reference = "", hypothesis = "hello world", lang = "en")
        self.assertEqual(counts(res), (0, 0, 2, 0))
        self.assertIsNone(res["basic"]["rate"])

    def test_punctuation_and_case(self):
        res = score_pair(reference = "Don't stop, Ñandú!", hypothesis = "dont stop ñandú", lang = "es")
        self.assertEqual(res["basic"]["errors"], 0)

    def test_diacritics_kept(self):
        res = score_pair(reference = "café", hypothesis = "cafe", lang = "fr")
        self.assertEqual(res["basic"]["errors"], 1)

    def test_russian_yo(self):
        res = score_pair(reference = "всё", hypothesis = "все", lang = "ru")
        self.assertEqual(res["basic"]["errors"], 0)


class TestCer(unittest.TestCase):
    def test_metrics(self):
        for lang in ("zh", "ja", "th", "ko"):
            self.assertEqual(METRIC_BY_LANG[lang], "cer")
        for lang in LANGS:
            if lang not in ("zh", "ja", "th", "ko"):
                self.assertEqual(METRIC_BY_LANG[lang], "wer")

    def test_zh_char(self):
        res = score_pair(reference = "今天天气很好", hypothesis = "今天天汽很好", lang = "zh")
        self.assertEqual(counts(res), (1, 0, 0, 6))

    def test_zh_traditional_and_punct(self):
        res = score_pair(reference = "今天天气很好。", hypothesis = "今天天氣很好", lang = "zh")
        self.assertEqual(res["basic"]["errors"], 0)

    def test_ja_kana(self):
        # 平假名与片假名统一后一致；少一个字
        res = score_pair(reference = "おはようございます", hypothesis = "オハヨウゴザイマ", lang = "ja")
        self.assertEqual(counts(res), (0, 1, 0, 9))

    def test_ko_ignores_spaces(self):
        res = score_pair(reference = "안녕하세요 여러분", hypothesis = "안녕하세요여러분", lang = "ko")
        self.assertEqual(res["basic"]["errors"], 0)
        res = score_pair(reference = "안녕하세요 여러분", hypothesis = "안녕하세요 여러부", lang = "ko")
        self.assertEqual(counts(res), (1, 0, 0, 8))

    def test_th_char(self):
        res = score_pair(reference = "สวัสดีครับ", hypothesis = "สวัสดีค่ะ", lang = "th")
        self.assertEqual(res["basic"]["ref_len"], len("สวัสดีครับ"))
        self.assertGreater(res["basic"]["errors"], 0)
        res = score_pair(reference = "สวัสดี ครับ", hypothesis = "สวัสดีครับ", lang = "th")
        self.assertEqual(res["basic"]["errors"], 0)


class TestNumerals(unittest.TestCase):
    def test_en_words_vs_digits(self):
        ref, hyp = "he has twenty five books", "he has 25 books"
        res = score_pair(reference = ref, hypothesis = hyp, lang = "en")
        # basic：「twenty five」两个词对「25」一个词 → 1 替换 + 1 删除
        self.assertEqual(res["basic"]["errors"], 2)
        # numeric：两边统一成 25
        self.assertEqual(res["numeric"]["errors"], 0)

    def test_zh_words_vs_digits(self):
        res = score_pair(reference = "他有二十五本书", hypothesis = "他有25本书", lang = "zh")
        self.assertEqual(counts(res), (2, 1, 0, 7))
        self.assertEqual(res["numeric"]["errors"], 0)

    def test_ru_case_error_remains(self):
        # 变格错误在数词统一后仍然算错：不会被当成同一个数
        res = score_pair(reference = "в пяти домах", hypothesis = "в 5 домах", lang = "ru")
        self.assertEqual(res["numeric"]["errors"], 0)
        res = score_pair(reference = "пять домов", hypothesis = "шесть домов", lang = "ru")
        self.assertEqual(res["numeric"]["errors"], 1)

    def test_real_error_stays(self):
        res = score_pair(reference = "he has twenty five books", hypothesis = "he has 26 books", lang = "en")
        self.assertEqual(res["numeric"]["errors"], 1)

    def test_basic_text_keeps_digits(self):
        self.assertEqual(normalize_basic(text = "It costs 3.5%, ok?", lang = "en"), "it costs 3.5% ok")


class TestCorpus(unittest.TestCase):
    ROWS = [
        {"sample_id": "a", "lang": "en", "reference": "one two three four", "hypothesis": "one two three"},
        {"sample_id": "b", "lang": "en", "reference": "five six", "hypothesis": "five seven"},
        {"sample_id": "c", "lang": "zh", "reference": "今天很好", "hypothesis": "今天很好"},
    ]

    def test_corpus_level_is_sum_over_sum(self):
        per_sample, summary = score_corpus(rows = self.ROWS)
        self.assertEqual(len(per_sample), 3)
        en = summary["en"]["basic"]
        # 编辑距离总和 / 参考长度总和 = 2 / 6，不是 (1/4 + 1/2) / 2
        self.assertEqual((en["errors"], en["ref_len"]), (2, 6))
        self.assertAlmostEqual(en["rate"], 2 / 6)
        self.assertEqual(summary["zh"]["basic"]["rate"], 0.0)
        self.assertAlmostEqual(summary["ALL"]["basic"]["rate"], (2 / 6 + 0.0) / 2)

    def test_cli(self):
        with tempfile.TemporaryDirectory() as d:
            inp = os.path.join(d, "in.jsonl")
            out = os.path.join(d, "out.jsonl")
            summ = os.path.join(d, "s.json")
            with open(inp, "w", encoding = "utf-8") as f:
                for r in self.ROWS:
                    f.write(json.dumps(r, ensure_ascii = False) + "\n")
            self.assertEqual(main(argv = ["--input", inp, "--output", out, "--summary", summ]), 0)
            with open(out, encoding = "utf-8") as f:
                lines = [json.loads(x) for x in f]
            self.assertEqual(lines[0]["sample_id"], "a")
            self.assertEqual(lines[0]["basic"]["diff"], [["del", "four", None]])
            with open(summ, encoding = "utf-8") as f:
                self.assertIn("ALL", json.load(f))

    def test_all_langs_run(self):
        for lang in LANGS:
            res = score_pair(reference = "test 1 2", hypothesis = "test", lang = lang)
            self.assertIn("numeric", res)


if __name__ == "__main__":
    unittest.main()
