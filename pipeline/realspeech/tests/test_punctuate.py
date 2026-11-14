"""标点还原的校验与放回：python -m unittest pipeline.realspeech.tests.test_punctuate"""

import unittest

from pipeline.realspeech.punctuate import needs_punctuation, score, strip_marks, transfer

ZWNJ = "‌"


class TransferTest(unittest.TestCase):
    def check(self, source: str, llm_output: str, status: str, output: str = None, reason: str = None) -> None:
        result = transfer(source = source, llm_output = llm_output, lang = "fa")
        self.assertEqual(result["status"], status, msg = result)
        if output is not None:
            self.assertEqual(result["output"], output)
        if reason is not None:
            self.assertEqual(result["reason"], reason)

    def test_marks_follow_source_spacing(self):
        # LLM 把「همه جانبه」连成带 ZWNJ 的一个词，放回时仍用原文本的空格
        self.check(source = "قدرت همه جانبه آن همه را", llm_output = f"قدرت همه{ZWNJ}جانبه، آن همه را.",
                   status = "ok", output = "قدرت همه جانبه، آن همه را.")

    def test_source_zwnj_kept(self):
        self.check(source = f"می{ZWNJ}خواهم بروم", llm_output = "میخواهم بروم.", status = "ok", output = f"می{ZWNJ}خواهم بروم.")

    def test_small_spelling_fix_is_aligned(self):
        # LLM 纠正了一个字母（پاشاه → پادشاه），词仍取原文本
        self.check(source = "از پاشاه گرفته تا افراد عادی", llm_output = "از پادشاه گرفته تا افراد عادی.",
                   status = "aligned", output = "از پاشاه گرفته تا افراد عادی.")

    def test_rewritten_words_fail(self):
        self.check(source = "از پاشاه گرفته تا افراد عادی", llm_output = "از شاه تا مردم عادی.", status = "failed",
                   reason = "words_changed")

    def test_mark_inside_word_fails(self):
        self.check(source = "کتابخانه بزرگ شهر", llm_output = "کتاب، خانه بزرگ شهر.", status = "failed",
                   reason = "mark_inside_word")

    def test_digit_conversion_fails(self):
        self.check(source = "در سال ۱۳۹۹ بود", llm_output = "در سال 1399 بود.", status = "failed", reason = "words_changed")

    def test_existing_decimal_point_kept(self):
        self.check(source = "۳.۵ درصد افزایش یافت", llm_output = "۳.۵ درصد افزایش یافت.", status = "ok",
                   output = "۳.۵ درصد افزایش یافت.")

    def test_latin_marks_normalized(self):
        self.check(source = "سلام دنیا چطوری", llm_output = "سلام, دنیا چطوری?", status = "ok", output = "سلام، دنیا چطوری؟")

    def test_opening_and_closing_quotes(self):
        self.check(source = "در زمین های لاس کانیتاس", llm_output = "در زمین های «لاس کانیتاس».", status = "ok",
                   output = "در زمین های «لاس کانیتاس».")


class RecaseTest(unittest.TestCase):
    def test_case_and_marks_restored(self):
        result = transfer(source = "hôm nay ở hà nội trời mưa", llm_output = "Hôm nay, ở Hà Nội trời mưa.", lang = "vi")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["output"], "Hôm nay, ở Hà Nội trời mưa.")

    def test_diacritic_change_keeps_source(self):
        # LLM 改了声调（mưa → múa）：只采用大小写和标点，字母仍取原文本
        result = transfer(source = "hôm nay ở hà nội trời mưa", llm_output = "Hôm nay ở Hà Nội trời múa.", lang = "vi")
        self.assertEqual(result["status"], "aligned")
        self.assertEqual(result["output"], "Hôm nay ở Hà Nội trời mưa.")

    def test_lowercase_text_needs_restoring(self):
        self.assertTrue(needs_punctuation(text = "hôm nay, trời mưa.", lang = "vi"))
        self.assertFalse(needs_punctuation(text = "Hôm nay, trời mưa.", lang = "vi"))

    def test_malay_and_filipino_restore_case(self):
        result = transfer(
            source = "untuk makluman yang berhormat dewan rakyat",
            llm_output = "Untuk makluman Yang Berhormat, Dewan Rakyat.",
            lang = "ms",
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["output"], "Untuk makluman Yang Berhormat, Dewan Rakyat.")
        # 叠词加连字符等于把两个词连成一个：连字符不是可放回的标点，词和空格都取原文本
        result = transfer(source = "rumah rumah lama", llm_output = "Rumah-rumah lama.", lang = "ms")
        self.assertEqual(result["output"], "Rumah rumah lama.")
        result = transfer(source = "ang ating mga farmers sa batangas", llm_output = "Ang ating mga farmers sa Batangas.", lang = "fil")
        self.assertEqual(result["output"], "Ang ating mga farmers sa Batangas.")

    def test_score_capitalized_words(self):
        result = score(references = ["Hôm nay ở Hà Nội."], predictions = ["Hôm nay ở hà Nội."], lang = "vi")
        self.assertEqual(result["capitalized_words"]["recall"], round(2 / 3, 4))
        self.assertEqual(result["capitalized_words"]["precision"], 1.0)


class HelpersTest(unittest.TestCase):
    def test_needs_punctuation(self):
        self.assertTrue(needs_punctuation(text = "سلام دنیا", lang = "fa"))
        self.assertTrue(needs_punctuation(text = "۳.۵ درصد", lang = "fa"))
        self.assertFalse(needs_punctuation(text = "سلام، دنیا", lang = "fa"))

    def test_strip_marks(self):
        self.assertEqual(strip_marks(text = "«لاس کانیتاس» اتفاق می‌افتد.", lang = "fa"), "لاس کانیتاس اتفاق می‌افتد")

    def test_score_period_only(self):
        result = score(references = ["سلام، دنیا."], predictions = ["سلام دنیا."], lang = "fa")
        self.assertEqual(result["marks"]["period"]["recall"], 1.0)
        self.assertEqual(result["marks"]["comma"]["recall"], 0.0)
        self.assertEqual(result["final_accuracy"], 1.0)


class TaggerLabelsTest(unittest.TestCase):
    def test_apply_labels_uses_word_end_token(self):
        from pipeline.realspeech.punctuate_tagger import apply_labels

        text = "walang kagubatan sa canaan kaya un ito"
        # 子词结束位置上的标签；「kagubatan」切成两个子词，只有词尾那个（位置 16）算数
        labels = {6: "0|Cap", 10: ".|UPPER", 16: "0|lower", 19: "0|lower", 26: ",|Cap", 31: "0|lower", 34: "-|UPPER", 38: ".|lower"}
        self.assertEqual(apply_labels(text = text, labels = labels), "Walang kagubatan sa Canaan, kaya UN ito.")
        result = transfer(source = text, llm_output = apply_labels(text = text, labels = labels), lang = "fil")
        self.assertEqual(result["output"], "Walang kagubatan sa Canaan, kaya UN ito.")

    def test_chinese_marks_after_any_character(self):
        from pipeline.realspeech.punctuate_tagger import apply_labels

        text = "今天天气很好我们去公园吧"
        labels = {2: "0|lower", 6: ",|lower", 12: ".|lower"}
        self.assertEqual(apply_labels(text = text, labels = labels), "今天天气很好，我们去公园吧。")


class ChineseTransferTest(unittest.TestCase):
    def test_marks_placed_between_characters(self):
        result = transfer(source = "今天天气很好我们去公园吧", llm_output = "今天天气很好，我们去公园吧。", lang = "zh")
        self.assertEqual(result["output"], "今天天气很好，我们去公园吧。")

    def test_latin_words_keep_source_case_and_spaces(self):
        # 模型输出转成了小写：对齐不分大小写，字母取原文本的写法，英文词之间的空格保留
        result = transfer(source = "我在用 iPhone Pro 听歌", llm_output = "我在用 iphone pro 听歌。", lang = "zh")
        self.assertEqual(result["output"], "我在用 iPhone Pro 听歌。")

    def test_needs_punctuation(self):
        self.assertTrue(needs_punctuation(text = "请搜索我附近的超市", lang = "zh"))
        self.assertFalse(needs_punctuation(text = "好的，谢谢", lang = "zh"))


if __name__ == "__main__":
    unittest.main()
