"""pipeline.tn 的单元测试：清理、片段检查和命令行。

    python -m unittest pipeline.tn.tests.test_tn
"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from pipeline.tn import find_issues, normalize_entry
from pipeline.tn.run import main as run_main

ZWNJ = chr(0x200C)
ZWSP = chr(0x200B)
NBSP = chr(0x00A0)


def main(argv: list) -> int:
    """运行命令行入口，不在测试输出中打印。"""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return run_main(argv = argv)


def kinds(text: str, lang: str) -> list:
    return [it.kind for it in find_issues(text = text, lang = lang)]


class CleanTest(unittest.TestCase):
    def test_zh_to_simplified(self):
        result = normalize_entry(original = "那時正在下雨，而 Joe 回到家時長髮已濕透了。", lang = "zh")
        self.assertEqual(result.status, "unchanged")
        self.assertEqual(result.script, "那时正在下雨，而 Joe 回到家时长发已湿透了。")
        self.assertEqual(result.note, "转为简体")

    def test_zh_already_simplified(self):
        result = normalize_entry(original = "真相只有一个。", lang = "zh")
        self.assertEqual(result, ("真相只有一个。", "unchanged", ""))

    def test_whitespace_and_invisible(self):
        result = normalize_entry(original = f"  He held{ZWSP} a  law{NBSP}degree. ", lang = "en")
        self.assertEqual(result.script, "He held a law degree.")

    def test_persian_zwnj_kept(self):
        text = f"این قشنگ{ZWNJ}ترین هزینه{ZWNJ}ای است."
        result = normalize_entry(original = text, lang = "fa")
        self.assertEqual(result.status, "unchanged")
        self.assertEqual(result.script, text)

    def test_other_languages_not_converted(self):
        result = normalize_entry(original = "這是日本語の文です。", lang = "ja")
        self.assertEqual(result.script, "這是日本語の文です。")


class DetectTest(unittest.TestCase):
    def test_digits_in_all_scripts(self):
        for text, lang in [
            ("In 1886, it was founded.", "en"),
            ("เขามี ๓ คน", "th"),
            ("عاش ٣١ سنة", "ar"),
            ("ساعت ۸ اینجاست", "fa"),
            ("全角の３です", "ja"),
            ("面积为 5½ 平方公里", "zh"),
        ]:
            with self.subTest(text = text):
                self.assertIn("digit", kinds(text = text, lang = lang))
                self.assertEqual(normalize_entry(original = text, lang = lang).status, "dropped")

    def test_chinese_numeral_words_are_text(self):
        self.assertEqual(kinds(text = "二〇二六年有十项提名。", lang = "zh"), [])

    def test_symbols(self):
        for text, lang in [
            ("Rendimento de 10%.", "pt"),
            ("It costs $5.", "en"),
            ("Johnson & Johnson", "en"),
            ("气温 30℃", "zh"),
            ("a − b", "de"),
            ("write to me @ home", "en"),
        ]:
            with self.subTest(text = text):
                self.assertIn("symbol", kinds(text = text, lang = lang))

    def test_abbreviations(self):
        for text, lang in [
            ("Some fruits, e.g. apples.", "en"),
            ("The book by Rev.W.Awdry.", "en"),
            ("Dr. Smith is here.", "en"),
            ("«мальчик», «чай» и т. д.", "ru"),
            ("Das ist z. B. gut.", "de"),
            ("Er wohnt in der Hauptstr. am Rand.", "de"),
            ("Il y avait des pommes, des poires, etc.", "fr"),
            ("Buku, pena, dll.", "id"),
            ("ปี พ.ศ. นั้น", "th"),
            ("เขาไปกรุงเทพฯ", "th"),
            ("توفي سنة هـ", "ar"),
            ("طوله كـم واحد", "ar"),
            ("قال د. أحمد", "ar"),
        ]:
            with self.subTest(text = text):
                self.assertIn("abbrev", kinds(text = text, lang = lang))

    def test_not_abbreviations(self):
        for text, lang in [
            ("No. I don't think so.", "en"),
            ("J. R. R. Tolkien wrote it.", "en"),
            ("Sie ist die stärkste ihrer Art.", "de"),
            ("C'était un vol.", "fr"),
            ("Tôi thích ca.", "vi"),
            ("تبنّت الجزائر التّجارة الحرّة.", "ar"),
            ("Ang lahat ay masaya.", "fil"),
            ("The OSCE Minsk Group met.", "en"),
        ]:
            with self.subTest(text = text):
                self.assertEqual(kinds(text = text, lang = lang), [])

    def test_roman_numerals(self):
        self.assertEqual(kinds(text = "Ramses III. regierte.", lang = "de"), ["roman"])
        self.assertEqual(kinds(text = "obras dos séculos XX e XXI", lang = "pt"), ["roman", "roman"])
        self.assertEqual(kinds(text = "Đĩa CD này rất hay.", lang = "vi"), [])
        self.assertEqual(kinds(text = "I think I can.", lang = "en"), [])

    def test_url(self):
        self.assertIn("url", kinds(text = "Visit www.example.org today", lang = "en"))

    def test_note_lists_spans(self):
        result = normalize_entry(original = "Ende 1989 und 1990 vergab Ramses III. etwas.", lang = "de")
        self.assertEqual(result.status, "dropped")
        self.assertIsNone(result.script)
        self.assertEqual(result.note, "含数字「1989」「1990」；含罗马数字「III」")


class CliTest(unittest.TestCase):
    def entry(self, **fields):
        row = {
            "entry_id": "en-test-000001",
            "lang": "en",
            "source": "test",
            "source_license": "CC0-1.0",
            "source_ref": "test:1",
            "original": "He held a law degree.",
            "n_chars": 21,
            "length_bin": "short",
            "est_duration_s": 1.5,
        }
        row.update(fields)
        return row

    def write(self, folder: Path, rows: list) -> Path:
        path = folder / "entries.jsonl"
        with open(file = path, mode = "w", encoding = "utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(obj = row, ensure_ascii = False) + "\n")
        return path

    def test_run_writes_scripts_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            rows = [
                self.entry(),
                self.entry(entry_id = "en-test-000002", original = "In 1886, it was founded."),
                self.entry(entry_id = "zh-test-000003", lang = "zh", original = "你的狗在這。"),
            ]
            path = self.write(folder = folder, rows = rows)
            output = folder / "out" / "scripts.jsonl"
            code = main(argv = ["--input", str(path), "--output", str(output)])
            self.assertEqual(code, 0)
            out = [json.loads(s = line) for line in output.read_text(encoding = "utf-8").splitlines()]
            self.assertEqual([r["entry_id"] for r in out], [r["entry_id"] for r in rows])
            self.assertEqual([r["tn_status"] for r in out], ["unchanged", "dropped", "unchanged"])
            self.assertEqual(out[0]["script"], "He held a law degree.")
            self.assertIsNone(out[1]["script"])
            self.assertEqual(out[2]["script"], "你的狗在这。")
            self.assertEqual(out[0]["source_ref"], "test:1")
            summary = json.loads(s = (folder / "out" / "scripts.summary.json").read_text(encoding = "utf-8"))
            self.assertEqual(summary["langs"]["en"]["dropped"], 1)
            self.assertEqual(summary["langs"]["zh"]["ratio"]["unchanged"], 1.0)

    def test_invalid_input_exits(self):
        bad_rows = [
            [{k: v for k, v in self.entry().items() if k != "source_license"}],
            [self.entry(lang = "xx", entry_id = "xx-test-1")],
            [self.entry(n_chars = "21")],
            [self.entry(entry_id = "zh-test-000001")],
            [self.entry(), self.entry()],
            [self.entry(length_bin = "huge")],
        ]
        for rows in bad_rows:
            with self.subTest(rows = rows), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                path = self.write(folder = folder, rows = rows)
                output = folder / "scripts.jsonl"
                code = main(argv = ["--input", str(path), "--output", str(output)])
                self.assertEqual(code, 2)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
