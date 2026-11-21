"""指令遵循探针（任务 #72）中不需要模型的部分：python -m unittest pipeline.eval.tests.test_instruct_probe

拼接音频的各段起止时间、拼接组的挑选、chrF、字符错误率、时间段输出的分类、按 fleurs_id 匹配参考译文、
请求的构造和汇总。
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.eval.instruct_probe import (
    CPT_INSTRUCTION,
    build_requests,
    char_error_rate,
    chrf_score,
    chrf_statistics,
    concatenate,
    corpus_chrf,
    format_markdown,
    parallel_references,
    pick_concat_parts,
    request_key,
    score_item,
    segment_bounds,
    segment_outcome,
    select_rows,
    summarize,
)


# 各语言的示例句子差别足够大，译文与原文的字符错误率远高于逐字的阈值。
TEMPLATES = {
    "en": "the quick brown fox jumps {number}",
    "fr": "le renard saute par dessus le chien {number}",
    "zh": "敏捷的棕色狐狸跳过了懒狗{number}",
}


def manifest_rows(lang, count, texts = None, durations = None):
    rows = []
    for number in range(count):
        rows.append({
            "sample_id": f"{lang}-test-{number:03d}",
            "audio_path": f"{lang}/test/{number:03d}.flac",
            "lang": lang,
            "fleurs_id": str(100 + number),
            "raw_transcription": texts[number] if texts else TEMPLATES[lang].format(number = number),
            "duration_s": durations[number] if durations else 5.0 + number % 7,
        })
    return rows


def frames_from_duration(row):
    return int(round(row["duration_s"] * 16000))


class SegmentTest(unittest.TestCase):
    def test_bounds(self):
        bounds = segment_bounds(lengths = [16000, 8000, 4000], gap = 8000)
        self.assertEqual(bounds, [(0.0, 1.0), (1.5, 2.0), (2.5, 2.75)])
        self.assertEqual(segment_bounds(lengths = [24000], gap = 8000), [(0.0, 1.5)])

    def test_concatenate(self):
        waves = [np.full(shape = (16000,), fill_value = 0.5), np.full(shape = (8000,), fill_value = -0.25)]
        wave, bounds = concatenate(waveforms = waves, gap = 8000)
        self.assertEqual(len(wave), 16000 + 8000 + 8000)
        self.assertEqual(wave.dtype, np.float32)
        for (start, end), source in zip(bounds, waves):
            piece = wave[int(start * 16000):int(end * 16000)]
            self.assertTrue(np.array_equal(piece, source.astype(np.float32)))
        self.assertTrue(np.all(wave[16000:24000] == 0))


class PickConcatTest(unittest.TestCase):
    def test_groups(self):
        rows = manifest_rows(lang = "en", count = 40, durations = [4.0 + (number % 9) * 3 for number in range(40)])
        groups, reordered = [
            pick_concat_parts(
                rows = ordered,
                count = 50,
                seed = 3,
                lang = "en",
                gap_s = 0.5,
                max_total_s = 30.0,
            )
            for ordered in (rows, list(reversed(rows)))
        ]
        # 结果与清单的行序无关。
        self.assertEqual(groups, reordered)
        sizes = set()
        for parts, target in groups:
            sizes.add(len(parts))
            self.assertIn(len(parts), (2, 3))
            self.assertLess(target, len(parts))
            self.assertLessEqual(sum(row["duration_s"] for row in parts) + 0.5 * (len(parts) - 1), 30.0)
            self.assertEqual(len({row["fleurs_id"] for row in parts}), len(parts))
        self.assertEqual(sizes, {2, 3})

    def test_too_long(self):
        rows = manifest_rows(lang = "en", count = 3, durations = [20.0, 25.0, 40.0])
        with self.assertRaises(ValueError):
            pick_concat_parts(
                rows = rows,
                count = 1,
                seed = 0,
                lang = "en",
                gap_s = 0.5,
                max_total_s = 30.0,
            )


class ScoreTest(unittest.TestCase):
    def test_chrf(self):
        self.assertAlmostEqual(chrf_score(stats = chrf_statistics(hypothesis = "abc", reference = "abc")), 100.0)
        # 1 元：精确率、召回率 2/3；2 元：1/2；3 元：0；4–6 元两边都没有，不计。平均 7/18，F2 也是 7/18。
        self.assertAlmostEqual(chrf_score(stats = chrf_statistics(hypothesis = "abc", reference = "abd")), 700 / 18)
        self.assertEqual(chrf_score(stats = chrf_statistics(hypothesis = "", reference = "abc")), 0.0)
        # 不计空白。
        self.assertAlmostEqual(chrf_score(stats = chrf_statistics(hypothesis = "a b c", reference = "abc")), 100.0)
        # 召回率权重更高：漏字比多字扣分多。
        short = chrf_score(stats = chrf_statistics(hypothesis = "hello", reference = "hello world"))
        long = chrf_score(stats = chrf_statistics(hypothesis = "hello world again", reference = "hello world"))
        self.assertLess(short, long)
        # 语料级：统计量相加后再算。
        pairs = [("abc", "abd"), ("hello world", "hello world")]
        total = [
            [a + b for a, b in zip(x, y)]
            for x, y in zip(
                chrf_statistics(hypothesis = "abc", reference = "abd"),
                chrf_statistics(hypothesis = "hello world", reference = "hello world"),
            )
        ]
        self.assertAlmostEqual(corpus_chrf(pairs = pairs), chrf_score(stats = total))
        self.assertIsNone(corpus_chrf(pairs = []))

    def test_char_error_rate(self):
        self.assertEqual(char_error_rate(reference = "Hello, World!", hypothesis = "hello world", lang = "en"), 0.0)
        self.assertAlmostEqual(char_error_rate(reference = "abcd", hypothesis = "abxd", lang = "en"), 0.25)
        self.assertEqual(char_error_rate(reference = "你好，世界。", hypothesis = "你好世界", lang = "zh"), 0.0)
        self.assertIsNone(char_error_rate(reference = "", hypothesis = "abc", lang = "en"))

    def test_segment_outcome(self):
        self.assertEqual(segment_outcome(segment_rate = 0.05, whole_rate = 0.6, threshold = 0.3), "segment")
        self.assertEqual(segment_outcome(segment_rate = 1.2, whole_rate = 0.05, threshold = 0.3), "whole")
        # 只差一小段时，两个都低于阈值，取更接近的一个。
        self.assertEqual(segment_outcome(segment_rate = 0.0, whole_rate = 0.2, threshold = 0.3), "segment")
        self.assertEqual(segment_outcome(segment_rate = 0.25, whole_rate = 0.1, threshold = 0.3), "whole")
        self.assertEqual(segment_outcome(segment_rate = 0.9, whole_rate = 0.8, threshold = 0.3), "other")


class RequestTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.manifests = {
            "en": manifest_rows(lang = "en", count = 12),
            "fr": manifest_rows(lang = "fr", count = 12),
            "zh": manifest_rows(lang = "zh", count = 10),
        }
        # 英文清单中同一个 fleurs_id 有两位说话人的录音。
        self.manifests["en"].append(dict(self.manifests["en"][0], sample_id = "en-test-999"))
        for lang, rows in self.manifests.items():
            (root / lang).mkdir()
            with open(file = root / lang / "test.jsonl", mode = "w", encoding = "utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(obj = row, ensure_ascii = False) + "\n")
        self.root = root

    def tearDown(self):
        self.directory.cleanup()

    def build(self, tasks, prompt_formats = ("chat",), cascade = False, langs = ("en", "fr", "zh")):
        references = parallel_references(manifest_dir = self.root, target_langs = ["en", "zh"])
        return build_requests(
            manifests = {lang: self.manifests[lang] for lang in langs},
            tasks = tasks,
            prompt_formats = prompt_formats,
            per_lang = 5,
            seed = 0,
            cascade = cascade,
            references = references,
            audio_frames = frames_from_duration,
        )

    def test_parallel_references(self):
        references = parallel_references(manifest_dir = self.root, target_langs = ["en", "zh", "ja"])
        self.assertEqual(references["en"]["100"], TEMPLATES["en"].format(number = 0))
        self.assertEqual(references["zh"]["105"], TEMPLATES["zh"].format(number = 5))
        self.assertNotIn("110", references["zh"])
        self.assertEqual(references["ja"], {})

    def test_translate_matches_fleurs_id(self):
        requests = self.build(tasks = ["translate"])
        ids = {(request["lang"], request["instruction_id"]) for request in requests}
        # 源语言与目标语言相同时不问。
        self.assertNotIn(("en", "translate_en"), ids)
        self.assertNotIn(("zh", "translate_zh"), ids)
        self.assertIn(("fr", "translate_en"), ids)
        self.assertIn(("en", "translate_zh"), ids)
        for request in requests:
            number = int(request["fleurs_ids"][0]) - 100
            expected = TEMPLATES[request["target_lang"]].format(number = number)
            if request["target_lang"] == "zh" and number >= 10:
                expected = None
            self.assertEqual(request["translation_reference"], expected)

    def test_same_items_across_tasks(self):
        requests = self.build(tasks = ["transcribe", "what_heard", "translate"])
        for lang in ("en", "fr"):
            items = {}
            for request in requests:
                if request["lang"] == lang:
                    items.setdefault(request["instruction_id"], set()).add(request["item_id"])
            self.assertEqual(len({frozenset(value) for value in items.values()}), 1)
            self.assertEqual(len(next(iter(items.values()))), 5)
        self.assertEqual(len({request_key(request = request) for request in requests}), len(requests))
        picked = [
            [
                row["sample_id"]
                for row in select_rows(
                    rows = ordered,
                    count = 5,
                    seed = 0,
                    lang = "fr",
                )
            ]
            for ordered in (self.manifests["fr"], list(reversed(self.manifests["fr"])))
        ]
        self.assertEqual(picked[0], picked[1])

    def test_timestamps_and_formats(self):
        requests = self.build(tasks = ["transcribe", "timestamps"], prompt_formats = ("chat", "cpt"), langs = ("fr",))
        cpt = [request for request in requests if request["prompt_format"] == "cpt"]
        self.assertEqual({request["instruction_id"] for request in cpt}, {CPT_INSTRUCTION["id"]})
        self.assertTrue(all(request["instruction"] is None and request["task"] == "transcribe" for request in cpt))
        segments = [request for request in requests if request["task"] == "timestamps"]
        self.assertEqual(len(segments), 10)
        for request in segments:
            start, end = request["segment_bounds"][request["segment_index"]]
            self.assertIn(f"{start:.1f}", request["instruction"])
            self.assertIn(f"{end:.1f}", request["instruction"])
            self.assertAlmostEqual(request["duration_s"], request["segment_bounds"][-1][1])
            self.assertIn(request["reference"], request["whole_reference"])
        # 级联参照没有音频，不做时间段任务，也没有 cpt。
        cascade = self.build(tasks = ["transcribe", "timestamps"], cascade = True, langs = ("fr",))
        self.assertTrue(all(request["task"] == "transcribe" and request["mode"] == "cascade" for request in cascade))

    def test_summary(self):
        requests = self.build(tasks = ["transcribe", "what_heard", "translate", "timestamps"], langs = ("fr",))
        items = []
        for request in requests:
            item = dict(request, n_new_tokens = 1)
            # translate_en 照做、translate_zh 只转写原文；segment_en 输出整段、segment_zh 只输出问的这一段。
            if request["instruction_id"] == "translate_en":
                item["output"] = request["translation_reference"]
            elif request["instruction_id"] == "segment_en":
                item["output"] = request["whole_reference"]
            elif request["instruction_id"] == "heard_zh":
                item["output"] = "我听到一个人在说话。"
            else:
                item["output"] = request["reference"]
            items.append(score_item(item = item, threshold = 0.3))
        summary = summarize(items = items)
        self.assertEqual(summary["transcribe"]["transcribe"]["per_lang"]["fr"]["rate_basic"], 0.0)
        self.assertEqual(summary["transcribe"]["transcribe"]["ALL"]["verbatim_share"], 1.0)
        self.assertEqual(summary["what_heard"]["heard_en"]["per_lang"]["fr"]["verbatim_share"], 1.0)
        self.assertEqual(summary["what_heard"]["heard_zh"]["per_lang"]["fr"]["descriptive_share"], 1.0)
        self.assertAlmostEqual(summary["translate"]["translate_en"]["per_lang"]["fr"]["chrf"], 100.0)
        self.assertEqual(summary["translate"]["translate_en"]["per_lang"]["fr"]["verbatim_share"], 0.0)
        self.assertEqual(summary["translate"]["translate_zh"]["per_lang"]["fr"]["verbatim_share"], 1.0)
        self.assertEqual(summary["timestamps"]["segment_en"]["per_lang"]["fr"]["whole_share"], 1.0)
        self.assertEqual(summary["timestamps"]["segment_zh"]["per_lang"]["fr"]["segment_share"], 1.0)
        table = format_markdown(summary = summary)
        for task in ("transcribe", "what_heard", "translate", "timestamps"):
            self.assertIn(f"## {task}", table)
        self.assertIn("| fr |", table)
        self.assertIn("| ALL |", table)


if __name__ == "__main__":
    unittest.main()
