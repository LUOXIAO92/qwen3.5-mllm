"""分词器修正与加载检查（#77）：python -m unittest pipeline.tests.test_tokenizer_fix

用仓库中阶段 1 模型目录的分词器和 processor 文件（不需要权重），模型目录可以用环境变量 MODEL_DIR 指定。
环境变量 ORIGINAL_DIR 指向原始 Qwen3.5 目录（默认是仓库根目录下的 weights/Qwen3.5-0.8B），各语言例句的参照
token 编号由 tokenizers 库按其中的 tokenizer.json 分词得到；找不到原始目录时跳过需要它的测试。

修正前的目录在临时目录中复现：tokenizer_config.json 的分词器类改回 Qwen2Tokenizer，tokenizer.json 的预分词器和
解码器换成 transformers 5.16.1 的 Qwen2Tokenizer 保存的写法。
"""

import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from pipeline.tokenizer_fix import (
    DECODER,
    PRE_TOKENIZER,
    PROBE_IDS,
    PROBE_TEXT,
    TOKENIZER_CLASS,
    check_tokenizer,
    fix_model_dir,
)

REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.environ.get(key = "MODEL_DIR", default = REPO / "qwen3.5-0.8b-mllm"))
ORIGINAL_DIR = Path(os.environ.get(key = "ORIGINAL_DIR", default = REPO / "weights" / "Qwen3.5-0.8B"))
HAS_MODEL = (MODEL_DIR / "processor_config.json").exists()
HAS_ORIGINAL = (ORIGINAL_DIR / "tokenizer.json").exists()
# 各语言的例句：泰语和带元音符号的阿拉伯语、波斯语会受预分词规则影响，其余语言用来确认没有别的变化
SAMPLES = {
    "th": "สวัสดีครับ วันนี้อากาศดีมาก ภาษาไทยมีวรรณยุกต์ห้าเสียง",
    "ar": "اَلسَّلَامُ عَلَيْكُمْ وَرَحْمَةُ اللهِ وَبَرَكَاتُهُ",
    "fa": "زبان فارسی یکی از زبان‌های هندواروپایی است. کِتابِ خوبی خواندَم و دانش‌آموزان شاد شُدند.",
    "vi": "Tiếng Việt có sáu thanh điệu: ngang, huyền, sắc, hỏi, ngã, nặng.",
    "ja": "日本語の文章を正しく分割できるか確かめます。カタカナもひらがなも。",
    "zh": "今天天气很好，我们去公园散步吧。",
    "en": "The quick brown fox jumps over the lazy dog, isn't it?",
}
# 受影响的语言：修正前的分词与原始结果不同
AFFECTED = ("th", "ar")


def reference_tokenizer():
    from tokenizers import Tokenizer

    return Tokenizer.from_file(path = str(ORIGINAL_DIR / "tokenizer.json"))


def reference_ids(text):
    return reference_tokenizer().encode(sequence = text, add_special_tokens = False).ids


def copy_model_dir(target):
    """复制模型目录中除权重以外的文件。"""
    shutil.copytree(
        src = MODEL_DIR,
        dst = target,
        ignore = shutil.ignore_patterns("*.safetensors", "*.before-77", "__pycache__"),
    )
    return Path(target)


def break_model_dir(model_dir):
    """把模型目录改成修正前的样子（transformers 5.16.1 按 Qwen2Tokenizer 保存的分词器文件）。"""
    from tokenizers import Tokenizer
    from transformers.models.qwen2.tokenization_qwen2 import PRETOKENIZE_REGEX as QWEN2_REGEX

    config_path = model_dir / "tokenizer_config.json"
    config = json.loads(s = config_path.read_text(encoding = "utf-8"))
    config["tokenizer_class"] = "Qwen2Tokenizer"
    config_path.write_text(data = json.dumps(obj = config, indent = 2, ensure_ascii = False) + "\n", encoding = "utf-8")
    tokenizer_path = model_dir / "tokenizer.json"
    tokenizer_json = json.loads(s = tokenizer_path.read_text(encoding = "utf-8"))
    tokenizer_json["pre_tokenizer"] = {
        "type": "Sequence",
        "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": QWEN2_REGEX}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
        ],
    }
    tokenizer_json["decoder"] = {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True}
    broken = Tokenizer.from_str(json = json.dumps(obj = tokenizer_json, ensure_ascii = False))
    broken.save(path = str(tokenizer_path), pretty = True)


def load_all(model_dir):
    """各加载入口得到的分词器：AutoTokenizer（加与不加 trust_remote_code）和 AutoProcessor 中的 tokenizer。"""
    from transformers import AutoProcessor, AutoTokenizer

    return {
        "AutoTokenizer": AutoTokenizer.from_pretrained(pretrained_model_name_or_path = str(model_dir)),
        "AutoTokenizer(trust_remote_code)": AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path = str(model_dir),
            trust_remote_code = True,
        ),
        "AutoProcessor": AutoProcessor.from_pretrained(
            pretrained_model_name_or_path = str(model_dir),
            trust_remote_code = True,
        ).tokenizer,
    }


def encode(tokenizer, text):
    return tokenizer(text = text, add_special_tokens = False)["input_ids"]


@unittest.skipUnless(HAS_ORIGINAL, f"没有找到原始 Qwen3.5 目录 {ORIGINAL_DIR}")
class OriginalConstantsTest(unittest.TestCase):
    """本模块中的常量取自原始 tokenizer.json。"""

    def test_constants_match_original_file(self):
        original = json.loads(s = (ORIGINAL_DIR / "tokenizer.json").read_text(encoding = "utf-8"))
        self.assertEqual(original["pre_tokenizer"], PRE_TOKENIZER)
        self.assertEqual(original["decoder"], DECODER)
        self.assertEqual(reference_ids(text = PROBE_TEXT), PROBE_IDS)


@unittest.skipUnless(HAS_MODEL, f"没有找到模型目录 {MODEL_DIR}")
class RepoModelDirTest(unittest.TestCase):
    """仓库中的模型目录已经修正，再运行修正不改动任何文件。"""

    def test_files_are_fixed(self):
        config = json.loads(s = (MODEL_DIR / "tokenizer_config.json").read_text(encoding = "utf-8"))
        tokenizer_json = json.loads(s = (MODEL_DIR / "tokenizer.json").read_text(encoding = "utf-8"))
        self.assertEqual(config["tokenizer_class"], TOKENIZER_CLASS)
        self.assertEqual(tokenizer_json["pre_tokenizer"], PRE_TOKENIZER)
        self.assertEqual(tokenizer_json["decoder"], DECODER)

    def test_fix_is_noop(self):
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = copy_model_dir(target = Path(temporary) / "model")
            before = {name: (model_dir / name).read_bytes() for name in ("tokenizer.json", "tokenizer_config.json")}
            self.assertEqual(fix_model_dir(model_dir = model_dir, backup_suffix = ".before-77"), [])
            for name, content in before.items():
                self.assertEqual((model_dir / name).read_bytes(), content)
                self.assertFalse((model_dir / f"{name}.before-77").exists())


@unittest.skipUnless(HAS_MODEL and HAS_ORIGINAL, f"没有找到模型目录 {MODEL_DIR} 或原始目录 {ORIGINAL_DIR}")
class FixAndLoadTest(unittest.TestCase):
    """修正前的目录被检查拦下；修正后各加载入口的分词与原始 tokenizer.json 相同。"""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.broken_dir = copy_model_dir(target = Path(cls.temporary.name) / "broken")
        break_model_dir(model_dir = cls.broken_dir)
        cls.fixed_dir = copy_model_dir(target = Path(cls.temporary.name) / "fixed")
        break_model_dir(model_dir = cls.fixed_dir)
        cls.broken_files = {name: (cls.fixed_dir / name).read_bytes() for name in ("tokenizer.json", "tokenizer_config.json")}
        cls.changed = fix_model_dir(model_dir = cls.fixed_dir, backup_suffix = ".before-77")
        cls.reference = {lang: reference_ids(text = text) for lang, text in SAMPLES.items()}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_broken_dir_splits_marks_and_fails_check(self):
        for entry, tokenizer in load_all(model_dir = self.broken_dir).items():
            with self.subTest(entry = entry):
                for lang in AFFECTED:
                    self.assertNotEqual(encode(tokenizer = tokenizer, text = SAMPLES[lang]), self.reference[lang])
                with self.assertRaisesRegex(RuntimeError, "#77") as caught:
                    check_tokenizer(tokenizer = tokenizer, source = self.broken_dir)
                self.assertIn("python -m pipeline.tokenizer_fix", str(caught.exception))

    def test_fix_changes_both_files_and_keeps_backups(self):
        self.assertEqual(sorted(self.changed), ["tokenizer.json", "tokenizer_config.json"])
        for name, content in self.broken_files.items():
            self.assertEqual((self.fixed_dir / f"{name}.before-77").read_bytes(), content)
        config = json.loads(s = (self.fixed_dir / "tokenizer_config.json").read_text(encoding = "utf-8"))
        self.assertEqual(config["tokenizer_class"], TOKENIZER_CLASS)
        tokenizer_json = json.loads(s = (self.fixed_dir / "tokenizer.json").read_text(encoding = "utf-8"))
        self.assertEqual(tokenizer_json["pre_tokenizer"], PRE_TOKENIZER)
        self.assertEqual(tokenizer_json["decoder"], DECODER)
        # 再运行一次不改动，已有的备份也不被覆盖
        self.assertEqual(fix_model_dir(model_dir = self.fixed_dir, backup_suffix = ".before-77"), [])
        for name, content in self.broken_files.items():
            self.assertEqual((self.fixed_dir / f"{name}.before-77").read_bytes(), content)

    def test_fixed_loading_matches_original(self):
        for entry, tokenizer in load_all(model_dir = self.fixed_dir).items():
            with self.subTest(entry = entry):
                self.assertEqual(type(tokenizer).__name__, TOKENIZER_CLASS)
                self.assertIs(check_tokenizer(tokenizer = tokenizer), tokenizer)
                for lang, text in SAMPLES.items():
                    ids = encode(tokenizer = tokenizer, text = text)
                    self.assertEqual(ids, self.reference[lang], msg = lang)
                    self.assertEqual(tokenizer.decode(token_ids = ids), text, msg = lang)

    def test_fixed_file_matches_original_with_tokenizers_library(self):
        from tokenizers import Tokenizer

        fixed = Tokenizer.from_file(path = str(self.fixed_dir / "tokenizer.json"))
        for lang, text in SAMPLES.items():
            self.assertEqual(fixed.encode(sequence = text, add_special_tokens = False).ids, self.reference[lang], msg = lang)

    def test_special_tokens_unchanged(self):
        before = load_all(model_dir = self.broken_dir)["AutoProcessor"]
        after = load_all(model_dir = self.fixed_dir)["AutoProcessor"]
        self.assertEqual(len(before), len(after))
        self.assertEqual(before.all_special_ids, after.all_special_ids)
        tokens = ["<|audio_start|>", "<|audio_pad|>", "<|audio_end|>", "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>"]
        self.assertEqual(before.convert_tokens_to_ids(tokens = tokens), after.convert_tokens_to_ids(tokens = tokens))
        prompt = "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>Transcribe the audio.<|im_end|>\n"
        self.assertEqual(encode(tokenizer = before, text = prompt), encode(tokenizer = after, text = prompt))


@unittest.skipUnless(HAS_MODEL, f"没有找到模型目录 {MODEL_DIR}")
class CheckTokenizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoProcessor

        cls.processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path = str(MODEL_DIR), trust_remote_code = True)

    def test_accepts_processor(self):
        self.assertIs(check_tokenizer(tokenizer = self.processor), self.processor.tokenizer)

    def test_reports_wrong_split(self):
        class SplitMarks:
            """把检查文字分成 25 个 token 的分词器（修正前的切法）。"""

            def encode(self, text, add_special_tokens = True):
                return list(range(25))

        with self.assertRaisesRegex(RuntimeError, "25 个 token.*11 个"):
            check_tokenizer(tokenizer = SplitMarks(), source = "某目录")

    def test_fast(self):
        start = time.perf_counter()
        for _ in range(100):
            check_tokenizer(tokenizer = self.processor)
        self.assertLess((time.perf_counter() - start) / 100, 0.005)


if __name__ == "__main__":
    unittest.main()
