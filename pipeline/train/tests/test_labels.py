"""空目标样本的训练标签：python -m unittest pipeline.train.tests.test_labels

用仓库中阶段 1 模型目录的 processor（只需要分词器和特征提取的配置，不需要权重）。模型目录可以用环境变量
MODEL_DIR 指定，默认是仓库根目录下的 qwen3.5-0.8b-mllm。
"""

import os
import unittest
from pathlib import Path

import numpy as np

from pipeline.train.data import IGNORE_INDEX, SAMPLING_RATE, TrainCollator, build_prompt
from pipeline.train.tests.fixtures import noise, tone

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parents[3] / "qwen3.5-0.8b-mllm"))


@unittest.skipUnless((MODEL_DIR / "processor_config.json").exists(), f"没有找到模型目录 {MODEL_DIR}")
class EmptyTargetLabelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoProcessor

        cls.processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path = str(MODEL_DIR),
            trust_remote_code = True,
        )
        cls.tokenizer = cls.processor.tokenizer
        cls.im_end = cls.tokenizer.convert_tokens_to_ids(tokens = "<|im_end|>")
        cls.collator = TrainCollator(processor = cls.processor)

    def item(self, sample_id, waveform, target):
        return {
            "sample_id": sample_id,
            "lang": "zxx" if target == "" else "en",
            "audio_path": f"{sample_id}.flac",
            "duration_s": len(waveform) / SAMPLING_RATE,
            "waveform": waveform,
            "target": target,
            "empty": target == "",
            "mix": None,
        }

    def prompt_ids(self, item):
        return self.processor(
            text = [build_prompt(processor = self.processor, audio_path = item["audio_path"])],
            audio = [item["waveform"]],
            sampling_rate = SAMPLING_RATE,
            return_tensors = "pt",
        )["input_ids"][0].tolist()

    def test_mixed_batch(self):
        items = [
            self.item(sample_id = "speech-long", waveform = tone(seconds = 6.0), target = "Hello world."),
            self.item(sample_id = "empty-short", waveform = noise(seconds = 1.5, seed = 0), target = ""),
            self.item(sample_id = "empty-long", waveform = noise(seconds = 9.0, seed = 1), target = ""),
            self.item(sample_id = "speech-short", waveform = tone(seconds = 2.0), target = "Ok"),
        ]
        batch = self.collator(items)
        self.assertEqual(batch["n_empty"], 2)
        for row, item in enumerate(items):
            input_ids = batch["input_ids"][row]
            labels = batch["labels"][row]
            supervised = (labels != IGNORE_INDEX).nonzero().view(-1)
            supervised_ids = input_ids[supervised].tolist()
            if item["target"] == "":
                # 只有 <|im_end|> 计算损失：一个 token，位于空思考块之后。
                self.assertEqual(supervised_ids, [self.im_end], msg = item["sample_id"])
            else:
                expected = self.tokenizer(text = item["target"], add_special_tokens = False)["input_ids"] + [self.im_end]
                self.assertEqual(supervised_ids, expected, msg = item["sample_id"])
            # 计算损失的位置连续，之前的部分与推理时的提示逐个相同。
            first = int(supervised[0])
            self.assertEqual(supervised.tolist(), list(range(first, first + len(supervised))))
            self.assertEqual(input_ids[:first].tolist(), self.prompt_ids(item = item), msg = item["sample_id"])
            self.assertTrue(
                self.tokenizer.decode(token_ids = input_ids[:first].tolist()).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            )
        # 填充位置一律不计损失。
        padding = batch["attention_mask"] == 0
        self.assertTrue(bool(np.all(a = (batch["labels"][padding] == IGNORE_INDEX).numpy())))


if __name__ == "__main__":
    unittest.main()
