"""样本格式 --prompt-format（任务 #71）的单元测试：python -m unittest pipeline.train.tests.test_prompt_format

只在 CPU 上运行，用仓库中阶段 1 模型目录的 processor（只需要分词器和特征提取的配置，不需要权重）。

- CptLabelTest：cpt 格式的样本是音频标记、目标文本、<|endoftext|>；计算损失的 token 恰好是目标文本的 token 加
  <|endoftext|>，从 <|audio_end|> 之后开始、连续，之前的部分与推理时只有音频标记的提示逐个相同；空目标样本只有
  <|endoftext|> 计算损失；填充位置（与 <|endoftext|> 同一个 id）不计损失。build_cpt_labels 对左填充同样成立，
  缺少音频结束标记、或最后一个有效 token 是其他 token 时报错。
- ChatUnchangedTest：默认的聊天格式与加入 cpt 之前的实现（本文件中的 reference_chat_batch，照抄原来的
  build_training_text 与 TrainCollator 的分词、标签步骤）逐个张量相同，预先计算元数据和拼接批次时也相同。
- ConfigTest：prompt_format 在续训核对的配置中（以前的配置按 chat 核对），不在训练计划的配置中。
"""

import os
import unittest
from pathlib import Path

import torch

from pipeline.train.data import (
    ASSISTANT_MASK_SPEC,
    IGNORE_INDEX,
    SAMPLING_RATE,
    TrainCollator,
    batch_metadata,
    build_cpt_labels,
    build_messages,
    build_prompt,
    pack_batch,
)
from pipeline.train.tests.fixtures import noise, tone

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parents[3] / "qwen3.5-0.8b-mllm"))


def make_item(sample_id, waveform, target, lang = "en"):
    return {
        "sample_id": sample_id,
        "lang": "zxx" if target == "" else lang,
        "audio_path": f"{sample_id}.flac",
        "duration_s": len(waveform) / SAMPLING_RATE,
        "waveform": waveform,
        "target": target,
        "empty": target == "",
        "mix": None,
    }


def mixed_items():
    """长短不一的语音、空目标样本和中文目标，覆盖右填充的各种长度。"""
    return [
        make_item(sample_id = "speech-long", waveform = tone(seconds = 6.0), target = "Hello world."),
        make_item(sample_id = "empty-short", waveform = noise(seconds = 1.5, seed = 0), target = ""),
        make_item(sample_id = "empty-long", waveform = noise(seconds = 9.0, seed = 1), target = ""),
        make_item(sample_id = "speech-short", waveform = tone(seconds = 2.0), target = "Ok"),
        make_item(
            sample_id = "speech-zh",
            waveform = tone(seconds = 3.3, frequency = 300.0),
            target = "你好，世界。",
            lang = "zh",
        ),
    ]


def reference_chat_batch(processor, items):
    """加入 cpt 之前 TrainCollator 的分词与标签步骤（聊天格式），用来核对默认格式没有改变。"""
    from mm_assistant_mask import build_assistant_labels

    texts = [
        processor.apply_chat_template(
            conversation = build_messages(audio_path = item["audio_path"]) + [
                {"role": "assistant", "content": [{"type": "text", "text": item["target"]}]},
            ],
            tokenize = False,
            add_generation_prompt = False,
            enable_thinking = False,
        )
        for item in items
    ]
    inputs = processor(
        text = texts,
        audio = [item["waveform"] for item in items],
        sampling_rate = SAMPLING_RATE,
        padding = True,
        return_tensors = "pt",
    )
    labels = build_assistant_labels(
        batch = inputs,
        tokenizer = processor.tokenizer,
        spec = ASSISTANT_MASK_SPEC,
    )
    return texts, inputs, labels


@unittest.skipUnless((MODEL_DIR / "processor_config.json").exists(), f"没有找到模型目录 {MODEL_DIR}")
class CptLabelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoProcessor

        cls.processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path = str(MODEL_DIR),
            trust_remote_code = True,
        )
        cls.tokenizer = cls.processor.tokenizer
        cls.eos = cls.tokenizer.convert_tokens_to_ids(tokens = "<|endoftext|>")
        cls.audio_end = cls.tokenizer.convert_tokens_to_ids(tokens = "<|audio_end|>")
        cls.collator = TrainCollator(processor = cls.processor, prompt_format = "cpt")

    def prompt_ids(self, item):
        return self.processor(
            text = [build_prompt(processor = self.processor, audio_path = item["audio_path"], prompt_format = "cpt")],
            audio = [item["waveform"]],
            sampling_rate = SAMPLING_RATE,
            return_tensors = "pt",
        )["input_ids"][0].tolist()

    def test_prompt_is_audio_markup(self):
        prompt = build_prompt(processor = self.processor, audio_path = "x.flac", prompt_format = "cpt")
        self.assertEqual(prompt, "<|audio_start|><|audio_pad|><|audio_end|>")
        # 与聊天格式中音频的写法相同。
        chat = build_prompt(processor = self.processor, audio_path = "x.flac")
        self.assertIn(prompt, chat)

    def test_labels(self):
        items = mixed_items()
        batch = self.collator(items)
        self.assertEqual(batch["n_empty"], 2)
        for row, item in enumerate(items):
            input_ids = batch["input_ids"][row]
            labels = batch["labels"][row]
            valid = batch["attention_mask"][row].bool()
            supervised = (labels != IGNORE_INDEX).nonzero().view(-1)
            expected = self.tokenizer(text = item["target"], add_special_tokens = False)["input_ids"] + [self.eos]
            self.assertEqual(input_ids[supervised].tolist(), expected, msg = item["sample_id"])
            self.assertEqual(labels[supervised].tolist(), expected, msg = item["sample_id"])
            first = int(supervised[0])
            self.assertEqual(supervised.tolist(), list(range(first, first + len(supervised))))
            # 计损失的部分从 <|audio_end|> 之后开始，到最后一个有效 token 为止。
            self.assertEqual(int(input_ids[first - 1]), self.audio_end, msg = item["sample_id"])
            self.assertEqual(int(supervised[-1]), int(valid.sum()) - 1, msg = item["sample_id"])
            # 之前的部分与推理时的提示（只有音频标记）展开后逐个相同，开头就是 <|audio_start|>。
            self.assertEqual(input_ids[:first].tolist(), self.prompt_ids(item = item), msg = item["sample_id"])
            text = self.tokenizer.decode(token_ids = input_ids[:first].tolist())
            self.assertTrue(text.startswith("<|audio_start|><|audio_pad|>"), msg = text[:40])
            self.assertNotIn("<|im_start|>", text)
            if item["target"] == "":
                self.assertEqual(input_ids[supervised].tolist(), [self.eos], msg = item["sample_id"])
        padding = batch["attention_mask"] == 0
        self.assertTrue(bool(padding.any()))
        self.assertTrue(bool((batch["labels"][padding] == IGNORE_INDEX).all()))
        # 填充用的也是 <|endoftext|>，只能靠 attention_mask 区分。
        self.assertTrue(bool((batch["input_ids"][padding] == self.eos).all()))

    def test_left_padding_and_errors(self):
        pad, start, audio, end = self.eos, 248070, 248076, self.audio_end
        input_ids = torch.tensor(data = [
            [pad, pad, start, audio, end, 11, 12, self.eos],
            [start, audio, audio, end, 21, 22, 23, self.eos],
            [pad, pad, pad, pad, start, audio, end, self.eos],
        ])
        attention_mask = (torch.arange(8)[None] >= torch.tensor(data = [2, 0, 4])[:, None]).long()
        labels = build_cpt_labels(
            input_ids = input_ids,
            attention_mask = attention_mask,
            audio_end_id = end,
            end_id = self.eos,
        )
        self.assertEqual(labels[0].tolist(), [IGNORE_INDEX] * 5 + [11, 12, self.eos])
        self.assertEqual(labels[1].tolist(), [IGNORE_INDEX] * 4 + [21, 22, 23, self.eos])
        self.assertEqual(labels[2].tolist(), [IGNORE_INDEX] * 7 + [self.eos])
        with self.assertRaises(ValueError):
            build_cpt_labels(
                input_ids = torch.tensor(data = [[start, audio, 11, self.eos]]),
                attention_mask = torch.ones(size = (1, 4), dtype = torch.long),
                audio_end_id = end,
                end_id = self.eos,
            )
        with self.assertRaises(ValueError):
            build_cpt_labels(
                input_ids = torch.tensor(data = [[start, audio, end, 11]]),
                attention_mask = torch.ones(size = (1, 4), dtype = torch.long),
                audio_end_id = end,
                end_id = self.eos,
            )

    def test_metadata_and_packing(self):
        """预先计算元数据时，计损失的位置与目标就是标签中的这些 token；拼接后逐条不变。"""
        items = mixed_items()
        collator = TrainCollator(processor = self.processor, audio_windows = (50, 800), prompt_format = "cpt")
        padded = collator(items)
        flat = padded["input_ids"].reshape(-1)
        self.assertTrue(torch.equal(flat[padded["loss_index"] + 1], padded["loss_targets"]))
        self.assertEqual(int(padded["loss_targets"].numel()), int((padded["labels"] != IGNORE_INDEX).sum()))
        self.assertEqual(padded["loss_targets"].tolist().count(self.eos), len(items))
        packed = pack_batch(batch = dict(padded))
        self.assertTrue(torch.equal(packed["loss_targets"], padded["loss_targets"]))
        self.assertTrue(torch.equal(packed["loss_rows"], padded["loss_rows"]))
        self.assertTrue(torch.equal(packed["input_ids"][0, packed["loss_index"] + 1], packed["loss_targets"]))


@unittest.skipUnless((MODEL_DIR / "processor_config.json").exists(), f"没有找到模型目录 {MODEL_DIR}")
class ChatUnchangedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoProcessor

        cls.processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path = str(MODEL_DIR),
            trust_remote_code = True,
        )

    def assert_same(self, want, got):
        self.assertEqual(set(want), set(got))
        for key, value in want.items():
            if isinstance(value, torch.Tensor):
                self.assertTrue(torch.equal(value, got[key]), msg = key)
            else:
                self.assertEqual(value, got[key], msg = key)

    def test_same_as_reference(self):
        items = mixed_items()
        texts, inputs, labels = reference_chat_batch(processor = self.processor, items = items)
        self.assertTrue(all(text.endswith("<|im_end|>\n") for text in texts))
        # 不给 prompt_format 与给出 chat 相同。
        collators = (TrainCollator(processor = self.processor), TrainCollator(processor = self.processor, prompt_format = "chat"))
        for collator in collators:
            batch = collator(items)
            self.assertTrue(torch.equal(batch["input_ids"], inputs["input_ids"]))
            self.assertTrue(torch.equal(batch["attention_mask"], inputs["attention_mask"]))
            self.assertTrue(torch.equal(batch["mm_token_type_ids"], inputs["mm_token_type_ids"]))
            self.assertTrue(torch.equal(batch["input_features"], inputs["input_features"]))
            self.assertTrue(torch.equal(batch["feature_attention_mask"], inputs["feature_attention_mask"]))
            self.assertTrue(torch.equal(batch["labels"], labels))

    def test_same_with_metadata_and_packing(self):
        items = mixed_items()
        _, inputs, labels = reference_chat_batch(processor = self.processor, items = items)
        reference = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
            "input_features": inputs["input_features"],
            "feature_attention_mask": inputs["feature_attention_mask"],
        }
        audio_token_id = self.processor.tokenizer.convert_tokens_to_ids(tokens = self.processor.audio_token)
        reference.update(batch_metadata(
            batch = reference,
            audio_token_id = audio_token_id,
            n_window = 50,
            n_window_infer = 800,
        ))
        batch = TrainCollator(processor = self.processor, audio_windows = (50, 800))(items)
        self.assert_same(want = reference, got = {key: batch[key] for key in reference})
        packed_reference = pack_batch(batch = {**reference, "mm_token_type_ids": inputs["mm_token_type_ids"]})
        packed = TrainCollator(processor = self.processor, audio_windows = (50, 800), pack = True)(items)
        self.assert_same(want = packed_reference, got = {key: packed[key] for key in packed_reference})


class ConfigTest(unittest.TestCase):
    ARGV = [
        "--model-dir", "/repo/qwen3.5-2b-mllm",
        "--train-manifest", "/data/train/manifest-v2b.jsonl",
        "--target", "original",
        "--output-dir", "/work/data/T71/cpt",
    ]

    def test_resume_key_with_chat_default(self):
        from pipeline.train.train import PLAN_KEYS, RESUME_KEYS, parse_args, plan_mismatches, resume_mismatches, run_config

        self.assertIn("prompt_format", RESUME_KEYS)
        self.assertNotIn("prompt_format", PLAN_KEYS)
        chat = run_config(args = parse_args(argv = self.ARGV))
        cpt = run_config(args = parse_args(argv = self.ARGV + ["--prompt-format", "cpt"]))
        self.assertEqual(chat["prompt_format"], "chat")
        # 以前的训练配置中没有这一项，按聊天格式核对。
        old = dict(chat)
        old.pop("prompt_format")
        self.assertEqual(resume_mismatches(saved = old, config = chat), [])
        self.assertEqual(resume_mismatches(saved = old, config = cpt), ["prompt_format"])
        self.assertEqual(resume_mismatches(saved = cpt, config = chat), ["prompt_format"])
        # 训练计划不变，按计划排列的分片索引照常可用。
        self.assertEqual(plan_mismatches(saved = {key: chat[key] for key in PLAN_KEYS}, config = cpt), [])


if __name__ == "__main__":
    unittest.main()
