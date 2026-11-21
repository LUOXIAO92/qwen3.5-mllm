"""拼接批次与 torch 交叉熵（任务 #66）的单元测试：python -m unittest pipeline.train.tests.test_packing

- PackBatchTest（CPU）：data.pack_batch 拼成的一行与填充的批次逐条样本对应：token、标签、计损失的位置和目标相同，
  位置编号每条从 0 开始，样本边界正确。聊天格式和纯续写格式（--prompt-format cpt）的批次各检查一次。
- TorchLossTest（CPU）：--loss-impl torch 与先投影再 F.cross_entropy 的参考值一致，隐状态和输出层权重的梯度都对。
- PackedTinyModelTest（需要 CUDA，以及 flash-attn、causal-conv1d、fla 的内核，缺一时跳过）：缩小的随机初始化
  模型上，拼接与填充的批次的损失、projector 与语言模型的梯度在 bfloat16 误差以内一致；改动一条样本的音频，
  拼接后的其他样本的隐状态逐位不变。真实模型和真实批次上的同样检查见 pipeline/train/check_packing.py。
"""

import copy
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from pipeline.train.model import linear_cross_entropy
from pipeline.train.tests.test_offload import CUDA, MODEL_DIR, tiny_batch, tiny_model

HAS_MODEL = (MODEL_DIR / "config.json").exists()


def kernels_available():
    try:
        import causal_conv1d  # noqa: F401
        import fla  # noqa: F401
        import flash_attn  # noqa: F401
    except Exception:
        return False
    return True


@unittest.skipUnless(HAS_MODEL, "需要阶段 1 模型目录中的代码与配置")
class PackBatchTest(unittest.TestCase):
    def test_rows_match_padded_batch(self):
        self.check_rows_match(padded = tiny_batch(audio_windows = (50, 800)))

    def test_rows_match_padded_batch_cpt(self):
        """纯续写格式（--prompt-format cpt，任务 #71）的批次同样逐条对应。"""
        self.check_rows_match(padded = tiny_batch(audio_windows = (50, 800), prompt_format = "cpt"))

    def check_rows_match(self, padded):
        from pipeline.train.data import IGNORE_INDEX, pack_batch

        packed = pack_batch(batch = copy.copy(padded))
        mask = padded["attention_mask"].bool()
        lengths = mask.sum(dim = -1).tolist()
        starts = [0]
        for length in lengths:
            starts.append(starts[-1] + length)
        self.assertEqual(packed["cu_seq_lens_q"].tolist(), starts)
        self.assertEqual(packed["cu_seq_lens_q"].dtype, torch.int32)
        self.assertEqual(packed["max_length_q"], max(lengths))
        self.assertEqual(tuple(packed["input_ids"].shape), (1, starts[-1]))
        self.assertEqual(tuple(packed["position_ids"].shape), (3, 1, starts[-1]))
        for row, (start, end) in enumerate(zip(starts[:-1], starts[1:])):
            self.assertTrue(torch.equal(packed["input_ids"][0, start:end], padded["input_ids"][row][mask[row]]))
            self.assertTrue(torch.equal(packed["audio_mask"][0, start:end], padded["audio_mask"][row][mask[row]]))
            self.assertEqual(packed["position_ids"][0, 0, start:end].tolist(), list(range(end - start)))
            self.assertTrue(bool((packed["seq_idx"][0, start:end] == row).all()))
            # 每条样本的第一个 token 不被预测（标签为 -100），其余标签与填充的批次相同。
            self.assertEqual(int(packed["labels"][0, start]), IGNORE_INDEX)
            self.assertTrue(torch.equal(packed["labels"][0, start + 1:end], padded["labels"][row][mask[row]][1:]))
        # 计损失的 token 与填充的批次一一对应、顺序相同，所在样本相同。
        self.assertTrue(torch.equal(packed["loss_targets"], padded["loss_targets"]))
        self.assertTrue(torch.equal(packed["loss_rows"], padded["loss_rows"]))
        self.assertTrue(torch.equal(packed["input_ids"][0, packed["loss_index"] + 1], packed["loss_targets"]))
        # 训练循环按 labels[:, 1:] 数计损失的 token，两种批次数出的个数相同。
        self.assertEqual(
            int((packed["labels"][:, 1:] != IGNORE_INDEX).sum()),
            int((padded["labels"][:, 1:] != IGNORE_INDEX).sum()),
        )
        # 音频塔的输入与填充无关，不变。
        self.assertTrue(torch.equal(packed["audio_padded_feature"], padded["audio_padded_feature"]))


class TorchLossTest(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(0)
        self.hidden = torch.randn(size = (50, 16), generator = generator, dtype = torch.float64)
        self.weight = torch.randn(size = (37, 16), generator = generator, dtype = torch.float64)
        self.targets = torch.randint(low = 0, high = 37, size = (50,), generator = generator)
        self.scale = torch.rand(size = (50,), generator = generator, dtype = torch.float64)

    def run_impl(self, impl, weight_grad):
        hidden = self.hidden.clone().requires_grad_(requires_grad = True)
        weight = self.weight.clone().requires_grad_(requires_grad = weight_grad)
        losses, correct = linear_cross_entropy(
            hidden_states = hidden,
            weight = weight,
            targets = self.targets,
            chunk_size = 8,
            impl = impl,
        )
        (losses * self.scale).sum().backward()
        return losses.detach(), correct, hidden.grad, weight.grad

    def test_matches_reference(self):
        for weight_grad in (False, True):
            hidden = self.hidden.clone().requires_grad_(requires_grad = True)
            weight = self.weight.clone().requires_grad_(requires_grad = weight_grad)
            logits = hidden @ weight.t()
            expected = F.cross_entropy(input = logits, target = self.targets, reduction = "none")
            (expected * self.scale).sum().backward()
            losses, correct, grad_hidden, grad_weight = self.run_impl(impl = "torch", weight_grad = weight_grad)
            torch.testing.assert_close(actual = losses.double(), expected = expected.detach(), rtol = 1e-5, atol = 1e-5)
            torch.testing.assert_close(actual = grad_hidden, expected = hidden.grad, rtol = 1e-5, atol = 1e-5)
            self.assertTrue(torch.equal(correct, logits.argmax(dim = -1) == self.targets))
            if weight_grad:
                torch.testing.assert_close(actual = grad_weight, expected = weight.grad, rtol = 1e-5, atol = 1e-5)
            else:
                self.assertIsNone(grad_weight)

    def test_same_as_chunked(self):
        for weight_grad in (False, True):
            chunked = self.run_impl(impl = "chunked", weight_grad = weight_grad)
            result = self.run_impl(impl = "torch", weight_grad = weight_grad)
            for want, got in zip(chunked, result):
                if want is None:
                    self.assertIsNone(got)
                elif want.dtype == torch.bool:
                    self.assertTrue(torch.equal(want, got))
                else:
                    torch.testing.assert_close(actual = got.double(), expected = want.double(), rtol = 1e-5, atol = 1e-5)


def relative_error(want, got):
    return float((want.float() - got.float()).norm() / want.float().norm().clamp(min = 1e-12))


@unittest.skipUnless(
    CUDA and HAS_MODEL and kernels_available(),
    "需要 CUDA、阶段 1 模型目录中的代码与配置，以及 flash-attn、causal-conv1d、fla",
)
class PackedTinyModelTest(unittest.TestCase):
    """缩小的随机初始化模型（bfloat16）上比较拼接与填充的批次。"""

    @classmethod
    def setUpClass(cls):
        cls.padded = tiny_batch(audio_windows = (50, 800))

    def model(self, packed, train_language_model = False, dtype = torch.bfloat16):
        from pipeline.train.model import LANGUAGE_MODEL_PREFIXES, PACKED_ATTENTION, check_packing_kernels

        model = tiny_model(dtype = dtype)
        if train_language_model:
            for name, parameter in model.named_parameters():
                if name.startswith(LANGUAGE_MODEL_PREFIXES):
                    parameter.requires_grad_(requires_grad = True)
        if packed:
            model.set_attn_implementation(attn_implementation = PACKED_ATTENTION)
            check_packing_kernels(model = model)
        return model

    def batch(self, packed, padded = None):
        from pipeline.train.data import pack_batch

        padded = self.padded if padded is None else padded
        return pack_batch(batch = copy.copy(padded)) if packed else padded

    def losses_and_grads(self, packed, train_language_model, dtype = torch.bfloat16):
        from pipeline.train.model import move_batch, token_losses

        model = self.model(packed = packed, train_language_model = train_language_model, dtype = dtype)
        inputs = move_batch(batch = self.batch(packed = packed), device = "cuda", feature_dtype = dtype)
        losses, _, rows = token_losses(model = model, inputs = inputs)
        losses.sum().backward()
        grads = {name: parameter.grad.float() for name, parameter in model.named_parameters() if parameter.requires_grad}
        return losses.detach().float(), rows, grads

    def test_losses_and_grads_match(self):
        """以 float32 填充的批次为参考：拼接的批次（bfloat16）的误差不超过填充的批次（bfloat16）本身的误差。

        线性注意力门控参数（dt_bias、A_log、in_proj_a）的梯度是大量正负项之和，bfloat16 下与 float32 本来就相差
        60%–100%，所以不用固定的相对误差，而是要求拼接不比填充更不准。
        """
        for train_language_model in (False, True):
            reference = self.losses_and_grads(
                packed = False,
                train_language_model = train_language_model,
                dtype = torch.float32,
            )
            padded = self.losses_and_grads(packed = False, train_language_model = train_language_model)
            packed = self.losses_and_grads(packed = True, train_language_model = train_language_model)
            self.assertTrue(torch.equal(padded[1], packed[1]))
            padded_error = float((padded[0] - reference[0]).abs().max())
            packed_error = float((packed[0] - reference[0]).abs().max())
            self.assertLessEqual(packed_error, 1.5 * padded_error + 1e-3)
            self.assertEqual(set(padded[2]), set(packed[2]))
            for name, want in reference[2].items():
                padded_error = relative_error(want = want, got = padded[2][name])
                packed_error = relative_error(want = want, got = packed[2][name])
                self.assertLessEqual(packed_error, 1.25 * padded_error + 1e-2, msg = name)

    def differences_after_change(self, model):
        """拼接的批次中把第 1 条（中间一条）的音频倒放（时长不变，各条的位置和总长都不变），返回各条隐状态的最大变化。"""
        from transformers import AutoProcessor

        from pipeline.train.data import TrainCollator
        from pipeline.train.model import hidden_states_precomputed, move_batch

        changed = tiny_items()
        changed[1]["waveform"] = np.flip(changed[1]["waveform"]).copy()
        processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path = str(MODEL_DIR), trust_remote_code = True)
        collator = TrainCollator(processor = processor, audio_windows = (50, 800), pack = True)
        outputs = []
        for batch_items in (tiny_items(), changed):
            inputs = move_batch(batch = collator(items = batch_items), device = "cuda", feature_dtype = torch.bfloat16)
            with torch.no_grad():
                outputs.append(hidden_states_precomputed(model = model, inputs = inputs)[0].float())
        starts = inputs["cu_seq_lens_q"].tolist()
        return [
            float((outputs[0][start:end] - outputs[1][start:end]).abs().max())
            for start, end in zip(starts[:-1], starts[1:])
        ]

    def test_no_leakage(self):
        differences = self.differences_after_change(model = self.model(packed = True))
        self.assertGreater(differences[1], 0.0)
        self.assertEqual(differences[0], 0.0, msg = "第 0 条样本受到了第 1 条的影响")
        self.assertEqual(differences[2], 0.0, msg = "第 2 条样本受到了第 1 条的影响")

    def test_reference_kernels_leak(self):
        """对照：线性注意力换回 transformers 的 PyTorch 参考实现（不认样本边界）时，改动传到了后一条样本。"""
        from pipeline.train.check_packing import reference_linear_attention

        model = self.model(packed = True)
        with reference_linear_attention():
            differences = self.differences_after_change(model = model)
        self.assertEqual(differences[0], 0.0)
        self.assertGreater(differences[2], 0.0)


def tiny_items():
    """与 test_offload.tiny_batch 相同的三条样本（不同时长、一条超过 8 秒、一条空目标）。"""
    rng = np.random.default_rng(seed = 0)
    items = []
    for number, seconds in enumerate((3.2, 9.5, 1.7)):
        items.append({
            "sample_id": f"t-{number}",
            "lang": "en",
            "audio_path": f"t-{number}.flac",
            "duration_s": seconds,
            "waveform": (0.1 * rng.standard_normal(size = int(seconds * 16000))).astype(np.float32),
            "target": "" if number == 2 else f"sentence number {number}",
            "empty": number == 2,
            "mix": None,
        })
    return items


if __name__ == "__main__":
    unittest.main()
