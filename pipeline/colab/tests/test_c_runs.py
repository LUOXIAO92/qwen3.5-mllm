"""C 组运行登记（pipeline/colab/c_runs.py）的单元测试：python -m unittest pipeline.colab.tests.test_c_runs

只在 CPU 上运行：

- SysRunsTest：第五轮的两个运行（c-sys-kl32tail-lora90、c-sys-kl32tail-lr5e-6）的回放组与权重、kl32 加尾部项、语言模型的
  训练方式（LoRA 秩 90 或全量微调 5e-6），回放组的写法能被训练程序解析；第三、四轮的同名设置不受影响。
- DriversTest：本机的三个评测驱动程序（c71_local、bench_local、sysprompt_local）接受这两个运行名。
"""

import contextlib
import io
import unittest

from pipeline.colab import c_runs
from pipeline.eval import bench_local, c71_local, sysprompt_local
from pipeline.train.replay import parse_groups

SYS_KEYS = ("c-sys-kl32tail-lora90", "c-sys-kl32tail-lr5e-6")


class SysRunsTest(unittest.TestCase):
    def test_replay_groups(self):
        for key in SYS_KEYS:
            self.assertEqual(c_runs.ALL_RUNS[key], "cont_raw=2,free_sys_chat=1,free_sys_chat_think=1")
            self.assertEqual(c_runs.replay_loss(key = key), "kl32")
            self.assertIn(key, c_runs.KL_TAIL_RUNS)
        self.assertEqual(
            parse_groups(spec = c_runs.SYS_REPLAY),
            {"cont_raw": 2.0, "free_sys_chat": 1.0, "free_sys_chat_think": 1.0},
        )

    def test_language_model_settings(self):
        lora = c_runs.full_arguments(
            init = "init",
            lm_lr = c_runs.LM_LR.get("c-sys-kl32tail-lora90", "1e-5"),
            lora_rank = c_runs.LORA_RANK.get("c-sys-kl32tail-lora90"),
        )
        self.assertEqual(lora[:4], ["--language-model-lora-rank", "90", "--language-model-lora-lr", "1e-4"])
        self.assertNotIn("--train-language-model", lora)
        full = c_runs.full_arguments(
            init = "init",
            lm_lr = c_runs.LM_LR.get("c-sys-kl32tail-lr5e-6", "1e-5"),
            lora_rank = c_runs.LORA_RANK.get("c-sys-kl32tail-lr5e-6"),
        )
        self.assertEqual(full[:3], ["--train-language-model", "--language-model-lr", "5e-6"])
        self.assertNotIn("--language-model-lora-rank", full)

    def test_replay_arguments(self):
        arguments = c_runs.replay_arguments(
            spec = c_runs.ALL_RUNS["c-sys-kl32tail-lr5e-6"],
            loss = "kl32",
            tail = True,
        )
        self.assertEqual(arguments[arguments.index("--replay-groups") + 1], c_runs.SYS_REPLAY)
        self.assertEqual(arguments[-1], "--replay-kl-tail")

    def test_earlier_runs_unchanged(self):
        self.assertEqual(c_runs.ALL_RUNS["c-cont_raw-kl32tail-lora90"], "cont_raw=1")
        self.assertEqual(c_runs.ALL_RUNS["c-cont_raw-kl32tail-lora32"], "cont_raw=1")
        self.assertEqual(c_runs.ALL_RUNS["c-cont_raw-kl32tail-lr5e-6"], "cont_raw=1")
        self.assertEqual(c_runs.LORA_RANK["c-cont_raw-kl32tail-lora90"], 90)
        self.assertEqual(c_runs.LM_LR["c-cont_raw-kl32tail-lr5e-6"], "5e-6")
        self.assertEqual(len(c_runs.ALL_RUNS), 18)  # 加上第六轮的两个 free_raw 运行和改名的 -8bit 运行

    def test_run_names(self):
        self.assertEqual(c_runs.run_name(key = "c-sys-kl32tail-lora90"), "v2b-2b-cpt77-c-sys-kl32tail-lora90" + c_runs.TAG)
        self.assertEqual(c_runs.run_name(key = "c-sys-kl32tail-lr5e-6"), "v2b-2b-cpt77-c-sys-kl32tail-lr5e-6" + c_runs.TAG)


class DriversTest(unittest.TestCase):
    def test_drivers_accept_sys_runs(self):
        runs = ",".join(SYS_KEYS)
        args = c71_local.parse_args(argv = ["--dataset", "data", "--runs", runs])
        self.assertEqual(args.keys, list(SYS_KEYS))
        args = bench_local.parse_args(argv = ["--dataset", "data", "--runs", runs])
        self.assertEqual(args.keys, list(SYS_KEYS))
        args = sysprompt_local.parse_args(argv = ["--runs-root", "runs", "--runs", runs, "--dry-run"])
        self.assertEqual(args.keys, list(SYS_KEYS))
        # 默认点名全部运行时也包含这两个
        args = sysprompt_local.parse_args(argv = ["--runs-root", "runs", "--dry-run"])
        self.assertTrue(set(SYS_KEYS) <= set(args.keys))

    def test_unknown_run_rejected(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(new_target = io.StringIO()):
            c71_local.parse_args(argv = ["--dataset", "data", "--runs", "c-sys-kl32"])


if __name__ == "__main__":
    unittest.main()


class FreeRunsTest(unittest.TestCase):
    """第六轮的两个运行：回放以 free_raw 为基础，kl32 加尾部项，语言模型学习率 5e-6，全量微调（不是 LoRA）。"""

    def test_free_runs(self):
        expected = {
            "c-free_raw-kl32tail-lr5e-6": "free_raw=1",
            "c-frsys-kl32tail-lr5e-6": "free_raw=2,free_sys_chat=1,free_sys_chat_think=1",
        }
        for key, spec in expected.items():
            self.assertEqual(c_runs.ALL_RUNS[key], spec)
            self.assertEqual(c_runs.replay_loss(key = key), "kl32")
            self.assertIn(key, c_runs.KL_TAIL_RUNS)
            self.assertEqual(c_runs.LM_LR[key], "5e-6")
            self.assertNotIn(key, c_runs.LORA_RANK)
            arguments = c_runs.replay_arguments(spec = spec, loss = "kl32", tail = True)
            self.assertEqual(arguments[arguments.index("--replay-groups") + 1], spec)
            self.assertIn("--replay-kl-tail", arguments)

    def test_8bit_run_kept(self):
        """第四轮 8 位状态的运行改名为 -8bit 后设置不变，c-sys-kl32tail-lr5e-6 仍在（默认精度重训）。"""
        for key in ("c-sys-kl32tail-lr5e-6", "c-sys-kl32tail-lr5e-6-8bit"):
            self.assertEqual(c_runs.ALL_RUNS[key], c_runs.SYS_REPLAY)
            self.assertIn(key, c_runs.KL_TAIL_RUNS)
            self.assertEqual(c_runs.LM_LR[key], "5e-6")
            self.assertNotIn(key, c_runs.LORA_RANK)

