"""带系统提示的自生成回放组（#71，free_sys_chat、free_sys_chat_think）的单元测试：

    python -m unittest pipeline.replay.tests.test_system_prompts

只在 CPU 上运行，用仓库中阶段 1 模型目录的分词器与 chat template（不需要权重），模型目录可以用环境变量 MODEL_DIR 指定。
测试把分词器文件复制到临时目录，去掉 processor 的 auto_map，加载时不需要自定义代码。

- TemplateTest：system_chat_segments 切出的各段；按「系统轮开头 + 系统提示 + <|im_end|> + 补上的用户轮开头 + 用户消息 +
  <|im_end|> + 补上的助手轮开头 + 模型写的内容」逐段分词拼出的 token，与 chat template 渲染 [系统, 用户, 助手] 完整对话
  （enable_thinking 为假或真）后整体分词的结果逐个相同；生成提示部分与 add_generation_prompt = True 的渲染相同；助手轮开头
  与 free_chat、free_chat_think 相同。
- PromptsTest：prompts 一步写出的提示清单与 templates.json；free_chat、free_chat_think 的模板不变。
- GenerateTest：generate 的分段生成（用假的 vLLM 引擎按脚本输出 token）：每一段的提示是前面全部片段的拼接；系统提示或用户
  消息没有正常结束的样本状态为 no_system_end、no_user_end；--drop-loops 时有循环的样本整条丢弃。finalize 写出系统、用户、
  助手文本，没有助手回答的样本不用，思考组没有写出 </think> 的样本不用。free_chat 的两段生成与改动之前相同。
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pipeline.replay.finalize import finalize_split
from pipeline.replay.generate import Generator
from pipeline.replay.prompts import (
    chat_segments,
    encode,
    system_chat_segments,
    system_conversation,
    thinking_between,
)
from pipeline.replay.prompts import main as prompts_main

REPO = Path(__file__).resolve().parents[3]
MODEL_DIR = Path(os.environ.get("MODEL_DIR", REPO / "qwen3.5-0.8b-mllm"))
HAS_MODEL = (MODEL_DIR / "tokenizer.json").exists()
IM_START, IM_END, ENDOFTEXT, THINK, THINK_END = 248045, 248046, 248044, 248068, 248069
SYSTEM_START = [IM_START, 8678, 198]
USER_START = [IM_START, 846, 198]
SYSTEM_BETWEEN = [198, IM_START, 846, 198]
CHAT_BETWEEN = [198, IM_START, 74455, 198, THINK, 271, THINK_END, 271]
THINK_BETWEEN = [198, IM_START, 74455, 198, THINK, 198]
# (系统提示, 用户消息, 思考, 回答)：英文、中文、多行与代码
EXAMPLES = [
    ("You are a concise assistant. Answer in French.", "What is the capital of Japan?", "The user wants the capital.", "Tokyo."),
    ("你是一名中文写作助手，请用简洁的语言回答。\n回答不超过三句话。", "帮我写一句生日祝福。", "用户想要一句祝福。", "祝你生日快乐，天天开心！"),
    (
        "You are a Python tutor.\n\nRules:\n- Use code blocks.\n- Keep answers under 100 words.",
        "How do I reverse a list?\nShow an example.",
        "Simple: slicing or reverse().\nMention both.",
        "Use slicing:\n\n```python\nitems = [1, 2, 3]\nprint(items[::-1])\n```",
    ),
]


def copy_tokenizer(target):
    """把分词器文件复制到 target，去掉 tokenizer_config.json 中的 auto_map（只用分词器，不加载 processor 的自定义代码）。"""
    target = Path(target)
    target.mkdir(parents = True, exist_ok = True)
    for name in ("tokenizer.json", "chat_template.jinja"):
        shutil.copy2(src = MODEL_DIR / name, dst = target / name)
    config = json.loads((MODEL_DIR / "tokenizer_config.json").read_text(encoding = "utf-8"))
    config.pop("auto_map", None)
    config.pop("processor_class", None)
    (target / "tokenizer_config.json").write_text(json.dumps(config, ensure_ascii = False, indent = 2), encoding = "utf-8")
    return target


@unittest.skipUnless(HAS_MODEL, "需要模型目录中的分词器文件")
class TemplateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls.tmp = tempfile.TemporaryDirectory()
        cls.model_dir = copy_tokenizer(target = Path(cls.tmp.name) / "model")
        cls.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = str(cls.model_dir))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def encode(self, text):
        return encode(tokenizer = self.tokenizer, text = text)

    def test_segments(self):
        for thinking in (False, True):
            segments = system_chat_segments(tokenizer = self.tokenizer, thinking = thinking)
            self.assertEqual(segments["system_start"], "<|im_start|>system\n")
            self.assertEqual(segments["system_between"], "\n<|im_start|>user\n")
            opening = "\n<|im_start|>assistant\n<think>\n" + ("" if thinking else "\n</think>\n\n")
            self.assertEqual(segments["user_between"], opening)
            self.assertEqual(segments["end"], "<|im_end|>\n")
            self.assertEqual(self.encode(text = segments["system_start"]), SYSTEM_START)
            self.assertEqual(self.encode(text = segments["system_between"]), SYSTEM_BETWEEN)
            self.assertEqual(self.encode(text = segments["user_between"]), THINK_BETWEEN if thinking else CHAT_BETWEEN)
        # 助手轮的打开方式与没有系统提示的 free_chat、free_chat_think 相同
        self.assertEqual(
            system_chat_segments(tokenizer = self.tokenizer, thinking = False)["user_between"],
            chat_segments(tokenizer = self.tokenizer)["between"].removeprefix("<|im_end|>"),
        )
        self.assertEqual(system_chat_segments(tokenizer = self.tokenizer, thinking = True)["user_between"], thinking_between(tokenizer = self.tokenizer))

    def test_token_identical_to_chat_template(self):
        for thinking in (False, True):
            segments = system_chat_segments(tokenizer = self.tokenizer, thinking = thinking)
            for system, user, reasoning, answer in EXAMPLES:
                with self.subTest(thinking = thinking, system = system[:20]):
                    prompt_ids = (
                        self.encode(text = segments["system_start"])
                        + self.encode(text = system)
                        + [IM_END]
                        + self.encode(text = segments["system_between"])
                        + self.encode(text = user)
                        + [IM_END]
                        + self.encode(text = segments["user_between"])
                    )
                    rendered_prompt = self.tokenizer.apply_chat_template(
                        conversation = system_conversation(system = system, user = user),
                        tokenize = False,
                        add_generation_prompt = True,
                        enable_thinking = thinking,
                    )
                    self.assertEqual(prompt_ids, self.encode(text = rendered_prompt))
                    # 模型写出的内容：非思考模式为回答，思考模式为思考、</think>、回答；都以 <|im_end|> 结束
                    written = f"{reasoning}\n</think>\n\n{answer}<|im_end|>" if thinking else f"{answer}<|im_end|>"
                    built = prompt_ids + self.encode(text = written)
                    rendered = self.tokenizer.apply_chat_template(
                        conversation = system_conversation(
                            system = system,
                            user = user,
                            assistant = answer,
                            reasoning = reasoning if thinking else None,
                        ),
                        tokenize = False,
                        add_generation_prompt = False,
                        enable_thinking = thinking,
                    )
                    expected = self.encode(text = rendered)
                    # 完整渲染以 <|im_end|>\n 结束；样本停在生成的 <|im_end|>，与 free_chat 相同
                    self.assertEqual(expected[-1], 198)
                    self.assertEqual(built, expected[:-1])
                    self.assertEqual(built.count(IM_END), 3)


def read_jsonl(path):
    with open(path, encoding = "utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_prompts(model_dir, replay_dir, groups, count):
    argv = [
        "prompts",
        "--model",
        str(model_dir),
        "--replay-dir",
        str(replay_dir),
        "--groups",
        *groups,
        "--free-count",
        f"eval={count},train=0",
    ]
    with mock.patch.object(sys, "argv", argv):
        prompts_main()


@unittest.skipUnless(HAS_MODEL, "需要模型目录中的分词器文件")
class PromptsTest(unittest.TestCase):
    def test_prompt_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = copy_tokenizer(target = Path(tmp) / "model")
            replay_dir = Path(tmp) / "replay"
            write_prompts(
                model_dir = model_dir,
                replay_dir = replay_dir,
                groups = ["free_chat", "free_chat_think", "free_sys_chat", "free_sys_chat_think"],
                count = 3,
            )
            templates = json.loads((replay_dir / "templates.json").read_text(encoding = "utf-8"))
            for group in ("free_sys_chat", "free_sys_chat_think"):
                rows = read_jsonl(path = replay_dir / group / "eval" / "prompts.jsonl")
                self.assertEqual([row["prompt_index"] for row in rows], [0, 1, 2])
                self.assertTrue(all(row["prompt_ids"] == SYSTEM_START for row in rows))
                self.assertEqual(templates[group]["start_ids"], SYSTEM_START)
                self.assertEqual(templates[group]["system_between_ids"], SYSTEM_BETWEEN)
                self.assertEqual(templates[group]["between_ids"], THINK_BETWEEN if group.endswith("think") else CHAT_BETWEEN)
            # 没有系统提示的两组与以前相同
            self.assertEqual(templates["free_chat"]["start_ids"], USER_START)
            self.assertEqual(templates["free_chat"]["between_ids"], CHAT_BETWEEN)
            self.assertEqual(templates["free_chat_think"]["between_ids"], THINK_BETWEEN)


class FakeEngine:
    """按脚本输出的假 vLLM 引擎：每个请求在下一步一次给出全部 token。script(请求编号, 提示) 返回 (token 列表, 结束方式)。"""

    def __init__(self, script):
        self.script = script
        self.pending = {}
        self.prompts = {}

    def add_request(self, request_id, prompt, params):
        prompt_ids = list(prompt["prompt_token_ids"])
        self.prompts[request_id] = prompt_ids
        tokens, reason = self.script(request_id, prompt_ids)
        if len(tokens) > params["max_tokens"]:
            tokens, reason = tokens[: params["max_tokens"]], "length"
        self.pending[request_id] = (list(tokens), reason)

    def has_unfinished_requests(self):
        return bool(self.pending)

    def do_log_stats_with_interval(self):
        pass

    def step(self):
        outputs = []
        for request_id, (tokens, reason) in list(self.pending.items()):
            del self.pending[request_id]
            completion = SimpleNamespace(
                token_ids = tokens,
                logprobs = SimpleNamespace(token_ids = tokens, logprobs = [-0.5] * len(tokens)),
                finish_reason = reason,
            )
            outputs.append(SimpleNamespace(request_id = request_id, finished = True, outputs = [completion]))
        return outputs

    def abort_request(self, request_ids):
        for request_id in request_ids:
            self.pending.pop(request_id, None)


def fake_generator(replay_dir, engine, max_new_tokens = 16384):
    """不加载 vLLM 的 Generator：只换掉引擎和采样参数，其余逻辑与正式运行相同（--drop-loops、--topk 0、--count-closed-think）。"""
    generator = Generator.__new__(Generator)
    generator.args = argparse.Namespace(
        replay_dir = str(replay_dir),
        count_closed_think = True,
        topk = 0,
        prior_mean = "1000",
        max_inflight = "0",
        max_num_seqs = 4,
        max_new_tokens = max_new_tokens,
        no_filter = False,
        drop_loops = True,
        check_every = 256,
        loop_margin = 1024,
        log_every = 600,
        seed = 0,
        model = "fake",
    )
    generator.SamplingParams = dict
    generator.delta = None
    generator.engine = engine
    generator.stop_ids = [IM_END, ENDOFTEXT]
    generator.im_end = IM_END
    generator.think_end = THINK_END
    generator.load_seconds = 0.0
    return generator


@unittest.skipUnless(HAS_MODEL, "需要模型目录中的分词器文件")
class GenerateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls.tmp = tempfile.TemporaryDirectory()
        cls.model_dir = copy_tokenizer(target = Path(cls.tmp.name) / "model")
        cls.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = str(cls.model_dir))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def encode(self, text):
        return encode(tokenizer = self.tokenizer, text = text)

    def scripts(self, thinking):
        """五条样本：0、4 正常（思考组的第 4 条没有写出 </think>），1 系统提示以 <|endoftext|> 结束，2 用户消息以
        <|endoftext|> 结束，3 助手回答原样循环。返回 script 函数与每条样本的 (系统, 用户, 助手) token。"""
        loop = self.encode(text = " ".join(f"w{number}" for number in range(12)))
        texts = {}
        for index in range(5):
            system, user, reasoning, answer = EXAMPLES[index % len(EXAMPLES)]
            system_ids = self.encode(text = system) + ([ENDOFTEXT] if index == 1 else [IM_END])
            user_ids = self.encode(text = user) + ([ENDOFTEXT] if index == 2 else [IM_END])
            if index == 3:
                assistant_ids = (loop * 12) + [IM_END]
            elif thinking and index != 4:
                assistant_ids = self.encode(text = f"{reasoning}\n</think>\n\n{answer}") + [IM_END]
            elif thinking:
                assistant_ids = self.encode(text = reasoning) + [IM_END]
            else:
                assistant_ids = self.encode(text = answer) + [IM_END]
            texts[index] = (system_ids, user_ids, assistant_ids)

        def script(request_id, prompt_ids):
            index, phase = map(int, request_id.split("-"))
            return texts[index][phase - 1], "stop"

        return script, texts

    def test_system_groups(self):
        for group in ("free_sys_chat", "free_sys_chat_think"):
            thinking = group.endswith("think")
            with self.subTest(group = group), tempfile.TemporaryDirectory() as tmp:
                replay_dir = Path(tmp) / "replay"
                write_prompts(model_dir = self.model_dir, replay_dir = replay_dir, groups = [group], count = 5)
                script, texts = self.scripts(thinking = thinking)
                engine = FakeEngine(script = script)
                generator = fake_generator(replay_dir = replay_dir, engine = engine)
                finished = generator.run_split(group = group, split = "eval", target = 10**9, deadline = time.time() + 600)
                self.assertTrue(finished)
                between = THINK_BETWEEN if thinking else CHAT_BETWEEN
                system_ids, user_ids, _ = texts[0]
                # 每一段的提示是前面全部片段的拼接
                self.assertEqual(engine.prompts["0-1"], SYSTEM_START)
                self.assertEqual(engine.prompts["0-2"], SYSTEM_START + system_ids + SYSTEM_BETWEEN)
                self.assertEqual(engine.prompts["0-3"], SYSTEM_START + system_ids + SYSTEM_BETWEEN + user_ids + between)
                # 没有正常结束的段之后不再生成
                self.assertNotIn("1-2", engine.prompts)
                self.assertNotIn("2-3", engine.prompts)
                finalize_split(
                    replay_dir = replay_dir,
                    group = group,
                    split = "eval",
                    model_dir = str(self.model_dir),
                    jobs = 1,
                    drop_loops = True,
                )
                rows = {row["prompt_index"]: row for row in read_jsonl(path = replay_dir / group / "eval" / "samples.jsonl")}
                self.assertEqual(rows[1]["status"], "no_system_end")
                self.assertEqual(rows[2]["status"], "no_user_end")
                reasons = {index: rows[index]["drop_reason"] for index in rows if not rows[index]["keep"]}
                expected = {1: "no_system_end", 2: "no_user_end", 3: "loop"}
                if thinking:
                    expected[4] = "think_unclosed"
                self.assertEqual(reasons, expected)
                system, user, reasoning, answer = EXAMPLES[0]
                self.assertEqual(rows[0]["system_text"], system)
                self.assertEqual(rows[0]["user_text"], user)
                if thinking:
                    self.assertTrue(rows[0]["think_closed"])
                    # answer_text 是 </think> 之后的全部内容（含模板中的两个换行），与 free_chat_think 相同
                    self.assertEqual(rows[0]["answer_text"], f"\n\n{answer}")
                    self.assertEqual(rows[0]["assistant_text"], f"{reasoning}\n</think>\n\n{answer}")
                else:
                    self.assertEqual(rows[0]["assistant_text"], answer)
                # 保存的片段依次拼起来，就是 chat template 渲染完整对话的 token（去掉结尾的换行）
                sequence = []
                for piece in rows[0]["pieces"]:
                    if "gen" in piece:
                        sequence += texts[0][("system", "user", "assistant").index(piece["gen"])]
                    else:
                        sequence += piece["ids"]
                rendered = self.tokenizer.apply_chat_template(
                    conversation = system_conversation(
                        system = system,
                        user = user,
                        assistant = answer,
                        reasoning = reasoning if thinking else None,
                    ),
                    tokenize = False,
                    add_generation_prompt = False,
                    enable_thinking = thinking,
                )
                self.assertEqual(sequence, self.encode(text = rendered)[:-1])
                self.assertEqual([piece.get("gen") for piece in rows[0]["pieces"]], [None, "system", None, "user", None, "assistant"])

    def test_free_chat_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay_dir = Path(tmp) / "replay"
            write_prompts(model_dir = self.model_dir, replay_dir = replay_dir, groups = ["free_chat"], count = 3)
            user_ids = self.encode(text = EXAMPLES[0][1]) + [IM_END]
            answer_ids = self.encode(text = EXAMPLES[0][3]) + [IM_END]

            def script(request_id, prompt_ids):
                index, phase = map(int, request_id.split("-"))
                if index == 2 and phase == 1:
                    return user_ids[:-1] + [ENDOFTEXT], "stop"
                return (user_ids if phase == 1 else answer_ids), "stop"

            engine = FakeEngine(script = script)
            generator = fake_generator(replay_dir = replay_dir, engine = engine)
            self.assertTrue(generator.run_split(group = "free_chat", split = "eval", target = 10**9, deadline = time.time() + 600))
            self.assertEqual(sorted(engine.prompts), ["0-1", "0-2", "1-1", "1-2", "2-1"])
            self.assertEqual(engine.prompts["0-1"], USER_START)
            self.assertEqual(engine.prompts["0-2"], USER_START + user_ids + CHAT_BETWEEN)
            records = read_jsonl(path = next((replay_dir / "free_chat" / "eval" / "raw").glob("shard-*.jsonl")))
            status = {record["prompt_index"]: record["status"] for record in records}
            self.assertEqual(status, {0: "ok", 1: "ok", 2: "no_user_end"})


if __name__ == "__main__":
    unittest.main()
