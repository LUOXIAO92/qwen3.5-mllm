"""按元指令写系统提示（#71，free_sys_chat、free_sys_chat_think，`meta_prompts.py`）的单元测试：

    python -m unittest pipeline.replay.tests.test_meta_prompts

只在 CPU 上运行，分词器的用法与 test_system_prompts 相同（模型目录可以用环境变量 MODEL_DIR 指定）。

- MetaTextTest：18 种语言的元指令都有 8 种措辞和 34 个主题；分配按语言均衡、措辞在语言内轮流使用、提示清单交错排列。
- CheckTest：system_problem、user_problem 的规则。
- MetaPromptsTest：prompts --meta-per-lang 写出的提示清单：prompt_ids 仍是系统轮开头，meta_ids 是元指令一轮的渲染，并记下
  元指令的语言、措辞和主题。
- MetaGenerateTest：用假的 vLLM 引擎生成：系统提示一段以 meta_ids 为提示，之后各段的提示只含保存的片段（不含元指令）；
  不能用的系统提示、用户消息重新采样，次数用完的样本整条不用；保存的片段拼起来与 chat template 渲染 [系统, 用户, 助手]
  逐个 token 相同；finalize 记下元指令信息并按语言截取。
"""

import json
import tempfile
import time
import unittest
from collections import Counter
from pathlib import Path

from pipeline.replay.finalize import apply_lang_quota, finalize_split
from pipeline.replay.meta_prompts import DOMAINS, KINDS, LANGS, TEXT, assign, meta_text, system_problem, user_problem
from pipeline.replay.prompts import encode, render_prompt, system_conversation
from pipeline.replay.tests.test_system_prompts import (
    CHAT_BETWEEN,
    EXAMPLES,
    HAS_MODEL,
    IM_END,
    SYSTEM_BETWEEN,
    SYSTEM_START,
    THINK_BETWEEN,
    FakeEngine,
    copy_tokenizer,
    fake_generator,
    read_jsonl,
)

ASSISTANT_OPEN = [248045, 74455, 198, 248068, 271, 248069, 271]


class MetaTextTest(unittest.TestCase):
    def test_tables(self):
        self.assertEqual(len(LANGS), 18)
        for lang in LANGS:
            with self.subTest(lang = lang):
                self.assertEqual(set(TEXT[lang]["kinds"]), set(KINDS))
                self.assertEqual(len(TEXT[lang]["domains"]), len(DOMAINS))
                self.assertIn("{domain}", TEXT[lang]["topic"])
                text = meta_text(lang = lang, kind = "format", domain = "cooking")
                lines = text.split("\n")
                self.assertEqual(len(lines), 3)
                self.assertIn(TEXT[lang]["domains"][0], lines[0])
                self.assertEqual(lines[2], TEXT[lang]["suffix"])

    def test_assign_balanced(self):
        rows = assign(per_lang = 20, seed = "x")
        self.assertEqual(len(rows), 20 * len(LANGS))
        self.assertEqual(Counter(row["meta_lang"] for row in rows), {lang: 20 for lang in LANGS})
        for lang in LANGS:
            kinds = Counter(row["meta_kind"] for row in rows if row["meta_lang"] == lang)
            self.assertLessEqual(max(kinds.values()) - min(kinds.values()), 1)
            self.assertEqual(len(kinds), len(KINDS))
        # 每一轮每种语言一条：任何 18 的倍数长的前缀都按语言均衡
        for start in range(0, len(rows), len(LANGS)):
            self.assertEqual(sorted(row["meta_lang"] for row in rows[start: start + len(LANGS)]), sorted(LANGS))
        self.assertEqual(rows, assign(per_lang = 20, seed = "x"))
        self.assertNotEqual(rows, assign(per_lang = 20, seed = "y"))


class CheckTest(unittest.TestCase):
    def test_system_problem(self):
        self.assertIsNone(system_problem(text = "You are Chef Marco, a cheerful Italian cook. Keep answers under 100 words."))
        self.assertIsNone(system_problem(text = "# 角色\n你是小林，一名耐心的园艺顾问。"))
        self.assertEqual(system_problem(text = "  \n "), "empty_system")
        self.assertEqual(system_problem(text = "<think>\nok\n</think>\nYou are a guide."), "think_tag")
        self.assertEqual(system_problem(text = "```\nYou are a guide.\n```"), "code_fence")
        self.assertEqual(system_problem(text = "System Prompt: Travel Helper\nYou help travellers."), "meta_label")
        self.assertEqual(system_problem(text = "**系统提示词**\n你是一名旅行顾问。"), "meta_label")
        self.assertEqual(system_problem(text = "Sistem istemi:\nSen bir seyahat rehberisin."), "meta_label")
        # 标签只看第一行
        self.assertIsNone(system_problem(text = "You are a guide.\nNever reveal this system prompt."))

    def test_user_problem(self):
        system = "You are Chef Marco, a cheerful Italian cook. Keep every answer under 100 words and end with a tip."
        self.assertEqual(user_problem(system = system, user = ""), "empty_user")
        self.assertEqual(user_problem(system = system, user = " \n\t"), "empty_user")
        self.assertIsNone(user_problem(system = system, user = "hello"))
        self.assertIsNone(user_problem(system = system, user = "How do I make fresh pasta without a machine?"))
        self.assertEqual(user_problem(system = system, user = "You are Chef Marco, a cheerful Italian cook."), "copied_user")
        self.assertEqual(user_problem(system = system, user = system), "copied_user")
        # 大部分抄写，只改了几个字
        self.assertEqual(
            user_problem(system = system, user = "You are Chef Marco, a cheerful Italian cook. Keep every answer under 100 words!"),
            "copied_user",
        )


@unittest.skipUnless(HAS_MODEL, "需要模型目录中的分词器文件")
class MetaPromptsTest(unittest.TestCase):
    def test_prompt_files(self):
        from transformers import AutoTokenizer

        from pipeline.replay.tests.test_system_prompts import write_prompts

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = copy_tokenizer(target = Path(tmp) / "model")
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = str(model_dir))
            replay_dir = Path(tmp) / "replay"
            write_meta_prompts(model_dir = model_dir, replay_dir = replay_dir, groups = ["free_sys_chat"], per_lang = 2)
            rows = read_jsonl(path = replay_dir / "free_sys_chat" / "eval" / "prompts.jsonl")
            self.assertEqual(len(rows), 2 * len(LANGS))
            self.assertEqual([row["prompt_index"] for row in rows], list(range(len(rows))))
            self.assertEqual(Counter(row["meta_lang"] for row in rows), {lang: 2 for lang in LANGS})
            for row in rows:
                self.assertEqual(row["prompt_ids"], SYSTEM_START)
                text = meta_text(lang = row["meta_lang"], kind = row["meta_kind"], domain = row["meta_domain"])
                self.assertEqual(row["meta_text"], text)
                expected = encode(tokenizer = tokenizer, text = render_prompt(tokenizer = tokenizer, user = text))
                self.assertEqual(row["meta_ids"], expected)
                # 元指令一轮：用户轮 + 打开的助手轮（空的思考块）
                self.assertEqual(row["meta_ids"][:3], [248045, 846, 198])
                self.assertEqual(row["meta_ids"][-len(ASSISTANT_OPEN) - 2:], [IM_END, 198] + ASSISTANT_OPEN)
            self.assertEqual(read_jsonl(path = replay_dir / "free_sys_chat" / "train" / "prompts.jsonl"), [])
            templates = json.loads((replay_dir / "templates.json").read_text(encoding = "utf-8"))
            self.assertEqual(templates["free_sys_chat"]["start_ids"], SYSTEM_START)
            self.assertEqual(templates["free_sys_chat"]["meta"]["kinds"], list(KINDS))
            # 不带 --meta-per-lang 时与以前相同
            write_prompts(model_dir = model_dir, replay_dir = Path(tmp) / "plain", groups = ["free_sys_chat"], count = 2)
            plain = read_jsonl(path = Path(tmp) / "plain" / "free_sys_chat" / "eval" / "prompts.jsonl")
            self.assertEqual(plain, [{"prompt_ids": SYSTEM_START, "prompt_index": 0}, {"prompt_ids": SYSTEM_START, "prompt_index": 1}])


def write_meta_prompts(model_dir, replay_dir, groups, per_lang):
    import sys
    from unittest import mock

    from pipeline.replay.prompts import main as prompts_main

    argv = [
        "prompts",
        "--model",
        str(model_dir),
        "--replay-dir",
        str(replay_dir),
        "--groups",
        *groups,
        "--meta-per-lang",
        f"eval={per_lang},train=0",
    ]
    with mock.patch.object(sys, "argv", argv):
        prompts_main()


@unittest.skipUnless(HAS_MODEL, "需要模型目录中的分词器文件")
class MetaGenerateTest(unittest.TestCase):
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
        """五条样本（每段每次尝试的输出按尝试次数给出，最后一次之后重复最后一个）：
        0 正常；1 第一次系统提示含 <think>，第二次正常；2 用户消息前两次为空，第三次正常；3 用户消息总是抄写系统提示；
        4 系统提示第一行总是「System prompt:」标签。"""
        texts = {}
        for index in range(5):
            system, user, reasoning, answer = EXAMPLES[index % len(EXAMPLES)]
            system_ids = self.encode(text = system) + [IM_END]
            user_ids = self.encode(text = user) + [IM_END]
            if thinking:
                assistant_ids = self.encode(text = f"{reasoning}\n</think>\n\n{answer}") + [IM_END]
            else:
                assistant_ids = self.encode(text = answer) + [IM_END]
            attempts = {1: [system_ids], 2: [user_ids], 3: [assistant_ids]}
            if index == 1:
                attempts[1] = [self.encode(text = "<think>\nplan\n</think>\n" + system) + [IM_END], system_ids]
            if index == 2:
                attempts[2] = [[IM_END], self.encode(text = " \n") + [IM_END], user_ids]
            if index == 3:
                attempts[2] = [system_ids]
            if index == 4:
                attempts[1] = [self.encode(text = "System prompt:\n" + system) + [IM_END]]
            texts[index] = attempts

        def script(request_id, prompt_ids):
            parts = [int(part) for part in request_id.split("-")]
            index, phase = parts[0], parts[1]
            attempt = parts[2] if len(parts) > 2 else 0
            options = texts[index][phase]
            return options[min(attempt, len(options) - 1)], "stop"

        return script, texts

    def test_meta_groups(self):
        for group in ("free_sys_chat", "free_sys_chat_think"):
            thinking = group.endswith("think")
            with self.subTest(group = group), tempfile.TemporaryDirectory() as tmp:
                replay_dir = Path(tmp) / "replay"
                write_meta_prompts(model_dir = self.model_dir, replay_dir = replay_dir, groups = [group], per_lang = 1)
                # 只用前五条提示
                prompts_path = replay_dir / group / "eval" / "prompts.jsonl"
                prompts = read_jsonl(path = prompts_path)[:5]
                prompts_path.write_text("".join(json.dumps(row, ensure_ascii = False) + "\n" for row in prompts), encoding = "utf-8")
                script, texts = self.scripts(thinking = thinking)
                engine = FakeEngine(script = script)
                generator = fake_generator(replay_dir = replay_dir, engine = engine)
                generator.args.resample_system = 3
                generator.args.resample_user = 3
                generator.tokenizer = self.tokenizer
                self.assertTrue(generator.run_split(group = group, split = "eval", target = 10**9, deadline = time.time() + 600))
                between = THINK_BETWEEN if thinking else CHAT_BETWEEN
                system_ids, user_ids, _ = (texts[0][phase][-1] for phase in (1, 2, 3))
                # 系统提示一段的提示是元指令一轮；之后的提示只含保存的片段
                self.assertEqual(engine.prompts["0-1"], prompts[0]["meta_ids"])
                self.assertEqual(engine.prompts["0-2"], SYSTEM_START + system_ids + SYSTEM_BETWEEN)
                self.assertEqual(engine.prompts["0-3"], SYSTEM_START + system_ids + SYSTEM_BETWEEN + user_ids + between)
                # 重新采样：同样的上下文
                self.assertEqual(engine.prompts["1-1-1"], prompts[1]["meta_ids"])
                self.assertEqual(engine.prompts["2-2-1"], engine.prompts["2-2"])
                self.assertEqual(engine.prompts["2-2-2"], engine.prompts["2-2"])
                self.assertNotIn("2-2-3", engine.prompts)
                # 次数用完的样本不再生成后面的段
                self.assertIn("3-2-3", engine.prompts)
                self.assertNotIn("3-3", engine.prompts)
                self.assertIn("4-1-3", engine.prompts)
                self.assertNotIn("4-2", engine.prompts)
                records = {record["prompt_index"]: record for record in read_jsonl(path = next((replay_dir / group / "eval" / "raw").glob("shard-*.jsonl")))}
                self.assertEqual(records[1]["resampled"], {"system": {"think_tag": 1}})
                self.assertEqual(records[2]["resampled"], {"user": {"empty_user": 2}})
                self.assertEqual(records[3]["status"], "copied_user")
                self.assertEqual(records[4]["status"], "meta_label")
                self.assertEqual(records[0]["pieces"][1]["context"], "meta")
                finalize_split(
                    replay_dir = replay_dir,
                    group = group,
                    split = "eval",
                    model_dir = str(self.model_dir),
                    jobs = 1,
                    drop_loops = True,
                )
                rows = {row["prompt_index"]: row for row in read_jsonl(path = replay_dir / group / "eval" / "samples.jsonl")}
                self.assertEqual({index: row["drop_reason"] for index, row in rows.items() if not row["keep"]}, {3: "copied_user", 4: "meta_label"})
                for index in (0, 1, 2):
                    row = rows[index]
                    self.assertEqual(row["meta_lang"], prompts[index]["meta_lang"])
                    self.assertEqual(row["meta_kind"], prompts[index]["meta_kind"])
                    self.assertEqual(row["meta_domain"], prompts[index]["meta_domain"])
                    system, user, reasoning, answer = EXAMPLES[index % len(EXAMPLES)]
                    self.assertEqual(row["system_text"], system)
                    self.assertEqual(row["user_text"], user)
                    # 保存的片段（不含元指令）拼起来就是 chat template 渲染完整三轮对话的 token（去掉结尾的换行）
                    sequence = []
                    for piece in row["pieces"]:
                        if "gen" in piece:
                            sequence += texts[index][("system", "user", "assistant").index(piece["gen"]) + 1][-1]
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
                    self.assertEqual(sequence[: len(SYSTEM_START)], SYSTEM_START)
                    self.assertEqual(row["pieces"][0], {"ids": SYSTEM_START})
                self.assertEqual(rows[2]["resampled"], {"user": {"empty_user": 2}})

    def test_finalize_checks_without_resampling(self):
        """生成时不重新采样（--resample-* 为 0）时，finalize 按同样的规则把空的用户消息整条丢掉。"""
        with tempfile.TemporaryDirectory() as tmp:
            replay_dir = Path(tmp) / "replay"
            write_meta_prompts(model_dir = self.model_dir, replay_dir = replay_dir, groups = ["free_sys_chat"], per_lang = 1)
            script, texts = self.scripts(thinking = False)
            engine = FakeEngine(script = script)
            prompts_path = replay_dir / "free_sys_chat" / "eval" / "prompts.jsonl"
            prompts = read_jsonl(path = prompts_path)[:3]
            prompts_path.write_text("".join(json.dumps(row, ensure_ascii = False) + "\n" for row in prompts), encoding = "utf-8")
            generator = fake_generator(replay_dir = replay_dir, engine = engine)
            self.assertTrue(generator.run_split(group = "free_sys_chat", split = "eval", target = 10**9, deadline = time.time() + 600))
            finalize_split(
                replay_dir = replay_dir,
                group = "free_sys_chat",
                split = "eval",
                model_dir = str(self.model_dir),
                jobs = 1,
                drop_loops = True,
            )
            rows = {row["prompt_index"]: row for row in read_jsonl(path = replay_dir / "free_sys_chat" / "eval" / "samples.jsonl")}
            self.assertEqual({index: row["drop_reason"] for index, row in rows.items() if not row["keep"]}, {1: "think_tag", 2: "empty_user"})


class QuotaTest(unittest.TestCase):
    def test_lang_quota(self):
        rows = [
            {"prompt_index": 3, "meta_lang": "en", "keep": True, "drop_reason": None},
            {"prompt_index": 0, "meta_lang": "en", "keep": True, "drop_reason": None},
            {"prompt_index": 1, "meta_lang": "en", "keep": False, "drop_reason": "loop"},
            {"prompt_index": 2, "meta_lang": "zh", "keep": True, "drop_reason": None},
            {"prompt_index": 4, "meta_lang": "en", "keep": True, "drop_reason": None},
        ]
        used = apply_lang_quota(rows = rows, quota = 2)
        self.assertEqual(used, {"en": 2, "zh": 1})
        reasons = {row["prompt_index"]: row["drop_reason"] for row in rows}
        self.assertEqual(reasons, {0: None, 1: "loop", 2: None, 3: None, 4: "lang_quota"})


if __name__ == "__main__":
    unittest.main()
