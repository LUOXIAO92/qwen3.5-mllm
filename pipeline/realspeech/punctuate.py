"""用 LLM 给没有标点的训练目标还原标点（只加标点，不改任何词；在 envs/runner 中运行，只用标准库）。

重新识别模式下训练目标取主 ASR 的识别结果。有的主 ASR 不输出标点（例如波斯语的 nemotron-farsi），这一步把训练目标
发给一个兼容 OpenAI 接口的 LLM 服务，让它加上标点，再把它加的标点按位置放回原文本：

1. 把原文本和 LLM 输出各自拆成「字母序列」（去掉空白、零宽非连接符 ZWNJ 和所有标点后剩下的字符）和标点位置
   （每个标点前面有几个字母）。
2. 两边字母序列相同时，LLM 加的标点按字母位置直接放回原文本（状态 ok）。LLM 顺手改了个别字母时（例如纠正拼写），
   用最长公共子序列对齐，只把落在两边一致片段上的标点放回（状态 aligned）；改动超过阈值时不采用（状态 failed）。
3. 闭合类标点（句号、逗号、问号、右引号等）只能放在原文本某个词的末尾，开启类标点（左引号、左括号）只能放在某个词
   的开头；落在词中间的标点说明 LLM 拆分或合并了词，整句不采用。

词的写法、空白和 ZWNJ 都取自原文本，所以采用的结果去掉标点后与原文本完全相同。

RECASE 中的语言（越南语：主 ASR 和交叉 ASR 都几乎只输出小写）同时还原大小写：比对字母时不分大小写，对齐上的字母
若只差大小写就取 LLM 的写法，其余（声调符号、拼写）仍取原文本，所以采用的结果转成小写、去掉标点后与原文本相同。

结果逐行追加到输出目录的 punctuation.jsonl（sample_id、lang、input、output、status、reason、llm_output），
可以中断后续跑：已有结果且输入相同的样本不再请求。manifest 步骤（pipeline/runner/report.py）生成清单时套用
状态为 ok 或 aligned 的结果，原文本存到 unpunctuated_target 字段。

用法::

    # 给一个真人语音输出目录的通过样本加标点（先跑过 manifest）
    python -m pipeline.realspeech.punctuate run --out-dir <输出目录> --langs fa --endpoint http://127.0.0.1:8081/v1
    # 评测：取带标点的参考文本（如 FLEURS 原文），去掉标点后还原，与参考比较
    python -m pipeline.realspeech.punctuate eval --input fleurs_dev.jsonl --lang fa --endpoint ... --output eval.json
"""

import argparse
import difflib
import json
import re
import threading
import time
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ZWNJ = "‌"

# 每种语言允许 LLM 加的标点：闭合类放在词尾，开启类放在词首；NORMALIZE 把 LLM 可能写成的拉丁形式换成该语言的写法。
LANG_MARKS = {
    "fa": {
        "closing": set(".،؛؟!:»)…"),
        "opening": set("«("),
        "normalize": {",": "،", ";": "؛", "?": "؟", "٬": "،"},
    },
    "vi": {
        "closing": set(".,;:?!…)”"),
        "opening": set("(“"),
        "normalize": {},
    },
    "ms": {
        "closing": set(".,;:?!…)”"),
        "opening": set("(“"),
        "normalize": {},
    },
    "fil": {
        "closing": set(".,;:?!…)”"),
        "opening": set("(“"),
        "normalize": {},
    },
}

LANG_MARKS["zh"] = {
    "closing": set("，。？！；：、…）”》"),
    "opening": set("（“《"),
    "normalize": {",": "，", ".": "。", "?": "？", "!": "！", ";": "；", ":": "："},
}

# 不以空格分词的语言：每个汉字（及假名）都是一个可以在前后放标点的位置，其余字母按空白连成词
SCAN = {"zh"}
CJK = re.compile(pattern = r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]")

# 同时还原大小写的语言（越南语的 ASR 几乎只输出小写；马来语、菲律宾语的 omniASR 只输出小写）
RECASE = {"vi", "ms", "fil"}

SYSTEM_PROMPTS = {
    "fa": (
        "You restore punctuation in Persian speech transcripts produced by a speech recognizer. "
        "Insert Persian punctuation marks (، . ؟ ! ؛ : « ») where a careful Persian editor would put them. "
        "Copy every word exactly as given: do not change, add, remove, reorder, split or join words, do not fix "
        "spelling even if a word looks wrong, do not convert digits, and keep every space and zero-width non-joiner "
        "as it is. The transcript is one segment of a longer recording and may start or stop in the middle of a "
        "sentence; end it with a period only when the segment ends a sentence. Output only the punctuated text."
    ),
    "vi": (
        "You restore punctuation and capitalization in Vietnamese speech transcripts produced by a speech recognizer, "
        "which writes everything in lower case. Capitalize the first word of each sentence and proper names (people, "
        "places, organizations, titles) following standard Vietnamese orthography, and insert punctuation marks "
        "(. , ? ! ; : …) where a careful Vietnamese editor would put them. Copy every word exactly as given: do not "
        "change, add, remove, reorder, split or join words, keep every diacritic and spelling even if a word looks "
        "wrong, and do not convert numbers. The transcript is one segment of a longer recording and may start or "
        "stop in the middle of a sentence; end it with a period only when the segment ends a sentence. Output only "
        "the corrected text."
    ),
    "ms": (
        "You restore punctuation and capitalization in Malay speech transcripts produced by a speech recognizer, "
        "which writes everything in lower case without punctuation. Capitalize the first word of each sentence and "
        "proper names (people, places, organizations, titles such as Yang Berhormat or Dewan Rakyat) following "
        "standard Malay orthography, and insert punctuation marks (. , ? ! ; :) where a careful Malay editor would put "
        "them. Copy every word exactly as given: do not change, add, remove, reorder, split or join words (do not add "
        "hyphens to reduplicated words), keep every spelling even if a word looks wrong, and do not convert numbers. "
        "The transcript is one segment of a longer recording and may start or stop in the middle of a sentence; end "
        "it with a period only when the segment ends a sentence. Output only the corrected text."
    ),
    "fil": (
        "You restore punctuation and capitalization in Filipino (Tagalog) speech transcripts produced by a speech "
        "recognizer, which writes everything in lower case without punctuation. The speech often mixes Tagalog and "
        "English; keep English words as they are and capitalize English proper names by English rules. Capitalize "
        "the first word of each sentence and proper names (people, places, organizations, titles) following standard "
        "Filipino orthography, and insert punctuation marks (. , ? ! ; :) where a careful Filipino editor would put "
        "them. Copy every word exactly as given: do not change, add, remove, reorder, split or join words, keep every "
        "spelling even if a word looks wrong, and do not convert numbers. The transcript is one segment of a longer "
        "recording and may start or stop in the middle of a sentence; end it with a period only when the segment ends "
        "a sentence. Output only the corrected text."
    ),
}

# 一次请求送多句时追加的说明：每句一行，以 [序号] 开头，回复也按序号逐行给出。
GROUP_INSTRUCTION = (
    " The input holds several separate segments, one per line, each starting with a label such as [1]. "
    "Punctuate each segment on its own and return every line with its label, in the same order, one line per segment."
)
LABELED_LINE = re.compile(pattern = r"^\s*\[(\d+)\]\s?(.*)$")

# aligned 状态允许 LLM 改动的字母数：不超过 max(2, 字母总数 × 3%)。
MAX_CHANGED_LETTERS = 2
MAX_CHANGED_RATIO = 0.03


def is_mark(char: str) -> bool:
    """Unicode 标点类别（P 开头）的字符都当作标点。"""
    return unicodedata.category(char).startswith("P")


def parse(text: str, lang: str) -> tuple:
    """把文本拆成字母序列和标点列表 [(前面的字母数, 标点), ...]；标点先按 NORMALIZE 换成该语言的写法。"""
    normalize = LANG_MARKS[lang]["normalize"]
    letters = []
    marks = []
    for char in text:
        if char.isspace() or char == ZWNJ:
            continue
        if is_mark(char = char) or char in normalize:
            marks.append((len(letters), normalize.get(char, char)))
        else:
            letters.append(char)
    return "".join(letters), marks


def fold(text: str) -> str:
    """逐字转小写；转换后长度会变的字符（如 İ）保持原样，保证字母位置一一对应。"""
    return "".join(char.lower() if len(char.lower()) == 1 else char for char in text)


def needs_punctuation(text: str, lang: str) -> bool:
    """文本的词尾没有任何闭合类标点时需要还原（数字里的小数点等词内标点不算）；RECASE 的语言没有任何大写字母时也需要。"""
    if lang in RECASE and not any(char.isupper() for char in text):
        return True
    closing = LANG_MARKS[lang]["closing"]
    ends = {end for _, end, _ in word_spans(text = text, lang = lang)}
    return not any(mark in closing and offset in ends for offset, mark in parse(text = text, lang = lang)[1])


def word_spans(text: str, lang: str) -> list:
    """按空白切词，返回每个词的 (起始字母数, 结束字母数, 词)。ZWNJ 和词内的标点不切分。

    SCAN 中的语言再把词拆开：每个汉字单独成一项，相连的其他字母成一项（词的部分只给出字母）。
    """
    spans = []
    count = 0
    for word in text.split():
        letters = parse(text = word, lang = lang)[0]
        if lang not in SCAN:
            spans.append((count, count + len(letters), word))
            count += len(letters)
            continue
        run = ""
        for char in letters:
            if CJK.match(char):
                if run:
                    spans.append((count, count + len(run), run))
                    count += len(run)
                    run = ""
                spans.append((count, count + 1, char))
                count += 1
            else:
                run += char
        if run:
            spans.append((count, count + len(run), run))
            count += len(run)
    return spans


def offset_map(source: str, output: str) -> tuple:
    """返回 (位置映射函数, 改动的字母数, 一致片段)：映射函数把 LLM 输出的字母位置换成原文本的字母位置。

    位置 k 指「第 k 个字母之前」。只有落在两边一致片段内部或边界上的位置能映射，其余返回 None。
    一致片段是 [(原文本起点, 输出起点, 长度)]。
    """
    if source == output:
        return (lambda k: k), 0, [(0, 0, len(source))]
    matcher = difflib.SequenceMatcher(a = source, b = output, autojunk = False)
    blocks = [block for block in matcher.get_matching_blocks() if block.size > 0]
    changed = len(source) + len(output) - 2 * sum(block.size for block in blocks)

    def mapped(k: int):
        for block in blocks:
            if block.b <= k <= block.b + block.size:
                return block.a + (k - block.b)
        return None

    return mapped, changed, [(block.a, block.b, block.size) for block in blocks]


def transfer(source: str, llm_output: str, lang: str) -> dict:
    """把 LLM 输出里的标点放回原文本，返回 {status, output, reason}。"""
    spec = LANG_MARKS[lang]
    source_letters, source_marks = parse(text = source, lang = lang)
    output_letters, output_marks = parse(text = llm_output.strip(), lang = lang)
    if not source_letters:
        return {"status": "failed", "output": None, "reason": "empty_input"}
    recase = lang in RECASE
    # 逐词分类模型先把输入转成小写：对齐一律不分大小写，不还原大小写的语言仍取原文本的写法
    key = fold
    mapped, changed, blocks = offset_map(source = key(source_letters), output = key(output_letters))
    if changed > max(MAX_CHANGED_LETTERS, int(len(source_letters) * MAX_CHANGED_RATIO)):
        return {"status": "failed", "output": None, "reason": "words_changed"}
    letters = list(source_letters)
    if recase:
        # 对齐上的字母只差大小写时取 LLM 的写法
        for a, b, size in blocks:
            for i in range(size):
                letters[a + i] = output_letters[b + i]
    spans = word_spans(text = source, lang = lang)
    ends = {end for _, end, _ in spans}
    starts = {start for start, _, _ in spans}
    existing = set(source_marks)
    before = {}
    after = {}
    for offset, mark in output_marks:
        if mark not in spec["closing"] and mark not in spec["opening"]:
            continue
        target = mapped(offset)
        if target is None:
            if changed:
                continue
            return {"status": "failed", "output": None, "reason": "unmapped_mark"}
        if (target, mark) in existing:
            continue
        if mark in spec["closing"]:
            if target not in ends or target == 0:
                return {"status": "failed", "output": None, "reason": "mark_inside_word"}
            after.setdefault(target, []).append(mark)
        else:
            if target not in starts or target == len(source_letters):
                return {"status": "failed", "output": None, "reason": "mark_inside_word"}
            before.setdefault(target, []).append(mark)
    if lang in SCAN:
        # 按原文本逐字重建，空格原样保留：标点紧贴在对应的字前后
        pieces = []
        count = 0
        for char in source:
            if char.isspace() or char == ZWNJ or is_mark(char = char) or char in spec["normalize"]:
                pieces.append(char)
                continue
            pieces.extend(before.get(count, []))
            pieces.append(letters[count] if recase else char)
            count += 1
            pieces.extend(after.get(count, []))
        result = "".join(pieces)
        if key(parse(text = result, lang = lang)[0]) != key(source_letters):
            return {"status": "failed", "output": None, "reason": "rebuild_mismatch"}
        return {"status": "aligned" if changed else "ok", "output": result, "reason": None}
    words = []
    for start, end, word in spans:
        if recase:
            chars, index = [], start
            for char in word:
                if char.isspace() or char == ZWNJ or is_mark(char = char) or char in spec["normalize"]:
                    chars.append(char)
                else:
                    chars.append(letters[index])
                    index += 1
            word = "".join(chars)
        words.append("".join(before.get(start, [])) + word + "".join(after.get(end, [])))
    result = " ".join(words)
    if key(parse(text = result, lang = lang)[0]) != key(source_letters):
        return {"status": "failed", "output": None, "reason": "rebuild_mismatch"}
    return {"status": "aligned" if changed else "ok", "output": result, "reason": None}


def complete(endpoint: str, model: str, system_prompt: str, text: str, timeout: float, extra_tokens: int = 0) -> str:
    """向兼容 OpenAI 接口的服务请求一次补全（温度 0，关闭思考），返回回复文本。

    输出只比输入多几个标点，max_tokens 先按每 2 个字符 1 个 token 加余量设定；被截断时放宽一次重新请求。
    """
    for max_tokens in (32 + extra_tokens + len(text) // 2, 64 + extra_tokens + 2 * len(text)):
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = urllib.request.Request(
            url = endpoint.rstrip("/") + "/chat/completions",
            data = json.dumps(obj = body).encode(encoding = "utf-8"),
            headers = {"Content-Type": "application/json"},
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout = timeout) as response:
                    choice = json.load(fp = response)["choices"][0]
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(5 * (attempt + 1))
        if choice.get("finish_reason") != "length":
            break
    return choice["message"]["content"] or ""


def parse_labeled(reply: str) -> dict:
    """把多句请求的回复拆成 {序号: 文本}；同一序号出现多次时整句作废（标为 None）。"""
    lines = {}
    for line in reply.splitlines():
        match = LABELED_LINE.match(line)
        if match:
            index = int(match.group(1))
            lines[index] = None if index in lines else match.group(2)
    return lines


def punctuate_all(
    items: list,
    lang: str,
    endpoint: str,
    model: str,
    concurrency: int,
    timeout: float,
    on_result,
    group: int = 1,
) -> None:
    """并发处理 [(key, text), ...]，每得到一条结果调用一次 on_result(key, text, llm_output, result)。

    group 大于 1 时每次请求送 group 句（按 items 的顺序分组，同一录音的相邻片段在同一组里），系统提示词只读一次；
    组内缺行、序号重复或校验不通过的句子再单独请求一次。结果的 request 字段记录采用的是哪种请求。
    """
    system_prompt = SYSTEM_PROMPTS[lang]

    def one(key: str, text: str) -> tuple:
        try:
            llm_output = complete(endpoint = endpoint, model = model, system_prompt = system_prompt, text = text, timeout = timeout)
        except OSError as error:
            return key, text, None, {"status": "error", "output": None, "reason": f"request: {error}", "request": "single"}
        result = transfer(source = text, llm_output = llm_output, lang = lang)
        result["request"] = "single"
        return key, text, llm_output, result

    def chunk_of(chunk: list) -> list:
        if len(chunk) == 1:
            return [one(*chunk[0])]
        prompt = "\n".join(f"[{index + 1}] {text}" for index, (_, text) in enumerate(chunk))
        try:
            reply = complete(
                endpoint = endpoint,
                model = model,
                system_prompt = system_prompt + GROUP_INSTRUCTION,
                text = prompt,
                timeout = timeout,
                extra_tokens = 4 * len(chunk),
            )
            lines = parse_labeled(reply = reply)
        except OSError:
            lines = {}
        results = []
        for index, (key, text) in enumerate(chunk):
            line = lines.get(index + 1)
            result = transfer(source = text, llm_output = line, lang = lang) if line else None
            if result and result["status"] != "failed":
                result["request"] = "group"
                results.append((key, text, line, result))
            else:
                results.append(one(key, text))
        return results

    chunks = [items[start:start + max(group, 1)] for start in range(0, len(items), max(group, 1))]
    with ThreadPoolExecutor(max_workers = concurrency) as pool:
        futures = [pool.submit(chunk_of, chunk) for chunk in chunks]
        for future in as_completed(futures):
            for result in future.result():
                on_result(*result)


def load_cache(path: Path) -> dict:
    """读取 punctuation.jsonl，同一 sample_id 取最后一行。"""
    cache = {}
    if path.exists():
        with open(file = path, mode = "r", encoding = "utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(s = line)
                    cache[row["sample_id"]] = row
    return cache


def run(args: argparse.Namespace) -> None:
    langs = set(args.langs.split(","))
    unknown = langs - set(LANG_MARKS)
    if unknown:
        raise SystemExit(f"不支持的语言：{sorted(unknown)}")
    cache_path = args.out_dir / "punctuation.jsonl"
    cache = load_cache(path = cache_path)
    todo = {lang: [] for lang in langs}
    with open(file = args.out_dir / "manifest.jsonl", mode = "r", encoding = "utf-8") as handle:
        for line in handle:
            row = json.loads(s = line)
            if row["lang"] not in langs:
                continue
            text = row.get("unpunctuated_target", row["original"])
            if not needs_punctuation(text = text, lang = row["lang"]):
                continue
            cached = cache.get(row["sample_id"])
            if cached and cached["input"] == text and (cached["status"] != "error"):
                continue
            todo[row["lang"]].append((row["sample_id"], text))
    if args.limit:
        todo = {lang: items[:args.limit] for lang, items in todo.items()}
    lock = threading.Lock()
    counts = {}
    started = time.time()
    with open(file = cache_path, mode = "a", encoding = "utf-8") as out:
        for lang, items in sorted(todo.items()):
            print(f"[punctuate] {lang}: {len(items)} 条待处理", flush = True)

            def on_result(key, text, llm_output, result, lang = lang):
                row = {"sample_id": key, "lang": lang, "input": text, "output": result["output"],
                       "status": result["status"], "reason": result["reason"], "llm_output": llm_output,
                       "request": result.get("request"), "model": args.model_name}
                with lock:
                    out.write(json.dumps(obj = row, ensure_ascii = False) + "\n")
                    out.flush()
                    counts[result["status"]] = counts.get(result["status"], 0) + 1
                    done = sum(counts.values())
                    if done % 200 == 0:
                        print(f"[punctuate] {done} 条，{done / (time.time() - started):.2f} 句/秒，{counts}", flush = True)

            punctuate_all(
                items = items,
                lang = lang,
                endpoint = args.endpoint,
                model = args.model_name,
                concurrency = args.concurrency,
                timeout = args.timeout,
                on_result = on_result,
                group = args.group,
            )
    elapsed = time.time() - started
    total = sum(counts.values())
    print(f"[punctuate] 完成 {total} 条，用时 {elapsed:.0f} 秒（{total / max(elapsed, 1e-9):.2f} 句/秒），{counts}", flush = True)


MARK_CLASSES = {".": "period", "،": "comma", ",": "comma", "؟": "question", "?": "question", "!": "exclamation",
                "؛": "semicolon", ";": "semicolon", ":": "colon", "«": "quote", "»": "quote", "“": "quote", "”": "quote"}


def slot_marks(text: str, lang: str) -> dict:
    """把带标点的文本换成 {(词序号, 前/后): 标点类别集合}；落在词中间的标点不计。"""
    spans = word_spans(text = strip_marks(text = text, lang = lang), lang = lang)
    starts = {start: index for index, (start, _, _) in enumerate(spans)}
    ends = {end: index for index, (_, end, _) in enumerate(spans)}
    slots = {}
    for offset, mark in parse(text = text, lang = lang)[1]:
        name = MARK_CLASSES.get(mark, "other")
        if mark in LANG_MARKS[lang]["opening"] and offset in starts:
            slots.setdefault((starts[offset], "before"), set()).add(name)
        elif offset in ends:
            slots.setdefault((ends[offset], "after"), set()).add(name)
    return slots


def strip_marks(text: str, lang: str) -> str:
    """去掉所有标点，合并多余空白（用于从参考文本造出无标点的输入）。"""
    normalize = LANG_MARKS[lang]["normalize"]
    kept = "".join(char for char in text if not (is_mark(char = char) or char in normalize))
    return " ".join(kept.split())


def final_mark(text: str, lang: str) -> str:
    """句末最后一个闭合类标点的类别（右引号、右括号跳过），没有时为 none。"""
    marks = parse(text = text, lang = lang)[1]
    letters = len(parse(text = text, lang = lang)[0])
    for offset, mark in reversed(marks):
        if offset != letters:
            break
        if mark in "»)":
            continue
        return MARK_CLASSES.get(mark, "other")
    return "none"


def score(references: list, predictions: list, lang: str) -> dict:
    """按「词后/词前位置 × 标点类别」统计精确率和召回率，并统计句末标点的正确率。"""
    tp, fp, fn = {}, {}, {}
    final_correct = 0
    for reference, prediction in zip(references, predictions):
        ref_slots = slot_marks(text = reference, lang = lang)
        pred_slots = slot_marks(text = prediction, lang = lang)
        for slot in set(ref_slots) | set(pred_slots):
            ref = ref_slots.get(slot, set())
            pred = pred_slots.get(slot, set())
            for name in ref & pred:
                tp[name] = tp.get(name, 0) + 1
            for name in pred - ref:
                fp[name] = fp.get(name, 0) + 1
            for name in ref - pred:
                fn[name] = fn.get(name, 0) + 1
        final_correct += final_mark(text = reference, lang = lang) == final_mark(text = prediction, lang = lang)
    table = {}
    for name in sorted(set(tp) | set(fp) | set(fn)):
        t, p, n = tp.get(name, 0), fp.get(name, 0), fn.get(name, 0)
        table[name] = {"reference": t + n, "predicted": t + p,
                       "precision": round(t / (t + p), 4) if t + p else None,
                       "recall": round(t / (t + n), 4) if t + n else None}
    t, p, n = sum(tp.values()), sum(fp.values()), sum(fn.values())
    table["all"] = {"reference": t + n, "predicted": t + p,
                    "precision": round(t / (t + p), 4) if t + p else None,
                    "recall": round(t / (t + n), 4) if t + n else None}
    result = {"marks": table, "final_accuracy": round(final_correct / max(len(references), 1), 4), "sentences": len(references)}
    if lang in RECASE:
        # 首字母大写的词：预测与参考逐词对应（预测由去掉标点、转小写的参考还原而来，词数相同）
        t = p = n = 0
        for reference, prediction in zip(references, predictions):
            ref_words = strip_marks(text = reference, lang = lang).split()
            pred_words = strip_marks(text = prediction, lang = lang).split()
            if len(ref_words) != len(pred_words):
                continue
            for a, b in zip(ref_words, pred_words):
                upper_ref, upper_pred = a[:1].isupper(), b[:1].isupper()
                t += upper_ref and upper_pred
                p += upper_pred and not upper_ref
                n += upper_ref and not upper_pred
        result["capitalized_words"] = {"reference": t + n, "predicted": t + p,
                                       "precision": round(t / (t + p), 4) if t + p else None,
                                       "recall": round(t / (t + n), 4) if t + n else None}
    return result


def evaluate(args: argparse.Namespace) -> None:
    references = {}
    with open(file = args.input, mode = "r", encoding = "utf-8") as handle:
        for line in handle:
            row = json.loads(s = line)
            if row["lang"] == args.lang:
                references[row["sample_id"]] = row["original"]
    if args.limit:
        references = dict(list(references.items())[:args.limit])
    plain = fold if args.lang in RECASE else (lambda text: text)
    items = [(key, plain(strip_marks(text = text, lang = args.lang))) for key, text in references.items()]
    results = {}
    started = time.time()
    punctuate_all(
        items = items,
        lang = args.lang,
        endpoint = args.endpoint,
        model = args.model_name,
        concurrency = args.concurrency,
        timeout = args.timeout,
        on_result = lambda key, text, llm_output, result: results.__setitem__(key, (text, llm_output, result)),
        group = args.group,
    )
    elapsed = time.time() - started
    keys = list(references)
    statuses = {}
    for key in keys:
        status = f"{results[key][2]['status']}/{results[key][2].get('request')}"
        statuses[status] = statuses.get(status, 0) + 1
    # 不采用的句子按「不加标点」计入，反映真正用于训练的结果
    predictions = [results[key][2]["output"] or results[key][0] for key in keys]
    # 简单做法：句末加句号；还原大小写的语言再把句首字母大写
    period_only = [(results[key][0][:1].upper() if args.lang in RECASE else results[key][0][:1]) + results[key][0][1:] + "."
                   for key in keys]
    report = {
        "lang": args.lang,
        "model": args.model_name,
        "group": args.group,
        "sentences": len(keys),
        "status": statuses,
        "seconds": round(elapsed, 1),
        "sentences_per_second": round(len(keys) / max(elapsed, 1e-9), 3),
        "llm": score(references = [references[key] for key in keys], predictions = predictions, lang = args.lang),
        "period_only": score(references = [references[key] for key in keys], predictions = period_only, lang = args.lang),
        "examples": [
            {"reference": references[key], "input": results[key][0], "llm_output": results[key][1],
             "status": results[key][2]["status"], "reason": results[key][2]["reason"], "output": results[key][2]["output"]}
            for key in keys
        ],
    }
    args.output.parent.mkdir(parents = True, exist_ok = True)
    args.output.write_text(json.dumps(obj = report, ensure_ascii = False, indent = 1), encoding = "utf-8")
    print(json.dumps(obj = {k: v for k, v in report.items() if k != "examples"}, ensure_ascii = False, indent = 1))


def main() -> None:
    parser = argparse.ArgumentParser(description = "用 LLM 给没有标点的训练目标还原标点")
    sub = parser.add_subparsers(dest = "command", required = True)
    for name in ("run", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--endpoint", default = "http://127.0.0.1:8081/v1", help = "兼容 OpenAI 接口的服务地址")
        p.add_argument("--model-name", default = "swift-1.5-iq3_s", help = "请求里的模型名，同时记入结果")
        p.add_argument("--concurrency", type = int, default = 4, help = "同时发出的请求数，与服务的并发槽数相同即可")
        p.add_argument("--timeout", type = float, default = 600)
        p.add_argument("--limit", type = int, default = 0, help = "只处理前若干条（试跑用）")
        p.add_argument("--group", type = int, default = 16, help = "每次请求送几句；1 为逐句请求")
    sub.choices["run"].add_argument("--out-dir", required = True, type = Path)
    sub.choices["run"].add_argument("--langs", required = True, help = "逗号分隔，如 fa")
    sub.choices["eval"].add_argument("--input", required = True, type = Path, help = "带标点参考文本的 jsonl（sample_id、lang、original）")
    sub.choices["eval"].add_argument("--lang", required = True)
    sub.choices["eval"].add_argument("--output", required = True, type = Path)
    args = parser.parse_args()
    run(args = args) if args.command == "run" else evaluate(args = args)


if __name__ == "__main__":
    main()
