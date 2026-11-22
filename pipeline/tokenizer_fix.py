"""修正 Qwen3.5 分词器的预分词规则，并在加载时检查（任务 #77）。

分词分两步：先按一条正则把文字切成小段（预分词），再在每一段里按词表合并成 token。Qwen3.5 自带的 tokenizer.json
里，这条正则的字母部分是 `[\\p{L}\\p{M}]+`：字母（\\p{L}）和紧跟在字母上的组合用符号（\\p{M}，例如泰语的元音与声调
符号、阿拉伯语和波斯语的元音符号）连成一段。原始目录的 tokenizer_config.json 写的分词器类是 Qwen2Tokenizer，
transformers 5.16.1 按这个类加载时，用 Qwen2Tokenizer 代码里自带的正则重建分词器，字母部分是 `\\p{L}+`，组合用符号
被切成单独的一段。于是同一段泰语、带元音符号的阿拉伯语分出更多 token，切法也和语言模型预训练时不同。融合模型目录
用 save_pretrained 保存，tokenizer.json 里写下的也是这条错误的正则。

修正只改模型目录中的两个文件：tokenizer_config.json 的 tokenizer_class 改为 Qwen3_5Tokenizer（transformers 中与
Qwen3.5 原始规则相同的类），tokenizer.json 的预分词器和解码器恢复成原始 Qwen3.5 的写法。词表、合并规则和特殊 token
都不变，已经训练的 projector 和语言模型权重照常加载。

    python -m pipeline.tokenizer_fix <模型目录> [<模型目录> ...] [--backup-suffix .before-77] [--verify]

检查函数 check_tokenizer 用一小段泰语加带元音符号的阿拉伯语分词，与原始 tokenizer.json 的结果比较，不一致就报错。
训练、评测、指令探针和回放文本加载分词器或 processor 之后都调用它，一次分词只需几十微秒。
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path

TOKENIZER_CLASS = "Qwen3_5Tokenizer"
# 原始 Qwen3.5 tokenizer.json 中的预分词器与解码器（0.8B、2B、4B 相同）
PRETOKENIZE_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+|\s+(?!\S)|\s+"
)
PRE_TOKENIZER = {
    "type": "Sequence",
    "pretokenizers": [
        {"type": "Split", "pattern": {"Regex": PRETOKENIZE_REGEX}, "behavior": "Isolated", "invert": False},
        {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
    ],
}
DECODER = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}
# 检查用的文字：泰语和带元音符号的阿拉伯语。原始 tokenizer.json 分成下面 11 个 token；把组合用符号切开的规则分成 25 个
PROBE_TEXT = "ภาษาไทยเป็นภาษาที่มีวรรณยุกต์ และ مُحَمَّدٌ رَسُولُ"
PROBE_IDS = [229936, 148423, 153871, 159124, 165249, 168921, 150015, 229208, 154876, 226377, 62422]
ISSUE = "#77"


def _pre_tokenizer_regex(tokenizer):
    """取出分词器预分词正则的写法，用于报错信息；取不到时返回 None。"""
    try:
        state = json.loads(s = tokenizer.backend_tokenizer.pre_tokenizer.__getstate__())
    except Exception:
        return None
    for item in state.get("pretokenizers", [state]):
        if item.get("type") == "Split":
            return item.get("pattern", {}).get("Regex")
    return None


def check_tokenizer(tokenizer, source = None):
    """确认分词器按原始 Qwen3.5 的规则分词，不一致时报错；也可以传入 processor（检查其中的 tokenizer）。

    source 是报错信息中的来源说明，通常是模型目录。返回检查过的分词器。
    """
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    ids = list(tokenizer.encode(text = PROBE_TEXT, add_special_tokens = False))
    if ids == PROBE_IDS:
        return tokenizer
    where = f"（{source}）" if source is not None else ""
    regex = _pre_tokenizer_regex(tokenizer = tokenizer)
    raise RuntimeError(
        f"分词器{where}的分词结果与原始 Qwen3.5 不一致（{ISSUE}）：检查文字分成 {len(ids)} 个 token，"
        f"原始 tokenizer.json 分成 {len(PROBE_IDS)} 个。加载到的分词器类是 {type(tokenizer).__name__}，"
        f"预分词正则是 {regex!r}；原始规则把组合用符号和字母连在一起（[\\p{{L}}\\p{{M}}]+）。"
        f"请先修正模型目录：python -m pipeline.tokenizer_fix <模型目录>（tokenizer_config.json 的 tokenizer_class "
        f"改为 {TOKENIZER_CLASS}，tokenizer.json 的预分词器恢复原始写法），再重新加载。"
    )


def _write_atomic(path, write):
    temporary = path.with_name(name = path.name + ".tmp")
    write(temporary)
    os.replace(src = temporary, dst = path)


def _fixed_config_text(text):
    """返回改好 tokenizer_class 的 tokenizer_config.json 文本；已经正确时返回 None。只改这一个值，其余写法不动。"""
    config = json.loads(s = text)
    if config.get("tokenizer_class") == TOKENIZER_CLASS:
        return None
    expected = {**config, "tokenizer_class": TOKENIZER_CLASS}
    new_text, count = re.subn(
        pattern = r'("tokenizer_class"\s*:\s*)"(?:[^"\\]|\\.)*"',
        repl = lambda match: match.group(1) + json.dumps(obj = TOKENIZER_CLASS),
        string = text,
        count = 1,
    )
    if count == 0 or json.loads(s = new_text) != expected:
        new_text = json.dumps(obj = expected, indent = 2, ensure_ascii = False) + "\n"
    return new_text


def fix_model_dir(model_dir, backup_suffix = None):
    """修正一个模型目录的分词器文件，可以重复运行；返回改动了的文件名列表（已经正确时为空）。

    backup_suffix 不为空时，改动前把原文件复制为「文件名 + backup_suffix」；同名备份已经存在时保留它，不覆盖。
    """
    from tokenizers import Tokenizer

    model_dir = Path(model_dir)
    config_path = model_dir / "tokenizer_config.json"
    tokenizer_path = model_dir / "tokenizer.json"
    changes = {}

    config_text = config_path.read_text(encoding = "utf-8")
    new_config_text = _fixed_config_text(text = config_text)
    if new_config_text is not None:
        changes[config_path] = lambda path: path.write_text(data = new_config_text, encoding = "utf-8")

    tokenizer_json = json.loads(s = tokenizer_path.read_text(encoding = "utf-8"))
    if tokenizer_json.get("pre_tokenizer") != PRE_TOKENIZER or tokenizer_json.get("decoder") != DECODER:
        tokenizer_json["pre_tokenizer"] = PRE_TOKENIZER
        tokenizer_json["decoder"] = DECODER
        # 用 tokenizers 库写回，与 transformers 保存的格式相同，文件中只有这两段改变
        fixed = Tokenizer.from_str(json = json.dumps(obj = tokenizer_json, ensure_ascii = False))
        ids = fixed.encode(sequence = PROBE_TEXT, add_special_tokens = False).ids
        if ids != PROBE_IDS:
            raise RuntimeError(f"{tokenizer_path} 换上原始预分词器后检查文字仍分成 {len(ids)} 个 token，词表可能不是 Qwen3.5 的")
        changes[tokenizer_path] = lambda path: fixed.save(path = str(path), pretty = True)

    for path, write in changes.items():
        if backup_suffix:
            backup = path.with_name(name = path.name + backup_suffix)
            if not backup.exists():
                shutil.copy2(src = path, dst = backup)
        _write_atomic(path = path, write = write)
    return [path.name for path in changes]


def verify_model_dir(model_dir):
    """用 transformers 加载模型目录的分词器（有 processor 配置时也加载 processor）并检查；返回加载到的分词器类名。"""
    from transformers import AutoProcessor, AutoTokenizer

    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = str(model_dir), trust_remote_code = True)
    check_tokenizer(tokenizer = tokenizer, source = model_dir)
    if (model_dir / "processor_config.json").exists():
        processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path = str(model_dir), trust_remote_code = True)
        check_tokenizer(tokenizer = processor, source = model_dir)
    return type(tokenizer).__name__


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dirs", nargs = "+", type = Path, help = "模型目录（含 tokenizer.json 与 tokenizer_config.json）")
    parser.add_argument("--backup-suffix", default = None, help = "改动前把原文件另存为「文件名 + 后缀」，例如 .before-77")
    parser.add_argument("--verify", action = "store_true", help = "修正后用 transformers 加载并检查")
    args = parser.parse_args()
    for model_dir in args.model_dirs:
        changed = fix_model_dir(model_dir = model_dir, backup_suffix = args.backup_suffix)
        state = f"已修正 {', '.join(changed)}" if changed else "已经正确，未改动"
        if args.verify:
            state += f"；transformers 加载为 {verify_model_dir(model_dir = model_dir)}，检查通过"
        print(f"{model_dir}：{state}", flush = True)


if __name__ == "__main__":
    main()
