"""各子命令启动时检查输入文件的格式（字段按 pipeline/CONVENTIONS.md），不合格时报出文件、行号和缺的字段。"""

from pathlib import Path

from .io_utils import read_jsonl

LANGS = {"zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it", "id", "ms", "th", "vi", "fil", "ar", "tr", "fa"}

# 文件类型 -> 必需字段
REQUIRED = {
    "entries": ["entry_id", "lang", "original"],
    "scripts": ["entry_id"],
    "voices": ["voice_id", "source_model", "ref_lang", "holdout"],
    "tasks": [
        "sample_id", "entry_id", "lang", "text", "original", "tts_model", "mode", "voice_id",
        "ref_lang", "cross_lingual", "seed",
    ],
    "synth": ["sample_id", "tts_model", "mode", "audio_path", "duration_s", "status"],
    "length": ["sample_id", "ok", "reason", "duration_s", "chars"],
    "asr_raw": ["sample_id", "asr_model", "asr_role", "hypothesis"],
    "asr_results": ["sample_id", "asr_model", "hypothesis", "match", "compare_detail", "asr_role"],
}


class InputError(Exception):
    """输入文件缺失或格式不符，需要先运行上一步或修正文件。"""


def load_checked(path: Path, kind: str, hint: str = "") -> list:
    """读取 JSONL 并检查必需字段与语言代码。"""
    path = Path(path)
    if not path.exists():
        raise InputError(f"找不到输入文件 {path}。{hint}")
    rows = read_jsonl(path = path)
    required = REQUIRED[kind]
    for number, row in enumerate(rows, start = 1):
        missing = [name for name in required if name not in row]
        if missing:
            raise InputError(f"{path} 第 {number} 行缺少字段 {missing}（{kind} 的格式见 pipeline/CONVENTIONS.md）")
        if "lang" in row and row["lang"] not in LANGS:
            raise InputError(f"{path} 第 {number} 行的语言代码 {row['lang']!r} 不在 {sorted(LANGS)} 中")
        if kind == "voices" and not (row.get("ref_audio") or row.get("builtin_name")):
            raise InputError(f"{path} 第 {number} 行的音色既没有 ref_audio 也没有 builtin_name")
    return rows
