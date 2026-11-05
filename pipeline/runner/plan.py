"""生成合成任务：为每条文本确定性地抽取 TTS 模型和音色。

抽取只依赖「配置种子 + 数据划分 + 条目 ID」，和条目的处理顺序、分片、是否续跑都无关，
所以同一份输入和配置在任何时候重新生成，得到的任务完全相同。
"""

import hashlib
import random
import re
from pathlib import Path

from .io_utils import read_jsonl
from .validate import load_checked


def norm_name(name: str) -> str:
    return re.sub(pattern = r"[^a-z0-9]", repl = "", string = str(name).lower())


def load_entries(
    paths: list,
    langs: list | None,
    limit_per_lang: int | None,
    fallback_original: bool,
    scripts: list | None = None,
) -> list:
    """读取文本条目。path 可以是 JSONL 文件，也可以是含 <lang>.jsonl 的目录。

    scripts 是单独存放朗读稿的 JSONL（含 entry_id、script、tn_status），按 entry_id 合并进条目；
    tn_status 为 dropped 的条目不合成。
    """
    files = []
    for raw in paths:
        path = Path(raw)
        files.extend(sorted(path.glob("*.jsonl")) if path.is_dir() else [path])
    script_rows = {}
    for raw in scripts or []:
        for row in load_checked(path = Path(raw), kind = "scripts"):
            script_rows[row["entry_id"]] = row
    entries = {}
    for file in files:
        for row in load_checked(path = file, kind = "entries"):
            if row["entry_id"] in script_rows:
                extra = script_rows[row["entry_id"]]
                row = {**row, "script": extra.get("script"), "tn_status": extra.get("tn_status", row.get("tn_status"))}
            if langs and row["lang"] not in langs:
                continue
            if row.get("tn_status") == "dropped":
                continue
            script = row.get("script")
            if not script and fallback_original:
                script = row.get("original")
            if not script or not script.strip():
                continue
            entries[row["entry_id"]] = {**row, "script": script}
    ordered = [entries[key] for key in sorted(entries)]
    if limit_per_lang:
        counts = {}
        kept = []
        for row in ordered:
            counts[row["lang"]] = counts.get(row["lang"], 0) + 1
            if counts[row["lang"]] <= limit_per_lang:
                kept.append(row)
        ordered = kept
    return ordered


def load_voices(path: Path) -> list:
    path = Path(path)
    voices = load_checked(path = path, kind = "voices")
    for voice in voices:
        ref = voice.get("ref_audio")
        if ref and not Path(ref).is_absolute():
            voice["ref_audio"] = str((path.parent / ref).resolve())
    return voices


def usable_voices(cfg: dict, model: str, lang: str, voices: list) -> list:
    """能为 model 合成 lang 的音色，附带合成模式。返回 [(voice, mode)]。"""
    spec = cfg["tts_models"][model]
    if lang not in spec.get("langs", []):
        return []
    names = {norm_name(name = model)} | {norm_name(name = alias) for alias in spec.get("aliases", [])}
    found = []
    for voice in voices:
        own = norm_name(name = voice.get("source_model", "")) in names
        if own and voice.get("builtin_name") and lang in spec.get("builtin_langs", []):
            found.append((voice, "builtin"))
            continue
        if not spec.get("clone", False) or not voice.get("ref_audio"):
            continue
        if spec.get("needs_ref_text", False) and not voice.get("ref_text"):
            continue
        found.append((voice, "clone"))
    return found


def build_tasks(cfg: dict, entries: list, voices: list, split: str) -> tuple:
    """返回 (任务列表, 统计信息)。split 为 train（只用非留出音色）或 test（只用留出音色）。"""
    if split not in ("train", "test"):
        raise ValueError("split 只能是 train 或 test")
    want_holdout = split == "test"
    pool = [voice for voice in voices if bool(voice.get("holdout", False)) == want_holdout]
    excluded_ids = {voice["ref_entry_id"] for voice in voices if voice.get("ref_entry_id")}
    seed = cfg.get("seed", 0)
    tasks = []
    info = {"excluded_ref_entries": 0, "no_lang_config": 0, "no_voice": 0}
    cache = {}
    for entry in entries:
        lang = entry["lang"]
        if entry["entry_id"] in excluded_ids:
            info["excluded_ref_entries"] += 1
            continue
        weights = cfg.get("langs", {}).get(lang, {}).get("tts", {})
        if not weights:
            info["no_lang_config"] += 1
            continue
        candidates = []
        for model in sorted(weights):
            key = (model, lang)
            if key not in cache:
                cache[key] = sorted(
                    usable_voices(cfg = cfg, model = model, lang = lang, voices = pool),
                    key = lambda pair: pair[0]["voice_id"],
                )
            if cache[key] and weights[model] > 0:
                candidates.append(model)
        if not candidates:
            info["no_voice"] += 1
            continue
        rng = random.Random(f"{seed}:{split}:{entry['entry_id']}")
        model = rng.choices(population = candidates, weights = [weights[name] for name in candidates], k = 1)[0]
        usable = cache[(model, lang)]
        prefer = cfg.get("langs", {}).get(lang, {}).get("prefer_builtin", cfg.get("prefer_builtin", False))
        if prefer and any(mode == "builtin" for _, mode in usable):
            usable = [pair for pair in usable if pair[1] == "builtin"]
        voice, mode = rng.choice(usable)
        sample_id = f"{split}-{entry['entry_id']}"
        task = {
            "sample_id": sample_id,
            "entry_id": entry["entry_id"],
            "lang": lang,
            "text": entry["script"],
            "original": entry.get("original", entry["script"]),
            "source": entry.get("source"),
            "source_license": entry.get("source_license"),
            "split": split,
            "tts_model": model,
            "mode": mode,
            "voice_id": voice["voice_id"],
            "holdout": bool(voice.get("holdout", False)),
            "ref_lang": voice.get("ref_lang"),
            "cross_lingual": voice.get("ref_lang") != lang,
            "seed": int(hashlib.sha1(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()[:8], 16) % (2**31),
        }
        if mode == "builtin":
            task["builtin_name"] = voice["builtin_name"]
        else:
            task["ref_audio"] = voice["ref_audio"]
            task["ref_text"] = voice.get("ref_text")
        tasks.append(task)
    tasks.sort(key = lambda row: row["sample_id"])
    return tasks, info


def in_shard(sample_id: str, shard: tuple) -> bool:
    index, count = shard
    return int(hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:8], 16) % count == index
