"""汇总：判定每个样本是否通过，写出识别结果、通过清单、全部样本明细和统计。

输出只由 tasks.jsonl 和各阶段块文件决定，按 sample_id 排序，和运行方式（一次跑完、分片、续跑）无关。
"""

from pathlib import Path

from .io_utils import read_jsonl, write_json_atomic, write_jsonl_atomic
from .stages import Layout
from .validate import load_checked


def judge(cfg: dict, task: dict, synth: dict, length: dict, asr: dict) -> tuple:
    """返回 (状态, 原因)。状态为 pass、fail 或 pending（阶段尚未跑到）。"""
    record = synth.get(task["sample_id"])
    if record is None:
        return "pending", "synth_pending"
    if record["status"] != "ok":
        reason = "tts_unsupported" if record["status"] in ("skipped", "unsupported_lang") else "tts_failed"
        return "fail", reason
    check = length.get(task["sample_id"])
    if check is None:
        return "pending", "length_pending"
    if not check["ok"]:
        return "fail", check["reason"]
    lang_cfg = cfg.get("langs", {}).get(task["lang"], {})
    for role, label in (("primary", "asr"), ("cross", "cross")):
        if role == "cross" and not lang_cfg.get("cross_asr"):
            continue
        row = asr.get((task["sample_id"], role))
        if row is None:
            return "pending", f"{role}_pending"
        if not row["match"]:
            detail = str(row.get("compare_detail", ""))
            if detail.startswith("asr_error"):
                kind = "asr_error"
            elif detail.startswith("asr_empty"):
                kind = "asr_empty"
            else:
                kind = "mismatch"
            return "fail", kind if role == "primary" else f"cross_{kind}"
    return "pass", ""


def _rate_table(rows: list, key: str) -> dict:
    table = {}
    for row in rows:
        group = table.setdefault(str(row[key]), {"total": 0, "pass": 0})
        group["total"] += 1
        group["pass"] += row["status"] == "pass"
    for group in table.values():
        group["rate"] = round(group["pass"] / group["total"], 4)
    return dict(sorted(table.items()))


def _reasons(rows: list) -> dict:
    counts = {}
    for row in rows:
        if row["status"] == "fail":
            counts[row["reason"]] = counts.get(row["reason"], 0) + 1
    return dict(sorted(counts.items(), key = lambda pair: (-pair[1], pair[0])))


def build_stats(rows: list) -> dict:
    decided = [row for row in rows if row["status"] != "pending"]
    stats = {
        "total": len(rows),
        "pending": len(rows) - len(decided),
        "decided": len(decided),
        "pass": sum(row["status"] == "pass" for row in decided),
        "by_lang": _rate_table(rows = decided, key = "lang"),
        "by_tts_model": _rate_table(rows = decided, key = "tts_model"),
        "by_voice": _rate_table(rows = decided, key = "voice_id"),
        "by_cross_lingual": _rate_table(rows = decided, key = "cross_lingual"),
        "fail_reasons": _reasons(rows = decided),
        "fail_reasons_by_lang": {
            lang: _reasons(rows = [row for row in decided if row["lang"] == lang])
            for lang in sorted({row["lang"] for row in decided})
        },
    }
    stats["rate"] = round(stats["pass"] / stats["decided"], 4) if stats["decided"] else None
    return stats


def render_markdown(stats: dict) -> str:
    lines = [
        "# 合成核对统计",
        "",
        f"共 {stats['total']} 条，已判定 {stats['decided']} 条，待处理 {stats['pending']} 条，通过 {stats['pass']} 条，通过率 {stats['rate']}。",
        "",
    ]
    for title, key in (
        ("按语言", "by_lang"),
        ("按 TTS 模型", "by_tts_model"),
        ("按音色", "by_voice"),
        ("按是否跨语言", "by_cross_lingual"),
    ):
        lines += [f"## {title}", "", "| 分组 | 总数 | 通过 | 通过率 |", "| --- | --- | --- | --- |"]
        for name, group in stats[key].items():
            lines.append(f"| {name} | {group['total']} | {group['pass']} | {group['rate']:.1%} |")
        lines.append("")
    lines += ["## 失败原因", "", "| 原因 | 条数 |", "| --- | --- |"]
    for reason, count in stats["fail_reasons"].items():
        lines.append(f"| {reason} | {count} |")
    lines += ["", "## 各语言的失败原因", ""]
    for lang, reasons in stats["fail_reasons_by_lang"].items():
        if reasons:
            lines.append(f"- {lang}：" + "，".join(f"{name} {count}" for name, count in reasons.items()))
    return "\n".join(lines) + "\n"


def write_report(cfg: dict, layout: Layout) -> dict:
    """读取 tasks、synth_results、length_check、asr_results，写出 samples、manifest 和统计。"""
    tasks = load_checked(path = layout.tasks, kind = "tasks", hint = "先运行 plan 子命令")
    synth = {row["sample_id"]: row for row in (load_checked(path = layout.synth, kind = "synth") if layout.synth.exists() else [])}
    length = {row["sample_id"]: row for row in (load_checked(path = layout.length, kind = "length") if layout.length.exists() else [])}
    asr = {}
    if layout.asr_results.exists():
        for row in load_checked(path = layout.asr_results, kind = "asr_results"):
            asr[(row["sample_id"], row["asr_role"])] = row
    rows = []
    manifest = []
    for task in tasks:
        status, reason = judge(cfg = cfg, task = task, synth = synth, length = length, asr = asr)
        record = synth.get(task["sample_id"]) or {}
        row = {
            "sample_id": task["sample_id"],
            "entry_id": task["entry_id"],
            "lang": task["lang"],
            "tts_model": task["tts_model"],
            "mode": task["mode"],
            "voice_id": task["voice_id"],
            "ref_lang": task["ref_lang"],
            "cross_lingual": task["cross_lingual"],
            "holdout": task["holdout"],
            "seed": task["seed"],
            "audio_path": record.get("audio_path"),
            "duration_s": record.get("duration_s"),
            "tts_status": record.get("status"),
            "status": status,
            "reason": reason,
        }
        rows.append(row)
        if status == "pass":
            primary = asr[(task["sample_id"], "primary")]
            cross = asr.get((task["sample_id"], "cross"))
            manifest.append(
                {
                    "sample_id": task["sample_id"],
                    "audio_path": record["audio_path"],
                    "duration_s": record["duration_s"],
                    "lang": task["lang"],
                    "original": task["original"],
                    "script": task["text"],
                    "source": task["source"],
                    "source_license": task["source_license"],
                    "tts_model": task["tts_model"],
                    "voice_id": task["voice_id"],
                    "holdout": task["holdout"],
                    "cross_lingual": task["cross_lingual"],
                    "asr_model": primary["asr_model"],
                    "cross_asr_model": cross["asr_model"] if cross else None,
                    "entry_id": task["entry_id"],
                    "mode": task["mode"],
                    "seed": task["seed"],
                }
            )
    write_jsonl_atomic(path = layout.out / "samples.jsonl", rows = rows)
    write_jsonl_atomic(path = layout.out / "manifest.jsonl", rows = manifest)
    stats = build_stats(rows = rows)
    write_json_atomic(path = layout.out / "stats.json", obj = stats)
    (layout.out / "stats.md").write_text(render_markdown(stats = stats), encoding = "utf-8")
    return stats
