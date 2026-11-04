"""生成内置音色参考库（实验 v1，任务 T07）。

做什么：为 Qwen3-TTS（9 个）、Supertonic-3（10 个）、VieNeu（23 个）的内置音色各合成一段约 8 到 12 秒的
参考朗读，再用 Qwen3-ASR 识别、用 pipeline.compare 与参考文本比对；不一致的音色换句子、换种子重试，
最多 --max-rounds 轮。
每一轮的合成结果和识别结果都保存在 --tmp-dir 下，中断后用同样的命令重跑即可从断点继续（已完成的合成、识别会直接读取）。通过核对的音色写入 voices.jsonl，并按固定随机种子挑出留出音色。

参考文本：从 T03 抽取的原文（<原文目录>/<lang>.jsonl）中选。只选估计时长 8 到 12 秒、字符全是文字和
常见标点（没有数字、缩写、括号、符号）的完整句子，不需要改写成读法。所用条目的 entry_id 写入
voices.jsonl 的 ref_entry_id，合成流水线据此把它们从训练文本中排除。

留出规则：Qwen3-TTS 留 1 个；Supertonic-3 留 2 个（一男一女）；VieNeu 留 3 个（兼顾性别）。

运行：本脚本只用标准库和 pipeline.compare，可用任意 Python 3.10+ 启动；各 TTS、ASR 模型通过
子进程在各自的虚拟环境中运行，GPU 命令用 flock 包起来排队。示例：

    python -m pipeline.voices.build_voices \
        --text-dir <T03 原文目录，如 data/T03/part1> \
        --out-dir <输出目录，如 data/T07> \
        --tmp-dir <临时目录，如 tmp/T07> \
        --gpu-lock <gpu.lock 路径> \
        --py-qwen3-tts <envs/tts-qwen3/bin/python> \
        --py-supertonic3 <envs/tts-supertonic3/bin/python> \
        --py-vieneu <envs/tts-vieneu/bin/python> \
        --py-asr <envs/asr-qwen3/bin/python>

输出：<out-dir>/voices.jsonl、<out-dir>/refs/<voice_id>.flac（16 kHz 单声道）、<out-dir>/report.json。
voices.jsonl 字段：voice_id、source_model、builtin_name、gender、ref_audio（相对 out-dir）、ref_text、
ref_lang、holdout，另有 ref_entry_id、ref_duration_s、ref_seed、ref_asr_hypothesis。
"""

import argparse
import json
import random
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

from pipeline.compare import compare

# ---------------------------------------------------------------- 音色表

# 名称 -> (母语, 性别)。性别来自 Qwen3-TTS 模型卡，未标明的填 None。
QWEN3_VOICES = {
    "Vivian": ("zh", "female"),
    "Serena": ("zh", "female"),
    "Uncle_Fu": ("zh", "male"),
    "Dylan": ("zh", "male"),
    "Eric": ("zh", "male"),
    "Ryan": ("en", "male"),
    "Aiden": ("en", "male"),
    "Ono_Anna": ("ja", "female"),
    "Sohee": ("ko", "female"),
}

# Supertonic-3：F 为女声，M 为男声。参考语言在它支持的目标语言里轮流分配。
SUPERTONIC_VOICES = ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]
SUPERTONIC_LANG_ORDER = ["ar", "de", "es", "fr", "id", "it", "pt", "ru", "tr", "ko"]
# 男女交错取语言，使每种性别都覆盖不同语言
SUPERTONIC_ASSIGN = dict(zip(
    ["F1", "M1", "F2", "M2", "F3", "M3", "F4", "M4", "F5", "M5"],
    SUPERTONIC_LANG_ORDER,
))

# VieNeu 23 个内置音色，名称与性别取自 vieneu 的 list_preset_voices()。
VIENEU_VOICES = {
    "Minh Đức": "male",
    "Phạm Tuyên": "male",
    "Thái Sơn": "male",
    "Xuân Vĩnh": "male",
    "Thanh Bình": "male",
    "Trúc Ly": "female",
    "Ngọc Linh": "female",
    "Đoan Trang": "female",
    "Mai Anh": "female",
    "Thục Đoan": "female",
    "Minh Triết": "male",
    "Thùy Dung": "female",
    "Quang Sơn": "male",
    "Ngọc Trân": "female",
    "Mỹ Duyên": "female",
    "Quỳnh Anh": "female",
    "Đức Trí": "male",
    "Kim Thanh": "female",
    "Ngọc Huyền": "female",
    "Adam": "male",
    "Mạnh Dũng": "male",
    "Minh Quân": "male",
    "Anh Khôi": "male",
}

HOLDOUT_SEED = 20260704
# 实测合成时长比估计时长短，下限放宽到 4.5 秒
MIN_REF_S = 4.5
MAX_REF_S = 14.0

# ---------------------------------------------------------------- 参考文本筛选

# 只允许文字、空白和常见句内、句末标点；数字、括号、引号以外的符号都不要
_BAD_CHARS = re.compile(r"[0-9０-９\(\)\[\]（）【】「」『』《》<>{}/\\@#%&*+=_~^|$€£¥°]")
_ALLOWED_PUNCT = set(",.?!;:'’-—、，。！？；：…")
_ABBREV = re.compile(r"[A-ZÀ-ÖØ-Þ]{2,}|\b\w{1,3}\.\s+\w|\.\w")
_SENT_END = re.compile(r"[.?!。！？…]$")


def is_clean_sentence(text: str) -> bool:
    """句子只含文字和常见标点，没有数字、缩写、括号、符号，且以句末标点收尾。"""
    if _BAD_CHARS.search(text) or _ABBREV.search(text) or not _SENT_END.search(text):
        return False
    for ch in text:
        category = unicodedata.category(ch)
        if category[0] in "LMZ" or ch in _ALLOWED_PUNCT:
            continue
        return False
    # 句中不应有多个句子，避免参考文本里混入缩写后的断句
    return len(re.findall(r"[.?!。！？]", text[:-1])) == 0


def candidate_pool(text_dir: Path, lang: str, rng: random.Random) -> list:
    """读取某语言的原文，筛出合格句子，用固定种子打乱顺序。"""
    pool = []
    with open(text_dir / f"{lang}.jsonl", encoding = "utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if 8.0 <= entry.get("est_duration_s", 0) <= 12.0 and is_clean_sentence(text = entry["original"]):
                pool.append(entry)
    pool.sort(key = lambda e: e["entry_id"])
    rng.shuffle(pool)
    # 估计时长 10 秒以上的排前面，使实际合成时长尽量接近 8 秒
    pool.sort(key = lambda e: e["est_duration_s"] < 10.0)
    return pool


# ---------------------------------------------------------------- 子进程

def run(cmd: list, cwd: Path, gpu_lock: str = None, env_extra: dict = None):
    """运行子进程；gpu_lock 非空时用 flock 排队。"""
    import os

    full = ["flock", gpu_lock] + cmd if gpu_lock else cmd
    env = {**os.environ, **(env_extra or {})}
    print("$", " ".join(full), file = sys.stderr)
    subprocess.run(full, cwd = str(cwd), env = env, check = True)


def read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, encoding = "utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list):
    with open(path, "w", encoding = "utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii = False) + "\n")


def ascii_slug(name: str) -> str:
    """去掉越南语声调与 đ，空格换下划线，用作 voice_id 和文件名。"""
    name = name.replace("Đ", "D").replace("đ", "d")
    name = unicodedata.normalize("NFKD", name)
    name = "".join(ch for ch in name if not unicodedata.combining(ch))
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


# ---------------------------------------------------------------- 主流程

def build_voice_list() -> list:
    voices = []
    for name, (lang, gender) in QWEN3_VOICES.items():
        voices.append(dict(
            voice_id = f"qwen3tts-{name}",
            source_model = "qwen3-tts",
            builtin_name = name,
            gender = gender,
            ref_lang = lang,
        ))
    for name in SUPERTONIC_VOICES:
        voices.append(dict(
            voice_id = f"supertonic3-{name}",
            source_model = "supertonic-3",
            builtin_name = name,
            gender = "female" if name.startswith("F") else "male",
            ref_lang = SUPERTONIC_ASSIGN[name],
        ))
    for name, gender in VIENEU_VOICES.items():
        voices.append(dict(
            voice_id = f"vieneu-{ascii_slug(name = name)}",
            source_model = "vieneu-v3-turbo",
            builtin_name = name,
            gender = gender,
            ref_lang = "vi",
        ))
    return voices


def synthesize_round(args, repo_root: Path, round_dir: Path, jobs_by_model: dict, seed: int):
    """合成一轮：每个模型一次子进程调用。返回 {sample_id: 结果行}。"""
    round_dir.mkdir(parents = True, exist_ok = True)
    commands = {
        "qwen3-tts": (args.py_qwen3_tts, "pipeline.tts.qwen3_tts", args.gpu_lock, ["--batch-size", "9", "--seed", str(seed)]),
        "supertonic-3": (args.py_supertonic3, "pipeline.tts.supertonic3", None, ["--batch-size", "16"]),
        "vieneu-v3-turbo": (args.py_vieneu, "pipeline.tts.vieneu", args.gpu_lock, ["--batch-size", "8"]),
    }
    results = {}
    for model, jobs in jobs_by_model.items():
        if not jobs:
            continue
        python, module, lock, extra = commands[model]
        model_dir = round_dir / model
        if (model_dir / "results.jsonl").exists():
            # 续跑：这一轮该模型已经合成完，直接读取结果
            for row in read_jsonl(path = model_dir / "results.jsonl"):
                results[row["sample_id"]] = row
            continue
        write_jsonl(path = round_dir / f"{model}.jobs.jsonl", rows = jobs)
        env_extra = {}
        if model == "supertonic-3" and args.supertonic_cache_dir:
            env_extra["SUPERTONIC_CACHE_DIR"] = args.supertonic_cache_dir
        run(
            cmd = [python, "-m", module, "--input", str(round_dir / f"{model}.jobs.jsonl"),
                   "--output-dir", str(model_dir), *extra],
            cwd = repo_root,
            gpu_lock = lock,
            env_extra = env_extra,
        )
        for row in read_jsonl(path = model_dir / "results.jsonl"):
            results[row["sample_id"]] = row
    return results


def recognize_round(args, repo_root: Path, round_dir: Path, rows: list) -> dict:
    """用 Qwen3-ASR 识别一轮中合成成功的片段，返回 {sample_id: hypothesis}。"""
    asr_in = [dict(sample_id = r["sample_id"], audio_path = r["audio_path"], lang = r["lang"]) for r in rows]
    if not asr_in:
        return {}
    write_jsonl(path = round_dir / "asr.in.jsonl", rows = asr_in)
    if (round_dir / "asr.out.jsonl").exists():
        # 续跑：这一轮已识别完
        return {r["sample_id"]: r["hypothesis"] for r in read_jsonl(path = round_dir / "asr.out.jsonl")}
    run(
        cmd = [args.py_asr, "-m", "pipeline.asr.qwen3_asr", "--input", str(round_dir / "asr.in.jsonl"),
               "--output", str(round_dir / "asr.out.jsonl"), "--batch-size", "16"],
        cwd = repo_root,
        gpu_lock = args.gpu_lock,
    )
    return {r["sample_id"]: r["hypothesis"] for r in read_jsonl(path = round_dir / "asr.out.jsonl")}


def main():
    parser = argparse.ArgumentParser(description = "生成内置音色参考库")
    parser.add_argument("--text-dir", required = True, help = "T03 原文目录（含 <lang>.jsonl）")
    parser.add_argument("--out-dir", required = True, help = "输出目录（voices.jsonl 与 refs/）")
    parser.add_argument("--tmp-dir", required = True, help = "各轮合成的临时目录")
    parser.add_argument("--gpu-lock", required = True, help = "GPU 锁文件")
    parser.add_argument("--py-qwen3-tts", required = True)
    parser.add_argument("--py-supertonic3", required = True)
    parser.add_argument("--py-vieneu", required = True)
    parser.add_argument("--py-asr", required = True, help = "Qwen3-ASR 环境的 python")
    parser.add_argument("--supertonic-cache-dir", default = None, help = "Supertonic 模型缓存目录，可不填")
    parser.add_argument("--max-rounds", type = int, default = 6)
    parser.add_argument("--seed", type = int, default = HOLDOUT_SEED, help = "选句子与选留出音色的随机种子")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    text_dir, out_dir, tmp_dir = Path(args.text_dir), Path(args.out_dir), Path(args.tmp_dir)
    (out_dir / "refs").mkdir(parents = True, exist_ok = True)
    tmp_dir.mkdir(parents = True, exist_ok = True)

    voices = build_voice_list()
    rng = random.Random(args.seed)
    pools = {lang: candidate_pool(text_dir = text_dir, lang = lang, rng = rng)
             for lang in sorted({v["ref_lang"] for v in voices})}
    cursor = {lang: 0 for lang in pools}
    used_entries = set()
    pending = {v["voice_id"]: v for v in voices}
    accepted = {}
    attempts = {v["voice_id"]: [] for v in voices}

    for round_index in range(args.max_rounds):
        if not pending:
            break
        round_dir = tmp_dir / f"round{round_index}"
        jobs_by_model = {"qwen3-tts": [], "supertonic-3": [], "vieneu-v3-turbo": []}
        chosen = {}
        for voice_id, voice in pending.items():
            lang = voice["ref_lang"]
            pool = pools[lang]
            while pool[cursor[lang]]["entry_id"] in used_entries:
                cursor[lang] += 1
            entry = pool[cursor[lang]]
            cursor[lang] += 1
            used_entries.add(entry["entry_id"])
            seed = args.seed + round_index
            sample_id = f"{voice_id}-r{round_index}"
            chosen[sample_id] = (voice, entry, seed)
            jobs_by_model[voice["source_model"]].append(dict(
                sample_id = sample_id,
                lang = lang,
                text = entry["original"],
                mode = "builtin",
                builtin_name = voice["builtin_name"],
                seed = seed,
            ))
        # Qwen3-TTS 的种子是整条命令的全局种子，由 --seed 传入
        results = synthesize_round(args = args, repo_root = repo_root, round_dir = round_dir, jobs_by_model = jobs_by_model, seed = args.seed + round_index)
        ok_rows = [r for r in results.values() if r.get("status") == "ok"]
        hypotheses = recognize_round(args = args, repo_root = repo_root, round_dir = round_dir, rows = ok_rows)

        for sample_id, (voice, entry, seed) in chosen.items():
            row = results.get(sample_id, {})
            record = dict(round = round_index, entry_id = entry["entry_id"], status = row.get("status", "missing"),
                          error = row.get("error"))
            if row.get("status") == "ok":
                duration = row["duration_s"]
                hypothesis = hypotheses.get(sample_id, "")
                outcome = compare(script = entry["original"], hypothesis = hypothesis, lang = voice["ref_lang"])
                record.update(duration_s = duration, hypothesis = hypothesis, match = outcome.match,
                              detail = outcome.detail)
                if outcome.match and MIN_REF_S <= duration <= MAX_REF_S:
                    dest = out_dir / "refs" / f"{voice['voice_id']}.flac"
                    shutil.copyfile(src = row["audio_path"], dst = dest)
                    accepted[voice["voice_id"]] = dict(
                        voice_id = voice["voice_id"],
                        source_model = voice["source_model"],
                        builtin_name = voice["builtin_name"],
                        gender = voice["gender"],
                        ref_audio = f"refs/{voice['voice_id']}.flac",
                        ref_text = entry["original"],
                        ref_lang = voice["ref_lang"],
                        holdout = False,
                        ref_entry_id = entry["entry_id"],
                        ref_duration_s = duration,
                        ref_seed = seed,
                        ref_asr_hypothesis = hypothesis,
                    )
                    pending.pop(voice["voice_id"], None)
            attempts[voice["voice_id"]].append(record)
        print(f"第 {round_index} 轮结束：通过 {len(accepted)}，待重试 {len(pending)}", file = sys.stderr)

    # 留出音色：在通过核对的音色中，用固定种子挑选
    pick = random.Random(args.seed)

    def choose(candidates: list, count: int) -> list:
        candidates = sorted(candidates, key = lambda v: v["voice_id"])
        return pick.sample(candidates, k = min(count, len(candidates)))

    by_model = {}
    for v in accepted.values():
        by_model.setdefault(v["source_model"], []).append(v)
    holdout = []
    holdout += choose(candidates = by_model.get("qwen3-tts", []), count = 1)
    st = by_model.get("supertonic-3", [])
    holdout += choose(candidates = [v for v in st if v["gender"] == "female"], count = 1)
    holdout += choose(candidates = [v for v in st if v["gender"] == "male"], count = 1)
    vi = by_model.get("vieneu-v3-turbo", [])
    first = choose(candidates = [v for v in vi if v["gender"] == "female"], count = 1)
    first += choose(candidates = [v for v in vi if v["gender"] == "male"], count = 1)
    holdout += first
    rest = [v for v in vi if v not in first]
    holdout += choose(candidates = rest, count = 1)
    for v in holdout:
        v["holdout"] = True

    order = {v["voice_id"]: i for i, v in enumerate(voices)}
    final = sorted(accepted.values(), key = lambda v: order[v["voice_id"]])
    write_jsonl(path = out_dir / "voices.jsonl", rows = final)
    report = dict(
        seed = args.seed,
        total = len(voices),
        accepted = len(final),
        failed = [v["voice_id"] for v in voices if v["voice_id"] not in accepted],
        holdout = [v["voice_id"] for v in holdout],
        attempts = attempts,
    )
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii = False, indent = 2), encoding = "utf-8")
    print(json.dumps({k: report[k] for k in ("seed", "total", "accepted", "failed", "holdout")}, ensure_ascii = False))


if __name__ == "__main__":
    main()
