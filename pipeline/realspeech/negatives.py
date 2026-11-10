"""把无人声片段切成「空目标」训练集、留出的评测集和混音背景声列表（实验 v2，任务 N1，#47；在 envs/vad 中运行，只用 CPU）。

输入是 ``pipeline.realspeech.nonspeech`` 筛出的 negatives.jsonl（FSD50K 与 MUSAN 的无人声录音）。三份输出：

- ``train/``：训练用空目标样本。目录内有 manifest.jsonl 和 audio/（16 kHz 单声道 FLAC），字段与真人语音训练清单一致
  （audio_path 相对该目录，original 和 script 为空字符串，lang 为 ``zxx``，即 ISO 639 的「无语言内容」），
  可直接交给 ``pipeline.runner.merge_manifests``。
- ``eval/``：留出的评测集，格式相同，约 ``--eval-size`` 段，覆盖各类声音。
- ``background.jsonl``：训练时实时混进语音样本的背景声列表，每行指向 ``train/`` 内的一段音频
  （audio_path 相对输出根目录），只用训练一侧的录音。

做法：

1. 一个「原始录音」是 FSD50K 的一个 clip，或 MUSAN 的一个 wav 文件。同一个原始录音切出的片段只会出现在训练、评测之一。
2. 切段：不到 ``--min-s`` 的录音丢弃；不超过 ``--max-s`` 的录音整段保留；更长的录音依次切成随机长度
   （均匀分布于 ``--cut-low`` 到 ``--cut-high`` 秒）的不重叠片段，不足 ``--min-s`` 的尾巴丢弃，每个录音最多留下
   ``--max-per-recording``（评测集为 ``--max-per-recording-eval``）段。
3. 声音类别：把 FSD50K 的类别标签（含上位类）按固定优先级归并成十来个大类，MUSAN 噪声为 noise、音乐为 music。
4. 评测集：候选是 FSD50K 官方 eval 部分的录音，加上按哈希抽出约 ``--musan-eval-percent`` % 的 MUSAN 文件。每个大类的名额
   与该类可用片段数的平方根成正比，类内随机选录音直到名额用完。评测集涉及的 FSD50K 上传者名下其余录音也不进训练集
   （FSD50K 的 dev 与 eval 本来就没有共同上传者），防止同一人的相近录音泄漏。
5. 每段记录来源、每段许可（FSD50K 取自 clips_info 里该 clip 的许可）、原始录音编号、起止时间、标签和类别。

用法::

    python -m pipeline.realspeech.negatives --negatives <negatives.jsonl> --fsd50k <FSD50k 目录> --out <输出根目录> --workers 16
"""

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SAMPLING_RATE = 16000
LANG = "zxx"

# 按优先级匹配：一个录音取第一个命中的大类；都不命中为 other
CATEGORY_MARKERS = [
    ("music", {"Music", "Musical_instrument", "music"}),
    ("noise", {"noise"}),
    ("animal", {"Animal"}),
    ("vehicle", {"Vehicle", "Engine"}),
    ("water_weather", {"Water", "Liquid", "Rain", "Thunderstorm", "Thunder", "Wind", "Fire", "Ocean", "Stream"}),
    ("human_body", {
        "Respiratory_sounds", "Hands", "Walk_and_footsteps", "Run", "Chewing_and_mastication", "Fart",
        "Burping_and_eructation", "Human_group_actions", "Finger_snapping", "Gasp", "Cough", "Sneeze", "Breathing",
        "Clapping", "Applause",
    }),
    ("alarm_signal", {"Alarm", "Bell", "Siren", "Telephone", "Ringtone", "Doorbell", "Vehicle_horn_and_car_horn_and_honking"}),
    ("explosion_impact", {"Explosion", "Gunshot_and_gunfire", "Fireworks", "Slam", "Boom"}),
    ("mechanical", {"Mechanisms", "Tools", "Printer", "Typewriter", "Computer_keyboard", "Typing"}),
    ("domestic", {"Domestic_sounds_and_home_sounds"}),
]
LICENSE_SHORT = {
    "http://creativecommons.org/licenses/by/3.0/": "CC-BY-3.0",
    "http://creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0",
    "http://creativecommons.org/licenses/by-nc/3.0/": "CC-BY-NC-3.0",
    "http://creativecommons.org/licenses/sampling+/1.0/": "CC-Sampling+-1.0",
}


def categorize(labels: list) -> str:
    label_set = set(labels)
    for name, markers in CATEGORY_MARKERS:
        if label_set & markers:
            return name
    return "other"


def recording_id(row: dict) -> str:
    """FSD50K 取文件名里的编号，MUSAN 取 ``musan-<子集>-<文件名>``。"""
    path = Path(row["path"])
    if row["source"].startswith("FSD50K"):
        return f"fsd50k-{path.stem}"
    return "musan-" + row["source"].split("/", maxsplit = 1)[1].replace("/", "-") + f"-{path.stem}"


def load_recordings(negatives: Path, fsd50k: Path) -> list:
    """读 negatives.jsonl，补上 recording_id、category、每段许可和上传者；跳过出错的行。"""
    info = {}
    for split in ("dev", "eval"):
        with open(file = fsd50k / "metadata" / f"{split}_clips_info_FSD50K.json", mode = "r", encoding = "utf-8") as handle:
            info[split] = json.load(fp = handle)
    recordings = []
    with open(file = negatives, mode = "r", encoding = "utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(s = line)
            if "error" in row or row.get("label_excluded"):
                continue
            row["recording_id"] = recording_id(row = row)
            row["category"] = categorize(labels = row["labels"])
            if row["source"].startswith("FSD50K"):
                split = row["source"].split("/")[1]
                clip = info[split][Path(row["path"]).stem]
                row["license_url"] = clip["license"]
                row["license"] = LICENSE_SHORT[clip["license"]]
                row["uploader"] = clip["uploader"]
                row["fsd_split"] = split
            else:
                row["license_url"] = None
                row["license"] = "MUSAN (per-file CC-BY 4.0 or public domain, see MUSAN LICENSE)"
                row["uploader"] = None
                row["fsd_split"] = None
            recordings.append(row)
    return recordings


def plan_segments(
    duration_s: float,
    rng: random.Random,
    min_s: float = 2.0,
    max_s: float = 30.0,
    cut_low: float = 5.0,
    cut_high: float = 25.0,
    max_per_recording: int = 12,
) -> list:
    """返回 [(起点秒, 终点秒), ...]。"""
    if duration_s < min_s:
        return []
    if duration_s <= max_s:
        return [(0.0, duration_s)]
    pieces = []
    start = 0.0
    while duration_s - start >= min_s:
        length = min(rng.uniform(cut_low, cut_high), duration_s - start)
        pieces.append((round(start, 3), round(start + length, 3)))
        start += length
    if len(pieces) > max_per_recording:
        pieces = sorted(rng.sample(population = pieces, k = max_per_recording))
    return pieces


def musan_in_eval_pool(rec_id: str, seed: int, percent: float) -> bool:
    digest = hashlib.sha1(f"{seed}:{rec_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % 10000 < percent * 100


def eval_quotas(available: dict, total: int) -> dict:
    """名额与各类可用片段数的平方根成正比，不超过可用数；某类不够时把差额按同样比例分给其余类。"""
    quotas = {name: 0 for name in available}
    open_names = {name for name, count in available.items() if count > 0}
    remaining = total
    while remaining > 0 and open_names:
        weights = {name: math.sqrt(available[name]) for name in open_names}
        weight_sum = sum(weights.values())
        capped = None
        for name in sorted(open_names):
            if quotas[name] + remaining * weights[name] / weight_sum >= available[name]:
                capped = name
                break
        if capped is None:
            for name in open_names:
                quotas[name] += int(remaining * weights[name] / weight_sum)
            break
        remaining -= available[capped] - quotas[capped]
        quotas[capped] = available[capped]
        open_names.discard(capped)
    return quotas


def split_recordings(recordings: list, plans: dict, plans_eval: dict, eval_size: int, musan_eval_percent: float, seed: int) -> tuple:
    """返回 (评测片段 [(录音, 起, 止)], 训练片段 [(录音, 起, 止)])，两边的原始录音互不相交。"""
    pool = defaultdict(list)
    for rec in recordings:
        in_pool = rec["fsd_split"] == "eval" or (
            rec["fsd_split"] is None and musan_in_eval_pool(rec_id = rec["recording_id"], seed = seed, percent = musan_eval_percent)
        )
        if in_pool and plans_eval[rec["recording_id"]]:
            pool[rec["category"]].append(rec)
    available = {name: sum(len(plans_eval[rec["recording_id"]]) for rec in recs) for name, recs in pool.items()}
    quotas = eval_quotas(available = available, total = eval_size)
    rng = random.Random(seed)
    eval_segments = []
    eval_ids = set()
    for name in sorted(pool):
        recs = sorted(pool[name], key = lambda rec: rec["recording_id"])
        rng.shuffle(recs)
        need = quotas[name]
        for rec in recs:
            if need <= 0:
                break
            pieces = plans_eval[rec["recording_id"]][:need]
            eval_segments.extend((rec, start, end) for start, end in pieces)
            eval_ids.add(rec["recording_id"])
            need -= len(pieces)
    eval_uploaders = {rec["uploader"] for rec in recordings if rec["recording_id"] in eval_ids and rec["uploader"]}
    train_segments = []
    for rec in recordings:
        if rec["recording_id"] in eval_ids or (rec["uploader"] and rec["uploader"] in eval_uploaders):
            continue
        train_segments.extend((rec, start, end) for start, end in plans[rec["recording_id"]])
    return eval_segments, train_segments


def cut_recording(job: dict) -> list:
    """解码一个录音，按计划切段写成 FLAC，返回每段的 (sample_id, 实际时长)。"""
    import torch
    from torchcodec.decoders import AudioDecoder
    from torchcodec.encoders import AudioEncoder

    torch.set_num_threads(1)
    wave = AudioDecoder(source = job["path"], sample_rate = SAMPLING_RATE, num_channels = 1).get_all_samples().data
    done = []
    for sample_id, start, end in job["segments"]:
        piece = wave[:, int(round(start * SAMPLING_RATE)):int(round(end * SAMPLING_RATE))]
        if piece.shape[1] < SAMPLING_RATE:
            continue
        AudioEncoder(samples = piece, sample_rate = SAMPLING_RATE).to_file(dest = str(Path(job["audio_dir"]) / f"{sample_id}.flac"))
        done.append((sample_id, round(piece.shape[1] / SAMPLING_RATE, 3)))
    return done


def manifest_row(rec: dict, sample_id: str, start: float, duration: float, holdout: bool) -> dict:
    return {
        "sample_id": sample_id,
        "audio_path": f"audio/{sample_id}.flac",
        "duration_s": duration,
        "lang": LANG,
        "original": "",
        "script": "",
        "source": rec["source"],
        "source_license": rec["license"],
        "tts_model": "nonspeech",
        "voice_id": "",
        "holdout": holdout,
        "cross_lingual": False,
        "asr_model": None,
        "cross_asr_model": None,
        "entry_id": sample_id,
        "mode": "nonspeech",
        "seed": 0,
        "recording_id": rec["recording_id"],
        "category": rec["category"],
        "labels": rec["labels"],
        "license_url": rec["license_url"],
        "uploader": rec["uploader"],
        "start_s": start,
    }


def write_split(name: str, segments: list, out_root: Path, holdout: bool, workers: int) -> list:
    directory = out_root / name
    audio_dir = directory / "audio"
    audio_dir.mkdir(parents = True, exist_ok = True)
    paths = {}
    by_recording = defaultdict(list)
    meta = {}
    for rec, start, end in segments:
        sample_id = f"nonspeech-{rec['recording_id']}-{int(round(start * 1000)):07d}"
        paths[rec["recording_id"]] = rec["path"]
        by_recording[rec["recording_id"]].append((sample_id, start, end))
        meta[sample_id] = (rec, start)
    jobs = [{"path": paths[rid], "audio_dir": str(audio_dir), "segments": segs} for rid, segs in by_recording.items()]
    rows = []
    with ProcessPoolExecutor(max_workers = workers) as pool:
        for index, done in enumerate(pool.map(cut_recording, jobs, chunksize = 16), start = 1):
            for sample_id, duration in done:
                rec, start = meta[sample_id]
                rows.append(manifest_row(rec = rec, sample_id = sample_id, start = start, duration = duration, holdout = holdout))
            if index % 2000 == 0:
                print(f"[negatives] {name}: {index}/{len(jobs)} 个录音", flush = True)
    rows.sort(key = lambda row: row["sample_id"])
    with open(file = directory / "manifest.jsonl", mode = "w", encoding = "utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(obj = row, ensure_ascii = False) + "\n")
    return rows


def background_list(train_rows: list, min_s: float) -> list:
    return [
        {
            "audio_path": f"train/{row['audio_path']}",
            "duration_s": row["duration_s"],
            "category": row["category"],
            "recording_id": row["recording_id"],
            "source": row["source"],
            "source_license": row["source_license"],
        }
        for row in train_rows
        if row["duration_s"] >= min_s
    ]


def summarize(rows: list) -> dict:
    hours = defaultdict(float)
    count = Counter()
    for row in rows:
        hours[row["category"]] += row["duration_s"] / 3600
        count[row["category"]] += 1
    return {
        "segments": len(rows),
        "hours": round(sum(hours.values()), 2),
        "by_category": {name: [count[name], round(hours[name], 2)] for name in sorted(count)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description = "切分无人声片段：空目标训练集、评测集、背景声列表")
    parser.add_argument("--negatives", required = True, type = Path)
    parser.add_argument("--fsd50k", required = True, type = Path)
    parser.add_argument("--out", required = True, type = Path)
    parser.add_argument("--workers", type = int, default = 16)
    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument("--eval-size", type = int, default = 1000)
    parser.add_argument("--musan-eval-percent", type = float, default = 20.0)
    parser.add_argument("--min-s", type = float, default = 1.0)
    parser.add_argument("--max-s", type = float, default = 30.0)
    parser.add_argument("--cut-low", type = float, default = 5.0)
    parser.add_argument("--cut-high", type = float, default = 25.0)
    parser.add_argument("--max-per-recording", type = int, default = 12)
    parser.add_argument("--max-per-recording-eval", type = int, default = 4)
    parser.add_argument("--background-min-s", type = float, default = 2.0)
    args = parser.parse_args()

    recordings = load_recordings(negatives = args.negatives, fsd50k = args.fsd50k)
    plans = {}
    plans_eval = {}
    for rec in recordings:
        rec_id = rec["recording_id"]
        plans[rec_id] = plan_segments(
            duration_s = rec["duration_s"],
            rng = random.Random(f"{args.seed}:{rec_id}"),
            min_s = args.min_s,
            max_s = args.max_s,
            cut_low = args.cut_low,
            cut_high = args.cut_high,
            max_per_recording = args.max_per_recording,
        )
        plans_eval[rec_id] = plans[rec_id][:args.max_per_recording_eval]
    eval_segments, train_segments = split_recordings(
        recordings = recordings,
        plans = plans,
        plans_eval = plans_eval,
        eval_size = args.eval_size,
        musan_eval_percent = args.musan_eval_percent,
        seed = args.seed,
    )
    print(f"[negatives] 评测 {len(eval_segments)} 段，训练 {len(train_segments)} 段", flush = True)
    eval_rows = write_split(name = "eval", segments = eval_segments, out_root = args.out, holdout = True, workers = args.workers)
    train_rows = write_split(name = "train", segments = train_segments, out_root = args.out, holdout = False, workers = args.workers)
    background = background_list(train_rows = train_rows, min_s = args.background_min_s)
    with open(file = args.out / "background.jsonl", mode = "w", encoding = "utf-8") as handle:
        for row in background:
            handle.write(json.dumps(obj = row, ensure_ascii = False) + "\n")
    summary = {"train": summarize(rows = train_rows), "eval": summarize(rows = eval_rows), "background": summarize(rows = background)}
    with open(file = args.out / "summary.json", mode = "w", encoding = "utf-8") as handle:
        json.dump(obj = summary, fp = handle, ensure_ascii = False, indent = 2)
    print(json.dumps(obj = {key: value["segments"] for key, value in summary.items()}), flush = True)


if __name__ == "__main__":
    main()
