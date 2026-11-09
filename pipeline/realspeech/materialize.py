"""把选中条目的音频转成 16 kHz 单声道 FLAC，写出合成—核对流水线的输入文件（在 envs/mllm-eval 中运行）。

用法（在仓库根目录运行）::

    python -m pipeline.realspeech.materialize --out-dir <select 的输出目录> --workers 16 [--split-max-seconds 15]

读取 <输出目录>/selected.jsonl，写出：

- audio/<sample_id>.flac：16 kHz 单声道 FLAC；已存在的文件直接复用，可以中断后续跑
- tasks.jsonl：合成任务格式，text 和 original 都是数据集的转写，tts_model 为数据集名，mode 为 real
- synth_results.jsonl：audio_path（相对输出目录）、duration_s、status

给出 --split-max-seconds 时，长于它的音频在停顿处切成不超过该长度的若干段，每段一个样本，样本编号为
<sample_id>-p<序号>。切开后数据集的整段转写对不上各段，所以各段的 text 留空并标 text_unknown，只能用于
重新识别模式（RETRANSCRIBE=1，先由主 ASR 识别出文字，再由交叉 ASR 核对）；时长检查跳过这些样本。
片段越长，两个 ASR 逐字一致的可能越小（WorldSpeech 的片段中位数近 30 秒），切短后核对才有意义。

之后用 pipeline.runner 的 length、asr、compare、manifest 子命令完成核对（见 run.sh）。
"""

import argparse
import json
import tarfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SAMPLING_RATE = 16000
# 停顿检测：25 毫秒帧、10 毫秒步长；比本段较安静的 10% 帧高不到 8 分贝、或比最响的帧低 35 分贝以上的帧算作静音
FRAME = 400
HOP = 160
MIN_PAUSE_S = 0.25
MIN_PIECE_S = 2.0


def decode(source) -> np.ndarray:
    """用 TorchCodec 解码（文件路径或字节），重采样为 16 kHz 单声道。"""
    from torchcodec.decoders import AudioDecoder

    decoder = AudioDecoder(source = source, sample_rate = SAMPLING_RATE, num_channels = 1)
    return decoder.get_all_samples().data.squeeze(dim = 0).numpy()


def pause_centers(wave: np.ndarray) -> list:
    """返回各段停顿（不短于 MIN_PAUSE_S）的中点位置（采样点）。"""
    n = 1 + max(0, len(wave) - FRAME) // HOP
    frames = np.lib.stride_tricks.sliding_window_view(x = wave, window_shape = FRAME)[:: HOP][:n]
    db = 10 * np.log10(np.mean(frames.astype(np.float64) ** 2, axis = 1) + 1e-10)
    threshold = max(np.percentile(a = db, q = 10) + 8, db.max() - 35)
    silent = db < threshold
    centers, start = [], None
    for index, flag in enumerate(np.append(silent, False)):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if (index - start) * HOP / SAMPLING_RATE >= MIN_PAUSE_S:
                centers.append((start + index) // 2 * HOP + FRAME // 2)
            start = None
    return centers


def split_points(wave: np.ndarray, max_seconds: float) -> list:
    """把音频切成不超过 max_seconds 的段，返回 [(起点, 终点)]（采样点）。优先在停顿处切，没有合适的停顿时在上限处切断。"""
    total = len(wave)
    limit = int(max_seconds * SAMPLING_RATE)
    if total <= limit:
        return [(0, total)]
    minimum = int(MIN_PIECE_S * SAMPLING_RATE)
    centers = pause_centers(wave = wave)
    pieces, start = [], 0
    while total - start > limit:
        candidates = [c for c in centers if start + minimum <= c <= start + limit]
        cut = candidates[-1] if candidates else start + limit
        pieces.append((start, cut))
        start = cut
    if total - start >= minimum or not pieces:
        pieces.append((start, total))
    else:
        # 剩下的尾巴太短：并进上一段（可能略超上限，仍远小于原长）
        pieces[-1] = (pieces[-1][0], total)
    return pieces


def write_group(job: tuple) -> list:
    """处理同一个 tar、parquet 文件或一批独立文件里的条目。

    返回 (条目 sample_id, [(样本编号, 时长)], 错误) 的列表；不切分时每个条目只有一个样本，编号就是条目的 sample_id。
    """
    import soundfile as sf

    kind, path, items, audio_dir, max_seconds = job
    audio_dir = Path(audio_dir)
    results = []
    pending = []
    for item in items:
        existing = sorted(audio_dir.glob(pattern = f"{item['sample_id']}-p*.flac")) if max_seconds else []
        whole = audio_dir / f"{item['sample_id']}.flac"
        if existing:
            pieces = [(p.stem, round(sf.info(file = str(p)).duration, 3)) for p in existing]
            results.append((item["sample_id"], pieces, None))
        elif not max_seconds and whole.exists():
            results.append((item["sample_id"], [(whole.stem, round(sf.info(file = str(whole)).duration, 3))], None))
        else:
            pending.append(item)
    if not pending:
        return results
    if kind == "tar":
        # 按包内顺序读一遍：.tar.gz 不能随机定位，按条目顺序逐个取会让每次读取都从头解压
        def waves():
            wanted = {item["audio"]["member"]: item for item in pending}
            with tarfile.open(name = path, mode = "r|*") as tar:
                for member in tar:
                    item = wanted.pop(member.name, None)
                    if item is None:
                        continue
                    try:
                        yield item, decode(source = tar.extractfile(member = member).read()), None
                    except Exception as error:
                        yield item, None, error
                    if not wanted:
                        break
            for item in wanted.values():
                yield item, None, KeyError(f"{item['audio']['member']} not in archive")
    else:
        if kind == "parquet":
            import pyarrow.parquet as pq

            audio_column = pq.read_table(source = path, columns = ["audio"]).column(0).to_pylist()
            load = lambda item: decode(source = audio_column[item["audio"]["row"]]["bytes"])
        elif kind == "segment":
            # 同一段长录音只解码一次，再按起止时间截取
            full = decode(source = path)
            load = lambda item: full[int(item["audio"]["start"] * SAMPLING_RATE):int(item["audio"]["end"] * SAMPLING_RATE)]
        else:
            load = lambda item: decode(source = item["audio"]["path"])

        def waves():
            for item in pending:
                try:
                    yield item, load(item), None
                except Exception as error:
                    yield item, None, error

    for item, wave, error in waves():
        if error is not None:
            results.append((item["sample_id"], [], f"{type(error).__name__}: {error}"[:200]))
            continue
        try:
            spans = split_points(wave = wave, max_seconds = max_seconds) if max_seconds else [(0, len(wave))]
            pieces = []
            for index, (begin, end) in enumerate(spans):
                name = f"{item['sample_id']}-p{index:02d}" if max_seconds else item["sample_id"]
                target = audio_dir / f"{name}.flac"
                tmp = target.with_suffix(".tmp.flac")
                sf.write(file = str(tmp), data = wave[begin:end], samplerate = SAMPLING_RATE, format = "FLAC")
                tmp.replace(target)
                pieces.append((name, round((end - begin) / SAMPLING_RATE, 3)))
            results.append((item["sample_id"], pieces, None))
        except Exception as error:
            results.append((item["sample_id"], [], f"{type(error).__name__}: {error}"[:200]))
    return results


def to_task(item: dict, sample_id: str, split: bool) -> dict:
    task = {
        "sample_id": sample_id,
        "entry_id": item["sample_id"],
        "lang": item["lang"],
        "text": "" if split else item["transcript"],
        "original": "" if split else item["transcript"],
        "source": item["source"],
        "source_license": item["source_license"],
        "source_ref": item["source_ref"],
        "tts_model": item["sample_id"].split("-")[1],
        "mode": "real",
        "voice_id": item.get("speaker") or item["source"],
        "ref_lang": item["lang"],
        "cross_lingual": False,
        "holdout": False,
        "split": "train",
        "seed": 0,
        "target_from": item["target_from"],
        **{key: value for key, value in (item.get("extra") or {}).items() if value is not None},
    }
    if item.get("human_verified"):
        task["human_verified"] = True
    if split:
        # 切开的片段没有对应的转写；整段转写留作参考
        task["text_unknown"] = True
        task["segment_transcript"] = item["transcript"]
    return task


def main() -> None:
    parser = argparse.ArgumentParser(description = "转出音频并写出流水线输入")
    parser.add_argument("--out-dir", required = True, type = Path)
    parser.add_argument("--workers", type = int, default = 16)
    parser.add_argument("--split-max-seconds", type = float, default = 0.0, help = "大于 0 时把长音频在停顿处切成不超过这么长的段")
    args = parser.parse_args()

    with open(file = args.out_dir / "selected.jsonl", mode = "r", encoding = "utf-8") as handle:
        items = [json.loads(s = line) for line in handle if line.strip()]
    audio_dir = args.out_dir / "audio"
    audio_dir.mkdir(parents = True, exist_ok = True)
    groups = defaultdict(list)
    for item in items:
        audio = item["audio"]
        key = (audio["kind"], audio["path"]) if audio["kind"] in ("tar", "parquet", "segment") else ("file", "")
        groups[key].append(item)
    # 独立文件按每 500 条一组分给进程
    jobs = []
    for (kind, path), group in groups.items():
        step = 500 if kind == "file" else len(group)
        for start in range(0, len(group), step):
            jobs.append((kind, path, group[start:start + step], str(audio_dir), args.split_max_seconds))
    pieces, errors = {}, {}
    with ProcessPoolExecutor(max_workers = args.workers) as pool:
        for done, results in enumerate(pool.map(write_group, jobs), start = 1):
            for sample_id, item_pieces, error in results:
                if error:
                    errors[sample_id] = error
                else:
                    pieces[sample_id] = item_pieces
            if done % 20 == 0 or done == len(jobs):
                print(f"[materialize] {done}/{len(jobs)} 组，成功 {len(pieces)}，失败 {len(errors)}", flush = True)

    split = args.split_max_seconds > 0
    tasks, synth = [], []
    for item in sorted(items, key = lambda row: row["sample_id"]):
        for name, duration in pieces.get(item["sample_id"], []):
            task = to_task(item = item, sample_id = name, split = split)
            tasks.append(task)
            synth.append({
                "sample_id": name,
                "tts_model": task["tts_model"],
                "mode": "real",
                "audio_path": f"audio/{name}.flac",
                "duration_s": duration,
                "status": "ok",
                "error": None,
            })
    for name, rows in (("tasks.jsonl", tasks), ("synth_results.jsonl", synth)):
        with open(file = args.out_dir / name, mode = "w", encoding = "utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(obj = row, ensure_ascii = False) + "\n")
    if errors:
        (args.out_dir / "materialize_errors.json").write_text(json.dumps(obj = errors, ensure_ascii = False, indent = 1), encoding = "utf-8")
    hours = sum(duration for item_pieces in pieces.values() for _, duration in item_pieces) / 3600
    print(json.dumps(obj = {"entries": len(pieces), "samples": len(tasks), "audio_hours": round(hours, 2), "errors": len(errors)}, ensure_ascii = False))


if __name__ == "__main__":
    main()
