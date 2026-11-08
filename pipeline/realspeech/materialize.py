"""把选中条目的音频转成 16 kHz 单声道 FLAC，写出合成—核对流水线的输入文件（在 envs/mllm-eval 中运行）。

用法（在仓库根目录运行）::

    python -m pipeline.realspeech.materialize --out-dir <select 的输出目录> --workers 16

读取 <输出目录>/selected.jsonl，写出：

- audio/<sample_id>.flac：16 kHz 单声道 FLAC；已存在的文件直接复用，可以中断后续跑
- tasks.jsonl：合成任务格式，text 和 original 都是数据集的转写，tts_model 为数据集名，mode 为 real
- synth_results.jsonl：audio_path（相对输出目录）、duration_s、status

之后用 pipeline.runner 的 length、asr、compare、manifest 子命令完成核对（见 run.sh）。
"""

import argparse
import io
import json
import tarfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SAMPLING_RATE = 16000


def decode(source) -> "numpy.ndarray":
    """用 TorchCodec 解码（文件路径或字节），重采样为 16 kHz 单声道。"""
    from torchcodec.decoders import AudioDecoder

    decoder = AudioDecoder(source = source, sample_rate = SAMPLING_RATE, num_channels = 1)
    return decoder.get_all_samples().data.squeeze(dim = 0).numpy()


def write_group(job: tuple) -> list:
    """处理同一个 tar、parquet 文件或一批独立文件里的条目，返回 (sample_id, 时长或 None, 错误) 列表。"""
    import soundfile as sf

    kind, path, items, audio_dir = job
    results = []
    pending = []
    for item in items:
        target = Path(audio_dir) / f"{item['sample_id']}.flac"
        if target.exists():
            results.append((item["sample_id"], round(sf.info(file = str(target)).duration, 3), None))
        else:
            pending.append((item, target))
    if not pending:
        return results
    if kind == "tar":
        tar = tarfile.open(name = path)
        load = lambda item: tar.extractfile(member = item["audio"]["member"]).read()
    elif kind == "parquet":
        import pyarrow.parquet as pq

        audio_column = pq.read_table(source = path, columns = ["audio"]).column(0).to_pylist()
        load = lambda item: audio_column[item["audio"]["row"]]["bytes"]
    else:
        load = lambda item: item["audio"]["path"]
    for item, target in pending:
        try:
            wave = decode(source = load(item))
            tmp = target.with_suffix(".tmp.flac")
            sf.write(file = str(tmp), data = wave, samplerate = SAMPLING_RATE, format = "FLAC")
            tmp.replace(target)
            results.append((item["sample_id"], round(len(wave) / SAMPLING_RATE, 3), None))
        except Exception as error:
            results.append((item["sample_id"], None, f"{type(error).__name__}: {error}"[:200]))
    return results


def to_task(item: dict) -> dict:
    return {
        "sample_id": item["sample_id"],
        "entry_id": item["sample_id"],
        "lang": item["lang"],
        "text": item["transcript"],
        "original": item["transcript"],
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


def main() -> None:
    parser = argparse.ArgumentParser(description = "转出音频并写出流水线输入")
    parser.add_argument("--out-dir", required = True, type = Path)
    parser.add_argument("--workers", type = int, default = 16)
    args = parser.parse_args()

    with open(file = args.out_dir / "selected.jsonl", mode = "r", encoding = "utf-8") as handle:
        items = [json.loads(s = line) for line in handle if line.strip()]
    audio_dir = args.out_dir / "audio"
    audio_dir.mkdir(parents = True, exist_ok = True)
    groups = defaultdict(list)
    for item in items:
        audio = item["audio"]
        key = (audio["kind"], audio["path"]) if audio["kind"] in ("tar", "parquet") else ("file", "")
        groups[key].append(item)
    # 独立文件按每 500 条一组分给进程
    jobs = []
    for (kind, path), group in groups.items():
        step = 500 if kind == "file" else len(group)
        for start in range(0, len(group), step):
            jobs.append((kind, path, group[start:start + step], str(audio_dir)))
    durations, errors = {}, {}
    with ProcessPoolExecutor(max_workers = args.workers) as pool:
        for done, results in enumerate(pool.map(write_group, jobs), start = 1):
            for sample_id, duration, error in results:
                if error:
                    errors[sample_id] = error
                else:
                    durations[sample_id] = duration
            if done % 20 == 0 or done == len(jobs):
                print(f"[materialize] {done}/{len(jobs)} 组，成功 {len(durations)}，失败 {len(errors)}", flush = True)

    tasks, synth = [], []
    for item in sorted(items, key = lambda row: row["sample_id"]):
        if item["sample_id"] not in durations:
            continue
        task = to_task(item = item)
        tasks.append(task)
        synth.append({
            "sample_id": task["sample_id"],
            "tts_model": task["tts_model"],
            "mode": "real",
            "audio_path": f"audio/{task['sample_id']}.flac",
            "duration_s": durations[item["sample_id"]],
            "status": "ok",
            "error": None,
        })
    for name, rows in (("tasks.jsonl", tasks), ("synth_results.jsonl", synth)):
        with open(file = args.out_dir / name, mode = "w", encoding = "utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(obj = row, ensure_ascii = False) + "\n")
    if errors:
        (args.out_dir / "materialize_errors.json").write_text(json.dumps(obj = errors, ensure_ascii = False, indent = 1), encoding = "utf-8")
    print(json.dumps(obj = {"tasks": len(tasks), "audio_hours": round(sum(durations.values()) / 3600, 2), "errors": len(errors)}, ensure_ascii = False))


if __name__ == "__main__":
    main()
