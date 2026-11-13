"""把切分模式转出的片段按顺序拼回原始音频（在 envs/mllm-eval 中运行，只用 CPU）。

切分模式（materialize --split-max-seconds）把每条长音频在停顿处切成 <sample_id>-p00.flac、-p01.flac……，切点是采样点
位置，各段首尾相接、不丢不重。按编号顺序拼接 16 位 PCM 采样，就得到与不切分时逐采样点相同的 <sample_id>.flac。
之后不加 --split-max-seconds 重新运行 materialize，它会直接复用这些整段文件，不再读取原始数据。

每条音频写出整段文件并核对采样点数等于各段之和后，才删除它的片段文件；中断后重跑会跳过已经拼好的条目。
整段文件只在核对通过后才改名就位，所以整段已存在时剩下的片段是删除到一半被中断留下的，重跑时直接删除。
用户 2026-11-14 决定：切分会把 20–30 秒的长音频拆开，并拆散「说话、停顿、接着说话」的结构，WorldSpeech 改回整段。

用法::

    python -m pipeline.realspeech.join_pieces --out-dir <select 的输出目录> --workers 16
"""

import argparse
import json
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PIECE = re.compile(pattern = r"^(?P<base>.+)-p(?P<index>\d{2})\.flac$")


def join_one(job: tuple) -> tuple:
    """拼接一条音频的全部片段，返回 (sample_id, 采样点数, 错误)。"""
    import numpy as np
    import soundfile as sf

    sample_id, pieces, audio_dir = job
    target = Path(audio_dir) / f"{sample_id}.flac"
    try:
        if target.exists():
            for path in pieces:
                Path(path).unlink()
            return sample_id, sf.info(file = str(target)).frames, None
        waves = []
        rate = None
        for path in pieces:
            data, piece_rate = sf.read(file = path, dtype = "int16")
            if rate is not None and piece_rate != rate:
                raise ValueError(f"采样率不一致：{path}")
            rate = piece_rate
            waves.append(data)
        joined = np.concatenate(waves)
        tmp = target.with_suffix(".tmp.flac")
        sf.write(file = str(tmp), data = joined, samplerate = rate, format = "FLAC", subtype = "PCM_16")
        written = sf.info(file = str(tmp)).frames
        if written != sum(len(wave) for wave in waves):
            tmp.unlink()
            raise ValueError(f"写出 {written} 个采样点，应为 {sum(len(wave) for wave in waves)}")
        tmp.replace(target)
        for path in pieces:
            Path(path).unlink()
        return sample_id, written, None
    except Exception as error:
        return sample_id, 0, f"{type(error).__name__}: {error}"[:200]


def plan(audio_dir: Path, sample_ids: set) -> list:
    """列一次目录，按 sample_id 收集片段并按编号排序；整段文件已存在且没有片段的条目视为已完成。

    中断时写到一半的临时文件直接删除；整段已存在时只删除剩下的片段，不检查编号是否连续。
    """
    groups = defaultdict(list)
    whole = set()
    for path in audio_dir.iterdir():
        if path.name.endswith(".tmp.flac"):
            path.unlink()
            continue
        match = PIECE.match(path.name)
        if match and match["base"] in sample_ids:
            groups[match["base"]].append((int(match["index"]), str(path)))
        elif path.suffix == ".flac":
            whole.add(path.stem)
    jobs = []
    for sample_id, items in sorted(groups.items()):
        indices = [index for index, _ in sorted(items)]
        if sample_id not in whole and indices != list(range(len(indices))):
            raise SystemExit(f"{sample_id} 的片段编号不连续：{indices}")
        jobs.append((sample_id, [path for _, path in sorted(items)], str(audio_dir)))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description = "把切分模式的片段拼回原始音频")
    parser.add_argument("--out-dir", required = True, type = Path)
    parser.add_argument("--workers", type = int, default = 16)
    args = parser.parse_args()

    with open(file = args.out_dir / "selected.jsonl", mode = "r", encoding = "utf-8") as handle:
        sample_ids = {json.loads(s = line)["sample_id"] for line in handle if line.strip()}
    audio_dir = args.out_dir / "audio"
    jobs = plan(audio_dir = audio_dir, sample_ids = sample_ids)
    print(f"[join] {len(jobs)} 条音频待拼接", flush = True)
    joined, errors, frames = 0, {}, 0
    with ProcessPoolExecutor(max_workers = args.workers) as pool:
        for done, (sample_id, written, error) in enumerate(pool.map(join_one, jobs, chunksize = 64), start = 1):
            if error:
                errors[sample_id] = error
            else:
                joined += 1
                frames += written
            if done % 20000 == 0:
                print(f"[join] {done}/{len(jobs)}", flush = True)
    whole = sum(1 for sample_id in sample_ids if (audio_dir / f"{sample_id}.flac").exists())
    if errors:
        (args.out_dir / "join_errors.json").write_text(json.dumps(obj = errors, ensure_ascii = False, indent = 1), encoding = "utf-8")
    print(json.dumps(obj = {"joined": joined, "errors": len(errors), "hours": round(frames / 16000 / 3600, 2), "whole_files": whole,
                            "selected": len(sample_ids)}, ensure_ascii = False))


if __name__ == "__main__":
    main()
