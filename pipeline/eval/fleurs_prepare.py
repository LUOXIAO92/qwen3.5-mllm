"""准备 FLEURS 真人语音评测集，并生成训练文本排除清单。

下载 google/fleurs 中指定语言的 dev、test 切分（音频压缩包与 TSV），把音频转为
16 kHz 单声道 FLAC，写出每种语言每个切分的 JSONL 清单。train 切分不下载音频，
只下载 TSV，取出其中的原始转写，与 dev、test 的转写一起写入排除清单。

用法：

    python -m pipeline.eval.fleurs_prepare --work-dir <工作目录的绝对路径>

工作目录下的产出：

    data/T02/<语言代码>/<切分>/<样本编号>.flac   16 kHz 单声道 FLAC
    data/T02/<语言代码>/<切分>.jsonl            每行一条样本的清单
    data/T02/exclusion.jsonl                     train、dev、test 三个切分的转写
    data/T02/summary.json                        条数与官方条数对照

临时下载的压缩包放在 tmp/T02/raw/ 下。

依赖：soundfile、numpy、scipy、pyyaml。
"""

import argparse
import io
import json
import math
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from scipy.signal import resample_poly


# Hugging Face 上 google/fleurs 数据集的文件根地址（公开、非 gated）
HF_BASE = "https://huggingface.co/datasets/google/fleurs/resolve/main"

TARGET_SR = 16000
LICENSE = "CC-BY-4.0"

# 项目语言代码 -> FLEURS 配置名，与 pipeline/CONVENTIONS.md 的语言代码一一对应
LANG_MAP = {
    "zh": "cmn_hans_cn",
    "en": "en_us",
    "ja": "ja_jp",
    "ko": "ko_kr",
    "de": "de_de",
    "fr": "fr_fr",
    "ru": "ru_ru",
    "pt": "pt_br",
    "es": "es_419",
    "it": "it_it",
    "id": "id_id",
    "ms": "ms_my",
    "th": "th_th",
    "vi": "vi_vn",
    "fil": "fil_ph",
    "ar": "ar_eg",
    "tr": "tr_tr",
    "fa": "fa_ir",
}

# 本地切分名 -> README 元数据中的切分名（FLEURS 的 dev 在元数据中叫 validation）
README_SPLIT = {
    "dev": "validation",
    "test": "test",
}

# TSV 没有表头，列顺序如下
TSV_COLUMNS = 7


def download(url: str, dest: Path, retries: int = 6) -> None:
    """下载文件到 dest，支持断点续传；dest 已存在则跳过。连接中断时指数退避重试。"""
    if dest.exists():
        return
    dest.parent.mkdir(parents = True, exist_ok = True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(retries + 1):
        try:
            start = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={start}-"} if start else {}
            request = urllib.request.Request(url = url, headers = headers)
            with urllib.request.urlopen(url = request, timeout = 60) as response:
                # 服务器不支持续传时会返回 200，此时只能从头写
                mode = "ab" if (start and response.status == 206) else "wb"
                with part.open(mode = mode) as out:
                    while True:
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
            part.replace(dest)
            return
        except (urllib.error.URLError, OSError) as err:
            if attempt == retries:
                raise
            wait = min(300, 2 ** attempt * 5)
            print(f"下载中断（{err}），{wait} 秒后第 {attempt + 1} 次重试：{dest.name}", flush = True)
            time.sleep(wait)


def read_tsv(path: Path) -> list[dict]:
    """读取 FLEURS 的 TSV 切分文件。列依次为：编号、文件名、原始转写、规范化转写、词列表、说话人、性别。"""
    rows = []
    with path.open(mode = "r", encoding = "utf-8") as f:
        for line_no, line in enumerate(f, start = 1):
            cols = line.rstrip("\r\n").split("\t")
            if len(cols) != TSV_COLUMNS:
                raise ValueError(f"{path} 第 {line_no} 行有 {len(cols)} 列，应为 {TSV_COLUMNS} 列")
            rows.append(
                {
                    "fleurs_id": cols[0],
                    "file_name": cols[1],
                    "raw_transcription": cols[2],
                    "transcription": cols[3],
                    "speaker_id": cols[5],
                    "gender": cols[6],
                }
            )
    return rows


def to_16k_mono(data: np.ndarray, sr: int) -> np.ndarray:
    """多声道取平均，采样率不是 16 kHz 时重采样。"""
    if data.ndim > 1:
        data = data.mean(axis = 1)
    if sr != TARGET_SR:
        g = math.gcd(sr, TARGET_SR)
        data = resample_poly(x = data, up = TARGET_SR // g, down = sr // g)
    return data.astype(np.float32)


def convert_split(
    tar_path: Path,
    rows: list[dict],
    audio_dir: Path,
    lang: str,
    split: str,
) -> list[dict]:
    """按顺序流式读取压缩包，只把清单中的音频转为 FLAC，不解压到磁盘。"""
    wanted = {row["file_name"]: row for row in rows}
    manifest = []
    audio_dir.mkdir(parents = True, exist_ok = True)
    with tarfile.open(name = tar_path, mode = "r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = Path(member.name).name
            row = wanted.pop(name, None)
            if row is None:
                continue
            raw = tar.extractfile(member).read()
            data, sr = sf.read(file = io.BytesIO(raw), dtype = "float32")
            data = to_16k_mono(data = data, sr = sr)
            stem = Path(name).stem
            out_path = audio_dir / f"{stem}.flac"
            sf.write(
                file = out_path,
                data = data,
                samplerate = TARGET_SR,
                format = "FLAC",
                subtype = "PCM_16",
            )
            manifest.append(
                {
                    "sample_id": f"{lang}-{split}-{stem}",
                    "stem": stem,
                    "row": row,
                    "duration_s": round(len(data) / TARGET_SR, 3),
                }
            )
    if wanted:
        missing = list(wanted.keys())[:5]
        raise FileNotFoundError(f"{tar_path} 中缺少 {len(wanted)} 个音频，例如 {missing}")
    return manifest


def write_jsonl(path: Path, records: list[dict]) -> None:
    """每行一个 JSON 对象，UTF-8 编码。"""
    path.parent.mkdir(parents = True, exist_ok = True)
    with path.open(mode = "w", encoding = "utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii = False) + "\n")


def official_counts(readme: Path) -> dict:
    """从 README 的元数据读取每个配置、每个切分的官方条数。"""
    text = readme.read_text(encoding = "utf-8")
    front = text.split("---", 2)[1]
    meta = yaml.safe_load(front)
    counts = {}
    for info in meta["dataset_info"]:
        counts[info["config_name"]] = {s["name"]: s["num_examples"] for s in info["splits"]}
    return counts


def manifest_complete(manifest_path: Path, expected: int) -> bool:
    """清单存在、条数与官方一致，且每条对应的 FLAC 都存在时，视为该切分已完成。"""
    if not manifest_path.exists():
        return False
    lines = manifest_path.read_text(encoding = "utf-8").splitlines()
    if len(lines) != expected:
        return False
    base = manifest_path.parent
    for line in lines:
        audio = json.loads(line)["audio_path"]
        if not (base.parent / audio).exists():
            return False
    return True


def process_language(
    lang: str,
    config: str,
    splits: list[str],
    raw_dir: Path,
    out_dir: Path,
    expected: dict,
) -> tuple[dict, list[dict]]:
    """处理一种语言：下载、转换 dev 与 test，并读取 train 的转写。"""
    summary = {}
    exclusion = []

    # 三个切分的 TSV 都读取，排除清单需要 train、dev、test 的转写
    for split in ["train", "dev", "test"]:
        tsv_name = f"{split}.tsv"
        tsv_path = raw_dir / config / tsv_name
        download(url = f"{HF_BASE}/data/{config}/{tsv_name}", dest = tsv_path)
        rows = read_tsv(path = tsv_path)
        for row in rows:
            exclusion.append(
                {
                    "lang": lang,
                    "split": split,
                    "fleurs_id": row["fleurs_id"],
                    "raw_transcription": row["raw_transcription"],
                    "transcription": row["transcription"],
                }
            )
        summary[split] = {"tsv_rows": len(rows)}

    for split in splits:
        manifest_path = out_dir / lang / f"{split}.jsonl"
        if manifest_complete(manifest_path = manifest_path, expected = expected[split]):
            print(f"[{lang}] {split} 已完成，跳过", flush = True)
            summary[split]["converted"] = expected[split]
            continue
        tar_name = f"{split}.tar.gz"
        tar_path = raw_dir / config / "audio" / tar_name
        download(url = f"{HF_BASE}/data/{config}/audio/{tar_name}", dest = tar_path)
        rows = read_tsv(path = raw_dir / config / f"{split}.tsv")
        audio_dir = out_dir / lang / split
        converted = convert_split(
            tar_path = tar_path,
            rows = rows,
            audio_dir = audio_dir,
            lang = lang,
            split = split,
        )
        records = []
        for item in converted:
            row = item["row"]
            records.append(
                {
                    "sample_id": item["sample_id"],
                    "audio_path": f"{lang}/{split}/{item['stem']}.flac",
                    "lang": lang,
                    "fleurs_config": config,
                    "split": split,
                    "fleurs_id": row["fleurs_id"],
                    "raw_transcription": row["raw_transcription"],
                    "transcription": row["transcription"],
                    "speaker_id": row["speaker_id"],
                    "gender": row["gender"],
                    "duration_s": item["duration_s"],
                    "sample_rate": TARGET_SR,
                    "source": "google/fleurs",
                    "source_license": LICENSE,
                }
            )
        write_jsonl(path = out_dir / lang / f"{split}.jsonl", records = records)
        summary[split]["converted"] = len(records)

    return summary, exclusion


def main(argv = None) -> int:
    parser = argparse.ArgumentParser(description = "准备 FLEURS 评测集与训练文本排除清单")
    parser.add_argument(
        "--work-dir",
        required = True,
        help = "工作目录的绝对路径，产出写入其下的 data/T02/",
    )
    parser.add_argument(
        "--langs",
        nargs = "+",
        default = list(LANG_MAP.keys()),
        help = "项目语言代码，默认 18 种",
    )
    parser.add_argument(
        "--splits",
        nargs = "+",
        default = ["dev", "test"],
        help = "要转换音频的切分，默认 dev 与 test",
    )
    args = parser.parse_args(argv)

    for lang in args.langs:
        if lang not in LANG_MAP:
            print(f"未知语言代码：{lang}", file = sys.stderr)
            return 2

    work = Path(args.work_dir)
    raw_dir = work / "tmp" / "T02" / "raw"
    out_dir = work / "data" / "T02"
    out_dir.mkdir(parents = True, exist_ok = True)

    readme = raw_dir / "README.md"
    download(url = f"{HF_BASE}/README.md", dest = readme)
    official = official_counts(readme = readme)

    all_exclusion = []
    summary = {}
    for lang in args.langs:
        config = LANG_MAP[lang]
        print(f"[{lang}] {config}", flush = True)
        lang_summary, exclusion = process_language(
            lang = lang,
            config = config,
            splits = args.splits,
            raw_dir = raw_dir,
            out_dir = out_dir,
            expected = {split: official[config][README_SPLIT[split]] for split in args.splits},
        )
        for split in args.splits:
            lang_summary[split]["official"] = official[config][README_SPLIT[split]]
        summary[lang] = {"fleurs_config": config, **lang_summary}
        all_exclusion.extend(exclusion)

    write_jsonl(path = out_dir / "exclusion.jsonl", records = all_exclusion)
    with (out_dir / "summary.json").open(mode = "w", encoding = "utf-8") as f:
        json.dump(obj = summary, fp = f, ensure_ascii = False, indent = 2)
    print(f"排除清单：{len(all_exclusion)} 条", flush = True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
