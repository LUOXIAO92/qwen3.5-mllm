"""各真人语音数据集的读取方式。

每个函数逐条产出候选条目（字典），字段：

- uid：数据集内的编号
- lang：目标语言代码（pipeline.text.langs.LANGS 中的写法）
- transcript：数据集提供的转写
- source、source_license、source_ref：来源数据集、许可、定位信息
- speaker：说话人编号（没有时为空字符串）
- duration_s：时长（秒，数据集给出时）
- audio：音频位置，{"kind": "tar", "path", "member"}、{"kind": "file", "path"} 或 {"kind": "parquet", "path", "row"}
- extra：数据集自带的其他信息（质量分、子来源等），原样写入任务
"""

import json
import tarfile
from pathlib import Path

# Emilia 的语言代码与本项目相同（小写），这里列出本项目会用到的六种
EMILIA_LANGS = {"zh", "en", "ja", "ko", "de", "fr"}
EMILIA_LICENSES = {"Emilia": "CC-BY-NC-4.0", "Emilia-YODAS": "CC-BY-4.0"}


def emilia_tar(paths: list, lang: str):
    """Hugging Face 上 amphion/Emilia-Dataset 的 tar 分片：每条一个 .json（含 text、duration、speaker、dnsmos）和一个 .mp3。"""
    for path in map(Path, paths):
        subset = path.parent.parent.name
        with tarfile.open(name = path) as tar:
            members = {member.name: member for member in tar if member.isfile()}
            for name in sorted(members):
                if not name.endswith(".json"):
                    continue
                stem = name[: -len(".json")]
                audio_name = next((stem + ext for ext in (".mp3", ".wav", ".flac") if stem + ext in members), None)
                if audio_name is None:
                    continue
                meta = json.load(tar.extractfile(member = members[name]))
                yield {
                    "uid": meta.get("_id") or stem,
                    "lang": lang,
                    "transcript": meta.get("text", ""),
                    "source": subset,
                    "source_license": EMILIA_LICENSES.get(subset, "CC-BY-NC-4.0"),
                    "source_ref": f"{path.name}:{audio_name}",
                    "speaker": meta.get("speaker", ""),
                    "duration_s": meta.get("duration"),
                    "audio": {"kind": "tar", "path": str(path.resolve()), "member": audio_name},
                    "extra": {"dnsmos": meta.get("dnsmos"), "emilia_language": meta.get("language")},
                }


def makesense_emilia(root: str, lang: str):
    """MakeSense 项目整理的 Emilia 子集：metadata/、transcription/、audio/ 按大写语言代码分目录，转写由 Qwen3-ASR 重新生成。"""
    root = Path(root)
    code = lang.upper()
    texts = {}
    for path in sorted((root / "transcription" / code).glob(pattern = "*.jsonl")):
        with open(file = path, mode = "r", encoding = "utf-8") as handle:
            for line in handle:
                row = json.loads(s = line)
                texts[row["uid"]] = row["text"]
    for path in sorted((root / "metadata" / code).glob(pattern = "*.jsonl")):
        with open(file = path, mode = "r", encoding = "utf-8") as handle:
            for line in handle:
                meta = json.loads(s = line)
                if meta["uid"] not in texts:
                    continue
                subset = meta.get("subset", "Emilia").split("/")[0]
                yield {
                    "uid": meta["uid"],
                    "lang": lang,
                    "transcript": texts[meta["uid"]],
                    "source": f"{subset} (MakeSense Qwen3-ASR)",
                    "source_license": EMILIA_LICENSES.get(subset, "CC-BY-NC-4.0"),
                    "source_ref": meta["file_name"],
                    "speaker": "",
                    "duration_s": meta.get("duration"),
                    "audio": {"kind": "file", "path": str((root / "audio" / code / meta["file_name"]).resolve())},
                    "extra": {"dnsmos": meta.get("quality")},
                }


def worldspeech(paths: list, lang: str, max_cer: float = 1.0):
    """disco-eth/WorldSpeech 的 parquet 文件：human_transcript 是对齐到片段上的官方记录，asr_transcript 与 cer 由数据集提供。

    max_cer 先按数据集自带的 cer（官方记录与其 ASR 结果的差异）粗筛，减少后续要核对的条数。
    """
    import pyarrow.parquet as pq

    columns = ["human_transcript", "cer", "duration", "source", "source_url", "segment_id", "session_date"]
    for path in map(Path, paths):
        table = pq.read_table(source = path, columns = columns)
        for row_index, row in enumerate(table.to_pylist()):
            if row["cer"] is not None and row["cer"] > max_cer:
                continue
            yield {
                "uid": row["segment_id"],
                "lang": lang,
                "transcript": row["human_transcript"] or "",
                "source": f"WorldSpeech/{row['source']}",
                "source_license": "CC-BY-NC-4.0",
                "source_ref": row["source_url"],
                "speaker": "",
                "duration_s": row["duration"],
                "audio": {"kind": "parquet", "path": str(path.resolve()), "row": row_index},
                "extra": {"worldspeech_cer": row["cer"], "session_date": row["session_date"]},
            }


SOURCES = {
    "emilia": emilia_tar,
    "makesense_emilia": makesense_emilia,
    "worldspeech": worldspeech,
}
