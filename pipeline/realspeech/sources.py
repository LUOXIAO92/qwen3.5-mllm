"""各真人语音数据集的读取方式。

每个函数逐条产出候选条目（字典），字段：

- uid：数据集内的编号
- lang：目标语言代码（pipeline.text.langs.LANGS 中的写法）
- transcript：数据集提供的转写
- source、source_license、source_ref：来源数据集、许可、定位信息
- speaker：说话人编号（没有时为空字符串）
- duration_s：时长（秒，数据集给出时）
- audio：音频位置，{"kind": "tar", "path", "member"}、{"kind": "file", "path"}、{"kind": "parquet", "path", "row"}
  或 {"kind": "segment", "path", "start", "end"}（长录音中的一段，单位秒）
- extra：数据集自带的其他信息（质量分、子来源等），原样写入任务
"""

import csv
import json
import re
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


def makesense_emilia(root: str, lang: str, audio_roots: tuple = ()):
    """MakeSense 项目整理的 Emilia 子集：metadata/、transcription/、audio/ 按大写语言代码分目录，转写由 Qwen3-ASR 重新生成。

    一部分音频可能放在另一个同样结构的目录里（MakeSense 先下载的一批），audio_roots 给出这些目录，按顺序查找。
    """
    root = Path(root)
    code = lang.upper()
    search = [root / "audio" / code] + [Path(extra) / "audio" / code for extra in audio_roots]
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
                audio_path = next((folder / meta["file_name"] for folder in search if (folder / meta["file_name"]).exists()), None)
                if audio_path is None:
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
                    "audio": {"kind": "file", "path": str(audio_path.resolve())},
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


def _read_tsv(path: Path) -> list:
    """读取 Common Voice 等的制表符分隔文件（不按引号解析，句子里的引号原样保留）。"""
    csv.field_size_limit(1 << 24)
    with open(file = path, mode = "r", encoding = "utf-8", newline = "") as handle:
        return list(csv.DictReader(handle, delimiter = "\t", quoting = csv.QUOTE_NONE))


def _common_voice_row(row: dict, lang: str, source: str, audio: dict, duration) -> dict:
    return {
        "uid": Path(row["path"]).stem,
        "lang": lang,
        "transcript": row["sentence"],
        "source": source,
        "source_license": "CC0-1.0",
        "source_ref": row["path"],
        "speaker": row.get("client_id", ""),
        "duration_s": duration,
        "audio": audio,
        "extra": {
            "cv_locale": row.get("locale"),
            "cv_up_votes": int(row["up_votes"]) if row.get("up_votes") else None,
            "cv_down_votes": int(row["down_votes"]) if row.get("down_votes") else None,
        },
    }


def common_voice(paths: list, lang: str):
    """Common Voice 的已验证录音（至少两人投票确认朗读与句子一致）。paths 中每一项可以是：

    - Mozilla Data Collective 下载、只保留已验证录音的语言目录：validated.tsv、clip_durations.tsv、clips/*.mp3
    - Hugging Face 镜像 fsicoli/common_voice_22_0 的音频 tar：<根>/audio/<语言>/<划分>/<语言>_<划分>_<n>.tar，
      句子在 <根>/transcript/<语言>/<划分>.tsv；镜像没有给时长
    台湾普通话（zh-TW）的录音同样归入 zh，清单阶段转成简体。
    """
    for path in map(Path, paths):
        if path.is_dir():
            durations = {}
            if (path / "clip_durations.tsv").exists():
                for row in _read_tsv(path = path / "clip_durations.tsv"):
                    durations[row["clip"]] = int(row["duration[ms]"]) / 1000
            for row in _read_tsv(path = path / "validated.tsv"):
                clip = path / "clips" / row["path"]
                if clip.exists():
                    yield _common_voice_row(
                        row = row,
                        lang = lang,
                        source = "Common Voice 27.0",
                        audio = {"kind": "file", "path": str(clip.resolve())},
                        duration = durations.get(row["path"]),
                    )
            continue
        split, locale, root = path.parent.name, path.parent.parent.name, path.parents[3]
        rows = {row["path"]: row for row in _read_tsv(path = root / "transcript" / locale / f"{split}.tsv")}
        with tarfile.open(name = path) as tar:
            members = [member.name for member in tar if member.isfile() and member.name.endswith(".mp3")]
        for member in members:
            row = rows.get(Path(member).name)
            if row is not None:
                yield _common_voice_row(
                    row = row,
                    lang = lang,
                    source = "Common Voice 22.0",
                    audio = {"kind": "tar", "path": str(path.resolve()), "member": member},
                    duration = None,
                )


def gigaspeech2(paths: list, lang: str):
    """speechcolab/gigaspeech2 的训练分片 data/<语言>/train/<n>.tar.gz，转写取 data/<语言>/train_refined.tsv。

    转写是模型迭代生成的伪标注（小写、无标点），用重新识别模式。tsv 的编号 a-b-c 对应分片 a.tar.gz 中的 a/b/a-b-c.wav，
    不在 train_refined.tsv 里的片段（精选时被去掉）不用。
    """
    paths = [Path(path) for path in paths]
    wanted = {path.name[: -len(".tar.gz")]: path for path in paths}
    tsv = paths[0].parent.parent / "train_refined.tsv"
    with open(file = tsv, mode = "r", encoding = "utf-8") as handle:
        for line in handle:
            uid, _, text = line.rstrip("\n").partition("\t")
            parts = uid.split("-")
            if len(parts) != 3 or parts[0] not in wanted:
                continue
            path = wanted[parts[0]]
            yield {
                "uid": uid,
                "lang": lang,
                "transcript": text,
                "source": "GigaSpeech 2",
                "source_license": "Apache-2.0 (non-commercial research terms)",
                "source_ref": f"{path.name}:{parts[0]}/{parts[1]}/{uid}.wav",
                "speaker": "",
                "duration_s": None,
                "audio": {"kind": "tar", "path": str(path.resolve()), "member": f"{parts[0]}/{parts[1]}/{uid}.wav"},
                "extra": {},
            }


def parsvoice(paths: list, lang: str):
    """MohammadJRanjbar/ParsVoice 的 data/*.parquet（segment_id、audio、transcript、speaker_id）。

    同名的 metadata/*.parquet 存在时读出时长和质量信息，跳过 is_clean 为假的片段。转写是伪标注、没有标点，用重新识别模式。
    """
    import pyarrow.parquet as pq

    for path in map(Path, paths):
        meta = {}
        meta_path = path.parent.parent / "metadata" / path.name
        if meta_path.exists():
            columns = ["segment_id", "duration_sec", "is_clean", "snr_db", "audio_quality_score", "has_background_music"]
            for row in pq.read_table(source = meta_path, columns = columns).to_pylist():
                meta[row["segment_id"]] = row
        table = pq.read_table(source = path, columns = ["segment_id", "transcript", "speaker_id"])
        for row_index, row in enumerate(table.to_pylist()):
            info = meta.get(row["segment_id"], {})
            if info and not info.get("is_clean", True):
                continue
            yield {
                "uid": row["segment_id"],
                "lang": lang,
                "transcript": row["transcript"] or "",
                "source": "ParsVoice",
                "source_license": "CC-BY-NC-4.0",
                "source_ref": f"{path.name}:{row_index}",
                "speaker": row["speaker_id"] or "",
                "duration_s": info.get("duration_sec"),
                "audio": {"kind": "parquet", "path": str(path.resolve()), "row": row_index},
                "extra": {
                    "snr_db": info.get("snr_db"),
                    "audio_quality_score": info.get("audio_quality_score"),
                    "has_background_music": info.get("has_background_music"),
                },
            }


def phoaudiobook(paths: list, lang: str):
    """thivux/phoaudiobook 的 data/*.parquet（audio、text、speaker）：越南语有声书，转写带标点但句首不大写。"""
    import pyarrow.parquet as pq

    for path in map(Path, paths):
        table = pq.read_table(source = path, columns = ["text", "speaker"])
        for row_index, row in enumerate(table.to_pylist()):
            yield {
                "uid": f"{path.stem}-{row_index:05d}",
                "lang": lang,
                "transcript": row["text"] or "",
                "source": "PhoAudiobook",
                "source_license": "unspecified (no redistribution)",
                "source_ref": f"{path.name}:{row_index}",
                "speaker": row["speaker"] or "",
                "duration_s": None,
                "audio": {"kind": "parquet", "path": str(path.resolve()), "row": row_index},
                "extra": {},
            }


def wenetspeech4tts(paths: list, lang: str):
    """Wenetspeech4TTS/WenetSpeech4TTS 的 Premium 分片（.tar.gz，wavs/ 与 txts/ 一一对应；txt 第一行为「编号\t带标点的文本」）。

    需要先读完一遍压缩包取出全部 txt；DNSMOS 分数（DNSMOS_P808Scores/Premium_DNSMOS.lst）存在时一并记录。
    """
    paths = [Path(path) for path in paths]
    dnsmos = {}
    scores = paths[0].parent.parent / "DNSMOS_P808Scores" / "Premium_DNSMOS.lst"
    if scores.exists():
        with open(file = scores, mode = "r", encoding = "utf-8") as handle:
            for line in handle:
                key, _, value = line.strip().partition("\t")
                dnsmos[key] = float(value) if value else None
    for path in paths:
        texts, wavs = {}, {}
        with tarfile.open(name = path, mode = "r|gz") as tar:
            for member in tar:
                stem = Path(member.name).stem
                if member.name.endswith(".txt"):
                    first = tar.extractfile(member = member).read().decode(encoding = "utf-8").split("\n", 1)[0]
                    texts[stem] = first.partition("\t")[2]
                elif member.name.endswith(".wav"):
                    wavs[stem] = member.name
        for stem in sorted(set(texts) & set(wavs)):
            yield {
                "uid": stem,
                "lang": lang,
                "transcript": texts[stem],
                "source": "WenetSpeech4TTS Premium",
                "source_license": "CC-BY-4.0",
                "source_ref": f"{path.name}:{wavs[stem]}",
                "speaker": stem.rsplit("_", 1)[0],
                "duration_s": None,
                "audio": {"kind": "tar", "path": str(path.resolve()), "member": wavs[stem]},
                "extra": {"dnsmos": dnsmos.get(stem)},
            }


# MAGICDATA 转写里的非语音标记：[FIL] 填充音（嗯、啊），[SPK] 说话人发出的杂音
_MAGICDATA_MARK = re.compile(pattern = r"\s*\[(?:FIL|SPK)\]\s*")


def _magicdata_gap(match) -> str:
    """去掉标记后，两侧都是拉丁字母或数字时留一个空格（英文词之间），否则直接相连（汉字之间不加空格）。"""
    text = match.string
    before = text[match.start() - 1] if match.start() > 0 else ""
    after = text[match.end()] if match.end() < len(text) else ""
    return " " if before.isascii() and before.isalnum() and after.isascii() and after.isalnum() else ""


def magicdata(paths: list, lang: str):
    """MAGICDATA Mandarin Chinese Read Speech（OpenSLR 68）的训练集 train_set.tar.gz（内含 train/<说话人>/<编号>.wav）。

    转写取 TRANS.txt（制表符分隔：UtteranceID、SpeakerID、Transcription，第一行是表头），可以放在 paths 中，
    否则在压缩包同级的 train/TRANS.txt 查找（它在压缩包末尾，要先单独解出）。转写由人工完成，没有标点，
    数字写成汉字；[FIL]、[SPK] 两种非语音标记从文本中去掉，去掉后为空的条目跳过。TRANS.txt 中有、压缩包里
    没有音频的条目在转出音频时记为失败。
    """
    paths = [Path(path) for path in paths]
    archives = [path for path in paths if path.name.endswith(".tar.gz")]
    trans = next((path for path in paths if path.name == "TRANS.txt"), None)
    if trans is None:
        trans = archives[0].parent / "train" / "TRANS.txt"
    archive = archives[0].resolve()
    with open(file = trans, mode = "r", encoding = "utf-8") as handle:
        next(handle)
        for line in handle:
            utterance, speaker, text = line.rstrip("\n").split("\t")
            text = _MAGICDATA_MARK.sub(_magicdata_gap, text).strip()
            if not text:
                continue
            member = f"train/{speaker}/{utterance}"
            yield {
                "uid": Path(utterance).stem,
                "lang": lang,
                "transcript": text,
                "source": "MAGICDATA Mandarin Chinese Read Speech",
                "source_license": "CC-BY-NC-ND-4.0",
                "source_ref": f"{archive.name}:{member}",
                "speaker": speaker,
                "duration_s": None,
                "audio": {"kind": "tar", "path": str(archive), "member": member},
                "extra": {},
            }


# ReazonSpeech 的电视字幕里，［拍手］（笑）之类说明声音的注记不是说出的话
_REAZON_NOTE = re.compile(pattern = r"[［\[（(【♪]")


def reazonspeech(paths: list, lang: str):
    """reazon-research/reazonspeech（v2）的音频 tar（<前缀>.tar，内含 <前缀>/<编号>.flac），转写取同级 tsv/all.tsv。

    转写来自电视字幕对齐，带「。」「、」；含声音注记（括号、♪）的字幕跳过。
    """
    paths = [Path(path) for path in paths]
    wanted = {path.stem: path for path in paths}
    tsv = paths[0].parent.parent / "tsv" / "all.tsv"
    with open(file = tsv, mode = "r", encoding = "utf-8") as handle:
        for line in handle:
            member, _, text = line.rstrip("\n").partition("\t")
            prefix = member.split("/", 1)[0]
            if prefix not in wanted or _REAZON_NOTE.search(text):
                continue
            yield {
                "uid": Path(member).stem,
                "lang": lang,
                "transcript": text,
                "source": "ReazonSpeech v2",
                "source_license": "CDLA-Sharing-1.0 (Japanese Copyright Act Art. 30-4)",
                "source_ref": member,
                "speaker": "",
                "duration_s": None,
                "audio": {"kind": "tar", "path": str(wanted[prefix].resolve()), "member": member},
                "extra": {},
            }


_ARABIC_MARK_SPACE = re.compile(pattern = r"\s+([،؛؟.!:,])")


def mgb2(paths: list, lang: str, max_wmer: float = 100.0):
    """QCRI/mgb2 训练集：wav/<编号>.wav 是整期节目，xml/utf8/<编号>.xml 给出人工转写对齐后的片段。

    每个片段带 WMER（人工转写与轻监督解码结果的词错误率，%），先按 max_wmer 粗筛。转写中的标点是独立的词，
    拼回时贴到前一个词后面。
    """
    from xml.etree import ElementTree

    for path in map(Path, paths):
        xml = path.parent.parent / "xml" / "utf8" / f"{path.stem}.xml"
        if not xml.exists():
            continue
        root = ElementTree.parse(source = xml).getroot()
        for segment in root.iter(tag = "segment"):
            wmer = float(segment.get("WMER", "100"))
            if wmer > max_wmer:
                continue
            words = [element.text or "" for element in segment.iter(tag = "element")]
            text = _ARABIC_MARK_SPACE.sub(repl = r"\1", string = " ".join(word for word in words if word))
            start, end = float(segment.get("starttime")), float(segment.get("endtime"))
            yield {
                "uid": segment.get("id"),
                "lang": lang,
                "transcript": text,
                "source": "MGB-2",
                "source_license": "research only",
                "source_ref": f"{path.name}:{start:.2f}-{end:.2f}",
                "speaker": segment.get("who", ""),
                "duration_s": round(end - start, 3),
                "audio": {"kind": "segment", "path": str(path.resolve()), "start": start, "end": end},
                "extra": {"mgb2_wmer": wmer, "mgb2_pmer": float(segment.get("PMER", "nan")), "mgb2_awd": float(segment.get("AWD", "nan"))},
            }


def _librispeech_ids(subset_dir: Path) -> set:
    """LibriSpeech 子集目录（<说话人>/<章节>/<说话人>-<章节>.trans.txt）里所有语句编号。"""
    ids = set()
    for trans in subset_dir.glob(pattern = "*/*/*.trans.txt"):
        with open(file = trans, mode = "r", encoding = "utf-8") as handle:
            for line in handle:
                if line.strip():
                    ids.add(line.split(maxsplit = 1)[0])
    return ids


def _librispeech_pc_rows(subset_dir: Path) -> list:
    """读取 <LibriSpeech 根>/pc/<子集>.json（LibriSpeech-PC 的 manifest，每行一个 json）。"""
    manifest = subset_dir.parent.parent / "pc" / f"{subset_dir.name}.json"
    with open(file = manifest, mode = "r", encoding = "utf-8") as handle:
        return [json.loads(s = line) for line in handle if line.strip()]


def librispeech_coverage(paths: list) -> dict:
    """各子集中：LibriSpeech 原有的语句数、有 LibriSpeech-PC 转写的语句数、没有 PC 转写（会被跳过）的语句数。"""
    result = {}
    for path in map(Path, paths):
        original = _librispeech_ids(subset_dir = path)
        covered = {Path(row["audio_filepath"]).stem for row in _librispeech_pc_rows(subset_dir = path)}
        result[path.name] = {"original": len(original), "with_pc": len(original & covered), "without_pc": len(original - covered)}
    return result


def librispeech(paths: list, lang: str):
    """LibriSpeech 训练集（OpenSLR SLR12，CC BY 4.0）配 LibriSpeech-PC（SLR145，CC BY 4.0）的带大小写、标点的转写。

    paths 中每一项是一个解压后的子集目录 <根>/LibriSpeech/train-clean-100（或 train-clean-360、train-other-500）；
    LibriSpeech-PC 的 manifests 解压到 <根>/pc/<子集>.json。按语句编号（audio_filepath 的文件名）把 PC 的 text
    （NVIDIA 说明为「可直接用于 ASR 训练」的版本，无引号、分号等符号）对上音频，text_raw 记在 extra 中。
    PC 没有收录的语句（约 9%）和找不到音频的语句跳过，数量见 librispeech_coverage。转写来自有声书原文，按人工转写处理。
    """
    for path in map(Path, paths):
        for row in _librispeech_pc_rows(subset_dir = path):
            relative = Path(row["audio_filepath"])
            audio = path.parent / relative
            if not audio.exists():
                continue
            speaker = relative.stem.split("-")[0]
            yield {
                "uid": relative.stem,
                "lang": lang,
                "transcript": row["text"],
                "source": f"LibriSpeech + LibriSpeech-PC ({path.name})",
                "source_license": "CC-BY-4.0",
                "source_ref": row["audio_filepath"],
                "speaker": speaker,
                "duration_s": row.get("duration"),
                "audio": {"kind": "file", "path": str(audio.resolve())},
                "extra": {"text_raw": row.get("text_raw")},
            }


SOURCES = {
    "emilia": emilia_tar,
    "makesense_emilia": makesense_emilia,
    "worldspeech": worldspeech,
    "common_voice": common_voice,
    "gigaspeech2": gigaspeech2,
    "parsvoice": parsvoice,
    "phoaudiobook": phoaudiobook,
    "wenetspeech4tts": wenetspeech4tts,
    "reazonspeech": reazonspeech,
    "magicdata": magicdata,
    "mgb2": mgb2,
    "librispeech": librispeech,
}
