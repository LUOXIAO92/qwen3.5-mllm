"""筛选无人声的环境声、噪声和器乐片段，作为「听到无人声音频时输出空文本」的训练素材（在 envs/vad 中运行）。

来源：

- FSD50K（Freesound 的 200 类声音事件，CC 系列许可）：labels/{dev,eval}.csv 给出每段的类别。带人声类别（说话、
  对话、人群嘈杂、喊叫、耳语、笑、哭、唱歌、叹气、欢呼、语音合成等）的片段直接排除；咳嗽、喷嚏、呼吸等不是说话的
  人体声音保留。
- MUSAN 的 noise 和 music 两部分：music/*/ANNOTATIONS 第三列标出是否有人声，有人声或缺标注的排除；speech 部分不用。

标签会漏标，所以留下的片段再用 Silero VAD 检查：把音频转成 16 kHz，逐 32 毫秒算出有人声的概率，记下判为人声的总秒数
和最大概率。是否采用由之后的训练数据步骤按这两个数决定，这里只记录。

用法::

    python -m pipeline.realspeech.nonspeech --fsd50k <FSD50k 目录> --musan <musan 目录> --out <输出 jsonl> --workers 8

输出每行一段：path、source、license、labels、duration_s、vad_speech_s、vad_max_prob、label_excluded（按标签排除的原因，
没有排除时为 null；被排除的片段不跑 VAD）。可以中断后续跑：已写出的 path 跳过。
"""

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SAMPLING_RATE = 16000
VOICE_LABELS = {
    "Human_voice", "Speech", "Male_speech_and_man_speaking", "Female_speech_and_woman_speaking",
    "Child_speech_and_kid_speaking", "Conversation", "Speech_synthesizer", "Chatter", "Crowd", "Cheering",
    "Whispering", "Shout", "Yell", "Screaming", "Laughter", "Giggle", "Chuckle_and_chortle", "Crying_and_sobbing",
    "Singing", "Male_singing", "Female_singing", "Sigh",
}


def fsd50k_items(root: Path) -> list:
    items = []
    for split in ("dev", "eval"):
        with open(file = root / "labels" / f"{split}.csv", mode = "r", encoding = "utf-8", newline = "") as handle:
            for row in csv.DictReader(handle):
                labels = row["labels"].split(",")
                voice = sorted(set(labels) & VOICE_LABELS)
                items.append({
                    "path": str((root / "clips" / split / f"{row['fname']}.wav").resolve()),
                    "source": f"FSD50K/{split}",
                    "license": "CC (per clip, see FSD50K metadata)",
                    "labels": labels,
                    "label_excluded": f"voice_label:{voice[0]}" if voice else None,
                })
    return items


def musan_items(root: Path) -> list:
    items = []
    for path in sorted((root / "noise").glob(pattern = "*/*.wav")):
        items.append({"path": str(path.resolve()), "source": f"MUSAN/noise/{path.parent.name}",
                      "license": "see MUSAN LICENSE", "labels": ["noise"], "label_excluded": None})
    vocals = {}
    for annotations in (root / "music").glob(pattern = "*/ANNOTATIONS"):
        with open(file = annotations, mode = "r", encoding = "utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) >= 3:
                    vocals[parts[0]] = parts[2]
    for path in sorted((root / "music").glob(pattern = "*/*.wav")):
        flag = vocals.get(path.stem)
        excluded = None if flag == "N" else ("music_vocals" if flag == "Y" else "music_unannotated")
        items.append({"path": str(path.resolve()), "source": f"MUSAN/music/{path.parent.name}",
                      "license": "see MUSAN LICENSE", "labels": ["music"], "label_excluded": excluded})
    return items


def screen(batch: list) -> list:
    """对一批片段跑 Silero VAD，返回补上 duration_s、vad_speech_s、vad_max_prob 的条目。"""
    import torch
    from silero_vad import load_silero_vad
    from torchcodec.decoders import AudioDecoder

    torch.set_num_threads(1)
    model = load_silero_vad()
    window = 512
    results = []
    for item in batch:
        try:
            wave = AudioDecoder(source = item["path"], sample_rate = SAMPLING_RATE, num_channels = 1).get_all_samples().data[0]
            duration = len(wave) / SAMPLING_RATE
            item = {**item, "duration_s": round(duration, 3)}
            if item["label_excluded"]:
                results.append(item)
                continue
            model.reset_states()
            pad = (-len(wave)) % window
            chunks = torch.nn.functional.pad(input = wave, pad = (0, pad)).reshape(-1, window)
            with torch.no_grad():
                probs = torch.stack([model(chunk, SAMPLING_RATE).reshape(()) for chunk in chunks])
            speech = float((probs > 0.5).sum()) * window / SAMPLING_RATE
            results.append({**item, "vad_speech_s": round(speech, 2), "vad_max_prob": round(float(probs.max()), 3)})
        except Exception as error:
            results.append({**item, "error": f"{type(error).__name__}: {error}"[:200]})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description = "筛选无人声片段并用 VAD 检查")
    parser.add_argument("--fsd50k", type = Path)
    parser.add_argument("--musan", type = Path)
    parser.add_argument("--out", required = True, type = Path)
    parser.add_argument("--workers", type = int, default = 8)
    args = parser.parse_args()

    items = (fsd50k_items(root = args.fsd50k) if args.fsd50k else []) + (musan_items(root = args.musan) if args.musan else [])
    done = set()
    if args.out.exists():
        with open(file = args.out, mode = "r", encoding = "utf-8") as handle:
            done = {json.loads(s = line)["path"] for line in handle if line.strip()}
    todo = [item for item in items if item["path"] not in done]
    print(f"[nonspeech] 共 {len(items)} 段，待处理 {len(todo)} 段", flush = True)
    batches = [todo[start:start + 200] for start in range(0, len(todo), 200)]
    args.out.parent.mkdir(parents = True, exist_ok = True)
    with open(file = args.out, mode = "a", encoding = "utf-8") as out, ProcessPoolExecutor(max_workers = args.workers) as pool:
        for index, results in enumerate(pool.map(screen, batches), start = 1):
            for row in results:
                out.write(json.dumps(obj = row, ensure_ascii = False) + "\n")
            out.flush()
            if index % 20 == 0 or index == len(batches):
                print(f"[nonspeech] {index}/{len(batches)} 批", flush = True)


if __name__ == "__main__":
    main()
