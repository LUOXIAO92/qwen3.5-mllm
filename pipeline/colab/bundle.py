"""把一份清单和它引用的音频打成一个 tar，在另一台机器上解开就能使用（任务 C1）。

tar 中的路径都相对 --root：清单和音频各自放在它们相对 --root 的位置；清单里的 audio_path 改写为相对清单所在
目录的路径，本来就是这样写的行原样保留。在新机器上用 tar -C <新的根目录> -xf 解开，读取清单时相对路径以清单所在
目录为基准（pipeline.train.data.read_manifest 和 mixing.BackgroundMixer 都是这样），音频就能找到。全部音频都必须在
--root 之下。

    # dev 清单：清单在 <工作目录>/data/T10，音频在 <工作目录>/data/T02 下，清单中改写为 ../T02/...
    python -m pipeline.colab.bundle \\
        --manifest <工作目录>/data/T10/fleurs_dev.jsonl \\
        --root <工作目录> \\
        --output fleurs-dev.tar
    # 背景声列表：路径本来就相对列表所在目录，列表内容不变；只打包列表引用的文件
    python -m pipeline.colab.bundle \\
        --manifest $DATASET/audio/qwen3.5-asr/nonspeech/v2/background.jsonl \\
        --root $DATASET \\
        --output noise-v2.tar
"""

import argparse
import io
import json
import os
import tarfile
import time
from pathlib import Path


def rewrite_manifest(manifest, root, audio_root = None):
    """返回 (改写后的清单文本, {tar 中的路径: 音频文件})。"""
    manifest = Path(manifest).absolute()
    root = Path(root).absolute()
    base = Path(audio_root).absolute() if audio_root is not None else manifest.parent
    lines = []
    files = {}
    with open(file = manifest, mode = "r", encoding = "utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(s = line)
            audio_path = Path(row["audio_path"])
            if not audio_path.is_absolute():
                audio_path = base / audio_path
            try:
                member = audio_path.relative_to(root)
            except ValueError:
                raise ValueError(f"{audio_path} 不在 {root} 之下") from None
            relative = os.path.relpath(path = audio_path, start = manifest.parent)
            if relative != row["audio_path"]:
                row["audio_path"] = relative
                line = json.dumps(obj = row, ensure_ascii = False) + "\n"
            lines.append(line if line.endswith("\n") else line + "\n")
            files[member.as_posix()] = audio_path
    return "".join(lines), files


def write_bundle(
    manifest,
    root,
    output,
    audio_root = None,
):
    """写出 tar，返回 (音频文件数, 音频字节数)。"""
    text, files = rewrite_manifest(manifest = manifest, root = root, audio_root = audio_root)
    data = text.encode(encoding = "utf-8")
    info = tarfile.TarInfo(name = Path(manifest).absolute().relative_to(Path(root).absolute()).as_posix())
    info.size = len(data)
    info.mtime = int(time.time())
    info.mode = 0o644
    output = Path(output)
    output.parent.mkdir(parents = True, exist_ok = True)
    tmp = output.with_name(name = output.name + ".tmp")
    total = 0
    with tarfile.open(name = tmp, mode = "w") as archive:
        archive.addfile(tarinfo = info, fileobj = io.BytesIO(initial_bytes = data))
        for member in sorted(files):
            archive.add(name = str(files[member]), arcname = member, recursive = False)
            total += files[member].stat().st_size
    tmp.replace(target = output)
    return len(files), total


def main(argv = None):
    parser = argparse.ArgumentParser(description = "把清单和它引用的音频打成 tar，路径相对 --root")
    parser.add_argument("--manifest", required = True, help = "清单 JSONL，须在 --root 之下")
    parser.add_argument("--root", required = True, help = "tar 中路径的基准目录")
    parser.add_argument("--audio-root", default = None, help = "清单中相对 audio_path 的基准目录，默认是清单所在目录")
    parser.add_argument("--output", required = True, help = "输出的 tar 文件")
    args = parser.parse_args(args = argv)
    count, total = write_bundle(
        manifest = args.manifest,
        root = args.root,
        output = args.output,
        audio_root = args.audio_root,
    )
    print(json.dumps(obj = {"output": args.output, "files": count, "gb": round(number = total / 1e9, ndigits = 2)}))


if __name__ == "__main__":
    main()
