"""测试用的假封装：不需要 GPU 和模型，行为由文本内容决定。

用法：python -m pipeline.runner.tests.fakes tts|asr ...（参数同真实封装）。
文本含 RUNAWAY 时合成的音频时长异常长；含 MISMATCH 时假 ASR 返回不同的文字；
含 TTSFAIL 时假 TTS 把该条标为失败。环境变量 FAKE_CRASH_FILE 指向的文件存在时，
第一次调用把它删除并以非零退出码退出，用来模拟封装崩溃。
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def run_tts(args: argparse.Namespace) -> int:
    crash = os.environ.get("FAKE_CRASH_FILE")
    if crash and Path(crash).exists():
        Path(crash).unlink()
        return 7
    out = Path(args.output_dir)
    out.mkdir(parents = True, exist_ok = True)
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding = "utf-8").splitlines() if line]
    with (out / "results.jsonl").open(mode = "w", encoding = "utf-8") as handle:
        for row in rows:
            text = row["text"]
            result = dict(row)
            if "TTSFAIL" in text:
                result.update(status = "failed", error = "fake failure", audio_path = None, duration_s = None)
            else:
                duration = len(text.replace(" ", "")) * (5.0 if "RUNAWAY" in text else 0.1)
                payload = hashlib.sha1(f"{row['sample_id']}|{text}|{row.get('seed')}|{row.get('voice_id')}".encode()).digest()
                (out / f"{row['sample_id']}.flac").write_bytes(payload)
                result.update(
                    tts_model = "fake-tts",
                    status = "ok",
                    error = None,
                    audio_path = str(out / f"{row['sample_id']}.flac"),
                    duration_s = round(duration, 3),
                )
            handle.write(json.dumps(result, ensure_ascii = False) + "\n")
    return 0


def run_asr(args: argparse.Namespace) -> int:
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding = "utf-8").splitlines() if line]
    # 假 ASR 从音频文件旁边的 texts 映射读不到文本，所以让测试通过环境变量给出 sample_id 到文本的映射
    texts = json.loads(Path(os.environ["FAKE_TEXTS"]).read_text(encoding = "utf-8"))
    with Path(args.output).open(mode = "w", encoding = "utf-8") as handle:
        for row in rows:
            text = texts[row["sample_id"]]
            hypothesis = "completely different words" if "MISMATCH" in text else text
            if args.variant == "cross" and "CROSSBAD" in text:
                hypothesis = "other"
            handle.write(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "lang": row["lang"],
                        "asr_model": f"fake-asr-{args.variant}",
                        "hypothesis": hypothesis,
                    },
                    ensure_ascii = False,
                )
                + "\n"
            )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices = ["tts", "asr"])
    parser.add_argument("--input", required = True)
    parser.add_argument("--output-dir")
    parser.add_argument("--output")
    parser.add_argument("--batch-size", type = int, default = 1)
    parser.add_argument("--variant", default = "primary")
    args = parser.parse_args()
    return run_tts(args = args) if args.kind == "tts" else run_asr(args = args)


if __name__ == "__main__":
    sys.exit(main())
