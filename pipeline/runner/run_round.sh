#!/usr/bin/env bash
# 跑一整轮合成：先跑留出音色的测试集（只在第一轮需要），再按语言依次跑训练集，每种语言一个输出目录。
# 已写出 samples.jsonl 的输出目录视为完成，直接跳过；中断后重新运行本脚本即可续跑。
#
# 用法：pipeline/runner/run_round.sh <轮次名，如 r1>
# 条目文件由 pipeline.runner.rounds 生成：$DATASET/text/qwen3.5-asr/rounds/test-all.jsonl 与 <轮次名>-<语言>.jsonl。
# 音频写到 $DATASET/audio/qwen3.5-asr/v1/test 与 $DATASET/audio/qwen3.5-asr/v1/train/<语言>（第二轮起为 train-<轮次名>/<语言>）。
set -uo pipefail
round="${1:?轮次名}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
S="${DATASET:?需要设置 DATASET}/text/qwen3.5-asr/rounds"
A="$DATASET/audio/qwen3.5-asr/v1"
train_dir="$A/train"
[ "$round" = r1 ] || train_dir="$A/train-$round"
if [ -f "$S/test-all.jsonl" ] && [ ! -f "$A/test/samples.jsonl" ]; then
  "$repo/pipeline/runner/run_dir.sh" test-all test "$S/test-all.jsonl" "$A/test" || echo "FAILED test-all"
fi
# 通过率高的语言放在前面，先拿到可用的数据
for lang in en tr es id ar it ko de pt ru vi fr th zh fil ja ms fa; do
  [ -f "$train_dir/$lang/samples.jsonl" ] && continue
  "$repo/pipeline/runner/run_dir.sh" "$round-$lang" train "$S/$round-$lang.jsonl" "$train_dir/$lang" || echo "FAILED $round-$lang"
done
echo "round $round finished $(date '+%F %T')"
