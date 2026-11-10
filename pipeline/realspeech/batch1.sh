#!/usr/bin/env bash
# 实验 v2 第一批真人语音：对已下载的各数据集走完导入与核对。已写出 manifest.jsonl 的目录直接跳过，可以中断后重新运行。
#
# 用法：pipeline/realspeech/batch1.sh asr   需要 ASR 核对的数据集，按顺序占用 GPU
#       pipeline/realspeech/batch1.sh cv    Common Voice（人工核验，只用 CPU），可以和 asr 同时运行
# 输入（需先下载，只下了每个数据集的一部分分片）：
#   $DATASET/audio/speechcolab/gigaspeech2/data/{th,id,vi}/{train/*.tar.gz,train_refined.tsv}
#   $DATASET/audio/MohammadJRanjbar/ParsVoice/{data,metadata}/*.parquet
#   $DATASET/audio/thivux/phoaudiobook/data/*.parquet
#   $DATASET/audio/QCRI/mgb2/train/{wav/*.wav,xml/utf8/*.xml}
#   $DATASET/audio/reazon-research/reazonspeech-v2/{data/*.tar,tsv/all.tsv}
#   $DATASET/audio/Wenetspeech4TTS/WenetSpeech4TTS/{Premium/*.tar.gz,DNSMOS_P808Scores/Premium_DNSMOS.lst}
#   $DATASET/audio/fsicoli/common_voice_22_0/{audio,transcript}/{de,fr,es}/   Common Voice 22.0 镜像的部分分片
#   $DATASET/audio/mozilla/common-voice-27/<语言>/                             Common Voice 27.0，只保留已验证录音
# 输出：$DATASET/audio/qwen3.5-asr/real-batch1/<名称>/
# 波斯语（ParsVoice）的训练目标没有标点：GPU 空出后启动 LLM 服务，再对其输出目录运行 pipeline/realspeech/punctuate.py。
set -uo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
run="$repo/pipeline/realspeech/run.sh"
real="$repo/pipeline/realspeech/config_real.yaml"
A="${DATASET:?需要设置 DATASET}/audio"
B="$A/qwen3.5-asr/real-batch1"
part="${1:?asr 或 cv}"

batch() {
  local name="$1"; shift
  [ -f "$B/$name/manifest.jsonl" ] && return 0
  "$@" || echo "FAILED $name"
}

if [ "$part" = asr ]; then
  # GigaSpeech 2（Whisper 加迭代重标）、ParsVoice（Google Web Speech）、PhoAudiobook（Whisper large-v3）的转写由早期 ASR
  # 生成：重新识别模式，主 ASR 的结果作训练目标，交叉 ASR 核对
  G="$A/speechcolab/gigaspeech2/data"
  for lang in th vi; do
    batch "gigaspeech2-$lang" env RETRANSCRIBE=1 "$run" "gigaspeech2-$lang" "$real" "$B/gigaspeech2-$lang" \
      --source gigaspeech2 --lang "$lang" --input "$G/$lang"/train/*.tar.gz
  done
  batch phoaudiobook-vi env RETRANSCRIBE=1 SPLIT_MAX_SECONDS=20 "$run" phoaudiobook-vi "$real" "$B/phoaudiobook-vi" \
    --source phoaudiobook --lang vi --input "$A"/thivux/phoaudiobook/data/*.parquet
  batch parsvoice-fa env RETRANSCRIBE=1 "$run" parsvoice-fa "$real" "$B/parsvoice-fa" \
    --source parsvoice --lang fa --input "$A"/MohammadJRanjbar/ParsVoice/data/*.parquet
  # MGB-2（半岛电视台网站的人工转写）和 ReazonSpeech（电视台字幕）是人工转写：按人工核验处理，不做 ASR 核对，
  # 转写作训练目标。MGB-2 只保留 WMER 不超过 10 的片段（数据集自带的对齐质量，排除转写与音频对不上的分段）
  batch mgb2-ar env HUMAN_VERIFIED=1 "$run" mgb2-ar "$real" "$B/mgb2-ar" \
    --source mgb2 --lang ar --input "$A"/QCRI/mgb2/train/wav/*.wav --max-wmer 10
  batch reazonspeech-ja env HUMAN_VERIFIED=1 "$run" reazonspeech-ja "$real" "$B/reazonspeech-ja" \
    --source reazonspeech --lang ja --input "$A"/reazon-research/reazonspeech-v2/data/*.tar
  # WenetSpeech4TTS 的转写由 Paraformer-large 生成：重新识别模式，与 GigaSpeech 2 相同
  batch wenetspeech4tts-zh env RETRANSCRIBE=1 "$run" wenetspeech4tts-zh "$real" "$B/wenetspeech4tts-zh" \
    --source wenetspeech4tts --lang zh --input "$A"/Wenetspeech4TTS/WenetSpeech4TTS/Premium/*.tar.gz
  batch gigaspeech2-id env RETRANSCRIBE=1 "$run" gigaspeech2-id "$real" "$B/gigaspeech2-id" \
    --source gigaspeech2 --lang id --input "$G"/id/train/*.tar.gz
elif [ "$part" = cv ]; then
  # Common Voice 的已验证录音视为人工核验：只做语言识别、FLEURS 去重和时长检查，不跑 ASR
  C="$A/fsicoli/common_voice_22_0/audio"
  for lang in de fr es; do
    batch "cv22-$lang" env HUMAN_VERIFIED=1 "$run" "cv22-$lang" "$real" "$B/cv22-$lang" \
      --source common_voice --lang "$lang" --input "$C/$lang"/*/*.tar
  done
  # Common Voice 27.0：下载完成（有 .done 标记）的语言才处理；同一句话最多取 3 条录音
  declare -A LANG_OF=([zh-CN]=zh [zh-TW]=zh [ja]=ja [fa]=fa [it]=it [ru]=ru [th]=th [pt]=pt [ar]=ar [tr]=tr [id]=id
                      [vi]=vi [ko]=ko [ms]=ms)
  for locale in zh-CN zh-TW ja fa it ru th pt ar tr id vi ko ms; do
    dir="$A/mozilla/common-voice-27/$locale"
    [ -f "$dir/.done" ] || continue
    batch "cv27-$locale" env HUMAN_VERIFIED=1 "$run" "cv27-$locale" "$real" "$B/cv27-$locale" \
      --source common_voice --lang "${LANG_OF[$locale]}" --input "$dir" --max-per-transcript 3
  done
fi
echo "batch1 $part finished $(date '+%F %T')"
