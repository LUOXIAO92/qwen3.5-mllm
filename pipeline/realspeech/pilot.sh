#!/usr/bin/env bash
# 实验 v2 的真人语音小规模试验：每个来源、每种语言最多 2000 条，走完导入与核对，比较不同核对方式的通过率。
# 已写出 manifest.jsonl 的试验目录直接跳过，可以中断后重新运行。
#
# 用法：pipeline/realspeech/pilot.sh
# 输入（需先下载）：
#   $DATASET/audio/amphion/Emilia-Dataset/{Emilia,Emilia-YODAS}/{DE,FR}/*.tar   Hugging Face 上的 Emilia 分片
#   $MAKESENSE_EMILIA                                                         MakeSense 重新识别的 Emilia 中英日韩子集
#   $MAKESENSE_EMILIA_EXTRA（可选）                                           存放其余一部分音频的同结构目录
#   $DATASET/audio/disco-eth/WorldSpeech/data/<配置>/*.parquet                WorldSpeech 的若干训练分片
# 输出：$DATASET/audio/qwen3.5-asr/real-pilot/<名称>/
set -uo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
run="$repo/pipeline/realspeech/run.sh"
real="$repo/pipeline/realspeech/config_real.yaml"
qwen3text="$repo/pipeline/realspeech/config_real_qwen3text.yaml"
P="${DATASET:?需要设置 DATASET}/audio/qwen3.5-asr/real-pilot"
E="$DATASET/audio/amphion/Emilia-Dataset"
W="$DATASET/audio/disco-eth/WorldSpeech/data"
M="${MAKESENSE_EMILIA:?需要设置 MAKESENSE_EMILIA（MakeSense 的 Emilia-Dataset 目录）}"
N=2000

pilot() {
  local name="$1"; shift
  [ -f "$P/$name/manifest.jsonl" ] && return 0
  "$@" || echo "FAILED $name"
}

# Emilia 德语、法语：原始转写不可靠，用重新识别模式（主 ASR Qwen3-ASR 识别，交叉 ASR Parakeet 核对）
for lang in de fr; do
  code=$(echo "$lang" | tr a-z A-Z)
  pilot "emilia-$lang" env RETRANSCRIBE=1 "$run" "emilia-$lang" "$real" "$P/emilia-$lang" \
    --source emilia --lang "$lang" --input "$E"/Emilia/"$code"/*.tar "$E"/Emilia-YODAS/"$code"/*.tar --limit "$N"
done

# MakeSense 的 Emilia 中英日韩：转写已由 Qwen3-ASR 重新生成，中文、韩语改用 Fun-ASR 核对，英语用 Parakeet，日语用 Haas-ja
for lang in zh en ja ko; do
  pilot "makesense-$lang" "$run" "makesense-$lang" "$qwen3text" "$P/makesense-$lang" \
    --source makesense_emilia --lang "$lang" --input "$M" ${MAKESENSE_EMILIA_EXTRA:-} --limit "$N"
done

# WorldSpeech：人工转写对齐到片段上，两端常对不齐。A 模式用人工转写核对；转写没有标点或空格的语言用 ASR 结果作训练目标
declare -A CONFIG=([ko]=ko_kr [ms]=ms_my [id]=id_id [fil]=tl_ph [th]=th_th [tr]=tr_tr [fa]=fa_ir [ja]=ja_jp [zh]=zh_tw [ru]=ru_ru)
declare -A TARGET=([ko]=asr [ms]=asr [id]=transcript [fil]=asr [th]=asr [tr]=transcript [fa]=transcript [ja]=asr [zh]=transcript [ru]=transcript)
for lang in ko ms id fil th tr fa ja zh ru; do
  pilot "worldspeech-$lang" "$run" "worldspeech-$lang" "$real" "$P/worldspeech-$lang" \
    --source worldspeech --lang "$lang" --input "$W/${CONFIG[$lang]}"/*.parquet --target "${TARGET[$lang]}" --max-cer 0.5 --limit "$N"
done

# WorldSpeech B 模式：人工转写两端常对不齐，片段又长（中位数近 30 秒），整段逐字一致很难。复制 A 模式选出的条目，
# 在停顿处切成不超过 15 秒的段，改为重新识别：主 ASR 识别每段，交叉 ASR 核对
for lang in ko ms id fil th tr fa ja zh ru; do
  src="$P/worldspeech-$lang"
  dst="$P/worldspeech-$lang-split"
  [ -f "$dst/manifest.jsonl" ] && continue
  [ -f "$src/selected.jsonl" ] || continue
  mkdir -p "$dst"
  cp "$src/selected.jsonl" "$dst/"
  pilot "worldspeech-$lang-split" env RETRANSCRIBE=1 SPLIT_MAX_SECONDS=15 "$run" "worldspeech-$lang-split" "$real" "$dst"
done
echo "pilot finished $(date '+%F %T')"
