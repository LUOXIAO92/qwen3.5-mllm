# 训练数据生成与 projector 训练流水线

这里的脚本把实验 v1 的完整流程串起来：从现有文本数据集抽取多语种原文，改写成朗读稿，用 TTS 合成音频，用 ASR 核对音频内容，把核对通过的音频—文本对整理成训练清单，训练音频 projector，再在真人录音上评测。讲解见仓库根目录的教程 notebook：[02 训练数据准备](../02-training-data-preparation-zh.ipynb)、[03 训练与评测](../03-projector-training-zh.ipynb)；实验结果见 [results/v1-zh.md](../results/v1-zh.md)。

各阶段之间交换的文件格式见 [CONVENTIONS.md](CONVENTIONS.md)，各模型虚拟环境的建立方法见 [ENVIRONMENTS.md](ENVIRONMENTS.md)。下面的命令都在仓库根目录运行。`<工作目录>` 存放虚拟环境、GPU 锁和中间产出，驱动脚本通过环境变量 `WORK_DIR` 找到它；数据集放在环境变量 `DATASET` 指向的目录下。需要 GPU 的命令都经 `<工作目录>/gpu.lock` 排队，同一时间只有一个模型占用 GPU。

## 1. 真人录音评测集与排除清单

```bash
python -m pipeline.eval.fleurs_prepare --work-dir <工作目录>
python -m pipeline.train.fleurs_dev \
    --fleurs-dir <工作目录>/data/T02 \
    --output <工作目录>/data/T10/fleurs_dev.jsonl \
    --overfit-output <工作目录>/data/T10/overfit32.jsonl
```

第一条下载 FLEURS 18 种语言的 dev、test 录音，写出评测清单，并把 FLEURS 全部转写写入排除清单 `exclusion.jsonl`，后面抽取训练文本时去掉与评测集相同的句子。第二条把 dev 集整理成训练时用来算验证损失的清单。

## 2. 抽取原文

```bash
python -m pipeline.text.fetch_web hplt --dest $DATASET/text --lang all
python -m pipeline.text.extract --source tatoeba --lang all \
    --raw-dir <Tatoeba 导出目录> --exclusion <exclusion.jsonl> \
    --langid-model <GlotLID model_v3.bin> --out-dir <候选目录>
python -m pipeline.text.build --cand-dir <候选目录> --exclusion <exclusion.jsonl> --out-dir <part1>
python -m pipeline.text.extract_web --source hplt --lang all --raw-root $DATASET/text \
    --exclusion <exclusion.jsonl> --langid-model <GlotLID model_v3.bin> --out-dir <网页候选目录> --jobs 6
python -m pipeline.text.merge --part1-dir <part1> --cand-dir <网页候选目录> \
    --exclusion <exclusion.jsonl> --out-dir $DATASET/text/qwen3.5-asr/source-text/final
```

`extract` 处理短句来源（Tatoeba、Common Voice 句库、维基百科段落），`extract_web` 处理网页与新闻来源（HPLT、FineWeb2-HQ-Plus、Wikinews 等，先用 `fetch_web` 下载）。两者都做清理、过滤、去重、排除 FLEURS 句子、语言检测和按长度分档，`merge` 把两部分合成每种语言的最终条目。

## 3. 生成朗读稿

```bash
pipeline/tn/run_all.sh
```

对每种语言的最终条目调用 `pipeline.tn.run`，把数字、单位、缩写等改写成实际读法，写出 `$DATASET/text/qwen3.5-asr/scripts-v1/<语言>.jsonl`。无法可靠改写的条目标为 `dropped`，不参加合成。

## 4. 音色库

```bash
python -m pipeline.voices.build_voices \
    --text-dir <原文目录，含 <语言>.jsonl> \
    --out-dir <工作目录>/data/T07 \
    --tmp-dir <工作目录>/tmp/T07 \
    --gpu-lock <工作目录>/gpu.lock \
    --py-qwen3-tts <工作目录>/envs/tts-qwen3/bin/python \
    --py-supertonic3 <工作目录>/envs/tts-supertonic3/bin/python \
    --py-vieneu <工作目录>/envs/tts-vieneu/bin/python \
    --py-asr <工作目录>/envs/asr-qwen3/bin/python
```

为三个 TTS 模型的内置音色各合成一段参考朗读并核对，写出 `voices.jsonl`。这些参考朗读同时作为 OmniVoice、VoxCPM2、IndexTTS 声音克隆的参考录音。每个模型留出几个音色只用于测试集。

## 5. 抽取每轮的条目并合成、核对

```bash
python -m pipeline.runner.rounds test \
    --scripts-dir $DATASET/text/qwen3.5-asr/scripts-v1 \
    --voices <工作目录>/data/T07/voices.jsonl \
    --per-lang 60 \
    --output $DATASET/text/qwen3.5-asr/rounds/test-all.jsonl
python -m pipeline.runner.rounds train \
    --name r1 \
    --hours 5 \
    --config pipeline/runner/config_v1.yaml \
    --scripts-dir $DATASET/text/qwen3.5-asr/scripts-v1 \
    --voices <工作目录>/data/T07/voices.jsonl \
    --exclude $DATASET/text/qwen3.5-asr/rounds/test-all.jsonl \
    --out-dir $DATASET/text/qwen3.5-asr/rounds
pipeline/runner/run_round.sh r1
```

`rounds` 抽出测试集条目（每种语言 60 条）和一轮训练条目（按预计通过率换算，目标是每种语言约 5 小时通过的音频）。`run_round.sh` 先用留出音色合成测试集，再按语言依次调用 `run_dir.sh`：合成、检查时长、主 ASR 识别并比对、交叉 ASR 识别并比对、写出训练清单。`pipeline.runner` 每一步的细节见 `pipeline/runner/__init__.py`，各语言用哪些 TTS、ASR 模型见 `pipeline/runner/config_v1.yaml`。

## 6. 训练

```bash
python -m pipeline.runner.merge_manifests $DATASET/audio/qwen3.5-asr/v1/train/manifest-r1.jsonl \
    $DATASET/audio/qwen3.5-asr/v1/train/{zh,en,ja,ko,de,fr,ru,pt,es,it,id,ms,th,vi,fil,ar,tr,fa}
pipeline/train/run.sh $DATASET/audio/qwen3.5-asr/v1/train/manifest-r1.jsonl original <工作目录>/data/T14/v1b
```

第一条把 18 种语言的训练清单合成一份。合并真人语音时加 `--cap-hours 600 --uncapped zh en`，除中文、英语外每种语言最多收 600 小时；目录按命令行顺序占用上限，人工核对过的 Common Voice 写在前面，同一目录超出上限时按 `--seed` 随机取子集。第二条只训练音频 projector，训练目标为 `original`（原文）或 `script`（朗读稿）；超参数见脚本内的注释，额外参数追加在后面，例如 `--max-steps 20000 --warmup-steps 1000`。

## 7. 评测

```bash
python -m pipeline.eval.synth_manifest $DATASET/audio/qwen3.5-asr/v1/test <工作目录>/data/T16/holdout-manifest
pipeline/eval/run.sh <工作目录>/data/T14/v1b $DATASET/audio/qwen3.5-asr/fleurs <工作目录>/data/T16/v1b-fleurs
pipeline/eval/run.sh <工作目录>/data/T14/v1b <工作目录>/data/T16/holdout-manifest <工作目录>/data/T16/v1b-holdout
python -m pipeline.eval.report --out-dir <报告目录> \
    --fleurs v1b=<工作目录>/data/T16/v1b-fleurs \
    --synth v1b:holdout=<工作目录>/data/T16/v1b-holdout@<工作目录>/data/T16/holdout-manifest
```

`run.sh` 取该次训练中 dev 损失最低的检查点，转写评测集并按语言打分；`report` 把多次评测汇总成表格。
