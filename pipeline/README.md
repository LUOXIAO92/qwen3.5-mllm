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

实验 v2 的训练清单还包含无人声片段（语言 `zxx`，训练目标为空），合并时用 `--uncapped zh en zxx` 整份收录。训练时无人声片段只学输出结束标记，`--empty-target-share`（默认 0.03）按抽样权重把它们控制在每轮样本的 3%；`--sample-weight 字段:取值=权重`（可重复，例如 `lang:fa=2`）让某些语言或来源多抽或少抽；加 `--noise-manifest $DATASET/audio/qwen3.5-asr/nonspeech/v2/background.jsonl` 后，30% 的语音样本（`--noise-prob`）在 0–20 dB 信噪比下实时混入背景声，`--noise-category-weight music=0.5` 这类参数调整各类背景声的比例。细节见 `pipeline/train/train.py`、`sampling.py` 和 `mixing.py` 的说明。

训练、评测和回放文本加载模型目录时都检查分词器（#77）：融合模型目录和参考模型目录的 `tokenizer_config.json` 要写 `Qwen3_5Tokenizer`，`tokenizer.json` 的预分词规则要与原始 Qwen3.5 相同（字母与组合用符号连在一起，`[\p{L}\p{M}]+`）。原始 Qwen3.5 目录写的是 `Qwen2Tokenizer`，transformers 5.16.1 按它加载时把泰语的元音与声调符号、阿拉伯语和波斯语的元音符号单独切开，FLEURS test 的泰语参考文本因此多出将近一倍的 token。检查不通过时程序停下，用 `python -m pipeline.tokenizer_fix <模型目录> --backup-suffix .before-77 --verify` 修正（只改这两个文件，可以重复运行），`pipeline/build_mllm.py` 构建新目录时会自动修正。

全量微调让语言模型连同 projector 一起训练，音频塔和视觉塔仍冻结：加 `--train-language-model --language-model-lr <语言模型的峰值学习率>`，projector 的峰值学习率仍由 `--lr` 给出；通常再加 `--init-projector <只训练 projector 的检查点目录>`，从已经训练过的 projector 出发。语言模型以 bfloat16 计算、优化器更新 float32 主权重，每个检查点另存一份 bfloat16 的 `language_model.safetensors`。精度、显存和检查点格式见 `pipeline/train/__init__.py`「全量微调」。

优化器默认是 AdamW。`--optimizer muon`（#78）是实验选项：代码与 CPU 测试已完成，还没有在 GPU 上做过训练对比，正式实验仍用 AdamW。加上它时，projector 的两个权重矩阵和语言模型各层的线性层（注意力、Gated DeltaNet 的投影、MLP）改用 PyTorch 自带的 Muon，词嵌入兼输出层、归一化层、偏置、`A_log`、`dt_bias`、短卷积等其余参数仍用 AdamW；2B 的语言模型中 13.73 亿个参数用 Muon、5.09 亿个（几乎全是词嵌入）用 AdamW，训练开始时打印分组。默认的 `--muon-lr-scaling match_rms_adamw` 把 Muon 的更新缩放到与 AdamW 相当的幅度，Muon 沿用 `--lr` 和 `--language-model-lr`（`--muon-lr`、`--language-model-muon-lr` 可以另给）；`original` 是 Keller Jordan 的缩放，必须另给学习率。动量、Nesterov、Newton–Schulz 迭代次数和权重衰减各有选项，语言模型 Muon 组的动量可以存为 bf16（`--muon-momentum-dtype`）。精度选项、梯度裁剪、学习率曲线、续训、`--stop-at-step` 和拼接批次照常可用；`--offload-optimizer` 只卸载 AdamW 部分，全量微调时不能用 `--lm-blocks-to-swap`。Muon 每个参数只有一个动量，默认精度下 2B 全量微调语言模型一侧常驻 28.4 GB（只用 AdamW 时 33.9 GB），但每步多出的正交化计算与语言模型线性层本身的计算相当。用途、显存表、计算量和文献见 `pipeline/train/__init__.py`「Muon 优化器」，实现见 `pipeline/train/muon.py`。

全量微调时可以混入自生成回放（#71 C 组，数据见第 9 节）：加 `--replay-dir $DATASET/text/qwen3.5-asr/replay-noloop --replay-groups free_raw=1 --replay-ratio 0.5`（需要 `--train-language-model` 和 `--pack-sequences`）。ASR 批次、训练计划和步数都不变，每一步按「回放损失 token 数 / ASR 损失 token 数」的长期平均为 r 整条取回放样本（从不截断），短样本接在 ASR 拼接行后面，超过 `--replay-inline-max-tokens`（默认 2,048）的长样本另成同一步内的纯文本微批。损失只算参考模型生成的部分：无模板组是全部生成 token（cont_raw 不含人写的文档开头），带模板的组只算助手轮（含结尾的 `<|im_end|>`，free_chat_think 含推理和 `</think>`），模型自己写的用户轮只作上下文。`--replay-groups` 写多组时按 token 数混合（例如五组各 `=1`）；`--replay-loss ntp`（默认）是对采样 token 的交叉熵，`kl32` 是对参考模型 top-32 的截断前向 KL（`--replay-kl-tail` 加尾部项）；`--replay-norm step`（默认）把一步内 ASR 与回放的全部损失 token 一起取平均。开启回放时训练日志的 `loss` 只统计 ASR，另有回放的总损失、各组损失和实际比例，取样记录在输出目录的 `replay_log.jsonl`，续训时抽样器从检查点恢复。做法与长样本对实际权重的影响见 `pipeline/train/replay.py` 开头，Colab 上第一轮的无人值守流程见 [colab/README.md](colab/README.md) 单元格 7 的 #71 C 组。

语言模型也可以用 LoRA 代替全量微调（#71）：加 `--language-model-lora-rank <秩>`（不加 `--train-language-model`），语言模型各解码层的全部线性层（Gated DeltaNet 层的 `in_proj_qkv`、`in_proj_z`、`in_proj_b`、`in_proj_a`、`out_proj`，全注意力层的 `q_proj`、`k_proj`、`v_proj`、`o_proj`，每层 MLP 的 `gate_proj`、`up_proj`、`down_proj`）加上低秩增量，基座权重、词嵌入兼输出层和归一化层都冻结，只训练增量和 projector。α 默认等于秩（`--language-model-lora-alpha`），LoRA 参数组的峰值学习率由 `--language-model-lora-lr` 给出（默认 1e-4），projector 仍用 `--lr`；回放、拼接批次、梯度检查点、续训和 dev 损失照常可用。每个检查点另存 `lora.safetensors`，并写出合并后的 `language_model.safetensors`，格式与全量微调的相同，评测程序直接使用；`python -m pipeline.train.lora merge` 从 `lora.safetensors` 重新生成它。2B 每个秩单位 1,051,200 个参数，秩 32 为 3,364 万个（语言模型参数的 1.79%），秩 90 为 9,461 万个（5.03%）。做法、显存和速度见 `pipeline/train/__init__.py`「语言模型的 LoRA」。

## 7. 评测

```bash
python -m pipeline.eval.synth_manifest $DATASET/audio/qwen3.5-asr/v1/test <工作目录>/data/T16/holdout-manifest
pipeline/eval/run.sh <工作目录>/data/T14/v1b $DATASET/audio/qwen3.5-asr/fleurs <工作目录>/data/T16/v1b-fleurs
pipeline/eval/run.sh <工作目录>/data/T14/v1b <工作目录>/data/T16/holdout-manifest <工作目录>/data/T16/v1b-holdout
python -m pipeline.eval.report --out-dir <报告目录> \
    --fleurs v1b=<工作目录>/data/T16/v1b-fleurs \
    --synth v1b:holdout=<工作目录>/data/T16/v1b-holdout@<工作目录>/data/T16/holdout-manifest
```

无人声误输出率（无人声片段中输出了非空文字的比例，按声音类别给出分布）用无人声评测集（语言 zxx，无参考文本）：

```bash
python -m pipeline.eval.synth_manifest $DATASET/audio/qwen3.5-asr/nonspeech/v2/eval <工作目录>/data/E1/manifest
LANGS=zxx pipeline/eval/run.sh <工作目录>/data/T14/v1b-20k <工作目录>/data/E1/manifest <工作目录>/data/E1/v1b-20k
python -m pipeline.eval.report --out-dir <报告目录> --nonspeech v1b-20k=<工作目录>/data/E1/v1b-20k
```

结果目录的 summary.json 中 `nonspeech` 项即该指标；`report --nonspeech` 把多个模型并列成表。

`run.sh` 取该次训练中 dev 损失最低的检查点，转写评测集并按语言打分，全量微调的检查点中的 `language_model.safetensors` 一并载入；提示格式取这次训练的 `prompt_format`（纯续写格式训练的是 cpt，环境变量 `PROMPT_FORMAT` 可以改）。`report` 把多次评测汇总成表格。

语音指令遵循探针（任务 #72）：同一批 FLEURS 音频（每种语言 30 条）换不同指令提问（转写及其改写、你听到了什么、翻译成英文或中文、拼接音频中某个时间段说了什么），统计照做与一律转写的比例；`--prompt-format chat cpt` 同时用纯续写格式的提示转写，`--cascade` 用参考转写代替音频，得到原始语言模型的参照。说明见 `pipeline/eval/instruct_probe.py` 开头。

```bash
flock <工作目录>/gpu.lock python -m pipeline.eval.instruct_probe \
    --model-dir qwen3.5-2b-mllm --projector <检查点>/projector.safetensors \
    --manifest-dir $DATASET/audio/qwen3.5-asr/fleurs --output-dir <工作目录>/data/T72/<名称>
flock <工作目录>/gpu.lock python -m pipeline.eval.instruct_probe \
    --model-dir qwen3.5-2b-mllm --cascade \
    --manifest-dir $DATASET/audio/qwen3.5-asr/fleurs --output-dir <工作目录>/data/T72/2b-cascade
```

全量微调后的遗忘与思考能力（#71）另有三个脚本，都用 `--model-dir <融合模型目录>` 加可选的 `--language-model <检查点>/language_model.safetensors`，不给时评测原始语言模型：

```bash
flock <工作目录>/gpu.lock python -m pipeline.eval.lm_ppl --model-dir qwen3.5-2b-mllm --language-model <检查点>/language_model.safetensors \
    --manifest-dir $DATASET/audio/qwen3.5-asr/fleurs --output <输出目录>/lm_ppl.json
flock <工作目录>/gpu.lock python -m pipeline.eval.replay_gap --model-dir qwen3.5-2b-mllm --language-model <检查点>/language_model.safetensors \
    --replay-dir $DATASET/text/qwen3.5-asr/replay-noloop --output-dir <输出目录>/replay_gap
flock <工作目录>/gpu.lock python -m pipeline.eval.think_check --model-dir qwen3.5-2b-mllm --language-model <检查点>/language_model.safetensors \
    --replay-dir $DATASET/text/qwen3.5-asr/replay-noloop --attn-implementation flash_attention_2 --fit-memory --output-dir <输出目录>/think
```

`lm_ppl` 是语言模型对 FLEURS 参考文本的每字困惑度（与 #66、#71 第一轮的计算相同）。`replay_gap` 在五组回放评测集上按训练时的损失位置算每 token 负对数似然，减去参考模型的（teacher 中的 `lp.f32`），得到前向 KL 的估计，free_chat_think 另分推理和回答。`think_check` 用 free_chat_think 评测集中参考模型自己写的 200 条提问开启思考生成（温度 1、不设 top-k / top-p，最多 16,384 个新 token），统计写完 `</think>` 的比例、推理与回答的长度和循环比例。`--attn-implementation flash_attention_2` 让语言模型的 6 个全注意力层用 FlashAttention-2（线性注意力层仍用 fla 的内核），载入或生成出错时自动改用 sdpa 从头重做，`summary.json` 的 `attn_implementation_used` 记下实际的实现；比较不同模型时要用同一种实现。在 32 GB 的显卡上加 `--fit-memory`，按剩余显存和注意力实现把 200 条分成几批（RTX 5090 上两种实现都是 3 批、每批 67 条，每条约 0.26 GiB 与 0.36 GiB），提问、温度和上限不变。`--smoke` 在 4 条左填充的提问上逐步比较 sdpa 与 FlashAttention-2 的分布，给出该用哪一种。

#71 C 组第一轮在 Colab 上只训练，全部评测在本机进行，由 `pipeline/eval/c71_local.py` 完成。从 Google Drive 下载每个运行 `runs/v2b-2b-cpt77-<c0|c-free_raw|c-free_chat|c-cont_raw|c-cont_chat|c-mix>/` 中的 `checkpoint-0030000/language_model.safetensors`（约 3.76 GB）、`projector.safetensors`、`projector.json`，以及运行目录中的 `train_config.json`、`train_log.jsonl`、`eval_log.jsonl`、`replay_log.jsonl`（C0 没有），保持同样的目录结构放到运行目录下（`--runs-root` 或环境变量 `C71_RUNS_ROOT` 指定，不指定时为 `<工作目录>/data/T71c/runs/`；这次放在 F 盘，见下面的启动命令），六个运行共约 22.8 GB，`trainer_state.pt` 不用下载。F 盘在 WSL 中是 9P 挂载，实测顺序读约 150 MB/s，每项评测载入语言模型约 25 秒。驱动程序对下载齐全的运行和参考模型做 Colab 驱动程序 `c_runs.py` 中同样的评测项目（命令由 `c_runs.eval_items` 生成，只换成本机的路径和 python），已有结果的项目跳过，单项失败记下后接着做，每一项持有 GPU 锁，结束后不留 GPU 进程；开始时先用参考模型做一次思考检查的冒烟比较（约 5 分钟），一致时所有模型的思考检查都用 FlashAttention-2，否则都用 sdpa；默认先对所有模型做回放差距与困惑度，再做 FLEURS、探针，最后做最费时的思考检查。结果、`summary/summary.md`（与 Colab 汇总同样的表格，另加与 C0 比较的两张表，并列出各模型思考检查实际所用的注意力实现）、冒烟比较的 `think_smoke/`、`status.json` 和 `driver.log` 都在 `<工作目录>/data/T71c/` 下。先用 `--dry-run` 检查输入、列出计划和时间估计，再在后台启动：

```bash
WORK_DIR=<工作目录> DATASET=<数据集目录> python -m pipeline.eval.c71_local --runs-root /mnt/f/models/qwen3.5-asr/c71-runs --dry-run
WORK_DIR=<工作目录> DATASET=<数据集目录> nohup setsid <工作目录>/envs/mllm-eval/bin/python -m pipeline.eval.c71_local \
    --runs-root /mnt/f/models/qwen3.5-asr/c71-runs >> <工作目录>/data/T71c/launch.out 2>&1 < /dev/null &
```

读音频的两项（FLEURS、探针）用 `envs/mllm-eval`（有 soundfile），纯文本的三项和冒烟比较用 `envs/mllm-train`（有 flash-linear-attention、flash-attn 与 mm_assistant_mask）。在 RTX 5090 上估计每个运行约 1.5 小时（思考检查约 34 分钟，用 sdpa 时约 49 分钟），参考模型约 42 分钟，全部约 9.6 小时；思考检查的估计按每步 K、V 的读写量和显存带宽算出，没有实测，驱动程序做完第一项后改按实测用时估计剩余时间。一晚做不完时用 `kill <status.json 中的 pid>` 停下，下次重新运行即可续跑。设置与续跑的细节见脚本开头。

#71 的文本基准衡量全量微调后的语言模型还保留多少 Qwen3.5-2B 原有的文本能力（`pipeline/eval/text_bench.py`，只用语言模型，不需要 projector）。MMLU-Redux（排除数据集标为有错的 278 题，用其余 2,722 题）、C-Eval 验证集（1,346 题）和 MMMLU 中与项目重合的 10 种语言（每种同一组 500 题）是零样本选择题，用非思考模式：提示要求只回答字母，不生成，比较助手回合第一个位置上 A、B、C、D 四个 token 的对数概率。IFEval（541 条）用非思考模式贪心生成，最多 1,280 个新 token，打分用 google-research 的官方代码（`pipeline/eval/ifeval/`，Apache-2.0，只改了导入和两个可以用标准库代替的依赖），给出 prompt 级与 instruction 级的 strict、loose 四个比例。GPQA Diamond（198 题）用思考模式，生成设置与思考检查相同（温度 1，最多 16,384 个新 token，全注意力层用 FlashAttention-2，按剩余显存分批），答案从 `</think>` 之后的 `"answer": "X"` 或 `\boxed{X}` 中解析，没写完 `</think>` 或找不到答案都算错；它的使用条款要求不公开题目，结果文件只记录题目编号、选项顺序与对错。数据放在 `$DATASET/text/<机构>/<数据集名>`，来源、许可和下载版本见 `datasets.md`「文本基准（#71）」。

`pipeline/eval/bench_local.py` 先对参考模型和 C 组六个运行各做一遍四项简单基准（每个模型一个进程，模型只载入一次），全部做完后再对每个模型做 GPQA。已有结果的项目跳过，单项失败记下后接着做，GPU 锁、显存等待和进程组的处理与 `c71_local.py` 相同（直接使用它的 `Driver`）。IFEval 的打分需要 langdetect 与 nltk（≥ 3.9.1，带 punkt_tab 数据），项目的 `envs/mllm-train`、`envs/mllm-eval` 都没有，用 `--py-ifeval` 指定另一个有这两个包的 python，只用 CPU；不给时只生成、不打分，以后补上参数重新运行即可补打分。结果在 `<工作目录>/data/T71c/bench/<模型>/`，`summary.md` 列出各模型的分数、与参考模型之差、模型卡上的分数、MMMLU 各语言和诊断量（四个字母的概率和、IFEval 回答长度、GPQA 写完 `</think>` 的比例等）。在 RTX 5090 上估计每个模型的简单基准约 6 分钟、GPQA 约 42 分钟（参考模型的思考检查实测 40.7 分钟，同样是 3 批、上限 16,384），七个模型共约 5.7 小时：

```bash
WORK_DIR=<工作目录> DATASET=<数据集目录> python -m pipeline.eval.bench_local --runs-root <运行目录> --py-ifeval <python> --dry-run
WORK_DIR=<工作目录> DATASET=<数据集目录> nohup setsid <工作目录>/envs/mllm-train/bin/python -m pipeline.eval.bench_local \
    --runs-root <运行目录> --py-ifeval <python> >> <工作目录>/data/T71c/bench/launch.out 2>&1 < /dev/null &
```

用户 2026-11-27 决定本机评测一律改用 vLLM 的 docker 镜像（`vllm/vllm-openai:nightly-x86_64`，vLLM 0.34），把 GPU 用满、尽快出结果。各评测程序（`think_check`、`text_bench` 的选择题与 IFEval、`run_eval`、`instruct_probe`、`lm_ppl`、`replay_gap`）都有 `--engine {hf,vllm}`：提示、打分和输出文件不变，只把模型的计算交给容器中的 `pipeline/eval/vllm_generate.py`，结果中另记 `engine` 与 vLLM 的设置；共用的部分（导出语言模型、写任务目录、启动容器）在 `pipeline/eval/vllm_host.py`，做法与 GPQA 的 vLLM 版相同。生成类评测用 vLLM 的连续批处理；选择题用 `logprob_token_ids` 取 A–D 四个 token 在全词表上的对数概率；困惑度与回放差距用 `prompt_logprobs` 取提示中每个位置实际 token 的对数概率。读音频的两项先在宿主机上按 transformers 版同样的批用音频塔与 projector 算出音频向量（`pipeline/eval/audio_embeds.py`），再以混合输入交给 vLLM：音频占位符的位置用这些向量，其余位置按 token id 查词嵌入；音频 token 与文字一样按顺序排位置，与融合模型相同。两个驱动程序的 `--engine` 默认为 vllm（`bench_local` 的 `--gpqa-engine` 也默认 vllm），默认顺序改为一个模型做完全部项目再做下一个，每个运行的语言模型只导出一次、做完删除；`c71_local` 另做严格版指令探针（`probe_strict/`）。思考检查按每条提问的种子采样（`--seed` 加序号），与 transformers 版统计上一致、逐条不同；贪心类评测两种引擎的差别只来自内核的 bfloat16 舍入，与 transformers 自身换注意力实现时的差别相当。核对结果与用时见 #71。在 RTX 5090 上每个运行的六项约 20 分钟（transformers 约 1.6 小时，其中思考检查 40 分钟、探针 27 分钟），四项简单基准连导出约 3 分钟。两个驱动程序共用导出目录（`<工作目录>/tmp/T71c/vllm-lm/<标签>`），要先后运行，不要同时运行：

```bash
RUNS=c0,c-free_raw,c-free_chat,c-cont_raw,c-cont_chat,c-mix,c-cont_raw-kl32,c-cont_think-kl32,c-cont_raw-kl32tail,c-cont_raw-kl32tail-lr5e-6,c-cont_raw-kl32tail-lr1e-6,c-cont_raw-kl32tail-lora32,c-cont_raw-kl32tail-lora90
mkdir -p <输出目录>/bench
WORK_DIR=<工作目录> DATASET=<数据集目录> nohup setsid bash -c "
    <工作目录>/envs/mllm-eval/bin/python -m pipeline.eval.c71_local --runs-root <运行目录> --runs $RUNS --out-dir <输出目录>
    <工作目录>/envs/mllm-train/bin/python -m pipeline.eval.bench_local --runs-root <运行目录> --runs $RUNS \
        --out-dir <输出目录>/bench --benches mmlu_redux,ceval,mmmlu,ifeval --py-ifeval <python>
" >> <输出目录>/launch.out 2>&1 < /dev/null &
```

系统提示遵循评测（#71）看全量微调后的模型是否还听从系统消息。C 组的训练数据都没有系统回合（ASR 为不套模板的续写格式，回放的对话组从 `<|im_start|>user` 开始），上面的评测也都把要求写在用户回合，所以另做这一项。文本部分（`pipeline/eval/sysprompt_bench.py`）用本仓库编写的 100 个中性英文开放问题（`pipeline/eval/sysprompt/tasks.jsonl`），每条加一个 IFEval 指令注册表中的约束：约束原文由注册表的 `build_description` 写出，打分用同一个类的检查与官方的 strict、loose 判定，不用其他外部指令数据。共 21 类、每类 14 条，按固定种子生成 294 条（`pipeline/eval/sysprompt/items.jsonl`，`build` 子命令可以原样重新生成；各类的参数范围与排除的 4 类及原因见脚本开头）。每条在三种条件下贪心生成（非思考模式，最多 1,280 个新 token）：A 不提要求，B 约束作为系统消息、问题作为用户消息，C 没有系统消息、约束接在问题之后。A 是碰巧满足约束的比例，主要看各模型的 B 与参考模型的 B 之差，以及 B − C。语音部分在 `instruct_probe` 上加 `--instruction-placement system`：严格版的转写与翻译指令放在系统消息中，用户回合只有音频，样本与严格版探针相同，与已有的 `probe_strict/`（指令在用户回合）对照；参考模型的语音用原始语言模型加 projector 只训练时的检查点（全量微调的起点），两种位置都在这里做。`pipeline/eval/sysprompt_local.py` 对参考模型和各运行依次做这几项（文本生成后用 `--py-ifeval` 打分），GPU 锁、显存等待与进程组的处理与 `c71_local.py` 相同，导出目录单独放在 `<工作目录>/tmp/T71e-sysprompt/` 下。结果在 `<输出目录>/sysprompt/`，`summary.md` 列出全部模型的表格，语音部分另给译文用目标语言文字写成的比例（只看是否照指令换了语言，与识别准不准无关）。在 RTX 5090 上实测每个运行约 2.7 分钟（含导出；文本 882 条约 1.4 分钟，语音 2,100 条约 1.3 分钟），参考模型与 13 个运行共 40 分钟：

```bash
WORK_DIR=<工作目录> python -m pipeline.eval.sysprompt_local --runs-root <运行目录> --py-ifeval <python> --dry-run
WORK_DIR=<工作目录> nohup setsid <工作目录>/envs/mllm-eval/bin/python -m pipeline.eval.sysprompt_local \
    --runs-root <运行目录> --py-ifeval <python> >> <输出目录>/sysprompt/launch.out 2>&1 < /dev/null &
```

## 8. 在 Google Colab 上训练

本机的训练可以在 Colab 上续训。训练音频按训练计划打包成约 1 GB 的 tar 分片上传到 Google Drive，Colab 上只在本地保留常驻分片和一段滚动窗口，训练程序按位置读出原始字节再解码（`--audio-shards`）。打包、上传、Colab 单元格和存储桶测试的步骤见 [colab/README.md](colab/README.md)。

## 9. 自生成回放文本

全量微调会让语言模型遗忘原有能力。回放的做法是在训练中混入参考模型（未经训练的 Qwen3.5-2B 语言模型，即阶段 1 融合模型中的原始权重）自己采样的文本（任务 #70，参见 arXiv 2605.26097）。脚本在 `pipeline/replay/`，完整流程由 `run.sh` 串起来：

```bash
WORK_DIR=<工作目录> DATASET=$DATASET MLLM=<2B 融合模型目录> ORIGINAL=<原始 Qwen3.5-2B 目录> \
    bash pipeline/replay/run.sh
```

各步依次为：`export` 把融合模型中的语言模型导出成 vLLM 可加载的目录；`prefixes` 从为 TTS 下载的 HPLT 3.0、FineWeb2-HQ-Plus、fineweb-edu、Wikinews 中按 18 种语言均匀抽取文档开头（排除含 FLEURS 句子的文档）；`prompts` 写出四组的提示；`generate` 用 vLLM 在温度 1、不设 top-k / top-p 的设置下采样，每个样本最多 16k token，同时保存每个位置的 top-20 对数概率；`finalize` 在循环起点截断并写出最终索引；`langid` 用 GlotLID 判断生成文本的语言；`nll` 计算参考模型在评测文本上的每 token 负对数似然；`report` 写出汇总。可以只跑其中几步，例如 `bash pipeline/replay/run.sh finalize langid report`。

四组回放文本是 free_raw（从一个换行开始自由生成，不加模板）、free_chat（从用户轮开头开始，模型自己写用户消息和助手回答）、cont_raw（续写文档开头）、cont_chat（文档开头放进用户轮并加续写指令，助手续写）。带模板的两组与 ASR 训练用同一套 chat template 设置。每组分为训练用的 `train` 和只作评测的 `eval`，两者的文档互不重叠。数据放在 `$DATASET/text/qwen3.5-asr/replay/<组>/<划分>/`，存放格式见 `pipeline/replay/store.py`，每个样本的字段见 `finalize.py`。

`generate` 生成时每 256 个 token 检查一次循环，发现循环后再多生成 1,024 个 token 就中止这条样本，最终在循环起点截断（规则见 `degeneration.py`）；无模板两组的样本有一半左右会陷入循环，不提前中止时大部分算力花在循环上。vLLM 0.28 对只有一个 token 的提示有状态残留的问题（见 `first_token.py`），所以 free_raw 的第一个生成 token 由 `prompts` 用 transformers 按参考模型的分布抽好，vLLM 从两个 token 的提示接着生成。`generate` 每次持有 GPU 锁约 12 分钟（`TIME_BUDGET`），时间用完就把还没写完的样本连同已生成的 token 存进 `raw/inflight.pkl` 并释放锁，下次运行先把它们接着生成（提示为原提示加已生成的 token），所以每次持锁的时间不受最长样本的影响；每次占用 GPU 之前先看 `nvidia-smi`，有其他进程占着显存时等待。

训练时建议直接使用保存的 token id（提示片段的 `ids` 加生成片段的 `[offset, offset + keep)`），不要把文本解码后重新分词。#77 修正之前，transformers 5.16.1 按原始目录 `tokenizer_config.json` 中的 `Qwen2Tokenizer` 加载 Qwen3.5 分词器，换掉了 `tokenizer.json` 中的预切分规则（原规则 `[\p{L}\p{M}]+` 变成 `\p{L}+`），泰语的元音、声调符号和阿拉伯语的元音符号都会把词切开：模型自己写出的泰语文本重新分词后 token 数是采样时的 1.1–1.8 倍。现在 `export` 把参考模型目录的分词器类改为 `Qwen3_5Tokenizer`，各步加载分词器后都检查切法。修正之前写出的 cont_raw、cont_chat 提示仍是旧的切法：泰语几乎全部、阿拉伯语约四成、波斯语约 6% 的文档开头与修正后的分词不同（#77）。free_chat 约 2% 的助手回答在空思考块之后又写了一段思考（含 `</think>`），按聊天模板重新渲染时会被拆成思考与回答两部分，用 token id 训练不受影响。

### 不筛选的自回放与丢弃循环的回放

按 arXiv 2605.26097 的做法，先生成了一份完全不筛选的回放文本，放在 `$DATASET/text/qwen3.5-asr/replay-unfiltered/<组>/<划分>/`，与上面筛选过的 `replay/` 并存。生成的设置与筛选版相同（同一个参考模型、四组的开头与模板、温度 1、不设 top-k / top-p、每条最多 16,384 个生成 token），区别在于：`generate --no-filter` 不检查循环、不提前中止，每条样本都生成到 `<|im_end|>` / `<|endoftext|>` 或 16,384 个 token 的上限；`finalize --no-filter` 不截断、不丢弃，循环起点（`loop_onset`、`pre_loop_tokens`）、gzip 压缩率和语言只作为元数据记录。生成时只保存采样到的 token id 和它自身的对数概率（`--topk 0`），每个 token 8 字节。`prompts` 沿用 `replay/prefixes/` 中的文档开头和 free_raw、free_chat 的提示文件（这两组的提示只有换行和聊天模板的特殊 token，不受分词器问题影响），续写两组的提示按修正后的分词器（#77）重新写出，并与 tokenizers 库直接读取原始 `tokenizer.json` 的结果逐条核对过。

不筛选的结果中，无模板两组约六成样本陷入循环，大多一直重复到 16k 上限，循环部分占这两组生成 token 的四分之三以上。用户据此决定（2026-11-22）：有循环（原样或近似，规则见 `degeneration.py`）的样本整条丢弃，其余样本原样保留，任何样本都不截断，避免截断的文本教会模型说到一半停下。保留下来的样本放在另一个目录 `$DATASET/text/qwen3.5-asr/replay-noloop/`，`replay-unfiltered/` 原样保留，便于以后比较：

```bash
FILTER=noloop WORK_DIR=<工作目录> DATASET=$DATASET MLLM=<2B 融合模型目录> ORIGINAL=<原始 Qwen3.5-2B 目录> \
    bash pipeline/replay/run.sh
```

各步依次为 `import think_prompts generate finalize langid estimate teacher verify report`。`import`（`noloop.py`）把 `replay-unfiltered/` 中已经生成完的样本逐条检查：没有循环的把 token 复制过来，有循环的只记一行元数据（状态 `loop`、出现循环的片段、起点、种类和花掉的 token 数），暂停在途的样本也一并带过来。`generate --drop-loops` 接着按提示清单的顺序生成，每 256 个 token 检查一次循环，发现就立即中止、整条丢弃，每段生成结束时再完整检查一次；循环检测只看前文，在前缀上发现的循环在完整序列上同样存在，所以提前中止得到的保留集合与「全部生成完再丢弃有循环的样本」完全相同，算力却省下大半。`finalize --drop-loops` 对保留的样本按同一套规则再检查一遍。目标按保留的 token 数计：每组 `eval` 约 100 万，free_chat、cont_chat 的 `train` 约 3,000 万，无模板两组的 `train` 约 1,000 万（用户 2026-11-22 调低）。每个划分的样本是提示清单前若干条中没有循环的全部样本；保留的样本比全部生成的样本短（长样本更容易陷入循环），长度分布偏向短文本，`summary.md` 给出两者的长度分布和丢弃比例。

为了保留思考能力，丢弃循环的目录另有一组 free_chat_think（用户 2026-11-22 加入）：与 free_chat 一样从 `<|im_start|>user\n` 开始，模型自己写用户消息，写到 `<|im_end|>` 后按 chat template 在 `enable_thinking = True` 时的生成提示打开助手轮（补上 `\n<|im_start|>assistant\n<think>\n`），模型接着写思考、`</think>` 和正式回答，直到 `<|im_end|>` 或 16k 上限。循环检查覆盖整个助手轮（思考也在内），有循环的样本同样整条丢弃；生成的 token 原样保存，不重新渲染。`finalize` 记录每条样本 `</think>` 的位置（`think_closed`、`reasoning_tokens`、`answer_tokens`）。直到结束符或 16k 上限都没有写出 `</think>` 的样本（大多一直想到上限，条数不多但占很多 token）与循环一样整条不用，不截断（用户 2026-11-22 决定）：`drop_reason` 为 `think_unclosed`，token 仍留在 `raw/` 中，不计入保留数量，也不做 top-32。生成时加 `--count-closed-think`，进度只计写出 `</think>` 的样本。`summary.md` 给出写出率、思考与回答的长度分布，以及与 free_chat 对比的循环丢弃比例。目标为评测用约 100 万、训练用约 1,000 万保留 token（`think_prompts` 一步写出提示，free_chat_think 只在丢弃循环的目录中）。

为了保持对系统提示的遵循（#71 的系统提示评测中，用对话模板回放的运行在系统提示上比用户轮多掉 10–14 个百分点，原有的带模板组都没有系统轮），另有两组从系统轮开始的回放：free_sys_chat 与 free_sys_chat_think（用户 2026-11-27 加入，没有系统轮的各组不变）。提示只有 `<|im_start|>system\n`，参考模型自己写系统提示，写到 `<|im_end|>` 后程序补上 `\n<|im_start|>user\n`，模型再写用户消息，写到 `<|im_end|>` 后按 chat template 的生成提示打开助手轮（free_sys_chat 补上 `\n<|im_start|>assistant\n<think>\n\n</think>\n\n`，free_sys_chat_think 补上 `\n<|im_start|>assistant\n<think>\n`），模型写回答（思考组先写思考和 `</think>`）。补上的各段由 `prompts.system_chat_segments` 从 chat template 对系统消息加用户消息的渲染切出，拼出的 token 与 chat template 渲染完整三轮对话的结果逐个相同（`pipeline/replay/tests/test_system_prompts.py`）。三段合计不超过 16k token，采样设置、循环检查和整条丢弃的规则与 free_chat 相同；系统提示或用户消息没有以 `<|im_end|>` 结束的样本没有回答，整条不用（`drop_reason` 为 `no_system_end` 或 `no_user_end`）。训练时系统提示和用户消息只作上下文，只对助手内容和结尾的 `<|im_end|>` 计算损失（`pipeline/train/replay.py` 中两组的来源编号为 6、7）。提示由 `run.sh sys_prompts` 写出，不在默认计划中。

从空的系统轮开始时，试运行（每组 400 条）中 42% 的系统提示只有 1–3 个词，20% 的用户消息为空。用户 2026-11-28 决定改为先让参考模型回答一句手写的元指令，把它写出的回答当作系统提示（`meta_prompts.py`）。元指令用项目的 18 种语言分别写成，并要求用该语言写；每条由主题行（34 个日常与专业领域，编程只是其中之一）、8 种措辞之一（人物或角色、任务助手、可检查的格式规则、语气与风格、安全与拒绝规则、分步辅导、具体机构的助手、很短的系统提示）、写成对助手的指示的要求和结尾要求组成，结尾要求不以「你是一位专家」开头、只输出系统提示本身；每种语言一半的元指令另加一句，要求至少一条能从回答中检查的规则。元指令这一轮只用于生成：系统提示一段的请求以「用户轮为元指令、助手轮按 `enable_thinking = False` 打开」为提示，写出的内容作为系统提示片段存下（记 `"context": "meta"`，它的对数概率是在元指令上下文中采样的），之后各段的提示和保存的样本都只有 `<|im_start|>system\n` 开始的三轮对话，与上面相同。每条样本记下元指令的语言、措辞、主题和是否要求规则（`meta_lang`、`meta_kind`、`meta_domain`、`meta_rule`、`meta_version`）。生成时系统提示为空、内容太少、含思考标签、以代码围栏开头或第一行是「系统提示」之类的标签，以及用户消息为空、含思考标签或抄写系统提示的，丢掉这一段、用同样的上下文重新采样（`generate --resample-system 4 --resample-user 4`），次数用完的样本整条不用，`finalize` 按同样的规则再检查一遍。

正式生成之前在 free_sys_chat 上做了三轮小规模检查（每种语言 40 条），通过标准事先写好：系统提示的语言与要求一致 ≥ 80%（GlotLID），最常见的开头 ≤ 10%（每种语言内 ≤ 25%），以被禁止的开头开始的 ≤ 3%，近似重复（字符 5-gram Jaccard > 0.7）< 5%，过滤后空的或抄写的用户消息为 0，人工判读 72 条中至少三分之一带可验证约束，人工判读 40 条中至少 70% 的回答按系统提示作答。第 1 轮只有可验证约束不达标（15/72），第 2 轮加了要求规则的一句后约束达标（38/72），回答按系统提示作答的比例差一条（27/40），第 3 轮加了写成对助手的指示的要求、碎片的判定改为按文字加权的字母数不足 20 后全部达标（语言一致 94.9%，最常见开头 0.6%，约束 24/72，按系统提示作答 28/40）。不达标的样本多是菲律宾语、阿拉伯语、土耳其语、马来语等语言中不成文的文本，这是参考模型本身的能力。

正式数据每种语言 free_sys_chat 的训练用 703 条、评测用 20 条提示，free_sys_chat_think 训练用 128 条、评测用 20 条，比目标多约 15%，用来补足丢弃的样本；`finalize --lang-quota eval=17,train=611,free_sys_chat_think:train=111` 按提示清单的顺序在每种语言中只用前这么多条保留的样本（提示清单按语言交错排列，截取与长度无关），多出的 `drop_reason` 为 `lang_quota`。两组放在 F 盘的 `<F 盘数据集目录>/text/qwen3.5-asr/replay-noloop/` 下，`$DATASET/text/qwen3.5-asr/replay-noloop/` 中的两个组目录是指向它们的符号链接，训练照常用 `--replay-dir $DATASET/text/qwen3.5-asr/replay-noloop` 读取；上传到 Colab 的打包方法见 [colab/README.md](colab/README.md)。命令：

```bash
FILTER=noloop SYS_META=eval=20,train=703 SYS_THINK_META=eval=20,train=128 \
    PLAN=free_sys_chat:eval,free_sys_chat_think:eval,free_sys_chat:train,free_sys_chat_think:train \
    TARGET=eval=1000000000,train=1000000000 LANG_QUOTA=eval=17,train=611,free_sys_chat_think:train=111 \
    ONLY_GROUPS="free_sys_chat free_sys_chat_think" \
    WORK_DIR=<工作目录> DATASET=$DATASET MLLM=<2B 融合模型目录> ORIGINAL=<原始 Qwen3.5-2B 目录> \
    bash pipeline/replay/run.sh sys_prompts generate finalize langid estimate teacher verify
```

目标 token 数设得很大，生成在提示清单用完时结束，条数由提示清单决定。

丢弃循环的目录中还准备了第五组 prompt_chat 的提示，暂不生成（用户 2026-11-22 决定先集中做模态对齐）：用户消息是指令数据集中真人写的提问，由参考模型写回答，提示由 `user_prompts.py` 写出（来源与用法见 `datasets.md`「回放文本的用户提问」）。提问来自 CohereLabs/aya_dataset 的 `inputs` 和 OpenAssistant/oasst2 每棵对话树的第一条用户消息，只取项目的 18 种语言，评测用取自两者的 test / validation 划分（1,840 条），训练用取自 train 划分（71,995 条），两者没有相同的提问；按规范化文本去重，少于 3 个 token 或多于 2,048 个 token 的丢弃，每个划分中任何一种语言最多占 25%。对话格式与 cont_chat 相同。提示文件在 `replay-noloop/prompt_chat/<划分>/prompts.jsonl`，需要时用 `PLAN=prompt_chat:eval@instruct,prompt_chat:train@instruct` 生成（`@` 后面是提示清单的 `part` 字段，进度按这一部分计）。

`estimate` 统计各划分保留的生成 token 数和下一步需要的磁盘空间；`teacher` 再用参考模型把每条保留的样本（提示加生成，就是保存的 token id）整条前向一遍，在每个生成位置保存参考模型概率最高的 32 个 token，供以后试 top-k 截断的近似 KL 损失。`teacher` 用 transformers 加载融合模型（与训练同一套实现），输出层用 bfloat16 输入、float32 输出计算全部 248,320 个 logit，再取 log-sum-exp 和前 32 个；`verify` 抽几条样本单独前向，在 float64 上算整个词表的 log softmax 与保存的数对照。结果放在每个划分的 `teacher/` 目录，与 `raw/` 的 shard 一一对应，第 i 行就是 raw shard 中第 i 个生成 token 的位置：

| 文件 | 类型 | 内容 |
| --- | --- | --- |
| `shard-<编号>.top_ids.i32` | int32，每行 32 个 | 概率最高的 32 个 token id，从高到低 |
| `shard-<编号>.top_lp.f16` | float16，每行 32 个 | 这 32 个 token 的对数概率，即 logit 减 log-sum-exp |
| `shard-<编号>.lse.f32` | float32 | 整个词表上的 log-sum-exp |
| `shard-<编号>.lp.f32` | float32 | 实际生成的 token 的对数概率 |

每个生成 token 占 200 字节。只有 `samples.jsonl` 中 `keep` 为真的样本做了计算（已完成的样本记在 `teacher/progress.jsonl`）；free_chat_think 中没有写出 `</think>` 的样本仍在 `raw/` 中，对应的行没有计算，读取时以 `keep` 为准。参考模型给前 32 个 token 的概率是 `p_i = exp(top_lp_i)`，前 32 个之外的尾部概率总量是 `1 - Σ p_i`；原始 logit 是 `top_lp_i + lse`，需要换温度时可以由它重新计算前 32 个的相对概率。对数概率用 float16 存放，概率最高的几个 token 对数概率接近 0，误差最小（绝对值小于 1 时误差不超过 0.0005）。用 top-32 近似 KL 时，常见的两种写法是只在前 32 个上把参考分布重新归一化，或者把尾部概率并成一个「其他」类别，两者都偏离完整的 KL，偏差大小取决于尾部概率，`report` 给出了它的分布。`lp.f32` 取负后的平均值就是参考模型在这些文本上的每 token 负对数似然，`report` 用它给出评测集的基线（`teacher_nll`）。两个目录各有自己的 `summary.md`：`replay-unfiltered/` 的汇总给出全部生成样本的长度、达到上限的数量、循环比例和语言分布；`replay-noloop/` 的汇总给出保留与丢弃的样本数和 token 数、保留样本的长度分布、语言分布和参考 NLL。
