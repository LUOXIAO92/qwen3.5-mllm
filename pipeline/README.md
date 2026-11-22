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

优化器默认是 AdamW。加 `--optimizer muon`（#78）时，projector 的两个权重矩阵和语言模型各层的线性层（注意力、Gated DeltaNet 的投影、MLP）改用 PyTorch 自带的 Muon，词嵌入兼输出层、归一化层、偏置、`A_log`、`dt_bias`、短卷积等其余参数仍用 AdamW；2B 的语言模型中 13.73 亿个参数用 Muon、5.09 亿个（几乎全是词嵌入）用 AdamW，训练开始时打印分组。默认的 `--muon-lr-scaling match_rms_adamw` 把 Muon 的更新缩放到与 AdamW 相当的幅度，Muon 沿用 `--lr` 和 `--language-model-lr`（`--muon-lr`、`--language-model-muon-lr` 可以另给）；`original` 是 Keller Jordan 的缩放，必须另给学习率。动量、Nesterov、Newton–Schulz 迭代次数和权重衰减各有选项，语言模型 Muon 组的动量可以存为 bf16（`--muon-momentum-dtype`）。精度选项、梯度裁剪、学习率曲线、续训、`--stop-at-step` 和拼接批次照常可用；`--offload-optimizer` 只卸载 AdamW 部分，全量微调时不能用 `--lm-blocks-to-swap`。Muon 每个参数只有一个动量，默认精度下 2B 全量微调语言模型一侧常驻 28.4 GB（只用 AdamW 时 33.9 GB），但每步多出的正交化计算与语言模型线性层本身的计算相当。用途、显存表、计算量和文献见 `pipeline/train/__init__.py`「Muon 优化器」，实现见 `pipeline/train/muon.py`。

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

### 不筛选的纯自回放

按 arXiv 2605.26097 的做法，另有一份完全不筛选的回放文本，放在 `$DATASET/text/qwen3.5-asr/replay-unfiltered/<组>/<划分>/`，与上面筛选过的 `replay/` 并存，方便以后比较两者。生成的设置与筛选版相同（同一个参考模型、四组的开头与模板、温度 1、不设 top-k / top-p、每条最多 16,384 个生成 token），区别只在于：`generate --no-filter` 不检查循环、不提前中止，每条样本都生成到 `<|im_end|>` / `<|endoftext|>` 或 16,384 个 token 的上限；`finalize --no-filter` 不截断、不丢弃，全部原样保留，循环起点（`loop_onset`、`pre_loop_tokens`）、gzip 压缩率和语言只作为元数据记录；free_chat 的用户消息没有正常结束时，这条样本只有用户消息，同样保留。因为样本都要写完，长样本与短样本一样计入，每个划分的样本就是提示清单的前若干条。生成时只保存采样到的 token id 和它自身的对数概率（`--topk 0`），每个 token 8 字节。

```bash
FILTER=0 WORK_DIR=<工作目录> DATASET=$DATASET MLLM=<2B 融合模型目录> ORIGINAL=<原始 Qwen3.5-2B 目录> \
    bash pipeline/replay/run.sh
```

各步依次为 `export prompts generate finalize langid estimate teacher verify report`。`prompts` 沿用 `replay/prefixes/` 中的文档开头和 free_raw、free_chat 的提示文件（这两组的提示只有换行和聊天模板的特殊 token，不受分词器问题影响），续写两组的提示按修正后的分词器（#77）重新写出，并与 tokenizers 库直接读取原始 `tokenizer.json` 的结果逐条核对过。每个划分的目标是全部生成 token 数：`eval` 约 100 万，`train` 约 3,000 万（`TARGET`）。在途的样本都要写完，所以 `generate` 按各组的平均样本长度（`PRIOR_MEAN`，由筛选版估计：无模板两组约有一半样本陷入循环、一直写到上限，平均约 8,000–10,000 token；带模板两组约 1,000 token）决定投放多少提示，避免超出目标太多。

`estimate` 统计各划分的生成 token 数和下一步需要的磁盘空间；`teacher` 再用参考模型把每条样本（提示加生成，就是保存的 token id）整条前向一遍，在每个生成位置保存参考模型概率最高的 32 个 token，供以后试 top-k 截断的近似 KL 损失。`teacher` 用 transformers 加载融合模型（与训练同一套实现），输出层用 bfloat16 输入、float32 输出计算全部 248,320 个 logit，再取 log-sum-exp 和前 32 个；`verify` 抽几条样本单独前向，在 float64 上算整个词表的 log softmax 与保存的数对照。结果放在每个划分的 `teacher/` 目录，与 `raw/` 的 shard 一一对应，第 i 行就是 raw shard 中第 i 个生成 token 的位置：

| 文件 | 类型 | 内容 |
| --- | --- | --- |
| `shard-<编号>.top_ids.i32` | int32，每行 32 个 | 概率最高的 32 个 token id，从高到低 |
| `shard-<编号>.top_lp.f16` | float16，每行 32 个 | 这 32 个 token 的对数概率，即 logit 减 log-sum-exp |
| `shard-<编号>.lse.f32` | float32 | 整个词表上的 log-sum-exp |
| `shard-<编号>.lp.f32` | float32 | 实际生成的 token 的对数概率 |

每个生成 token 占 200 字节。参考模型给前 32 个 token 的概率是 `p_i = exp(top_lp_i)`，前 32 个之外的尾部概率总量是 `1 - Σ p_i`；原始 logit 是 `top_lp_i + lse`，需要换温度时可以由它重新计算前 32 个的相对概率。对数概率用 float16 存放，概率最高的几个 token 对数概率接近 0，误差最小（绝对值小于 1 时误差不超过 0.0005）。用 top-32 近似 KL 时，常见的两种写法是只在前 32 个上把参考分布重新归一化，或者把尾部概率并成一个「其他」类别，两者都偏离完整的 KL，偏差大小取决于尾部概率，`report` 给出了它的分布。`lp.f32` 取负后的平均值就是参考模型在这些文本上的每 token 负对数似然，`report` 用它给出评测集的基线（`teacher_nll`）。汇总写在 `replay-unfiltered/summary.md`，包括各划分的样本与 token 数、长度分布、达到上限的样本数、出现循环的样本比例、语言分布和参考 NLL。
