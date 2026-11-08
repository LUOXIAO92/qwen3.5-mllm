"""真人语音的导入与核对（实验 v2，任务 #32 之后的数据扩充）。

把下载好的真人语音数据集整理成合成—核对流水线（pipeline.runner）能直接读取的输出目录，后续的时长检查、
ASR 识别、比对和训练清单都复用 pipeline.runner。流程分三步：

1. select（envs/t03）：从数据集读出每条的转写和音频位置，清理转写的空白；用 GlotLID 检查转写的语言，
   去掉不是目标语言的条目（Emilia 等数据集的语言标签不可靠，例如法语子集里混有英语）；去掉与 FLEURS
   相同的句子（评测集不得进入训练）。写出 selected.jsonl 和 select_stats.json。
2. materialize（envs/mllm-eval）：把选中条目的音频解码成 16 kHz 单声道 FLAC，写出 tasks.jsonl、
   synth_results.jsonl 和 audio/，格式与合成流水线相同，tts_model 记数据集名，mode 记 real。
3. 用 pipeline.runner 依次运行 length、asr（主、交叉）、compare、manifest。

转写没有标点或大小写的数据集，导入时给任务加 target_from = asr：比对通过后，训练清单的训练目标改用主 ASR 的
识别结果（它已经与人工转写一致，又带标点和大小写），原转写保存在 source_transcript 字段。

核对用的 ASR 不能和生成转写的是同一个模型。例如 MakeSense 里重新识别的 Emilia 转写来自 Qwen3-ASR，而 v1 配置中
中文、英语、韩语的主 ASR 也是 Qwen3-ASR，这时用 config_real_qwen3text.yaml，改用 Whisper 核对。

数据集适配见 sources.py；完整的命令见 run.sh。
"""
