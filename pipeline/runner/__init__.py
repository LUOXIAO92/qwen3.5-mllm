"""合成—核对流水线（实验 v1，任务 T11）。

输入文本条目（带 script 朗读稿）、音色库 voices.jsonl 和按语言的配置，生成合成任务，按 TTS 模型分块合成，
检查音频时长与文本长度的比例，按语言调用主 ASR 和可选的交叉 ASR，用 pipeline.compare 比对，
写出识别结果、通过清单和统计。运行本流水线的 Python 只需要标准库和 PyYAML（读配置）；
TTS、ASR 封装用各自虚拟环境的 python 以子进程运行，需要 GPU 的经 ``<工作目录>/gpu.lock``。

用法（在仓库根目录运行）::

    python -m pipeline.runner run \\
        --config pipeline/runner/config_v1.yaml \\
        --work-dir <工作目录> \\
        --texts <文本条目 JSONL 或目录> \\
        --voices <voices.jsonl> \\
        --out-dir <输出目录> \\
        [--split train|test] [--langs ja en] [--limit-per-lang 10] \\
        [--shard K/N] [--fallback-original] [--stop-after-calls N]

子命令：``plan`` 只生成 tasks.jsonl；``run`` 生成任务并执行全部阶段、汇总；``report`` 只按已有的块文件重新汇总。
``--split train`` 只用非留出音色，``--split test`` 只用留出音色，两者分别运行、分别放在不同输出目录。

任务怎么生成
============

每条文本按「配置种子 + split + entry_id」得到独立的随机数序列，先按 ``langs.<语言>.tts`` 的权重抽 TTS 模型
（只在有可用音色的模型里抽），再从该模型可用的音色里均匀抽一个。音色属于该 TTS 模型、音色有 builtin_name
且该模型的 ``builtin_langs`` 含目标语言时用 ``builtin`` 模式，否则用参考片段克隆（``clone``，要求模型
``clone: true`` 且音色有 ref_audio，需要时还要有 ref_text）。``cross_lingual`` 记录音色的 ref_lang 与目标语言
是否不同。每个任务的 seed 由配置种子和 sample_id 导出。音色库中 ``ref_entry_id`` 指向的条目不会成为任务。
抽取与条目的处理顺序、分片、是否续跑都无关，同一份输入重跑得到完全相同的 tasks.jsonl。
sample_id 为 ``<split>-<entry_id>``。

执行与续跑
==========

任务先按 tts_model 分组，组内按 sample_id 排序后每 ``chunk_size`` 条切成一块；一块调用一次封装（每次 GPU
调用都经 flock）。块跑完才把块文件原子地写入 ``state/``，块文件名由块内 sample_id 的哈希决定。重新运行时已有
块文件的块直接跳过，其余整块重做，音频文件按 sample_id 命名，重做只会覆盖同名文件，所以中途终止后续跑的结果
与一次跑完完全一致。VoxCPM2、IndexTTS 只能逐条合成，其 ``batch_size`` 只影响写盘与分组粒度，不假设所有 TTS
都支持真正的批量。ASR 阶段对通过长度检查的样本运行：先跑主 ASR，再对主 ASR 已一致的样本跑交叉 ASR。

长度检查：音频时长除以朗读稿的非空白字符数，超出 ``langs.<语言>.sec_per_char`` 的 [下限, 上限] 的样本直接判失败，
原因是 ``length_too_long`` 或 ``length_too_short``，不送入 ASR。

分片：``--shard K/N`` 只处理 sample_id 哈希对 N 取余为 K 的任务，各片可以共用一个输出目录，
全部片完成后运行一次 ``report`` 得到整体结果。

输出（都在 --out-dir 下，音频路径相对该目录）
=============================================

- ``tasks.jsonl``：合成任务（CONVENTIONS 的字段，外加 original、source、split、holdout 等）
- ``audio/<sample_id>.flac``：合成音频
- ``asr_results.jsonl``：识别结果，每个样本每个 ASR 一行，含 sample_id、asr_model、hypothesis、match、
  compare_detail 和 asr_role（primary 或 cross）
- ``manifest.jsonl``：核对通过的清单（训练清单格式，另含 cross_asr_model、entry_id、mode、seed）
- ``samples.jsonl``：全部样本的判定（status 为 pass、fail 或 pending，fail 时给出 reason）
- ``stats.json`` 与 ``stats.md``：按语言、TTS 模型、音色、是否跨语言的通过率，以及失败原因分布
- ``state/``、``logs/``：块文件与各封装的日志

失败原因：``tts_failed``、``tts_unsupported``、``length_too_long``、``length_too_short``、``length_zero_duration``、
``asr_error``、``asr_empty``、``mismatch``，交叉 ASR 的失败加前缀 ``cross_``。

测试::

    python -m unittest pipeline.runner.tests.test_runner
"""
