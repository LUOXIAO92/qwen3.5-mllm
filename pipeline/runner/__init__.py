"""合成—核对流水线（实验 v1，任务 T11）。

输入文本条目（带 script 朗读稿）、音色库 voices.jsonl 和按语言的配置，生成合成任务，按 TTS 模型分块合成，
检查音频时长与文本长度的比例，按语言调用主 ASR 和可选的交叉 ASR，用 pipeline.compare 比对，
写出识别结果、通过清单和统计。本流水线用工作目录的 ``envs/runner`` 运行，这个环境装有 PyYAML 6.0.3（读配置）、
opencc 1.4.2（中文繁简转换）、fugashi 1.5.2 和 unidic-lite 1.0.8（日语按读音比对）；比对与朗读稿生成用同一种
繁简转换，日语可以按读音比较（见 pipeline.compare）。TTS、ASR 封装用各自虚拟环境的 python 以子进程运行，
需要 GPU 的经 ``<工作目录>/gpu.lock``。

建立环境（在工作目录运行）::

    uv venv --python 3.12 envs/runner
    uv pip install --python envs/runner/bin/python pyyaml==6.0.3 opencc==1.4.2 fugashi==1.5.2 unidic-lite==1.0.8

用法（在仓库根目录运行，``python`` 指 ``<工作目录>/envs/runner/bin/python``）。每个子命令都是独立的一步，只读取上一步写在 ``--out-dir`` 里的文件，启动时按
pipeline/CONVENTIONS.md 检查输入格式（缺文件、缺字段、语言代码不对时报输入错误，退出码 2）::

    C="--config pipeline/runner/config_v1.yaml --work-dir <工作目录> --out-dir <输出目录>"
    python -m pipeline.runner plan $C --texts <条目 JSONL 或目录> --scripts <朗读稿 JSONL> --voices <voices.jsonl> \\
        [--split train|test] [--langs ja en] [--limit-per-lang 10] [--fallback-original]
    python -m pipeline.runner tts $C [--shard K/N]            # 合成，写 synth_results.jsonl 与 audio/
    python -m pipeline.runner length $C                       # 时长比例检查，写 length_check.jsonl
    python -m pipeline.runner asr $C --role primary [--shard K/N]   # 主 ASR，写 asr_primary.jsonl
    python -m pipeline.runner compare $C                      # 比对，写 asr_results.jsonl
    python -m pipeline.runner asr $C --role cross             # 交叉 ASR，只处理主 ASR 已一致的样本
    python -m pipeline.runner compare $C
    python -m pipeline.runner manifest $C                     # 训练清单、samples.jsonl、统计
    python -m pipeline.runner all $C ...                      # 按上面的顺序依次执行全部（不含分片）

``tts`` 与 ``asr`` 的 ``--models a b`` 只处理列出的模型，所以 CPU 模型（如 supertonic3）与 GPU 模型可以在两个进程里
同时跑同一个输出目录（块文件按模型分开，progress.json 合并各进程写入的阶段）；配置里 ``processes`` 大于 1 的模型同时开多个
封装子进程，各处理一块，整个模型阶段只持一次 gpu.lock。配置里的 ``prefer_builtin: true``（可按语言覆盖）让抽到的模型在有
内置音色可用时只用内置音色，其余语言用克隆。批次与吞吐的测量用 ``pipeline.runner.bench``，结果表用 ``bench_table``。

``--scripts`` 的朗读稿按 entry_id 合并进条目，``tn_status`` 为 dropped 的条目不合成。``--split train`` 只用非留出
音色，``--split test`` 只用留出音色，两者分别运行、分别放在不同输出目录。每一步可以单独重跑：已完成的块直接跳过，
下游文件按 sample_id 排序重写，内容不变。

进度：每个阶段在每个块完成时、以及等待封装期间每 ``--progress-interval`` 秒（默认 30）写一行日志到标准错误和
``<out-dir>/logs/progress.log``：阶段、模型、已完成数/总数、样本/秒、合成（或识别）音频秒/墙钟秒、预计剩余时间；
同时更新 ``<out-dir>/progress.json``（各阶段数字、计数器、整体通过情况，manifest 之后含最终通过率）。
速度含排队等待 GPU 锁的时间，续跑时已完成的块不计入速度。

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

分片：``tts``、``asr`` 的 ``--shard K/N`` 只处理 sample_id 哈希对 N 取余为 K 的任务，各片共用一个输出目录；
所有片的 tts 完成后运行 ``length``，再各片运行 asr，最后运行 compare 与 manifest。

输出（都在 --out-dir 下，音频路径相对该目录）
=============================================

- ``tasks.jsonl``：合成任务（CONVENTIONS 的字段，外加 original、source、split、holdout 等）
- ``audio/<sample_id>.flac``、``synth_results.jsonl``：合成音频与每个样本的合成结果（status、duration_s）
- ``length_check.jsonl``：时长比例检查结果（ok、reason、sec_per_char）
- ``asr_primary.jsonl``、``asr_cross.jsonl``：ASR 原始识别结果（hypothesis、error）
- ``asr_results.jsonl``：比对后的识别结果，每个样本每个 ASR 一行，含 sample_id、asr_model、hypothesis、match、
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
