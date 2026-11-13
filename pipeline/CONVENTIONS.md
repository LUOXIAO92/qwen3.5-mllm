# 实验约定

本文件是各次实验（v1、v2）任务共用的约定：数据放在哪里、各阶段之间交换什么格式、怎样共用唯一的一张 GPU。任务拆解和进度见 GitHub issue（标签 `exp-v1`、`exp-v2`，v2 总览为 #41）。修改本文件需要由协调者（主会话）完成，并在相关 issue 中说明。

## 目录

代码放在仓库的 `pipeline/` 下，按阶段分子目录：`text/`（抽取原文）、`tn/`（生成朗读稿）、`compare/`（比对规范化与数词统一）、`tts/`、`asr/`、`runner/`（合成—核对流水线）、`train/`、`eval/`。合成数据、下载的数据集和虚拟环境都不进仓库，统一放在工作目录（下文写作 `<工作目录>`）。驱动脚本通过环境变量 `WORK_DIR` 找到工作目录，没有设置时使用与仓库目录并列的 `qwen3.5-asr-work/`；各虚拟环境的建立方法见 [ENVIRONMENTS.md](ENVIRONMENTS.md)。工作目录包含：

- `data/<任务编号>/`：各任务的数据产出，例如 `data/T03/`。下游只读取上游目录，不在其中写文件。
- `envs/<名称>/`：用 `uv venv` 建立的虚拟环境，一个模型或一组相容的模型一个环境。conda 环境 `py3.14` 不安装、不升级任何包。合成—核对流水线（`pipeline.runner`）和比对（`pipeline.compare`）统一用 `envs/runner` 的 python 运行，这个环境装有 PyYAML 6.0.3、opencc 1.4.2、fugashi 1.5.2 和 unidic-lite 1.0.8，建立方法见 `pipeline/runner/__init__.py`。
- `src/<仓库名>/`：需要长期保留的上游源码克隆（含随源码存放的权重），例如可编辑安装的 TTS 源码、通过 `PYTHONPATH` 使用的推理包。
- `tmp/<任务编号>/`：可再生的中间文件，删除后不影响已有环境和产出。
- `logs/`：长时间任务的日志。
- `gpu.lock`：GPU 锁文件，见下文。

模型权重使用默认的 Hugging Face 缓存 `~/.cache/huggingface`。不要在家目录下创建其他目录；需要临时文件时放在 `qwen3.5-asr-work/tmp/` 下，系统 `/tmp` 重启即清空，不放任何需要保留的东西。数据集（下载的原始语料、评测集、原文、朗读稿、合成音频）放 `$DATASET` 下：本项目产出的音频为 `$DATASET/audio/qwen3.5-asr/<名称>`，文本为 `$DATASET/text/qwen3.5-asr/<名称>`，下载的原始语料为 `$DATASET/text/<机构>/<数据集名>`。不接受任何 gated 模型或数据集的条款，不调用付费 API。

## GPU

整机只有一张 RTX 5090（32 GB，sm_120，需要 CUDA 12.8 及以上的 PyTorch）。所有使用 GPU 的命令都通过锁文件排队：

```bash
flock <工作目录>/gpu.lock python -m pipeline.tts.omnivoice ...
```

冒烟测试控制在几分钟内；长时间任务（正式合成、训练）只由协调者启动。

## 数据格式

各阶段之间用 JSONL 交换数据，一行一个对象，UTF-8 编码，字段名使用英文。音频统一保存为 16 kHz 单声道 FLAC。

**文本条目**（T03 产出，T04 补充字段）：

| 字段 | 含义 |
| --- | --- |
| `entry_id` | 全局唯一，格式 `<lang>-<source>-<序号>` |
| `lang` | 语言代码，取值见下文 |
| `source` | 来源数据集，使用 `datasets.md` 中的名称 |
| `source_license` | 来源许可 |
| `source_ref` | 来源中的定位信息，例如文档编号或 URL |
| `original` | 原文，保持来源写法 |
| `n_chars` | 原文字符数 |
| `length_bin` | 长度档（T03 补充）：`short`（估计朗读时长 5 秒以内）、`medium`（5–15 秒）、`long`（15–30 秒） |
| `est_duration_s` | 按各语言估计语速（字符/秒）算出的朗读时长，单位秒（T03 补充） |
| `script` | 朗读稿（T04 补充） |
| `tn_status` | `unchanged`（无需改写）、`normalized`（已改写）或 `dropped`（无法可靠改写，丢弃） |
| `tn_note` | 改写或丢弃的原因，可为空 |

语言代码：`zh en ja ko de fr ru pt es it id ms th vi fil ar tr fa`。

**音色**（T07 产出）：`voice_id`、`source_model`、`builtin_name`、`gender`（已知时填写）、`ref_audio`、`ref_text`、`ref_lang`、`holdout`（是否留出不参与训练）。

**合成任务与结果**（T11 产出）：`sample_id`、`entry_id`、`lang`、`text`（送入 TTS 的朗读稿）、`tts_model`、`mode`（`builtin` 或 `clone`）、`voice_id`、`ref_lang`、`cross_lingual`、`seed`、`audio_path`、`duration_s`、`status`。

**识别结果**：`sample_id`、`asr_model`、`hypothesis`、`match`、`compare_detail`。

**核对通过的训练清单**（T13 产出）：合并上述字段，每行一个通过核对的样本，至少包含 `sample_id`、`audio_path`、`duration_s`、`lang`、`original`、`script`、`source`、`source_license`、`tts_model`、`voice_id`、`holdout`、`cross_lingual`、`asr_model`。可选字段 `offset_s`：虚拟切分的片段（`pipeline/realspeech/materialize.py`），`audio_path` 指向整段文件，样本是从 `offset_s` 秒开始、长 `duration_s` 秒的一段，读取时在内存中截取；切点记录在该输出目录的 `virtual_split.jsonl`。

## 训练与评测的对话格式

训练（T10）和评测推理（T15）使用同一种输入，保证评测时看到的提示与训练一致。用户消息的内容依次为一段音频和一句固定指令 `Transcribe the audio.`，不给语言提示；用阶段 1 模型自带的 chat template 展开，`add_generation_prompt = True`、`enable_thinking = False`，于是助手回合以空的思考块 `<think>\n\n</think>\n\n` 开头。训练时只对其后的目标文本和结尾的 `<|im_end|>` 计算损失；评测时从同一位置开始生成。

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text", "text": "Transcribe the audio."},
        ],
    },
]
```

## 模型封装的命令行接口

每个 TTS、ASR 模型在 `pipeline/tts/`、`pipeline/asr/` 下有一个封装模块，在该模型自己的虚拟环境中运行，只依赖该环境中的包和标准库。

```bash
# TTS：读取合成任务，逐条写出 <sample_id>.flac，并写出结果 JSONL
python -m pipeline.tts.<模型> --input jobs.jsonl --output-dir <目录> --batch-size <N>

# ASR：读取 sample_id、audio_path、lang，写出识别结果 JSONL
python -m pipeline.asr.<模型> --input audio.jsonl --output results.jsonl --batch-size <N>
```

封装必须关闭模型自带的文本规范化，输出采样率统一转为 16 kHz。每个封装的模块说明写明：支持的语言代码、需要的环境、显存占用和实测吞吐。

## 代码风格

遵守仓库根目录 `AGENTS.md` 的 Python 写法：支持关键字传参的接口显式写出 `参数名 = 值`，`=` 两侧各留一个空格，四个及以上参数逐行书写。脚本会作为教程的一部分对外发布，注释和说明使用中文，不写入本机以外的路径、账号或密钥。
