# v1 流水线环境搭建手册

本文说明如何重建 v1 流水线用到的全部虚拟环境、上游源码克隆和模型权重。流水线的每个 TTS、ASR 封装（`pipeline/tts/*.py`、`pipeline/asr/*.py`）在自己的虚拟环境里以子进程运行，环境与模型的对应关系写在 `pipeline/runner/config_v1.yaml` 的 `tts_models` 和 `asr_models` 中。下文所有版本号和提交号都读取自已经能正常运行的那套环境。

## 1. 工作目录布局

文中的 `<工作目录>` 指放在仓库之外的工作目录，所有路径都相对它；数据集放在 `$DATASET` 下。

```text
<工作目录>/
├── envs/          # 每个环境一个子目录，用 uv venv 建立
├── src/           # 需要长期保留的上游源码克隆（含随源码存放的权重）
├── gpu.lock       # 所有 GPU 任务共用的锁文件，用 flock <工作目录>/gpu.lock <命令> 排队
├── data/ logs/ reports/ tmp/ tools/   # 数据、日志、报告、临时文件
```

本文的命令默认在 `<工作目录>` 下运行。环境的版本以 `uv pip freeze --python envs/<名称>/bin/python` 的输出为准，下面只列出关键包。

### 1.1 两类环境

建立环境前先分清两类，它们的 `uv venv` 写法不同。

| 类别 | 说明 | 包含的环境 |
| --- | --- | --- |
| 独立环境 | 自带 Python，不继承任何包 | runner、tn、全部 `tts-*`、asr-gigaam、asr-nemo |
| 叠加在 conda 环境上 | 用 `--system-site-packages` 继承项目 conda 环境 `py3.14`（Python 3.14.7、torch 2.13.0+cu130、torchaudio 2.11.0、transformers 5.16.1、numpy 2.4.6），只在上面补装少量包 | asr-qwen3、asr-nemo-fa、asr-hf、mllm-train、mllm-eval |

conda 环境 `py3.14` 本身不安装、不升级任何包。叠加类环境的创建命令是：

```bash
uv venv --python <conda py3.14 的 python> --system-site-packages envs/<名称>
```

其中 `<conda py3.14 的 python>` 是 conda 环境 `py3.14` 里的 `bin/python`。

### 1.2 PyTorch 的 CUDA 版本

TTS 环境用 CUDA 12.8 的 PyTorch 构建（RTX 5090 的 sm_120 可用），安装时加 PyTorch 官方的 cu128 索引：

```bash
--index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
```

asr-hf 和 asr-gigaam 里的 torch 2.15.0 是 `+cu132` 构建，asr-nemo 里的 torch 2.14.0 是 `+cu130` 构建，索引分别换成 `https://download.pytorch.org/whl/cu132` 和 `https://download.pytorch.org/whl/cu130`。

## 2. 环境一览

| 环境 | Python | 使用者 | 类别 |
| --- | --- | --- | --- |
| `envs/runner` | 3.12.13 | `pipeline.runner`、`pipeline.compare`（调度与比对） | 独立 |
| `envs/tn` | 3.14.7 | `pipeline.tn`（文本规范化） | 独立 |
| `envs/tts-omnivoice` | 3.11.15 | `pipeline.tts.omnivoice` | 独立 |
| `envs/tts-voxcpm2` | 3.12.13 | `pipeline.tts.voxcpm2` | 独立 |
| `envs/tts-qwen3` | 3.12.13 | `pipeline.tts.qwen3_tts` | 独立 |
| `envs/tts-indextts25` | 3.11.15 | `pipeline.tts.indextts25` | 独立 |
| `envs/tts-vieneu` | 3.12.13 | `pipeline.tts.vieneu` | 独立 |
| `envs/tts-supertonic3` | 3.12.13 | `pipeline.tts.supertonic3` | 独立 |
| `envs/asr-qwen3` | 3.14.7 | `pipeline.asr.qwen3_asr`、`pipeline.asr.whisper` | 叠加 conda |
| `envs/asr-nemo-fa` | 3.14.7 | `pipeline.asr.whisper_fa`、`pipeline.asr.nemotron_farsi` | 叠加 conda |
| `envs/asr-gigaam` | 3.12.13 | `pipeline.asr.gigaam` | 独立 |
| `envs/asr-nemo` | 3.12.13 | `pipeline.asr.haas_ja` | 独立 |
| `envs/asr-hf` | 3.14.7 | `pipeline.asr.typhoon`、`pipeline.asr.jais_asr` | 叠加 conda |
| `envs/mllm-train` | 3.14.7 | 阶段 3 训练（`pipeline/train/run.sh`） | 叠加 conda |
| `envs/mllm-eval` | 3.14.7 | 真人语音评测（`pipeline/eval/run.sh`） | 叠加 conda |
| `envs/asr-funasr-mlt` | 3.14.7 | `pipeline.asr.fun_asr_mlt`（v2，建立命令见 #34） | 叠加 conda |
| `envs/asr-omni` | 3.12.13 | `pipeline.asr.omni_asr`（v2，建立命令见 #35） | 独立 |
| `envs/asr-cohere` | 3.14.7 | `pipeline.asr.cohere_transcribe`（v2，建立命令见 #38） | 叠加 conda |
| `envs/asr-nemotron` | 3.12.13 | `pipeline.asr.nemotron_transcribe`（v2，vLLM 0.32.0 加 NeMo Speech 插件，建立命令见 #37；运行时占满整张 32 GB 显卡） | 独立 |

Parakeet（`pipeline.asr.parakeet_v3`）用 `envs/asr-nemo`。

`envs/` 下还有 `data-tools`、`t03`、`t04b`、`t05-dev`，它们是早期任务留下的环境，`config_v1.yaml` 没有引用，本文不覆盖。

## 3. 上游源码克隆

需要从源码安装或通过 `PYTHONPATH` 使用的仓库克隆在 `<工作目录>/src/`，每个仓库固定到下表的提交。

```bash
cd <工作目录>/src
git clone <地址> <目录名>
git -C <目录名> checkout <提交>
```

| 目录 | 仓库地址 | 提交 | 被谁使用 |
| --- | --- | --- | --- |
| `OmniVoice` | https://github.com/k2-fsa/OmniVoice | `08be0b4ccbac3e13e374e86fbfead4b4cac343e2` | tts-omnivoice（非可编辑安装） |
| `VoxCPM` | https://github.com/OpenBMB/VoxCPM | `cce58a5b59303c9bd63d12f23afe6a49a4a80c59` | tts-voxcpm2（非可编辑安装） |
| `Qwen3-TTS` | https://github.com/QwenLM/Qwen3-TTS.git | `022e286b98fbec7e1e916cb940cdf532cd9f488e` | tts-qwen3（可编辑安装） |
| `index-tts` | https://github.com/index-tts/index-tts.git | `d9e41aac89fd00b3d71497fddb287b7f24613712` | tts-indextts25（可编辑安装），权重在其 `checkpoints/` |
| `Haas-ASR` | https://github.com/third-intelligence/Haas-ASR.git | `acc614f68769dcbfae9dba96b48439c4a66295af` | asr-nemo（通过 `PYTHONPATH` 使用，不安装） |

`src/` 下还有 `VieNeu-TTS`、`nanovllm-voxcpm`、`supertonic` 三个克隆和 `supertonic3-model` 目录。当前 `config_v1.yaml` 引用的环境都不依赖它们（tts-vieneu、tts-supertonic3 直接从 PyPI 安装），重建 v1 环境时不需要克隆。它们的提交分别是 `VieNeu-TTS` 为 `04044e1b8f822f946bafb2c88b716459f4164152`（https://github.com/pnnbao97/VieNeu-TTS），`nanovllm-voxcpm` 为 `5fa1d74d7ad6072661d3c3da02ac734a752a993b`（https://github.com/a710128/nanovllm-voxcpm），`supertonic` 为 `1e9799e964ea4c0dad7cde993b65c3c813a7b373`（https://github.com/supertone-inc/supertonic）。

## 4. 调度与文本规范化环境

### 4.1 `envs/runner`

供 `pipeline.runner` 与 `pipeline.compare` 使用，包括任务规划、调度子进程、日语按读音比对和中文繁简转换。创建命令与 `pipeline/runner/__init__.py` 中记录的一致：

```bash
uv venv --python 3.12 envs/runner
uv pip install --python envs/runner/bin/python pyyaml==6.0.3 opencc==1.4.2 fugashi==1.5.2 unidic-lite==1.0.8
```

### 4.2 `envs/tn`

供 `pipeline.tn`（文本规范化）使用。Python 3.14.7，独立环境，已安装的包是 num2words 0.5.14、opencc 1.4.2、pymorphy3 2.0.6、pymorphy3-dicts-ru 2.4.417150.4580142、dawg2-python 0.9.0、docopt 0.6.2。创建这个环境时用的原始命令没有记录，下面的命令按已安装的包整理而成：

```bash
uv venv --python 3.14 envs/tn
uv pip install --python envs/tn/bin/python num2words==0.5.14 opencc==1.4.2 pymorphy3==2.0.6
```

俄语规范化依赖 pymorphy3，它会带入上面的词典包。

## 5. TTS 环境

TTS 环境都是独立环境，PyTorch 一律用 cu128 构建。下面每个环境的安装命令里，`$TORCH` 代表：

```bash
TORCH="--index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple"
```

这些环境最初是手工建立的，下面的命令按已安装的版本整理，依赖解析由 uv 完成。

### 5.1 `envs/tts-omnivoice`

使用者：`pipeline.tts.omnivoice`。模型 `k2-fsa/OmniVoice`。Python 3.11.15。

| 包 | 版本 |
| --- | --- |
| torch / torchaudio | 2.8.0+cu128 |
| transformers | 5.22.1 |
| omnivoice | 0.2.1（来自 `src/OmniVoice`，非可编辑） |
| accelerate | 1.15.0 |
| librosa | 0.11.0 |
| numpy | 2.4.6 |
| huggingface-hub | 2.4.0 |

```bash
uv venv --python 3.11 envs/tts-omnivoice
uv pip install --python envs/tts-omnivoice/bin/python $TORCH torch==2.8.0 torchaudio==2.8.0
uv pip install --python envs/tts-omnivoice/bin/python transformers==5.22.1 accelerate==1.15.0 librosa==0.11.0 numpy==2.4.6 pyyaml==6.0.3 soundfile==0.14.0
uv pip install --python envs/tts-omnivoice/bin/python src/OmniVoice
```

### 5.2 `envs/tts-voxcpm2`

使用者：`pipeline.tts.voxcpm2`。模型 `openbmb/VoxCPM2`。Python 3.12.13。

| 包 | 版本 |
| --- | --- |
| torch / torchaudio | 2.11.0+cu128 |
| torchcodec | 0.17.0 |
| transformers | 5.22.1 |
| voxcpm | 来自 `src/VoxCPM`（非可编辑，版本号 0.0.post1+gcce58a5b5） |
| funasr / modelscope | 1.4.21 / 1.40.2 |
| datasets / hydra-core | 3.6.0 / 1.3.8 |
| librosa / numpy | 1.0.0 / 2.5.4 |

```bash
uv venv --python 3.12 envs/tts-voxcpm2
uv pip install --python envs/tts-voxcpm2/bin/python $TORCH torch==2.11.0 torchaudio==2.11.0
uv pip install --python envs/tts-voxcpm2/bin/python transformers==5.22.1 pyyaml==6.0.3 soundfile==0.14.0
uv pip install --python envs/tts-voxcpm2/bin/python src/VoxCPM
```

funasr、modelscope、datasets、hydra-core、librosa、torchcodec 由 VoxCPM 的依赖带入。

### 5.3 `envs/tts-qwen3`

使用者：`pipeline.tts.qwen3_tts`。模型 `Qwen/Qwen3-TTS-12Hz-1.7B-Base`（克隆模式）和 `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`（内置音色模式）。Python 3.12.13。

| 包 | 版本 |
| --- | --- |
| torch / torchaudio | 2.11.0+cu128 |
| transformers | 4.57.3 |
| qwen_tts | 0.1.1（来自 `src/Qwen3-TTS`，可编辑安装） |
| accelerate | 1.12.0 |
| onnxruntime | 1.31.1 |
| librosa / numpy | 1.0.0 / 2.5.4 |

```bash
uv venv --python 3.12 envs/tts-qwen3
uv pip install --python envs/tts-qwen3/bin/python $TORCH torch==2.11.0 torchaudio==2.11.0
uv pip install --python envs/tts-qwen3/bin/python transformers==4.57.3 accelerate==1.12.0 pyyaml==6.0.3 soundfile==0.14.0
uv pip install --python envs/tts-qwen3/bin/python -e src/Qwen3-TTS
```

### 5.4 `envs/tts-indextts25`

使用者：`pipeline.tts.indextts25`。模型 `IndexTeam/IndexTTS-2.5`。Python 3.11.15。这个环境按上游 `uv.lock` 原样安装（任务 #12 的记录），所以版本由上游锁文件决定。

| 包 | 版本 |
| --- | --- |
| torch / torchaudio | 2.8.0+cu128 |
| transformers | 4.52.1 |
| indextts | 2.0.0（来自 `src/index-tts`，可编辑安装） |
| accelerate | 1.8.1 |
| librosa / numpy | 0.10.2.post1 / 2.2.6 |
| cn2an / modelscope | 0.5.22 / 1.27.0 |

```bash
uv venv --python 3.11 envs/tts-indextts25
cd src/index-tts
uv sync --frozen --python ../../envs/tts-indextts25/bin/python   # 按上游 uv.lock 安装；具体参数没有记录，以上游 README 为准
```

这个环境的具体安装命令没有记录，上面是按"原样使用上游 `uv.lock`"整理的做法，建议重建后用 `uv pip freeze` 与下表的版本核对。权重放在 `src/index-tts/checkpoints/`（含 `config.yaml`、`gpt.pth`、`s2mel.pth` 等），辅助模型放在 `src/index-tts/checkpoints/hf_cache/`（含 `w2v-bert-2.0`、`campplus_cn_common.bin`、`bigvgan`、`semantic_codec`）。权重下载方式没有记录，来源是 Hugging Face 的 `IndexTeam/IndexTTS-2.5`，封装通过 `--model-dir <工作目录>/src/index-tts/checkpoints` 读取（`config_v1.yaml` 里已配置）。

### 5.5 `envs/tts-vieneu`

使用者：`pipeline.tts.vieneu`。模型 `pnnbao-ump/VieNeu-TTS-v3-Turbo` 和编解码器 `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano`。Python 3.12.13。

| 包 | 版本 |
| --- | --- |
| torch / torchaudio | 2.8.0+cu128 |
| transformers | 4.57.6 |
| vieneu | 3.7.1（PyPI） |
| onnxruntime | 1.31.1 |
| librosa / numpy | 1.0.0 / 2.5.4 |

```bash
uv venv --python 3.12 envs/tts-vieneu
uv pip install --python envs/tts-vieneu/bin/python $TORCH torch==2.8.0 torchaudio==2.8.0
uv pip install --python envs/tts-vieneu/bin/python vieneu==3.7.1 transformers==4.57.6 pyyaml==6.0.3 soundfile==0.14.0
```

封装用 `snapshot_download` 把两个仓库固定到提交 `61b85e3d937fbbacb387714180e8182823512523`（VieNeu-TTS-v3-Turbo）和 `6aa02b01e445cc585582cf0ba480bc3ea6c8dd68`（MOSS-Audio-Tokenizer-Nano），第一次运行时自动下载。

### 5.6 `envs/tts-supertonic3`

使用者：`pipeline.tts.supertonic3`（CPU，不经 `gpu.lock`）。模型 `Supertone/supertonic-3`。Python 3.12.13，没有 PyTorch。

| 包 | 版本 |
| --- | --- |
| supertonic | 1.3.1（PyPI） |
| onnxruntime | 1.31.1（CPU 版） |
| numpy | 2.5.4 |

```bash
uv venv --python 3.12 envs/tts-supertonic3
uv pip install --python envs/tts-supertonic3/bin/python supertonic==1.3.1 onnxruntime==1.31.1 pyyaml==6.0.3 soundfile==0.14.0
```

模型文件第一次运行时由 supertonic 包下载，位置由环境变量 `SUPERTONIC_CACHE_DIR` 指定，也可以用 `--model-dir` 指向已有目录。`<工作目录>/src/supertonic3-model/` 保存着一份模型目录（含 `onnx/`、`voice_styles/`），它的下载方式没有记录。

## 6. ASR 环境

### 6.1 `envs/asr-qwen3`

使用者：`pipeline.asr.qwen3_asr`（模型 `Qwen/Qwen3-ASR-1.7B-hf`）和 `pipeline.asr.whisper`（模型 `openai/whisper-large-v3`）。Python 3.14.7，叠加在 conda 环境上，PyTorch 和 transformers 5.16.1 都来自 conda（torch 2.13.0+cu130）。环境里额外安装的只有 numpy 2.5.4、soundfile 0.14.0 及其依赖。

```bash
uv venv --python <conda py3.14 的 python> --system-site-packages envs/asr-qwen3
uv pip install --python envs/asr-qwen3/bin/python numpy==2.5.4 soundfile==0.14.0
```

### 6.2 `envs/asr-nemo-fa`

使用者：`pipeline.asr.nemotron_farsi`（模型 `mehdi-hf/nemotron-asr-streaming-farsi`）和 `pipeline.asr.whisper_fa`（模型 `MohammadGholizadeh/whisper-large-v3-turbo-persian`）。Python 3.14.7，叠加在 conda 环境上，torch 2.13.0+cu130 来自 conda。nemotron_farsi 用 Transformers 的 `AutoModelForRNNT`，要求 transformers 不低于 5.18，所以这个环境单独升级到 5.22.1。

| 包 | 版本 |
| --- | --- |
| transformers | 5.22.1 |
| huggingface-hub | 2.4.0 |
| librosa / numpy / soundfile | 1.0.0 / 2.5.4 / 0.14.0 |

```bash
uv venv --python <conda py3.14 的 python> --system-site-packages envs/asr-nemo-fa
uv pip install --python envs/asr-nemo-fa/bin/python transformers==5.22.1 librosa==1.0.0 numpy==2.5.4 soundfile==0.14.0 pyyaml==6.0.3
```

### 6.3 `envs/asr-gigaam`

使用者：`pipeline.asr.gigaam`。模型 `ai-sage/GigaAM-v3`，仓库的不同 revision 对应不同变体（`ctc`、`rnnt`、`e2e_rnnt`）。Python 3.12.13，独立环境。transformers 必须用 4.57.1，5.x 加载 GigaAM 会报 meta 设备错误。

| 包 | 版本 |
| --- | --- |
| torch / torchaudio | 2.15.0+cu132 / 2.11.1 |
| transformers | 4.57.1 |
| huggingface-hub | 0.36.2 |
| hydra-core / sentencepiece | 1.3.8 / 0.2.2 |
| soundfile / soxr / numpy | 0.14.0 / 1.1.0 / 2.5.4 |

```bash
uv venv --python 3.12 envs/asr-gigaam
uv pip install --python envs/asr-gigaam/bin/python --index-url https://download.pytorch.org/whl/cu132 --extra-index-url https://pypi.org/simple torch==2.15.0 torchaudio==2.11.1
uv pip install --python envs/asr-gigaam/bin/python transformers==4.57.1 huggingface-hub==0.36.2 hydra-core==1.3.8 sentencepiece==0.2.2 soundfile==0.14.0 soxr==1.1.0 numpy==2.5.4 pyyaml==6.0.3
```

### 6.4 `envs/asr-nemo`

使用者：`pipeline.asr.haas_ja`。模型 `third-intelligence/Haas-ja-v1.0`。Python 3.12.13，独立环境。torch 和 triton 的版本与 Haas-ASR 官方 `uv.lock` 一致，triton 3.9 下 flash-linear-attention 会报 `BT` 参数错误，所以固定 triton 3.8.0。`haas_asr` 包不安装，运行时把 `src/Haas-ASR/src` 加进 `PYTHONPATH`（`config_v1.yaml` 已设置）。

| 包 | 版本 |
| --- | --- |
| torch | 2.14.0+cu130 |
| triton | 3.8.0 |
| nemo-toolkit | 2.7.3 |
| flash-linear-attention | 0.5.2 |
| transformers | 4.57.6 |
| huggingface-hub | 0.36.2 |
| accelerate / peft | 1.15.0 / 0.21.2 |
| lightning / torchmetrics | 2.4.0 / 1.9.1 |
| datasets / jiwer | 5.0.1 / 3.1.0 |
| hydra-core / numpy / librosa | 1.3.2 / 2.5.4 / 1.0.0 |

```bash
uv venv --python 3.12 envs/asr-nemo
uv pip install --python envs/asr-nemo/bin/python --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple torch==2.14.0 triton==3.8.0
uv pip install --python envs/asr-nemo/bin/python "nemo-toolkit[asr]==2.7.3" flash-linear-attention==0.5.2 transformers==4.57.6
```

这是按已安装的版本整理的命令，完整依赖以 `src/Haas-ASR/uv.lock` 为准。

### 6.5 `envs/asr-hf`

使用者：`pipeline.asr.typhoon`（模型 `typhoon-ai/typhoon-asr-qwen-1.7b-ctx`）和 `pipeline.asr.jais_asr`（模型 `IFM/Jais-ASR`，需要 `trust_remote_code`）。Python 3.14.7，叠加在 conda 环境上；transformers 5.16.1 来自 conda，在上面换装了更新的 torch。

| 包 | 版本 |
| --- | --- |
| torch / torchaudio / torchvision | 2.15.0+cu132 / 2.11.1 / 0.29.2 |
| torchcodec | 0.17.0 |
| huggingface-hub | 1.33.0（小于 2） |
| accelerate | 1.15.0 |
| soundfile / soxr / sentencepiece | 0.14.0 / 1.1.0 / 0.2.2 |

```bash
uv venv --python <conda py3.14 的 python> --system-site-packages envs/asr-hf
uv pip install --python envs/asr-hf/bin/python --index-url https://download.pytorch.org/whl/cu132 --extra-index-url https://pypi.org/simple torch==2.15.0 torchaudio==2.11.1 torchvision==0.29.2
uv pip install --python envs/asr-hf/bin/python "huggingface-hub<2" accelerate==1.15.0 soundfile==0.14.0 soxr==1.1.0 sentencepiece==0.2.2
```

## 7. 训练与评测环境

### 7.1 `envs/mllm-train`

使用者：阶段 3 的 projector 训练（`pipeline/train/run.sh`、`pipeline.train.*`）。Python 3.14.7，叠加在 conda 环境上，PyTorch（2.13.0+cu130）、transformers 5.16.1 和 torchcodec 都来自 conda。额外装 flash-linear-attention 与 fla-core 0.6.0，并加 `--no-deps`，避免改动 torch 和 transformers；还装有 opencc 1.4.2，以及生成训练标签用的 [mm-assistant-mask](https://github.com/LUOXIAO92/MultimodalAssistantMask)（固定在提交 95e15d7，同样加 `--no-deps`）。没有 fla 时语言模型的线性注意力层退回 PyTorch 实现，结果相同但更慢。

```bash
uv venv --python <conda py3.14 的 python> --system-site-packages envs/mllm-train
uv pip install --python envs/mllm-train/bin/python --no-deps flash-linear-attention==0.6.0 fla-core==0.6.0
uv pip install --python envs/mllm-train/bin/python opencc==1.4.2
uv pip install --python envs/mllm-train/bin/python --no-deps \
    "mm-assistant-mask @ git+https://github.com/LUOXIAO92/MultimodalAssistantMask@95e15d78b5e418e11164fa94a694ee850db4f9f7"
```

### 7.2 `envs/mllm-eval`

使用者：真人语音评测（`pipeline/eval/run.sh`）。Python 3.14.7，叠加在 conda 环境上，额外安装 numpy 2.5.4、soundfile 0.14.0 和 opencc 1.4.2。opencc 让打分时的中文繁简转换与比对、朗读稿生成一致；没有安装时打分按单字表转换，并给出警告。

```bash
uv venv --python <conda py3.14 的 python> --system-site-packages envs/mllm-eval
uv pip install --python envs/mllm-eval/bin/python numpy==2.5.4 soundfile==0.14.0 opencc==1.4.2
```

## 8. 模型权重

除 IndexTTS-2.5 外，所有权重在第一次运行对应封装时，由 `huggingface_hub` 下载到默认的 Hugging Face 缓存，不需要手工下载。表中列出每个封装使用的仓库。

| 封装 | Hugging Face 仓库 | 获取方式 |
| --- | --- | --- |
| `tts.omnivoice` | `k2-fsa/OmniVoice` | 首次运行自动下载 |
| `tts.voxcpm2` | `openbmb/VoxCPM2` | 首次运行自动下载 |
| `tts.qwen3_tts` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base`、`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 首次运行自动下载 |
| `tts.indextts25` | `IndexTeam/IndexTTS-2.5` | 放在 `src/index-tts/checkpoints/`，用 `--model-dir` 指定（见 5.4） |
| `tts.vieneu` | `pnnbao-ump/VieNeu-TTS-v3-Turbo`（提交 `61b85e3d…`）、`OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano`（提交 `6aa02b01…`） | 首次运行自动下载 |
| `tts.supertonic3` | `Supertone/supertonic-3` | 由 supertonic 包下载，位置见 `SUPERTONIC_CACHE_DIR` |
| `asr.qwen3_asr` | `Qwen/Qwen3-ASR-1.7B-hf` | 首次运行自动下载 |
| `asr.whisper` | `openai/whisper-large-v3` | 首次运行自动下载 |
| `asr.whisper_fa` | `MohammadGholizadeh/whisper-large-v3-turbo-persian` | 首次运行自动下载 |
| `asr.nemotron_farsi` | `mehdi-hf/nemotron-asr-streaming-farsi` | 首次运行自动下载 |
| `asr.gigaam` | `ai-sage/GigaAM-v3`，revision 为 `ctc`、`rnnt`、`e2e_rnnt` | 预先下载（见下） |
| `asr.haas_ja` | `third-intelligence/Haas-ja-v1.0` | 预先下载（见下） |
| `asr.typhoon` | `typhoon-ai/typhoon-asr-qwen-1.7b-ctx` | 预先下载（见下） |
| `asr.jais_asr` | `IFM/Jais-ASR` | 预先下载（见下） |

GigaAM、Haas、Typhoon 和 Jais-ASR 在任务 #8b 里先用一小段脚本整体下载到 Hugging Face 缓存，之后封装直接读缓存。脚本只有一个 `snapshot_download` 调用：

```python
import sys
from huggingface_hub import snapshot_download

repo = sys.argv[1]
revision = sys.argv[2] if len(sys.argv) > 2 else None
print(snapshot_download(repo_id = repo, revision = revision))
```

存为 `dl.py`，用 conda 环境 `py3.14` 的 python 运行，命令如下：

```bash
python dl.py typhoon-ai/typhoon-asr-qwen-1.7b-ctx
python dl.py third-intelligence/Haas-ja-v1.0
python dl.py ai-sage/GigaAM-v3 ctc
python dl.py ai-sage/GigaAM-v3 rnnt
python dl.py ai-sage/GigaAM-v3 e2e_rnnt
python dl.py IFM/Jais-ASR
```

下载体积参考：Jais-ASR 约 11 GB，Typhoon 约 3.9 GB，GigaAM 三个 revision 共约 1.3 GB，Haas 约 1.2 GB。

## 9. 验证

环境建好后，在 `<工作目录>` 下检查每个环境的 Python 和关键包版本是否与本文一致：

```bash
for e in envs/*/; do echo "$e $($e/bin/python --version)"; done
uv pip freeze --python envs/tts-omnivoice/bin/python | grep -E "^(torch|transformers|omnivoice)"
```

之后按 `pipeline/CONVENTIONS.md` 和 `pipeline/runner/__init__.py` 里的用法，用 `envs/runner` 的 python 跑一遍小规模的合成与识别，确认各封装都能启动。

## 10. 本文没有覆盖的信息

- `envs/tts-indextts25`、`envs/tn`、`envs/asr-nemo`、`envs/asr-hf` 等环境当初的原始安装命令没有记录，对应章节里的命令是按已安装的版本整理的，其中 `tts-indextts25` 只能确定使用了上游 `uv.lock`。
- IndexTTS-2.5 权重与 `hf_cache/` 辅助模型的下载命令没有记录。
- `src/supertonic3-model/` 的获取方式没有记录。
- 本文只记录关键包的版本，传递依赖的完整版本以实际环境的 `uv pip freeze` 为准。
