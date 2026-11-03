# TTS 与 ASR 选型

调查日期：2026-11-04，替换 2026-09-26 的版本。目标语言共 18 种：中文、英语、日语、韩语、德语、法语、俄语、葡萄牙语、西班牙语、意大利语、印尼语、马来语、泰语、越南语、菲律宾语、阿拉伯语、土耳其语、波斯语。

用途是生成 projector 的训练数据：TTS 把朗读稿合成为音频，ASR 再把音频转回文字，与朗读稿核对后保留一致的样本。合成和核对都在本地一张 RTX 5090（32 GB）上大批量运行。项目纯开源，不涉及商业使用，合成数据和权重都不发布，限非商用的模型也可以使用，但需要标明。TTS 的许可必须允许用其输出训练本项目这样的非商用模型。

这次调查从最新发布出发，没有沿用旧的候选名单。具体做了四件事：一是用 HF API 按创建时间、热度、点赞和下载量列出 TTS、ASR 模型，并按团队和关键词补查；二是查看排行榜和评测榜单；三是检索 2026 年 5 月以后的 arXiv 论文；四是查看各主要团队的发布页面。每个模型都核对了发布日期、许可和一手出处，拿不准的标注「未核实」。旧版文档中的说法逐条核对，更正列在文末。

## 结论

TTS 首轮以 OmniVoice、VoxCPM2、Qwen3-TTS 为主力。三者都有统一口径的第三方成绩，都支持批量推理，32 GB 显存够用，并且能通过声音克隆或文字描述提供大量说话人。三者覆盖的语言不同：OmniVoice 覆盖全部 18 种语言，权重为 CC-BY-NC；VoxCPM2 缺波斯语；Qwen3-TTS 只覆盖最初的十种语言。阿拉伯语另加 IndexTTS-2.5，它在统一口径的榜上阿拉伯语排名第一，许可允许用输出改进非商用模型。FireRedTTS3、Confucius4-TTS、MOSS-TTS-v1.5 只有厂商自报的数字，作为补充候选实测。Fish Audio Drama 3、Fish S2 Pro、Higgs TTS 3 的许可不论是否商用，都禁止用输出训练本项目这类模型，予以排除。

ASR 没有一个模型能在所有语言上领先，需要按语言选择，每种语言选一个首选模型、一个交叉核对模型。新发布的 NVIDIA Nemotron-3.5-Transcribe 是欧洲语言和英语上最强的开放权重模型，可以按读法输出数字，但不支持马来语和波斯语，在 32 GB 上只能用 NVFP4 量化版。Qwen3-ASR 覆盖全部 18 种语言，在中文、韩语、印尼语、越南语上领先，在菲律宾语、波斯语、阿拉伯语、马来语、土耳其语上明显偏弱。这几种语言改用 Whisper-large-v3 或各语言的专用模型。

| 语言 | TTS 首轮候选 | ASR 首选 | ASR 交叉核对 |
| --- | --- | --- | --- |
| zh | Qwen3-TTS、OmniVoice、FireRedTTS3 | Qwen3-ASR-1.7B | FireRedASR2-LLM 或 MiMo-V2.5-ASR |
| en | OmniVoice、Qwen3-TTS、MOSS-TTS-v1.5 | Nemotron-3.5-Transcribe | Qwen3-ASR-1.7B 或 parakeet-tdt-0.6b-v3 |
| ja | OmniVoice、Qwen3-TTS、FireRedTTS3 | Haas-ja-v1.0 | Nemotron 或 Qwen3-ASR |
| ko | Qwen3-TTS、OmniVoice、Confucius4-TTS | Qwen3-ASR-1.7B | Nemotron |
| de | Qwen3-TTS、OmniVoice、VoxCPM2 或 Confucius4-TTS | Nemotron | Cohere Transcribe（gated）或 Whisper-large-v3 |
| fr | OmniVoice、Qwen3-TTS | Nemotron | Cohere Transcribe 或 Whisper-large-v3 |
| ru | OmniVoice、VoxCPM2、FireRedTTS3 | GigaAM-v3 | Nemotron |
| pt | VoxCPM2、Qwen3-TTS、OmniVoice | Nemotron | Cohere Transcribe 或 Whisper-large-v3 |
| es | Qwen3-TTS、OmniVoice、VoxCPM2 | Nemotron | Cohere Transcribe 或 Whisper-large-v3 |
| it | OmniVoice、Qwen3-TTS、VoxCPM2 | Nemotron | Cohere Transcribe 或 Whisper-large-v3 |
| id | VoxCPM2、OmniVoice、Confucius4-TTS | Qwen3-ASR-1.7B | Whisper-large-v3 |
| ms | VoxCPM2、OmniVoice、MOSS-TTS-v1.5 | Whisper-large-v3 | MERaLiON-3-3B-ASR 或 Qwen3-ASR |
| th | OmniVoice、VoxCPM2、FireRedTTS3 或 Confucius4-TTS | Typhoon ASR | Qwen3-ASR-1.7B |
| vi | OmniVoice、VoxCPM2、VieNeu-TTS-v3 | Qwen3-ASR-1.7B | Cohere Transcribe 或 Nemotron |
| fil | VoxCPM2、OmniVoice、MOSS-TTS-v1.5 | Whisper-large-v3 | NCSpeech FastConformer 或 Nemotron |
| ar | IndexTTS-2.5、OmniVoice、MOSS-TTS-v1.5 或 FireRedTTS3 | Jais-ASR 或 cohere-transcribe-arabic（gated） | 两者交叉 |
| tr | VoxCPM2、FireRedTTS3、OmniVoice | Whisper-large-v3 | Nemotron |
| fa | OmniVoice、MOSS-TTS-v1.5 | nemotron-asr-streaming-farsi | whisper-large-v3-turbo-persian |

## TTS

### 候选

| 模型 | 发布 / 最近更新 | 许可 | 覆盖（缺） | 推理与显存 |
| --- | --- | --- | --- | --- |
| [OmniVoice](https://github.com/k2-fsa/OmniVoice) | HF 2026-03-30，权重 04-13；0.2.1 于 07-16 | 代码 Apache-2.0，**权重 CC-BY-NC**（训练数据含 Emilia） | 全部 18 种 | 0.8B，约 3.3 GB；自带批量推理和多卡脚本，支持 FlashInfer、vLLM-Omni |
| [VoxCPM2](https://github.com/OpenBMB/VoxCPM) | HF 2026-04-03，权重 04-16；v2.0.3 于 05-11 | Apache-2.0 | 17 种（缺 fa） | 2B，约 8 GB；Nano-vLLM、vLLM-Omni |
| [Qwen3-TTS-12Hz-1.7B](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)（Base / VoiceDesign / CustomVoice） | 2026-01-21，之后权重未变 | Apache-2.0 | 最初的十种 | 1.7B，约 4.5 GB；原生批量，vLLM-Omni |
| [FireRedTTS3](https://github.com/FireRedTeam/FireRedTTS3)（Base / Instruct） | 2026-08-13 | Apache-2.0 | 15 种（缺 ms、fil、fa） | 骨干由 Qwen3-1.7B 初始化，另有 RedAE 等模块；没有官方批量推理或 vLLM 说明 |
| [MOSS-TTS-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5) | 2026-05-25 | Apache-2.0 | 17 种（缺 id） | 8.49B，bf16 约 17 GB，放得下但余量小；transformers 批量、SGLang、vLLM-Omni |
| [Confucius4-TTS](https://huggingface.co/netease-youdao/Confucius4-TTS) | 2026-06-02 / 10-23 | Apache-2.0 | 14 种（缺 fil、ar、tr、fa） | 约 3 GB 加若干组件；需打补丁接入 vLLM |
| [VieNeu-TTS-v3-Turbo](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo) | 2026-06-05 | Apache-2.0 | 越南语专用 | 23 个内置音色加克隆；规格未逐项核实 |
| [IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5) | 2026-08-10 | bilibili 模型许可：输出只能用于改进 IndexTTS 自身或非商用模型 | 5 种：zh、en、ja、es、ar，只在阿拉伯语上列为候选 | 共 1.67B，约 6 GB；官方代码逐条推理，需 Python 3.10–3.11，榜上 RTFx 2.51（OmniVoice 为 32.98）；批量可用 vLLM-Omni，单卡配置未经官方验证 |

还查过以下模型，未列为首轮候选：Audio8-TTS-Realistic-Air 的 codec 与 Fish S2 一致，可能受 Fish 许可约束，并在 TTS Arena 因刷票被封；ZONOS2 是 8B MoE，缺 ms、tr、fa；dots.tts 在日、韩、阿拉伯语上偏弱；Chatterbox Multilingual V3 缺 id、th、vi、fil、fa，输出带水印；Supertonic-3 和 NVIDIA Magpie 只有少量固定音色，不能克隆；Ming-3.0-tts 只覆盖 7 种语言；Fun-CosyVoice3 不覆盖葡萄牙语和新增的八种语言。

以下模型的许可不论是否商用，都禁止用输出训练本项目这类模型，因此排除：[Fish Audio Drama 3](https://huggingface.co/fishaudio/drama-3/blob/main/LICENSE)、[Fish S2 Pro](https://huggingface.co/fishaudio/s2-pro/blob/main/LICENSE.md)、[Higgs TTS 3](https://huggingface.co/bosonai/higgs-tts-3-4b/blob/main/LICENSE)。Drama 3 是 Artificial Analysis 英语榜上排名最高的开放权重模型，35B-A3B，FP8 版也放不进 32 GB。

### 统一口径的成绩

HF 官方的 [Open TTS Leaderboard](https://huggingface.co/spaces/hf-audio/open_tts_leaderboard) 于 2026-09 上线，最新数据为 2026-11-03。它在 Seed-TTS-eval、CV3-Eval 和 MiniMax 多语种测试集上用同一个 Qwen3-ASR-1.7B 给所有模型打分，说话人相似度（SIM）用 WavLM 计算。这种打分方式与我们用 ASR 核对 TTS 的流程属于同一类。下表为克隆模式的 WER%（中、日、韩、泰为 CER%），括号内为 SIM×100。

| 语言 | Qwen3-TTS-1.7B | VoxCPM2 | OmniVoice | MOSS-TTS v1.0 | IndexTTS-2.5 |
| --- | --- | --- | --- | --- | --- |
| zh | **0.75**（78） | 1.06（80） | 0.81（78） | 0.89（76） | 1.02（75） |
| en | 0.91（73） | 1.54（75） | **0.82**（77） | 1.16（70） | 1.31（60） |
| ja | 4.67（75） | 5.21（78） | **4.56**（77） | 8.52（74） | 5.49（74） |
| ko | **2.82**（78） | 3.54（79） | 3.01（79） | 4.05（77） | – |
| de | **1.47**（75） | 2.38（78） | 1.79（78） | 1.94（73） | – |
| fr | 3.36（71） | 3.88（72） | **2.79**（76） | 3.83（67） | – |
| ru | 4.71（75） | 4.25（78） | **3.46**（75） | 5.38（74） | – |
| pt | 1.33（81） | **0.90**（83） | 1.57（86） | 2.52（80） | – |
| es | **1.24**（78） | 1.94（79） | 1.45（80） | 2.42（76） | 2.58（73） |
| it | 1.98（75） | 2.37（77） | **1.86**（78） | 2.53（74） | – |
| id | – | **0.75**（81） | 0.83（78） | – | – |
| th | – | 1.71（82） | **1.34**（83） | – | – |
| vi | – | 1.34（80） | **0.54**（78） | – | – |
| ar | – | 10.11（77） | 5.39（75） | 7.42（73） | **4.12**（76） |
| tr | – | **2.07**（86） | 3.53（85） | 4.57（81） | – |

FireRedTTS3、Confucius4-TTS、MOSS-TTS-v1.5 不在这张榜上，只有各自在 MiniMax 测试集上的自报数字。各家的参考音频、打分 ASR 和重生成策略不同，自报数字之间不能直接比较。自报与第三方结果有时差距很大：例如 FireRedTTS3 自报阿拉伯语 1.75、越南语 0.86，小规模第三方复测为 20.67、19.82。这几个模型需要实测确认。

马来语、菲律宾语、波斯语没有公共基准，只有厂商内部测试。马来语：VoxCPM2 内部测试 WER 1.75，OmniVoice 在 FLEURS 上 CER 1.19。菲律宾语：VoxCPM2 为 2.63，OmniVoice 为 1.24。波斯语：只有 OmniVoice 在 FLEURS 上的 CER 1.90。这三种语言必须自己实测。

### 音色与文本规范化

训练数据需要多说话人。Qwen3-TTS 的 Base 支持克隆，VoiceDesign 支持用文字描述生成音色，CustomVoice 有 9 个内置音色；VoxCPM2 支持克隆和文字设计音色；OmniVoice 支持克隆，不给参考音频时生成随机音色，但它的文字设计音色只用中英数据训练过；FireRedTTS3 的 Base 支持克隆，Instruct 支持文字设计；MOSS-TTS 支持克隆和随机音色；IndexTTS-2.5 只支持克隆，另外可以控制情感和语速。克隆用的参考音频要与目标语言一致，OmniVoice 和 FireRedTTS3 都说明跨语言克隆会带口音。

自带固定音色的模型有：Qwen3-TTS CustomVoice 9 个，母语为中、英、日、韩；Supertonic-3 10 个（5 女 5 男），覆盖 13 种目标语言，缺 zh、th、ms、fil、fa，许可为 OpenRAIL-M；VieNeu-TTS-v3 23 个，只说越南语。三者合计约 40 个音色。第一版不收集外部参考音频，用这些内置音色合成参考片段，再交给支持克隆的模型合成其他语言，每条样本的音色和语言随机组合。跨语言克隆带来的口音和音色数量是否够用，由第一版训练后的指标判断；如果不理想，再加入外部参考音频。每条样本记录 TTS 模型、音色和参考片段的语言，便于按音色和是否跨语言拆分指标。

朗读稿已经由我们自己完成文本规范化，合成时要确认模型不再改动文本：

- VoxCPM2、OmniVoice、dots.tts 默认不做规范化。
- FireRedTTS3 示例默认开启（`do_tn=True`），需要关闭；Ming、ZONOS2 也需要显式关闭。
- IndexTTS-2.5 默认开启（`text_normalization=True`），但只处理中、英、日、西四种语言，合成阿拉伯语时不改动文本。
- Confucius4-TTS 对中、日以外的语言会按英文规则把数字展开成英文单词，必须传 `raw=True`。
- MOSS-TTS 会改写标点、换行和 Markdown，不展开数字，无法关闭。

Qwen3-TTS 在参考文本有错时质量下降，长文本下偶有失控，合成时应按句切分。

## ASR

### 候选

| 模型 | 发布 / 更新 | 许可 | 覆盖 | 输出形式 | 推理与显存 |
| --- | --- | --- | --- | --- | --- |
| [Nemotron-3.5-Transcribe-30B-A4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B) | HF 2026-10-07，正式发布 10-28 | OpenMDW-1.1 | 16 种（缺 ms、fa）。官方分级：ko、de、fr、ru、pt、es、it 为 Leading，ja、th、tr 为 Production-ready，zh、ar、id、vi、fil 只是 Supported | 用 prompt 控制标点和大小写，可以要求按读法写出数字；读法输出只在英语上有公开评测 | 总参数 34B，激活 4.3B；BF16 约 68 GB、FP8 约 34 GB，32 GB 上只能用 NVFP4 版（官方报告平均错误率相对上升 3.6%）；vLLM 上游支持尚未发布，目前需走 NeMo 插件 |
| [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | 2026-01-28；-hf 版 06-26 | Apache-2.0 | 全部 18 种（共 30 种） | 带标点；没有关闭反向规范化（ITN）的开关，数字形式未核实 | 约 2.3B；官方 vLLM 后端，批量吞吐高 |
| [Whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | 2023-11 | Apache-2.0 | 全部 18 种 | 带标点；数字写法不稳定，多为阿拉伯数字 | 1.55B；vLLM、CTranslate2 均支持批量 |
| [Cohere Transcribe 03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) | 2026-03-24 | Apache-2.0，**gated** | 11 种（缺 ru、id、ms、th、fil、tr、fa） | 默认带标点；必须指定语言 | 2B；vLLM |
| [GigaAM-v3](https://huggingface.co/ai-sage/GigaAM-v3) | 2025-11 | MIT | ru | ctc/rnnt 版无标点、按读法输出数字 | 约 240M |
| [Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0) | 2026-10-08 | Apache-2.0 | ja | 未说明 | 616M；NeMo |
| [Typhoon ASR](https://huggingface.co/typhoon-ai/typhoon-asr-qwen-1.7b-ctx) | 2026-09-03 | Apache-2.0 | th | 数字转成泰文读法，ๆ 展开 | 约 2B |
| [Jais-ASR](https://huggingface.co/IFM/Jais-ASR) | 2026-10-06 | Apache-2.0 | ar | 可选输出元音符号 | 5.7B，约 12 GB |
| [cohere-transcribe-arabic-07-2026](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026) | 2026-06-18 | Apache-2.0，gated | ar | 必须指定语言 | 2B |
| [nemotron-asr-streaming-farsi](https://huggingface.co/mehdi-hf/nemotron-asr-streaming-farsi) | 2026-10-03 | OpenMDW-1.1 | fa | 无标点，按读法输出数字 | 0.6B |
| [MERaLiON-3-3B-ASR](https://huggingface.co/MERaLiON/MERaLiON-3-3B-ASR) | 2026-05-24 | MERaLiON 公共许可（需署名） | en、zh、ms、id、th、vi | 带标点 | 3.3B |
| [FireRedASR2-LLM](https://huggingface.co/FireRedTeam/FireRedASR2-LLM) | 2026-02-25 | Apache-2.0 | zh、en | 无标点，标点由单独模块添加 | 8B 以上；vLLM |
| [Fun-ASR-MLT-Nano-2512](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512) | 2025-12 | 卡片写 Apache-2.0，另有说法为 FunASR 许可，未核实 | 10 种 | `itn=False` 可关闭数字转写 | 0.8B |

Qwen 在 2026 年 5 月以后只发布了 API 模型（Qwen-Audio-3.0/3.1-ASR），开放权重仍是 Qwen3-ASR。Meta Omnilingual ASR、IBM Granite Speech 5.0、Voxtral、Parakeet 等也查过：Omnilingual 覆盖全部语言，但标点和数字输出形式未核实；Granite 5.0 和 Parakeet 只覆盖 6–7 种欧洲语言；Voxtral-Small 放不进 32 GB。

### 成绩

HF [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) 的多语种榜只有德、法、意、西、葡等少数语言，数据截至 2026-10-30，各模型在同一口径下可比。下表为 WER%。

| 测试集 | Nemotron | Cohere | Qwen3-ASR-1.7B | Whisper-v3 | Voxtral-Small-24B | omniASR-7B-v2 |
| --- | --- | --- | --- | --- | --- | --- |
| de FLEURS | 2.62 | 3.33 | 3.35 | 3.20 | 2.61 | 3.72 |
| fr FLEURS | 3.46 | 4.33 | 4.06 | 4.84 | 3.61 | 5.19 |
| it FLEURS | 1.49 | 2.33 | 2.23 | 2.28 | 2.38 | 2.29 |
| es FLEURS | 2.35 | 3.26 | 2.92 | 2.30 | 2.79 | 2.93 |
| pt FLEURS | 3.11 | 4.24 | 3.97 | 3.50 | 3.64 | 4.15 |
| 五语平均（含 MCV、MLS） | **3.47** | 3.78 | 约 5.04 | 4.60 | 3.61 | 4.36 |

英语榜从 2026-11-02 起换了默认平均口径，开放权重中 Nemotron 6.10、Qwen3-ASR-1.7B 6.14、Cohere 6.72。其他语言没有统一的榜单，下表汇总了能找到的数字，同一列内可比，跨列不可比：「Qwen 报告」与「Nemotron 卡」为各自的 FLEURS 自测；「Handy」为同一流程下的 FLEURS 评测（Q8_0 量化）；最后一列为区域榜单或专用模型的数字。

| 语言 | Qwen 报告：Qwen3 | Nemotron 卡 | Handy 同表 | 区域榜或专用模型 |
| --- | --- | --- | --- | --- |
| zh | 2.41 | 8.62 | Qwen3 7.14、Whisper 7.98 | 四个普通话集：FireRedASR2-LLM 2.89、Qwen3 3.76（FireRed 自报） |
| ja | 5.20 | 4.42 | Whisper 4.81、Qwen3 5.29 | asr-ja-bench：Haas 6.12、Nemotron 6.89、Qwen3 7.59（作者自评） |
| ko | 2.57 | 2.78 | Qwen3 4.60、Whisper 4.89 | — |
| ru | 5.99 | 3.80 | Whisper 4.96、Qwen3 6.25 | 俄语榜 11 集平均：GigaAM-v3 7.19、Whisper 15.44、Qwen3 16.55 |
| id | 5.16 | 14.51 | Qwen3 5.37、Whisper 6.08 | MERaLiON 自报与其他来源冲突 |
| ms | 10.39 | 不支持 | Whisper 7.59、Qwen3 10.42 | 另一独立对照：Whisper 7.08、Qwen3 9.49 |
| th | 6.32 | 7.34 | Qwen3 6.89、Whisper 8.78 | Typhoon 自报比 Qwen3 低 30–40% |
| vi | 5.55 | 2.90 | Qwen3 6.15、Whisper 8.74 | — |
| fil | 24.29 | 18.28 | Whisper 11.82、Qwen3 24.29 | NCSpeech FastConformer 自报 9.34 |
| ar | 16.98 | 6.35 | cohere-arabic 11.06、Qwen3 14.91、Whisper 14.92 | 阿拉伯语榜：Jais 和 cohere-arabic 明显领先 |
| tr | 9.47 | 6.75 | Whisper 6.51、Qwen3 9.46 | 土耳其语榜：Whisper 6.06、Qwen3 8.19 |
| fa | 29.90 | 不支持 | 各模型都在 28–30 | nemotron-farsi 第三方复测 11.51 |

### 核对用法

所有公开数字都来自真人录音。在 TTS 合成语音上，整句完全一致率、误收率和误杀率都要用我们自己的数据测量。以 LLM 为解码器的 ASR（如 Qwen3、Nemotron）语言先验强，可能把 TTS 漏读或错读的地方「补」回原文，造成误收；CTC 或 Transducer 结构的模型（如 GigaAM、Parakeet、FastConformer）更贴近声学。这是推断，没有公开测量，需要实测确认。交叉核对时，第二个模型尽量选结构或训练来源不同的。「两者都一致才收」误收最少，「任一一致就收」误杀最少，先用一小批人工听审过的样本量出两种规则的误收率和误杀率，再决定采用哪一种。

朗读稿中的数字已写成读法，而 Qwen3、Whisper、Cohere 多数输出阿拉伯数字。比对方法见 `research/text-normalization-zh.md` 的「比对环节」：两边的数词统一映射成数值记号后再比较。能按读法输出数字的模型（Nemotron 的读法 prompt、GigaAM、Typhoon、nemotron-farsi、Fun-ASR 的 `itn=False`）可以减少这一步的误差。各模型的数字写法和中文简繁形式，需要先抽样确认再写比对规则。

## API 服务

API 只作对照，本项目不使用。原因有两个：一是用量大，二是多数服务的条款限制把输出用于训练模型。具体条款如下：

- ElevenLabs 的[使用政策](https://elevenlabs.io/use-policy)明确禁止把输出放进用于训练、微调或改进 AI 模型的数据集。
- OpenAI、Google、AssemblyAI、Speechmatics 限制用输出开发竞争或同类模型。
- Deepgram 连 benchmark 用途也加以限制，Soniox 则禁止把输出用作训练、评测或验证数据。
- 阿里云的产品条款禁止用输出训练与阿里竞争的产品。
- MiniMax 的条款中没有找到禁止条款。

## 需要实测的缺口

- **TTS**：马来语、菲律宾语、波斯语没有公共基准；FireRedTTS3、Confucius4-TTS、MOSS-TTS-v1.5 不在统一口径的榜上；阿拉伯语、越南语、日语的自报和第三方结果冲突明显。
- **ASR**：马来语、菲律宾语、波斯语没有可信的同条件数字；印尼语、泰语的专用模型只有自报；Nemotron 的 NVFP4 版在 RTX 5090 上的逐语言精度和吞吐没有公开数字，读法输出 prompt 在非英语语言上是否有效也未核实。
- **TTS 与 ASR 的组合**：在合成语音上的完全一致率、误收率和误杀率，需要按语言实测。

## 与 2026-09-26 版相比的更正

- 旧版称四个 TTS 候选都采用 Apache-2.0。OmniVoice 的权重实为 CC-BY-NC，模型卡自 2026-07-03 起已写明。
- 旧版称 VoxCPM2 对比表中的 Qwen3-TTS 未写明 checkpoint。该列数字与 Qwen3-TTS-12Hz-1.7B-Base 的官方数字逐项相同，因此 checkpoint 可以确定；它仍然算不上复现实验。
- 旧版称没有同口径评测，现在已有 Open TTS Leaderboard。旧版推荐重点比较 VoxCPM2 的德语，这一点没有得到统一口径数据的支持：该榜德语成绩为 Qwen3-TTS 1.47、OmniVoice 1.79、VoxCPM2 2.38。
- Cohere Transcribe 需要登录并同意条件才能下载（gated），旧版没有写明。旧版把 ElevenLabs Scribe v2 列为可比较的候选，没有提到它的条款禁止把输出用于训练。
- 英语榜换口径后，Qwen3-ASR-1.7B（6.14）已经领先 Cohere（6.72）。
- NVIDIA Nemotron-3.5-Transcribe 于 2026-10-28 发布，成为欧洲语言和英语上的首选，旧版「NVIDIA 模型没有足够证据替代」的判断已不成立。
- 在马来语、菲律宾语、土耳其语上，Whisper-large-v3 优于 Qwen3-ASR，旧版「Whisper 暂不放首轮」需要修正。俄语榜已有 GigaAM-v3 与 Qwen3-ASR 的同条件对照。
- 旧版引用的 `hf-audio/multilingual_evals` 数据集已改为私有，原链接无法访问。
- Qwen 和阿里的服务端 TTS 已更新到 Qwen-Audio-3.1-TTS（2026-10）。
