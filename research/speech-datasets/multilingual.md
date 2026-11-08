# 多语种真人语音语料调研（18 种目标语言）

调研日期：2026-11-08。范围：覆盖多个目标语言的大型有转写真人语音语料。只做网页和 Hugging Face API 元数据查询，没有下载数据、没有登录、没有接受任何协议。

目标语言：zh en ja ko de fr ru pt es it id ms th vi fil ar tr fa。

## 0. 读表说明

- 「未核实」表示在一手来源（数据卡、官网、论文）里没有找到，没有猜测。
- 小时数（h）按各语料自己的口径写明。「转写形式」一栏里，「标点+大小写」指文本与普通书面文字接近；「小写无标点」指需要额外恢复才能当训练目标。
- 体积（GB，十进制）来源写在每个数据集章节里：`datasets-server size` 是 Hugging Face 的 `/size` 接口（原始文件字节数）；`tree` 是 `api/datasets/<id>/tree` 的文件大小求和；`卡片` 是数据卡或官网。
- 粗估公式：16 kHz 16 bit 单声道 WAV 约 0.115 GB/小时（压缩格式更小）。
- FLEURS 约束：FLEURS 句子来自 FLoRes。本文只有 FLEURS 自身和（可能）以 FLoRes/FLEURS 为来源的数据需要标记，见各章节「与 FLEURS 的关系」。

## 1. 汇总表

格式：小时数 / 体积。「—」表示该语料不含该语言；「未核实」表示没有一手数字。

### 1.1 大规模伪标签或自动字幕类（量大，转写质量参差）

| 语言 | YODAS2（手动字幕 / 自动字幕，小时*） | YODAS2 体积(GB)** | Emilia + Emilia-YODAS（h） | Granary（伪标签，有标点） | GigaSpeech 2（伪标签，训练 raw/refined） |
|---|---|---|---|---|---|
| zh | 299 / 0 | 60 | 50.3k（Emilia 49.9k + YODAS 0.3k） | — | — |
| en | 29,155 / 140,697 | 25,888 | 139.0k | 有（h 未核实） | — |
| ja | 1,094 / 2,930 | 712 | 2.8k | — | — |
| ko | 2,774 / 17,673 | 3,179 | 7.5k | — | — |
| de | 3,064 / 10,949 | 2,116 | 7.2k | 有（h 未核实） | — |
| fr | 2,424 / 18,450 | 3,229 | 8.8k | 有（h 未核实） | — |
| ru | 5,596 / 34,026 | 6,183 | — | 有（h 未核实） | — |
| pt | 1,631 / 18,678 | 3,068 | — | 有（h 未核实） | — |
| es | 3,738 / 33,423 | 6,866 | — | 有（h 未核实） | — |
| it | 1,455 / 9,233 | 1,617 | — | 有（h 未核实） | — |
| id | 1,420 / 8,464 | 2,105 | — | — | 8,113 / 5,714 h |
| ms | 28 / 0 | 10 | — | — | — |
| th | 497 / 2 | 162 | — | — | 12,902 / 10,262 h |
| vi | 780 / 9,203 | 2,121 | — | — | 7,324 / 6,039 h |
| fil | —（无分片） | — | — | — | — |
| ar | 290 / 0 | 227 | — | — | — |
| tr | 589 / 4,068 | 850 | — | — | — |
| fa | 43 / 0 | 15 | — | — | — |

\* YODAS2 的数据卡没有分语言小时表。这里的小时数取自 YODAS（v1）数据卡的分片统计表，两者分片名一一对应（YODAS2 的 `xx000` 是手动字幕分片，`xx1nn` 是自动字幕分片），但 YODAS2 是否与 v1 小时数完全相同「未核实」，请当作量级参考。
\*\* YODAS2 体积为 `tree` 求和（含音频和文本，24 kHz）。部分分片接口单页上限 1000 条，个别分片可能被截断，应视为下限。

Emilia / Emilia-YODAS 分语言小时与体积（卡片 + tree）：

| 语言 | Emilia h | Emilia GB | Emilia-YODAS h | Emilia-YODAS GB |
|---|---|---|---|---|
| zh | 49.9k | 1,264.5 | 0.3k | 8.5 |
| en | 46.8k | 1,156.1 | 92.2k | 1,489.0 |
| ja | 1.7k | 84.5 | 1.1k | 30.9 |
| ko | 0.2k | 5.0 | 7.3k | 217.6 |
| de | 1.6k | 37.6 | 5.6k | 167.5 |
| fr | 1.4k | 62.7 | 7.4k | 222.1 |

### 1.2 人工转写或人工字幕类（质量较高，量较小）

| 语言 | Common Voice 27（总 h / 已验证 h / GB） | MLS（h / GB） | VoxPopuli 转写子集（h / GB） | MediaSpeech（h / GB） | Multilingual TEDx（GB，h 未核实） | WorldSpeech（论文口径 h / HF 仓库 GB） | MSR-86K（训练 h） |
|---|---|---|---|---|---|---|---|
| zh | zh-CN 1,075 / 240 / 23.0；zh-TW 131.7 / 79.9 / 3.2；zh-HK 143.4 / 108.6 / 3.7 | — | — | — | — | 1,482（台湾立法院）/ 40 | — |
| en | 3,796 / 2,827 / 95.0 | 44,660 / 未核实 | 543 / 98.8 | — | — | 5,312 / 155 | 9,795 |
| ja | 726 / 373 / 15.4 | — | — | — | — | 1,387（LibriVox 朗读）/ 38 | 1,779 |
| ko | 10.4 / 2.6 / 0.22 | — | — | — | — | 1,454 / 38 | 10,339 |
| de | 1,492 / 1,395 / 37.4 | 1,967 / 31.5 | 282 / 52.6 | — | 2.6 GB | 1,907 / 53 | 6,862 |
| fr | 1,217 / 1,108 / 30.7 | 1,077 / 17.4 | 211 / 42.1 | 约 10 / 0.64 | 34 GB | 6,029（含魁北克）/ 29 | 8,317 |
| ru | 293 / 257 / 7.1 | — | — | — | 10 GB | 1,537 / 40 | 3,189 |
| pt | 230 / 189 / 5.2 | 161 / 2.8 | — | — | 29 GB | 1,764 / 45 | 3,738 |
| es | 2,283 / 598 / 51.9 | 918 / 15.0 | 166 / 31.7 | 约 10 / 0.59 | 35 GB | 6,792 / 186 | 13,977 |
| it | 431 / 364 / 10.5 | 247 / 4.2 | 91 / 17.1 | — | 19 GB | —（不含） | 5,691 |
| id | 67.5 / 34.3 / 1.5 | — | — | — | — | 340 / 14 | 1,983 |
| ms | 4.05 / 0.22 / 0.09 | — | — | — | — | 432 / 11.5 | — |
| th | 427 / 173 / 9.0 | — | — | — | — | 1,176 / 33 | 3,675 |
| vi | 23.0 / 7.8 / 0.48 | — | — | — | — | 726（仅元数据，无音频）/ — | 5,957 |
| fil | CV 27 中无该语言 | — | — | — | — | 219（Tagalog，tl_ph）/ 5.9 | — |
| ar | 158 / 93 / 3.5 | — | — | 约 10 / 0.62 | 3.6 GB | 1,001（多国方言）/ 28 | 874 |
| tr | 136 / 130 / 3.0 | — | — | 约 10 / 0.62 | — | 1,008 / 26 | — |
| fa | 431 / 373 / 11.2 | — | — | — | — | 28 / 0.8 | — |

FLEURS 训练集（需标记，不得用于训练）：每种语言约 10 小时，见第 12 节。

## 2. YODAS2（espnet/yodas2）

- 链接：https://huggingface.co/datasets/espnet/yodas2 ；论文 https://arxiv.org/abs/2406.00899 ；v1 数据卡（含分片统计表）https://huggingface.co/datasets/espnet/yodas
- 覆盖语言：除 fil 外全部（17 种）。分片列表来自 `api/datasets/espnet/yodas2/tree/main/data`，没有 `fil`/`tl` 分片。
- 小时数：见 1.1 汇总表。来源是 v1 数据卡的 Statistics 表（YODAS2 自己的数据卡没有分语言小时表；总量见下）。v1 总计 369,510 小时；YODAS2 论文摘要称 YODAS 整体「超过 50 万小时、100 多种语言」。
- 许可：数据卡元数据 `cc-by-3.0`；卡片原文「We made sure that our dataset only consisted of videos with CC licenses during our downloading」。论文页面指向 CC BY 4.0 的说法与数据卡不一致，以数据卡 `cc-by-3.0` 为准，需标记。
- 获取方式：直接下载，未设门槛（`gated: false`），支持 `streaming=True`。
- 转写：YouTube 字幕，非模型逐句转写。分片名第一位数字 0 表示用户上传的手动字幕，1 表示自动字幕；数据卡明确写道「manual caption only indicates that it is uploaded by users, but not necessarily transcribed by a human」。标点和大小写：数据卡未说明（未核实）。数字写法：未说明（未核实）。
- 域和说话人：YouTube 视频，多种场景。说话人数：未核实。
- 格式和体积：整段视频（非切片）加 `utterances` 时间戳；24 kHz；总计 60.3 TB（`usedStorage`）。分语言体积见汇总表（tree 求和，下限）。数据卡提示下载很慢。
- 与 FLEURS 的关系：无已知关系。
- 质量问题：手动字幕也可能是机器翻译或错位；时间戳需要自行切片和对齐；数据卡没有质量指标。
- 使用建议：手动字幕分片（`xx000`）更可靠；ja、ko、de、fr 等自动字幕分片量大但是 YouTube 自动识别结果，标点和大小写基本缺失，使用前需检查。

## 3. Emilia / Emilia-YODAS（amphion/Emilia-Dataset）

- 链接：https://huggingface.co/datasets/amphion/Emilia-Dataset
- 覆盖：zh en ja ko de fr（6 种）。小时数见 1.1 的 Emilia 表（来源：数据卡）。总计 Emilia 101.7k h，Emilia-YODAS 113.9k h，合计 215.6k h。
- 许可：数据卡元数据 `cc-by-4.0`，但正文写「Users are permitted to use Emilia dataset only for non-commercial purposes under the CC BY-NC-4.0 license」，使用条款写 Emilia 部分按 CC-BY-NC、Emilia-YODAS 部分按 CC-BY。**非商业，需标记，且卡片内部标注不一致。**
- 获取：HF gated（`gated: auto`），需要登录并同意使用条款（含赔偿和终止条款）。中国大陆用户可经 OpenDataLab 获得 Emilia（不含 Emilia-YODAS）。我没有接受任何条款。
- 转写：Emilia 管线用 ASR 自动打标（伪标签）。数据卡未写明 ASR 模型（未核实）。卡片里的 JSONL 示例显示有大小写和标点，开头带空格。数字写法：未核实。
- 域：互联网视频和播客，脱口秀、访谈、辩论、体育解说、有声书等，自发语音为主。说话人数：未核实。
- 格式和体积：tar 分片，Emilia 2.4 TB（2,360 个 tar），Emilia-YODAS 2.1 TB（1,983 个 tar）；仓库 `usedStorage` 7.17 TB。分语言体积见汇总（tree 求和）。
- 与 FLEURS 的关系：无已知关系。
- 质量问题：伪标签，语音合成导向的数据管线，过滤偏向 TTS 质量而不是转写准确度。Emilia-YODAS 来自 YODAS 的 CC-BY 子集，与第 2 节和第 4 节存在音频重叠，不要重复计算。

## 4. NVIDIA Granary（nvidia/Granary）与 YODAS-Granary

- 链接：https://huggingface.co/datasets/nvidia/Granary ；https://huggingface.co/datasets/espnet/yodas-granary ；论文 https://arxiv.org/abs/2505.13404
- 覆盖 25 种欧洲语言，我们的目标语言里有 en de es fr it pt ru（7 种）。
- 小时数：总计约 100 万小时伪标签 ASR 数据（卡片）。YODAS-Granary 共 192k 小时、23 种欧洲语言；MOSEL 部分 451k 小时（卡片）。分语言小时在卡片的图中，没有文字表，**各目标语言小时未核实**。YODAS-Granary 的话语条数（datasets-server）：en 40.8M，es 8.9M，ru 9.9M，fr 5.3M，it 1.3M。
- 许可：`nvidia/Granary` 为 `cc-by-4.0`（清单文件）；`espnet/yodas-granary` 为 `cc-by-3.0`。音频本身沿用来源语料的许可（YODAS2 CC-BY、VoxPopuli CC0、YouTube-Commons 等），需分别核对。
- 获取：直接下载，未设门槛。`nvidia/Granary` 只含 NeMo 清单（jsonl，约 203 GB），音频要另外下载来源语料；`yodas-granary` 内嵌 16 kHz 音频。
- 转写：伪标签。流程为 VAD+强制对齐切片，Whisper-large-v3（faster-whisper）两遍识别并做语种检查，过滤幻觉和低质量，再用 Qwen2.5-7B 恢复标点和大小写（卡片写作 P&C normalization）。因此标点+大小写：有，但是模型恢复结果。数字写法：未核实。
- 域：YouTube（YODAS、YouTube-Commons）、欧洲议会（VoxPopuli）、有声书（LibriLight，仅英语）。说话人数：未核实。
- 体积：`yodas-granary` 总 18,958 GB（datasets-server），en 11,264 GB，es 3,022 GB，ru 2,296 GB，fr 1,393 GB，it 697 GB；de、pt 未能取到分语言体积。
- 与 FLEURS 的关系：无已知关系。
- 质量问题：Whisper 幻觉和语言误判经过过滤但仍有残留；Qwen 恢复的标点可能改写内容。对我们的价值：在欧洲语言上提供了「带标点和大小写的伪标签」，格式与目标文本最接近。

## 5. Common Voice（当前版本 27.0，2026-09-11）

- 官方入口：Mozilla Data Collective（https://mozilladatacollective.com/datasets ，原域名 datacollective.mozillafoundation.org 已 301 跳转）。统计来源：官方元数据库 https://github.com/common-voice/cv-dataset ，文件 `datasets/scripted-speech/cv-corpus-27.0-2026-09-11.json`。
- 发布节奏：约每季度一版（v24 2025-12，v25 2026-03，v26 2026-06，v27 2026-09）。另有 Spontaneous Speech（sps-corpus 5.0，2026-09-11）和 code-switching 数据集，本次未统计语言覆盖。
- 小时数：见 1.2 表。总时长含全部提交录音（包括未验证、被否决的）；训练建议只用已验证（validated）部分。**注意：zh-CN 总 1,075 h 但已验证仅 240 h；es 总 2,283 h 已验证仅 598 h。** CV 27 中没有 fil（元数据里没有 `fil`/`tl` 键）。体积是官方元数据的 `size` 字段（mp3 压缩包字节数）。
- 许可：Mozilla Data Collective 上的 Common Voice 条目标注 CC0-1.0（搜索结果来自二手页面，官方条款页我没有拿到全文，**许可和条款原文未核实**）。
- 获取：自 2025 年 10 月起仅通过 Mozilla Data Collective 分发，需要账号并同意其服务条款（来源：Mozilla 在 HF 上的说明页 https://huggingface.co/spaces/mozilla-foundation/README ，二手检索，需到官网确认）。HF 上的 `mozilla-foundation/common_voice_*` 已不再是主要入口。我没有登录。
- 转写：人工朗读，文本为事先收集的句子，由志愿者朗读后社区验证。标点和大小写：句子集保留书面形式（我对此有常识性印象，数据卡里没有逐字核实，**未核实**）。数字写法：取决于句子本身，未核实。
- 域：朗读，短句（平均 3.4 到 5.4 秒，见元数据 `avgDurationSecs`），说话人众多（总计约 38 万贡献者，跨语言重复计数）。
- 与 FLEURS 的关系：句子来源是各语言的句子收集（维基、社区提交、公共领域文本），与 FLoRes 的重叠**未核实**。若句子来自维基百科，有可能与 FLoRes 的源句有极少量重合，建议在构造训练集时对 FLEURS dev/test 的转写做去重过滤。
- 质量问题：说话人性别和年龄分布偏斜（男性和 20 岁左右居多）；部分语言录音设备差异大；ko 只有 10.4 h（已验证 2.6 h）、ms 仅 4 h、vi 23 h，不足以单独使用。

## 6. Multilingual LibriSpeech（facebook/multilingual_librispeech）

- 链接：https://huggingface.co/datasets/facebook/multilingual_librispeech
- 覆盖：en de nl fr es it pt pl；我们的目标语言里有 en de fr es it pt。训练小时（卡片）：en 44,659.74；de 1,966.51；fr 1,076.58；es 917.68；it 247.38；pt 160.96。
- 体积（datasets-server，原始文件字节）：de 31.5 GB，fr 17.4 GB，es 15.0 GB，it 4.2 GB，pt 2.8 GB；英语未取到。
- 许可：CC-BY-4.0。获取：直接下载（`gated: false`）。
- 转写：LibriVox 有声书的人工朗读，文本取自 Project Gutenberg 等公共领域书籍。标点和大小写：我记得转写是小写无标点，但数据卡未写明，**未核实**，使用前需抽样确认。
- 域：有声书朗读，说话人数见卡片（训练集 de 约 176 人、fr 142、es 86、it 65、pt 42）。
- 与 FLEURS 的关系：无关。
- 质量问题：书面古旧文本，词汇偏文学；单人朗读环境较干净。

## 7. VoxPopuli（facebook/voxpopuli）转写子集

- 链接：https://huggingface.co/datasets/facebook/voxpopuli
- 覆盖：我们的目标语言里有 en de fr es it。转写小时（卡片）：en 543；de 282；fr 211；es 166；it 91。另有 29 小时非母语英语口音数据。
- 体积（datasets-server）：en 98.8 GB，de 52.6 GB，fr 42.1 GB，es 31.7 GB，it 17.1 GB（与 `usedStorage` 888 GB 不同，后者包含无转写音频）。
- 许可：卡片 `cc0-1.0`（欧洲议会法律声明另适用于原始数据）。获取：直接下载。
- 转写：欧洲议会官方记录与音频强制对齐，再用 ASR 预测过滤（CER 阈值 20%）。数据卡的字段是 `normalized_text`（规范化文本，示例为小写无标点）；是否同时提供带标点的 `raw_text` 在本次检索到的卡片段落里**未核实**。
- 域：欧洲议会全会发言，说话人 en 1,313 人、de 531、fr 534、es 305、it 306。
- 与 FLEURS 的关系：无关。
- 质量问题：官方记录经过编辑，与口语不完全一致；对齐误差导致部分首尾截断。

## 8. People's Speech（MLCommons/peoples_speech）

- 链接：https://huggingface.co/datasets/MLCommons/peoples_speech
- 只有英语，超过 30,000 小时；体积 2.12 TB（datasets-server：clean 398.6 GB，dirty 1,496.7 GB，clean_sa 72.5 GB，dirty_sa 149.3 GB）。不覆盖其余 17 种目标语言，只列为英语补充来源。
- 许可：CC-BY 与 CC-BY-SA 子集（卡片：「We provide CC-BY and CC-BY-SA subsets of the dataset」）。获取：直接下载。
- 转写：卡片示例为小写无标点，并说明部分训练转写很可能是 ASR 输出、部分数据对齐较差。域：archive.org 上的法庭、听证、口述历史，几乎全是美式英语。

## 9. GigaSpeech 2（speechcolab/gigaspeech2）

- 链接：https://huggingface.co/datasets/speechcolab/gigaspeech2 ；论文 https://arxiv.org/abs/2406.11546
- 覆盖：th id vi。小时（卡片）：th 训练 raw 12,901.8 / refined 10,262.0，dev 10.0，test 10.0；id raw 8,112.9 / refined 5,714.0，dev 10.0，test 10.0；vi raw 7,324.0 / refined 6,039.0，dev 10.2，test 11.0。
- 体积：卡片总 3.09 TB；`tree` 求和 th 1,278 GB，id 984 GB，vi 828 GB（下限）。
- 许可：HF 标签 `apache-2.0`，但使用条款写「only for non-commercial research and educational purposes」，且 SpeechColab 不拥有音频版权。**非商业，需标记。** 论文页面的 CC BY-NC-SA 4.0 是论文本身的许可。
- 获取：HF gated（`auto`，需登录并同意条款）。我没有接受。
- 转写：训练集是自动转写（论文称用 Whisper 做初始转写、MMS 做强制对齐，再用 Noisy Student Training 迭代修正伪标签；refined 是过滤后的子集）。dev/test 由专业人工转写。标点和大小写：卡片未说明（未核实）。数字写法：未核实。
- 格式：16 kHz 16 bit 单声道 WAV，打包为 .tar.gz，TSV 转写。域：YouTube 多领域。说话人数：未核实。
- 与 FLEURS 的关系：无已知关系。
- 质量问题：伪标签错误率较高，Thai 无空格分词和 Vietnamese 标点处理方式需抽样检查。

## 10. MediaSpeech（ymoslem/MediaSpeech）

- 链接：https://huggingface.co/datasets/ymoslem/MediaSpeech
- 覆盖：ar es fr tr，每种约 10 小时，话语数约 2,500 条（ar 2,505；es 2,507；fr 2,498；tr 2,513）。体积（datasets-server）ar 0.62 GB，es 0.59 GB，fr 0.64 GB，tr 0.62 GB，合计 2.47 GB。
- 许可：CC-BY-4.0；直接下载。
- 转写：从 YouTube 媒体视频自动切段，人工转写，并做了预处理和后处理（细节、标点、大小写均未核实）。域：新闻媒体。说话人数：未核实。
- 与 FLEURS 的关系：无关。量太小，定位是评测集（卡片称目的为 ASR 测试），可作验证或小规模补充。

## 11. Multilingual TEDx（OpenSLR 100）

- 链接：https://www.openslr.org/100/ ；论文 https://arxiv.org/abs/2102.01757
- 覆盖：es fr pt it ru el ar de（我们的目标语言里有 es fr pt it ru ar de）。小时数：官网和论文摘要页没有给出，**未核实**。下载体积（官网）：es 35 GB，fr 34 GB，pt 29 GB，it 19 GB，ru 10 GB，ar 3.6 GB，de 2.6 GB（el 5.5 GB）。
- 许可：**CC BY-NC-ND 4.0**（非商业，且禁止演绎，需重点标记；对训练模型是否算「演绎」需要用户自行判断）。获取：直接下载。
- 转写：TEDx 演讲，人工转写和翻译，按句对齐。标点和大小写：未核实。域：演讲。
- 与 FLEURS 的关系：无关。

## 12. FLEURS 训练集（标记，不得用于训练）

- 链接：https://huggingface.co/datasets/google/fleurs ；许可 CC-BY-4.0（卡片）。
- 全部 18 种目标语言都有。每种语言约 1,000 条训练话语、约 400 条验证、约 400 条测试（卡片「Data Splits」）。每个句子有 1 到 3 条录音。训练小时约 10 h（未精确核实）。体积（datasets-server，含 train/validation/test）：ar 1.9 GB，zh 3.2 GB，de 3.1 GB，en 2.4 GB，es 2.9 GB，fa 3.9 GB，fil 3.3 GB，fr 2.9 GB，id 2.9 GB，it 3.2 GB，ja 2.4 GB，ko 2.3 GB，ms 2.9 GB，pt 3.4 GB，ru 2.6 GB，th 3.1 GB，tr 2.7 GB，vi 3.0 GB。
- 转写：人工朗读维基句子，提供 `transcription`（规范化）和 `raw_transcription`（保留标点大小写）。
- 约束：FLEURS 句子来自 FLoRes，按 FLoRes+ 条款**不得用于训练**，包括 FLEURS 的 train 划分。所有训练数据都应与 FLEURS dev/test 的转写做去重。

## 13. 其他发现的大型多语种语料

### 13.1 WorldSpeech（disco-eth/WorldSpeech，2026，NeurIPS 2026 收录）

- 链接：https://huggingface.co/datasets/disco-eth/WorldSpeech ；论文 https://arxiv.org/abs/2605.09167 ；代码 https://github.com/ETH-DISCO/worldspeech
- 规模：65,072 小时、88 种语言、127 个语言-地区变体；24 kHz，OGG Opus。
- 覆盖：我们 18 种里有 17 种（缺 it）。小时数见 1.2 表（卡片 Language Coverage 表与论文表一致）。分国家配置明细：ar 9 个国家 1,001 h（bh 272.5，iq 291.9，kw 175.5，dz 92.9，ma 78.3，tn 50.9，eg 22.0，un 11.1，sa 6.1）；de：奥地利 1,077.5 + 列支敦士登 829.7；en：美国 3,725.9 等 8 个国家；es：9 个国家，西班牙 2,097.4、智利 1,740.5；fr：论文称 6,029 h（含魁北克），HF 卡片分国家表中我只核到刚果 28.0 h 与科特迪瓦 11.9 h，魁北克（`fr_ca`）在仓库目录里存在（约 28 GB），其小时数未能在卡片表中核实；fil 为 Tagalog（`tl_ph`）219.0 h；zh 为台湾国语（`zh_tw`）1,482.1 h；ja 为 LibriVox 朗读（`ja_jp`）1,387.2 h；vi 726 h 只发布对齐元数据，音频要自行从 RFA 取回，HF 仓库没有 vi 数据。
- 体积（`tree` 求和）：zh 40.2 GB，en 155.4 GB，ja 38.1 GB，ko 38.2 GB，de 52.9 GB，fr 29.4 GB，ru 39.7 GB，pt 44.5 GB，es 186.0 GB，id 14.1 GB，ms 11.5 GB，th 33.0 GB，fil 5.9 GB，ar 27.9 GB，tr 25.8 GB，fa 0.8 GB；整个仓库 2.76 TB（`usedStorage`）。
- 许可：HF 卡片 `cc-by-nc-4.0`（**非商业，需标记**）。每个配置另有来源版权依据（议会记录、美国联邦政府作品公有领域、LibriVox CC0、联合国官方记录和国际刑事法院记录标注非商业或教育用途）。论文列出 en_au 为 CC BY-NC-ND 4.0。
- 获取：直接下载（`gated: false`）。
- 转写：人工提供的转写（议会记录、广播稿、书籍文本），ASR 只用于检索和对齐，不作为真值；每行同时存 `human_transcript`、`asr_transcript` 和 CER。标点和大小写、数字写法：论文和卡片均未报告（**未核实**），官方议会记录一般有标点，但需抽样确认。
- 域：议会发言（多数）、国际机构、公共广播（VOA/RFA）、公有领域有声书（ja）。说话人数：未核实。
- 与 FLEURS 的关系：论文只把 FLEURS 作为评测基准和「先前最大公开语料」对比对象，没有把 FLEURS 当训练来源。FLoRes 重叠：未核实，但议会记录与维基句子重合概率很低。
- 质量问题：DNSMOS 平均 2.83，部分配置（tr 2.22、es_es 2.52）较差；用 CER 过滤对齐错误；议会发言语体正式；ja 是文学文本朗读，内容分布偏窄。对我们的价值：ko、th、id、ms、fil、tr、ar、zh（台湾）、ja 这类 CV 量小的语言有成规模的人工转写真人语音，是本次调研中对 18 种语言覆盖最广的新来源。

### 13.2 MSR-86K（Alex-Song/MSR-86K，Meituan，2024）

- 链接：https://huggingface.co/datasets/Alex-Song/MSR-86K ；论文 https://arxiv.org/abs/2406.18301
- 覆盖：ko es en fr de hi vi it nl pt th ru id ja ar，其中属于我们目标语言的 13 种（无 zh、ms、fil、tr、fa）。训练小时（论文 Table 2）：见 1.2 表；合计 86,299.91 h，dev 180.11 h。
- 许可：**CC BY-NC-ND 4.0**，HF 门控类型为 `manual`（需提交申请并获人工批准），条款还写明 MSR-86K 不拥有音频版权、须同意不尝试识别说话人。我没有申请。
- 转写：用户上传的 YouTube 手动字幕（不是 ASR 伪标签），与音频强制对齐，原始字幕时间戳常不准确。**文本规范化会转小写、删标点和表情符号、转换数字，因此是小写无标点的形式**，要训练书面输出需要额外恢复，不推荐直接使用。
- 格式：16 kHz 单声道 WAV。体积：未核实（HF 仓库约 10.9 GB，推测只含元数据和下载脚本，音频需要自行从 YouTube 获取）。按公式估算 86,300 h × 0.115 GB/h ≈ 9.9 TB（未压缩 WAV 估算）。
- 与 FLEURS 的关系：论文只在对比表中列出 FLEURS；测试集取自 Common Voice 并经人工核查。
- 质量问题：字幕可能存在错位；YouTube 视频可能被删除导致不可复现。

### 13.3 其他条目（只列线索，未深入）

- EuroSpeech（WorldSpeech 论文引为 de/fr 最大先前语料，de 2,184 h、fr 2,250 h）：只覆盖欧洲议会语言，细节未核实。
- ParsVoice、Neyshekar（波斯语）：属于单语言语料，由其他调研覆盖。
- 论文 arXiv 2512.07277 的 3,000 小时波斯-阿拉伯语系语料：预训练用的无标注数据，没有转写。

## 14. 与 FLEURS/FLoRes 相关的总标记

- 必须排除：FLEURS 全部划分（含 train）。
- 需要做去重核对：Common Voice（句子来自维基和社区收集，重合未核实）；其他语料无已知关系。
- 其余所有语料的音频均与 FLEURS 无来源关系。

## 15. 阻碍使用的问题清单

1. 许可：Emilia（CC BY-NC，卡片自相矛盾）、GigaSpeech 2（apache 标签与非商业条款冲突，以条款为准）、WorldSpeech（cc-by-nc-4.0）、MSR-86K 与 Multilingual TEDx（CC BY-NC-ND）。项目为非商业开源且不发布数据和权重，非商业可接受，但 ND 条款需用户判断；Common Voice 的官方条款原文未核实。
2. 门槛：Emilia、GigaSpeech 2 为 HF 点击同意；MSR-86K 为人工批准；Common Voice 需要 Mozilla Data Collective 账号。以上我都没有操作。
3. 文本形式：MSR-86K、People's Speech 为小写无标点；MLS、VoxPopuli 很可能同样（MLS 未核实，VoxPopuli 的 `normalized_text` 是小写无标点）；Granary 有标点大小写但是 Qwen 恢复；YODAS2 手动字幕标点未核实。
4. 数据重叠：YODAS2、Emilia-YODAS、Granary 的 YODAS 部分来自同一批 YouTube 视频，不要把小时数相加。
5. 量级：YODAS2 数百 TB 到数十 TB；单语言体积见汇总表，英语、西班牙语、俄语占大头。

## 16. 来源汇总

- YODAS2 数据卡与 API：https://huggingface.co/datasets/espnet/yodas2 ；https://huggingface.co/api/datasets/espnet/yodas2/tree/main/data
- YODAS v1 数据卡：https://huggingface.co/datasets/espnet/yodas
- Emilia：https://huggingface.co/datasets/amphion/Emilia-Dataset
- Granary：https://huggingface.co/datasets/nvidia/Granary ；https://huggingface.co/datasets/espnet/yodas-granary ；https://arxiv.org/abs/2505.13404
- Common Voice 统计：https://github.com/common-voice/cv-dataset ；Mozilla Data Collective：https://mozilladatacollective.com/datasets
- MLS：https://huggingface.co/datasets/facebook/multilingual_librispeech
- VoxPopuli：https://huggingface.co/datasets/facebook/voxpopuli
- People's Speech：https://huggingface.co/datasets/MLCommons/peoples_speech
- GigaSpeech 2：https://huggingface.co/datasets/speechcolab/gigaspeech2 ；https://arxiv.org/abs/2406.11546
- MediaSpeech：https://huggingface.co/datasets/ymoslem/MediaSpeech
- Multilingual TEDx：https://www.openslr.org/100/
- FLEURS：https://huggingface.co/datasets/google/fleurs
- WorldSpeech：https://huggingface.co/datasets/disco-eth/WorldSpeech ；https://arxiv.org/html/2605.09167
- MSR-86K：https://huggingface.co/datasets/Alex-Song/MSR-86K ；https://arxiv.org/html/2406.18301v1
- 体积接口：https://datasets-server.huggingface.co/size?dataset=<id> ；https://huggingface.co/api/datasets/<id>/tree/main
