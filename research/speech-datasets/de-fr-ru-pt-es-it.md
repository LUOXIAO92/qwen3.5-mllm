# 德、法、俄、葡、西、意：单语或少语种真人语音语料调查

调查日期：2026-11-08。方法：只读 Hugging Face 数据集卡、`huggingface.co/api/datasets/<id>`、`datasets-server.huggingface.co/size`（体积与行数）、官方页面。没有下载任何数据内容，没有登录，没有接受任何协议。

## 约定

- 「未核实」表示在上述来源里没有找到可核对的数字或原文。
- 体积：标注「HF 元数据」的来自 datasets-server 的 `num_bytes_original_files`（即 HF 上存储的文件字节数，通常是 parquet 内嵌音频）。标注「估算」的按 16 kHz 16-bit 单声道 WAV 约 0.115 GB/小时推算，压缩格式会更小。
- 转写形式：「标点+大小写」表示文本可直接作为书面文本目标；「小写无标点」或「伪标签」需要注明。
- 与 FLEURS/FLoRes 的关系：本次所查数据集的卡片和官网均没有提到使用 FLEURS 或 FLoRes 句子，各节统一写「卡片未提及，未核实」。来自维基百科文本的朗读语料（Tuda-De、SWC、CIEMPIESS Wikipedia）与 FLoRes 的维基来源可能有句子级重叠，需要训练前做句子去重。

## 一、汇总表（每种语言）

### 德语（de）

| 数据集 | 时长 | 许可 | 获取 | 转写 | 体积 |
|---|---|---|---|---|---|
| CML-TTS de（真人有声书，来自 MLS 重切分） | 约 2350 h（估算，见 CML-TTS 一节） | CC-BY-4.0 | HF 直接下载 | 人工书籍文本，标点+大小写；老书文本，拼写偏旧 | 280.4 GB（HF 元数据） |
| EuroSpeech de（德国联邦及州议会） | 2184 h（CER<20%）；对齐总量 2472 h | 各议会各自条款，HF 标 other | HF 直接下载 | 议会官方非逐字记录，经对齐；标点情况未核实 | 247.0 GB（HF 元数据） |
| WorldSpeech de_at + de_li（奥地利、列支敦士登议会） | 1077.5 + 829.7 h | CC BY-NC 4.0 | HF 直接下载 | 人工转写（human_transcript）；标点未核实 | 27.6 + 25.3 GB（HF 元数据） |
| Spoken Wikipedia Corpus (SWC) de | 约 386 h，其中 249 h 词级对齐 | CC BY-SA 4.0 | 官方 handle 下载 | 维基文章原文，已对齐；标点+大小写 | 未核实；估算约 44 GB（386 h WAV） |
| M-AILABS de | 237.4 h | 允许含商用的再分发（BSD 风格文本） | 直接下载 | 书籍原文+数字展开版两份 | 约 20 GiB（官方页） |
| Tuda-De | 时长未核实；27,858 条（含多麦克风重复） | CC-BY-4.0 | HF 直接下载 | 朗读句子；`sentence` 含标点，`cleaned_sentence` 无标点 | 7.22 GB（HF 元数据，parquet） |

### 法语（fr）

| 数据集 | 时长 | 许可 | 获取 | 转写 | 体积 |
|---|---|---|---|---|---|
| EuroSpeech fr（法国议会） | 2249.8 h（CER<20%）；对齐总量 5476.8 h | Licence Ouverte | HF 直接下载 | 议会官方记录，经对齐；标点未核实 | 251.8 GB（HF 元数据） |
| CML-TTS fr | 约 436 h（估算） | CC-BY-4.0 | HF 直接下载 | 书籍文本，标点+大小写 | 48.4 GB（HF 元数据） |
| M-AILABS fr | 190.5 h | 允许含商用的再分发 | 直接下载 | 书籍原文+数字展开版 | 约 15 GiB |
| African Accented French (OpenSLR 57) | 约 22 h | Apache 2.0 | OpenSLR 直接下载 | 有转写；标点未核实 | 1.8 GB（官方页） |
| WorldSpeech fr_cd/fr_ci（国际刑事法院庭审） | 28.0 h + 11.9 h | CC BY-NC 4.0，且受 ICC 条款约束 | HF 直接下载 | 人工转写 | 0.7 + 0.3 GB（HF 元数据） |

法语没有找到 100 h 以上、带标点、非议会、非有声书的开放单语真人语料（Att-Hack 等小库未纳入，见第五节）。

### 俄语（ru）

| 数据集 | 时长 | 许可 | 获取 | 转写 | 体积 |
|---|---|---|---|---|---|
| Russian Open STT | 约 20,108 h | CC-BY-NC（商用需另议） | 脚本/直链/种子 | 多为自动对齐、字幕、ASR 标签，仅验证集人工；标点未说明 | 2,369 GB WAV；356 GB opus（官方页） |
| Balalaika SFT (lab260) | 约 11,603 h（估算，4,775,837 条） | 卡片未写（未核实）；含 Golos 等有许可限制的来源 | HF 直接下载（webdataset tar） | 7 个 ASR 的 ROVER 共识 + RUPunct 恢复标点和大小写（伪标签） | 3.98 TB（HF `usedStorage`） |
| Sber Golos | 约 1240 h（crowd 1095 h + farfield 132.4 h 训练；测试 12.6 h） | 页面链接 PDF 许可，原文未核实 | 直链 | 人工标注；标点未核实 | opus 20.5 GB；WAV 约 144 GB（GitHub 页） |
| WorldSpeech ru_ru（杜马）+ ru_by | 1534.2 h + 2.9 h | CC BY-NC 4.0 | HF 直接下载 | 人工转写；标点未核实 | 39.6 + 0.1 GB（HF 元数据） |
| SOVA RuAudiobooksDevices + RuDevices | 298 h + 101 h | CC BY 4.0 | GitHub/HF 直接下载 | 人工标注；标点未核实（常见为小写无标点，未核实） | 30.24 + 10.42 GB（GitHub 页）；HF 版 29.7 GB + 6.25 GB |
| Russian LibriSpeech (RuLS, OpenSLR 96) | 约 98 h | Public Domain（美国） | OpenSLR 直接下载 | 书籍文本；格式未核实 | 9.1 GB（官方页） |
| M-AILABS ru | 46.8 h | 允许含商用的再分发 | 直接下载 | 书籍原文+数字展开版 | 约 3.6 GiB |

另有 `Muncy/AudiobookRu`（8.7 TB 有声书章节）和 `Sinoosoida/SpeechRu`（7.4 TB 播客）：无转写，不能直接使用。

### 葡萄牙语（pt）

| 数据集 | 时长 | 许可 | 获取 | 转写 | 体积 |
|---|---|---|---|---|---|
| TAGARELA（播客，巴葡 8130 h + 欧葡 842 h） | 8,972 h | CC BY-NC-SA 4.0 | HF 直接下载 | 伪标签（ElevenLabs Scribe 引导 + 微调 Whisper large-v3，Wav2Vec2 一致性过滤） | 1,402.8 GB（HF 元数据） |
| EuroSpeech pt（葡萄牙议会，欧葡） | 3293.5 h（CER<20%）；对齐总量 5096.3 h | 葡萄牙著作权法第 75 条 | HF 直接下载 | 议会官方记录；标点未核实 | 379.6 GB（HF 元数据） |
| WorldSpeech pt_br（巴西参议院） | 1763.5 h | CC BY-NC 4.0 | HF 直接下载 | 人工转写；标点未核实 | 44.5 GB（HF 元数据） |
| CORAA-MUPE-ASR | 365 h，317,743 条 | 卡片无许可字段（未核实） | HF 直接下载 | 人工；`normalized_text` 小写无标点，`original_text` 为归一化前文本 | 41.8 GB（HF 元数据） |
| CORAA v1.1 | 290.77 h，402,456 条 | CC BY-NC-ND 4.0 | HF 直接下载 | 人工校验；标点未核实 | 66.9 GB（HF 元数据） |
| CORAA NURC-SP | 239.68 h | CC BY-NC-ND 4.0 | HF 直接下载 | 人工校验；标点未核实 | 27.5 GB（HF 元数据） |
| CML-TTS pt | 约 139 h（估算） | CC-BY-4.0 | HF 直接下载 | 书籍文本，标点+大小写 | 14.4 GB（HF 元数据） |

### 西班牙语（es）

| 数据集 | 时长 | 许可 | 获取 | 转写 | 体积 |
|---|---|---|---|---|---|
| WorldSpeech es_*（西班牙、智利、墨西哥、乌拉圭等 9 个议会） | 合计约 6,792 h（es_es 2097.4、es_cl 1740.5、es_mx 901.6、es_uy 894.4、es_pe 387.4、es_ar 251.7、es_pr 228.6、es_co 156.8、es_py 133.2） | CC BY-NC 4.0 | HF 直接下载 | 人工转写；标点未核实 | 合计约 186 GB（HF 元数据逐项相加） |
| HiTZ composite_corpus_es_v1.0 | 训练 2596.3 h | 卡片 cc-by-4.0；组成成分各有许可 | HF 直接下载 | 组合自 CV、MLS、VoxPopuli、OpenSLR、巴斯克议会；已做数字和缩写清洗 | 18.5 GB（HF 元数据，parquet） |
| CML-TTS es | 约 661 h（估算） | CC-BY-4.0 | HF 直接下载 | 书籍文本，标点+大小写 | 73.7 GB（HF 元数据） |
| CIEMPIESS 系列（light 18.4 h、LibriVox 73 h、VoxForge 49.7 h、TEDx 24.5 h、Wikipedia 25.6 h） | 见各节 | 各不相同 | HF 直接下载 | 人工，小写无标点 | 1.1 / 5.1 / 3.4 / 1.6 / 1.9 GB（HF 元数据） |
| M-AILABS es | 108.6 h | 允许含商用的再分发 | 直接下载 | 书籍原文+数字展开版 | 约 8.3 GiB |

### 意大利语（it）

| 数据集 | 时长 | 许可 | 获取 | 转写 | 体积 |
|---|---|---|---|---|---|
| EuroSpeech it（意大利议会） | 2813.7 h（CER<20%）；对齐总量 4812.8 h | 卡片引用意大利参议院网站的 CC BY 4.0 | HF 直接下载 | 议会官方记录；标点未核实 | 317.4 GB（HF 元数据） |
| CML-TTS it | 约 204 h（估算） | CC-BY-4.0 | HF 直接下载 | 书籍文本，标点+大小写 | 22.0 GB（HF 元数据） |
| M-AILABS it | 127.7 h | 允许含商用的再分发 | 直接下载 | 书籍原文+数字展开版 | 约 9.5 GiB |

意大利语的单语真人语料很少，除以上外没有找到可核实的 100 h 以上开放集。

## 二、跨语言要点

1. CML-TTS 和 M-AILABS 都来自 LibriVox 有声书，CML-TTS 又由 MLS 重新切分。它们与另一位同事负责的 MLS 在音频上重叠，合并使用要先去重。
2. 有标点和大小写的真人转写主要来自三类：CML-TTS（书籍文本）、议会记录（EuroSpeech、WorldSpeech）、SWC（维基原文）。CORAA、CIEMPIESS、SOVA 多为小写无标点。
3. 议会转写通常是编辑过的非逐字记录，与实际说话有出入。EuroSpeech 用 CER 过滤缓解，训练时建议至少使用 CER<10% 子集。
4. 大规模伪标签：Balalaika（俄）和 TAGARELA（葡）。可以用，但转写由 ASR 产生，要记录为「伪标签」。
5. 非商用许可（可用，需标注）：WorldSpeech（CC BY-NC）、TAGARELA（NC-SA）、CORAA（NC-ND）、Open STT（NC）。

## 三、各数据集详情

### 3.1 CML-TTS（de / fr / it / pt / es 子集）
- 链接：HF https://huggingface.co/datasets/ylacombe/cml-tts ；官方 https://www.openslr.org/146/ ；论文 https://arxiv.org/abs/2306.10097
- 时长：论文摘要给出 CML-TTS 训练 YourTTS 使用 3,176.13 h，没有逐语言表。HF 版逐语言行数来自 datasets-server（de 619,076；fr 115,100；it 53,945；pt 36,696；es 174,752，含 dev/test）。估算时长 = 行数 × 前几批样本的平均时长（约 13.6 s）：de 约 2352 h，fr 约 436 h，it 约 204 h，pt 约 139 h，es 约 661 h。这个平均值来自 datasets-server `/statistics`（只覆盖部分行），所以全部为估算，且与论文 3,176 h 的总量口径不同，未核实。
- 许可：HF 卡 `cc-by-4.0`；GitHub 称 CC-BY 4.0；arXiv 页徽章链接指向 CC BY-NC-SA 4.0，两处不一致，使用前以 OpenSLR 页为准（未核实）。
- 获取：HF 直接下载；OpenSLR 有分语言 tar.bz。
- 转写：`text` 为书籍原文，带标点和大小写（德语样本已核对，含旧拼写如「daß」）；另有 `transcript_wav2vec`（小写无标点的 ASR 结果，用于过滤）。数字归一化情况未核实。
- 领域/说话人：LibriVox 志愿者朗读的公版书；有 `speaker_id` 字段，人数未核实。
- 格式/体积：parquet 内嵌音频，体积见汇总表（de 280.4、fr 48.4、it 22.0、pt 14.4、es 73.7 GB；全 7 语合计 580 GB）。
- 与 FLEURS：卡片未提及，未核实。公版老书文本，与现代 FLoRes 句子重叠的可能性很低，但未核实。
- 已知问题：书籍文本老旧（1884–1964 年出版的占多数），拼写和用词偏古；片段长度集中在 10.9–16.3 s（统计的最小/最大值）；是 TTS 数据集，只保证文音基本对应，需要用 `levenshtein` 字段过滤。

### 3.2 EuroSpeech
- 链接：https://huggingface.co/datasets/disco-eth/EuroSpeech ；论文 https://arxiv.org/abs/2510.00514
- 范围：22 国议会，不含西班牙和俄罗斯。本调查相关：de、fr、it、pt（葡萄牙，欧葡）。
- 时长（卡片表）：de 2184.4 h、fr 2249.8 h、it 2813.7 h、pt 3293.5 h（CER<20%，即 HF 默认提供的子集）；对齐总量 de 2472.2、fr 5476.8、it 4812.8、pt 5096.3 h。
- 许可：HF 标 `other`，卡片称各国议会条款不同，并说明不保证许可信息准确。德国一行未逐条核对（未核实）；法国 Licence Ouverte；意大利引用 CC BY 4.0；葡萄牙引用著作权法第 75 条。
- 获取：HF 直接下载，无门槛。
- 转写：议会的非逐字官方记录，经对齐流水线与音频对应，并用 CER 过滤；标点、大小写、数字写法未核实。
- 领域：议会发言，正式书面化口语。说话人数未核实。
- 格式/体积：16 kHz，parquet 内嵌音频。体积（HF 元数据）：de 247.0、fr 251.8、it 317.4、pt 379.6 GB，四国合计 1,195.8 GB。另有 24 kHz 版 `disco-eth/EuroSpeech-24kHz`。
- 与 FLEURS：卡片未提及，未核实。
- 已知问题：转写非逐字，有漏词和改写；CER 过滤后仍可能含对齐错误。

### 3.3 WorldSpeech
- 链接：https://huggingface.co/datasets/disco-eth/WorldSpeech ；论文 https://arxiv.org/abs/2605.09167
- 总量：65,072 h，88 种语言，127 个语言-地区变体；24 kHz OGG Opus。
- 相关配置与时长（卡片表）：de_at 1077.5、de_li 829.7；es_es 2097.4、es_cl 1740.5、es_mx 901.6、es_uy 894.4、es_pe 387.4、es_ar 251.7、es_pr 228.6、es_co 156.8、es_py 133.2；fr_cd 28.0、fr_ci 11.9（国际刑事法院庭审）；pt_br 1763.5；ru_ru 1534.2、ru_by 2.9。没有 it 配置，没有法国本土配置。
- 许可：整体 CC BY-NC 4.0，卡片说明各配置的来源许可更严格时以更严格者为准；ICC 两个配置注明非商用/教育用途。
- 获取：HF 直接下载，无门槛。
- 转写：字段 `human_transcript`（人提供）、`asr_transcript`、`cer`。标点和大小写情况未核实。
- 体积（HF 元数据）：de_at 27.6、de_li 25.3、es_* 合计约 186.1、pt_br 44.5、ru_ru 39.6 GB（体积小是因为 Opus 压缩）。
- 与 FLEURS：卡片未提及，未核实。
- 已知问题：非商用；卡片列的 DNSMOS 均值约 2.5–3.1，音质一般；议会转写非逐字。

### 3.4 Spoken Wikipedia Corpus（SWC，德语）
- 链接：https://nats.gitlab.io/swc/
- 时长：德语 1,010 篇文章、339 位朗读者，约 386 h，其中约 249 h 词级对齐，129 h 还有音素级对齐。
- 许可：CC BY-SA 4.0。
- 获取：官方 handle 下载 http://hdl.handle.net/11022/0000-0007-C641-0（2017 年 10 月版，含英、德、荷），也可用 Bitbucket 脚本自行对齐。无需登录（页面所述）。
- 转写：维基文章原文，已分句、分词并对齐；原文含标点和大小写；数字归一化是对齐用的 normalized 形式，与书写形式并存。
- 领域：志愿者朗读维基百科文章，不是录音棚。
- 格式/体积：体积未核实；估算 386 h × 0.115 GB/h ≈ 44 GB（WAV 假设；原始多为 ogg/mp3，会更小）。
- 与 FLEURS/FLoRes：文本来自维基百科，FLoRes 的来源包含维基系站点（Wikinews、Wikijunior、Wikivoyage），句子级重叠需要去重，未核实。
- 已知问题：需要自行解析 XML 并按对齐切分片段；长录音需切句；对齐质量各篇不同。

### 3.5 M-AILABS Speech Dataset（de / fr / es / it / ru）
- 链接：https://github.com/imdatceleste/m-ailabs-dataset （官方站点 caito.de 在本环境无法解析）
- 时长与体积（GitHub 页）：de 237h22m ≈ 20 GiB；fr 190h30m ≈ 15 GiB；es 108h34m ≈ 8.3 GiB；it 127h40m ≈ 9.5 GiB；ru 46h47m ≈ 3.6 GiB。
- 许可（页面原文）：「Redistribution and use in any form, including any commercial use, with or without modification are permitted.」
- 获取：直接下载。
- 转写：每条有两份文本，第一份是原文（数字未归一化，带标点），第二份是归一化文本（数字写成单词，外来字符转写）。
- 领域：LibriVox 真人有声书，文本来自 Project Gutenberg（1884–1964 年出版）。说话人少（每语言数位到十余位，具体未核实）。
- 与 FLEURS：卡片未提及，未核实。
- 已知问题：与 CML-TTS/MLS 的 LibriVox 音频可能重叠；说话人很少；老书文本。HF 上有第三方镜像（例如 `gigant/m-ailabs_speech_dataset_fr` 许可字段只写 `cc`，应以官方页为准）。

### 3.6 Tuda-De（德语）
- 链接：https://huggingface.co/datasets/uhhlt/Tuda-De
- 时长：卡片未给小时数，未核实；共 27,858 行，每句有多个麦克风版本（Kinect-Beam、Kinect-RAW、Realtek、Samson、Yamaha），独立句子数小于行数。
- 许可：HF 卡 `cc-by-4.0`。
- 获取：HF 直接下载。
- 转写：朗读句子，`sentence` 含原标点（样例里标点前有空格，例如「erweitert ,」），`cleaned_sentence` 去掉标点、数字展开。大小写保留。
- 领域：德语维基百科、Europarl、命令控制句和网页句，在同一房间距麦克风约 1 m 录音；多数说话人来自黑森州，年龄 21–30 岁。
- 体积：7.22 GB（HF 元数据）。
- 与 FLEURS：句子取自维基百科，与 FLoRes 可能有重叠，未核实。
- 已知问题：远场、多麦克风重复；说话人数量少于句子数；`sentence` 标点带空格，需要重新整理。

### 3.7 African Accented French（OpenSLR 57）
- 链接：https://www.openslr.org/57/
- 时长约 22 h，1.8 GB（官方页）；许可 Apache 2.0；OpenSLR 直接下载。转写覆盖所有录音，标点和大小写未核实。语言为非洲口音法语，说话人数和领域未核实。与 FLEURS 关系：未提及。HF 镜像 `gigant/african_accented_french` 用脚本加载，datasets-server 无法给出体积。

### 3.8 Russian Open STT
- 链接：https://github.com/snakers4/open_stt
- 时长：约 20,108 h，约 1650 万条；WAV 共 2,369 GB，opus 约 356 GB（官方页）。主要子集：radio_v4 10,430 h、public_speech 2,709 h、audiobook_2 1,511 h、radio_2 1,439 h、public_youtube1120 1,104 h、phone calls 601+211 h、tts_russian_addresses 754 h（合成语音，需排除）。
- 许可：页面写「CC-BY-NC and commercial usage available after agreement with dataset authors」，徽章为 CC BY NC 4.0。
- 获取：`download.sh`/`download.py` 脚本，DigitalOcean Spaces、Azure Open Datasets 直链、torrent、Zenodo 镜像；无需账号。
- 转写：多数是自动对齐（电台、公开演讲、有声书）、YouTube 字幕、电话和短故事的 ASR 标签；只有验证集为人工标注。标点和大小写：页面未说明，仅要求把「ё」统一为「е」。
- 质量：官方标注多数子集约 95%，电话约 70%（页面所述）。
- 与 FLEURS：未提及。
- 已知问题：转写质量参差；含合成子集；许可为非商用；子集中的 radio_v4 单文件 176 GB。

### 3.9 Balalaika SFT（lab260，ROVER7 + 标点）
- 链接：https://huggingface.co/datasets/lab260/balalaika-sft-rover7-punct
- 规模：4,775,837 条，约 11,603 h（按 43,484 条分层抽样估算，95% 区间 11,016–12,227 h）。来源（by_source.csv）：sova_balalaika 约 9,723 h、youtube_balalaika 约 1,485 h、igm 118 h、webinars 109 h、upvote 105 h、golos 40 h、buldjat 21 h、tuchniyzhab 2.6 h。
- 许可：卡片无许可字段，正文也没有找到许可条款（未核实）。含 Golos 等有自身许可的来源，使用前必须逐源确认。
- 获取：HF 直接下载（webdataset tar），无门槛。
- 转写：伪标签。7 个 ASR（GigaAM 三种、Vosk、T-One、Parakeet TDT 0.6B v3、Whisper large-v3-turbo）做 ROVER 共识，再用 RUPunct_big 等恢复标点和大小写；主要训练文本字段是 `restored_punctuation`。卡片明确说自动一致性不等于人工准确率。
- 体积：约 3.98 TB（HF API `usedStorage`）。
- 与 FLEURS：未提及，未核实。
- 已知问题：伪标签；标点为模型恢复；来源混杂，许可不明。

### 3.10 Sber Golos
- 链接：https://github.com/sberdevices/golos ；HF 镜像 https://huggingface.co/datasets/SberDevices/Golos 、10 h 子集 `bond005/sberdevices_golos_10h_crowd`（18,780 行，2.27 GB）
- 时长：crowd 训练 1,095 h、测试 11.2 h；farfield 训练 132.4 h、测试 1.4 h，合计约 1,240 h。
- 许可：GitHub 页链接了中英文 PDF，没有贴出原文，原文未核实；HF 10 h 子集标 `other`。
- 获取：sc.link 直链；HF 子集直接下载。
- 转写：人工标注。标点和大小写、数字写法：卡片未说明，未核实。
- 领域：众包朗读（crowd）与对智能设备说话（farfield）。
- 体积：opus 20.5 GB，WAV 约 144 GB（GitHub 页，按各 tar 相加）。
- 与 FLEURS：未提及。
- 已知问题：短指令式语句较多，领域偏窄。

### 3.11 SOVA RuDevices / RuAudiobooksDevices
- 链接：https://github.com/sovaai/sova-dataset ；HF https://huggingface.co/datasets/bond005/sova_rudevices 、https://huggingface.co/datasets/dangrebenkin/sova_rudevices_audiobooks
- 时长：RuAudiobooksDevices 298 h（30.24 GB），RuDevices 101 h（10.42 GB）（GitHub 页）。HF 版：RuDevices 54,586 行 6.25 GB；有声书 203,150 行 29.7 GB。
- 许可：CC BY 4.0（权利人 Virtual Assistant, LLC）。获取：直接下载。
- 转写：人工标注，官方称质量 99%/98%；标点、大小写未核实。与 FLEURS：未提及。
- 已知问题：RuDevices 为设备端短句；HF 镜像为第三方整理。

### 3.12 Russian LibriSpeech（RuLS）
- 链接：https://www.openslr.org/96/ 。约 98 h，9.1 GB，Public Domain（美国）。转写格式未核实。与 FLEURS：未提及。

### 3.13 TAGARELA（葡萄牙语播客）
- 链接：https://huggingface.co/datasets/freds0/TAGARELA ；论文 https://arxiv.org/abs/2603.15326 （ICASSP 2026）
- 规模：8,972 h，16,806 集播客，2,094 档节目，13,368 位说话人；巴葡 8,130 h，欧葡 842 h；另有约 2,800 h 的 clean 子集。8,262,923 行。
- 许可：CC BY-NC-SA 4.0（卡片原文）。获取：HF 直接下载。
- 转写：伪标签。先用 ElevenLabs Scribe 做种子语料，再微调 Whisper large-v3 做伪标注，并用 Wav2Vec2-XLS-R 一致性过滤；卡片声明这些是大规模 ASR 生成标签，不是人工标注。标点和大小写：卡片未明说，未核实。
- 体积：1,402.8 GB（HF 元数据；卡片称下载约 1.21 TB）。
- 与 FLEURS：未提及。已知问题：伪标签偏差；播客含口语不流利和背景音；NC-SA。

### 3.14 CORAA 系列（巴西葡语）
- CORAA v1.1：https://huggingface.co/datasets/Racoci/CORAA-v1.1 （第三方镜像），官方 https://github.com/nilc-nlp/CORAA 。290.77 h，402,456 条，CC BY-NC-ND 4.0，66.9 GB。来源 ALIP、C-ORAL Brazil、NURC-Recife、SP2010、TEDx。含欧葡变体标签。转写由标注员校验；标点和大小写未核实。
- CORAA NURC-SP：https://huggingface.co/datasets/nilc-nlp/CORAA-NURC-SP-Audio-Corpus ，239.68 h，355,635 行（original+filtered 两配置，实际音频约 27.5 GB 一份），CC BY-NC-ND 4.0，转写称 human-verified。
- CORAA-MUPE-ASR：https://huggingface.co/datasets/nilc-nlp/CORAA-MUPE-ASR ，289 场生活故事访谈共 365 h，317,743 条，41.8 GB，16 kHz。卡片没有许可字段（未核实）；`normalized_text` 是小写无标点，`original_text` 为归一化前文本，标点情况未核实。论文 COLING 2025。
- 共同问题：多为自发口语，含犹豫和重复；NC-ND 禁止衍生，是否包括训练模型需要使用者自行判断；与 FLEURS 关系未提及。

### 3.15 HiTZ composite_corpus_es_v1.0（西班牙语）
- 链接：https://huggingface.co/datasets/HiTZ/composite_corpus_es_v1.0
- 训练 2596.32 h（CV18 538.82、OpenSLR 45.58、MLS 922.47、VoxPopuli 142.96、巴斯克议会 949.27），1,142,586 句。许可卡标 cc-by-4.0，成分各有自己的许可。
- 转写：对 OpenSLR 和 VoxPopuli 部分清除了缩写和数字；其余成分的标点情况未核实。
- 体积：18.5 GB（HF 元数据，parquet；音频可能经过压缩或抽样，与时长不成比例，未核实）。
- 价值：这个数据集大部分成分（CV、MLS、VoxPopuli）属于另一位同事的范围，真正新增的是 OpenSLR 西班牙语子集（SLR 39、61、67、71–75、108）和巴斯克议会（西语-巴斯克语双语，西语部分约 949 h）。
- 与 FLEURS：卡片未提及；CV 和 VoxPopuli 与 FLEURS 无句子关联的结论未核实。

### 3.16 CIEMPIESS 系列（墨西哥西语）
- 链接：https://huggingface.co/ciempiess
- light：18 h 25 min，16,663 条，87 位说话人，CC-BY-SA-4.0，1.12 GB；电台/电视访谈，法律话题为主。
- LibriVox Spanish：73 h 1 min，36,338 条，154 位说话人，CC-BY-SA-4.0，5.09 GB。
- VoxForge Spanish：49 h 42 min，21,692 条，GPL-3.0，3.44 GB。
- TEDx Spanish：24 h 29 min，11,243 条，CC-BY-NC-ND-4.0，1.61 GB。
- Wikipedia Spanish：25 h 37 min，11,569 条，CC-BY-SA-3.0，1.89 GB（朗读维基文章，与 FLoRes 可能重叠，未核实）。
- 转写：人工，`normalized_text` 为小写无标点，卡片样例已核对。
- 总体量小，合计约 190 h，适合作为拉丁美洲口音补充。

### 3.17 不能直接使用的大型资源
- `Muncy/AudiobookRu`（https://huggingface.co/datasets/Muncy/AudiobookRu ）：俄语有声书章节 MP3，892,483 章，8.7 TB，许可 `other`（「scraped-from-web-for-research-use」）；卡片只提到有些章节有关联文本版本，没有逐段转写。
- `Sinoosoida/SpeechRu`（https://huggingface.co/datasets/Sinoosoida/SpeechRu ）：约 18.6 万集俄语播客，7.4 TB，无转写。

## 四、阻碍与风险

1. 俄语 Golos 的许可原文只在 PDF 里，本次未能读取；Balalaika 卡片无许可且含多来源，Open STT 为 CC-BY-NC。
2. CML-TTS 许可在 HF/GitHub（CC-BY-4.0）与 arXiv 徽章（CC BY-NC-SA 4.0）之间不一致，应以 OpenSLR 页确认。
3. 议会类语料（EuroSpeech、WorldSpeech）的转写是非逐字记录，标点和大小写情况本次没有核实，训练前需要抽样检查。
4. CORAA 系列带 ND（禁止衍生）条款；MuPe 无许可字段。
5. 意大利语和法语的开放真人语料明显偏少：意大利语只有议会和有声书，法语除议会和有声书外只有 22 h 的 African Accented French。
6. 所有数据集卡片均未提及 FLEURS/FLoRes。维基文本来源的语料（Tuda-De、SWC、CIEMPIESS Wikipedia）需要句子级去重。

## 五、已查看但未纳入详述

- Att-Hack（`FrancophonIA/Att-Hack`）：小型法语语气数据集，未核实许可与规模，不适合作为大规模来源。
- Cnam-LMSSC/vibravox：法语骨传导麦克风语料，音频形态特殊，未纳入。
- Common Voice、MLS、VoxPopuli、YODAS、MediaSpeech、FLEURS、Multilingual TEDx 属另一位同事范围。
- 各类商业小数据集（SpeechDataAI、UniDataPro 等）为付费样本，未纳入。
