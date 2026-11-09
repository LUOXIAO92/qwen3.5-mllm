# 真人语音训练数据调查

调查日期：2026-11-08。实验 v1 只用 81 小时合成音频训练 projector，验证损失约 6 轮后就不再下降，瓶颈在数据量和多样性（见 [results/v1-zh.md](../results/v1-zh.md)）。本文调查 18 种目标语言有哪些带转写的真人语音可以拿来训练，记录每个数据集的时长、容量、许可、获取方式和转写形式，最后给出取用建议。

调查分五组进行，各组的详细报告（每个数据集一节，附数字来源）放在附录：[多语种大型语料](speech-datasets/multilingual.md)、[中英日韩](speech-datasets/zh-en-ja-ko.md)、[德法俄葡西意](speech-datasets/de-fr-ru-pt-es-it.md)、[印尼、马来、泰、越、菲律宾](speech-datasets/id-ms-th-vi-fil.md)、[阿拉伯、土耳其、波斯](speech-datasets/ar-tr-fa.md)。调查只读取网页、数据集卡片和 Hugging Face 的元数据接口，没有下载数据，没有登录，也没有接受任何需要同意的条款。

## 1. 读表说明

- 时长按各数据集自己的口径（小时）；「未核实」表示在官方来源里没有找到。
- 容量是下载或存储的大小（GB），来源写在附录各节；按语言分开的数据集写该语言的容量。
- 许可按原文写。项目纯开源、不商用，只发布脚本和教程，合成或下载的数据和训练出的权重都不发布，所以限非商用（NC）的许可可以用；禁止改编（ND）的许可，训练模型是否算改编需要另行判断。
- 获取方式：「直接下载」不需要账号；「HF 需同意」要登录 Hugging Face 并点同意条款；「需账号」要注册对应平台；「需申请」要填表或签协议，等对方批准。
- 转写形式一栏，标「抽样」的是本文从 Hugging Face 的 datasets-server 接口读出前几条转写亲自看过的；没有标的来自数据集卡片或论文。
- 评测用 FLEURS 的 dev、test 集，它的句子来自 FLoRes，FLoRes 的条款要求评测内容不得进入训练数据。所有候选数据下载后都要先按 FLEURS 句子去重；明确由 FLEURS 派生的数据不能用（FLEURS 自己的 train 切分、Bayanat-Labs/arabic-asr-channel-pairs、Peacockery/farsi-asr-corpus-v4）。

## 2. 抽样看到的转写情况

数据集卡片很少写明转写有没有标点、是不是人工逐字听写。下表是抽样读到的实际情况，它决定了这些数据能不能直接当训练目标。

| 数据集 | 抽样看到的转写 |
| --- | --- |
| WorldSpeech（多语种，议会、有声书等） | 「human_transcript」是把官方记录或书籍原文按时间切到每个片段上，片段两端常常多出或少掉几个词，甚至截在半个词上；另附一份 ASR 识别结果和两者的字错误率（cer）。各语言差别很大：俄语、土耳其语、奥地利德语、台湾中文、波斯语带标点和大小写；中文为繁体，词与词之间插了空格；日语是文学有声书原文，无标点、旧式写法；巴西葡萄牙语、西班牙语、马来语、菲律宾语全小写无标点，巴西葡萄牙语的转写开头还混进了发言人身份信息；菲律宾语样本是参议院的英语发言；阿拉伯语抽到的是《古兰经》诵读（全带元音符号）、周五讲道（官方转写断字错乱）和从 PDF 抽取的联合国发言稿（带换行、字母编码损坏） |
| MLS（多语种有声书） | 德语全小写无标点 |
| CML-TTS（多语种有声书） | `text` 字段是书籍原文，带标点和大小写，拼写较旧；另有全小写的 wav2vec 转写 |
| Granary（多语种，伪标注） | 带标点和大小写，来自 Whisper 识别加标点恢复 |
| EuroSpeech（欧洲议会） | 只提供 ASR 识别出的转写，带标点，属于伪标注 |
| SADA22（沙特电视） | 沙特方言，带标点和部分元音符号，另有去掉符号的版本 |
| Neyshekar（波斯语众包朗读） | 带标点 |
| SOVA（俄语） | 全小写无标点，口语短句 |
| Porjai（泰语） | 词与词之间用空格隔开，无标点 |

## 3. 按语言汇总

每种语言列出主要候选，完整列表见附录。CV 指 Common Voice 第 27 版，时长写作「全部 / 已验证」；它的许可是 CC0，但现在只在 Mozilla Data Collective 发布，需要账号下载。

### 中文 zh

| 数据集 | 时长 | 容量 | 许可 | 获取 | 转写 |
| --- | --- | --- | --- | --- | --- |
| WorldSpeech zh_tw（台湾立法院） | 1,482 h | 40 GB | CC BY-NC 4.0 | 直接下载 | 繁体，带标点，词间有空格，两端常对不齐（抽样） |
| CV zh-CN | 1,075 / 240 h | 23 GB | CC0 | 需账号 | 未抽样 |
| Emilia zh | 49,900 h | 约 1.26 TB | CC BY-NC（卡片标签与正文矛盾） | HF 需同意 | 伪标注，带标点 |
| WenetSpeech | 10,005 h | 未核实（约 1.15 TB，估算） | CC BY 4.0，非商用 | 需申请 | 字幕识别加 ASR，非人工 |
| MAGICDATA（SLR68） | 755 h | 约 55 GB | CC BY-NC-ND 4.0 | 直接下载 | 人工，未提标点 |
| AISHELL-1 | 未核实 | 15 GB | Apache-2.0 | 直接下载 | 人工，无标点 |

### 英语 en

英语不是瓶颈。最省事的带标点选择是 LibriSpeech（约 1,000 h，CC BY 4.0，直接下载，训练集约 59 GB）加 LibriSpeech-PC 提供的恢复了大小写和标点的转写（约 25 MB，CC BY 4.0）。更大的有 Libriheavy（中等规模 5,042 h，音频 321 GB，转写有带标点版本，但音频来源 Libri-Light 的页面没写许可）、CV（3,800 / 2,800 h）、People's Speech（30,000 h，全小写无标点）。

### 日语 ja

| 数据集 | 时长 | 容量 | 许可 | 获取 | 转写 |
| --- | --- | --- | --- | --- | --- |
| ReazonSpeech v2（电视） | 约 35,000 h；可下载子集 1,000 h / 5,000 h | 65 GB / 330 GB | CDLA-Sharing-1.0，限日本著作权法第 30 条之 4 的信息分析用途 | HF 需同意 | 电视字幕对齐，未全部人工核对，带「。」 |
| WorldSpeech ja_jp（LibriVox 有声书） | 1,387 h | 38 GB | CC BY-NC 4.0 | 直接下载 | 文学原文，无标点，旧式写法，两端对不齐（抽样） |
| CV ja | 726 / 373 h | 15 GB | CC0 | 需账号 | 未抽样 |
| Emilia ja | 2,800 h | 见附录 | CC BY-NC | HF 需同意 | 伪标注 |

### 韩语 ko

韩语的单语开放数据最少。

| 数据集 | 时长 | 容量 | 许可 | 获取 | 转写 |
| --- | --- | --- | --- | --- | --- |
| WorldSpeech ko_kr（国会） | 1,454 h | 38 GB | CC BY-NC 4.0 | 直接下载 | 未能抽样（接口读取超限） |
| Zeroth-Korean | 51.6 h | 2.88 GB | CC BY 4.0 | 直接下载 | 约 3,000 个不同句子的朗读 |
| Emilia-YODAS ko | 7,300 h | 见附录 | CC BY 4.0（以卡片为准） | HF 需同意 | 伪标注 |
| MSR-86K ko | 10,300 h | 未核实 | CC BY-NC-ND 4.0 | 需申请 | 全小写无标点 |
| KsponSpeech | 约 1,000 h | 123 GB | 页面未写 | 只接受韩国国民申请 | 人工 |

### 德语 de、法语 fr、意大利语 it、西班牙语 es、葡萄牙语 pt、俄语 ru

这六种语言在 v1 中本来就较好，开放数据也最多，主要来自议会和有声书。

| 数据集 | 覆盖与时长 | 容量 | 许可 | 获取 | 转写 |
| --- | --- | --- | --- | --- | --- |
| CV | de 1,492 / 1,395 h，fr 1,217 / 1,108 h，es 2,283 / 598 h，it 431 / 364 h，pt 230 / 189 h，ru 293 / 257 h | de 37 GB，fr 31 GB，es 52 GB | CC0 | 需账号 | 未抽样 |
| MLS | de 1,967 h，fr 1,077 h，es 918 h，it 247 h，pt 161 h | de 31.5 GB | CC BY 4.0 | 直接下载 | 全小写无标点（抽样） |
| CML-TTS | de 约 2,350 h，es 约 661 h，fr 约 436 h，it 约 204 h，pt 约 139 h | de 280 GB，es 73.7 GB，fr 48 GB，it 22 GB，pt 14.4 GB | HF 与 GitHub 写 CC BY 4.0，论文徽章写 CC BY-NC-SA，需以 OpenSLR 页面确认 | 直接下载 | 书籍原文，带标点和大小写（抽样）；与 MLS、M-AILABS 同源，合并前要去重 |
| WorldSpeech | ru_ru 1,534 h，pt_br 1,763 h，es 各国合计 6,790 h，de_at + de_li 1,907 h | ru 39.6 GB，pt 44.5 GB，es 186 GB，de 53 GB | CC BY-NC 4.0 | 直接下载 | 俄语、奥地利德语带标点；巴西葡萄牙语、西班牙语全小写（抽样） |
| EuroSpeech（议会） | de 2,184 h，fr 2,250 h，it 2,814 h，pt 3,293 h | 247–380 GB | 各国议会条款 | 直接下载 | 只有 ASR 转写，伪标注（抽样） |
| Golos（ru） | 约 1,240 h | opus 20.5 GB | 许可原文在 PDF 中，未核实 | 直接下载 | 人工 |
| SOVA（ru） | 399 h | 约 41 GB | CC BY 4.0 | 直接下载 | 全小写无标点（抽样） |

### 印尼语 id、马来语 ms、泰语 th、越南语 vi、菲律宾语 fil

| 数据集 | 覆盖与时长 | 容量 | 许可 | 获取 | 转写 |
| --- | --- | --- | --- | --- | --- |
| WorldSpeech | th 1,176 h，ms 432 h，id 340 h，fil（tl_ph）219 h；vi 只有对齐信息，音频要另外获取 | th 33 GB，id 14 GB，ms 11.5 GB，fil 5.9 GB | CC BY-NC 4.0 | 直接下载 | 泰语少量标点；马来语、菲律宾语全小写无标点，菲律宾语样本为英语（抽样） |
| GigaSpeech 2 | th 10,262 h，id 5,714 h，vi 6,039 h（精选部分） | 合计数 TB，见附录 | 卡片标 Apache-2.0，条款写限非商用研究 | HF 需同意 | 伪标注 |
| YODAS2（YouTube） | id 1,420 h，vi 780 h，th 497 h，ms 28 h（上传字幕） | 见附录 | CC BY 3.0（卡片） | 直接下载 | 上传者字幕，不一定是人工逐字 |
| Porjai（th） | 卡片 700 h，与条数不符，未核实 | 7.5 GB | CC BY-SA 4.0 | 直接下载 | 人工，词间空格，无标点（抽样） |
| CV | th 427 / 173 h，id 67 / 34 h，vi 23 h，ms 4 h，fil 无 | th 9 GB | CC0 | 需账号 | 未抽样 |
| Bud500（vi） | 约 500 h | 53–98 GB（记录不一致） | 许可写法矛盾，按非商用处理 | HF 需同意 | 全小写无标点 |
| VietSpeech（vi） | 1,100 h 以上 | 130 GB | Apache-2.0 | HF 需同意 | 未写明人工还是伪标注 |

菲律宾语最缺：除了 WorldSpeech tl_ph，开放数据中整句的只有约 7 小时；较大的 IARPA Babel Tagalog（约 213 h，8 kHz 电话录音）需要从 LDC 付费购买。马来语的大型数据（mesolitica 系列）几乎都是 Whisper 伪标注，而且没有许可。

### 阿拉伯语 ar、土耳其语 tr、波斯语 fa

| 数据集 | 覆盖与时长 | 容量 | 许可 | 获取 | 转写 |
| --- | --- | --- | --- | --- | --- |
| MASC（ar） | 907 h 有转写 | HF 镜像 87.7 GB | CC BY 4.0 | 直接下载（HF 镜像） | 训练集为 YouTube 字幕，多方言 |
| SADA22（ar） | 约 438 h 有转写 | 50.4 GB | CC BY-NC-SA 4.0 | 直接下载 | 人工，沙特方言，带标点（抽样） |
| WorldSpeech ar_* | 约 1,001 h | 28 GB | CC BY-NC 4.0 | 直接下载 | 宗教诵读与讲道、联合国 PDF 文本，质量差（抽样） |
| MGB-2（ar） | 1,200 h | 约 120 GB | 研究用途 | HF 需同意 | 未核实 |
| QASR（ar） | 约 2,000 h | 未核实 | 非商用 | 需申请 | 带标点（官方） |
| WorldSpeech tr_tr（大国民议会） | 1,008 h | 25.8 GB | CC BY-NC 4.0 | 直接下载 | 带标点和大小写，数字有时写成词（抽样） |
| TSC（tr） | 218 h | 21.4 GB | CC BY 4.0（HF 标 MIT） | 直接下载 | 人工，全小写无标点 |
| CV | fa 431 / 373 h，ar 158 / 93 h，tr 136 / 130 h | fa 11 GB，ar 3.5 GB | CC0 | 需账号 | 未抽样 |
| Neyshekar（fa） | 99 h | 11.4 GB | CC0 | 直接下载 | 众包朗读，带标点（抽样） |
| WorldSpeech fa_ir（VOA） | 27.6 h | 0.8 GB | 公有领域 | 直接下载 | 带标点（抽样） |
| PerSets youtube / filimo（fa） | 385 h / 245 h | 23 GB / 36.7 GB | 整理者声明 CC0，原视频版权未说明 | 直接下载 | 字幕，未经核对 |

## 4. 取用建议

### 4.1 用 v1 的核对流程清洗真人数据

上面的数据大多不能直接当训练目标：片段两端对不齐、转写不是逐字听写、混入其他语言、没有标点。v1 已经有一套「ASR 识别后与文本比对」的流程，可以原样用在真人语音上：用每种语言配置的主 ASR 识别片段，用 `pipeline.compare` 与数据集提供的转写比对，只保留一致的片段。一致的片段里，转写本身带标点和大小写的，直接用数据集的转写作为训练目标；转写全小写、无标点的，改用 ASR 的识别结果作为训练目标——它已经通过与人工转写的比对，又带有标点和大小写。另外用语言识别（抽取原文时用过的 GlotLID）去掉语言不对的片段，例如菲律宾参议院里的英语发言。

### 4.2 先小规模试验，再整批下载

每个候选数据集先下载一小部分（例如一个分片、约 1–2 GB），跑一遍上面的清洗流程，看通过率、通过后的时长和文本是否像样，再决定整批下载哪些。这和 v1 先做小规模端到端试验的做法一致。数据盘目前剩余约 804 GB。

不需要账号、许可可以接受、能直接下载的候选，按缺口从大到小排列（容量为整批下载的大小）：

| 语言 | 候选 | 容量 |
| --- | --- | ---: |
| ko | WorldSpeech ko_kr、Zeroth-Korean | 38 + 2.9 GB |
| fil | WorldSpeech tl_ph（需去掉英语片段） | 5.9 GB |
| ms | WorldSpeech ms_my | 11.5 GB |
| id | WorldSpeech id_id | 14 GB |
| fa | Neyshekar、WorldSpeech fa_ir | 11.4 + 0.8 GB |
| th | WorldSpeech th_th、Porjai | 33 + 7.5 GB |
| ar | SADA22、MASC；WorldSpeech 阿拉伯语按来源挑选 | 50.4 + 87.7 GB |
| tr | WorldSpeech tr_tr、TSC | 25.8 + 21.4 GB |
| ja | WorldSpeech ja_jp | 38 GB |
| zh | WorldSpeech zh_tw、AISHELL-1 | 40 + 15 GB |
| ru | WorldSpeech ru_ru | 39.6 GB |
| en | LibriSpeech 加 LibriSpeech-PC 转写 | 59 GB |
| de、fr、es、it、pt | CML-TTS（先确认许可）、MLS | 见上表 |

越南语没有能直接下载的大规模候选：WorldSpeech 只给对齐信息，GigaSpeech 2、Bud500、VietSpeech 都要在 Hugging Face 上同意条款。

### 4.3 获取方式与已定事项

已定事项（2026-11-09）：

- 训练模型不视为改编，禁止改编（ND）许可的数据集可以用于训练；本项目不发布数据和训练出的权重。涉及的有 MAGICDATA（OpenSLR SLR68、SLR123，许可为 CC BY-NC-ND 4.0）、CORAA、MSR-86K、Multilingual TEDx、ViMD。Hugging Face 上的 `quanhaol/MagicData` 是 MagicMotion 论文的视频生成数据集（约 2.3 万段视频和分割标注），与这里的普通话语音库只是重名。
- Emilia 的中英日韩已在关联项目 MakeSense 中扩充并重新识别：每种语言约 8,900 条（Emilia 与 Emilia-YODAS 各约一半），各约 37–38 小时，合计约 152 小时，转写由 Qwen3-ASR 重新生成，附词级时间戳。Emilia 原始转写质量偏低，常把粤语、英语判成普通话，法语子集里也抽到英语文本，所以每条都要重新识别并检查语言。Emilia 只有中英日韩德法六种语言；德语、法语从 Hugging Face 补下，每个分片约 1 GB、含 34–44 小时音频。核对时要换一个模型：中文、英语、韩语的 v1 主 ASR 也是 Qwen3-ASR，用它核对自己的转写没有意义，这三种语言改用交叉 ASR（Whisper）核对。

以下数据集需要用户本人登录并同意条款（或提交申请）后才能下载：

| 数据集 | 语言 | 地址 | 方式 | 许可 |
| --- | --- | --- | --- | --- |
| Common Voice 第 27 版 | 17 种（无菲律宾语） | <https://mozilladatacollective.com> | 注册账号，按语言下载 | CC0 |
| GigaSpeech 2 | th、id、vi | <https://huggingface.co/datasets/speechcolab/gigaspeech2> | HF 点同意 | 卡片 Apache-2.0，条款限非商用研究 |
| ReazonSpeech | ja | <https://huggingface.co/datasets/reazon-research/reazonspeech> | HF 点同意 | CDLA-Sharing-1.0，限信息分析用途 |
| J-CHAT | ja | <https://huggingface.co/datasets/sarulab-speech/J-CHAT> | HF 点同意 | 其他（非商用），伪标注 |
| WenetSpeech4TTS | zh | <https://huggingface.co/datasets/Wenetspeech4TTS/WenetSpeech4TTS> | HF 点同意 | CC BY 4.0（论文写非商用） |
| WenetSpeech | zh | <https://wenet.org.cn/WenetSpeech/> | 填表申请密码 | CC BY 4.0，非商用 |
| AISHELL-2 | zh | <https://www.aishelltech.com/aishell_2> | 申请并签协议 | 非商用研究 |
| MSR-86K | ko 等 | <https://huggingface.co/datasets/Alex-Song/MSR-86K> | HF 人工审批 | CC BY-NC-ND 4.0 |
| Bud500 | vi | <https://huggingface.co/datasets/linhtran92/viet_bud500> | HF 点同意 | CC BY-NC-SA 4.0 |
| VietSpeech | vi | <https://huggingface.co/datasets/NhutP/VietSpeech> | HF 点同意 | Apache-2.0 |
| PhoAudiobook | vi | <https://huggingface.co/datasets/thivux/phoaudiobook> | HF 点同意 | 未写许可，条款禁止再分发 |
| MGB-2 | ar | <https://huggingface.co/datasets/QCRI/mgb2> | HF 点同意 | 研究用途 |
| QASR | ar | <https://forms.gle/4U34R3Sy9xcmtuRw8>（见 <https://huggingface.co/datasets/QCRI/QASR>） | 填表申请 | CC BY-NC 2.0（HF 卡片） |
| ParsVoice | fa | <https://huggingface.co/datasets/MohammadJRanjbar/ParsVoice> | HF 点同意 | CC BY-NC 4.0，伪标注 |
| Persian-Farsi-Speech | fa | <https://huggingface.co/datasets/Thomcles/Persian-Farsi-Speech> | HF 点同意 | CC BY 4.0 |

Babel Tagalog（LDC2016S13，菲律宾语，约 213 小时电话录音）需要从 LDC 付费购买，不在上表中。

- 伪标注数据（Granary、EuroSpeech、GigaSpeech 2、Emilia 原始转写等）和来自 YouTube 等视频平台的数据（YODAS2、MASC 的字幕部分、PerSets 等）都可以用，统一经过 4.1 节的核对流程。

## 5. 小规模试验（2026-11-09）

用 `pipeline/realspeech/pilot.sh` 对能直接取得的数据各取最多 2000 条，走完导入和核对，核对搭配见 `pipeline/runner/config_v2.yaml`（不用 Whisper）。三种核对方式：

- 转写核对：拿数据集的转写与主 ASR 比对（MakeSense 的 Emilia，转写由 Qwen3-ASR 生成，所以中文、韩语由 Fun-ASR、英语由 Parakeet、日语由 Haas-ja 核对）。
- 重新识别：不用数据集的转写，主 ASR 识别、交叉 ASR 核对，两个模型逐字一致才收（Emilia 德语、法语）。
- 切分后重新识别：WorldSpeech 的片段中位数近 30 秒，先在停顿处切成不超过 15 秒的段，再重新识别。

WorldSpeech 按人工转写直接核对时十种语言都只有 0–9% 通过（转写两端对不齐、PDF 抽取错误、非逐字记录），整段重新识别时韩语 5.9%、马来语 1.9%；切分后提高到下表的水平。

| 目录 | 样本 | 通过 | 通过率 | 全部时长 | 通过时长 | 时长通过率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| emilia-de | 2000 | 819 | 40.9% | 6.25 h | 2.24 h | 35.8% |
| emilia-fr | 2000 | 630 | 31.5% | 5.20 h | 1.32 h | 25.3% |
| makesense-zh | 2000 | 925 | 46.2% | 7.91 h | 3.41 h | 43.1% |
| makesense-en | 2000 | 749 | 37.5% | 8.11 h | 2.96 h | 36.5% |
| makesense-ja | 2000 | 444 | 22.2% | 8.65 h | 1.76 h | 20.3% |
| makesense-ko | 2000 | 285 | 14.2% | 8.44 h | 1.10 h | 13.0% |
| worldspeech-fa-split | 2876 | 799 | 27.8% | 7.55 h | 1.76 h | 23.3% |
| worldspeech-fil-split | 1590 | 347 | 21.8% | 1.01 h | 0.20 h | 19.9% |
| worldspeech-id-split | 2849 | 900 | 31.6% | 8.26 h | 2.19 h | 26.5% |
| worldspeech-ja-split | 3937 | 906 | 23.0% | 12.56 h | 2.44 h | 19.4% |
| worldspeech-ko-split | 4589 | 1043 | 22.7% | 14.72 h | 2.74 h | 18.6% |
| worldspeech-ms-split | 2753 | 278 | 10.1% | 8.81 h | 0.68 h | 7.7% |
| worldspeech-ru-split | 4292 | 1401 | 32.6% | 13.58 h | 4.05 h | 29.8% |
| worldspeech-th-split | 2900 | 967 | 33.3% | 8.40 h | 2.36 h | 28.1% |
| worldspeech-tr-split | 2871 | 767 | 26.7% | 8.38 h | 1.89 h | 22.6% |
| worldspeech-zh-split | 2862 | 833 | 29.1% | 8.22 h | 1.96 h | 23.9% |
| 合计 | 43519 | 12093 | | 136.05 h | 33.04 h | |

通过率按条（段）计算，时长通过率是通过音频占全部音频的比例。各来源都只取了一小部分，可用于估算整批的产出：通过时长约为原始时长乘以时长通过率。

抽查通过的训练目标发现三处问题：俄语、波斯语的主 ASR（GigaAM、nemotron-farsi）不输出标点和大小写，重新识别模式下训练目标也没有标点；日语 LibriVox 有声书的开场声明「この録音はパブリックドメインです。」混入；切出的半句被 Qwen3-ASR 加上句号。

俄语已改由 Qwen3-ASR 作主 ASR、GigaAM 作交叉 ASR（`pipeline/runner/config_v2.yaml`）。重跑后通过 1401 段（32.6%），通过时长不变，1401 条训练目标中 1335 条带标点。波斯语的主 ASR 仍待决定。
