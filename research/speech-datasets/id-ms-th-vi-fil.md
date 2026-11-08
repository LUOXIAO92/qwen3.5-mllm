# 印尼语、马来语、泰语、越南语、菲律宾语（他加禄语）真人语音语料调研

调查日期：2026-11-08。只做网页调研，没有下载任何数据内容，没有登录，没有同意任何协议。

## 约定

- 「未核实」表示没有在一手来源（数据集卡片、官网、论文）上确认到。
- 存储大小栏的来源标记：`HF-API` 是 `https://huggingface.co/api/datasets/<id>` 返回的 `usedStorage`（整个仓库占用，可能含重复文件或多个版本，只能当上限参考）；`DS-size` 是 `datasets-server.huggingface.co/size`；`卡片` 是数据集卡片里的 `download_size`；`官网` 是论文或官方页面。
- 转写形式缩写：P+C 表示有标点和大小写；无P 表示无标点；pseudo 表示机器伪标注。
- 与 FLEURS/FLoRes 的关系：下面所有数据集都没有在卡片里声明使用 FLEURS/FLoRes 句子。凡是由 Common Voice、YODAS、Emilia、GigaSpeech 2 派生的数据集，另一份报告负责的语料可能与之重叠，需要另行去重。FLoRes 句子来自维基新闻等来源，新闻类文本（ViMD、TITML-IDN 的报刊文本）理论上可能撞句，建议训练前对 FLEURS dev/test 文本做字符串去重。

## 一、汇总表

| 语言 | 数据集 | 时长 | 许可 | 获取方式 | 转写形式 | 大小 |
|---|---|---|---|---|---|---|
| id | TITML-IDN | 约 14.5 h（未核实一手） | other，仅研究 | HF 直下 | 人工；形式未核实 | 7.2 GB（HF-API） |
| id | ID2 (Zenodo) | 7.7 h | CC BY 4.0 | Zenodo 直下 | 人工 | 1.3 GB（Zenodo） |
| id | LibriVox Indonesia | 约 8 h（7 种语言合计，印尼语更少） | cc（含糊） | HF 直下 | 对齐自书本文本，P+C 未核实 | 0.97 GB（HF-API） |
| id | mesolitica pseudolabel-indonesian | 未核实 | 无声明 | HF 直下 | pseudo（Whisper large-v3） | 64.7 GB（HF-API） |
| ms | mesolitica Malaysian-STT-Whisper | 未核实（数万小时量级） | 无声明 | HF 直下 | pseudo（Whisper large-v3 + 翻译后处理） | 585 GB（HF-API） |
| ms | mesolitica Malaysian-STT-Whisper-Stage2 | 未核实 | 无声明 | HF 直下 | pseudo，较高置信度 | 62.8 GB（HF-API） |
| ms | malaya-speech-malay-stt | 未核实（1,635,599 条） | 无声明 | HF 直下 | pseudo（Google STT）+ 私有数据 | 53.3 GB（卡片） |
| ms | malaysia-parliament-youtube | 3985.4 h（无转写） | CC-BY-NC-4.0 | HF 直下 | 仅音频，无转写 | 224 GB（HF-API） |
| ms | Malaysian-Emilia | 数千小时 | CC-BY-NC-4.0 | HF 直下 | pseudo | 4.15 TB（HF-API） |
| ms | IMDA NSC 镜像 | 约 10,659 h（6 部分，卡片标语言为 en） | Singapore Open Data Licence | HF 直下 | 人工，主要英语 | 163 GB（HF-API） |
| th | Porjai 中部泰语 | 卡片称 700 h | CC-BY-SA-4.0 | HF 直下 | 人工，应无标点 | 7.5 GB（DS-size） |
| th | Porjai 方言（korat 等三种） | 各 40 h | korat CC-BY-SA；khummuang、pattani CC-BY-NC-SA | HF 直下 | 人工 | korat 0.51 GB（DS-size） |
| th | processed-voice-th-169k | 169,550 条（时长未核实，约 165 h 为推算） | CC-BY-SA-4.0（来源不明） | HF 直下 | 来源不明 | 12.0 GB（卡片） |
| th | Gowajee | 约 17 h，20,308 条 | MIT | Google Drive | 人工，形式未核实 | 原版未核实（HF 镜像 39 GB） |
| th | Thai Elderly Speech | 17 h 11 min，19,200 条 | CC-BY-SA-4.0（SEACrowd 卡片） | GitHub release | 人工，形式未核实 | 未核实（HF 镜像 9.4 GB） |
| th | LOTUSDIS | 114 h | CC-BY-SA-4.0 | 下载位置未核实 | 未核实 | 未核实 |
| th | LOTUS-TRD | 约 180 h，四种方言 | 以 CC-BY-SA 4.0 为基础 | 未核实 | 未核实 | 未核实 |
| vi | VIVOS | 15 h，12,420 条 | CC BY-NC-SA 4.0 | HF 直下 | 人工，无标点 | 1.48 GB（卡片） |
| vi | Bud500 | 约 500 h，634,158 条训练 | 卡片冲突：CC-BY-NC-SA 与 Apache-2.0 | HF 门控（自动） | 人工，小写无标点 | 59.5 GB（HF-API）；卡片写 98 GB，文件列表约 52.7 GB |
| vi | VietSpeech | 1100+ h，约 1,026,047 条 | Apache-2.0 | HF 门控（自动） | 未核实（疑似伪标注） | 130.5 GB（HF-API） |
| vi | VietMed | 有标注 16 h，另有无标注 | MIT | HF 直下 | 人工，医疗领域 | 0.18 GB（卡片） |
| vi | ViMD | 102.56 h，约 19,000 条 | CC-BY-NC-ND-4.0 | HF 直下 | 人工 | 74.2 GB（HF-API）；parquet 14.4 GB（DS-size，部分） |
| vi | PhoAudiobook | 941 h（论文） | 无声明，有条款 | HF 门控（自动） | 面向 TTS，转写来源未核实 | 167 GB（HF-API） |
| vi | VietSuperSpeech | 103.18 h，32,267 条 | 卡片写 MIT（来源为 YouTube，存疑） | HF 直下 | pseudo（Zipformer-30M-RNNT-6000h） | 64.3 GB（HF-API） |
| vi | Gemini 转写 YouTube 集 | 未核实 | other | HF 门控（人工审批） | pseudo（Gemini） | 41.2 GB（HF-API） |
| fil | Filipino Speech Corpus (FSC) | 总 65.1 h，句子级仅约 7 h | MIT（HF 重打包，原始语料授权未核实） | HF 直下 | 人工/机器分词混合 | 6.9 GB（卡片） |
| fil | Silencio Tagalog 样本 | 2.03 h，90 条 | CC-BY-NC-4.0 | HF 直下 | 人工 | 1.0 GB（HF-API） |
| fil | InfoBayAI 呼叫中心 | 卡片称 FIL 169 h、TL 477 h（搜索摘要，未在一手页面确认） | cc-by-4.0 | HF 门控（自动） | 未核实 | 未核实 |
| fil | IARPA Babel Tagalog (LDC2016S13) | 约 213 h | LDC 许可 | 付费或 LDC 会员 | 人工，电话语音 8 kHz | 未核实 |

## 二、各语言结论摘要

- 印尼语：开放的真人转写语料极少，TITML-IDN（约 14.5 h，仅研究）、ID2（7.7 h，CC BY 4.0）、LibriVox Indonesia（很小）合计不到 30 h。大规模数据只能靠 GigaSpeech 2（另一份报告）、Common Voice、YODAS，或 mesolitica 的 Whisper 伪标注。属于数据缺口，没有找到新的 2025–2026 大型开放真人语料。
- 马来语：量大但几乎全是伪标注，且 mesolitica 大部分仓库没有许可声明。可用性最好的真人 + 人工转写数据是 IMDA NSC，但卡片语言标注为英语，马来语占比未核实。
- 泰语：Porjai 中部泰语（700 h，CC-BY-SA，人工转写）是最好的单语言真人语料，另有 Gowajee、Thai Elderly Speech、LOTUSDIS 等小而干净的数据集。转写预计无标点。
- 越南语：Bud500（500 h）和 VietSpeech（1100+ h）量大，但都是门控下载，转写无标点；Bud500 许可有冲突，VietSpeech 转写来源未核实。VIVOS、VietMed、ViMD 是干净的人工转写小集，ViMD 为 ND 条款。
- 菲律宾语：没有找到开放的大规模人工转写语料。FSC 句子级只有约 7 h；Babel Tagalog（约 213 h）需要 LDC 付费。这一语言仍然是最薄弱的。

## 三、逐数据集说明

### 印尼语

#### TITML-IDN
- 链接：https://huggingface.co/datasets/holylovenia/TITML-IDN ；SEACrowd 加载脚本 https://huggingface.co/datasets/SEACrowd/titml_idn
- 语言与时长：印尼语；2025 年论文引用约 14.5 h，20 名说话人（11 男 9 女）（二手搜索摘要，一手论文未逐页核对，标未核实）。
- 许可：卡片元数据 `license: other`；SEACrowd 脚本描述为 "For research purposes only. If you use this corpus, you have to cite (Lestari et al, 2006)"（经搜索摘要转述，原文未直接打开，标未核实）。
- 获取：HF 直下，无门控。
- 转写：卡片说文本来自报纸和杂志文章，人工录音朗读；是否带标点大小写未核实。
- 领域：朗读新闻，20 位说话人，说话人太少。
- 大小：7.2 GB（HF-API）。
- 与 FLEURS：新闻文本，建议去重。
- 质量问题：说话人极少，录音为 2000 年代实验室录音。

#### ID2 (Indonesian Dataset 2)
- 链接：https://zenodo.org/records/15946165
- 时长：7 h 40 min 48 s，31 名说话人（16 男 15 女，17–25 岁），爪哇、巽他、巴塔克、巴厘、米南加保族。
- 许可：CC BY 4.0（Zenodo 页面）。
- 获取：Zenodo 直下，无限制。
- 转写：人工；页面句子数前后矛盾（330 句 vs 10,230 句），未核实；标点形式未核实。
- 领域：棚内安静录音，方言口音。
- 大小：1.3 GB（Zenodo，ID2.rar）。
- 质量问题：说话人年龄窄，句子重复度高（句数少）。

#### LibriVox Indonesia 1.0
- 链接：https://huggingface.co/datasets/indonesian-nlp/librivox-indonesia
- 时长：卡片写 7 种印尼语言合计 8 h，印尼语单独时长未核实。
- 许可：卡片 `license: cc`（未指明具体版本，LibriVox 原始音频为公有领域）；含糊，需标注。
- 获取：HF 直下。
- 转写：用自研强制对齐从书本文本切出；标点形式未核实。
- 大小：0.97 GB（HF-API）。
- 质量问题：体量很小；对齐错误率未核实。

#### mesolitica pseudolabel-indonesian-large-v3-timestamp
- 链接：https://huggingface.co/datasets/mesolitica/pseudolabel-indonesian-large-v3-timestamp （该仓库没有 README）
- 内容：来自 Malaysian-STT-Whisper 卡片的引用，为 Whisper large-v3 伪标注的印尼语语音；时长、来源 YouTube 频道均未核实。
- 许可：无声明（缺失，需标注）。
- 大小：64.7 GB（HF-API）。
- 质量问题：伪标注，来源频道版权状态未知。

#### 未采用或需要联系的条目
- IDSV 约 80 h 汇编（arXiv 2024，来自搜索摘要，未核实）；INDspeech_NEWS_LVCSR 超过 40 h、400 名说话人（NusaCrowd 提及，获取方式未核实）。
- SpeechDataAI/indonesian-conversational-speech：约 2000 h，商业数据，HF 仓库只有卡片（license: other），不可用。
- BabelSpeech/40hours_Indonesian_Colloquial_ASR_Speech_Dataset：门控人工审批，标 apache-2.0，来源是数据商样本，可信度未核实。

### 马来语

#### mesolitica Malaysian-STT-Whisper 与 Stage2
- 链接：https://huggingface.co/datasets/mesolitica/Malaysian-STT-Whisper ；https://huggingface.co/datasets/mesolitica/Malaysian-STT-Whisper-Stage2
- 内容：Whisper large-v3 伪标注，经重复三元组检查、Silero-VAD、强制对齐打分，并用 nanot5 翻译模型做后翻译；合并了马来西亚 YouTube、新加坡（IMDA）、印尼、普通话、泰米尔、科学类英语、砂拉越马来语、Iban 等。含词级时间戳。
- 语言：ms、en、zh、ta、id 混合；ms 单独时长未核实。
- 许可：无声明（缺失）。
- 获取：HF 直下，zip 分卷。
- 转写：pseudo（Whisper large-v3）；带标点大小写与否未核实，Whisper 输出通常有。
- 大小：585 GB 与 62.8 GB（HF-API）。
- 质量问题：伪标注，30 秒块合并长音频；来源 YouTube 版权不明；Stage2 为较高置信度子集，适合后期微调。

#### malaya-speech-malay-stt
- 链接：https://huggingface.co/datasets/mesolitica/malaya-speech-malay-stt
- 内容：1,635,599 条，16 kHz；卡片称由 Google Speech-to-Text 半监督标注与私有数据合并。时长未核实。
- 许可：无声明。
- 转写：pseudo（Google STT）；形式未核实。
- 大小：53.3 GB（卡片 download_size）。

#### malaysia-ai/malaysia-parliament-youtube
- 链接：https://huggingface.co/datasets/malaysia-ai/malaysia-parliament-youtube
- 内容：马来西亚国会 YouTube 全场视频音频，2068 个文件，3985.4 h；仅音频，无转写（要转写需另行伪标注）。
- 许可：卡片 `cc-by-nc-4.0`，同时带有版权免责声明，称各视频版权归原所有者，依 fair use 使用。
- 大小：224 GB（HF-API）。

#### mesolitica Malaysian-Emilia
- 链接：https://huggingface.co/datasets/mesolitica/Malaysian-Emilia
- 内容：基于 Emilia 流程处理的马来西亚/新加坡 YouTube、播客、国会、卡通语音，用于 TTS；转写为自动生成。例：YouTube 3168.8 h、播客 622.8 h、卡通 774.5 h。
- 许可：cc-by-nc-4.0（非商业，需标注）。
- 大小：4.15 TB（HF-API）。
- 质量问题：派生自 Emilia 流程，与另一份报告负责的 Emilia/YODAS 可能重叠。

#### IMDA National Speech Corpus（mesolitica 镜像）
- 链接：https://huggingface.co/datasets/mesolitica/IMDA-STT ；官网 https://www.imda.gov.sg/how-we-can-help/national-speech-corpus（页面正文未能取到）
- 内容：6 个部分，时长分别为 1117.99、1052.18、2162.50、2133.69、2044.52、2148.28 h（卡片）；卡片语言标 en。
- 许可：卡片称遵循 Singapore Open Data Licence。
- 转写：人工，以新加坡英语为主；马来语内容未核实。
- 大小：163 GB（HF-API，mp3 7z 压缩）。
- 判断：对马来语帮助有限，建议先确认 NSC 各部分语言构成。

#### 其他
- malaysia-ai/malay-conversational-speech-corpus-whisper-format（0.45 GB）、scripted-malay-daily-use-speech-corpus-whisper-format（0.91 GB）：没有 README，来源、时长、许可全部未核实。
- clt013/malay-speech-3k-rows-dataset（cc-by-nc-nd-4.0）、Speech-data/Malay-Speech-Dataset（cc-by-nc-nd-4.0）：数据商样本，体量小。
- BabelSpeech 50 h Malay：门控人工审批，数据商样本。

### 泰语

#### Porjai Thai Voice Dataset（朱拉隆功大学 / CMKL）
- 链接：https://huggingface.co/datasets/CMKL/Porjai-Thai-voice-dataset-central （方言：-korat、-khummuang、-pattani）
- 时长：卡片称中部泰语 700 h，三种方言各 40 h；HF 仓库中 central 有 335,674 条，实际小时数未核实。
- 许可：central 与 korat 为 CC-BY-SA-4.0；khummuang、pattani 为 CC-BY-NC-SA-4.0（HF 标签）。
- 获取：HF 直下。
- 转写：人工；泰语通常无空格标点，是否有标点未核实（字段有 `sentence` 与 `utterance` 两列）。
- 领域：Wang Data Market 采集的朗读语音，部分测试集已划给 ML-SUPERB 未公开。
- 大小：central 7.48 GB（DS-size）；korat 0.51 GB（DS-size）。
- 注意：700 h 对应 7.5 GB，压缩率很高，建议抽样确认采样率与时长。

#### processed-voice-th-169k
- 链接：https://huggingface.co/datasets/Porameht/processed-voice-th-169k
- 内容：169,550 条（149,953 / 7,614 / 11,983），卡片称是"cleaned open-source Thai speech dataset"；VIZINTZOR/Thai-Speech-Dataset 把它列为主数据集，三部分合计约 165 h（50+60+55，TTS 用途，推算）。
- 许可：CC-BY-SA-4.0 标签，但原始来源未写明，需标注。
- 大小：12.0 GB（卡片）。
- 质量问题：来源与转写方式不明，疑似含 TTS 合成音频，使用前必须抽样试听或与来源核对。

#### Gowajee
- 链接：https://github.com/ekapolc/gowajee_corpus ；SEACrowd 镜像 https://huggingface.co/datasets/SEACrowd/gowajee
- 时长：约 17 h，20,308 条，188 名说话人（163 男 25 女）。
- 许可：MIT。
- 获取：Google Drive（v0.9.3）；HF 镜像 huangjianbyte2024/thai_gowajee_corpus 无许可声明。
- 转写：人工，形式未核实。
- 领域：智能家居指令，"Gowajee" 唤醒词为中心，另有青少年俚语子集。内容单一、句子重复。
- 大小：原版未核实；HF 镜像 39 GB（HF-API）。

#### Thai Elderly Speech（Data Wow + VISAI）
- 链接：https://github.com/VISAI-DATAWOW/Thai-Elderly-Speech-dataset/releases/tag/v1.0.0 ；SEACrowd 卡片 https://huggingface.co/datasets/SEACrowd/thai_elderly_speech
- 时长：19,200 个文件，17 h 11 min，5,156 句独立句子，32 名老年说话人（10 男 22 女）。
- 许可：CC-BY-SA-4.0（SEACrowd 卡片转述，GitHub 原页面我没能取到正文，标未核实）。
- 领域：医疗与智能家居指令；句子重复多。
- 大小：未核实；Phonsiri 镜像 9.4 GB（HF-API，该镜像的转写表含 Typhoon 模型识别结果，不能直接当人工转写）。

#### LOTUSDIS / LOTUS-TRD（NECTEC）
- 链接：https://arxiv.org/abs/2509.18722 （LOTUSDIS）
- LOTUSDIS：114 h 自发会议对话，训练 88 h，开发 12.8 h，评测 13.3 h，CC-BY-SA 4.0，远场、重叠语音多（论文）。论文页面没有给出下载位置，转写形式未核实。
- LOTUS-TRD：约 180 h 四种泰语方言，CC-BY-SA 4.0 为基础的开放许可（搜索摘要，未在一手页面核实）；`nectec/LOTUS-TRD` 页面无法访问（401）。
- 判断：远场会议语音与我们的近讲训练分布差距大，优先级低。

#### 其他
- shunyalabs/thai-speech-dataset：4,062 条，3.06 GB，无许可声明，来源未核实。
- hotdogs/thai-speech-20k：衍生自 GigaSpeech 2 的单说话人筛选，仅元数据，无音频，不使用。
- GigaSpeech 2 泰语（另一份报告）：搜索摘要称精炼版约 10,000 h，Apache-2.0，自动转写。

### 越南语

#### VIVOS
- 链接：https://huggingface.co/datasets/AILAB-VNUHCM/vivos ；https://doi.org/10.5281/zenodo.7068130
- 时长：约 15 h；train 11,660 条，test 760 条。
- 许可：卡片 CC BY-NC-SA 4.0（HF 标签与 Licensing 部分一致）。
- 获取：HF 直下。
- 转写：人工朗读稿；无标点（传统 VIVOS 文本为大写无标点，卡片未明说，标未核实）。
- 领域：朗读，约 46 名说话人（未核实），录音棚级。
- 大小：1.48 GB（卡片 download_size）。
- 注意：卡片标明为研究用途；SEACrowd/vivos 镜像标签是 CC-BY-SA-4.0，与原卡片不同，以原卡片为准。

#### Bud500（VietAI）
- 链接：https://huggingface.co/datasets/linhtran92/viet_bud500
- 时长：约 500 h；train 634,158 条，validation 7,500，test 7,500。
- 许可：HF 标签 CC-BY-NC-SA-4.0，卡片版权段写 Apache License 2.0（VietAI Research），两处冲突，需按更严格的非商业处理。
- 获取：HF 门控（自动审批，需登录并同意分享联系信息）；我没有登录，未查看内容。
- 转写：人工，示例为小写无标点；domain 为播客、旅行、书籍、美食等混合，南/中/北三区口音。
- 大小：59.5 GB（HF-API）；卡片写 98 GB，文件列表约 52.7 GB，三者不一致，以实际下载为准。
- 质量问题：小写无标点，需要标点恢复才能作为目标文本。

#### VietSpeech
- 链接：https://huggingface.co/datasets/NhutP/VietSpeech
- 时长：1100+ h，约 1,026,047 条，16 kHz；来自社交媒体，多口音。
- 许可：Apache-2.0（HF 标签）。
- 获取：HF 门控（自动）。
- 转写：卡片没有说明人工或伪标注，未核实；规模和来源提示很可能是伪标注，用前必须核实。
- 大小：130.5 GB（HF-API）。

#### VietMed
- 链接：https://huggingface.co/datasets/leduckhai/VietMed ；论文 https://arxiv.org/abs/2404.05659
- 时长：有标注 16 h，另有 1000 h 医疗领域无标注与 1200 h 通用领域无标注（后两者在 Google Drive，无转写）。
- 许可：MIT（HF 标签）。
- 获取：HF 直下。
- 转写：人工；2026 年 10 月修复过音频-转写错位。是否有标点未核实。
- 大小：0.18 GB（卡片 download_size）；仓库整体 10.5 GB（HF-API，含其他文件）。
- 注意：医疗领域，体量小。

#### ViMD（多方言）
- 链接：https://huggingface.co/datasets/nguyendv02/ViMD_Dataset ；论文 https://aclanthology.org/2024.emnlp-main.426
- 时长：102.56 h，约 19,000 条，63 个省份方言，来自各省电视台新闻节目。
- 许可：CC-BY-NC-ND-4.0。ND（禁止衍生）条款对训练模型是否构成衍生没有定论，需标注并谨慎使用；论文称"available for research purposes"。
- 获取：HF 直下。
- 转写：人工；标点形式未核实。
- 大小：仓库 74.2 GB（HF-API）；datasets-server 报 parquet 14.4 GB（只覆盖部分行）。
- 与 FLEURS：新闻语音，建议去重。

#### PhoAudiobook
- 链接：https://huggingface.co/datasets/thivux/phoaudiobook ；论文 https://arxiv.org/abs/2506.01322
- 时长：论文称从 2,697 本有声书 23K 小时中筛出 941 h，735 名说话人；HF 称训练 940 h 加 554 h 增强短音频合计 1,494 h。
- 许可：HF 无许可标签；条款禁止分发数据集及其任何部分，使用时需引用 ACL 2025 论文（搜索摘要转述，未核实完整条款）。
- 获取：HF 门控（自动，需分享联系信息）。
- 转写：为零样本 TTS 构建，转写来源未核实。
- 大小：167 GB（HF-API）。
- 质量问题：有声书版权状态不明；禁止分发条款与我们只发布脚本的做法兼容，但仍需阅读完整条款。

#### VietSuperSpeech 与 Gemini 转写集
- https://huggingface.co/datasets/thanhnew2001/VietSuperSpeech ：103.18 h，32,267 条；来源是 YouTube 新闻、个人频道，转写由 Zipformer-30M-RNNT-6000h 生成；卡片写 MIT，但音频版权归原频道，许可存疑；64.3 GB（HF-API）。
- https://huggingface.co/datasets/nmcuong/vietnamese-youtube-audio-transcribed-gemini-3.5-transcribe ：YouTube 音频，由 Gemini 转写，license other，人工审批门控，41.2 GB（HF-API），时长未核实。两者均为伪标注，版权风险高。

#### 排除
- Dolly-Audio（dolly-vn/dolly-audio-1000h-vietnamese）：卡片标签为 synthetic、tts，约 1000 h、664,125 条、152 名说话人，CC-BY-NC-SA-4.0，需机构邮箱审批。属于合成语音，不符合"真人语音"要求，157.8 GB（卡片）。
- VLSP 评测数据：本次没有找到公开直接下载页，需要向组织方申请，获取方式未核实。

### 菲律宾语（他加禄语）

#### Filipino Speech Corpus (FSC, Sagum)
- 链接：https://huggingface.co/datasets/sapinsapin/filipinospeechcorpus
- 时长：总 65.1 h，313,322 个原始片段（发布 305,246 条：train 274,730，test 30,516），125 名说话人，16 kHz。
- 重要限制：56% 是机器切出的单词片段（中位 0.62 s）。卡片建议过滤后得到约 8,500 条、约 7 h 的句子级朗读/自发语音。
- 许可：HF 标签 MIT；这是第三方对原始 FSC（菲律宾大学 Sagum）的重打包，原始语料的授权未核实，需标注。
- 获取：HF 直下。
- 转写：Transcriber 软件人工/机器转写，两部分转写规范不一致；标点大小写未核实。
- 领域：录音棚朗读与自发语音，97.5% 说话人在 20–27 岁。
- 大小：6.9 GB（卡片）。

#### Silencio Tagalog/Filipino 样本
- 链接：https://huggingface.co/datasets/SilencioNetwork/tagalog-filipino-speech
- 时长：2.03 h，90 条，16 名说话人，48 kHz 立体声；自发语音，人工转写并带词级时间戳。
- 许可：CC-BY-NC-4.0。
- 说明：卡片自称是商业目录的样本，完整数据需另行授权。大小 1.0 GB（HF-API）。体量太小，只可作评估参考。

#### InfoBayAI 呼叫中心音频
- 链接：https://huggingface.co/datasets/InfoBayAI/Filipino-Tagalog-Call-Center-Audio-Dataset-Single-Channel
- 内容：搜索摘要称已处理的 Filipino 169 h、Tagalog 477 h；HF 标签 cc-by-4.0；门控自动审批，需分享联系信息。我没有登录，时长与转写形式均未在一手页面确认。数据商来源的数据，授权可信度需要核实。
- 另有 KarlShane11/handsfree-filipino-speech-combined（cc-by-4.0，门控，卡片无法读取，3.35 GB），内容未核实。

#### IARPA Babel Tagalog (LDC2016S13) 与相关
- 链接：https://catalog.ldc.upenn.edu/LDC2016S13
- 内容：约 213 h 北部、中部、南部方言的电话对话与朗读语音，含转写，8 kHz 电话语音。
- 获取：LDC 付费或会员许可。大小未核实。
- 同类：MATERIAL Tagalog-English（LDC2026S05）约 52 h 语音，LDC 许可；DataOcean 423.5 h、Defined.ai 224 h 为商业销售，不可用。
- Philippine Languages Database（PLD，ACL SIGUL 2024）：10 种菲律宾语言共 454+ h，获取方式与许可未核实。

#### 其他
- PiotrSty/yodas3-transcripts-filipino：仅含 YODAS3 菲律宾语字幕文本（6,565 条，CC-BY-3.0 声明，作者归属未解决），无音频。YODAS 本身由另一份报告负责；字幕为 YouTube 自动或用户字幕，质量不稳定。

## 四、阻碍使用的问题汇总

1. 印尼语和菲律宾语没有找到开放的大规模人工转写真人语料；只能依赖另一份报告负责的多语言大库（GigaSpeech 2、YODAS、Common Voice 等）补量。
2. 马来语数据基本是伪标注，且 mesolitica 系列大多没有许可声明，音频来自 YouTube，版权状态不明。
3. 越南语的 Bud500 许可冲突、门控且转写小写无标点；ViMD 带 ND 条款；PhoAudiobook 条款禁止分发。
4. 泰语的 Porjai 中部语料转写预计无标点；processed-voice-th-169k 来源不明。
5. 多数人工转写集（VIVOS、Bud500、泰语集）无标点或标点未核实，作为"带标点大小写的书面文本"目标需要标点恢复，或只作为低权重补充。
6. 门控数据集（Bud500、VietSpeech、PhoAudiobook、InfoBayAI）我没有登录，没有同意协议，条款以公开卡片为准。

## 五、主要来源

- HF API 元数据：`https://huggingface.co/api/datasets/<id>`（本报告所有 HF-API 大小、门控、许可标签）
- datasets-server：`https://datasets-server.huggingface.co/size?dataset=<id>`（Porjai、ViMD）
- VIVOS：https://huggingface.co/datasets/AILAB-VNUHCM/vivos
- Bud500：https://huggingface.co/datasets/linhtran92/viet_bud500
- VietMed：https://huggingface.co/datasets/leduckhai/VietMed
- ViMD：https://huggingface.co/datasets/nguyendv02/ViMD_Dataset
- Gowajee：https://github.com/ekapolc/gowajee_corpus
- ID2：https://zenodo.org/records/15946165
- LOTUSDIS：https://arxiv.org/abs/2509.18722
- FSC：https://huggingface.co/datasets/sapinsapin/filipinospeechcorpus
- mesolitica 系列：https://huggingface.co/mesolitica
- Babel Tagalog：https://catalog.ldc.upenn.edu/LDC2016S13
- 调查时读取的数据集卡片未随本文保存。
