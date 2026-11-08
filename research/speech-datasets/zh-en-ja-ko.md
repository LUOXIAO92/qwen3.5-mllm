# 中英日韩单语种真人语音语料调研

调查日期：2026-11-08。只做网页检索与 Hugging Face 元数据读取，没有下载数据、没有登录、没有接受任何协议。
「未核实」表示没有在一手来源中确认。体积列写明来源：OpenSLR 页面（OSLR）、HF 数据集页面或 tree API（HF）、论文/官网、估算。
估算假设：16 kHz 16 bit 单声道 WAV 约 0.115 GB/小时；FLAC 约为其一半；压缩格式更小。

范围说明：本报告不含 YODAS2、Emilia、Granary、Common Voice、MLS、VoxPopuli、People's Speech、GigaSpeech 2、MediaSpeech、FLEURS、Multilingual TEDx。
FLEURS/FLoRes 关系一栏：下列语料均未在一手来源中看到与 FLEURS/FLoRes 的声明性关联，一律写「未核实，未见声明」。其中 Wikipedia 或网页文本为朗读稿的语料（JSUT、Zeroth、Deeply Korean 等）理论上可能与 FLoRes 句子重叠，需在训练前做文本去重。

## 一、汇总表

### 中文（普通话）

| 数据集 | 小时 | 许可 | 获取 | 转写形式 | 体积 |
|---|---|---|---|---|---|
| WenetSpeech | 10005 h 高质量（另 2478 h 弱标签，约 9952 h 无标签）；YouTube + 播客 | 论文称 CC BY 4.0 且限非商用；官网需阅读许可 | 填 Google 表单申请密码 | OCR 字幕/ASR 标签，置信度筛选；标点未核实 | 未核实（估算：L 约 1.15 TB WAV，FLAC/opus 更小） |
| AISHELL-1 | 约 170 h（未在 OSLR 页面核实小时数） | Apache 2.0（页面另写“free for academic use”） | OpenSLR 直接下载 | 人工转写，无标点，朗读 | 15 GB（OSLR） |
| AISHELL-3 | 85 h，88035 句，218 人 | Apache 2.0 | OpenSLR 直接下载 | 汉字+拼音，人工，朗读（TTS 语料） | 19 GB（OSLR） |
| AISHELL-4 | 约 120 h，会议 | CC BY-SA 4.0 | OpenSLR 直接下载 | 人工，会议转写，标点未核实 | 约 51 GB（OSLR） |
| AliMeeting | 118.75 h，会议 | CC BY-SA 4.0 | OpenSLR 直接下载 | 人工，标点未核实 | 约 108 GB（OSLR 求和，含 8 通道） |
| AISHELL-2 | 约 1000 h，1991 人，iOS 通道 | 仅限非商用研究教育 | 填申请表，签协议 | 人工，无标点（未核实） | 未核实（估算约 115 GB WAV） |
| MAGICDATA 朗读 | 755 h，1080 人 | CC BY-NC-ND 4.0 | OpenSLR 直接下载 | 人工，标点未核实 | 约 55 GB（OSLR：52+1.0+2.2） |
| MagicData-RAMC | 180 h 对话，663 人 | CC BY-NC-ND 4.0 | OpenSLR 直接下载 | 人工，口语对话，标点未核实 | 15 GB（OSLR） |
| KeSpeech | 1542 h，27237 人，含 8 种方言 | 未核实（二手来源 CC BY-NC-SA，NeurIPS 称学术免费） | 需向作者申请（未核实）；HF 镜像仅为测试子集 | 人工，标点未核实 | HF 镜像 3.45 GB 仅测试集；全集未核实 |
| CS-Dialogue | 104 h，中英混合，200 人 | 论文称将对学术免费，未给许可名 | 论文未给下载链接 | 人工，有中英文标点，数字写成汉字/英文单词 | 估算约 12 GB WAV（104 h × 0.115） |
| Free ST-CMDS | 102600 句，855 人（小时数 OSLR 未列） | CC BY-NC-ND 4.0 | OpenSLR 直接下载 | 人工校对，短语音，无标点 | 8.2 GB（OSLR） |
| Primewords Set 1 | 100 h，296 人 | CC BY-NC-ND 4.0 | OpenSLR 直接下载 | 人工，标点未核实 | 9.0 GB（OSLR） |
| THCHS-30 | 约 30 h（页面未列） | Apache 2.0 | OpenSLR 直接下载 | 朗读，无标点 | 6.4 GB（OSLR） |
| aidatatang_200zh | 200 h，600 人 | CC BY-NC-ND 4.0，OpenSLR 注明应数据方要求撤回 | 不建议使用 | 人工 | 未核实 |
| WenetSpeech4TTS | 12800 h（Premium 945 / Standard 4056 / Basic 7226 / Rest 5574） | CC BY 4.0，限非商用研究教育 | HF 门控 | Paraformer-large 重新转写（伪标签），标点未核实 | 972 GB（HF 页面）；HF tree 合计约 972 GB |

### 英文

| 数据集 | 小时 | 许可 | 获取 | 转写形式 | 体积 |
|---|---|---|---|---|---|
| LibriSpeech | 约 1000 h | CC BY 4.0 | 直接下载 / HF 公开 | 大写，无标点，朗读 | train 共 59.3 GB tar.gz（OSLR）；HF parquet 123.7 GB（含 dev/test 全部配置） |
| LibriSpeech-PC | 同 LibriSpeech（少量句子被丢弃） | CC BY 4.0 | 直接下载，仅清单，音频另取 | 已恢复标点和大小写 | 清单约 25 MB（OSLR） |
| Libriheavy | small 509 h / medium 5042 h / large 50794 h | Apache-2.0（清单） | 清单公开（HF 无门控）；音频取自 Libri-Light | cases_and_punc 有标点和大小写；upper_no_punc 另备 | 清单 15.9 GB（HF tree）；音频：Libri-Light small 35 GB、medium 321 GB、large 3.05 TB（GitHub） |
| GigaSpeech | XS10 / S250 / M1000 / L2500 / XL10000 h，dev 12 h，test 40 h | HF 标签 apache-2.0，但仅限非商用研究教育，版权不属于 SpeechColab | 填 Google 表单；HF 门控 | 大写，标点为 `<COMMA>` 等标签，需转换 | 总计 2.6 TB（HF 卡片），分档大小未核实 |
| SPGISpeech | 约 5000 h，约 5 万说话人，财报电话会 | other；仅限学术研究和内部使用，禁止再分发 | HF 门控，需同意条款 | 大小写、标点、反规范化都有，多编辑校对 | S 22 GB / M 107 GB / L 530 GB / dev 11 / test 11；总计 781 GB（HF 页面） |
| TED-LIUM 3 | 452 h | CC BY-NC-ND 3.0 | OpenSLR 公开 | 自动对齐的 STM，大小写与标点未核实 | 下载约 50.6 GiB（TFDS 目录，二手） |
| Earnings-21 / 22 | 39 h / 119 h | 未核实（GitHub 页面未写） | Git LFS 公开 | 有标点（Earnings-21），其余未核实 | 未核实 |
| AMI | 约 100 h，会议，多为非母语者 | cc-by-4.0（HF 标签） | HF 公开 | 大写，几乎无标点（Kaldi 规范化） | 29 GB（HF，含 ihm 与 sdm） |
| Loquacious Set | small 250 / medium 2500 / large 25000 / clean 13000 h | CC BY 3.0 / 4.0 / CC0（按来源） | HF 公开 | 大写，无标点，数字拼写，已规范化 | 整个数据集 3.13 TB（datasets-server）；clean 0.90 TB |

### 日文

| 数据集 | 小时 | 许可 | 获取 | 转写形式 | 体积 |
|---|---|---|---|---|---|
| ReazonSpeech v2 | tiny 8.5 / small 100 / medium 1000 / large 5000 / all 约 35000 h | CDLA-Sharing-1.0；限日本著作权法第 30 条之 4 的目的 | HF 门控，需同意条款并登录 | 电视字幕经 CTC 对齐，自举筛选，有标点（示例带「。」），未经全量人工核对 | tiny 600 MB / small 6 GB / medium 65 GB / large 330 GB / all 2.3 TB（HF 卡片） |
| J-CHAT | 未核实（论文称约 6.9 万小时，此处未核实） | CC-BY-NC 4.0，限第 30 条之 4，禁商用 | HF 门控 | reazonspeech-nemo-v2 伪标签 | 未核实 |
| JSUT | 10 h，单一女声 | 文本 CC-BY-SA 4.0 等；音频限学术/非商用研究，不得再分发 | Google Drive 直接下载 | 人工朗读稿，有标点 | 2.7 GB（官网，48 kHz） |
| JVS | 30 h，100 人 | 标签 CC-BY-SA 4.0；音频限学术/非商用 | Google Drive 直接下载 | 文本取自 JSUT | 3.5 GB（官网） |
| CSJ | 未核实 | 个案审批，付费 USB 版 | 付费或申请 | 未核实 | 未核实 |
| 动漫/社区 HF 集 | 约 100–400 h | 多为版权存疑 | 公开 | 来源不明 | 不建议 |

### 韩文

| 数据集 | 小时 | 许可 | 获取 | 转写形式 | 体积 |
|---|---|---|---|---|---|
| Zeroth-Korean | 训练 51.6 h + 测试 1.2 h，22263+457 句，105+10 人 | CC BY 4.0 | OpenSLR / HF 公开 | 人工转写朗读，标点未核实 | HF 2.88 GB；OSLR tar 10 GB |
| KsponSpeech（AI Hub「한국어 음성」） | 约 1000 h，约 2000 人 | 页面未写许可，仅链接使用政策 | AI Hub 申请，页面写明「只有内国人可申请」 | 拼写转写（含填充词标记），EUC-KR，无需标点 | 123 GB（AI Hub） |
| Pansori-TEDxKR | 约 3 h，41 人 | CC BY-NC-ND 4.0 | OpenSLR 直接下载 | TEDx 社区字幕 | 174 MB（OSLR） |
| Deeply Korean read speech | 公开样本约 3 h（全集约 290 h） | CC BY-NC-ND 4.0 | OpenSLR 直接下载（仅样本） | 朗读 | 样本 281 MB（OSLR） |
| ClovaCall | 约 60 h（二手，说法不一） | 仅限非商用研究，要求署名 NAVER | GitHub，申请方式未核实 | 人工，客服通话 | 未核实 |
| KeSpeech 等 | 见中文 | | | | |

## 二、各数据集详述

### 中文

**WenetSpeech**（https://github.com/wenet-e2e/WenetSpeech ，https://www.openslr.org/121/ ，论文 arXiv:2110.03370）
- 语言与时长：普通话。L 子集 10005 h（置信度 0.95–1.0），M 1000 h，S 100 h；弱标签 2478 h；无标签约 9952 h；合计 22435 h。评测集 DEV 20 h、TEST_NET 23 h、TEST_MEETING 15 h。话语数未核实。
- 许可：OpenSLR 与论文写 CC BY 4.0，论文称“licensed for non-commercial usage under CC-BY 4.0”；GitHub 仓库标 Apache-2.0 但没有许可文本。音频版权归 YouTube/播客原作者。
- 获取：先到官网阅读许可，填表申请 PASSWORD，再用官方脚本从腾讯会议云或 ModelScope 下载。属于申请制。
- 转写：YouTube 部分来自视频字幕 OCR，播客部分来自 ASR，之后用端到端错误检测筛选。人工检查的是三个评测集。标点形式、数字规范化方式在上述页面中没有说明，标「未核实」，不应预设有标点。
- 领域：YouTube 6113 h + 播客 3892 h（高标签部分），说话人数未核实。
- 已知问题：2026-01-09 公告称 TEST_NET 有少量标签错误，见 issue #63。
- 体积：官方页面未给出。估算 10005 h 的 16 kHz WAV 约 1.15 TB，实际压缩格式更小。
- 与 FLEURS 关系：未见声明。

**WenetSpeech4TTS**（https://huggingface.co/datasets/Wenetspeech4TTS/WenetSpeech4TTS ，arXiv:2406.05763）
- 12800 h，由 WenetSpeech 重新切分并增强音质，用 Paraformer-large 重新转写，所以是 ASR 伪标签，带词级时间戳。
- 许可 CC BY 4.0，卡片写明限非商用研究教育，HF 门控。总体积 972 GB。标点未核实。适合在 WenetSpeech 之外扩充，但转写由模型产生。

**AISHELL-1**（https://www.openslr.org/33/）：约 400 人，安静室内高保真麦克风，16 kHz；人工转写，准确率“above 95%”；Apache License v.2.0，并写明“free for academic use”；data_aishell.tgz 15 GB。朗读文本、无标点，领域偏财经新闻类（未在页面核实）。小时数页面未列（常见说法 170 h，未核实）。HF 镜像 AISHELL/AISHELL-1，标签 apache-2.0，无门控。

**AISHELL-3**（https://www.openslr.org/93/）：85 h，88035 句，218 人，汉字与拼音两级标注，准确率 98% 以上；Apache License v.2.0；19 GB。TTS 语料，朗读为主，无标点。

**AISHELL-4**（https://www.openslr.org/111/）：约 120 h，211 场会议（每场 4–8 人），8 通道麦克风阵列，含重叠语音与噪声；CC BY-SA 4.0；约 51 GB（HF tree 显示 50.2 GB）。多人重叠，直接用于单说话人训练需要选近场或裁剪。

**AliMeeting**（https://www.openslr.org/119/）：118.75 h（训练 104.75 h），240 场会议；CC BY-SA 4.0；远场 8 通道约 108 GB 全套，近场耳麦训练集 22.85 GB。训练 projector 推荐使用近场耳麦部分。

**AISHELL-2**（论文 arXiv:1808.10583；申请页 https://www.aishelltech.com/aishell_2 未能读取）：约 1000 h，1991 人，iOS 通道为免费学术版，Android、Mic 为开发/测试。仅限非商用研究教育，需填申请表并签协议，硬盘或云端交付。体积未核实。官方页面这次没能打开，申请细节来自镜像 README 与目录站，需到官网复核。

**MAGICDATA 朗读语料**（https://www.openslr.org/68/）：755 h，1080 人，手机录制，安静室内，句级准确率 98% 以上，场景含问答、音乐搜索、SNS 消息、命令控制；CC BY-NC-ND 4.0；train 52 GB。短句为主，页面未提标点。

**MagicData-RAMC**（https://www.openslr.org/123/）：180 h 自发对话，663 人，手机录制；CC BY-NC-ND 4.0；15 GB；人工标注并经质检。对话口语，适合补充口语风格。

**KeSpeech**（NeurIPS 2021 数据集赛道论文；HF 镜像 TwinkStart/KeSpeech）：1542 h，27237 人，34 个城市，含普通话和 8 种方言。NeurIPS 页面称学术用途免费；二手来源 CodecBench 写 CC BY-NC-SA；VoxKitchen 写“见来源条款”且手动获取。许可未能从官方页面核实。HF 镜像为 3.45 GB、19723 行的测试子集，不是全集。GitHub 官方仓库本次返回 404，未核实。

**CS-Dialogue**（arXiv:2502.18913）：104.02 h，38917 句，200 人，中英混说的自发对话；16 kHz 16 bit 单声道 WAV；转写带中英文标点，数字写成汉字或英文单词，非语音事件用 `<FIL/>` 等标签，TextGrid 格式；论文称将免费用于学术，未给许可名和下载链接。是少数带标点的人工转写，但获取方式未核实。

**Free ST-CMDS**（https://www.openslr.org/38/）：102600 句，855 人，每人约 120 句，手机安静室内录制；CC BY-NC-ND 4.0；8.2 GB。句子很短，多为指令类。

**Primewords Set 1**（https://www.openslr.org/47/）：100 h，296 人，智能手机录制，准确率 98% 以上（95% 置信度）；CC BY-NC-ND 4.0；9.0 GB。

**THCHS-30**（https://www.openslr.org/18/）：Apache License v.2.0；data_thchs30.tgz 6.4 GB，小时数、人数页面未列。朗读新闻，无标点。

**aidatatang_200zh**（https://openslr.org/62 ，二手检索结果）：200 h，600 人，CC BY-NC-ND 4.0，OpenSLR 页面注明应数据所有方要求撤回。本次直接抓取该页返回“Resource not found”，与撤回说法一致。不建议使用。

### 英文

**LibriSpeech**（https://www.openslr.org/12/ ，HF openslr/librispeech_asr）：约 1000 h 有声书朗读；CC BY 4.0。train-clean-100 6.3 G、train-clean-360 23 G、train-other-500 30 G、dev/test 各约 0.3 GB（OSLR）。HF 版本为 parquet，全部配置共 123.7 GB（datasets-server，clean 30.35 GB、other 31.49 GB，“all” 61.84 GB 为 train+dev+test 并行配置）。转写全大写、无标点。

**LibriSpeech-PC**（https://www.openslr.org/145/）：为 LibriSpeech 恢复标点和大小写，只提供 JSON 清单（约 25 MB），音频需另取自 LibriSpeech；CC BY 4.0；部分样本在恢复时被丢弃。这是获得有标点英文真人语音成本最低的办法。

**Libriheavy**（https://github.com/k2-fsa/libriheavy ，https://arxiv.org/abs/2309.08105）：由 Libri-Light 标注而来，small 509 h、medium 5042 h、large 50794 h，评测 dev 22.3 h 等。清单 Apache-2.0。提供 cases_and_punc（原书文本，有大小写和标点）与 upper_no_punc 两种。音频与 Libri-Light 相同：small 577 h/35 GB，medium 5193 h/321 GB，large 51934 h/3.05 TB（Libri-Light 数据准备文档）。Libri-Light 页面未写音频许可，LibriVox 录音为公有领域是常识，但此处未从一手来源核实。有声书朗读，说话人多，域较单一。HF pkufool/libriheavy 清单合计 15.9 GB。

**GigaSpeech**（https://github.com/SpeechColab/GigaSpeech ，HF speechcolab/gigaspeech）：10000 h 转写（XL），有声书 2655 h、播客 3498 h、YouTube 3845 h。HF 标签 apache-2.0，卡片又写版权不属于 SpeechColab、只能非商用研究教育。填 Google 表单后获取；HF 门控。大写，标点为 `<COMMA> <PERIOD> <QUESTIONMARK> <EXCLAMATIONPOINT>` 标签，需转换成符号并恢复大小写。有 `<SIL> <MUSIC> <NOISE> <OTHER>` 垃圾标签需丢弃。总体积 2.6 TB。

**SPGISpeech**（https://huggingface.co/datasets/kensho/spgispeech）：约 5000 h，约 5 万说话人，S&P Global 财报电话会 2007–2020，切成 5–15 秒。转写带大小写、标点和反规范化，多位专业编辑校对，是人工转写里形式最接近“书面文本”的一个。许可 other：限学术研究和内部使用，禁止未经书面同意再分发；HF 门控，Kensho 可随时终止访问。体积 781 GB（S 22 / M 107 / L 530 / dev 11 / test 11 GB）。域单一（财经、男性占约 90%）。

**TED-LIUM 3**（https://lium.univ-lemans.fr/?p=22550 ，HF LIUM/tedlium 需登录，抓取失败）：452 h；CC BY-NC-ND 3.0（检索结果）；STM 自动对齐转写。大小写和标点未核实（常见说法为小写无标点，未核实）。下载约 50.59 GiB（TFDS 目录，二手）。OpenSLR/51 页面本次返回“Resource not found”。

**Earnings-21 / Earnings-22**（https://github.com/revdotcom/speech-datasets）：39 h/44 个文件；119 h/125 个文件；Earnings-21 转写含标点和实体标签；许可、体积页面未写。规模小，适合作为补充或评测，不适合当主力。

**AMI**（HF edinburghcstr/ami）：约 100 h 会议，多为非母语说话人；标签 cc-by-4.0，许可节为空；Kaldi 规范化，大写、几乎无标点；29 GB。

**Loquacious Set**（https://huggingface.co/datasets/speechbrain/LoquaciousSet ，arXiv:2505.21578）：25000 h，由 VoxPopuli、Libriheavy、LibriSpeech、YODAS、People's Speech、Common Voice 组成，许可按来源 CC0/CC BY。转写已规范化为大写 26 字母加撇号，数字拼写，无标点。其中不少来源在另一位调研者范围内，且转写形式不符合本项目优先要求，仅作规模参考。HF 公开、无门控，体积 3.13 TB。

### 日文

**ReazonSpeech**（https://huggingface.co/datasets/reazon-research/reazonspeech ，https://research.reazon.jp/projects/ReazonSpeech/）：地面电视广播，v2 约 35000 h，16 kHz FLAC。许可 CDLA-Sharing-1.0，并要求“SOLELY FOR THE PURPOSE OF JAPANESE COPYRIGHT ACT ARTICLE 30-4”（卡片原文）。法律依据是日本著作权法第 30 条之 4（允许为信息分析而利用作品）。HF 门控，需登录同意条款。转写来源：ANLP 2023 论文称先用 CTC 对齐把电视字幕与音频匹配，再迭代自举筛选，所以是字幕对齐加模型筛选的结果，没有逐条人工核对。示例文本带「。」，说话人数未核实。小子集 small（100 h，6 GB）适合先做小规模实验。

**J-CHAT**（https://huggingface.co/datasets/sarulab-speech/J-CHAT ，arXiv:2407.15828）：YouTube 与播客日语对话。CC-BY-NC 4.0，禁商用，限第 30 条之 4，HF 门控。转写由 reazonspeech-nemo-v2 生成（伪标签，其错误会被学到）。卡片没有写小时数和体积，论文中的小时数未在此核实。

**JSUT**（https://sites.google.com/site/shinnosuketakamichi/publication/jsut ）：10 h，单一女声，48 kHz，消声室；音频限学术、非商用研究（含公司内）和个人使用，不得再分发，商用需联系；文本 CC-BY-SA 4.0 等。2.7 GB。说话人单一，只适合作少量补充。

**JVS**（https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus ）：30 h，100 名专业说话人，文本取自 JSUT；音频条款与 JSUT 相同；3.5 GB。

**CSJ**（https://clrd.ninjal.ac.jp/csj/en/ ）：国立国语研究所自发日语语料；可经 Chunagon 在线检索，或购买付费 USB。商用个案审批。小时数、转写形式、价格页面未列，未核实。

**其他 HF 日语集**：检索只看到动漫/游戏语音（如 joujiboi/japanese-anime-speech-v2）和商业公司样例（SpeechDataAI、DataoceanAI 等）。动漫版权存疑，商业样例体积很小，本项目不建议。未发现 ReazonSpeech v3。

### 韩文

**Zeroth-Korean**（https://www.openslr.org/40/ ，HF kresnik/zeroth_korean 与 Bingsu/zeroth-korean）：训练 51.6 h（22263 句，105 人）、测试 1.2 h（457 句，10 人），16 kHz，CC BY 4.0，来自约 3000 个句子的朗读。HF 版本 2.88 GB parquet；OpenSLR tar 10 GB。句子重复度高（22263 句来自约 3000 个文本）。

**KsponSpeech / AI Hub「한국어 음성」**（https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=123 ）：约 1000 h，约 2000 人，安静环境，自发开放域对话；拼写转写（含填充词等标记），EUC-KR 编码，16 kHz 16 bit PCM 切句；123 GB。页面未写许可，仅链接使用政策；页面明确写“内国人만 데이터 신청이 가능합니다”（只有本国人可以申请数据）。对本项目是阻塞项。HF 上的 KsponSpeech 镜像来源和授权未核实，不建议使用。

**Pansori-TEDxKR**（https://www.openslr.org/58/）：约 3 h，41 人，TEDx 演讲，社区字幕；CC BY-NC-ND 4.0；174 MB。

**Deeply Korean read speech**（https://www.openslr.org/97/）：公开版为全集约 1% 的样本，约 3 h（全集约 290 h），CC BY-NC-ND 4.0，281 MB。仅一对说话人。

**ClovaCall**（https://github.com/clovaai/ClovaCall ）：客服通话；GitHub 称仅限非商用 AI 研究并要求署名 NAVER；时长说法不一（60 h 到 125 h），获取流程、体积未核实。

**HF 上的其他韩语集**：检索只看到 zeroth 镜像、电影片段（missycuo/korean-movies-asr，来源和授权未核实）和商业样例，不建议使用。

## 三、各语言结论

- 中文：最大的免申请路径有限。WenetSpeech 量最大（L 10005 h），申请制，标点未核实，标签为 OCR/ASR 衍生。干净、授权明确的人工转写有 MAGICDATA 朗读 755 h、MagicData-RAMC 180 h（对话）、AliMeeting 近场、AISHELL-1/3，但大多无标点，多数许可为 NC-ND。CS-Dialogue 带标点，但目前没有下载入口。
- 英文：想要带大小写与标点的人工转写，首选 Libriheavy cases_and_punc（有声书）和 LibriSpeech-PC，其次 SPGISpeech（财报，门控、限学术内部使用）。GigaSpeech 需要把标点标签转成符号。
- 日文：真人语料几乎只有 ReazonSpeech 一个量级足够的来源，门控，转写是电视字幕对齐的结果，用途限第 30 条之 4。其余都是 10–30 h 的 TTS 语料。
- 韩文：Zeroth 51.6 h 是仅有的开放可用大块；KsponSpeech 因“仅限本国人申请”不可用。韩文真人数据是最大的缺口，需要靠另一位调研者负责的多语种语料（如 Common Voice、YODAS、Emilia 等）补充。
