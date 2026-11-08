# 阿拉伯语、土耳其语、波斯语真人语音语料调查

调查日期：2026-11-08。方法：只读网页和 Hugging Face（HF）元数据接口（`api/datasets/<id>`、`tree`、`datasets-server.huggingface.co/size`），没有下载数据内容，没有登录，没有接受任何协议。下面每个数字都注明出处；查不到的写「未核实」。

## 先读这里

1. 三种语言的转写几乎都不带标点。TSC（土耳其语）有第三方论文明确说明「全小写、无标点」；波斯语的几个大型网络语料是字幕或 ASR 伪标注，标点情况基本未核实；阿拉伯语里只有 QASR 官方说明带标点，但它需要申请，且只限非商业。训练目标是带标点的书面文本，所以标点要么靠规范化前的原文保留，要么需要用另一个模型补标点，这一点要在选数据时就定下来。
2. 许可最干净、可直接下载、转写为人工产生的候选：阿拉伯语 MASC（CC BY 4.0，字幕转写）；土耳其语 TSC（CC BY 4.0 / HF 标 MIT，人工转写但无标点）；波斯语 Neyshekar（CC0，朗读，人工录音，含数字未展开）。
3. 与 FLEURS 的关系：Peacockery/farsi-asr-corpus-v4 直接混入了 FLEURS fa_ir（13 h），不能用。Bayanat-Labs/arabic-asr-channel-pairs 完全由 FLEURS ar_eg 衍生，不能用。其余被查到的语料没有已知的 FLEURS 重叠，但「未核实」不等于「没有」，开训前仍应对 FLEURS dev/test 句子做去重。
4. 已排除的合成语料（会混入合成语音，与本次要找真人语音的目的相反）：atasoglu/turkish-emotional-speech-100h（全合成）、turkmedstt/medv3-turkish-medical-asr（合成 TTS）、The0Sanity/sesmetin-v2-speech-corpus（Kokoro 合成）。

## 汇总表

「存储」栏的出处标记：[T]=HF `tree` 文件大小之和；[S]=datasets-server `size` 接口；[C]=数据集卡片；[P]=官方页面；[估]=按小时数估算，假设 16 kHz 16 bit 单声道 WAV 约 0.115 GB/小时，压缩格式更小。

### 阿拉伯语

| 语料 | 时长 | 许可 | 获取方式 | 转写形式 | 存储 |
|---|---|---|---|---|---|
| MASC（pain/MASC） | 声称 1000 h，有转写 907.4 h（训练 867 h，dev 20.8 h，test 19.4 h） | CC BY 4.0（官方 IEEE DataPort 与 HF 卡片一致） | HF 镜像无门槛；官方站需 IEEE DataPort 账号/订阅 | 训练集为 YouTube 原始字幕，评测集为人工标注；方言混合；标点、变音符号未核实 | HF 87.7 GB [T]；官方 masc.tar.gz 169.41 GB [P] |
| SADA（MohamedRashad/SADA22） | 声称 667 h，有转写约 438 h（训练 418 h）；HF 行数 253,166 | CC BY-NC-SA 4.0（非商业，需标注） | HF 无门槛；官方在 Kaggle（需账号） | 人工转写，沙特方言为主；有 `text` 与 `cleaned_text` 两列；标点未核实 | 50.4 GB [S] |
| SADA 的 MSA 子集（badrex/arabic-speech-SADA22-MSA） | 4,091 条，时长未核实 | CC BY-NC-SA 4.0 | HF 无门槛 | 同上，仅 MSA | 0.835 GB [S] |
| QASR（QCRI/QASR） | 约 2000 h | cc-by-nc-2.0 标签；卡片正文写「Non-Commercial Purpose ONLY!」 | 填写申请表（Google 表单），HF 仓库不含文件 | 半监督，有标点、分段和说话人信息；多方言，含语码转换 | 未核实；[估] 上限约 230 GB（WAV） |
| MGB-2（QCRI/mgb2） | 1200 h | QCRI-ALT 非独占研究许可（`license:other`） | HF 受限（需登录并接受协议），本次未接受 | 半监督字幕转写；标点未核实 | 约 120.4 GB [T]（train 118.4 GB + dev 1.0 + test 1.0） |
| WorldSpeech 阿拉伯语各配置（disco-eth/WorldSpeech） | 合计约 1001 h（见下文） | 整体 CC BY-NC 4.0，各配置另有来源许可 | HF 无门槛 | 议会记录等官方文字，人工转写；标点未核实 | 27.9 GB [S]（9 个 ar_* 配置之和） |
| ClArTTS（MBZUAI/ClArTTS） | 12 h 10 min，单人，9,705 条 | CC BY 4.0 | HF 无门槛 | 人工转写，完整变音符号；古典阿拉伯语 | 3.20 GB [S] |
| Arabic Speech Corpus（halabi2016） | 约 3.4 h（论文）；第三方统计 3.0 h | CC BY 4.0（第三方统计）；HF 卡片未写 | HF 仓库只有加载脚本，数据在作者网站 | 大马士革口音，棚内单人，Buckwalter 转写 | 未核实；[估] 约 0.4 GB |
| Egyptian-Arabic-Lectures（ismaeeelxd） | 约 30 h | MIT | HF 无门槛 | Gemini 生成后人工复核；埃及方言，夹英语术语 | 3.83 GB [S] |
| Casablanca（UBC-NLP） | 仅 dev+test，8 国各约 2 h，合计约 16 h | CC BY-NC-ND 4.0 | HF 无门槛 | 人工转写方言 | 9.16 GB [S]；各国 0.79–1.27 GB [C] |

### 土耳其语

| 语料 | 时长 | 许可 | 获取方式 | 转写形式 | 存储 |
|---|---|---|---|---|---|
| ISSAI TSC（issai/Turkish_Speech_Corpus） | 218.2 h，186,171 条 | 官方页面 CC BY 4.0；HF 标签 MIT | HF 无门槛 | 人工转写；全小写、无标点，并有拆词问题；来源 TV 和访谈节目 | 21.37 GB [T]（单个 tar.gz） |
| WorldSpeech tr_tr | 1007.6 h，239,654 条 | CC BY-NC 4.0 | HF 无门槛 | 土耳其大国民议会（TBMM）速记，人工转写；标点未核实 | 25.8 GB [S] |
| TSC 过滤修复版（KaanAydinli/tsc-tr-filtered-94h-clean） | 约 94 h，72,245 条 | MIT（继承自 TSC） | HF 无门槛 | 修复了 `İ` 和撇号后缀错误；仍是 TSC 文本风格 | 16.09 GB [S] |
| 土耳其议会演讲（alimetin/turkish-parliament-speech） | 2,777 条，时长未核实 | 未声明 | HF 无门槛 | 卡片无说明 | 1.94 GB [S] |
| Antalia 单人朗读（cloud0day3/antalia-voice-corpus） | 5.0 h，1,073 条 | CC BY 4.0 | HF 无门槛 | 与脚本对齐；单个职业配音演员，已签署再分发协议 | 0.80 GB [T] |
| Turkish Broadcast News（LDC2012S06） | 约 107 h（README）或 130 h（目录页） | LDC 用户协议，收费 | 购买；2012 年非会员价 2000 美元，现价未核实 | 人工转写；VOA 土耳其语广播 | 未核实；[估] 约 12 GB |
| Risale-i Nur 文本-音频（risaleinur/risalei-nur-text-audio） | 全库 203.02 h，土耳其语/阿拉伯语/波斯语混合，各语言分项未核实 | CC BY 4.0（卡片称获得出版社书面许可） | HF 无门槛 | 真人朗读宗教文本，转写来自官方原文；用 Whisper 检查过音文匹配 | 18.0 GB [S] |
| serdarcaglar/turkish-tts-audiobooks | 时长未核实 | `license:other` | HF 受限（auto） | 有声书，Whisper 二次转写（伪标注）；版权不明 | 172.4 GB [T] |

### 波斯语

| 语料 | 时长 | 许可 | 获取方式 | 转写形式 | 存储 |
|---|---|---|---|---|---|
| Neyshekar（shekar-ai/Neyshekar） | 99.02 h，62,279 条，190 位说话人 | CC0 1.0 | HF 无门槛 | 众包朗读，提示句 82% 由 LLM 生成并人工审核；数字未展开；标点未核实 | 11.42 GB [S] |
| PerSets/youtube-persian-asr | 385 h，40 万行以上 | CC0 1.0 | HF 无门槛（需 `trust_remote_code`） | YouTube 字幕，卡片称「原始、未验证、自动生成」，伪标注；标点未核实 | 22.99 GB [T] |
| PerSets/filimo-persian-asr | 245 h，40 万行以上 | CC0 1.0 | HF 无门槛 | Filimo 点播平台电影/剧集的波斯语字幕，原始未验证 | 36.66 GB [T] |
| ManaTTS（MahtaFetrat/Mana-TTS） | 卡片 114 h；论文摘要 86 h（两处不一致） | CC0 1.0（卡片）；伦理说明限研究开发 | HF 无门槛 | 单人，盲人杂志有声版；文本为杂志原文，经 ASR 搜索加强制对齐；标点未核实 | 35.90 GB [T]（44.1 kHz） |
| Mana-TTS v2（mmahdi-sz/mana-tts-v2-persian） | 约 64.4 h，49,532 条 | CC BY-NC-SA 4.0 | HF 无门槛 | 在 ManaTTS 基础上用 FastConformer 复核并修正 | 4.36 GB [T] |
| ShEMO（minoosh/shEMO 等） | 3 h 25 min，3,000 条，87 位说话人 | 未声明；作者说明「仅限学术使用，免费」 | GitHub/Dropbox 或 HF 镜像，无门槛 | 广播剧截取的情感语音，有正字法转写；标点未核实 | 1.04 GB [S] |
| Thomcles/Persian-Farsi-Speech | 过滤后约 417 h（卡片）；farsi v4 中记为 298 h | CC BY 4.0 | HF 受限（auto，需登录点击同意） | 由其他波斯语数据合并并降噪（DNSMOS≥3.0），伪标注；原目的为 TTS | 127.6 GB [T]（卡片写 128 GB） |
| farsi-asr/farsi-asr-dataset（srezas） | 未核实 | MIT | HF 无门槛 | YouTube 频道音频加手工字幕（GitHub 说明）；标点未核实 | 101.2 GB [T] |
| ParsVoice（MohammadJRanjbar/ParsVoice） | 2177.9 h，1,360,521 段，约 1,803 位说话人 | CC BY-NC 4.0（音频限非商业研究） | HF 受限（auto） | 有声书（IranSeda），ASR 伪标注，ASR 模型未说明，无原书文本 | 150.4 GB [T] |
| WorldSpeech fa_ir | 27.6 h，7,317 条 | 美国联邦机构内容（VOA），公有领域 | HF 无门槛 | 人工转写；标点未核实 | 0.8 GB [S] |
| kiarashQ/farsi-asr-unified-cleaned | 约 1,392.8 h，1,278,937 条 | 未声明 | HF 无门槛 | 聚合库，来源与标注混杂；数字有的写数字有的写词 | 131.4 GB [S] |
| Peacockery/farsi-asr-corpus-v4 | 985 h | CC BY 4.0（声明） | HF 无门槛 | 含 FLEURS fa_ir 13 h，不可用；文本为 CTC 规范化（数字展开为词） | 63.6 GB [S] |
| Reza2kn/visualears-persian-asr-16k | 3.93M 行，时长未核实 | Apache-2.0 | HF 无门槛 | Shenava 项目，包含伪标注，混杂 | 512.2 GB [T] |

---

## 阿拉伯语

### MASC（Massive Arabic Speech Corpus）
- 链接：官方 https://ieee-dataport.org/open-access/masc-massive-arabic-speech-corpus ；HF 镜像 https://huggingface.co/datasets/pain/MASC ；论文 IEEE SLT 2022，doi 10.1109/SLT54892.2023.10022652。
- 语言和时长：多地区、多体裁、多方言阿拉伯语，取自 700 多个 YouTube 频道，16 kHz。官方称 1000 h。Bayanat Labs 对 pain/MASC 的分段表做了统计，有转写的共 913,400 段、3,266,739 s，约 907.4 h（训练 875,873 段约 867.2 h，dev 19,521 段约 20.8 h，test 18,006 段约 19.4 h），并分干净（c）和噪声（n）两类。出处：https://huggingface.co/datasets/Bayanat-Labs/arabic-speech-hours/blob/main/evidence/measurements/masc-transcribed.json （此为第三方统计，非官方数字）。
- 许可：官方页面写「Creative Commons Attribution」（CC BY 4.0）。pain/MASC 卡片也写 cc-by-4.0。另一个镜像 abdusah/masc 标 CC BY-NC 4.0，与官方不一致，以官方为准。
- 获取：HF 镜像 API 显示 `gated: false`，卡片有「不得试图识别说话人」的提示语；官方 IEEE DataPort 页面写「需要登录」，并提到需要 IEEE DataPort 订阅。
- 转写：训练集是「从 YouTube 为每个视频提取的原始字幕」，评测集为人工标注。作者建议自己预处理（用 Maha 库）。标点和变音符号：未核实。
- 领域和说话人：YouTube 多体裁；说话人数未核实。
- 格式和大小：HF 上音频为 tar.xz/tar.gz 压缩包，合计 87.7 GB（`tree` 之和）；官方 masc.tar.gz 为 169.41 GB。
- 与 FLEURS 关系：无已知关系。
- 已知问题：字幕转写质量不一，需要自行过滤；字幕为原始文本，可能含时间戳错位。

### SADA（Saudi Audio Dataset for Arabic）
- 链接：HF 镜像 https://huggingface.co/datasets/MohamedRashad/SADA22 ；官方 https://www.kaggle.com/datasets/sdaiancai/sada2022 （页面需登录，本次未能读取正文）；MSA 子集 https://huggingface.co/datasets/badrex/arabic-speech-SADA22-MSA 。
- 内容：SDAIA 国家人工智能中心与沙特广播局合作，取自 57 个以上电视节目，主要为沙特各方言（Najdi、Hijazi、Khaliji），少量 MSA。卡片称 667 h，训练约 647 h，dev/test 约 20 h。Bayanat 统计显示有转写的只有约 438 h（训练 418 h，dev/test 各约 10 h），其余未转写，二者数字不一致，需要实际检查后确认。HF 行数 253,166。
- 许可（原文）：「Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International」。需标注非商业且相同方式共享。
- 获取：HF 镜像无门槛。官方在 Kaggle，需账号。
- 转写：人工转写（国家人工智能中心人工转写）；提供 `text`（原始）和 `cleaned_text`（规范化）两列；标点、变音符号：未核实；以沙特方言为主，MSA 只占其中一个方言标签。
- 大小：50.4 GB（datasets-server `size` 接口 `num_bytes_original_files`）。
- 与 FLEURS 关系：无。
- 已知问题：电视节目含背景音乐和重叠语音；沙特方言与 FLEURS 的埃及口音 MSA 朗读有距离。

### QASR
- 链接：https://huggingface.co/datasets/QCRI/QASR ；论文 https://arxiv.org/abs/2106.13000 （ACL 2021）。
- 内容：约 2000 h，半岛电视台（Aljazeera）新闻广播，多方言，含语码转换，16 kHz。
- 许可：卡片元数据 `cc-by-nc-2.0`，正文「Non-Commercial Purpose ONLY!」。
- 获取：HF 仓库没有数据文件，只给一个申请表（Google 表单），按页面说明走申请流程。本次没有填写。
- 转写：半监督（卡片称 lightly supervised），有语言学分段、标点、说话人信息，论文给出标点恢复基线。变音符号：未核实。
- 大小：未核实；[估] 2000 h 的 16 kHz WAV 上限约 230 GB。
- 与 FLEURS 关系：无。与 MGB-2 同源频道，重叠程度未测量。
- 价值：阿拉伯语里唯一明确带标点的大规模语料，但需要申请且限非商业。

### MGB-2
- 链接：https://huggingface.co/datasets/QCRI/mgb2 。
- 内容：约 1200 h 半岛电视台节目，轻监督转写（来源：Bayanat 统计）。
- 许可：「NON-EXCLUSIVE RESEARCH USE LICENSE FOR QCRI-ALT Corpus」，音频版权归 Al Jazeera，经 QF 授权。
- 获取：HF 受限（auto），需要登录并接受协议。本次没有接受，所以卡片内容和转写细节读不到。
- 大小：train.tar.gz 118.36 GB，dev 1.01 GB，test 1.01 GB，合计约 120.4 GB（`tree` 接口）。
- 转写形式：轻监督字幕；标点、变音符号：未核实。

### Casablanca
- 链接：https://huggingface.co/datasets/UBC-NLP/Casablanca 。
- 内容：八个国家（阿尔及利亚、埃及、约旦、毛里塔尼亚、摩洛哥、巴勒斯坦、阿联酋、也门）方言，只发布 validation 和 test，每国约 2 h。
- 许可（原文标签）：cc-by-nc-nd-4.0（禁止演绎，用于训练是否算演绎需自行判断）。
- 获取：HF 无门槛。
- 大小：合计 9.16 GB；各国下载大小：Algeria 1.215、Egypt 1.157、Jordan 0.788、Mauritania 1.191、Morocco 1.226、Palestine 1.213、UAE 1.108、Yemen 1.266 GB（卡片 `download_size`）。
- 结论：体量太小，且为评测用途，不适合作训练集。

### ClArTTS
- 链接：https://huggingface.co/datasets/MBZUAI/ClArTTS ；论文 Interspeech 2023（https://www.isca-archive.org/interspeech_2023/kulkarni23_interspeech.pdf ）。
- 内容：LibriVox 有声书中单个男性说话人的古典阿拉伯语朗读，12 h 10 min，9,705 条（9,500 训练 + 205 测试）。
- 许可：cc-by-4.0。
- 转写：论文称人工标注并给出完整变音符号（fully diacritized）；标点未核实。
- 大小：3.20 GB（`size` 接口）。
- 问题：单人、古典阿拉伯语、完整变音符号，与目标的 MSA 书面文本风格不同；用它需要去掉变音符号。

### Arabic Speech Corpus（Halabi）
- 链接：https://huggingface.co/datasets/halabi2016/arabic_speech_corpus ；官网 http://en.arabicspeechcorpus.com/ 。
- 内容：南黎凡特（大马士革口音）棚内单人录音。论文称约 3.4 h（ClArTTS 论文引用）；第三方统计 3.0 h。
- 许可：HF 卡片未写；第三方统计记为 CC BY 4.0，需核对官网。
- 转写：音素级 Buckwalter 音译。体量太小，不适合作主力数据。

### Egyptian-Arabic-Lectures
- 链接：https://huggingface.co/datasets/ismaeeelxd/Egyptian-Arabic-Lectures 。
- 约 30 h，埃及方言教学讲座，夹英语技术词；转写由 Gemini 生成后人工复核，并把俚语写法统一；MIT；3.83 GB。
- 问题：方言，且转写做了俚语归一化，与自然书写有差异。

### 其他阿拉伯语资源（简记）
- ArabicSpeech/MGB-3、MGB-5、sawtarabi：HF 无卡片、无许可声明（Bayanat 将其记为 `unknown`）。MGB-3 为 1.57 GB，MGB-5 为 6.65 GB（`size` 接口）。许可不明，不建议使用。
- WorldSpeech 阿拉伯语各配置：ar_bh 272.5 h、ar_dz 92.9 h、ar_eg 22.0 h、ar_iq 291.9 h、ar_kw 175.5 h、ar_ma 78.3 h、ar_sa 6.1 h、ar_tn 50.9 h、ar_un 11.1 h，合计约 1001 h（卡片表）；大小分别 7.5、2.3、0.7、8.8、4.6、2.0、0.1、1.5、0.3 GB（`size` 接口）。来源主要是各国议会记录，ar_un 为联合国记录且注明非商业/研究用途。说话风格是正式议会演讲，接近 MSA；各国口音不同。整体许可 CC BY-NC 4.0，且「更严格的来源许可优先」。出处：https://huggingface.co/datasets/disco-eth/WorldSpeech 。该库是多语言大库，与另一位调查员的范围可能重叠，这里只列出三种目标语言的部分。
- Bayanat-Labs/arabic-asr-channel-pairs：完全由 FLEURS ar_eg 的 500 条加信道仿真生成，不可用于训练。
- Bayanat-Labs/arabic-speech-hours（https://huggingface.co/datasets/Bayanat-Labs/arabic-speech-hours ）：第三方整理的 17 个阿拉伯语语料小时数与许可表，截至 2026-10-30，可作交叉核对用，其结论以官方来源为准。
- Quran 朗读类数据集（tarteel-ai/everyayah、zaibihassan/Quranic-Recitation-Data 等）：文本带完整变音符号，领域与目标差距大，未深入调查。
- 若要求 MSA 且带标点：搜索未发现 2025–2026 年新发布的满足条件的开放数据集（只有 IQRA 2026 的 79 h、有变音符号，和 KSAA-2026 约 5 h，二者都不适合）。

---

## 土耳其语

### ISSAI Turkish Speech Corpus（TSC）
- 链接：https://huggingface.co/datasets/issai/Turkish_Speech_Corpus ；https://github.com/IS2AI/TurkicASR ；论文 https://www.mdpi.com/2078-2489/14/2/74 ；ISSAI 数据页 https://issai.nu.edu.kz/issai-datasets/ 。
- 内容：218.2 h，186,171 条（ISSAI 数据页）。
- 许可：ISSAI 页面原文「Creative Commons Attribution 4.0 International License」；HF 标签为 MIT。二者宽松程度相当，需署名。
- 获取：HF 无门槛，仓库只有一个 ISSAI_TSC_218.tar.gz。
- 转写：人工转写，但全小写、无标点，并有拆词问题。出处：2026 年 arXiv 论文 https://arxiv.org/html/2611.03605 称 ISSAI 参考文本「lowercase and unpunctuated and split some words」；数字写法未说明。派生库 KaanAydinli/tsc-tr-filtered-94h-clean 的卡片还指出源转写存在 `İ`（如 `i stanbul`）和撇号后缀（如 `kavala nın`）的系统性错误，该派生库修复了 11.6% 的行。
- 领域：该派生库卡片描述源语料为土耳其电视和访谈节目；原始论文和 ISSAI 页面均未说明来源，这一点出自第三方卡片。说话人数：未核实。
- 格式和大小：21.37 GB 的单个 tar.gz（`tree` 接口）。
- 与 FLEURS 关系：未见。
- 已知问题：无标点且全小写，目标文本含标点和大小写，需额外补标点和大小写恢复；2026 年论文中 Whisper large-v3 在其 dev 上 WER 为 0.157，一部分来自参考文本本身的拆词问题。

### WorldSpeech tr_tr
- 链接：https://huggingface.co/datasets/disco-eth/WorldSpeech （配置 `tr_tr`）。
- 内容：1007.6 h，239,654 条，土耳其大国民议会（TBMM）议会记录；DNSMOS OVR 平均 2.22（偏低，卡片表）。
- 许可：整体 CC BY-NC 4.0；该配置来源标注为「Parliamentary Proceedings」，没有给出更具体的许可条款：未核实。
- 获取：HF 无门槛。
- 转写：卡片称「human-provided transcript」，并附 ASR 转写和 CER；标点未核实。
- 大小：25.8 GB（`size` 接口）；音频为 24 kHz OGG Opus。
- 领域：议会正式演讲，说话人数未核实。
- 问题：只有议会这一种领域；需要先看抽样文本的标点情况。

### 其他土耳其语语料
- alimetin/turkish-parliament-speech：2,777 条，1.94 GB，无许可声明，卡片无说明，不建议用。
- cloud0day3/antalia-voice-corpus：5.0 h 单人配音演员朗读，CC BY 4.0，已签署再分发协议；体量小，可作补充。
- Turkish Broadcast News（LDC2012S06）：Boğaziçi 大学，VOA 土耳其语广播，约 107 h（README）/ 130 h（目录页）；LDC 非会员收费（2012 年公告价 2000 美元，现价未核实）。出处：https://catalog.ldc.upenn.edu/LDC2012S06 。
- risaleinur/risalei-nur-text-audio：203.02 h，三语混合，CC BY 4.0，18.0 GB。内容为宗教文本朗读，用词偏奥斯曼时期风格，领域窄；三种语言的分项小时数卡片未给出。波斯语和阿拉伯语部分是否带变音符号：未核实。
- serdarcaglar/turkish-tts-audiobooks：172.4 GB，受限，许可 `other`，Whisper 二次转写，版权不明，不建议。
- Appenlimited/700h-tr-turkish-text-to-speech、Speech-data/Turkish-Speech-Dataset、SpeechDataAI/*：商业公司样本库（许可 unknown/NC-ND/other），只是样本，不作考虑。
- 未发现 MediaSpeech 以外的、可直接下载的大规模土耳其语人工转写语料；Common Voice、YODAS 按任务说明不在本范围。2026 年论文中用于对比的土耳其语数据（Common Voice 27.0 的 67.9 h、antalia 4.16 h 等）也都在其他范围。

---

## 波斯语

### Neyshekar
- 链接：https://huggingface.co/datasets/shekar-ai/Neyshekar ；论文 https://arxiv.org/abs/2609.14542 ；GitHub amirivojdan/neyshekar；Zenodo 10.5281/zenodo.18073632。
- 内容：众包网页平台上志愿者朗读，V6.0（2026-09-01）共 62,279 条、99.02 h、190 位说话人。分割按说话人划分：train 91.99 h / 134 人，validation 3.14 h / 26 人，test 3.88 h / 30 人。16 kHz 单声道 16 bit PCM。
- 许可：CC0 1.0；使用条款禁止尝试识别说话人。
- 获取：HF 无门槛。
- 转写：即朗读提示文本。提示句来源：约 5% 来自 HomoRich 手写部分，约 13% 为人工写的日常短语，82% 由 Claude Sonnet 4.5、GPT-4.5、GPT-5.1 生成并经人工审核。用 shekar 库规范化，数字不展开（论文原文「Digit scripts are unified without number verbalisation」）。评测时去标点，说明发布文本中可能有标点，但论文没有直接说，所以标点：未核实。
- 录音：用户自己的设备，不限制设备和信噪比，要求安静环境。
- 大小：11.42 GB（`size` 接口，parquet）。
- 与 FLEURS 关系：论文只与 Common Voice 做了对比，未报告与 FLEURS 重叠；提示句有 82% 由 LLM 生成，与 FLEURS 句子重复的概率低，但未做核实。
- 价值：许可最干净，朗读语音，说话人多，适合作波斯语主力真人语音之一。注意 15,222 条（24.44%）是口语化句子。

### PerSets 系列（YouTube / Filimo / Tarjoman）
- 链接：https://huggingface.co/datasets/PerSets/youtube-persian-asr ；https://huggingface.co/datasets/PerSets/filimo-persian-asr ；https://huggingface.co/datasets/PerSets/tarjoman-persian-asr 。
- 内容：youtube 385 h 以上，40 万行以上，播客、电视节目、教学、访谈、纪录片，到 2024 年中；filimo 约 245 h，40 万行以上，Filimo 点播平台上带波斯语字幕的电影、剧集、节目；tarjoman 为 507 篇文章及其音频，文章未分句，时长未核实。
- 许可：三者卡片均为 CC0 1.0。要注意：内容的原始版权属于视频制作者，CC0 是整理者的声明，不能消除原始版权风险。
- 获取：HF 无门槛，使用加载脚本（`trust_remote_code=True`）。
- 转写：youtube 卡片原文「raw, unvalidated auto-generated transcriptions」，即自动生成字幕，属伪标注；filimo 为「raw, unvalidated transcriptions」，来自平台字幕；两者按字幕时间切分，时间戳偶有不准。标点：未核实。
- 大小：youtube 22.99 GB，filimo 36.66 GB，tarjoman 5.94 GB（`tree` 接口之和）。
- 与 FLEURS 关系：未见。
- 问题：卡片本身警告需自己做质量评估；filimo 的字幕为影视对白，口语化，且可能含听障字幕标注。

### ManaTTS
- 链接：https://huggingface.co/datasets/MahtaFetrat/Mana-TTS ；论文 https://aclanthology.org/2025.naacl-long.464/ （NAACL 2025）；GitHub MahtaFetrat/ManaTTS-Persian-Speech-Dataset。
- 内容：单一说话人，来自 Nasl-e-Mana（伊朗盲人群体月刊）的有声内容，44.1 kHz。时长两处不一致：HF 卡片和 GitHub 称 114 h 以上，论文摘要称约 86 h。
- 许可：数据集 CC0 1.0，处理流程 MIT；但伦理声明称「仅用于研究和开发」，与 CC0 的说法不完全一致。
- 是否真人录音：来源是杂志的录音版本，论文摘要没有明说是真人还是合成，需要核实后再用。
- 转写：来自杂志原文，通过先用 ASR 得到近似转写、再在原文里搜索匹配，加强制对齐得到；有 `match_quality`（HIGH/MIDDLE）、CER 等列。标点、数字规范化：未核实。
- 大小：35.90 GB（`tree` 接口，44.1 kHz parquet）。
- 衍生：mmahdi-sz/mana-tts-v2-persian（64.4 h，49,532 条，CC BY-NC-SA 4.0，16 kHz FLAC，4.36 GB），用 FastConformer 复核并修复 ZWNJ。
- 问题：单人，风格单一；是否带标点要抽样确认。

### ShEMO
- 链接：https://github.com/pariajm/sharif-emotional-speech-dataset ；HF 镜像 https://huggingface.co/datasets/minoosh/shEMO 。
- 内容：3,000 条半自然语音，约 3 h 25 min，87 位母语说话人（31 女 56 男），截取自网络广播剧；情绪为愤怒、恐惧、快乐、悲伤、惊讶和中性。
- 许可：GitHub 页面没有给出许可名称，只说「可免费用于学术目的」。HF 镜像无许可标签。按「许可不明确」处理。
- 获取：GitHub 提供 Dropbox 下载（female.zip、male.zip）和 shemo.json（标签和转写）。HF 镜像无门槛。
- 转写：有正字法转写和音素转写；示例中有波斯语问号，疑似保留标点，页面未明确说明。
- 大小：1.04 GB（HF 镜像 `size` 接口）。
- 问题：体量小，情感语音；可作少量补充，价值有限。

### Thomcles/Persian-Farsi-Speech
- 链接：https://huggingface.co/datasets/Thomcles/Persian-Farsi-Speech 。
- 内容：由 farsi_voice_dataset 和 asr-farsi-youtube-chunked-30-seconds 两个 Apache-2.0 数据集合并并降噪（SepFormer），按 DNSMOS≥3.0 过滤。卡片：过滤前约 236,255 条 / 695 h，过滤后约 109,401 条 / 417 h。Peacockery v4 卡片中记为 298 h，与此不一致，原因未核实。
- 许可：CC BY 4.0（卡片）；但上游是网络爬取的 YouTube 内容，原始版权未处理。
- 获取：HF 受限（auto），需登录并点击同意共享 HF 信息，本次未接受。
- 转写：伪标注（来自上游，上游为 YouTube 字幕）；标点未核实。卡片写明目的是 TTS。降噪会改变音色，对 ASR 训练有影响。
- 大小：127.6 GB（`tree` 接口），卡片写 128 GB。

### farsi-asr/farsi-asr-dataset（srezas）
- 链接：https://huggingface.co/datasets/farsi-asr/farsi-asr-dataset ；代码 https://github.com/srezasm/farsi-asr-dataset 。
- 内容：GitHub 说明：抓取指定 YouTube 频道的高质量音频并配人工制作的字幕。时长：未核实；HF 页面只给出 100 万至 1000 万条的规模类别。
- 许可：MIT（HF 标签），但内容来自 YouTube，版权风险同上。
- 转写：人工字幕（来自频道，质量依赖频道）；标点：未核实。
- 大小：101.2 GB（`tree` 接口）；格式为 WebDataset tar，HF 查看器无法显示。

### ParsVoice
- 链接：https://huggingface.co/datasets/MohammadJRanjbar/ParsVoice 。
- 内容：1,877 本 IranSeda 有声书，共 2177.9 h，1,360,521 段，约 1,803 位推断出的说话人 ID。EMNLP 2026 主会议接收。
- 许可：标注为 CC BY-NC-4.0；音频仅限非商业研究；流程代码 Apache-2.0。作者声明版权归 IranSeda 和朗读者，只提供过滤后的短片段。
- 获取：HF 受限（auto），需登录并同意条件。本次未接受。
- 转写：ASR 伪标注（卡片未说明模型），没有书的原文参与；标点未核实。
- 大小：约 150.4 GB（`tree` 接口），卡片写音频约 130 GB，合计 150 GB。
- 价值：体量大，真人朗读，但许可限非商业、受限访问、伪标注，需谨慎。

### 其他波斯语聚合库（简记）
- kiarashQ/farsi-asr-unified-cleaned：1,392.8 h，131.4 GB，无许可声明，聚合多个网络爬取数据集，混有少量阿拉伯语音频配波斯语文本的错误（卡片称低于 0.01%），数字写法不统一。
- Peacockery/farsi-asr-corpus-v4：985 h（CV25 324 h、Thomcles 298 h、Farsi YouTube 205 h、Mana TTS 97 h、Neyshekar 37 h、FLEURS fa_ir 13 h、WorldSpeech 11 h），63.6 GB。**含 FLEURS fa_ir，不能用于本项目**；网络爬取部分的标签由 ElevenLabs Scribe 核验，文本已为 CTC 做规范化（数字展开为词，无标点），与目标不符。
- Reza2kn/visualears-persian-asr-16k：3.93M 行、512 GB，Apache-2.0，含伪标注，来源和时长不透明，不建议。
- pourmand1376/asr-farsi-youtube-chunked-30-seconds：约 40,933 行（HF 行数）、18.6 GB，卡片称 300 h 以上，Apache-2.0；与 Thomcles 的上游相同。
- MohammadGholizadeh/filimo-farsi：402,906 行、33.25 GB，无许可，疑似 Filimo 数据的另一份拷贝。
- WorldSpeech fa_ir：27.6 h，VOA 内容，美国联邦机构内容属公有领域（卡片），0.8 GB；体量小但许可清晰。
- SeyedAli/Persian-Speech-Dataset（MIT，1.0 GB，2,838 条）、persianvox（受限，manual）、ganjoor 朗诵类：未深入调查。
- 未找到波斯语 Common Voice 以外、带人工校对标点的开放真人语音语料。

---

## 给选型的建议

| 语言 | 首选 | 备选 | 阻碍 |
|---|---|---|---|
| 阿拉伯语 | MASC（907 h，CC BY 4.0，字幕转写，HF 镜像无门槛，87.7 GB） | SADA（约 438 h 有转写，NC-SA，沙特方言）；WorldSpeech ar_*（约 1001 h，NC，议会 MSA）；QASR（2000 h，唯一带标点，需申请，NC） | 标点和变音符号均未核实；MASC 字幕转写质量不一；FLEURS 是埃及口音 MSA 朗读，两个首选都有域偏移 |
| 土耳其语 | TSC（218.2 h，CC BY 4.0，人工转写） | WorldSpeech tr_tr（1007.6 h，NC，议会） | TSC 无标点全小写，需要补标点和大小写；WorldSpeech 标点未核实，需要抽样 |
| 波斯语 | Neyshekar（99 h，CC0，朗读） | PerSets youtube + filimo（约 630 h，CC0 声明，伪标注/字幕）；ManaTTS（单人，需核实真人性和标点） | 波斯语没有「人工转写 + 带标点」的大型开放库；网络库的原始版权风险；farsi v4 含 FLEURS |

## 方法说明与局限

- 本次没有读取任何数据内容（未取样转写文本），所以「标点」栏凡是卡片或论文没有明说的都写「未核实」。下一步最有价值的是对 MASC、SADA、WorldSpeech tr_tr、Neyshekar 各取几十条转写文本检查标点和数字写法，这需要用户允许下载少量样本。
- 本次没有登录，也没有接受任何受限库的协议，因此 MGB-2、Thomcles、ParsVoice、serdarcaglar 等受限库的卡片正文没有读到，信息来自 API 的元数据和描述片段。
- Bayanat Labs 的小时数统计（截至 2026-10-30）为第三方整理，作交叉核对用；关键数字尽量回到官方来源，回不去的已注明。
