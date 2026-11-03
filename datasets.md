# 朗读文本数据集清单

核查日期：2026-11-03。目标语言共 18 种：中文、英语、日语、韩语、德语、法语、俄语、葡萄牙语、西班牙语、意大利语，以及印尼语、马来语、泰语、越南语、菲律宾语、阿拉伯语、土耳其语、波斯语。

这份清单服务于训练数据准备：从现有数据集抽取适合直接朗读的条目，用 TTS 合成音频，再用 ASR 转回文字，规范化后与原文完全一致的样本保留下来，用于训练音频 projector。它替换了 2026-10-24 按「LLM 生成种子」用途整理的旧版清单，旧版中的数据集都按新标准重新核查过。

清单前半部分针对最初的十种语言，下文简称「十种语言」；最后一节针对东南亚和中东的八种语言。

各项结论来自实际读取的数据卡、HF API 和样本行。样本通过 datasets-server 的 rows 接口读取；查看器不可用时，只读取文件开头的一小段，没有下载完整数据。标注「未核实」的内容没有拿到一手信息。

## 筛选标准

朗读文本会原样交给 TTS，正文本身要能直接读出来：只用目标语言，由完整、连贯的句子组成，不含代码、表格、列表、公式、URL 和网页样板。文本需要保留标点和大小写，TTS 依靠它们断句和处理语调；很多 ASR 数据集的转写被规范化成全小写、无标点，这类文本不适合直接使用。TTS 对数字、缩写和符号的读法不固定，会让 ASR 核对对不上，因此这些内容先转写成实际读法，再交给 TTS。

许可需要允许把原文用于合成音频和训练模型。数据要同时提供短条目（一两句，几秒音频）和长条目（成段文字，几十秒音频），体裁上覆盖百科、新闻、叙事和口语。

## 推荐组合

十种语言共用一套主干来源，再按语言补充。

| 用途 | 来源 | 许可 |
| --- | --- | --- |
| 短条目 | Common Voice 句库 | CC0 |
| 短条目 | Tatoeba | CC BY 2.0 FR |
| 百科段落 | singletongue/wikipedia-paragraphs | CC BY-SA 4.0 |
| 网页段落，多体裁 | HPLT 3.0，按 web-register 体裁标签挑选 | 包装 CC0，原文版权归原作者 |
| 网页段落，高质量 | epfml/FineWeb2-HQ-Plus；英语用 HuggingFaceFW/fineweb-edu | ODC-By，原文版权归原作者 |
| 新闻 | Wikinews 各语言 dump | 多数为 CC BY 4.0 |

各语言的补充来源和注意事项：

| 语言 | 补充来源 | 注意 |
| --- | --- | --- |
| zh | opencsg/chinese-cosmopedia 的故事类（补叙事体裁；许可限制商用） | 维基、Wikinews、Tatoeba 都繁简混用，统一转为简体；网页文本有赌博关键词注入，需要段落级过滤 |
| en | fineweb-edu（说明文）；SimpleStories（叙事）；structured-wikipedia 的 abstract；Europarl；Cosmopedia v2 的故事类 | fineweb-edu 以说明文为主，叙事要从别处补 |
| ja | fineweb-2-edu-japanese 的 small_tokens_cleaned（短条目）；青空文库「新字新仮名」作品（叙事）；SimpleStories-JA | 网页博客常一句一行，需要合并；人名汉字读音不定，会影响 ASR 核对 |
| ko | Zeroth-Korean 的新闻稿文本（需补标点） | 十种语言中资源最少，叙事只能靠 HPLT 的 NA 体裁 |
| de | Europarl；FineWeb2 关联 propella-annotations 的体裁标签；Gutenberg | |
| fr | Europarl；structured-wikipedia；propella-annotations；Gutenberg | Wikinews fr 为 CC BY-SA 4.0 |
| ru | Wikinews ru（147 万篇） | 维基文本带重音符 U+0301，朗读前删除；Wikinews ru 的来源构成需抽查 |
| pt | Europarl | Europarl 是欧洲葡语、AO1990 改革前的拼写，需要和 TTS 音色的地区变体对应 |
| es | Europarl；propella-annotations；Gutenberg | |
| it | Europarl v7；propella-annotations；Gutenberg | Europarl v10 没有意大利语 |

目前的缺口在叙事和口语体裁。中文可以用 chinese-cosmopedia 的故事类，但它是合成文本，需要去重；韩语和俄语没有找到合适的叙事语料，只能用 HPLT 3.0 的 NA（叙事）体裁标签从网页里挑选叙事博客和新闻报道。合成故事语料中，只有英语和日语的 SimpleStories 质量较好；多语种 TinyStories 翻译版题材高度重复，不予采用。

## 真实语音验证集

google/fleurs 十种语言齐全，可作为真实人声验证集。它的句子来自 FLoRes，FLoRes+ 的条款要求评测内容不得出现在训练数据中，因此 FLEURS 和 FLoRes 的文本都不进入训练朗读稿。

其他带真实录音、可作验证的数据有 Common Voice、VoxPopuli（en/de/fr/es/it）、MLS（de/fr/es/it/pt）、Zeroth-Korean，以及在日本著作权法第 30 条之 4 范围内使用的 ReazonSpeech。如果 Common Voice 的句子用作训练朗读稿，验证时要排除相同的句子。

## 推荐来源

### [Common Voice 句库](https://github.com/common-voice/common-voice)

- **覆盖**：十种语言。GitHub 句子文件行数：zh-CN 约 5 万，ja 约 1.1 万，ko 6,408，ru 46,091，pt 41,140；de、en、es、fr、it 另有数十到上百 MB 的维基抽句文件。
- **文本**：保留标点和大小写，数字已写成文字，没有非语言标注和填充词。只有短句，字符数中位数 zh 17、ja 18、ko 30、欧洲语言 36–60，平台限制最长约 14 词。部分条目是词组或专名，以句末标点结尾的比例只有 ja 33%、zh 43%、es 和 pt 55%，需要过滤。
- **许可**：句子为 CC0。
- **获取**：自 2025 年 10 月起，HF 上的 `mozilla-foundation/common_voice_*` 只剩跳转说明，数据只在 [Mozilla Data Collective](https://mozilladatacollective.com) 发布，需要账号和 API key，平台条款禁止转托管。MDC 最新版本为 Common Voice Scripted Speech 27.0（2026-09-17，按语言分包），另有约 1 GB 的纯文本包「Mozilla Common Voice Text Language Identification dataset」（1900 万句，300 多种语言，CC0，列结构未核实）。GitHub 上的句子文件仍可直接下载，但众包句子的自动导出停在 2023 年。
- **附带真实录音**。

### [Tatoeba](https://tatoeba.org/en/downloads)

- **覆盖**：2026-10-31 导出的句数为 eng 2,040,024，rus 1,233,736，ita 990,954，deu 785,673，fra 729,625，por 445,567，spa 443,052，jpn 248,947，cmn 89,248，kor 15,947。
- **文本**：语言学习用的例句，偏口语，有标点和大小写，只有 0.5–5% 含数字。全是短句，字符数中位数 cmn 10、kor 15、jpn 16、欧洲语言 30–39。cmn 繁简混杂；人名 Tom 出现得很频繁，需要控制比例。
- **筛选字段**：`sentences_detailed` 有贡献者和日期，`sentences_base` 区分原创句和翻译句。
- **许可**：CC BY 2.0 FR，可商用，需署名。CC0 子集很小。
- **获取**：`https://downloads.tatoeba.org/exports/per_language/<iso3>/<iso3>_sentences.tsv.bz2`，每个 0.2–25 MB。

### [singletongue/wikipedia-paragraphs](https://huggingface.co/datasets/singletongue/wikipedia-paragraphs)

- **覆盖**：十种语言加阿拉伯语，最新 config 为 `<lang>wiki-20260607-v1.2.1`。行数（含分类页和模板页）：en 10,539,509，zh 3,146,592，ru 3,036,516，ja 1,929,103，ko 1,247,301。
- **文本**：从 HTML 按段落抽取，每页的段落放在 `paragraph_texts` 列表里，保留标点。百科体，数字和年份多。删模板后留下的空洞仍然存在，例如「史特拉斯堡座堂（法語：，德語：）」「以142（466英尺）的高度」；zh 繁简混用；en 首段带 IPA 和 ⓘ 符号。
- **筛选字段**：`paragraph_html_tags`（只保留 `p`）、`paragraph_sections`（`__LEAD__` 为导言段，可排除参考文献、外部链接等章节）、`page_type`、`templates`（排除消歧义页）、`num_inlinks`（热门度）。
- **许可**：CC BY-SA 4.0 + GFDL。
- **用法**：导言段作短、中条目，正文 `p` 段作长条目。

### [HPLT 3.0](https://huggingface.co/datasets/HPLT/HPLT3.0)

- **覆盖**：198 种语言，十种齐全。文档数：eng 18.1B，rus 3.30B，cmn_Hans 2.21B（另有 cmn_Hant 113M），spa 726M，jpn 667M，deu 645M，fra 604M，ita 363M，por 343M，kor 74.8M。zh、en、ru 没有做全局去重。
- **文本**：trafilatura 抽取，保留中文全角标点。最高质量档里也有垃圾页和电商规格页，中文有 SEO 关键词注入，质量分和「适合朗读」相关性弱。
- **筛选字段**：`web-register` 体裁标签，例如 NA 叙事（其中 ne 新闻、nb 叙事博客）、IN 信息说明、OP 观点、HI 操作指南、ID 互动讨论、SP 口语、MT 机器翻译或生成；`xml` 保留 `<p>` 段落结构；另有 `doc_scores`（WDS 质量分档）、`seg_langs`、`pii`、`cluster_size`。抽样中以 NA 为主标签的文档：ko 63/159，zh 46/168，ru 27/99，ja 13/124。
- **许可**：包装为 CC0。卡片声明不拥有任何文本，使用者自行确保符合欧盟 DSM 指令和 GDPR。
- **获取**：数据不在 HF 托管。`https://data.hplt-project.org/three/sorted/<lang>.map` 列出各 shard（`<WDS>_<n>.jsonl.zst`），可以按语言、按质量档下载或流式读取。
- **用途**：唯一同时覆盖十种语言并带体裁标签的网页语料，用来补叙事、新闻、观点等体裁。

### [epfml/FineWeb2-HQ-Plus](https://huggingface.co/datasets/epfml/FineWeb2-HQ-Plus)

- **覆盖**：2026-10-05 发布，100 种语言，含韩语，缺英语。行数：rus 64.65M，cmn 52.65M，deu 48.1M，spa 42.75M，fra 34.5M，jpn 34.0M，ita 21.9M，por 18.36M，kor 5.52M。
- **文本**：每种语言取 FineWeb2 v2.1.1 质量最高的 10%，多为完整文章，例如散文、评论、科普；仍有目录式文档和页尾样板。
- **筛选字段**：`quality_score`（偏向知识密集的内容）、`language_score`、`minhash_cluster_size`（数值极大的多为模板页）、`token_count`。`embeddings` 列很大，读取时只选文本列。
- **许可**：ODC-By 1.0 + Common Crawl 条款。
- **注意**：发布不到一个月。

### [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)

- **覆盖**：仅英语，1.3T tokens，有 `sample-10BT` 等抽样子集。
- **文本**：本次核查的网页语料中最干净，偶有代码片段和参考文献尾巴。以科普、教程等说明文为主，叙事少。
- **筛选字段**：`score`、`int_score`（教育分）、`language_score`、`token_count`。
- **许可**：ODC-By 1.0 + Common Crawl 条款。

### [openeurollm/propella-annotations](https://huggingface.co/datasets/openeurollm/propella-annotations)

- **覆盖**：只有标注，按 `id` 关联源数据。包括 FineWeb2 的 deu、spa、fra、ita，FinePDFs 的 eng、deu、fra、spa、ita、por 等，HPLT 3.0 的 deu，以及 finewiki。没有 zh、ja、ko、ru 的网页标注。
- **筛选字段**：由 LLM 标注的 18 项属性，其中 `content_ratio`（正文占比）、`content_integrity`（是否完整）、`content_type`（多标签，含 news_report、creative、opinion_editorial、boilerplate、source_code 等）与本任务最相关。
- **用法**：筛出 `content_ratio = complete_content`，且 `content_type` 包含 news_report、creative 或 opinion_editorial 的文档。
- **许可**：标注为 CC BY 4.0，正文跟随源数据集。

### [Wikinews](https://dumps.wikimedia.org/)

- **覆盖**：文章数 ru 1,473,500，pt 36,989，fr 24,204，en 22,237，zh 18,686，de 14,240，it 12,704，es 12,331，ja 4,104，ko 827。项目自 2026-05-04 起转为只读，官方 dump 仍可下载。
- **文本**：新闻正文，带电头和「消息來源」章节，需要自行解析 wikitext；zh 繁简混用。ru 数量很大，抽到的样本像转载的体育报道，来源构成未核实。
- **许可**：zh、en、ja、ko、ru、pt、es、it 为 CC BY 4.0，de 为 CC BY 2.5，fr 为 CC BY-SA 4.0。
- **获取**：`https://dumps.wikimedia.org/<lang>wikinews/latest/`。

### [Europarl](https://www.statmt.org/europarl/)

- **覆盖**：v10 句数 en 2,295,044，de 2,107,971，fr 2,083,223，es 2,038,039，pt 2,016,635；意大利语只在 v7。没有 zh、ja、ko、ru。
- **文本**：欧洲议会发言的编辑记录，98% 以句末标点结尾，没有填充词；约 11% 的句子含数字，4–5% 带程序性括号。一行一句，字符数中位数 130–143；按发言人和章节拼接相邻句子，可以得到自然的长段落。pt 为 AO1990 改革前的欧洲葡语拼写；不少发言由其他语言翻译而来。
- **筛选字段**：`file_id`、`chapter_id`、`speaker_id`、`language`（发言原语言）。
- **许可**：statmt 表示未发现版权限制；欧洲议会法律声明允许复用，包括商用，需注明来源。
- **获取**：`https://www.statmt.org/europarl/v10/training-monolingual/europarl-v10.<lang>.tsv.gz`；意大利语取 v7 的 `europarl.tgz`。

### [hotchpotch/fineweb-2-edu-japanese](https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese)

- **覆盖**：仅日语，120M 文档，由 FineWeb2 日语按教育分 ≥2.5 筛选。
- **文本**：`small_tokens_cleaned` 子集只含 512 token 以内的文本，并用噪声检测模型删掉了导航和广告片段，样本很干净。体裁为新闻、机构公告和说明文。README 提醒 `small_*` 子集的前 2 万行有重复，建议跳过前 10000 行。
- **筛选字段**：`score`、`token_count`、`is_cleaned`。
- **许可**：ODC-By + Common Crawl 条款。

### [globis-university/aozorabunko-clean](https://huggingface.co/datasets/globis-university/aozorabunko-clean)

- **覆盖**：日语，16,951 部作品，其中「新字新仮名」10,246 部。
- **文本**：页眉和注释已删除，振假名也被删除，人名读音因此丢失。「新字旧仮名」作品含「ゐ」「ほんたうに」等旧假名，要排除。体裁为小说、童话、随笔。
- **筛选字段**：`meta` 中的 `文字遣い種別`、`分類番号`、作者生卒年。
- **许可**：数据集为 CC BY 4.0，收录作品的著作权均已过期。

### [SimpleStories](https://huggingface.co/datasets/SimpleStories/SimpleStories) 与 [SimpleStories-JA](https://huggingface.co/datasets/SimpleStories/SimpleStories-JA)

- **覆盖**：en 训练集 2,115,696 篇，ja 2,087,410 篇。
- **文本**：干净的叙事段落，en 每篇约 150 词，ja 每篇约 100–450 字。人名集中，例如 Mia、Leo 反复出现，需要去重。ja 版开头受首字约束，读起来别扭；人名用汉字书写，读音不定。
- **筛选字段**：`topic`、`theme`、`style`、`word_count`、`num_paragraphs` 等；en 另有可读性分数。
- **许可**：MIT。文本由 gpt-4o-mini 生成，OpenAI 条款对下游的约束未核实。

### [wikimedia/structured-wikipedia](https://huggingface.co/datasets/wikimedia/structured-wikipedia)

- **覆盖**：只有 en（7,597,149 页）和 fr（2,871,732 页），抽取日期 2026-05-13。
- **文本**：`sections` 按段落、列表、表格、图片分类型存放，可以只取段落。段落会丢掉上下标，`abstract` 字段保留完整。
- **筛选字段**：`abstract`、`description`、`version.scores`、`maintenance_tags`、`number_of_characters`。
- **许可**：CC BY-SA 4.0。
- **用法**：`abstract` 作短、中条目，段落作长条目。

### [HuggingFaceFW/finewiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki)

- **覆盖**：325 种语言，十种齐全，来自 2025 年 8 月的 Wikipedia HTML 快照。
- **文本**：模板已渲染，删除了参考文献、外部链接等章节和消歧义页。正文为 Markdown，保留标题、列表、表格和公式，需要按行过滤。zh 同样有删除外文名后留下的空洞；ru 保留重音符。
- **筛选字段**：`has_math`、`infoboxes`、`wikidata_id`。
- **许可**：CC BY-SA 4.0 + GFDL。
- **用途**：wikipedia-paragraphs 的备选。

## 条件可用

| 数据集 | 适用语言 | 使用条件 |
| --- | --- | --- |
| [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) | 十种 | 快照只有 2023-11-01；保留列表、章节标题、参考文献标题和分类名，删模板后有空洞。优先用 wikipedia-paragraphs |
| [epfml/FineWeb2-HQ](https://huggingface.co/datasets/epfml/FineWeb2-HQ) | 缺 en、ko | 已被 HQ-Plus 取代 |
| [HuggingFaceFW/finepdfs-edu](https://huggingface.co/datasets/HuggingFaceFW/finepdfs-edu)、[finepdfs](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) | 十种 | 类 Markdown 输出，含表格、LaTeX、URL 和跨行拼接造成的错误空格（韩语会影响核对）；只建议给 ru、de、en 补长篇说明文 |
| [facebook/voxpopuli](https://huggingface.co/datasets/facebook/voxpopuli) 的 `original_text` | en/de/fr/es/it | CC0；5–12% 从句中开始，需要去掉；附真实录音。只取文本可下载 `asr_<lang>.tsv.gz` |
| [nvidia/Granary](https://huggingface.co/datasets/nvidia/Granary) | en/de/fr/es/it/pt | CC BY 4.0；Whisper 伪标注再由 LLM 补标点，长片段中 13–54% 从句中开始，需要裁到句边界；yodas 部分和 ru 不用 |
| [kresnik/zeroth_korean](https://huggingface.co/datasets/kresnik/zeroth_korean) | ko | 新闻稿朗读，句子完整，数字已写成韩文读法，但没有标点；CC BY 4.0；附真实录音 |
| [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) 的 `cosmopedia-v2` | en | 只取故事类，删除 Markdown，按开头句式去重 |
| [opencsg/chinese-cosmopedia](https://huggingface.co/datasets/opencsg/chinese-cosmopedia) | zh | 只取故事类，删除 Markdown；商用需向 OpenCSG 申请许可 |
| [manu/project_gutenberg](https://huggingface.co/datasets/manu/project_gutenberg) | en/fr/de/es/it | 删除 PG 页眉页脚，重新拼接硬换行；注意旧正字法 |
| Wikivoyage | de/en/it/fr | 只取叙述段落；zh 版几乎全是条目清单 |
| [日本国会会議録 API](https://kokkai.ndl.go.jp/help.html) | ja | 发言记录可拼成长段落，数字用汉字书写；需要清理说话人前缀和程序性表头。著作权归发言人，需自行判断著作权法的权利限制条款是否适用 |
| [OPUS GlobalVoices](https://huggingface.co/datasets/wecover/OPUS_GlobalVoices) | 多数 | 原站为 CC BY 3.0；有句子切分残片 |
| [opencsg/Fineweb-Edu-Chinese-V2.1](https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1)、[openbmb/Ultra-FineWeb](https://huggingface.co/datasets/openbmb/Ultra-FineWeb) | zh | 前者商用需申请许可，样本未核实；后者上游许可需逐一核对，列表式短行多 |
| [minpeter/fineweb-2-edu-korean](https://huggingface.co/datasets/minpeter/fineweb-2-edu-korean) | ko | 没有许可标签和方法说明，只作备选 |

## 不推荐

| 数据集 | 原因 |
| --- | --- |
| HuggingFaceFW/fineweb-2 原版 | 没有质量分，样板噪声最重；改用 HQ-Plus |
| hotchpotch/FineWeb2-IR | 只是 FineWeb2-HQ 的截断副本，缺韩语，没有许可标签 |
| allenai/MADLAD-400 | 没有任何元数据；中日文 clean 版大量是赌博和仿冒品页面；许可标签与 README 不一致 |
| uonlp/CulturaX | 需要申请访问，没有质量和体裁字段，已被 FineWeb2 和 HPLT 3.0 取代；样本未核实 |
| Wikidata | label 和 description 是专名和名词短语，太短，也不成句 |
| WikiLingua | 实际许可为 CC BY-NC-SA 3.0；步骤首句被抽走，文本不连贯 |
| CohereLabs/aya_collection_language_split | 指令格式；大部分为机器翻译，出现 `<unk>`、分词空格和截断 |
| FLoRes+ | 条款禁止用作训练数据，并且与 FLEURS 是同一批句子 |
| MLS | 全小写、无标点，片段在句中切断；可作 de/fr/es/it/pt 的真实语音验证 |
| Emilia / Emilia-YODAS | 文本是 ASR 输出，有错字、填充词和截断句；Emilia 部分仅限非商用 |
| YODAS2 | 字幕断行，不成句；自动字幕没有标点 |
| TED 演讲转写 | CC BY-NC-ND，禁止衍生作品和配音 |
| WenetSpeech、KsponSpeech、Golos | 无标点或带转写标注，口语片段；WenetSpeech 仅限非商用，Golos 要求相同方式共享，KsponSpeech 许可未核实 |
| ReazonSpeech | 字幕文本与语音不总是一致；只可在日本著作权法第 30 条之 4 范围内使用 |
| Wikisource、Wikibooks | Wikisource 多数条目没有展开正文，夹带旧正字法；Wikibooks 带导航、注音和代码 |
| PleIAs/common_corpus 及 PD 系列 | OCR 错误多，旧正字法和文语体，不能按语言下载 |
| Gabrui/multilingual_TinyStories | 翻译版题材高度重复，人名译法不一致 |
| stanford-oval/ccnews | 几乎没有中日韩文，没有许可标签 |
| csebuetnlp/xlsum、reciTAL/mlsum | 新闻摘要数据集；短条目和新闻体裁已有其他来源覆盖，并且仅限非商用 |

## 段落级清洗

所有来源都需要在段落级处理，文档级质量分无法保证干净：HPLT 3.0 最高质量档里有德语垃圾页，FineWeb2-HQ 的中文样本带 cookie 提示。核查中见到的问题和对应处理如下。

- 网页首尾样板，例如「上一篇/下一篇」、cookie 提示、「ПОХОЖИЕ ПУБЛИКАЦИИ」：删除首尾样板行。
- FineWeb 系列把邮箱、IP 换成固定占位符（如 `email@example.com`）：含占位符或 URL 的段落整段丢弃。
- 中文网页段落中注入的赌博、彩票关键词，以及打乱标点的洗稿文本：按关键词和标点异常过滤。
- 维基文本删模板后留下的空括号和缺失数值，以及以冒号结尾、引出列表或表格的段落（如「以下為各獎項結果：」）：整段丢弃。
- PDF 和合成语料中的 Markdown 标题、加粗、表格、列表和 LaTeX：删除标记，或整段丢弃。
- 日语博客一句一行：按空行重新拼成段落。
- 俄语维基的重音符 U+0301：删除。中文繁简混用：统一转为简体。
- 数字、单位和缩写：按各语言统一的读法规范转写成文字后保留。

## 许可分类

本项目对外只发布教程和清洗、TTS 合成脚本，合成的音频—文本数据和训练出的权重都不发布；使用者运行脚本时从各来源自行下载原始数据。原文再分发、署名和相同方式共享的义务在分发数据时产生，本项目不涉及。本项目纯开源，不涉及商业使用，限非商用的来源也可以使用。脚本和教程需要标明每个来源的许可，使用者按各自的用途判断；每条数据保存来源数据集，以后如果要公开权重，可以据此筛掉受限来源。

- **CC0**：Common Voice 句库、VoxPopuli、HPLT 3.0 的包装（原文版权仍归作者）。
- **署名类**：Tatoeba（CC BY 2.0 FR）、Wikinews（多数为 CC BY 4.0，de 为 CC BY 2.5）、青空文库整理版（CC BY 4.0）、Zeroth-Korean（CC BY 4.0）、Granary（CC BY 4.0）、Europarl 的欧洲议会原始记录（允许商用）、propella-annotations 的标注（CC BY 4.0）。
- **ODC-By + Common Crawl 条款**：FineWeb 系列、FinePDFs 系列。数据库许可只覆盖整理工作，网页和 PDF 原文的版权仍归原作者，本项目只用于训练。
- **相同方式共享**：维基百科各版本、Wikivoyage、Wikinews fr（CC BY-SA 4.0）。
- **访问条款**：Common Voice 需用 Mozilla Data Collective 账号下载，平台禁止转托管，脚本只能引导使用者自行下载；HPLT 3.0 要求使用者自行确保符合欧盟 DSM 指令和 GDPR。
- **MIT**：SimpleStories。文本由 gpt-4o-mini 生成，OpenAI 条款对下游的约束未核实。
- **限非商用或需另取许可**：XL-Sum、WikiLingua、Emilia（Emilia-YODAS 为 CC BY 4.0）、WenetSpeech、TED 系列；chinese-cosmopedia 和 Fineweb-Edu-Chinese-V2.1 商用需向 OpenCSG 申请。

## 东南亚与中东语言

这部分调查东南亚和中东各国的国家语言或官方语言，核查日期同为 2026-11-03。流程要先合成音频、再用 ASR 转回文字核对，因此目标语言只纳入 TTS 和 ASR 都有候选的八种：印尼语、马来语、泰语、越南语、菲律宾语、阿拉伯语、土耳其语、波斯语。希伯来语、缅甸语、高棉语、老挝语、库尔德语、阿塞拜疆语、普什图语、爪哇语、宿务语缺少 ASR，德顿语和巽他语 TTS 与 ASR 都缺，暂不纳入。

### TTS 与 ASR 支持

依据各模型官方页面列出的支持语言。Qwen3-TTS 不支持下表任何一种语言。OmniVoice 括号内是它在该语言上的训练时长，时长很短的语言合成质量需要实测。

| 语言 | TTS | ASR |
| --- | --- | --- |
| 印尼语 id | FireRedTTS3、VoxCPM2、OmniVoice | Qwen3-ASR |
| 马来语 ms | VoxCPM2、OmniVoice（9.6 h） | Qwen3-ASR |
| 泰语 th | FireRedTTS3、VoxCPM2、OmniVoice | Qwen3-ASR |
| 越南语 vi | FireRedTTS3、VoxCPM2、OmniVoice | Qwen3-ASR、Cohere |
| 菲律宾语 fil | VoxCPM2（Tagalog）、OmniVoice（7.7 h） | Qwen3-ASR |
| 阿拉伯语 ar | IndexTTS-2.5、FireRedTTS3、VoxCPM2、OmniVoice | Qwen3-ASR、Cohere |
| 土耳其语 tr | FireRedTTS3、VoxCPM2、OmniVoice | Qwen3-ASR |
| 波斯语 fa | OmniVoice | Qwen3-ASR |

### 现有通用来源的覆盖

十种语言的主干来源在这些语言上明显变弱。singletongue/wikipedia-paragraphs 只覆盖其中的阿拉伯语（254 万页）。finewiki 全部覆盖，但东南亚语言的条目多为短小的 stub、列表和表格，越南语约一半是物种条目。Wikinews 在东南亚只有已关闭的泰语版（25 篇），中东为 ar 3,392、fa 1,624、tr 1,184 篇，ALT 可以部分替代。

HPLT 3.0 和 FineWeb2-HQ-Plus 覆盖全部八种语言（FineWeb2-HQ-Plus 的波斯语 config 为 `fas_Arab`），HPLT 都带 web-register 体裁标签。菲律宾语网页中机器翻译多，HPLT 约 38% 的文档被标为 MT，需要按标签或域名排除；样本中还有机器翻译站点和赌博 SEO 页面。两者的波斯语都完全丢失了零宽不连字（ZWNJ）。

Common Voice 和 Tatoeba 的质量因语言而异。Common Voice 的 id、ms 句子 94–96% 以句末标点结尾；vi 只有 4%，多为小写片段；th 以词组和标题为主；ar 约四分之一全标元音，另有 1,316 行把句号写在行首。Tatoeba 的 tgl 有 7.9 万句，是菲律宾语最大的短句来源；tur 有 75 万句，其中约 24% 含人名 Tom。

### 各语言来源

| 语言 | 短条目 | 长条目 | 叙事、口语 | 真实语音验证 |
| --- | --- | --- | --- | --- |
| id | Tatoeba、ALT、TALPCo | id_newspapers_2018；FineWeb2-HQ-Plus；HPLT 3.0 | indo_story_cloze | FLEURS、Common Voice |
| ms | Tatoeba（去掉 Jawi）、ALT、TALPCo | FineWeb2-HQ-Plus；HPLT 3.0（叙事、观点多） | 马来西亚 Hansard | FLEURS |
| th | ALT、TALPCo、scb_mt_enth_2020 对话类 | prd_news、thaigov-v2（政府新闻，CC0）；FineWeb2-HQ-Plus；HPLT 3.0 | GlotStoryBook | FLEURS、Common Voice |
| vi | Tatoeba、ALT、TALPCo | FineWeb2-HQ-Plus；HPLT 3.0；BKAINewsCorpus（未标许可） | GlotStoryBook | FLEURS、Common Voice |
| fil | Tatoeba（7.9 万句）、ALT | NewsPH；Gutenberg tl（旧拼写） | GlotStoryBook | FLEURS |
| ar | Tatoeba、Common Voice、wikipedia-paragraphs 导言段 | wikipedia-paragraphs；HPLT 3.0 的叙事、新闻体裁；FineWeb2-HQ-Plus；UNPC | 缺 | FLEURS、Common Voice |
| tr | Tatoeba、Common Voice、TR-News 的 abstract | ParlaMint-TR；OzenliDerlem；FineWeb2-HQ-Plus | TinyStories_Turkish | FLEURS、Common Voice |
| fa | Tatoeba、Common Voice、pn_summary 的 summary | pn_summary；finewiki（保留 ZWNJ） | TinyStories-Persian | FLEURS、Common Voice、Mana-TTS |

### 新增来源

| 数据集 | 语言 | 许可 | 结论与用法 |
| --- | --- | --- | --- |
| [mutiyama/alt](https://huggingface.co/datasets/mutiyama/alt)（ALT） | id ms th vi fil | CC BY 4.0 | 推荐。约 2 万句英文 Wikinews 的人工译文，带标点，作中短条目；外文专名多 |
| [TALPCo](https://github.com/matbahasa/TALPCo) | id ms th vi | CC BY 4.0 | 推荐。每种语言 1,372 句日常口语短句 |
| [cis-lmu/GlotStoryBook](https://huggingface.co/datasets/cis-lmu/GlotStoryBook) | fil vi th | 逐条 CC BY / CC BY-NC | 推荐。童书故事，每语数百段，小规模叙事补充 |
| [pythainlp/prd_news_30112023](https://huggingface.co/datasets/pythainlp/prd_news_30112023) | th | CC0 | 推荐。24.3 万条政府新闻；缩写和佛历年份多，「ำ」常被拆写 |
| [pythainlp/thaigov-v2-corpus-31032024](https://huggingface.co/datasets/pythainlp/thaigov-v2-corpus-31032024) | th | CC0 | 推荐。4.5 万篇；需删「•」分隔符 |
| [pythainlp/scb_mt_enth_2020](https://huggingface.co/datasets/pythainlp/scb_mt_enth_2020) | th | CC BY-SA 4.0 | 条件可用。只取 task_master、mozilla 子集补口语短句 |
| [indonesian-nlp/id_newspapers_2018](https://huggingface.co/datasets/indonesian-nlp/id_newspapers_2018) | id | CC BY-SA 4.0，限研究 | 推荐。约 50 万篇新闻；需删电头，缩写和括号日期多 |
| [indolem/indo_story_cloze](https://huggingface.co/datasets/indolem/indo_story_cloze) | id | CC BY-SA 4.0 | 条件可用。2,325 个五句小故事 |
| [malaysia-ai/malaysia-hansard](https://huggingface.co/datasets/malaysia-ai/malaysia-hansard) | ms | 无许可标签 | 条件可用。议会 PDF 抽取，只取近年文档 |
| [jcblaise/newsph](https://huggingface.co/datasets/jcblaise/newsph) | fil | GPL-3.0 | 条件可用。夹带大量英语机构名 |
| [bkai-foundation-models/BKAINewsCorpus](https://huggingface.co/datasets/bkai-foundation-models/BKAINewsCorpus) | vi | 未标注 | 条件可用。1,676 万篇标题加导语 |
| [UNPC](https://huggingface.co/datasets/Helsinki-NLP/un_pc) 阿拉伯语 | ar | 卡片称自由许可，原文条款未核实 | 条件可用。联合国公文；1990 年代文档有展示形连字、倒序数字和反向括号，需丢弃 |
| [HeshamHaroon/arabic-msa-25k-saudi-male-tashkeel](https://huggingface.co/datasets/HeshamHaroon/arabic-msa-25k-saudi-male-tashkeel) | ar | CC BY 4.0 | 条件可用。LLM 生成并全标元音的标准语短句，元音准确度未核实 |
| [HooshvareLab/pn_summary](https://huggingface.co/datasets/HooshvareLab/pn_summary) | fa | MIT | 推荐。9.3 万篇新闻，summary 作短条目，数字为波斯数字 |
| [s-pouyan/TinyStories-Persian](https://huggingface.co/datasets/s-pouyan/TinyStories-Persian) | fa | CDLA-Sharing-1.0 | 推荐。约 200 万篇翻译故事，部分经人工修订；仍在更新，需固定版本 |
| [MahtaFetrat/Mana-TTS](https://huggingface.co/datasets/MahtaFetrat/Mana-TTS) | fa | CC0 | 条件可用。114 h 单人朗读，转写需拼回整句 |
| [MahtaFetrat/HomoRich-G2P-Persian](https://huggingface.co/datasets/MahtaFetrat/HomoRich-G2P-Persian) | fa | CC0 | 条件可用。每句含多音同形词，用于测试读音 |
| [ParlaMint-TR 4.1](http://hdl.handle.net/11356/1912) | tr | CC BY 4.0 | 推荐。议会记录约 5,000 万词，TEI 格式，段落完整 |
| [turkish-nlp-suite/OzenliDerlem](https://huggingface.co/datasets/turkish-nlp-suite/OzenliDerlem) | tr | CC BY-SA 4.0 | 推荐。139 万篇新闻、游记、科普、随笔 |
| [esat-krky/TinyStories_Turkish](https://huggingface.co/datasets/esat-krky/TinyStories_Turkish) | tr | CDLA-Sharing-1.0 | 推荐。40 万篇故事，生成方式未说明 |
| [batubayk/TR-News](https://huggingface.co/datasets/batubayk/TR-News) | tr | 未声明 | 条件可用。abstract 作短条目 |

不推荐：aisingapore/SEA-PILE-v1、v2（没有筛选字段，按行过滤打断段落）；Sailor2 系列数据集（标签错乱或无许可）；越南语、阿拉伯语的 TinyStories 机器翻译版（题材重复，有误译）；ClusterlabAi/101_billion_arabic_words_dataset 和 khalidalt/ultimate_arabic_news（标点被删除）；lightonai/ArabicWeb24（需申请，没有筛选字段）；community-datasets/tashkeela（古典宗教文本，只适合作元音加注器的参照）；issai/Turkish_Speech_Corpus（全小写、无标点）；LSCP（CC BY-NC-ND）。

### 书写处理要点

- **泰语**：句末不用句号，靠空格分句，存在歧义，可用 PyThaiNLP 的 `sent_tokenize` 或 wtpsplit。「ำ」常被拆写成「ํา」，朗读前合并；规范化用 NFC，NFKC 会把这个字符拆开。还要处理泰文数字、ๆ、ฯ、带点缩写和佛历年份。
- **马来语、越南语、菲律宾语**：马来语按字符集去掉 Jawi；越南语统一声调符号位置（hòa / hoà）；菲律宾语网页常混写英语，Gutenberg 是旧拼写。
- **阿拉伯语**：通常不写短元音，TTS 读音可能有歧义，可用 CATT 加注，TTS 是否从加注中受益未核实。核对时删除元音符号和 tatweel，并定下 أ/إ/آ、ة/ه、ى/ي 是否统一。数词要与名词保持性、格一致，转写读法的规则比其他语言复杂。
- **波斯语**：优先使用保留 ZWNJ 的来源；网页语料可用 hazm 补全一部分 ZWNJ，并把 ي、ك 统一为 ی、ک；核对时把 ZWNJ 与空格视为等价。
- **从右到左文本**：阿拉伯语、波斯语中删除 LRM、RLM 控制符，修正写在行首的句末标点，含拉丁字母的句子视情况丢弃。
- **数字**：ASCII、阿拉伯-印度（٠–٩）、波斯（۰–۹）三套数字并存，先统一再转写读法；土耳其语数字后的后缀要随读法做元音和谐。
