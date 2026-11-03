# 用户的编程与教材要求

## Python 写法

- 支持关键字传参的接口必须显式写出 `参数名 = 值`。仅在接口强制要求位置参数时保留位置参数。
- 赋值、关键字参数和默认参数中的 `=` 两侧各留一个空格。
- 赋值很长时换行；调用参数达到四个及以上时，参数逐行书写。
- Notebook、独立 `.py` 和文档中的 Python 示例遵守同一套格式。

示例：

```python
with safe_open(filename = shard_path, framework = "pt", device = "cpu") as shard:
    ...

model = AutoModel.from_pretrained(
    pretrained_model_name_or_path = export_dir,
    trust_remote_code = True,
    dtype = torch.float32,
    attn_implementation = "eager",
)
```

## Jupyter 自学教材

- 使用 Jupyter notebook 组织教学，按理解和操作的先后顺序说明每一步的目的、代码及结果。
- 使用容易理解的中文。必要术语就地解释，说明直接围绕当前操作展开。
- 自定义模块可以放在 `.py` 中；notebook 必须展示相关代码，说明用途、修改位置和对应文件。
- `.py` 源码用普通 Markdown Python 代码块展示。
- 教材只包含用户要求的学习内容。内部检查在教材之外执行；未经要求，不加入版本哈希固定、环境版本打印、成串的 `assert`、新进程与缓存隔离验证等工程步骤。
- 输出用于帮助理解当前操作，避免把教材扩展成验收流程。

## 范围与表达

- 依据完整对话中的用户要求确定范围；技术上相关的内容也需要符合用户指定的学习目标。
- 用户后续明确删减的内容从教材中移除，不以此前助手提出的计划作为保留理由。
- 直接说明是什么、做什么。禁止使用“不是……而是……”句式，禁止预设用户没有提出的误解或问题，再展开防御性解释。
- 撰写或修改讲解文案时，必须阅读并参考 `$eli5` 技能（项目：[dreambigou/eli5](https://github.com/dreambigou/eli5)）。文案以容易理解为前提保持简洁，必要概念和步骤解释清楚。
- 解释围绕用户当前问题展开，只保留理解当前操作所需的细节，禁止无限扩张到大量工程细节。
- 使用连贯的自然段，同一意思的相关句子放在同一段中。禁止一句话换一行；按内容层次分段，仅在确有并列或步骤关系时使用列表。
- 用户要求审查写作结果时，审查需求对应、教学顺序、表达和范围；按完整对话历史判断。

## 当前进度

- 阶段 1 模块准备已完成，教材为 `01-module-preparation-zh.ipynb`，源码在 `qwen3.5-0.8b-mllm/`。音频塔从 Qwen3-Omni 提取，只将 `thinker.audio_tower.*` 改为 `model.audio_tower.*`，保留内部结构、参数值和原有 `forward`；自定义 `.py` 随模型目录保存，支持 remote code 加载。Projector 为 `Linear → SiLU → Linear`，参数新初始化，尚未训练。音频塔、projector 已与 Qwen3.5 的视觉塔、语言模型融合，完整模型可以保存、加载并调用 `generate`。
- 阶段 2 训练数据准备正在进行，教材为 `02-training-data-preparation-zh.ipynb`。目标语言共 18 种：中文、英语、日语、韩语、德语、法语、俄语、葡萄牙语、西班牙语、意大利语，以及印尼语、马来语、泰语、越南语、菲律宾语、阿拉伯语、土耳其语、波斯语。朗读文本从现有数据集抽取，再合成 TTS 音频用于训练 projector；数据集清单见 `datasets.md`。项目纯开源，不涉及商业使用；对外只发布教程和清洗、TTS 合成脚本，合成数据和训练权重都不发布。LLM 无提示自由续写会产出大量代码，按主题提示生成又因种子提示词单一而内容重复严重，因此不再用 LLM 生成文本。`research/` 保存 TTS、ASR 选型调查 `research/training-data-selection-zh.md` 和文本规范化调查 `research/text-normalization-zh.md`。
- 实验 v1 覆盖阶段 2 的数据生成、阶段 3 的 projector 训练和真人语音评测，任务与进度用 GitHub issue 管理，总览为 #26；工作分支 `exp/v1`，共用约定见 `pipeline/CONVENTIONS.md`，数据与环境放在仓库外的 `/home/luoxiao/workspace/qwen3.5-asr-work/`。每个关键节点完成、代码或文档修改合并后都要 commit 并 push。
- 阶段 4 完整模块融合等待用户后续指示。
