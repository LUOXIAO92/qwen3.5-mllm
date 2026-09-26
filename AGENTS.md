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

## 当前音频塔任务

- 提取 Qwen3-Omni 音频塔，只将 `thinker.audio_tower.*` 改为 `model.audio_tower.*`，保留内部结构、参数值和原有 `forward`。
- 补齐类与 config，自定义 `.py` 随模型目录保存，支持 remote code 加载。
- Projector 和接入 Qwen3.5 等待用户后续指示。
