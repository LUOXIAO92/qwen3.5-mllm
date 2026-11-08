# qwen3.5-mllm

这个项目以 Qwen3.5-0.8B 为例，记录如何为已有的大语言模型（LLM）或视觉语言模型（VLM）增加音频输入能力。音频编码器提取声音特征，projector（特征投影器）将其转换为语言模型接收的维度，再与文本及视觉输入一起交给语言模型，最终输出文本。

项目使用 Qwen3-Omni 的音频编码器和 Qwen3.5-0.8B 的语言、视觉骨干。教程分为以下阶段：

| 阶段 | 内容 | 教程与源码 |
| --- | --- | --- |
| 1. 模块准备 | 提取音频编码器，准备 projector、模型配置和 processor，串联前向调用 | [模块准备（中文）](01-module-preparation-zh.ipynb) · [源码](qwen3.5-0.8b-mllm/) |
| 2. 训练数据准备 | 从现有数据集抽取多语种朗读文本，合成音频并核对文字 | [数据准备（中文）](02-training-data-preparation-zh.ipynb) · [源码](pipeline/) |
| 3. 训练 | 训练音频 projector，在真人录音上评测 | [训练与评测（中文）](03-projector-training-zh.ipynb) · [源码](pipeline/train/) |
| 4. 完整模块融合 | 将训练后的音频组件与原有语言、视觉骨干整合为完整模型 | 待编写 |

阶段 2、3 的完整命令顺序见 [pipeline/README.md](pipeline/README.md)，各模型虚拟环境的建立方法见 [pipeline/ENVIRONMENTS.md](pipeline/ENVIRONMENTS.md)。第一次完整实验（v1）用 81 小时合成音频只训练 projector，模型已能转写 18 种语言，英语、西班牙语、葡萄牙语、意大利语在 FLEURS 真人录音上的词错误率约 20%–30%；结果与分析见 [results/v1-zh.md](results/v1-zh.md)。

项目采用 [Apache License 2.0](LICENSE)，保留所复用源码的原有版权声明。预训练模型权重遵循各自的许可证。
