"""自生成回放文本（任务 #70）：用参考模型（未经训练的 Qwen3.5-2B 语言模型）自己采样的文本，减轻全量微调造成的遗忘。

四组文本：free_raw（从换行开始自由生成）、free_chat（模型自己写用户消息和助手回答）、cont_raw（续写文档开头）、
cont_chat（文档开头放进用户轮，助手续写）。丢弃循环的目录另有 free_chat_think、free_sys_chat、free_sys_chat_think
（见 groups.py）。每组分 train 与 eval，eval 不参与任何训练。

模块：`export_lm` 导出参考模型，`prefixes` 抽取文档开头，`prompts` 写提示清单，`first_token` 处理单 token 提示，
`generate` 用 vLLM 采样并保存 top-20 对数概率，`store` 是存放格式，`degeneration` 检测循环，`finalize` 截断并整理，
`langid` 判断语言，`nll` 计算参考模型的每 token 负对数似然，`report` 汇总。完整流程见 `run.sh` 和 `pipeline/README.md` 第 9 节。
"""
