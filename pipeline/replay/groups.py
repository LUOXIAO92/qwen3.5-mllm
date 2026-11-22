"""回放文本的组与划分。单独放在一个不依赖 transformers 的模块里：generate 在 vLLM 启动引擎进程之前导入它，
提前导入 transformers 与分词器会让引擎进程在退出时卡住。"""

GROUPS = ("free_raw", "free_chat", "cont_raw", "cont_chat")
# 只在丢弃循环的回放目录中的组：free_chat_think（与 free_chat 相同，但助手轮按思考模式打开）；prompt_chat（数据集中的
# 真人提问 + 参考模型写的回答，提示由 user_prompts 写出，暂不生成）
ALL_GROUPS = GROUPS + ("free_chat_think", "prompt_chat")
# 带聊天模板的组（生成的主体是助手回答），以及用户消息也由模型自己写的组
CHAT_GROUPS = ("free_chat", "free_chat_think", "cont_chat", "prompt_chat")
SELF_USER_GROUPS = ("free_chat", "free_chat_think")
SPLITS = ("eval", "train")
