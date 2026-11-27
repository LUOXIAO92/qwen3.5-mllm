"""回放文本的组与划分。单独放在一个不依赖 transformers 的模块里：generate 在 vLLM 启动引擎进程之前导入它，
提前导入 transformers 与分词器会让引擎进程在退出时卡住。"""

GROUPS = ("free_raw", "free_chat", "cont_raw", "cont_chat")
# 只在丢弃循环的回放目录中的组：free_chat_think（与 free_chat 相同，但助手轮按思考模式打开）；prompt_chat（数据集中的
# 真人提问 + 参考模型写的回答，提示由 user_prompts 写出，暂不生成）；free_sys_chat、free_sys_chat_think（#71，模型先自己写
# 系统提示，再写用户消息和助手回答，助手轮的打开方式分别与 free_chat、free_chat_think 相同）
ALL_GROUPS = GROUPS + ("free_chat_think", "prompt_chat", "free_sys_chat", "free_sys_chat_think")
# 带聊天模板的组（生成的主体是助手回答），以及用户消息也由模型自己写的组
CHAT_GROUPS = ("free_chat", "free_chat_think", "cont_chat", "prompt_chat", "free_sys_chat", "free_sys_chat_think")
SELF_USER_GROUPS = ("free_chat", "free_chat_think")
# 系统提示也由模型自己写的组：依次生成系统提示、用户消息、助手回答
SELF_SYSTEM_GROUPS = ("free_sys_chat", "free_sys_chat_think")
# 助手轮按思考模式打开的组：模型先写思考、</think>，再写正式回答
THINK_GROUPS = ("free_chat_think", "free_sys_chat_think")
SPLITS = ("eval", "train")


def self_roles(group):
    """模型自己写的各段依次的角色：free_chat 类为用户、助手，free_sys_chat 类为系统、用户、助手；其他组返回 None。"""
    if group in SELF_SYSTEM_GROUPS:
        return ("system", "user", "assistant")
    if group in SELF_USER_GROUPS:
        return ("user", "assistant")
    return None
