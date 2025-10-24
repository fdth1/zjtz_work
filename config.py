"""
DeepSeek Chat 配置文件
"""

# DeepSeek API 配置
DEEPSEEK_CONFIG = {
    "base_url": "https://api.deepseek.com/v1",
    "models": {
        "chat": "deepseek-chat",
        "coder": "deepseek-coder",
        "reasoner": "deepseek-reasoner"
    },
    "default_params": {
        "temperature": 1.0,
        "max_tokens": 4096,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0
    }
}

# 常用系统提示词
SYSTEM_PROMPTS = {
    "assistant": "你是一个有用的AI助手，能够回答各种问题并提供帮助。",
    "coder": "你是一个专业的程序员，擅长编写高质量的代码并解决技术问题。",
    "translator": "你是一个专业的翻译专家，能够准确地在不同语言之间进行翻译。",
    "writer": "你是一个专业的写作专家，能够创作各种类型的文章和内容。",
    "teacher": "你是一个耐心的老师，善于用简单易懂的方式解释复杂的概念。"
}