# DeepSeek Chat API 使用指南

这个项目提供了一个简单易用的 DeepSeek Chat API 调用工具，支持多种对话模式和功能。

## 文件说明

- `deepseek_chat.py` - 主要的 DeepSeek API 客户端类
- `config.py` - 配置文件，包含API设置和系统提示词
- `example_usage.py` - 使用示例和交互式聊天工具
- `requirements.txt` - Python依赖包列表
- `.env.example` - 环境变量配置模板

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置API密钥

### 方法1: 环境变量（推荐）

```bash
# Linux/Mac
export DEEPSEEK_API_KEY="your-api-key-here"

# Windows
set DEEPSEEK_API_KEY=your-api-key-here
```

### 方法2: .env文件

1. 复制 `.env.example` 为 `.env`
2. 在 `.env` 文件中填入你的API密钥

```bash
cp .env.example .env
# 编辑 .env 文件，填入API密钥
```

### 方法3: 代码中直接传入

```python
client = DeepSeekChat(api_key="your-api-key-here")
```

## 基本使用

### 1. 简单对话

```python
from deepseek_chat import DeepSeekChat

client = DeepSeekChat()
response = client.chat("你好，请介绍一下你自己")
print(response['choices'][0]['message']['content'])
```

### 2. 带系统提示的对话

```python
response = client.chat(
    "写一首关于春天的诗",
    system_prompt="你是一位古典诗词专家，擅长创作优美的诗词。"
)
```

### 3. 流式对话

```python
for chunk in client.chat("解释一下机器学习", stream=True):
    if 'choices' in chunk and len(chunk['choices']) > 0:
        delta = chunk['choices'][0].get('delta', {})
        if 'content' in delta:
            print(delta['content'], end="", flush=True)
```

### 4. 多轮对话

```python
messages = [
    {"role": "user", "content": "我想学习Python"},
    {"role": "assistant", "content": "很好！你想从哪里开始？"},
    {"role": "user", "content": "从基础语法开始"}
]

response = client.conversation(messages)
```

## 运行示例

### 交互式聊天

```bash
python example_usage.py
```

选择选项1进入交互式聊天模式，支持以下命令：
- `quit` 或 `exit` - 退出程序
- `clear` - 清空对话历史
- `stream` - 切换流式/非流式模式

### 直接运行主文件

```bash
python deepseek_chat.py
```

这将运行内置的示例，展示各种使用方式。

## API参数说明

### chat() 方法参数

- `message` (str): 用户消息内容
- `model` (str): 模型名称，默认 "deepseek-chat"
- `system_prompt` (str, 可选): 系统提示词
- `temperature` (float): 温度参数，控制回复随机性 (0.0-2.0)
- `max_tokens` (int, 可选): 最大生成token数
- `stream` (bool): 是否使用流式输出

### conversation() 方法参数

- `messages` (list): 消息历史列表
- 其他参数同 `chat()` 方法

## 支持的模型

- `deepseek-chat` - 通用对话模型
- `deepseek-coder` - 代码专用模型
- `deepseek-reasoner` - 推理专用模型

## 常用系统提示词

项目在 `config.py` 中预定义了一些常用的系统提示词：

- `assistant` - 通用助手
- `coder` - 编程专家
- `translator` - 翻译专家
- `writer` - 写作专家
- `teacher` - 教学专家

使用方式：

```python
from config import SYSTEM_PROMPTS

response = client.chat(
    "写一个排序算法",
    system_prompt=SYSTEM_PROMPTS["coder"]
)
```

## 错误处理

程序包含完整的错误处理机制：

- API密钥验证
- 网络请求异常处理
- JSON解析错误处理
- 流式输出异常处理

## 注意事项

1. 确保API密钥有效且有足够配额
2. 注意API调用频率限制
3. 流式输出时注意处理网络中断
4. 大量对话时注意token限制

## 获取API密钥

1. 访问 [DeepSeek 官网](https://platform.deepseek.com/)
2. 注册账号并登录
3. 在控制台中创建API密钥
4. 将密钥配置到环境变量中

## 故障排除

### 常见问题

1. **API密钥错误**
   - 检查密钥是否正确设置
   - 确认密钥未过期

2. **网络连接问题**
   - 检查网络连接
   - 确认防火墙设置

3. **配额不足**
   - 检查账户余额
   - 确认API调用限制

4. **模型不存在**
   - 确认使用的模型名称正确
   - 检查账户权限

如有其他问题，请查看错误信息或联系技术支持。