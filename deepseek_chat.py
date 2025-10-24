#!/usr/bin/env python3
"""
DeepSeek Chat API 调用工具
支持流式和非流式对话
"""

import requests
import json
import os
from typing import Optional, Dict, Any, Iterator


class DeepSeekChat:
    """DeepSeek Chat API 客户端"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化DeepSeek客户端
        
        Args:
            api_key: DeepSeek API密钥，如果不提供则从环境变量DEEPSEEK_API_KEY获取
        """
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("请提供API密钥或设置环境变量DEEPSEEK_API_KEY")
        
        self.base_url = "https://api.deepseek.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def chat(self, 
             message: str, 
             model: str = "deepseek-chat",
             system_prompt: Optional[str] = None,
             temperature: float = 1.0,
             max_tokens: Optional[int] = None,
             stream: bool = False) -> Dict[str, Any]:
        """
        发送聊天消息
        
        Args:
            message: 用户消息
            model: 模型名称，默认为deepseek-chat
            system_prompt: 系统提示词
            temperature: 温度参数，控制回复的随机性
            max_tokens: 最大token数
            stream: 是否使用流式输出
            
        Returns:
            API响应结果
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": message})
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream
        }
        
        if max_tokens:
            data["max_tokens"] = max_tokens
        
        if stream:
            return self._stream_chat(data)
        else:
            return self._normal_chat(data)
    
    def _normal_chat(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """普通聊天请求"""
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=data
        )
        
        if response.status_code != 200:
            raise Exception(f"API请求失败: {response.status_code} - {response.text}")
        
        return response.json()
    
    def _stream_chat(self, data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """流式聊天请求"""
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=data,
            stream=True
        )
        
        if response.status_code != 200:
            raise Exception(f"API请求失败: {response.status_code} - {response.text}")
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str.strip() == '[DONE]':
                        break
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
    
    def conversation(self, 
                    messages: list,
                    model: str = "deepseek-chat",
                    temperature: float = 1.0,
                    max_tokens: Optional[int] = None,
                    stream: bool = False) -> Dict[str, Any]:
        """
        多轮对话
        
        Args:
            messages: 消息列表，格式为[{"role": "user/assistant/system", "content": "..."}]
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            stream: 是否使用流式输出
            
        Returns:
            API响应结果
        """
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream
        }
        
        if max_tokens:
            data["max_tokens"] = max_tokens
        
        if stream:
            return self._stream_chat(data)
        else:
            return self._normal_chat(data)


def main():
    """示例用法"""
    try:
        # 初始化客户端
        client = DeepSeekChat()
        
        print("=== DeepSeek Chat 示例 ===\n")
        
        # 示例1: 简单对话
        print("1. 简单对话:")
        response = client.chat("你好，请介绍一下你自己")
        print(f"回复: {response['choices'][0]['message']['content']}\n")
        
        # 示例2: 带系统提示的对话
        print("2. 带系统提示的对话:")
        response = client.chat(
            "写一首关于春天的诗",
            system_prompt="你是一位古典诗词专家，擅长创作优美的诗词。"
        )
        print(f"回复: {response['choices'][0]['message']['content']}\n")
        
        # 示例3: 流式对话
        print("3. 流式对话:")
        print("问题: 解释一下机器学习的基本概念")
        print("回复: ", end="", flush=True)
        
        for chunk in client.chat("解释一下机器学习的基本概念", stream=True):
            if 'choices' in chunk and len(chunk['choices']) > 0:
                delta = chunk['choices'][0].get('delta', {})
                if 'content' in delta:
                    print(delta['content'], end="", flush=True)
        print("\n")
        
        # 示例4: 多轮对话
        print("4. 多轮对话:")
        messages = [
            {"role": "user", "content": "我想学习Python编程"},
            {"role": "assistant", "content": "很好！Python是一门非常适合初学者的编程语言。你想从哪个方面开始学习呢？"},
            {"role": "user", "content": "从基础语法开始吧"}
        ]
        
        response = client.conversation(messages)
        print(f"回复: {response['choices'][0]['message']['content']}\n")
        
    except Exception as e:
        print(f"错误: {e}")
        print("\n请确保:")
        print("1. 已设置环境变量 DEEPSEEK_API_KEY")
        print("2. API密钥有效且有足够的配额")
        print("3. 网络连接正常")


if __name__ == "__main__":
    main()