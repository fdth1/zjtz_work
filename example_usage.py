#!/usr/bin/env python3
"""
DeepSeek Chat API 使用示例
"""

import os
from deepseek_chat import DeepSeekChat
from config import SYSTEM_PROMPTS


def interactive_chat():
    """交互式聊天"""
    try:
        client = DeepSeekChat()
        print("=== DeepSeek 交互式聊天 ===")
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'clear' 清空对话历史")
        print("输入 'stream' 切换流式/非流式模式")
        print("-" * 40)
        
        messages = []
        stream_mode = False
        
        while True:
            user_input = input("\n你: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("再见！")
                break
            elif user_input.lower() == 'clear':
                messages = []
                print("对话历史已清空")
                continue
            elif user_input.lower() == 'stream':
                stream_mode = not stream_mode
                print(f"已切换到{'流式' if stream_mode else '非流式'}模式")
                continue
            elif not user_input:
                continue
            
            messages.append({"role": "user", "content": user_input})
            
            print("AI: ", end="", flush=True)
            
            if stream_mode:
                # 流式输出
                response_content = ""
                for chunk in client.conversation(messages, stream=True):
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        delta = chunk['choices'][0].get('delta', {})
                        if 'content' in delta:
                            content = delta['content']
                            print(content, end="", flush=True)
                            response_content += content
                print()  # 换行
                messages.append({"role": "assistant", "content": response_content})
            else:
                # 非流式输出
                response = client.conversation(messages)
                response_content = response['choices'][0]['message']['content']
                print(response_content)
                messages.append({"role": "assistant", "content": response_content})
                
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n错误: {e}")


def code_assistant():
    """代码助手示例"""
    try:
        client = DeepSeekChat()
        
        print("=== 代码助手示例 ===")
        
        # 使用代码专家系统提示
        code_question = """
        请帮我写一个Python函数，实现以下功能：
        1. 读取CSV文件
        2. 对数据进行基本的统计分析
        3. 生成简单的可视化图表
        """
        
        response = client.chat(
            code_question,
            system_prompt=SYSTEM_PROMPTS["coder"],
            temperature=0.7
        )
        
        print("问题:", code_question)
        print("\n回答:")
        print(response['choices'][0]['message']['content'])
        
    except Exception as e:
        print(f"错误: {e}")


def translation_example():
    """翻译示例"""
    try:
        client = DeepSeekChat()
        
        print("=== 翻译示例 ===")
        
        text_to_translate = "人工智能正在改变我们的世界，它在医疗、教育、交通等各个领域都有广泛的应用。"
        
        response = client.chat(
            f"请将以下中文翻译成英文：{text_to_translate}",
            system_prompt=SYSTEM_PROMPTS["translator"],
            temperature=0.3
        )
        
        print("原文:", text_to_translate)
        print("译文:", response['choices'][0]['message']['content'])
        
    except Exception as e:
        print(f"错误: {e}")


def main():
    """主函数"""
    print("请选择示例:")
    print("1. 交互式聊天")
    print("2. 代码助手")
    print("3. 翻译示例")
    print("4. 退出")
    
    while True:
        choice = input("\n请输入选项 (1-4): ").strip()
        
        if choice == '1':
            interactive_chat()
        elif choice == '2':
            code_assistant()
        elif choice == '3':
            translation_example()
        elif choice == '4':
            print("再见！")
            break
        else:
            print("无效选项，请重新输入")


if __name__ == "__main__":
    # 检查API密钥
    if not os.getenv('DEEPSEEK_API_KEY'):
        print("警告: 未设置环境变量 DEEPSEEK_API_KEY")
        print("请设置API密钥后再运行程序")
        print("例如: export DEEPSEEK_API_KEY='your-api-key-here'")
        exit(1)
    
    main()