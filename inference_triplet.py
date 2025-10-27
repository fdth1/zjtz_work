#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChatGLM-6B QLoRA三元组抽取推理脚本
纯Python实现，支持交互式和批量推理
"""

import os
import sys
import json
from pathlib import Path

def main():
    """主推理函数"""
    print("=" * 60)
    print("ChatGLM-6B QLoRA 三元组抽取推理")
    print("=" * 60)
    
    # 设置项目根目录
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # 添加src目录到Python路径
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # 检查模型是否存在
    base_model = "THUDM/chatglm-6b"
    lora_model = "output/chatglm-6b-triplet-qlora"
    
    lora_path = Path(lora_model)
    if not lora_path.exists():
        print(f"错误: LoRA模型不存在 {lora_path.absolute()}")
        print("请先运行训练脚本: python train_triplet.py")
        return False
    
    print(f"基础模型: {base_model}")
    print(f"LoRA模型: {lora_path.absolute()}")
    
    try:
        # 导入推理模块
        from inference import TripletExtractor
        
        print("\n正在加载模型...")
        extractor = TripletExtractor(
            base_model_path=base_model,
            lora_model_path=str(lora_path)
        )
        print("模型加载完成！")
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保已安装所有依赖: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"模型加载失败: {e}")
        return False
    
    # 交互式推理
    print("\n" + "=" * 60)
    print("交互式三元组抽取")
    print("输入文本进行三元组抽取，输入 'quit' 或 'exit' 退出")
    print("=" * 60)
    
    # 示例文本
    examples = [
        "马云是阿里巴巴的创始人，阿里巴巴总部位于杭州。",
        "张三毕业于清华大学，现在在腾讯担任CTO职位。",
        "李四是小米的产品经理，小米总部在北京。",
        "王五从北京大学毕业后，在字节跳动工作，担任技术总监。"
    ]
    
    print("\n示例文本（可直接输入数字选择）:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")
    
    while True:
        try:
            user_input = input("\n请输入文本 (或输入数字选择示例): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("退出推理程序")
                break
            
            if not user_input:
                continue
            
            # 检查是否是数字选择
            if user_input.isdigit():
                idx = int(user_input) - 1
                if 0 <= idx < len(examples):
                    text = examples[idx]
                    print(f"选择的示例: {text}")
                else:
                    print("无效的示例编号")
                    continue
            else:
                text = user_input
            
            print(f"\n输入文本: {text}")
            print("正在抽取三元组...")
            
            # 执行推理
            result = extractor.extract_triplets(text)
            
            print(f"抽取结果:")
            if result.strip():
                print(result)
            else:
                print("未找到三元组")
            
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"推理过程中出现错误: {e}")
            continue
    
    return True

def batch_inference(texts, output_file=None):
    """批量推理函数"""
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    try:
        from inference import TripletExtractor
        
        print("正在加载模型...")
        extractor = TripletExtractor(
            base_model_path="THUDM/chatglm-6b",
            lora_model_path="output/chatglm-6b-triplet-qlora"
        )
        
        results = []
        for i, text in enumerate(texts, 1):
            print(f"处理第 {i}/{len(texts)} 个文本...")
            result = extractor.extract_triplets(text)
            results.append({
                "input": text,
                "output": result
            })
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"结果已保存到: {output_file}")
        
        return results
        
    except Exception as e:
        print(f"批量推理失败: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        # 批量推理模式
        test_texts = [
            "马云是阿里巴巴的创始人，阿里巴巴总部位于杭州。",
            "张三毕业于清华大学，现在在腾讯担任CTO职位。",
            "李四是小米的产品经理，小米总部在北京。"
        ]
        batch_inference(test_texts, "batch_results.json")
    else:
        # 交互式推理模式
        success = main()
        sys.exit(0 if success else 1)