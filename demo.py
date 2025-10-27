#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChatGLM-6B三元组抽取演示脚本
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def demo_without_model():
    """在没有模型的情况下演示数据格式和处理逻辑"""
    print("=" * 60)
    print("ChatGLM-6B QLoRA 三元组抽取演示")
    print("=" * 60)
    
    # 演示数据格式
    print("\n1. 训练数据格式演示:")
    print("-" * 40)
    
    sample_data = {
        "instruction": "请从以下文本中抽取所有的三元组，格式为(主体, 关系, 客体)：",
        "input": "马云是阿里巴巴的创始人，阿里巴巴总部位于杭州。",
        "output": "(马云, 创立, 阿里巴巴)\n(阿里巴巴, 总部在, 杭州)"
    }
    
    print(f"指令: {sample_data['instruction']}")
    print(f"输入: {sample_data['input']}")
    print(f"期望输出: {sample_data['output']}")
    
    # 演示支持的关系类型
    print("\n2. 支持的关系类型:")
    print("-" * 40)
    
    relations = {
        "工作于": "人员与公司的雇佣关系",
        "担任": "人员与职位的关系", 
        "位于": "实体与地理位置的关系",
        "毕业于": "人员与教育机构的关系",
        "创立": "人员与公司的创建关系",
        "总部在": "公司与地理位置的关系",
        "任职": "人员与职位的关系"
    }
    
    for relation, description in relations.items():
        print(f"- {relation}: {description}")
    
    # 演示更多样本
    print("\n3. 更多训练样本示例:")
    print("-" * 40)
    
    examples = [
        {
            "input": "张三毕业于清华大学，现在在腾讯担任CTO职位。",
            "output": "(张三, 毕业于, 清华大学)\n(张三, 工作于, 腾讯)\n(张三, 担任, CTO)"
        },
        {
            "input": "李四是小米的产品经理，小米总部在北京。",
            "output": "(李四, 工作于, 小米)\n(李四, 担任, 产品经理)\n(小米, 总部在, 北京)"
        },
        {
            "input": "王五从北京大学毕业后，在字节跳动工作，担任技术总监。",
            "output": "(王五, 毕业于, 北京大学)\n(王五, 工作于, 字节跳动)\n(王五, 担任, 技术总监)"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n示例 {i}:")
        print(f"输入: {example['input']}")
        print(f"输出: {example['output']}")
    
    # 使用说明
    print("\n4. 使用说明:")
    print("-" * 40)
    print("要开始使用本项目，请按照以下步骤操作：")
    print()
    print("步骤1: 安装依赖")
    print("  pip install -r requirements.txt")
    print()
    print("步骤2: 生成训练数据")
    print("  cd data && python generate_triplet_data.py")
    print()
    print("步骤3: 开始训练")
    print("  bash scripts/train.sh")
    print()
    print("步骤4: 模型推理")
    print("  bash scripts/inference.sh")
    print()
    print("步骤5: 模型评估")
    print("  bash scripts/evaluate.sh")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)

if __name__ == "__main__":
    demo_without_model()