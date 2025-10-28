#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯Python演示脚本 - 展示如何使用训练好的模型
无需shell脚本，直接在Python中加载和使用模型
"""

import os
import json
from pathlib import Path
from typing import List, Tuple

def demo_without_torch():
    """不依赖torch的演示 - 展示项目结构和配置"""
    print("=" * 70)
    print("ChatGLM-6B QLoRA 三元组抽取 - 纯Python演示")
    print("=" * 70)
    
    print("\n📁 项目结构:")
    project_files = [
        "train_pure_python.py - 基础训练脚本",
        "train_with_config.py - 配置化训练脚本", 
        "example_train.py - 示例训练脚本",
        "inference_triplet.py - 推理脚本",
        "evaluate_triplet.py - 评估脚本",
        "config/train_config_simple.py - 训练配置",
        "data/train_triplet.jsonl - 训练数据",
        "data/val_triplet.jsonl - 验证数据"
    ]
    
    for file_desc in project_files:
        filename = file_desc.split(" - ")[0]
        desc = file_desc.split(" - ")[1]
        if Path(filename).exists():
            print(f"  ✅ {filename} - {desc}")
        else:
            print(f"  ❌ {filename} - {desc} (不存在)")
    
    print("\n🔧 可用的训练配置:")
    configs = [
        ("small", "小显存GPU (8-12GB)", "batch_size=1, lora_r=4"),
        ("medium", "中等显存GPU (12-24GB)", "batch_size=4, lora_r=8"),
        ("large", "大显存GPU (24GB+)", "batch_size=8, lora_r=16"),
        ("fast", "快速测试", "epochs=1, batch_size=2"),
        ("auto", "自动检测", "根据GPU自动选择")
    ]
    
    for config_name, desc, params in configs:
        print(f"  🎯 {config_name:6} - {desc:20} ({params})")
    
    print("\n📊 训练数据统计:")
    train_file = Path("data/train_triplet.jsonl")
    val_file = Path("data/val_triplet.jsonl")
    
    if train_file.exists() and val_file.exists():
        with open(train_file, 'r', encoding='utf-8') as f:
            train_count = sum(1 for line in f if line.strip())
        
        with open(val_file, 'r', encoding='utf-8') as f:
            val_count = sum(1 for line in f if line.strip())
        
        print(f"  📈 训练样本: {train_count} 条")
        print(f"  📊 验证样本: {val_count} 条")
        
        # 显示几个样本
        print("\n📝 训练数据示例:")
        with open(train_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 3:  # 只显示前3个
                    break
                if line.strip():
                    data = json.loads(line.strip())
                    print(f"  样本 {i+1}:")
                    print(f"    输入: {data.get('input', '')[:50]}...")
                    print(f"    输出: {data.get('output', '')[:50]}...")
    else:
        print("  ❌ 训练数据不存在，请运行: cd data && python generate_triplet_data.py")
    
    print("\n🚀 使用方法:")
    usage_examples = [
        "基础训练: python train_pure_python.py",
        "自动配置: python train_with_config.py --config auto",
        "小显存GPU: python train_with_config.py --config small",
        "快速测试: python train_with_config.py --config fast",
        "自定义参数: python train_with_config.py --config medium --epochs 2 --batch_size 2",
        "交互式训练: python example_train.py"
    ]
    
    for i, example in enumerate(usage_examples, 1):
        print(f"  {i}. {example}")
    
    print("\n💡 训练建议:")
    tips = [
        "新手用户: 使用 train_pure_python.py 开始",
        "进阶用户: 使用 train_with_config.py 进行调参",
        "显存不足: 使用 --config small 或减小 batch_size",
        "快速验证: 使用 --config fast 进行测试",
        "自动选择: 使用 --config auto 让程序自动检测GPU"
    ]
    
    for tip in tips:
        print(f"  💡 {tip}")

def demo_model_usage_code():
    """展示模型使用代码示例"""
    print("\n" + "=" * 70)
    print("模型使用代码示例")
    print("=" * 70)
    
    print("\n🔍 1. 加载训练好的模型:")
    print("""
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

# 加载基础模型
base_model = AutoModel.from_pretrained(
    "THUDM/chatglm-6b", 
    trust_remote_code=True,
    device_map="auto"
)

# 加载LoRA适配器
model = PeftModel.from_pretrained(
    base_model, 
    "output/chatglm-6b-triplet-qlora"
)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    "THUDM/chatglm-6b", 
    trust_remote_code=True
)
""")
    
    print("\n🎯 2. 进行三元组抽取:")
    print("""
def extract_triplets(text):
    prompt = f"请从以下文本中抽取三元组，格式为(主体, 关系, 客体):\\n{text}"
    
    response, history = model.chat(
        tokenizer, 
        prompt, 
        history=[]
    )
    
    return response

# 使用示例
text = "苹果公司是一家美国的科技公司，总部位于加利福尼亚州。"
result = extract_triplets(text)
print(result)
# 输出: (苹果公司, 是, 科技公司), (苹果公司, 总部位于, 加利福尼亚州)
""")
    
    print("\n📊 3. 批量处理:")
    print("""
def batch_extract(texts):
    results = []
    for text in texts:
        result = extract_triplets(text)
        results.append({
            'input': text,
            'output': result
        })
    return results

# 批量处理示例
texts = [
    "北京是中国的首都。",
    "张三在清华大学学习计算机科学。",
    "特斯拉公司生产电动汽车。"
]

results = batch_extract(texts)
for result in results:
    print(f"输入: {result['input']}")
    print(f"输出: {result['output']}")
    print("-" * 50)
""")

def demo_training_process():
    """展示训练过程"""
    print("\n" + "=" * 70)
    print("训练过程演示")
    print("=" * 70)
    
    print("\n🔄 训练流程:")
    steps = [
        "1. 数据准备 - 生成三元组抽取训练数据",
        "2. 模型加载 - 加载ChatGLM-6B基础模型",
        "3. 量化配置 - 设置4bit量化以节省显存",
        "4. LoRA配置 - 设置低秩适应参数",
        "5. 数据处理 - 将数据转换为模型输入格式",
        "6. 训练执行 - 使用Trainer进行训练",
        "7. 模型保存 - 保存LoRA适配器权重"
    ]
    
    for step in steps:
        print(f"  {step}")
    
    print("\n⚙️ 关键配置参数:")
    params = [
        ("LoRA Rank", "8", "控制适配器参数量，越大容量越大"),
        ("LoRA Alpha", "32", "缩放因子，通常为rank的2-4倍"),
        ("Batch Size", "4", "批次大小，根据显存调整"),
        ("Learning Rate", "2e-4", "学习率，LoRA通常用较大值"),
        ("Epochs", "3", "训练轮数，避免过拟合"),
        ("Max Length", "512", "最大序列长度")
    ]
    
    for param, value, desc in params:
        print(f"  {param:15}: {value:8} - {desc}")
    
    print("\n📈 训练监控:")
    monitoring = [
        "训练损失 (Training Loss) - 应该逐渐下降",
        "验证损失 (Validation Loss) - 用于判断过拟合",
        "学习率调度 - 预热后逐渐衰减",
        "显存使用 - 监控是否超出限制",
        "训练速度 - 每步训练时间"
    ]
    
    for item in monitoring:
        print(f"  📊 {item}")

def main():
    """主演示函数"""
    demo_without_torch()
    demo_model_usage_code()
    demo_training_process()
    
    print("\n" + "=" * 70)
    print("🎉 演示完成！")
    print("\n下一步:")
    print("1. 安装依赖: pip install -r requirements.txt")
    print("2. 生成数据: cd data && python generate_triplet_data.py")
    print("3. 开始训练: python train_with_config.py --config auto")
    print("4. 测试模型: python inference_triplet.py")
    print("\n详细文档: PYTHON_TRAINING_GUIDE.md")
    print("快速开始: QUICK_START.md")
    print("=" * 70)

if __name__ == "__main__":
    main()