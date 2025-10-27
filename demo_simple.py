#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChatGLM-6B QLoRA三元组抽取简单演示
不需要实际模型，展示数据处理和训练流程
"""

import json
import os
from pathlib import Path

def demo_data_processing():
    """演示数据处理流程"""
    print("=" * 60)
    print("数据处理演示")
    print("=" * 60)
    
    # 检查训练数据
    train_file = Path("data/train_triplet.jsonl")
    val_file = Path("data/val_triplet.jsonl")
    
    if not train_file.exists() or not val_file.exists():
        print("训练数据不存在，正在生成...")
        os.system("cd data && python generate_triplet_data.py")
    
    # 读取并展示数据样本
    print(f"\n训练数据文件: {train_file}")
    print(f"验证数据文件: {val_file}")
    
    # 统计数据
    train_count = 0
    val_count = 0
    
    if train_file.exists():
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    train_count += 1
    
    if val_file.exists():
        with open(val_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    val_count += 1
    
    print(f"\n数据统计:")
    print(f"训练样本数: {train_count}")
    print(f"验证样本数: {val_count}")
    print(f"总样本数: {train_count + val_count}")
    
    # 展示几个样本
    print(f"\n训练数据样本:")
    print("-" * 40)
    
    if train_file.exists():
        with open(train_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 3:  # 只显示前3个样本
                    break
                if line.strip():
                    data = json.loads(line)
                    print(f"\n样本 {i+1}:")
                    print(f"输入: {data['input']}")
                    print(f"输出: {data['output']}")

def demo_training_config():
    """演示训练配置"""
    print("\n" + "=" * 60)
    print("训练配置演示")
    print("=" * 60)
    
    config = {
        "模型": "THUDM/chatglm-6b",
        "训练方法": "QLoRA (4-bit量化)",
        "LoRA参数": {
            "r": 8,
            "alpha": 32,
            "dropout": 0.1,
            "target_modules": ["query_key_value"]
        },
        "训练参数": {
            "epochs": 3,
            "batch_size": 4,
            "learning_rate": 2e-4,
            "warmup_steps": 100,
            "max_length": 512
        },
        "量化配置": {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4"
        }
    }
    
    def print_config(cfg, indent=0):
        for key, value in cfg.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                print_config(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")
    
    print_config(config)

def demo_inference_format():
    """演示推理格式"""
    print("\n" + "=" * 60)
    print("推理格式演示")
    print("=" * 60)
    
    examples = [
        {
            "input": "马云是阿里巴巴的创始人，阿里巴巴总部位于杭州。",
            "expected": "(马云, 创立, 阿里巴巴)\n(阿里巴巴, 总部在, 杭州)"
        },
        {
            "input": "张三毕业于清华大学，现在在腾讯担任CTO职位。",
            "expected": "(张三, 毕业于, 清华大学)\n(张三, 工作于, 腾讯)\n(张三, 担任, CTO)"
        },
        {
            "input": "李四是小米的产品经理，小米总部在北京。",
            "expected": "(李四, 工作于, 小米)\n(李四, 担任, 产品经理)\n(小米, 总部在, 北京)"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n示例 {i}:")
        print(f"输入文本: {example['input']}")
        print(f"期望输出: {example['expected']}")
        print("-" * 40)

def demo_evaluation_metrics():
    """演示评估指标"""
    print("\n" + "=" * 60)
    print("评估指标演示")
    print("=" * 60)
    
    # 模拟评估结果
    metrics = {
        "precision": 0.85,
        "recall": 0.82,
        "f1_score": 0.835,
        "exact_match_accuracy": 0.78,
        "total_samples": 101,
        "total_predicted": 245,
        "total_ground_truth": 238,
        "total_correct": 208
    }
    
    print("评估指标说明:")
    print("- 精确率 (Precision): 预测正确的三元组 / 所有预测的三元组")
    print("- 召回率 (Recall): 预测正确的三元组 / 所有真实的三元组")
    print("- F1分数: 精确率和召回率的调和平均数")
    print("- 完全匹配准确率: 完全正确预测所有三元组的样本比例")
    
    print(f"\n模拟评估结果:")
    print(f"精确率: {metrics['precision']:.3f}")
    print(f"召回率: {metrics['recall']:.3f}")
    print(f"F1分数: {metrics['f1_score']:.3f}")
    print(f"完全匹配准确率: {metrics['exact_match_accuracy']:.3f}")
    
    print(f"\n详细统计:")
    print(f"测试样本数: {metrics['total_samples']}")
    print(f"预测三元组总数: {metrics['total_predicted']}")
    print(f"真实三元组总数: {metrics['total_ground_truth']}")
    print(f"正确预测数: {metrics['total_correct']}")

def demo_usage_instructions():
    """演示使用说明"""
    print("\n" + "=" * 60)
    print("使用说明")
    print("=" * 60)
    
    instructions = [
        ("1. 安装依赖", "pip install -r requirements.txt"),
        ("2. 生成训练数据", "cd data && python generate_triplet_data.py"),
        ("3. 开始训练 (纯Python)", "python train_triplet.py"),
        ("4. 开始训练 (Shell脚本)", "bash scripts/train.sh"),
        ("5. 交互式推理", "python inference_triplet.py"),
        ("6. 批量推理", "python inference_triplet.py batch"),
        ("7. 模型评估", "python evaluate_triplet.py"),
        ("8. 快速测试", "python evaluate_triplet.py quick")
    ]
    
    for step, command in instructions:
        print(f"{step}:")
        print(f"  {command}")
        print()

def main():
    """主函数"""
    print("ChatGLM-6B QLoRA 三元组抽取完整演示")
    
    # 演示各个组件
    demo_data_processing()
    demo_training_config()
    demo_inference_format()
    demo_evaluation_metrics()
    demo_usage_instructions()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    print("\n项目特点:")
    print("✓ 完整的QLoRA微调实现")
    print("✓ 自动生成训练数据")
    print("✓ 支持多种关系类型")
    print("✓ 完善的评估指标")
    print("✓ 纯Python实现，易于使用")
    print("✓ 详细的文档和示例")
    
    print("\n注意事项:")
    print("• 需要GPU环境进行训练")
    print("• 首次运行会下载ChatGLM-6B模型（约12GB）")
    print("• 建议使用24GB+显存的GPU")
    print("• 训练时间约1-2小时（V100）")

if __name__ == "__main__":
    main()