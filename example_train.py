#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChatGLM-6B QLoRA训练示例
展示如何使用纯Python进行LoRA微调
"""

import os
import sys
from pathlib import Path

def example_basic_training():
    """基础训练示例"""
    print("=" * 60)
    print("示例1: 基础训练")
    print("=" * 60)
    
    # 导入训练模块
    from train_pure_python import main as train_main
    
    print("使用默认配置进行训练...")
    print("配置:")
    print("  - 模型: THUDM/chatglm-6b")
    print("  - 训练轮数: 3")
    print("  - 批次大小: 4")
    print("  - LoRA rank: 8")
    print("  - 学习率: 2e-4")
    
    # 开始训练
    success = train_main()
    
    if success:
        print("✅ 基础训练完成！")
    else:
        print("❌ 基础训练失败！")
    
    return success

def example_custom_config():
    """自定义配置训练示例"""
    print("\n" + "=" * 60)
    print("示例2: 自定义配置训练")
    print("=" * 60)
    
    # 设置命令行参数
    sys.argv = [
        'train_with_config.py',
        '--config', 'medium',  # 使用中等显存配置
        '--epochs', '2',       # 训练2轮
        '--batch_size', '2',   # 批次大小2
        '--lora_r', '16'       # LoRA rank 16
    ]
    
    from train_with_config import main as config_train_main
    
    print("使用中等显存配置进行训练...")
    print("配置:")
    print("  - 配置类型: medium")
    print("  - 训练轮数: 2")
    print("  - 批次大小: 2")
    print("  - LoRA rank: 16")
    
    # 开始训练
    success = config_train_main()
    
    if success:
        print("✅ 自定义配置训练完成！")
    else:
        print("❌ 自定义配置训练失败！")
    
    return success

def example_fast_test():
    """快速测试训练示例"""
    print("\n" + "=" * 60)
    print("示例3: 快速测试训练")
    print("=" * 60)
    
    # 设置命令行参数
    sys.argv = [
        'train_with_config.py',
        '--config', 'fast',    # 使用快速配置
        '--epochs', '1',       # 只训练1轮
        '--batch_size', '1'    # 最小批次大小
    ]
    
    from train_with_config import main as config_train_main
    
    print("使用快速测试配置进行训练...")
    print("配置:")
    print("  - 配置类型: fast")
    print("  - 训练轮数: 1")
    print("  - 批次大小: 1")
    print("  - 用途: 快速验证代码和环境")
    
    # 开始训练
    success = config_train_main()
    
    if success:
        print("✅ 快速测试训练完成！")
    else:
        print("❌ 快速测试训练失败！")
    
    return success

def example_programmatic_config():
    """编程式配置示例"""
    print("\n" + "=" * 60)
    print("示例4: 编程式配置")
    print("=" * 60)
    
    # 导入配置类
    sys.path.insert(0, str(Path(__file__).parent / "config"))
    from train_config_simple import TrainingConfig
    from train_with_config import ChatGLMTrainer
    
    # 创建自定义配置
    config = TrainingConfig()
    
    # 自定义参数
    config.NUM_EPOCHS = 1
    config.BATCH_SIZE = 2
    config.LORA_R = 4
    config.LORA_ALPHA = 16
    config.OUTPUT_DIR = "output/custom-triplet-model"
    config.LEARNING_RATE = 1e-4
    
    print("使用编程式自定义配置...")
    print("配置:")
    print(f"  - 训练轮数: {config.NUM_EPOCHS}")
    print(f"  - 批次大小: {config.BATCH_SIZE}")
    print(f"  - LoRA rank: {config.LORA_R}")
    print(f"  - 学习率: {config.LEARNING_RATE}")
    print(f"  - 输出目录: {config.OUTPUT_DIR}")
    
    try:
        # 创建训练器
        trainer = ChatGLMTrainer(config)
        
        # 设置模型和分词器
        trainer.setup_model_and_tokenizer()
        
        # 准备数据集
        trainer.prepare_datasets()
        
        # 开始训练
        success = trainer.train()
        
        if success:
            print("✅ 编程式配置训练完成！")
        else:
            print("❌ 编程式配置训练失败！")
        
        return success
        
    except Exception as e:
        print(f"❌ 编程式配置训练出错: {e}")
        return False

def check_prerequisites():
    """检查前置条件"""
    print("检查前置条件...")
    
    # 检查数据文件
    train_file = Path("data/train_triplet.jsonl")
    val_file = Path("data/val_triplet.jsonl")
    
    if not train_file.exists() or not val_file.exists():
        print("❌ 训练数据不存在")
        print("正在生成训练数据...")
        
        # 生成数据
        os.system("cd data && python generate_triplet_data.py")
        
        if train_file.exists() and val_file.exists():
            print("✅ 训练数据生成成功")
        else:
            print("❌ 训练数据生成失败")
            return False
    else:
        print("✅ 训练数据已存在")
    
    # 检查依赖
    try:
        import torch
        import transformers
        import peft
        import bitsandbytes
        print("✅ 所有依赖已安装")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    # 检查GPU
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU可用: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("⚠️  未检测到GPU，将使用CPU训练（非常慢）")
    
    return True

def main():
    """主函数"""
    print("ChatGLM-6B QLoRA训练示例")
    print("=" * 80)
    
    # 检查前置条件
    if not check_prerequisites():
        print("❌ 前置条件检查失败，请解决问题后重试")
        return
    
    print("\n可用的训练示例:")
    print("1. 基础训练 (默认配置)")
    print("2. 自定义配置训练")
    print("3. 快速测试训练")
    print("4. 编程式配置训练")
    print("5. 全部运行")
    
    choice = input("\n请选择要运行的示例 (1-5): ").strip()
    
    if choice == "1":
        example_basic_training()
    elif choice == "2":
        example_custom_config()
    elif choice == "3":
        example_fast_test()
    elif choice == "4":
        example_programmatic_config()
    elif choice == "5":
        print("运行所有示例...")
        example_basic_training()
        example_custom_config()
        example_fast_test()
        example_programmatic_config()
    else:
        print("无效选择，运行快速测试...")
        example_fast_test()
    
    print("\n" + "=" * 80)
    print("示例运行完成！")
    print("\n接下来可以:")
    print("1. 查看训练结果: ls output/")
    print("2. 运行推理测试: python inference_triplet.py")
    print("3. 评估模型性能: python evaluate_triplet.py")
    print("=" * 80)

if __name__ == "__main__":
    main()