#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试训练环境设置
验证是否可以开始训练
"""

import os
import sys
import torch

def test_training_imports():
    """测试训练所需的模块导入"""
    print("🚀 测试训练模块导入")
    print("=" * 50)
    
    try:
        # 测试基础模块
        from glm_config import ProjectConfig
        print("✅ 配置模块导入成功")
        
        # 测试transformers模块
        from transformers import (
            AutoTokenizer, 
            AutoModel, 
            TrainingArguments, 
            Trainer,
            DataCollatorForSeq2Seq
        )
        print("✅ Transformers模块导入成功")
        
        # 测试PEFT模块
        from peft import (
            get_peft_model, 
            LoraConfig, 
            TaskType,
            PeftModel
        )
        print("✅ PEFT模块导入成功")
        
        # 测试数据处理模块
        import json
        from datasets import Dataset
        print("✅ 数据处理模块导入成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def test_config_loading():
    """测试配置加载"""
    print("\n🔧 测试配置加载")
    print("-" * 30)
    
    try:
        from glm_config import ProjectConfig
        pc = ProjectConfig()
        
        print(f"📋 模型路径: {pc.pre_model}")
        print(f"📋 训练数据: {pc.train_path}")
        print(f"📋 验证数据: {pc.dev_path}")
        print(f"📋 输出目录: {pc.save_dir}")
        print(f"📋 批次大小: {pc.batch_size}")
        print(f"📋 梯度累积: {pc.gradient_accumulation_steps}")
        print(f"📋 学习率: {pc.learning_rate}")
        print(f"📋 训练轮数: {pc.epochs}")
        
        # 检查数据文件
        if os.path.exists(pc.train_path):
            print("✅ 训练数据文件存在")
        else:
            print("❌ 训练数据文件不存在")
            return False
            
        if os.path.exists(pc.dev_path):
            print("✅ 验证数据文件存在")
        else:
            print("❌ 验证数据文件不存在")
            return False
        
        # 检查输出目录
        os.makedirs(pc.save_dir, exist_ok=True)
        print("✅ 输出目录已准备")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False

def test_data_loading():
    """测试数据加载"""
    print("\n📊 测试数据加载")
    print("-" * 30)
    
    try:
        from glm_config import ProjectConfig
        pc = ProjectConfig()
        
        # 读取训练数据
        with open(pc.train_path, 'r', encoding='utf-8') as f:
            train_lines = f.readlines()
        
        print(f"📈 训练样本数量: {len(train_lines)}")
        
        # 读取验证数据
        with open(pc.dev_path, 'r', encoding='utf-8') as f:
            dev_lines = f.readlines()
        
        print(f"📊 验证样本数量: {len(dev_lines)}")
        
        # 检查数据格式
        import json
        sample = json.loads(train_lines[0])
        
        if 'context' in sample and 'target' in sample:
            print("✅ 数据格式正确")
            print(f"📝 样本示例:")
            print(f"   输入: {sample['context'][:50]}...")
            print(f"   输出: {sample['target'][:50]}...")
        else:
            print("❌ 数据格式错误")
            print(f"   实际字段: {list(sample.keys())}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return False

def test_device_setup():
    """测试设备设置"""
    print("\n🖥️ 测试设备设置")
    print("-" * 30)
    
    try:
        print(f"📋 PyTorch版本: {torch.__version__}")
        print(f"📋 CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"📋 CUDA版本: {torch.version.cuda}")
            print(f"📋 GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"📋 GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("⚠️ CUDA不可用，将使用CPU训练（速度较慢）")
        
        return True
        
    except Exception as e:
        print(f"❌ 设备检查失败: {e}")
        return False

def test_model_config_only():
    """仅测试模型配置（不加载权重）"""
    print("\n🤖 测试模型配置")
    print("-" * 30)
    
    try:
        from transformers import AutoConfig
        from glm_config import ProjectConfig
        
        pc = ProjectConfig()
        
        print("🔄 加载模型配置...")
        config = AutoConfig.from_pretrained(
            pc.pre_model, 
            trust_remote_code=True
        )
        
        print("✅ 模型配置加载成功")
        print(f"📊 模型类型: {config.model_type}")
        print(f"📊 词汇表大小: {config.vocab_size}")
        print(f"📊 隐藏层大小: {config.hidden_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型配置加载失败: {e}")
        return False

def main():
    """主函数"""
    tests = [
        ("模块导入", test_training_imports),
        ("配置加载", test_config_loading),
        ("数据加载", test_data_loading),
        ("设备设置", test_device_setup),
        ("模型配置", test_model_config_only),
    ]
    
    success_count = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        if test_func():
            print(f"✅ {test_name} 通过")
            success_count += 1
        else:
            print(f"❌ {test_name} 失败")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {success_count}/{total_tests} 通过")
    
    if success_count == total_tests:
        print("🎉 训练环境设置完成！")
        print("\n📝 下一步:")
        print("1. 运行 python train.py 开始训练")
        print("2. 使用 nvidia-smi 监控GPU使用情况")
        print("3. 训练过程中可以使用 Ctrl+C 安全停止")
        return 0
    else:
        print("⚠️ 部分测试失败，请解决问题后再开始训练")
        return 1

if __name__ == "__main__":
    sys.exit(main())