#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试纯Python训练脚本的结构和配置
不依赖torch等深度学习库
"""

import sys
import json
from pathlib import Path

def test_config_loading():
    """测试配置加载"""
    print("测试配置加载...")
    
    try:
        # 添加config目录到路径
        config_path = Path(__file__).parent / "config"
        sys.path.insert(0, str(config_path))
        
        from train_config_simple import (
            TrainingConfig,
            SmallGPUConfig,
            MediumGPUConfig,
            LargeGPUConfig,
            FastTrainingConfig
        )
        
        # 测试默认配置
        config = TrainingConfig()
        print(f"✓ 默认配置加载成功")
        print(f"  模型: {config.MODEL_NAME}")
        print(f"  批次大小: {config.BATCH_SIZE}")
        print(f"  LoRA rank: {config.LORA_R}")
        
        # 测试预设配置
        small_config = SmallGPUConfig()
        print(f"✓ 小显存配置: batch_size={small_config.BATCH_SIZE}, lora_r={small_config.LORA_R}")
        
        medium_config = MediumGPUConfig()
        print(f"✓ 中等显存配置: batch_size={medium_config.BATCH_SIZE}, lora_r={medium_config.LORA_R}")
        
        large_config = LargeGPUConfig()
        print(f"✓ 大显存配置: batch_size={large_config.BATCH_SIZE}, lora_r={large_config.LORA_R}")
        
        fast_config = FastTrainingConfig()
        print(f"✓ 快速配置: epochs={fast_config.NUM_EPOCHS}, batch_size={fast_config.BATCH_SIZE}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False

def test_data_files():
    """测试数据文件"""
    print("\n测试数据文件...")
    
    train_file = Path("data/train_triplet.jsonl")
    val_file = Path("data/val_triplet.jsonl")
    
    if not train_file.exists():
        print(f"❌ 训练文件不存在: {train_file}")
        return False
    
    if not val_file.exists():
        print(f"❌ 验证文件不存在: {val_file}")
        return False
    
    try:
        # 检查文件格式
        with open(train_file, 'r', encoding='utf-8') as f:
            train_count = 0
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    # 检查必要字段
                    if 'instruction' in data and 'input' in data and 'output' in data:
                        train_count += 1
                    else:
                        print(f"❌ 训练数据格式错误: 缺少必要字段")
                        return False
        
        with open(val_file, 'r', encoding='utf-8') as f:
            val_count = 0
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    val_count += 1
        
        print(f"✓ 训练数据: {train_count} 条")
        print(f"✓ 验证数据: {val_count} 条")
        
        if train_count == 0 or val_count == 0:
            print("❌ 数据文件为空")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 数据文件检查失败: {e}")
        return False

def test_script_structure():
    """测试脚本结构"""
    print("\n测试脚本结构...")
    
    scripts = [
        "train_pure_python.py",
        "train_with_config.py",
        "example_train.py"
    ]
    
    for script in scripts:
        script_path = Path(script)
        if not script_path.exists():
            print(f"❌ 脚本不存在: {script}")
            return False
        
        # 检查脚本内容
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查关键类和函数
            if script == "train_pure_python.py":
                required = ["TrainingConfig", "TripletDataset", "ChatGLMTrainer", "main"]
            elif script == "train_with_config.py":
                required = ["TripletDataset", "ChatGLMTrainer", "detect_gpu_config", "main"]
            else:
                required = ["main"]
            
            for req in required:
                if req not in content:
                    print(f"❌ {script} 缺少必要组件: {req}")
                    return False
            
            print(f"✓ {script} 结构正确")
            
        except Exception as e:
            print(f"❌ 检查 {script} 失败: {e}")
            return False
    
    return True

def test_import_structure():
    """测试导入结构 (不实际导入torch)"""
    print("\n测试导入结构...")
    
    # 检查train_pure_python.py的导入
    try:
        with open("train_pure_python.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_imports = [
            "import torch",
            "from transformers import",
            "from peft import",
            "import bitsandbytes",
            "BitsAndBytesConfig"
        ]
        
        for imp in required_imports:
            if imp not in content:
                print(f"❌ train_pure_python.py 缺少导入: {imp}")
                return False
        
        print("✓ train_pure_python.py 导入结构正确")
        
        # 检查关键类定义
        key_classes = [
            "class TrainingConfig",
            "class TripletDataset",
            "class ChatGLMTrainer"
        ]
        
        for cls in key_classes:
            if cls not in content:
                print(f"❌ train_pure_python.py 缺少类定义: {cls}")
                return False
        
        print("✓ train_pure_python.py 类定义正确")
        
        return True
        
    except Exception as e:
        print(f"❌ 检查导入结构失败: {e}")
        return False

def test_configuration_examples():
    """测试配置示例"""
    print("\n测试配置示例...")
    
    try:
        # 添加config目录到路径
        config_path = Path(__file__).parent / "config"
        sys.path.insert(0, str(config_path))
        
        from train_config_simple import TrainingConfig
        
        # 测试自定义配置
        config = TrainingConfig()
        
        # 模拟不同的配置修改
        test_configs = [
            {"BATCH_SIZE": 1, "LORA_R": 4, "name": "小显存配置"},
            {"BATCH_SIZE": 8, "LORA_R": 16, "name": "大显存配置"},
            {"NUM_EPOCHS": 1, "BATCH_SIZE": 2, "name": "快速测试配置"}
        ]
        
        for test_config in test_configs:
            config = TrainingConfig()
            for key, value in test_config.items():
                if key != "name":
                    setattr(config, key, value)
            
            print(f"✓ {test_config['name']}: batch_size={config.BATCH_SIZE}, lora_r={config.LORA_R}, epochs={config.NUM_EPOCHS}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置示例测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 70)
    print("ChatGLM-6B 纯Python训练脚本测试")
    print("=" * 70)
    
    tests = [
        ("配置加载", test_config_loading),
        ("数据文件", test_data_files),
        ("脚本结构", test_script_structure),
        ("导入结构", test_import_structure),
        ("配置示例", test_configuration_examples)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} 测试通过")
        else:
            print(f"❌ {test_name} 测试失败")
    
    print("\n" + "=" * 70)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！纯Python训练脚本准备就绪。")
        print("\n使用方法:")
        print("1. 基础训练: python train_pure_python.py")
        print("2. 配置训练: python train_with_config.py --config auto")
        print("3. 示例训练: python example_train.py")
        print("\n注意: 需要先安装依赖 (pip install -r requirements.txt)")
    else:
        print("❌ 部分测试失败，请检查相关文件。")
    
    print("=" * 70)

if __name__ == "__main__":
    main()