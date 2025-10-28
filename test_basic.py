#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChatGLM-6B QLoRA 三元组抽取基础测试脚本
"""

import os
import sys
import json

def test_environment():
    """测试运行环境"""
    print("🔍 测试运行环境...")
    
    # 检查Python版本
    print(f"Python版本: {sys.version}")
    
    # 检查PyTorch
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    except ImportError:
        print("❌ PyTorch未安装")
        return False
    
    # 检查transformers
    try:
        import transformers
        print(f"Transformers版本: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers未安装")
        return False
    
    # 检查peft
    try:
        import peft
        print(f"PEFT版本: {peft.__version__}")
    except ImportError:
        print("❌ PEFT未安装")
        return False
    
    print("✅ 环境检查通过")
    return True

def test_config():
    """测试配置"""
    print("\n🔍 测试配置...")
    
    try:
        from glm_config import ProjectConfig
        pc = ProjectConfig()
        print(f"设备: {pc.device}")
        print(f"预训练模型: {pc.pre_model}")
        print(f"训练数据路径: {pc.train_path}")
        print(f"验证数据路径: {pc.dev_path}")
        print(f"输出目录: {pc.save_dir}")
        print(f"批次大小: {pc.batch_size}")
        print(f"梯度累积步数: {pc.gradient_accumulation_steps}")
        print(f"有效批次大小: {pc.batch_size * pc.gradient_accumulation_steps}")
        print(f"序列长度: {pc.max_source_seq_len} + {pc.max_target_seq_len} = {pc.max_seq_length}")
        print(f"LoRA rank: {pc.lora_rank}")
        print(f"混合精度: {pc.fp16}")
        
        print("✅ 配置检查通过")
        return True
        
    except Exception as e:
        print(f"❌ 配置检查失败: {e}")
        return False

def test_data_format():
    """测试数据格式"""
    print("\n🔍 测试数据格式...")
    
    try:
        from glm_config import ProjectConfig
        pc = ProjectConfig()
        
        # 检查数据文件是否存在
        if not os.path.exists(pc.train_path):
            print(f"❌ 训练数据文件不存在: {pc.train_path}")
            return False
        if not os.path.exists(pc.dev_path):
            print(f"❌ 验证数据文件不存在: {pc.dev_path}")
            return False
        
        # 读取几个样本进行测试
        with open(pc.train_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:3]  # 只测试前3个样本
        
        valid_count = 0
        for i, line in enumerate(lines):
            try:
                data = json.loads(line.strip())
                context = data.get('context', '')
                target = data.get('target', '')
                
                print(f"样本 {i+1}:")
                print(f"  输入长度: {len(context)}")
                print(f"  输出长度: {len(target)}")
                
                # 简单验证格式
                if context and target and 'json' in target:
                    valid_count += 1
                    print(f"  格式: ✅ 有效")
                else:
                    print(f"  格式: ❌ 无效")
                    
            except Exception as e:
                print(f"  解析失败: {e}")
        
        print(f"\n有效样本: {valid_count}/{len(lines)}")
        
        if valid_count == 0:
            print("❌ 没有有效的样本")
            return False
        
        print("✅ 数据格式检查通过")
        return True
        
    except Exception as e:
        print(f"❌ 数据格式检查失败: {e}")
        return False

def test_imports():
    """测试模块导入"""
    print("\n🔍 测试模块导入...")
    
    try:
        # 测试配置导入
        from glm_config import ProjectConfig
        print("✅ 配置模块导入成功")
        
        # 测试工具模块导入
        from utils.common_utils import CastOutputToFloat, save_model
        print("✅ 工具模块导入成功")
        
        print("✅ 模块导入检查通过")
        return True
        
    except Exception as e:
        print(f"❌ 模块导入检查失败: {e}")
        return False

def test_model_path():
    """测试模型路径"""
    print("\n🤖 测试模型路径...")
    
    try:
        from glm_config import ProjectConfig
        pc = ProjectConfig()
        
        print(f"模型路径: {pc.pre_model}")
        
        # 检查模型路径是否存在
        if not os.path.exists(pc.pre_model):
            print(f"❌ 模型路径不存在: {pc.pre_model}")
            print("💡 请确认:")
            print("   1. 模型已下载到指定路径")
            print("   2. 或修改 glm_config.py 中的 self.pre_model 路径")
            print("   3. 或使用在线模型: 'THUDM/chatglm-6b'")
            return False
        
        # 检查关键文件
        required_files = ['config.json']
        for file in required_files:
            file_path = os.path.join(pc.pre_model, file)
            if not os.path.exists(file_path):
                print(f"❌ 缺少关键文件: {file}")
                return False
        
        print("✅ 模型路径检查通过")
        return True
        
    except Exception as e:
        print(f"❌ 模型路径检查失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 ChatGLM-6B QLoRA 三元组抽取基础测试")
    print("=" * 60)
    
    tests = [
        ("环境检查", test_environment),
        ("配置检查", test_config),
        ("模块导入检查", test_imports),
        ("模型路径检查", test_model_path),
        ("数据格式检查", test_data_format),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 通过")
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
    
    print(f"\n{'='*60}")
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有基础测试通过！")
        print("\n📝 下一步:")
        print("1. 运行 python train.py 开始训练")
        print("2. 使用 nvidia-smi 监控GPU使用情况")
        print("3. 训练完成后使用 python inference_triplet.py --interactive 进行推理测试")
        return True
    else:
        print("⚠️ 部分测试失败，请检查配置和环境。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)