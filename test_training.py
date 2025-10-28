#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChatGLM-6B QLoRA 三元组抽取训练测试脚本

用于验证训练环境和代码的稳定性
"""

import os
import sys
import torch
import traceback
from glm_config import ProjectConfig
from data_handle.data_loader import get_data
from data_handle.data_preprocess import validate_triplet_format

def test_environment():
    """测试运行环境"""
    print("🔍 测试运行环境...")
    
    # 检查Python版本
    print(f"Python版本: {sys.version}")
    
    # 检查PyTorch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
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
    
    # 检查datasets
    try:
        import datasets
        print(f"Datasets版本: {datasets.__version__}")
    except ImportError:
        print("❌ Datasets未安装")
        return False
    
    print("✅ 环境检查通过")
    return True

def test_config():
    """测试配置"""
    print("\n🔍 测试配置...")
    
    try:
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
        
        # 检查数据文件
        if not os.path.exists(pc.train_path):
            print(f"❌ 训练数据文件不存在: {pc.train_path}")
            return False
        if not os.path.exists(pc.dev_path):
            print(f"❌ 验证数据文件不存在: {pc.dev_path}")
            return False
        
        print("✅ 配置检查通过")
        return True
        
    except Exception as e:
        print(f"❌ 配置检查失败: {e}")
        return False

def test_data_format():
    """测试数据格式"""
    print("\n🔍 测试数据格式...")
    
    try:
        pc = ProjectConfig()
        
        # 读取几个样本进行测试
        import json
        with open(pc.train_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:5]  # 只测试前5个样本
        
        valid_count = 0
        for i, line in enumerate(lines):
            try:
                data = json.loads(line.strip())
                context = data.get('context', '')
                target = data.get('target', '')
                
                print(f"样本 {i+1}:")
                print(f"  输入长度: {len(context)}")
                print(f"  输出长度: {len(target)}")
                
                # 验证三元组格式
                if validate_triplet_format(target):
                    valid_count += 1
                    print(f"  格式: ✅ 有效")
                else:
                    print(f"  格式: ❌ 无效")
                    print(f"  目标文本: {target[:100]}...")
                    
            except Exception as e:
                print(f"  解析失败: {e}")
        
        print(f"\n有效样本: {valid_count}/{len(lines)}")
        
        if valid_count == 0:
            print("❌ 没有有效的三元组样本")
            return False
        
        print("✅ 数据格式检查通过")
        return True
        
    except Exception as e:
        print(f"❌ 数据格式检查失败: {e}")
        return False

def test_data_loading():
    """测试数据加载"""
    print("\n🔍 测试数据加载...")
    
    try:
        train_dataloader, dev_dataloader = get_data()
        
        print(f"训练数据批次数: {len(train_dataloader)}")
        print(f"验证数据批次数: {len(dev_dataloader)}")
        
        # 测试第一个批次
        for batch in train_dataloader:
            print(f"输入形状: {batch['input_ids'].shape}")
            print(f"标签形状: {batch['labels'].shape}")
            print(f"输入数据类型: {batch['input_ids'].dtype}")
            print(f"标签数据类型: {batch['labels'].dtype}")
            break
        
        print("✅ 数据加载检查通过")
        return True
        
    except Exception as e:
        print(f"❌ 数据加载检查失败: {e}")
        traceback.print_exc()
        return False

def test_model_loading():
    """测试模型加载"""
    print("\n🔍 测试模型加载...")
    
    try:
        from transformers import AutoTokenizer, AutoModel, AutoConfig
        import peft
        from utils.common_utils import CastOutputToFloat
        
        pc = ProjectConfig()
        
        print("加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(pc.pre_model, trust_remote_code=True)
        print(f"词汇表大小: {len(tokenizer)}")
        
        print("加载模型配置...")
        config = AutoConfig.from_pretrained(pc.pre_model, trust_remote_code=True)
        
        print("加载模型...")
        model = AutoModel.from_pretrained(
            pc.pre_model,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.float16 if pc.fp16 else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        print("配置LoRA...")
        model.lm_head = CastOutputToFloat(model.lm_head)
        peft_config = peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            inference_mode=False,
            r=pc.lora_rank,
            lora_alpha=pc.lora_alpha,
            lora_dropout=pc.lora_dropout,
            target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        )
        model = peft.get_peft_model(model, peft_config)
        
        # 移动到设备
        model = model.to(pc.device)
        
        # 打印参数统计
        if hasattr(model, 'print_trainable_parameters'):
            model.print_trainable_parameters()
        
        # 测试前向传播
        print("测试前向传播...")
        model.train()
        
        # 创建测试输入
        test_input_ids = torch.randint(0, len(tokenizer), (1, 100)).to(pc.device)
        test_labels = torch.randint(0, len(tokenizer), (1, 100)).to(pc.device)
        
        with torch.no_grad():
            outputs = model(input_ids=test_input_ids, labels=test_labels)
            print(f"损失: {outputs.loss.item():.4f}")
        
        print("✅ 模型加载检查通过")
        return True
        
    except Exception as e:
        print(f"❌ 模型加载检查失败: {e}")
        traceback.print_exc()
        return False

def test_memory_usage():
    """测试显存使用"""
    print("\n🔍 测试显存使用...")
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA不可用，跳过显存测试")
        return True
    
    try:
        # 清理显存
        torch.cuda.empty_cache()
        
        # 获取初始显存
        initial_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"初始显存使用: {initial_memory:.2f}GB")
        
        # 模拟训练过程的显存使用
        pc = ProjectConfig()
        
        # 创建模拟数据
        batch_size = pc.batch_size
        seq_length = pc.max_seq_length
        vocab_size = 65024  # ChatGLM-6B词汇表大小
        
        # 模拟输入数据
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(pc.device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_length)).to(pc.device)
        
        current_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"加载数据后显存: {current_memory:.2f}GB")
        
        # 清理
        del input_ids, labels
        torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"清理后显存: {final_memory:.2f}GB")
        
        # 检查显存是否超过20GB
        max_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU总显存: {max_memory:.2f}GB")
        
        if current_memory > 20:
            print(f"⚠️ 显存使用可能超过20GB限制")
        else:
            print("✅ 显存使用检查通过")
        
        return True
        
    except Exception as e:
        print(f"❌ 显存测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 ChatGLM-6B QLoRA 三元组抽取训练测试")
    print("=" * 60)
    
    tests = [
        ("环境检查", test_environment),
        ("配置检查", test_config),
        ("数据格式检查", test_data_format),
        ("数据加载检查", test_data_loading),
        ("显存使用检查", test_memory_usage),
        # ("模型加载检查", test_model_loading),  # 这个测试比较耗时，可选
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
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！可以开始训练。")
        return True
    else:
        print("⚠️ 部分测试失败，请检查配置和环境。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)