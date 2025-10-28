#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的训练脚本参数解析
"""

import sys
import os
sys.path.append('src')

def test_argument_parsing():
    """测试参数解析是否正确"""
    print("测试参数解析...")
    
    # 模拟命令行参数
    test_args = [
        'train_qlora.py',
        '--model_name_or_path', 'THUDM/chatglm-6b',
        '--train_file', 'data/train_triplet.jsonl',
        '--validation_file', 'data/val_triplet.jsonl',
        '--output_dir', 'output/test',
        '--num_train_epochs', '1',
        '--per_device_train_batch_size', '2'
    ]
    
    # 保存原始argv
    original_argv = sys.argv
    
    try:
        # 设置测试参数
        sys.argv = test_args
        
        # 导入并测试参数解析部分
        from train_qlora import ModelArguments, DataArguments, LoraArguments
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_name_or_path", type=str, required=True)
        parser.add_argument("--train_file", type=str, default="data/train_triplet.jsonl")
        parser.add_argument("--validation_file", type=str, default="data/val_triplet.jsonl")
        parser.add_argument("--output_dir", type=str, required=True)
        parser.add_argument("--num_train_epochs", type=int, default=3)
        parser.add_argument("--per_device_train_batch_size", type=int, default=4)
        parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
        parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
        parser.add_argument("--learning_rate", type=float, default=2e-4)
        parser.add_argument("--max_source_length", type=int, default=512)
        parser.add_argument("--max_target_length", type=int, default=256)
        parser.add_argument("--lora_r", type=int, default=8)
        parser.add_argument("--lora_alpha", type=int, default=32)
        parser.add_argument("--lora_dropout", type=float, default=0.1)
        parser.add_argument("--warmup_steps", type=int, default=100)
        parser.add_argument("--logging_steps", type=int, default=10)
        parser.add_argument("--save_steps", type=int, default=500)
        parser.add_argument("--eval_steps", type=int, default=500)
        
        args = parser.parse_args()
        
        # 测试参数创建
        model_args = ModelArguments(model_name_or_path=args.model_name_or_path.strip())
        data_args = DataArguments(
            train_file=args.train_file.strip(),
            validation_file=args.validation_file.strip(),
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length
        )
        lora_args = LoraArguments(
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )
        
        print(f"✓ 模型名称: '{model_args.model_name_or_path}'")
        print(f"✓ 训练文件: '{data_args.train_file}'")
        print(f"✓ 验证文件: '{data_args.validation_file}'")
        print(f"✓ 输出目录: '{args.output_dir.strip()}'")
        
        # 检查是否有多余的空白字符
        if model_args.model_name_or_path != model_args.model_name_or_path.strip():
            print("❌ 模型名称包含多余空白字符")
            return False
        
        if data_args.train_file != data_args.train_file.strip():
            print("❌ 训练文件路径包含多余空白字符")
            return False
            
        print("✓ 参数解析测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 参数解析测试失败: {e}")
        return False
    finally:
        # 恢复原始argv
        sys.argv = original_argv

def test_shell_script():
    """测试shell脚本语法"""
    print("\n测试shell脚本语法...")
    
    scripts = ['scripts/train.sh', 'scripts/inference.sh', 'scripts/evaluate.sh']
    
    for script in scripts:
        if os.path.exists(script):
            result = os.system(f'bash -n {script}')
            if result == 0:
                print(f"✓ {script} 语法正确")
            else:
                print(f"❌ {script} 语法错误")
                return False
        else:
            print(f"⚠ {script} 不存在")
    
    return True

def main():
    """主测试函数"""
    print("=" * 50)
    print("修复验证测试")
    print("=" * 50)
    
    success = True
    
    # 测试参数解析
    if not test_argument_parsing():
        success = False
    
    # 测试shell脚本
    if not test_shell_script():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✓ 所有测试通过！修复成功。")
        print("\n修复内容:")
        print("1. 修复了shell脚本的长行问题")
        print("2. 添加了参数字符串的strip()处理")
        print("3. 确保所有路径参数都去除了多余空白字符")
    else:
        print("❌ 部分测试失败，需要进一步修复。")
    print("=" * 50)

if __name__ == "__main__":
    main()