#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChatGLM-6B QLoRA三元组抽取训练启动脚本
纯Python实现，无需shell脚本
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """主训练函数"""
    print("=" * 60)
    print("ChatGLM-6B QLoRA 三元组抽取模型训练")
    print("=" * 60)
    
    # 设置项目根目录
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # 添加src目录到Python路径
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # 设置环境变量
    os.environ["PYTHONPATH"] = f"{os.environ.get('PYTHONPATH', '')}:{src_path}"
    
    # 训练参数配置
    config = {
        "model_name_or_path": "THUDM/chatglm-6b",
        "train_file": "data/train_triplet.jsonl",
        "validation_file": "data/val_triplet.jsonl",
        "output_dir": "output/chatglm-6b-triplet-qlora",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "max_source_length": 512,
        "max_target_length": 256,
        "lora_r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "warmup_steps": 100,
        "logging_steps": 10,
        "save_steps": 500,
        "eval_steps": 500
    }
    
    # 创建输出目录
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir.absolute()}")
    
    # 检查训练数据是否存在
    train_file = Path(config["train_file"])
    val_file = Path(config["validation_file"])
    
    if not train_file.exists():
        print(f"错误: 训练文件不存在 {train_file}")
        print("请先运行: cd data && python generate_triplet_data.py")
        return False
    
    if not val_file.exists():
        print(f"错误: 验证文件不存在 {val_file}")
        print("请先运行: cd data && python generate_triplet_data.py")
        return False
    
    print(f"训练文件: {train_file.absolute()}")
    print(f"验证文件: {val_file.absolute()}")
    
    # 构建训练命令
    cmd = [
        sys.executable, "src/train_qlora.py"
    ]
    
    # 添加参数
    for key, value in config.items():
        cmd.extend([f"--{key}", str(value)])
    
    print("\n开始训练...")
    print("训练配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print(f"\n执行命令: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        # 执行训练
        result = subprocess.run(cmd, check=True, cwd=project_root)
        
        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("训练完成！")
            print(f"模型保存在: {output_dir.absolute()}")
            print("=" * 60)
            return True
        else:
            print(f"训练失败，退出码: {result.returncode}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"训练过程中出现错误: {e}")
        return False
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        return False
    except Exception as e:
        print(f"未知错误: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)