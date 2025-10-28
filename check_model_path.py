#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型路径检查脚本
检查本地ChatGLM-6B模型是否存在并可用
"""

import os
import sys
from glm_config import ProjectConfig

def check_model_path():
    """检查模型路径是否存在"""
    pc = ProjectConfig()
    model_path = pc.pre_model
    
    print(f"🔍 检查模型路径: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        print("\n💡 解决方案:")
        print("1. 确认模型已下载到指定路径")
        print("2. 或者修改 glm_config.py 中的 self.pre_model 路径")
        print("3. 或者使用在线模型: self.pre_model = 'THUDM/chatglm-6b'")
        return False
    
    # 检查关键文件
    required_files = [
        'config.json',
        'tokenizer.model',
        'pytorch_model.bin'
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if not os.path.exists(file_path):
            # 检查是否有分片模型文件
            if file == 'pytorch_model.bin':
                shard_files = [f for f in os.listdir(model_path) if f.startswith('pytorch_model-') and f.endswith('.bin')]
                if not shard_files:
                    missing_files.append(file)
            else:
                missing_files.append(file)
    
    if missing_files:
        print(f"⚠️ 缺少关键文件: {missing_files}")
        print("模型可能不完整，建议重新下载")
        return False
    
    print("✅ 模型路径检查通过")
    
    # 显示模型信息
    try:
        config_path = os.path.join(model_path, 'config.json')
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"📋 模型信息:")
            print(f"   - 模型名称: {config.get('name_or_path', 'ChatGLM-6B')}")
            print(f"   - 词汇表大小: {config.get('vocab_size', 'Unknown')}")
            print(f"   - 隐藏层大小: {config.get('hidden_size', 'Unknown')}")
            print(f"   - 层数: {config.get('num_layers', 'Unknown')}")
    except Exception as e:
        print(f"⚠️ 读取模型配置时出错: {e}")
    
    return True

def check_disk_space():
    """检查磁盘空间"""
    pc = ProjectConfig()
    model_path = pc.pre_model
    
    try:
        import shutil
        total, used, free = shutil.disk_usage(os.path.dirname(model_path))
        
        print(f"\n💾 磁盘空间信息:")
        print(f"   - 总空间: {total // (1024**3):.1f} GB")
        print(f"   - 已使用: {used // (1024**3):.1f} GB")
        print(f"   - 可用空间: {free // (1024**3):.1f} GB")
        
        if free < 10 * 1024**3:  # 小于10GB
            print("⚠️ 可用磁盘空间不足10GB，可能影响训练")
            return False
        
        return True
    except Exception as e:
        print(f"⚠️ 检查磁盘空间时出错: {e}")
        return False

def main():
    """主函数"""
    print("🚀 ChatGLM-6B 模型路径检查")
    print("=" * 50)
    
    # 检查模型路径
    model_ok = check_model_path()
    
    # 检查磁盘空间
    disk_ok = check_disk_space()
    
    print("\n" + "=" * 50)
    if model_ok and disk_ok:
        print("✅ 所有检查通过，可以开始训练！")
        return 0
    else:
        print("❌ 检查未通过，请解决上述问题后再开始训练")
        return 1

if __name__ == "__main__":
    sys.exit(main())