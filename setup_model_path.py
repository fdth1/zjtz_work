#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型路径配置助手
帮助用户快速配置正确的ChatGLM-6B模型路径
"""

import os
import sys
import json
import shutil
from pathlib import Path

def find_chatglm_models():
    """查找系统中可能的ChatGLM-6B模型路径"""
    possible_paths = [
        # ModelScope缓存路径
        "/root/.cache/modelscope/hub/models/ZhipuAI/ChatGLM-6B",
        "/home/user/.cache/modelscope/hub/models/ZhipuAI/ChatGLM-6B",
        os.path.expanduser("~/.cache/modelscope/hub/models/ZhipuAI/ChatGLM-6B"),
        
        # Hugging Face缓存路径
        "/root/.cache/huggingface/hub/models--THUDM--chatglm-6b",
        "/home/user/.cache/huggingface/hub/models--THUDM--chatglm-6b",
        os.path.expanduser("~/.cache/huggingface/hub/models--THUDM--chatglm-6b"),
        
        # 常见的本地路径
        "/models/ChatGLM-6B",
        "/data/models/ChatGLM-6B",
        "/workspace/models/ChatGLM-6B",
        "./models/ChatGLM-6B",
        "../models/ChatGLM-6B",
    ]
    
    found_models = []
    
    for path in possible_paths:
        if os.path.exists(path):
            # 检查是否包含必要的模型文件
            config_file = os.path.join(path, "config.json")
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    
                    # 检查是否是ChatGLM模型
                    if "chatglm" in config.get("model_type", "").lower() or \
                       "chatglm" in config.get("name_or_path", "").lower():
                        found_models.append(path)
                except:
                    pass
    
    return found_models

def validate_model_path(path):
    """验证模型路径是否有效"""
    if not os.path.exists(path):
        return False, "路径不存在"
    
    required_files = ["config.json"]
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(os.path.join(path, file)):
            missing_files.append(file)
    
    if missing_files:
        return False, f"缺少文件: {missing_files}"
    
    # 检查模型权重文件
    model_files = [
        "pytorch_model.bin",
        "pytorch_model-00001-of-00002.bin",  # 分片模型
        "model.safetensors"
    ]
    
    has_model_file = any(
        os.path.exists(os.path.join(path, f)) for f in model_files
    )
    
    if not has_model_file:
        return False, "未找到模型权重文件"
    
    return True, "模型路径有效"

def update_config_file(model_path):
    """更新配置文件中的模型路径"""
    config_file = "glm_config.py"
    
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        return False
    
    # 读取配置文件
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找并替换模型路径
    import re
    
    # 匹配 self.pre_model = '...' 或 self.pre_model = "..."
    pattern = r"(self\.pre_model\s*=\s*['\"])[^'\"]*(['\"])"
    
    if re.search(pattern, content):
        new_content = re.sub(pattern, f"\\g<1>{model_path}\\g<2>", content)
        
        # 备份原文件
        backup_file = f"{config_file}.backup"
        shutil.copy2(config_file, backup_file)
        print(f"📋 已备份原配置文件到: {backup_file}")
        
        # 写入新配置
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"✅ 已更新配置文件，模型路径设置为: {model_path}")
        return True
    else:
        print("❌ 未找到模型路径配置项")
        return False

def interactive_setup():
    """交互式设置模型路径"""
    print("🔍 正在搜索系统中的ChatGLM-6B模型...")
    
    found_models = find_chatglm_models()
    
    if found_models:
        print(f"\n✅ 找到 {len(found_models)} 个可能的模型路径:")
        for i, path in enumerate(found_models, 1):
            valid, msg = validate_model_path(path)
            status = "✅" if valid else "❌"
            print(f"  {i}. {status} {path}")
            if not valid:
                print(f"     {msg}")
        
        print(f"  {len(found_models) + 1}. 手动输入路径")
        print(f"  {len(found_models) + 2}. 使用在线模型 (THUDM/chatglm-6b)")
        
        while True:
            try:
                choice = input(f"\n请选择模型路径 (1-{len(found_models) + 2}): ").strip()
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(found_models):
                    selected_path = found_models[choice_num - 1]
                    valid, msg = validate_model_path(selected_path)
                    if valid:
                        return selected_path
                    else:
                        print(f"❌ 选择的路径无效: {msg}")
                        continue
                elif choice_num == len(found_models) + 1:
                    # 手动输入
                    manual_path = input("请输入模型路径: ").strip()
                    if manual_path:
                        valid, msg = validate_model_path(manual_path)
                        if valid:
                            return manual_path
                        else:
                            print(f"❌ 输入的路径无效: {msg}")
                            continue
                elif choice_num == len(found_models) + 2:
                    # 使用在线模型
                    return "THUDM/chatglm-6b"
                else:
                    print("❌ 无效的选择")
                    continue
                    
            except ValueError:
                print("❌ 请输入有效的数字")
                continue
            except KeyboardInterrupt:
                print("\n\n👋 用户取消操作")
                return None
    else:
        print("\n❌ 未找到本地ChatGLM-6B模型")
        print("\n💡 您可以:")
        print("1. 手动输入模型路径")
        print("2. 使用在线模型 (需要网络下载)")
        print("3. 退出并手动下载模型")
        
        while True:
            choice = input("\n请选择 (1-3): ").strip()
            
            if choice == "1":
                manual_path = input("请输入模型路径: ").strip()
                if manual_path:
                    valid, msg = validate_model_path(manual_path)
                    if valid:
                        return manual_path
                    else:
                        print(f"❌ 输入的路径无效: {msg}")
                        continue
            elif choice == "2":
                return "THUDM/chatglm-6b"
            elif choice == "3":
                return None
            else:
                print("❌ 无效的选择")
                continue

def main():
    """主函数"""
    print("🚀 ChatGLM-6B 模型路径配置助手")
    print("=" * 50)
    
    # 检查当前配置
    try:
        from glm_config import ProjectConfig
        pc = ProjectConfig()
        current_path = pc.pre_model
        print(f"📋 当前配置的模型路径: {current_path}")
        
        if current_path != "THUDM/chatglm-6b":
            valid, msg = validate_model_path(current_path)
            if valid:
                print("✅ 当前模型路径有效")
                
                choice = input("\n是否要重新配置模型路径? (y/N): ").strip().lower()
                if choice not in ['y', 'yes']:
                    print("👋 保持当前配置")
                    return 0
            else:
                print(f"❌ 当前模型路径无效: {msg}")
        else:
            print("📡 当前使用在线模型")
            
    except Exception as e:
        print(f"⚠️ 读取当前配置时出错: {e}")
    
    # 交互式设置
    selected_path = interactive_setup()
    
    if selected_path is None:
        print("\n👋 配置已取消")
        return 1
    
    # 更新配置文件
    print(f"\n🔧 正在更新配置文件...")
    if update_config_file(selected_path):
        print("\n✅ 模型路径配置完成！")
        
        # 验证新配置
        print("\n🧪 验证新配置...")
        try:
            # 重新导入配置
            import importlib
            import glm_config
            importlib.reload(glm_config)
            
            pc = glm_config.ProjectConfig()
            print(f"📋 新的模型路径: {pc.pre_model}")
            
            if pc.pre_model != "THUDM/chatglm-6b":
                valid, msg = validate_model_path(pc.pre_model)
                if valid:
                    print("✅ 新配置验证通过")
                else:
                    print(f"❌ 新配置验证失败: {msg}")
                    return 1
            else:
                print("📡 将使用在线模型")
            
        except Exception as e:
            print(f"⚠️ 验证新配置时出错: {e}")
        
        print("\n🎉 配置完成！现在可以运行训练了:")
        print("   python check_model_path.py  # 检查模型路径")
        print("   python test_basic.py        # 运行基础测试")
        print("   ./start_training.sh         # 开始训练")
        
        return 0
    else:
        print("\n❌ 配置更新失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())