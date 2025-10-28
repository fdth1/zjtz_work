#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试ChatGLM-6B tokenizer加载
用于验证在线模型是否可以正常加载
"""

import os
import sys
from transformers import AutoTokenizer

def test_tokenizer_loading():
    """测试tokenizer加载"""
    print("🚀 测试ChatGLM-6B Tokenizer加载")
    print("=" * 50)
    
    try:
        from glm_config import ProjectConfig
        pc = ProjectConfig()
        model_path = pc.pre_model
        
        print(f"📋 模型路径: {model_path}")
        print("🔄 正在加载tokenizer...")
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        print("✅ Tokenizer加载成功！")
        
        # 测试tokenizer功能
        test_text = "你好，这是一个测试文本。"
        print(f"\n🧪 测试文本: {test_text}")
        
        # 编码
        tokens = tokenizer.encode(test_text)
        print(f"📝 编码结果: {tokens}")
        print(f"📊 Token数量: {len(tokens)}")
        
        # 解码
        decoded = tokenizer.decode(tokens)
        print(f"🔄 解码结果: {decoded}")
        
        # 测试特殊token
        print(f"\n🔧 特殊Token:")
        print(f"   - BOS Token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
        print(f"   - EOS Token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
        print(f"   - PAD Token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        print(f"   - UNK Token: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
        
        # 获取词汇表大小
        try:
            vocab_size = len(tokenizer)
            print(f"\n📊 词汇表大小: {vocab_size}")
        except:
            try:
                vocab_size = tokenizer.vocab_size
                print(f"\n📊 词汇表大小: {vocab_size}")
            except:
                print(f"\n📊 词汇表大小: 无法获取")
        
        print("\n🎉 Tokenizer测试完成！")
        return True
        
    except Exception as e:
        print(f"❌ Tokenizer加载失败: {e}")
        print(f"📋 错误类型: {type(e).__name__}")
        
        # 提供解决建议
        print("\n💡 可能的解决方案:")
        print("1. 检查网络连接，确保可以访问Hugging Face")
        print("2. 尝试设置代理或使用镜像源")
        print("3. 检查transformers版本是否兼容")
        print("4. 清理缓存后重试")
        
        return False

def test_model_loading():
    """测试模型加载（仅检查是否可以初始化）"""
    print("\n🤖 测试模型初始化...")
    
    try:
        from transformers import AutoModel
        from glm_config import ProjectConfig
        
        pc = ProjectConfig()
        model_path = pc.pre_model
        
        print("🔄 正在初始化模型（仅检查配置）...")
        
        # 只加载配置，不加载权重
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        print("✅ 模型配置加载成功！")
        print(f"📊 模型信息:")
        print(f"   - 模型类型: {config.model_type}")
        print(f"   - 隐藏层大小: {config.hidden_size}")
        print(f"   - 层数: {config.num_layers}")
        print(f"   - 注意力头数: {config.num_attention_heads}")
        print(f"   - 词汇表大小: {config.vocab_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型配置加载失败: {e}")
        return False

def main():
    """主函数"""
    success_count = 0
    total_tests = 2
    
    # 测试tokenizer
    if test_tokenizer_loading():
        success_count += 1
    
    # 测试模型配置
    if test_model_loading():
        success_count += 1
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {success_count}/{total_tests} 通过")
    
    if success_count == total_tests:
        print("🎉 所有测试通过！可以开始训练了。")
        return 0
    else:
        print("⚠️ 部分测试失败，请检查配置。")
        return 1

if __name__ == "__main__":
    sys.exit(main())