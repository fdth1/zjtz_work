#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的训练配置文件
用户可以直接修改这个文件来调整训练参数
"""

class TrainingConfig:
    """训练配置类 - 用户可直接修改参数"""
    
    # ========== 基础配置 ==========
    # 模型名称或路径
    MODEL_NAME = "THUDM/chatglm-6b"
    
    # 数据文件路径
    TRAIN_FILE = "data/train_triplet.jsonl"
    VALIDATION_FILE = "data/val_triplet.jsonl"
    
    # 输出目录
    OUTPUT_DIR = "output/chatglm-6b-triplet-qlora"
    
    # ========== 训练参数 ==========
    # 训练轮数
    NUM_EPOCHS = 3
    
    # 批次大小 (根据GPU显存调整)
    # 12GB GPU: 建议 1-2
    # 24GB GPU: 建议 4-8  
    # 40GB+ GPU: 建议 8-16
    BATCH_SIZE = 4
    EVAL_BATCH_SIZE = 4
    
    # 梯度累积步数 (有效批次大小 = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    GRADIENT_ACCUMULATION_STEPS = 4
    
    # 学习率
    LEARNING_RATE = 2e-4
    
    # 预热步数
    WARMUP_STEPS = 100
    
    # ========== LoRA参数 ==========
    # LoRA rank (越大模型容量越大，但训练越慢)
    # 建议值: 4, 8, 16, 32
    LORA_R = 8
    
    # LoRA alpha (通常设为 LORA_R 的 2-4 倍)
    LORA_ALPHA = 32
    
    # LoRA dropout
    LORA_DROPOUT = 0.1
    
    # LoRA目标模块 (ChatGLM-6B的注意力和前馈网络层)
    LORA_TARGET_MODULES = [
        "query_key_value",  # 注意力查询、键、值
        "dense",            # 注意力输出层
        "dense_h_to_4h",    # 前馈网络第一层
        "dense_4h_to_h"     # 前馈网络第二层
    ]
    
    # ========== 序列长度 ==========
    # 输入序列最大长度
    MAX_SOURCE_LENGTH = 512
    
    # 输出序列最大长度
    MAX_TARGET_LENGTH = 256
    
    # ========== 量化配置 ==========
    # 是否使用4bit量化
    LOAD_IN_4BIT = True
    
    # 量化计算类型
    BNB_4BIT_COMPUTE_DTYPE = "float16"
    
    # 是否使用双重量化
    BNB_4BIT_USE_DOUBLE_QUANT = True
    
    # 量化类型 ("nf4" 或 "fp4")
    BNB_4BIT_QUANT_TYPE = "nf4"
    
    # ========== 日志和保存 ==========
    # 日志记录步数
    LOGGING_STEPS = 10
    
    # 模型保存步数
    SAVE_STEPS = 500
    
    # 评估步数
    EVAL_STEPS = 500
    
    # ========== 高级配置 ==========
    # 是否使用梯度检查点 (节省显存但稍微降低速度)
    GRADIENT_CHECKPOINTING = True
    
    # 是否使用FP16混合精度训练
    FP16 = True
    
    # 随机种子
    SEED = 42
    
    # 是否在训练结束时加载最佳模型
    LOAD_BEST_MODEL_AT_END = True
    
    # 最佳模型评估指标
    METRIC_FOR_BEST_MODEL = "eval_loss"
    
    # 指标是否越大越好
    GREATER_IS_BETTER = False

# ========== 预设配置 ==========

class SmallGPUConfig(TrainingConfig):
    """小显存GPU配置 (12GB以下)"""
    BATCH_SIZE = 1
    EVAL_BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 16
    LORA_R = 4
    LORA_ALPHA = 16
    MAX_SOURCE_LENGTH = 256
    MAX_TARGET_LENGTH = 128

class MediumGPUConfig(TrainingConfig):
    """中等显存GPU配置 (12-24GB)"""
    BATCH_SIZE = 4
    EVAL_BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    LORA_R = 8
    LORA_ALPHA = 32

class LargeGPUConfig(TrainingConfig):
    """大显存GPU配置 (24GB+)"""
    BATCH_SIZE = 8
    EVAL_BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 2
    LORA_R = 16
    LORA_ALPHA = 64

class FastTrainingConfig(TrainingConfig):
    """快速训练配置 (用于测试)"""
    NUM_EPOCHS = 1
    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION_STEPS = 2
    SAVE_STEPS = 100
    EVAL_STEPS = 100
    LOGGING_STEPS = 5

# ========== 使用说明 ==========
"""
使用方法:

1. 默认配置:
   config = TrainingConfig()

2. 小显存GPU:
   config = SmallGPUConfig()

3. 中等显存GPU:
   config = MediumGPUConfig()

4. 大显存GPU:
   config = LargeGPUConfig()

5. 快速测试:
   config = FastTrainingConfig()

6. 自定义配置:
   config = TrainingConfig()
   config.BATCH_SIZE = 2
   config.LORA_R = 16
   config.NUM_EPOCHS = 5

显存使用估算:
- 基础模型 (4bit): ~3.5GB
- LoRA参数: ~50-200MB (取决于LORA_R)
- 训练状态: ~1-2GB
- 批次数据: BATCH_SIZE * 序列长度 * 0.5MB

总显存需求 ≈ 5-6GB + BATCH_SIZE * 0.5GB
"""