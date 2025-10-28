#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用配置文件的ChatGLM-6B QLoRA训练脚本
支持多种预设配置，用户可根据GPU显存选择合适的配置
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModel,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from transformers import BitsAndBytesConfig

# 添加config目录到Python路径
config_path = Path(__file__).parent / "config"
sys.path.insert(0, str(config_path))

from train_config_simple import (
    TrainingConfig,
    SmallGPUConfig,
    MediumGPUConfig,
    LargeGPUConfig,
    FastTrainingConfig
)

# 设置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class TripletDataset(Dataset):
    """三元组数据集类"""
    
    def __init__(self, tokenizer, data_file: str, max_source_length: int = 512, max_target_length: int = 256):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.data = self.load_data(data_file)
        
    def load_data(self, data_file: str) -> List[Dict]:
        """加载JSONL数据"""
        data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        logger.info(f"加载了 {len(data)} 条数据从 {data_file}")
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构建输入文本
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")
        
        # 构建完整的prompt
        if instruction and input_text:
            prompt = f"{instruction}\n{input_text}"
        elif instruction:
            prompt = instruction
        else:
            prompt = input_text
        
        # 编码输入
        source_encoding = self.tokenizer(
            prompt,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 编码输出
        target_encoding = self.tokenizer(
            output_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": source_encoding["input_ids"].flatten(),
            "attention_mask": source_encoding["attention_mask"].flatten(),
            "labels": target_encoding["input_ids"].flatten()
        }

class ChatGLMTrainer:
    """ChatGLM训练器"""
    
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        
    def setup_model_and_tokenizer(self):
        """设置模型和分词器"""
        logger.info("加载分词器...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.MODEL_NAME,
            trust_remote_code=True
        )
        
        # 设置pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("设置量化配置...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=self.config.LOAD_IN_4BIT,
            bnb_4bit_compute_dtype=getattr(torch, self.config.BNB_4BIT_COMPUTE_DTYPE),
            bnb_4bit_use_double_quant=self.config.BNB_4BIT_USE_DOUBLE_QUANT,
            bnb_4bit_quant_type=self.config.BNB_4BIT_QUANT_TYPE
        )
        
        logger.info("加载基础模型...")
        self.model = AutoModel.from_pretrained(
            self.config.MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # 准备模型进行k-bit训练
        self.model = prepare_model_for_kbit_training(self.model)
        
        logger.info("设置LoRA配置...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.LORA_R,
            lora_alpha=self.config.LORA_ALPHA,
            lora_dropout=self.config.LORA_DROPOUT,
            target_modules=self.config.LORA_TARGET_MODULES
        )
        
        # 应用LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # 打印可训练参数
        self.model.print_trainable_parameters()
        
    def prepare_datasets(self):
        """准备训练和验证数据集"""
        logger.info("准备训练数据集...")
        self.train_dataset = TripletDataset(
            self.tokenizer,
            self.config.TRAIN_FILE,
            self.config.MAX_SOURCE_LENGTH,
            self.config.MAX_TARGET_LENGTH
        )
        
        logger.info("准备验证数据集...")
        self.eval_dataset = TripletDataset(
            self.tokenizer,
            self.config.VALIDATION_FILE,
            self.config.MAX_SOURCE_LENGTH,
            self.config.MAX_TARGET_LENGTH
        )
        
    def train(self):
        """开始训练"""
        logger.info("设置训练参数...")
        
        # 创建输出目录
        output_dir = Path(self.config.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            overwrite_output_dir=True,
            num_train_epochs=self.config.NUM_EPOCHS,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            per_device_eval_batch_size=self.config.EVAL_BATCH_SIZE,
            gradient_accumulation_steps=self.config.GRADIENT_ACCUMULATION_STEPS,
            learning_rate=self.config.LEARNING_RATE,
            warmup_steps=self.config.WARMUP_STEPS,
            logging_steps=self.config.LOGGING_STEPS,
            save_steps=self.config.SAVE_STEPS,
            eval_steps=self.config.EVAL_STEPS,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=self.config.LOAD_BEST_MODEL_AT_END,
            metric_for_best_model=self.config.METRIC_FOR_BEST_MODEL,
            greater_is_better=self.config.GREATER_IS_BETTER,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            fp16=self.config.FP16,
            gradient_checkpointing=self.config.GRADIENT_CHECKPOINTING,
            report_to=None,
            run_name="chatglm-6b-triplet-qlora"
        )
        
        # 数据整理器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
            pad_to_multiple_of=None,
            padding=True
        )
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        logger.info("开始训练...")
        trainer.train()
        
        # 保存模型
        logger.info("保存模型...")
        trainer.save_model()
        self.tokenizer.save_pretrained(str(output_dir))
        
        logger.info(f"训练完成！模型保存在: {output_dir.absolute()}")
        
        return True

def detect_gpu_config():
    """自动检测GPU配置并推荐合适的训练配置"""
    if not torch.cuda.is_available():
        logger.warning("未检测到CUDA，将使用CPU训练（不推荐）")
        return SmallGPUConfig()
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"检测到GPU内存: {gpu_memory:.1f} GB")
    
    if gpu_memory < 12:
        logger.info("推荐使用小显存配置")
        return SmallGPUConfig()
    elif gpu_memory < 24:
        logger.info("推荐使用中等显存配置")
        return MediumGPUConfig()
    else:
        logger.info("推荐使用大显存配置")
        return LargeGPUConfig()

def print_config_info(config):
    """打印配置信息"""
    logger.info("=" * 50)
    logger.info("训练配置:")
    logger.info(f"  模型: {config.MODEL_NAME}")
    logger.info(f"  训练轮数: {config.NUM_EPOCHS}")
    logger.info(f"  批次大小: {config.BATCH_SIZE}")
    logger.info(f"  梯度累积: {config.GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"  有效批次大小: {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"  学习率: {config.LEARNING_RATE}")
    logger.info(f"  LoRA rank: {config.LORA_R}")
    logger.info(f"  LoRA alpha: {config.LORA_ALPHA}")
    logger.info(f"  输入长度: {config.MAX_SOURCE_LENGTH}")
    logger.info(f"  输出长度: {config.MAX_TARGET_LENGTH}")
    logger.info(f"  输出目录: {config.OUTPUT_DIR}")
    
    # 估算显存使用
    estimated_memory = 5.5 + config.BATCH_SIZE * 0.5
    logger.info(f"  预估显存需求: ~{estimated_memory:.1f} GB")
    logger.info("=" * 50)

def check_data_files(config):
    """检查数据文件"""
    train_file = Path(config.TRAIN_FILE)
    val_file = Path(config.VALIDATION_FILE)
    
    if not train_file.exists():
        logger.error(f"训练文件不存在: {train_file}")
        logger.info("请先运行: python data/generate_triplet_data.py")
        return False
    
    if not val_file.exists():
        logger.error(f"验证文件不存在: {val_file}")
        logger.info("请先运行: python data/generate_triplet_data.py")
        return False
    
    return True

def main():
    """主函数"""
    print("=" * 80)
    print("ChatGLM-6B QLoRA 三元组抽取训练 - 配置化版本")
    print("=" * 80)
    
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="ChatGLM-6B QLoRA训练")
    parser.add_argument("--config", type=str, default="auto", 
                       choices=["auto", "small", "medium", "large", "fast", "default"],
                       help="选择配置类型")
    parser.add_argument("--epochs", type=int, help="训练轮数")
    parser.add_argument("--batch_size", type=int, help="批次大小")
    parser.add_argument("--lora_r", type=int, help="LoRA rank")
    parser.add_argument("--learning_rate", type=float, help="学习率")
    parser.add_argument("--output_dir", type=str, help="输出目录")
    
    args = parser.parse_args()
    
    # 选择配置
    if args.config == "auto":
        config = detect_gpu_config()
        logger.info("使用自动检测的配置")
    elif args.config == "small":
        config = SmallGPUConfig()
        logger.info("使用小显存GPU配置")
    elif args.config == "medium":
        config = MediumGPUConfig()
        logger.info("使用中等显存GPU配置")
    elif args.config == "large":
        config = LargeGPUConfig()
        logger.info("使用大显存GPU配置")
    elif args.config == "fast":
        config = FastTrainingConfig()
        logger.info("使用快速训练配置")
    else:
        config = TrainingConfig()
        logger.info("使用默认配置")
    
    # 应用命令行参数覆盖
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
        config.EVAL_BATCH_SIZE = args.batch_size
    if args.lora_r:
        config.LORA_R = args.lora_r
        config.LORA_ALPHA = args.lora_r * 4  # 自动调整alpha
    if args.learning_rate:
        config.LEARNING_RATE = args.learning_rate
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    
    # 设置随机种子
    set_seed(config.SEED)
    
    # 打印配置信息
    print_config_info(config)
    
    # 检查数据文件
    if not check_data_files(config):
        return False
    
    try:
        # 创建训练器
        trainer = ChatGLMTrainer(config)
        
        # 设置模型和分词器
        trainer.setup_model_and_tokenizer()
        
        # 准备数据集
        trainer.prepare_datasets()
        
        # 开始训练
        success = trainer.train()
        
        if success:
            print("\n" + "=" * 80)
            print("🎉 训练成功完成！")
            print(f"📁 模型保存位置: {Path(config.OUTPUT_DIR).absolute()}")
            print("\n🚀 接下来可以:")
            print("   1. 使用推理脚本测试模型: python inference_triplet.py")
            print("   2. 评估模型性能: python evaluate_triplet.py")
            print("   3. 查看训练日志: ls output/chatglm-6b-triplet-qlora/")
            print("=" * 80)
        
        return success
        
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
        return False
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)