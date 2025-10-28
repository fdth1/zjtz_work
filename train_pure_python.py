#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChatGLM-6B QLoRA三元组抽取训练 - 纯Python实现
直接在Python中进行LoRA微调，无需shell脚本或subprocess调用
"""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass
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
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig

# 设置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """训练配置类"""
    # 模型配置
    model_name_or_path: str = "THUDM/chatglm-6b"
    
    # 数据配置
    train_file: str = "data/train_triplet.jsonl"
    validation_file: str = "data/val_triplet.jsonl"
    max_source_length: int = 512
    max_target_length: int = 256
    
    # 训练配置
    output_dir: str = "output/chatglm-6b-triplet-qlora"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    
    # LoRA配置
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    
    # 量化配置
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]

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
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        
    def setup_model_and_tokenizer(self):
        """设置模型和分词器"""
        logger.info("加载分词器...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=True
        )
        
        # 设置pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("设置量化配置...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type
        )
        
        logger.info("加载基础模型...")
        self.model = AutoModel.from_pretrained(
            self.config.model_name_or_path,
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
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules
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
            self.config.train_file,
            self.config.max_source_length,
            self.config.max_target_length
        )
        
        logger.info("准备验证数据集...")
        self.eval_dataset = TripletDataset(
            self.tokenizer,
            self.config.validation_file,
            self.config.max_source_length,
            self.config.max_target_length
        )
        
    def train(self):
        """开始训练"""
        logger.info("设置训练参数...")
        
        # 创建输出目录
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            fp16=True,
            gradient_checkpointing=True,
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

def check_environment():
    """检查运行环境"""
    logger.info("检查运行环境...")
    
    # 检查CUDA
    if torch.cuda.is_available():
        logger.info(f"CUDA可用，设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.warning("CUDA不可用，将使用CPU训练（不推荐）")
    
    # 检查内存
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU内存: {gpu_memory:.1f} GB")
        if gpu_memory < 12:
            logger.warning("GPU内存可能不足，建议使用24GB+显存的GPU")
    
    return True

def check_data_files(config: TrainingConfig):
    """检查数据文件"""
    train_file = Path(config.train_file)
    val_file = Path(config.validation_file)
    
    if not train_file.exists():
        logger.error(f"训练文件不存在: {train_file}")
        logger.info("请先运行: python data/generate_triplet_data.py")
        return False
    
    if not val_file.exists():
        logger.error(f"验证文件不存在: {val_file}")
        logger.info("请先运行: python data/generate_triplet_data.py")
        return False
    
    # 检查文件内容
    try:
        with open(train_file, 'r', encoding='utf-8') as f:
            train_count = sum(1 for line in f if line.strip())
        
        with open(val_file, 'r', encoding='utf-8') as f:
            val_count = sum(1 for line in f if line.strip())
        
        logger.info(f"训练样本数: {train_count}")
        logger.info(f"验证样本数: {val_count}")
        
        if train_count == 0 or val_count == 0:
            logger.error("数据文件为空")
            return False
            
    except Exception as e:
        logger.error(f"读取数据文件失败: {e}")
        return False
    
    return True

def main():
    """主函数"""
    print("=" * 70)
    print("ChatGLM-6B QLoRA 三元组抽取模型训练 - 纯Python实现")
    print("=" * 70)
    
    # 设置随机种子
    set_seed(42)
    
    # 创建配置
    config = TrainingConfig()
    
    # 打印配置
    logger.info("训练配置:")
    for key, value in config.__dict__.items():
        if not key.startswith('_'):
            logger.info(f"  {key}: {value}")
    
    # 检查环境
    if not check_environment():
        return False
    
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
            print("\n" + "=" * 70)
            print("🎉 训练成功完成！")
            print(f"📁 模型保存位置: {Path(config.output_dir).absolute()}")
            print("🚀 现在可以使用以下命令进行推理:")
            print("   python inference_triplet.py")
            print("=" * 70)
        
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