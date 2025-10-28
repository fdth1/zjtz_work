#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用QLoRA微调ChatGLM-6B进行三元组抽取
"""

import os
import json
import torch
import argparse
from typing import Dict, List
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer, 
    AutoModel,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import Dataset
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """模型相关参数"""
    model_name_or_path: str = field(default="THUDM/chatglm-6b")
    cache_dir: str = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)

@dataclass
class DataArguments:
    """数据相关参数"""
    train_file: str = field(default="data/train_triplet.jsonl")
    validation_file: str = field(default="data/val_triplet.jsonl")
    max_source_length: int = field(default=512)
    max_target_length: int = field(default=256)
    ignore_pad_token_for_loss: bool = field(default=True)

@dataclass
class LoraArguments:
    """LoRA相关参数"""
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.1)
    lora_target_modules: List[str] = field(default_factory=lambda: ["query_key_value"])

class TripletDataset:
    """三元组抽取数据集处理类"""
    
    def __init__(self, tokenizer, data_args: DataArguments):
        self.tokenizer = tokenizer
        self.data_args = data_args
        
    def load_data(self, file_path: str) -> List[Dict]:
        """加载JSONL格式数据"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def preprocess_function(self, examples: Dict) -> Dict:
        """数据预处理函数"""
        inputs = []
        targets = []
        
        for i in range(len(examples['instruction'])):
            instruction = examples['instruction'][i]
            input_text = examples['input'][i]
            output_text = examples['output'][i]
            
            # 构建输入格式
            prompt = f"{instruction}\n输入：{input_text}\n输出："
            inputs.append(prompt)
            targets.append(output_text)
        
        # 编码输入
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.data_args.max_source_length,
            truncation=True,
            padding=False
        )
        
        # 编码标签
        labels = self.tokenizer(
            targets,
            max_length=self.data_args.max_target_length,
            truncation=True,
            padding=False
        )
        
        # 设置标签
        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs
    
    def get_dataset(self, file_path: str) -> Dataset:
        """获取处理后的数据集"""
        raw_data = self.load_data(file_path)
        
        # 转换为Dataset格式
        dataset_dict = {
            'instruction': [item['instruction'] for item in raw_data],
            'input': [item['input'] for item in raw_data],
            'output': [item['output'] for item in raw_data]
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        processed_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return processed_dataset

def setup_model_and_tokenizer(model_args: ModelArguments):
    """设置模型和分词器"""
    
    # 量化配置
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
        trust_remote_code=True
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 准备模型进行k-bit训练
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def setup_lora(model, lora_args: LoraArguments):
    """设置LoRA配置"""
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=lora_args.lora_target_modules,
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

def main():
    parser = argparse.ArgumentParser()
    
    # 添加参数
    parser.add_argument("--model_name_or_path", type=str, default="THUDM/chatglm-6b")
    parser.add_argument("--train_file", type=str, default="data/train_triplet.jsonl")
    parser.add_argument("--validation_file", type=str, default="data/val_triplet.jsonl")
    parser.add_argument("--output_dir", type=str, default="output/chatglm-6b-triplet-qlora")
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
    
    # 创建参数对象，确保去除空白字符
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
    
    # 设置模型和分词器
    logger.info("加载模型和分词器...")
    model, tokenizer = setup_model_and_tokenizer(model_args)
    
    # 设置LoRA
    logger.info("设置LoRA配置...")
    model = setup_lora(model, lora_args)
    
    # 准备数据
    logger.info("准备训练数据...")
    dataset_processor = TripletDataset(tokenizer, data_args)
    train_dataset = dataset_processor.get_dataset(data_args.train_file)
    eval_dataset = dataset_processor.get_dataset(data_args.validation_file)
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(eval_dataset)}")
    
    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
        padding=True
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir.strip(),
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
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
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    # 开始训练
    logger.info("开始训练...")
    trainer.train()
    
    # 保存模型
    logger.info("保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir.strip())
    
    logger.info("训练完成！")

if __name__ == "__main__":
    main()