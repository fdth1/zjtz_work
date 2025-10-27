#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChatGLM-6B三元组抽取推理脚本
"""

import torch
import argparse
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TripletExtractor:
    """三元组抽取器"""
    
    def __init__(self, base_model_path: str, lora_model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 加载分词器
        logger.info("加载分词器...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        
        # 加载基础模型
        logger.info("加载基础模型...")
        self.model = AutoModel.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # 如果有LoRA权重，则加载
        if lora_model_path:
            logger.info("加载LoRA权重...")
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_model_path,
                torch_dtype=torch.float16
            )
        
        self.model.eval()
        
    def extract_triplets(self, text: str, max_length: int = 512) -> str:
        """从文本中抽取三元组"""
        
        # 构建提示
        instruction = "请从以下文本中抽取所有的三元组，格式为(主体, 关系, 客体)："
        prompt = f"{instruction}\n输入：{text}\n输出："
        
        # 编码输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_length,
            truncation=True
        ).to(self.device)
        
        # 生成回复
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码输出
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        return response
    
    def batch_extract(self, texts: list, max_length: int = 512) -> list:
        """批量抽取三元组"""
        results = []
        for text in texts:
            result = self.extract_triplets(text, max_length)
            results.append(result)
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="THUDM/chatglm-6b", 
                       help="基础模型路径")
    parser.add_argument("--lora_model", type=str, default=None,
                       help="LoRA模型路径")
    parser.add_argument("--text", type=str, default=None,
                       help="要抽取三元组的文本")
    parser.add_argument("--interactive", action="store_true",
                       help="交互式模式")
    
    args = parser.parse_args()
    
    # 初始化抽取器
    extractor = TripletExtractor(args.base_model, args.lora_model)
    
    if args.interactive:
        # 交互式模式
        print("=== ChatGLM-6B 三元组抽取器 ===")
        print("输入文本进行三元组抽取，输入 'quit' 退出")
        print("-" * 50)
        
        while True:
            text = input("\n请输入文本: ").strip()
            if text.lower() == 'quit':
                break
            
            if text:
                print("\n抽取结果:")
                result = extractor.extract_triplets(text)
                print(result)
                print("-" * 50)
    
    elif args.text:
        # 单次抽取模式
        print(f"输入文本: {args.text}")
        result = extractor.extract_triplets(args.text)
        print(f"抽取结果: {result}")
    
    else:
        # 演示模式
        demo_texts = [
            "马云是阿里巴巴的创始人，阿里巴巴总部位于杭州。",
            "张三毕业于清华大学，现在在腾讯担任CTO职位。",
            "李四是小米的产品经理，小米总部在北京。",
            "王五从北京大学毕业后，在字节跳动工作，担任技术总监。"
        ]
        
        print("=== 演示模式 ===")
        for i, text in enumerate(demo_texts, 1):
            print(f"\n示例 {i}:")
            print(f"输入: {text}")
            result = extractor.extract_triplets(text)
            print(f"输出: {result}")
            print("-" * 50)

if __name__ == "__main__":
    main()