#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三元组抽取模型评估脚本
"""

import json
import re
import argparse
from typing import List, Tuple, Set
from inference import TripletExtractor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TripletEvaluator:
    """三元组抽取评估器"""
    
    def __init__(self, extractor: TripletExtractor):
        self.extractor = extractor
    
    def parse_triplets(self, text: str) -> Set[Tuple[str, str, str]]:
        """解析三元组文本，返回三元组集合"""
        triplets = set()
        
        # 使用正则表达式匹配三元组格式 (主体, 关系, 客体)
        pattern = r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)'
        matches = re.findall(pattern, text)
        
        for match in matches:
            subject = match[0].strip()
            relation = match[1].strip()
            object_entity = match[2].strip()
            triplets.add((subject, relation, object_entity))
        
        return triplets
    
    def calculate_metrics(self, predicted: Set, ground_truth: Set) -> dict:
        """计算精确率、召回率和F1分数"""
        
        if len(predicted) == 0 and len(ground_truth) == 0:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        
        if len(predicted) == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        if len(ground_truth) == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        # 计算交集
        intersection = predicted.intersection(ground_truth)
        
        # 计算指标
        precision = len(intersection) / len(predicted) if len(predicted) > 0 else 0.0
        recall = len(intersection) / len(ground_truth) if len(ground_truth) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predicted_count": len(predicted),
            "ground_truth_count": len(ground_truth),
            "correct_count": len(intersection)
        }
    
    def evaluate_dataset(self, test_file: str) -> dict:
        """评估整个测试数据集"""
        
        logger.info(f"开始评估测试文件: {test_file}")
        
        # 加载测试数据
        test_data = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                test_data.append(json.loads(line.strip()))
        
        logger.info(f"测试样本数量: {len(test_data)}")
        
        all_metrics = []
        correct_samples = 0
        
        for i, sample in enumerate(test_data):
            input_text = sample['input']
            expected_output = sample['output']
            
            # 模型预测
            predicted_output = self.extractor.extract_triplets(input_text)
            
            # 解析三元组
            predicted_triplets = self.parse_triplets(predicted_output)
            ground_truth_triplets = self.parse_triplets(expected_output)
            
            # 计算指标
            metrics = self.calculate_metrics(predicted_triplets, ground_truth_triplets)
            all_metrics.append(metrics)
            
            # 完全匹配的样本
            if predicted_triplets == ground_truth_triplets:
                correct_samples += 1
            
            # 打印详细信息（前几个样本）
            if i < 5:
                print(f"\n样本 {i+1}:")
                print(f"输入: {input_text}")
                print(f"预期输出: {expected_output}")
                print(f"模型输出: {predicted_output}")
                print(f"预期三元组: {ground_truth_triplets}")
                print(f"预测三元组: {predicted_triplets}")
                print(f"指标: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
                print("-" * 80)
        
        # 计算平均指标
        avg_precision = sum(m['precision'] for m in all_metrics) / len(all_metrics)
        avg_recall = sum(m['recall'] for m in all_metrics) / len(all_metrics)
        avg_f1 = sum(m['f1'] for m in all_metrics) / len(all_metrics)
        exact_match_accuracy = correct_samples / len(test_data)
        
        results = {
            "total_samples": len(test_data),
            "exact_match_accuracy": exact_match_accuracy,
            "average_precision": avg_precision,
            "average_recall": avg_recall,
            "average_f1": avg_f1,
            "correct_samples": correct_samples
        }
        
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="THUDM/chatglm-6b",
                       help="基础模型路径")
    parser.add_argument("--lora_model", type=str, default=None,
                       help="LoRA模型路径")
    parser.add_argument("--test_file", type=str, default="data/val_triplet.jsonl",
                       help="测试数据文件")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json",
                       help="评估结果输出文件")
    
    args = parser.parse_args()
    
    # 初始化抽取器
    logger.info("初始化三元组抽取器...")
    extractor = TripletExtractor(args.base_model, args.lora_model)
    
    # 初始化评估器
    evaluator = TripletEvaluator(extractor)
    
    # 执行评估
    results = evaluator.evaluate_dataset(args.test_file)
    
    # 打印结果
    print("\n" + "="*60)
    print("评估结果:")
    print("="*60)
    print(f"总样本数: {results['total_samples']}")
    print(f"完全匹配准确率: {results['exact_match_accuracy']:.4f}")
    print(f"平均精确率: {results['average_precision']:.4f}")
    print(f"平均召回率: {results['average_recall']:.4f}")
    print(f"平均F1分数: {results['average_f1']:.4f}")
    print(f"完全正确样本数: {results['correct_samples']}")
    
    # 保存结果
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"评估结果已保存到: {args.output_file}")

if __name__ == "__main__":
    main()