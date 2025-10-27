#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChatGLM-6B QLoRA三元组抽取评估脚本
纯Python实现，计算精确率、召回率和F1分数
"""

import os
import sys
import json
from pathlib import Path

def main():
    """主评估函数"""
    print("=" * 60)
    print("ChatGLM-6B QLoRA 三元组抽取模型评估")
    print("=" * 60)
    
    # 设置项目根目录
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # 添加src目录到Python路径
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # 检查模型和测试文件
    base_model = "THUDM/chatglm-6b"
    lora_model = "output/chatglm-6b-triplet-qlora"
    test_file = "data/val_triplet.jsonl"
    
    lora_path = Path(lora_model)
    test_path = Path(test_file)
    
    if not lora_path.exists():
        print(f"错误: LoRA模型不存在 {lora_path.absolute()}")
        print("请先运行训练脚本: python train_triplet.py")
        return False
    
    if not test_path.exists():
        print(f"错误: 测试文件不存在 {test_path.absolute()}")
        print("请先运行: cd data && python generate_triplet_data.py")
        return False
    
    print(f"基础模型: {base_model}")
    print(f"LoRA模型: {lora_path.absolute()}")
    print(f"测试文件: {test_path.absolute()}")
    
    try:
        # 导入评估模块
        from evaluate import TripletEvaluator
        
        print("\n正在加载模型...")
        evaluator = TripletEvaluator(
            base_model_path=base_model,
            lora_model_path=str(lora_path)
        )
        print("模型加载完成！")
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保已安装所有依赖: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"模型加载失败: {e}")
        return False
    
    # 加载测试数据
    print(f"\n正在加载测试数据...")
    test_data = []
    try:
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    test_data.append(json.loads(line))
        print(f"加载了 {len(test_data)} 个测试样本")
    except Exception as e:
        print(f"加载测试数据失败: {e}")
        return False
    
    # 执行评估
    print("\n开始评估...")
    print("-" * 40)
    
    try:
        results = evaluator.evaluate_dataset(test_data)
        
        # 显示评估结果
        print("\n" + "=" * 60)
        print("评估结果")
        print("=" * 60)
        
        print(f"总样本数: {results['total_samples']}")
        print(f"精确率 (Precision): {results['precision']:.4f}")
        print(f"召回率 (Recall): {results['recall']:.4f}")
        print(f"F1分数: {results['f1_score']:.4f}")
        print(f"完全匹配准确率: {results['exact_match_accuracy']:.4f}")
        
        print(f"\n详细统计:")
        print(f"预测的三元组总数: {results['total_predicted']}")
        print(f"真实的三元组总数: {results['total_ground_truth']}")
        print(f"正确预测的三元组数: {results['total_correct']}")
        print(f"完全正确的样本数: {results['exact_match_count']}")
        
        # 保存评估结果
        output_file = "evaluation_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n评估结果已保存到: {output_file}")
        
        # 显示一些示例
        if 'examples' in results and results['examples']:
            print(f"\n示例预测结果 (前5个):")
            print("-" * 40)
            for i, example in enumerate(results['examples'][:5], 1):
                print(f"\n示例 {i}:")
                print(f"输入: {example['input']}")
                print(f"真实: {example['ground_truth']}")
                print(f"预测: {example['prediction']}")
                print(f"正确: {'是' if example['correct'] else '否'}")
        
        print("\n" + "=" * 60)
        print("评估完成！")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        return False

def quick_test():
    """快速测试函数"""
    print("=" * 60)
    print("快速测试模式")
    print("=" * 60)
    
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    try:
        from evaluate import TripletEvaluator
        
        print("正在加载模型...")
        evaluator = TripletEvaluator(
            base_model_path="THUDM/chatglm-6b",
            lora_model_path="output/chatglm-6b-triplet-qlora"
        )
        
        # 测试样本
        test_samples = [
            {
                "input": "马云是阿里巴巴的创始人，阿里巴巴总部位于杭州。",
                "output": "(马云, 创立, 阿里巴巴)\n(阿里巴巴, 总部在, 杭州)"
            },
            {
                "input": "张三毕业于清华大学，现在在腾讯担任CTO职位。",
                "output": "(张三, 毕业于, 清华大学)\n(张三, 工作于, 腾讯)\n(张三, 担任, CTO)"
            }
        ]
        
        results = evaluator.evaluate_dataset(test_samples)
        
        print(f"\n快速测试结果:")
        print(f"精确率: {results['precision']:.4f}")
        print(f"召回率: {results['recall']:.4f}")
        print(f"F1分数: {results['f1_score']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"快速测试失败: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # 快速测试模式
        success = quick_test()
    else:
        # 完整评估模式
        success = main()
    
    sys.exit(0 if success else 1)