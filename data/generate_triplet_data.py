#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成三元组抽取训练数据
"""

import json
import random
from typing import List, Dict, Tuple

def generate_sample_data() -> List[Dict]:
    """生成示例三元组抽取数据"""
    
    # 定义一些实体和关系
    persons = ["张三", "李四", "王五", "赵六", "孙七", "周八", "吴九", "郑十", "马云", "马化腾", "雷军", "刘强东"]
    companies = ["阿里巴巴", "腾讯", "百度", "京东", "小米", "华为", "字节跳动", "美团", "滴滴", "拼多多"]
    positions = ["CEO", "CTO", "总裁", "副总裁", "技术总监", "产品经理", "工程师", "设计师"]
    locations = ["北京", "上海", "深圳", "杭州", "广州", "成都", "武汉", "南京", "西安", "苏州"]
    universities = ["清华大学", "北京大学", "复旦大学", "上海交通大学", "浙江大学", "中科大", "南京大学", "华中科技大学"]
    
    # 定义关系类型
    relations = {
        "工作于": ["person", "company"],
        "担任": ["person", "position"],
        "位于": ["company", "location"],
        "毕业于": ["person", "university"],
        "创立": ["person", "company"],
        "总部在": ["company", "location"],
        "任职": ["person", "position"]
    }
    
    samples = []
    
    # 生成训练样本
    for i in range(1000):
        # 随机选择关系
        relation = random.choice(list(relations.keys()))
        entity_types = relations[relation]
        
        # 根据关系类型选择实体
        if entity_types == ["person", "company"]:
            entity1 = random.choice(persons)
            entity2 = random.choice(companies)
        elif entity_types == ["person", "position"]:
            entity1 = random.choice(persons)
            entity2 = random.choice(positions)
        elif entity_types == ["company", "location"]:
            entity1 = random.choice(companies)
            entity2 = random.choice(locations)
        elif entity_types == ["person", "university"]:
            entity1 = random.choice(persons)
            entity2 = random.choice(universities)
        else:
            continue
            
        # 生成句子模板
        templates = {
            "工作于": [
                f"{entity1}在{entity2}工作。",
                f"{entity1}是{entity2}的员工。",
                f"{entity1}就职于{entity2}。"
            ],
            "担任": [
                f"{entity1}担任{entity2}职位。",
                f"{entity1}是{entity2}。",
                f"{entity1}的职位是{entity2}。"
            ],
            "位于": [
                f"{entity1}位于{entity2}。",
                f"{entity1}的地址在{entity2}。",
                f"{entity1}坐落在{entity2}。"
            ],
            "毕业于": [
                f"{entity1}毕业于{entity2}。",
                f"{entity1}是{entity2}的毕业生。",
                f"{entity1}从{entity2}毕业。"
            ],
            "创立": [
                f"{entity1}创立了{entity2}。",
                f"{entity1}是{entity2}的创始人。",
                f"{entity2}由{entity1}创建。"
            ],
            "总部在": [
                f"{entity1}总部在{entity2}。",
                f"{entity1}的总部位于{entity2}。",
                f"{entity1}总部设在{entity2}。"
            ],
            "任职": [
                f"{entity1}在公司任职{entity2}。",
                f"{entity1}的职务是{entity2}。",
                f"{entity1}担任{entity2}一职。"
            ]
        }
        
        text = random.choice(templates[relation])
        
        # 构建训练样本
        sample = {
            "text": text,
            "triplets": [(entity1, relation, entity2)]
        }
        samples.append(sample)
    
    # 添加一些复杂的多三元组样本
    complex_samples = [
        {
            "text": "马云是阿里巴巴的创始人，阿里巴巴总部位于杭州。",
            "triplets": [("马云", "创立", "阿里巴巴"), ("阿里巴巴", "总部在", "杭州")]
        },
        {
            "text": "张三毕业于清华大学，现在在腾讯担任CTO职位。",
            "triplets": [("张三", "毕业于", "清华大学"), ("张三", "工作于", "腾讯"), ("张三", "担任", "CTO")]
        },
        {
            "text": "李四是小米的产品经理，小米总部在北京。",
            "triplets": [("李四", "工作于", "小米"), ("李四", "担任", "产品经理"), ("小米", "总部在", "北京")]
        },
        {
            "text": "王五从北京大学毕业后，在字节跳动工作，担任技术总监。",
            "triplets": [("王五", "毕业于", "北京大学"), ("王五", "工作于", "字节跳动"), ("王五", "担任", "技术总监")]
        }
    ]
    
    samples.extend(complex_samples)
    
    return samples

def format_for_chatglm(samples: List[Dict]) -> List[Dict]:
    """将数据格式化为ChatGLM训练格式"""
    formatted_data = []
    
    for sample in samples:
        text = sample["text"]
        triplets = sample["triplets"]
        
        # 构建指令
        instruction = "请从以下文本中抽取所有的三元组，格式为(主体, 关系, 客体)："
        
        # 构建输出
        output_triplets = []
        for triplet in triplets:
            output_triplets.append(f"({triplet[0]}, {triplet[1]}, {triplet[2]})")
        output = "\n".join(output_triplets)
        
        # ChatGLM格式
        formatted_sample = {
            "instruction": instruction,
            "input": text,
            "output": output
        }
        
        formatted_data.append(formatted_sample)
    
    return formatted_data

def save_data(data: List[Dict], filename: str):
    """保存数据到文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    # 生成数据
    print("正在生成三元组抽取训练数据...")
    samples = generate_sample_data()
    
    # 格式化数据
    formatted_data = format_for_chatglm(samples)
    
    # 划分训练集和验证集
    random.shuffle(formatted_data)
    split_idx = int(len(formatted_data) * 0.9)
    train_data = formatted_data[:split_idx]
    val_data = formatted_data[split_idx:]
    
    # 保存数据
    save_data(train_data, "train_triplet.jsonl")
    save_data(val_data, "val_triplet.jsonl")
    
    print(f"训练数据生成完成！")
    print(f"训练集样本数: {len(train_data)}")
    print(f"验证集样本数: {len(val_data)}")
    print(f"数据已保存到 train_triplet.jsonl 和 val_triplet.jsonl")
    
    # 显示几个样本
    print("\n样本示例:")
    for i, sample in enumerate(train_data[:3]):
        print(f"\n样本 {i+1}:")
        print(f"指令: {sample['instruction']}")
        print(f"输入: {sample['input']}")
        print(f"输出: {sample['output']}")