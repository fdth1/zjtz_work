import json
# 返回的字符串包含有关异常的详细信
import traceback
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from functools import partial
import sys
sys.path.append('..')

from glm_config import *


def convert_example(
        examples: dict,
        tokenizer,
        max_source_seq_len: int,
        max_target_seq_len: int,
    ):
    """
    将三元组抽取样本数据转换为ChatGLM模型接收的输入数据。

    Args:
        examples (dict): 训练数据样本, e.g. -> {
                                                "text": [
                                                    '{"context": "Instruction: 你现在是一个很厉害的阅读理解器...", "target": "```json\\n[{...}]\\n```"}',
                                                    ...
                                                ]
                                            }
        max_source_seq_len (int): 输入序列最大长度
        max_target_seq_len (int): 输出序列最大长度

    Returns:
        dict (str: np.array) -> tokenized_output = {
                            'input_ids': [[1525, 10, ...], [758, 2345, ...]],
                            'labels': [[822, 10, ...], [125, 58...]]
                        }
    """
    tokenized_output = {
        'input_ids': [],
        'labels': []
    }

    max_seq_length = max_source_seq_len + max_target_seq_len

    for example in examples['text']:
        try:
            example = json.loads(example)
            context = example["context"]
            target = example["target"]
            
            # 验证三元组数据格式
            if not validate_triplet_format(target):
                print(f"⚠️ 跳过格式不正确的样本: {target[:100]}...")
                continue

            # 编码输入文本（指令+问题）
            prompts_ids = tokenizer.encode(
                text=context,
                add_special_tokens=False
            )

            # 编码目标文本（JSON格式的三元组）
            target_ids = tokenizer.encode(
                text=target,
                add_special_tokens=False
            )
            
            # 截断处理 - 为特殊token预留空间
            if len(prompts_ids) >= max_source_seq_len:
                prompts_ids = prompts_ids[:max_source_seq_len - 1]
            if len(target_ids) >= max_target_seq_len - 1:
                target_ids = target_ids[:max_target_seq_len - 2]

            # 构建完整的输入序列: source_ids + [gMASK] + <sop> + target_ids + <eos>
            input_ids = tokenizer.build_inputs_with_special_tokens(prompts_ids, target_ids)

            # 找到目标序列的开始位置（bos token位置）
            context_length = input_ids.index(tokenizer.bos_token_id)

            # 构建标签：只有目标序列部分参与损失计算
            labels = [-100] * context_length + input_ids[context_length:]

            # 填充到固定长度
            pad_len = max_seq_length - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [-100] * pad_len
            elif pad_len < 0:
                # 如果序列太长，进行截断
                input_ids = input_ids[:max_seq_length]
                labels = labels[:max_seq_length]

            tokenized_output['input_ids'].append(input_ids)
            tokenized_output['labels'].append(labels)
            
        except Exception as e:
            print(f'❌ 处理样本时出错: "{str(example)[:100]}..." -> {str(e)}')
            continue

    # 转换为numpy数组
    for k, v in tokenized_output.items():
        tokenized_output[k] = np.array(v)

    return tokenized_output


def validate_triplet_format(target_text: str) -> bool:
    """
    验证三元组数据格式是否正确
    
    Args:
        target_text (str): 目标文本，应该是JSON格式的三元组
        
    Returns:
        bool: 格式是否正确
    """
    try:
        # 检查是否包含```json标记
        if "```json" not in target_text or "```" not in target_text:
            return False
            
        # 提取JSON部分
        json_start = target_text.find("```json") + 7
        json_end = target_text.rfind("```")
        json_str = target_text[json_start:json_end].strip()
        
        # 解析JSON
        triplets = json.loads(json_str)
        
        # 验证是否为列表
        if not isinstance(triplets, list):
            return False
            
        # 验证每个三元组的格式
        for triplet in triplets:
            if not isinstance(triplet, dict):
                return False
            required_keys = {"subject", "predicate", "object", "subject_type", "object_type"}
            if not required_keys.issubset(triplet.keys()):
                return False
                
        return True
        
    except (json.JSONDecodeError, ValueError, KeyError):
        return False


def get_max_length(
        tokenizer,
        dataset_file: str
    ):
    """
    测试数据集最大的输入/输出tokens是多少。

    Args:
        dataset_file (str): _description_
    """
    source_seq_len_list = []
    target_seq_len_list = []
    with open(dataset_file, 'r') as f:
        for line in tqdm(f.readlines()):
            line = json.loads(line)

            source_len = tokenizer.encode(line['context'])
            source_seq_len_list.append(len(source_len))

            target_len = tokenizer.encode(line['target'])
            target_seq_len_list.append(len(target_len))

    print(dataset_file)
    print(f"【Source Sequence】 Max: {max(source_seq_len_list)}, Avg: {int(sum(source_seq_len_list) / len(source_seq_len_list))}, Middle: {sorted(source_seq_len_list)[int(len(source_seq_len_list) / 2)]}.")
    print(f"【Target Sequence】 Max: {max(target_seq_len_list)}, Avg: {int(sum(target_seq_len_list) / len(target_seq_len_list))}, Middle: {sorted(target_seq_len_list)[int(len(target_seq_len_list) / 2)]}.")




if __name__ == '__main__':
    pc = ProjectConfig()
    train_dataset = load_dataset('text', data_files={'train': pc.train_path})
    # print(type(train_dataset))
    # print(train_dataset)
    # print('*'*80)
    # print(train_dataset['train'])
    # print('*'*80)
    # print(train_dataset['train']['text'])
    tokenizer = AutoTokenizer.from_pretrained(pc.pre_model, trust_remote_code=True)
    tokenized_output = convert_example(examples=train_dataset['train'],
                                       tokenizer=tokenizer,
                                       max_source_seq_len=30,
                                       max_target_seq_len=20)
    print(len(tokenized_output["input_ids"][0]))
    print(len(tokenized_output["labels"][0]))

    get_max_length(tokenizer, pc.train_path)