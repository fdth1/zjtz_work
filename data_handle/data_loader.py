# coding:utf-8
import os
import sys
from torch.utils.data import DataLoader
from transformers import default_data_collator, AutoTokenizer
from functools import partial

# 添加父目录到路径以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .data_preprocess import *
from glm_config import *

pc = ProjectConfig() # 实例化项目配置文件

# 延迟初始化tokenizer，避免在导入时就加载模型
tokenizer = None

def get_tokenizer():
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(pc.pre_model, trust_remote_code=True)
    return tokenizer

def get_data():
    """获取训练和验证数据加载器"""
    # 检查数据文件是否存在
    if not os.path.exists(pc.train_path):
        raise FileNotFoundError(f"训练数据文件不存在: {pc.train_path}")
    if not os.path.exists(pc.dev_path):
        raise FileNotFoundError(f"验证数据文件不存在: {pc.dev_path}")
    
    dataset = load_dataset('text', data_files={'train': pc.train_path,
                                               'dev': pc.dev_path})

    # 使用配置中的序列长度
    new_func = partial(convert_example,
                       tokenizer=get_tokenizer(),
                       max_source_seq_len=pc.max_source_seq_len,
                       max_target_seq_len=pc.max_target_seq_len)

    dataset = dataset.map(new_func, batched=True, remove_columns=['text'])
    train_dataset = dataset["train"]
    dev_dataset = dataset["dev"]
    
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=pc.batch_size,
        num_workers=pc.dataloader_num_workers,
        pin_memory=pc.dataloader_pin_memory
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        collate_fn=default_data_collator,
        batch_size=pc.batch_size,
        num_workers=pc.dataloader_num_workers,
        pin_memory=pc.dataloader_pin_memory
    )
    return train_dataloader, dev_dataloader


if __name__ == '__main__':
    train_dataloader, dev_dataloader = get_data()
    print(len(train_dataloader))
    print(len(dev_dataloader))
    for i, value in enumerate(train_dataloader):
        print(i)
        print(value)
        print(value['input_ids'].shape)
        print(value['labels'].shape)
        break
