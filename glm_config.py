# -*- coding:utf-8 -*-
import torch
import os


class ProjectConfig(object):
    def __init__(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # 使用相对路径，适配Linux服务器环境
        self.pre_model = 'THUDM/chatglm-6b'  # 使用HuggingFace模型名
        self.train_path = os.path.join(os.path.dirname(__file__), 'data', 'mixed_train_dataset.jsonl')
        self.dev_path = os.path.join(os.path.dirname(__file__), 'data', 'mixed_dev_dataset.jsonl')
        
        # LoRA配置 - 针对20GB显存优化
        self.use_lora = True
        self.use_ptuning = False
        self.lora_rank = 8  # 保持较小的rank以节省显存
        self.lora_alpha = 32
        self.lora_dropout = 0.1
        
        # 训练配置 - 20GB显存优化
        self.batch_size = 1  # 保持小批次
        self.gradient_accumulation_steps = 8  # 使用梯度累积模拟更大批次
        self.epochs = 3  # 减少训练轮数，避免过拟合
        self.learning_rate = 2e-4  # 适当提高学习率
        self.weight_decay = 0.01
        self.warmup_ratio = 0.1
        
        # 序列长度配置 - 显存优化
        self.max_source_seq_len = 256  # 减少输入序列长度
        self.max_target_seq_len = 128  # 减少输出序列长度
        self.max_seq_length = self.max_source_seq_len + self.max_target_seq_len
        
        # 日志和保存配置
        self.logging_steps = 50
        self.save_freq = 500  # 减少保存频率
        self.eval_steps = 200  # 评估频率
        
        # P-tuning配置（备用）
        self.pre_seq_len = 128
        self.prefix_projection = False
        
        # 保存路径
        self.save_dir = os.path.join(os.path.dirname(__file__), 'output', 'chatglm-6b-triplet-qlora')
        
        # 显存优化配置
        self.fp16 = True  # 启用半精度训练
        self.dataloader_num_workers = 4  # 数据加载器工作进程数
        self.remove_unused_columns = False
        self.dataloader_pin_memory = True
        
        # 确保输出目录存在
        os.makedirs(self.save_dir, exist_ok=True)


if __name__ == '__main__':
    pc = ProjectConfig()
    print(pc.save_dir)