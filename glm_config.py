# -*- coding:utf-8 -*-
import torch


class ProjectConfig(object):
    def __init__(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'
        self.pre_model = r'D:\01-LLM\02-code\04-chatglm_lora\checkpoints\ptune\model_best'
        self.train_path = r'D:\01-LLM\02-code\04-chatglm_lora\data\mixed_train_dataset.jsonl'
        self.dev_path = r'D:\01-LLM\02-code\04-chatglm_lora\data\mixed_dev_dataset.jsonl'
        self.use_lora = True
        self.use_ptuning = False
        self.lora_rank = 8
        self.batch_size = 1
        self.epochs = 64
        self.learning_rate = 3e-5
        self.weight_decay = 0
        self.warmup_ratio = 0.06
        self.max_source_seq_len = 400
        self.max_target_seq_len = 300
        self.logging_steps = 10
        self.save_freq = 1
        self.pre_seq_len = 128
        self.prefix_projection = False # 默认为False,即p-tuning,如果为True，即p-tuning-v2
        self.save_dir = r'D:\01-LLM\02-code\04-chatglm_lora\checkpoints\ptune'


if __name__ == '__main__':
    pc = ProjectConfig()
    print(pc.save_dir)