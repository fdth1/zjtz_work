import os
import time
import copy
import argparse
import gc
from functools import partial
import torch
import peft
# autocast是PyTorch中一种混合精度的技术，可在保持数值精度的情况下提高训练速度和减少显存占用。
from torch.cuda.amp import GradScaler
from transformers import AutoTokenizer, AutoConfig, AutoModel, get_scheduler, AutoModelForCausalLM
from utils.common_utils import *
from data_handle.data_loader import *
from glm_config import *


# 兼容不同 PyTorch 版本的 autocast API（2.x 使用 torch.amp.autocast("cuda")，早期版本使用 torch.cuda.amp.autocast）
try:
    from torch.amp import autocast as _amp_autocast  # type: ignore
    def autocast_cuda():
        return _amp_autocast("cuda")
except Exception:  # pragma: no cover
    from torch.cuda.amp import autocast as _cuda_autocast  # type: ignore
    def autocast_cuda():
        return _cuda_autocast()

pc = ProjectConfig()

def print_gpu_memory():
    """打印GPU显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
def clear_gpu_cache():
    """清理GPU缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def evaluate_model(model, dev_dataloader):
    """
    在测试集上评估当前模型的训练效果。

    Args:
        model: 当前模型
        data_loader: 测试集的dataloader
    """
    model.eval()
    loss_list = []
    with torch.no_grad():
        for batch in dev_dataloader:
            if pc.use_lora:
                with autocast_cuda():
                    loss = model(
                        input_ids=batch['input_ids'].to(dtype=torch.long, device=pc.device),
                        labels=batch['labels'].to(dtype=torch.long, device=pc.device)
                    ).loss
            else:
                loss = model(
                    input_ids=batch['input_ids'].to(dtype=torch.long, device=pc.device),
                    labels=batch['labels'].to(dtype=torch.long, device=pc.device)
                ).loss
            loss_list.append(float(loss.cpu().detach()))
    model.train()
    return sum(loss_list) / len(loss_list)


def model2train():
    print("🚀 开始初始化模型和训练配置...")
    print_gpu_memory()
    
    # 初始化tokenizer
    print("📝 加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(pc.pre_model, trust_remote_code=True)

    # 初始化模型配置
    print("⚙️ 加载模型配置...")
    config = AutoConfig.from_pretrained(pc.pre_model, trust_remote_code=True)

    if pc.use_ptuning:
        config.pre_seq_len = pc.pre_seq_len
        config.prefix_projection = pc.prefix_projection
    
    # 加载模型
    print("🤖 加载ChatGLM-6B模型...")
    model = AutoModel.from_pretrained(
        pc.pre_model,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float16 if pc.fp16 else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    print("🔧 配置模型优化设置...")
    # 启用梯度检查点以节省显存
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    # 禁用缓存以节省显存
    model.config.use_cache = False

    if pc.use_ptuning:
        if pc.fp16:
            model.transformer.prefix_encoder.half()
        else:
            model.transformer.prefix_encoder.float()
    
    print(f'🎯 模型输出层: {model.lm_head}')
    
    if pc.use_lora:
        print("🔗 配置LoRA...")
        model.lm_head = CastOutputToFloat(model.lm_head)
        peft_config = peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            inference_mode=False,
            r=pc.lora_rank,
            lora_alpha=pc.lora_alpha,
            lora_dropout=pc.lora_dropout,
            target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]  # ChatGLM-6B的目标模块
        )
        model = peft.get_peft_model(model, peft_config)

    # 移动模型到设备
    if not hasattr(model, 'device') or model.device != torch.device(pc.device):
        model = model.to(pc.device)
    
    print('📊 模型训练参数统计:')
    if hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters()
    
    print_gpu_memory()

    # 配置优化器
    print("🎯 配置优化器...")
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": pc.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=pc.learning_rate)
    
    # 初始化混合精度训练
    scaler = GradScaler() if pc.fp16 and torch.cuda.is_available() else None
    
    # 加载数据
    print("📚 加载训练数据...")
    train_dataloader, dev_dataloader = get_data()
    
    # 计算训练步数
    num_update_steps_per_epoch = len(train_dataloader) // pc.gradient_accumulation_steps
    max_train_steps = pc.epochs * num_update_steps_per_epoch
    warm_steps = int(pc.warmup_ratio * max_train_steps)
    
    print(f"📈 训练配置:")
    print(f"  - 训练样本数: {len(train_dataloader.dataset)}")
    print(f"  - 验证样本数: {len(dev_dataloader.dataset)}")
    print(f"  - 批次大小: {pc.batch_size}")
    print(f"  - 梯度累积步数: {pc.gradient_accumulation_steps}")
    print(f"  - 有效批次大小: {pc.batch_size * pc.gradient_accumulation_steps}")
    print(f"  - 训练轮数: {pc.epochs}")
    print(f"  - 总训练步数: {max_train_steps}")
    print(f"  - 预热步数: {warm_steps}")
    
    # 配置学习率调度器
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=max_train_steps,
    )
    
    # 开始训练
    print("🎓 开始训练...")
    loss_list = []
    tic_train = time.time()
    global_step, best_eval_loss = 0, float('inf')
    
    model.train()
    
    for epoch in range(1, pc.epochs + 1):
        print(f"\n🔄 Epoch {epoch}/{pc.epochs}")
        print_gpu_memory()
        
        epoch_loss = []
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_dataloader):
            # 前向传播
            if pc.fp16 and scaler is not None:
                with autocast_cuda():
                    loss = model(
                        input_ids=batch['input_ids'].to(dtype=torch.long, device=pc.device),
                        labels=batch['labels'].to(dtype=torch.long, device=pc.device)
                    ).loss
                    # 梯度累积
                    loss = loss / pc.gradient_accumulation_steps
            else:
                loss = model(
                    input_ids=batch['input_ids'].to(dtype=torch.long, device=pc.device),
                    labels=batch['labels'].to(dtype=torch.long, device=pc.device)
                ).loss
                loss = loss / pc.gradient_accumulation_steps
            
            # 反向传播
            if pc.fp16 and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            epoch_loss.append(float(loss.cpu().detach()) * pc.gradient_accumulation_steps)
            
            # 梯度累积和优化器步骤
            if (step + 1) % pc.gradient_accumulation_steps == 0:
                if pc.fp16 and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # 记录日志
                if global_step % pc.logging_steps == 0:
                    time_diff = time.time() - tic_train
                    loss_avg = sum(epoch_loss[-pc.logging_steps*pc.gradient_accumulation_steps:]) / min(len(epoch_loss), pc.logging_steps*pc.gradient_accumulation_steps)
                    current_lr = lr_scheduler.get_last_lr()[0]
                    
                    print(f"📊 Step {global_step}/{max_train_steps} ({global_step/max_train_steps*100:.1f}%) | "
                          f"Epoch {epoch} | Loss: {loss_avg:.5f} | LR: {current_lr:.2e} | "
                          f"Speed: {pc.logging_steps/time_diff:.2f} step/s")
                    print_gpu_memory()
                    
                    tic_train = time.time()
                
                # 评估和保存模型
                if global_step % pc.eval_steps == 0:
                    print("🔍 开始评估...")
                    eval_loss = evaluate_model(model, dev_dataloader)
                    print(f"📈 Evaluation Loss: {eval_loss:.5f}")
                    
                    if eval_loss < best_eval_loss:
                        print(f"🎉 最佳模型更新: {best_eval_loss:.5f} → {eval_loss:.5f}")
                        best_eval_loss = eval_loss
                        best_model_dir = os.path.join(pc.save_dir, "model_best")
                        save_model(model, best_model_dir)
                        tokenizer.save_pretrained(best_model_dir)
                        print(f"💾 最佳模型已保存至: {best_model_dir}")
                    
                    model.train()  # 重新设置为训练模式
                    clear_gpu_cache()
                
                # 定期保存检查点
                if global_step % pc.save_freq == 0:
                    checkpoint_dir = os.path.join(pc.save_dir, f"checkpoint-{global_step}")
                    save_model(model, checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    print(f"💾 检查点已保存至: {checkpoint_dir}")
        
        # 每个epoch结束后的统计
        epoch_avg_loss = sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0
        print(f"✅ Epoch {epoch} 完成 | 平均损失: {epoch_avg_loss:.5f}")
        
        # 清理显存
        clear_gpu_cache()
    
    print("🎊 训练完成!")
    print(f"🏆 最佳验证损失: {best_eval_loss:.5f}")
    print_gpu_memory()


if __name__ == '__main__':
    model2train()