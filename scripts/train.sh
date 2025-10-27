#!/bin/bash

# ChatGLM-6B QLoRA微调训练脚本

echo "开始ChatGLM-6B三元组抽取模型训练..."

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 训练参数
MODEL_NAME="THUDM/chatglm-6b"
TRAIN_FILE="data/train_triplet.jsonl"
VAL_FILE="data/val_triplet.jsonl"
OUTPUT_DIR="output/chatglm-6b-triplet-qlora"
NUM_EPOCHS=3
BATCH_SIZE=4
GRAD_ACCUM=4
LEARNING_RATE=2e-4
LORA_R=8
LORA_ALPHA=32
LORA_DROPOUT=0.1

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 执行训练
python src/train_qlora.py \
    --model_name_or_path $MODEL_NAME \
    --train_file $TRAIN_FILE \
    --validation_file $VAL_FILE \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --warmup_steps 100 \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 500

echo "训练完成！模型保存在: $OUTPUT_DIR"