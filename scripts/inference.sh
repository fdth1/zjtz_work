#!/bin/bash

# ChatGLM-6B三元组抽取推理脚本

echo "ChatGLM-6B三元组抽取推理..."

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 模型路径
BASE_MODEL="THUDM/chatglm-6b"
LORA_MODEL="output/chatglm-6b-triplet-qlora"

# 检查LoRA模型是否存在
if [ -d "$LORA_MODEL" ]; then
    echo "使用微调后的模型进行推理..."
    python src/inference.py \
        --base_model $BASE_MODEL \
        --lora_model $LORA_MODEL \
        --interactive
else
    echo "LoRA模型不存在，使用基础模型进行推理..."
    python src/inference.py \
        --base_model $BASE_MODEL \
        --interactive
fi