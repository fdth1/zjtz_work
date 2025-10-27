#!/bin/bash

# ChatGLM-6B三元组抽取模型评估脚本

echo "开始评估ChatGLM-6B三元组抽取模型..."

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 模型路径
BASE_MODEL="THUDM/chatglm-6b"
LORA_MODEL="output/chatglm-6b-triplet-qlora"
TEST_FILE="data/val_triplet.jsonl"
OUTPUT_FILE="evaluation_results.json"

# 检查LoRA模型是否存在
if [ -d "$LORA_MODEL" ]; then
    echo "评估微调后的模型..."
    python src/evaluate.py \
        --base_model $BASE_MODEL \
        --lora_model $LORA_MODEL \
        --test_file $TEST_FILE \
        --output_file $OUTPUT_FILE
else
    echo "LoRA模型不存在，评估基础模型..."
    python src/evaluate.py \
        --base_model $BASE_MODEL \
        --test_file $TEST_FILE \
        --output_file $OUTPUT_FILE
fi

echo "评估完成！结果保存在: $OUTPUT_FILE"