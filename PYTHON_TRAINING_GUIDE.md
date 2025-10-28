# ChatGLM-6B QLoRA 纯Python训练指南

本指南介绍如何使用纯Python脚本进行ChatGLM-6B的QLoRA微调，无需shell脚本。

## 🚀 快速开始

### 1. 环境准备
```bash
# 安装依赖
pip install -r requirements.txt

# 生成训练数据
cd data && python generate_triplet_data.py && cd ..
```

### 2. 选择训练方式

#### 方式1: 基础训练 (推荐新手)
```python
python train_pure_python.py
```

#### 方式2: 配置化训练 (推荐进阶用户)
```python
# 自动检测GPU配置
python train_with_config.py --config auto

# 手动选择配置
python train_with_config.py --config medium --epochs 2 --batch_size 4
```

#### 方式3: 示例训练 (推荐学习)
```python
python example_train.py
```

## 📋 训练脚本说明

### 1. `train_pure_python.py` - 基础训练脚本
- **特点**: 简单直接，参数固定
- **适用**: 新手用户，快速开始
- **配置**: 内置默认配置，无需修改

```python
# 主要配置
model_name = "THUDM/chatglm-6b"
batch_size = 4
epochs = 3
lora_r = 8
learning_rate = 2e-4
```

### 2. `train_with_config.py` - 配置化训练脚本
- **特点**: 灵活配置，多种预设
- **适用**: 进阶用户，需要调参
- **配置**: 支持命令行参数和配置文件

```bash
# 使用预设配置
python train_with_config.py --config small    # 小显存GPU (12GB以下)
python train_with_config.py --config medium   # 中等显存GPU (12-24GB)
python train_with_config.py --config large    # 大显存GPU (24GB+)
python train_with_config.py --config fast     # 快速测试

# 自定义参数
python train_with_config.py --config medium --epochs 5 --batch_size 2 --lora_r 16
```

### 3. `example_train.py` - 示例训练脚本
- **特点**: 交互式选择，多种示例
- **适用**: 学习用户，了解不同配置
- **配置**: 包含4种不同的训练示例

## ⚙️ 配置详解

### GPU显存配置建议

| GPU显存 | 推荐配置 | 批次大小 | LoRA Rank | 预估训练时间 |
|---------|----------|----------|-----------|--------------|
| 8-12GB  | small    | 1-2      | 4-8       | 4-6小时      |
| 12-24GB | medium   | 4-8      | 8-16      | 2-3小时      |
| 24GB+   | large    | 8-16     | 16-32     | 1-2小时      |

### 配置文件 `config/train_config_simple.py`

```python
# 基础配置类
class TrainingConfig:
    MODEL_NAME = "THUDM/chatglm-6b"
    NUM_EPOCHS = 3
    BATCH_SIZE = 4
    LORA_R = 8
    LEARNING_RATE = 2e-4
    # ... 更多配置

# 预设配置
class SmallGPUConfig(TrainingConfig):
    BATCH_SIZE = 1
    LORA_R = 4
    # 适合小显存GPU

class MediumGPUConfig(TrainingConfig):
    BATCH_SIZE = 4
    LORA_R = 8
    # 适合中等显存GPU
```

## 🔧 自定义配置

### 1. 修改配置文件
编辑 `config/train_config_simple.py`:

```python
class MyCustomConfig(TrainingConfig):
    NUM_EPOCHS = 5          # 训练轮数
    BATCH_SIZE = 2          # 批次大小
    LORA_R = 16            # LoRA rank
    LEARNING_RATE = 1e-4   # 学习率
    OUTPUT_DIR = "output/my-model"  # 输出目录
```

### 2. 编程式配置
在Python代码中直接配置:

```python
from config.train_config_simple import TrainingConfig
from train_with_config import ChatGLMTrainer

# 创建配置
config = TrainingConfig()
config.NUM_EPOCHS = 2
config.BATCH_SIZE = 1
config.LORA_R = 4

# 开始训练
trainer = ChatGLMTrainer(config)
trainer.setup_model_and_tokenizer()
trainer.prepare_datasets()
trainer.train()
```

### 3. 命令行配置
使用命令行参数覆盖默认配置:

```bash
python train_with_config.py \
    --config medium \
    --epochs 5 \
    --batch_size 2 \
    --lora_r 16 \
    --learning_rate 1e-4 \
    --output_dir output/my-custom-model
```

## 📊 训练监控

### 1. 训练日志
训练过程中会显示详细日志:
```
INFO - 加载分词器...
INFO - 设置量化配置...
INFO - 加载基础模型...
INFO - 设置LoRA配置...
trainable params: 1,949,696 || all params: 6,245,533,696 || trainable%: 0.03
INFO - 准备训练数据集...
INFO - 加载了 903 条数据从 data/train_triplet.jsonl
INFO - 准备验证数据集...
INFO - 加载了 101 条数据从 data/val_triplet.jsonl
INFO - 开始训练...
```

### 2. 训练进度
```
{'loss': 2.1234, 'learning_rate': 0.0002, 'epoch': 0.1}
{'eval_loss': 1.8765, 'eval_runtime': 12.34, 'epoch': 0.5}
```

### 3. 模型保存
训练完成后模型保存在指定目录:
```
output/chatglm-6b-triplet-qlora/
├── adapter_config.json
├── adapter_model.bin
├── tokenizer_config.json
├── tokenizer.model
└── special_tokens_map.json
```

## 🐛 常见问题

### 1. 显存不足 (CUDA out of memory)
**解决方案**:
```python
# 减小批次大小
python train_with_config.py --config small --batch_size 1

# 或使用更小的LoRA参数
python train_with_config.py --lora_r 4
```

### 2. 模型下载失败
**解决方案**:
```bash
# 设置镜像源
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载模型到本地
git lfs clone https://huggingface.co/THUDM/chatglm-6b
```

### 3. 数据文件不存在
**解决方案**:
```bash
# 生成训练数据
cd data && python generate_triplet_data.py
```

### 4. 依赖包缺失
**解决方案**:
```bash
# 安装所有依赖
pip install -r requirements.txt

# 或单独安装
pip install torch transformers peft bitsandbytes datasets accelerate
```

## 🎯 训练技巧

### 1. 参数调优建议
- **LoRA Rank**: 从小开始 (4→8→16)，观察效果
- **学习率**: 2e-4 是好的起点，可尝试 1e-4 或 5e-4
- **批次大小**: 根据显存调整，保持有效批次大小 ≥ 8
- **训练轮数**: 3-5轮通常足够，避免过拟合

### 2. 显存优化
- 使用梯度检查点 (已默认开启)
- 减小序列长度
- 使用更小的LoRA参数
- 启用4bit量化 (已默认开启)

### 3. 训练加速
- 使用更大的批次大小
- 减少评估频率
- 使用FP16混合精度 (已默认开启)

## 📈 进阶用法

### 1. 多GPU训练
```bash
# 使用torchrun进行多GPU训练
torchrun --nproc_per_node=2 train_with_config.py --config large
```

### 2. 断点续训
```python
# 在配置中设置resume_from_checkpoint
config.resume_from_checkpoint = "output/chatglm-6b-triplet-qlora/checkpoint-500"
```

### 3. 自定义数据集
```python
# 修改数据文件路径
config.TRAIN_FILE = "path/to/your/train.jsonl"
config.VALIDATION_FILE = "path/to/your/val.jsonl"
```

## 🔍 验证训练结果

### 1. 快速测试
```python
python inference_triplet.py
```

### 2. 批量评估
```python
python evaluate_triplet.py
```

### 3. 交互式测试
```python
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

# 加载模型
base_model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, "output/chatglm-6b-triplet-qlora")
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)

# 测试推理
text = "苹果公司是一家美国的科技公司。"
prompt = f"请从以下文本中抽取三元组，格式为(主体, 关系, 客体):\n{text}"
response, history = model.chat(tokenizer, prompt, history=[])
print(response)
```

---

**总结**: 本指南提供了多种纯Python训练方式，从简单的一键训练到高度自定义的配置训练，满足不同用户的需求。建议新手从 `train_pure_python.py` 开始，熟悉后使用 `train_with_config.py` 进行更精细的调参。