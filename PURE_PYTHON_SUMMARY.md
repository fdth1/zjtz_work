# ChatGLM-6B QLoRA 纯Python训练方案总结

## 🎯 项目概述

本项目提供了完整的ChatGLM-6B QLoRA微调解决方案，专门用于三元组抽取任务。**完全使用纯Python实现，无需shell脚本**，解决了Windows/Linux环境兼容性问题。

## 📦 完整文件列表

### 核心训练脚本
- `train_pure_python.py` - 基础训练脚本，内置默认配置
- `train_with_config.py` - 配置化训练脚本，支持多种预设和自定义参数
- `example_train.py` - 交互式示例脚本，包含4种不同训练方式

### 配置文件
- `config/train_config_simple.py` - 训练配置类，包含5种预设配置
  - `TrainingConfig` - 默认配置
  - `SmallGPUConfig` - 小显存GPU配置 (8-12GB)
  - `MediumGPUConfig` - 中等显存GPU配置 (12-24GB)
  - `LargeGPUConfig` - 大显存GPU配置 (24GB+)
  - `FastTrainingConfig` - 快速测试配置

### 数据和推理
- `data/generate_triplet_data.py` - 训练数据生成脚本
- `data/train_triplet.jsonl` - 训练数据 (903条)
- `data/val_triplet.jsonl` - 验证数据 (101条)
- `inference_triplet.py` - 推理脚本
- `evaluate_triplet.py` - 评估脚本

### 演示和测试
- `demo_pure_python.py` - 纯Python演示脚本
- `test_python_training.py` - 测试脚本，验证所有组件
- `demo_simple.py` - 简单演示脚本

### 文档
- `QUICK_START.md` - 快速开始指南
- `PYTHON_TRAINING_GUIDE.md` - 详细训练指南
- `README.md` - 项目说明
- `TROUBLESHOOTING.md` - 故障排除指南

## 🚀 使用方式

### 1. 一键开始 (推荐新手)
```bash
# 安装依赖
pip install -r requirements.txt

# 生成数据
cd data && python generate_triplet_data.py && cd ..

# 开始训练
python train_pure_python.py
```

### 2. 配置化训练 (推荐进阶用户)
```bash
# 自动检测GPU配置
python train_with_config.py --config auto

# 手动选择配置
python train_with_config.py --config medium --epochs 3 --batch_size 4

# 小显存GPU
python train_with_config.py --config small

# 快速测试
python train_with_config.py --config fast
```

### 3. 交互式训练 (推荐学习)
```bash
python example_train.py
# 然后选择 1-5 中的任意选项
```

### 4. 编程式使用
```python
from config.train_config_simple import MediumGPUConfig
from train_with_config import ChatGLMTrainer

# 创建配置
config = MediumGPUConfig()
config.NUM_EPOCHS = 2
config.BATCH_SIZE = 2

# 开始训练
trainer = ChatGLMTrainer(config)
trainer.setup_model_and_tokenizer()
trainer.prepare_datasets()
trainer.train()
```

## ⚙️ 配置选择指南

| GPU显存 | 推荐配置 | 命令 | 批次大小 | LoRA Rank | 预估时间 |
|---------|----------|------|----------|-----------|----------|
| 8-12GB  | small    | `--config small` | 1 | 4 | 4-6小时 |
| 12-24GB | medium   | `--config medium` | 4 | 8 | 2-3小时 |
| 24GB+   | large    | `--config large` | 8 | 16 | 1-2小时 |
| 测试用  | fast     | `--config fast` | 2 | 8 | 30分钟 |
| 自动    | auto     | `--config auto` | 自动 | 自动 | 自动 |

## 🔧 核心特性

### 1. 纯Python实现
- ✅ 无需shell脚本
- ✅ 跨平台兼容 (Windows/Linux/macOS)
- ✅ 避免命令行参数解析问题
- ✅ 直接在Python中配置和运行

### 2. 多种配置方式
- 🎯 内置配置 - 开箱即用
- 🎯 预设配置 - 根据GPU选择
- 🎯 命令行配置 - 灵活调参
- 🎯 编程式配置 - 完全自定义

### 3. 智能化特性
- 🧠 自动GPU检测和配置推荐
- 🧠 显存使用估算
- 🧠 参数验证和错误提示
- 🧠 训练进度监控

### 4. 完整的工作流
- 📊 数据生成 → 模型训练 → 推理测试 → 性能评估
- 📊 支持断点续训
- 📊 支持多GPU训练
- 📊 完整的日志记录

## 📈 训练数据

### 数据规模
- **训练集**: 903条三元组抽取样本
- **验证集**: 101条三元组抽取样本
- **数据格式**: JSONL格式，包含instruction、input、output字段

### 数据示例
```json
{
  "instruction": "请从以下文本中抽取三元组，格式为(主体, 关系, 客体):",
  "input": "苹果公司是一家美国的科技公司。",
  "output": "(苹果公司, 是, 科技公司), (苹果公司, 国籍, 美国)"
}
```

### 数据类型
- 人物关系: 任职、学习、居住等
- 组织关系: 总部、成立、合作等
- 地理关系: 位于、属于、毗邻等
- 产品关系: 生产、开发、销售等

## 🎯 模型性能

### LoRA配置
- **Rank**: 4-32 (可调)
- **Alpha**: Rank的2-4倍
- **Target Modules**: query_key_value, dense, dense_h_to_4h, dense_4h_to_h
- **Dropout**: 0.1

### 量化配置
- **4bit量化**: 节省显存，保持性能
- **计算类型**: float16
- **量化类型**: NF4

### 训练参数
- **学习率**: 2e-4 (LoRA推荐值)
- **批次大小**: 1-16 (根据显存调整)
- **训练轮数**: 1-5轮
- **序列长度**: 输入512，输出256

## 🔍 验证和测试

### 自动测试
```bash
python test_python_training.py
# 验证所有组件是否正常工作
```

### 功能演示
```bash
python demo_pure_python.py
# 展示项目结构和使用方法
```

### 模型推理
```bash
python inference_triplet.py
# 测试训练好的模型
```

### 性能评估
```bash
python evaluate_triplet.py
# 批量评估模型性能
```

## 💡 最佳实践

### 1. 新手用户
1. 使用 `train_pure_python.py` 开始
2. 使用默认配置，无需修改参数
3. 关注训练日志，观察损失下降

### 2. 进阶用户
1. 使用 `train_with_config.py` 进行调参
2. 根据GPU显存选择合适配置
3. 监控验证损失，避免过拟合

### 3. 专业用户
1. 编程式配置，完全自定义
2. 使用多GPU训练加速
3. 实现自定义数据集和评估指标

## 🐛 常见问题解决

### 显存不足
```bash
# 使用小显存配置
python train_with_config.py --config small --batch_size 1

# 或减小序列长度
python train_with_config.py --config medium --max_source_length 256
```

### 训练速度慢
```bash
# 使用大显存配置
python train_with_config.py --config large

# 或增加批次大小
python train_with_config.py --config medium --batch_size 8
```

### 模型效果不好
```bash
# 增加训练轮数
python train_with_config.py --config medium --epochs 5

# 或增加LoRA参数
python train_with_config.py --config medium --lora_r 16
```

## 📚 相关文档

- [快速开始指南](QUICK_START.md) - 3分钟上手
- [详细训练指南](PYTHON_TRAINING_GUIDE.md) - 完整教程
- [故障排除指南](TROUBLESHOOTING.md) - 问题解决
- [项目说明](README.md) - 项目概述

## 🎉 总结

本项目提供了完整的ChatGLM-6B QLoRA微调解决方案，具有以下优势：

1. **纯Python实现** - 无需shell脚本，跨平台兼容
2. **多种使用方式** - 从简单到复杂，满足不同需求
3. **智能化配置** - 自动检测GPU，推荐最佳配置
4. **完整工作流** - 数据生成到模型评估的全流程
5. **详细文档** - 从快速开始到深度定制的完整指南

**开始使用只需要3个命令：**
```bash
pip install -r requirements.txt
cd data && python generate_triplet_data.py && cd ..
python train_with_config.py --config auto
```

🚀 **立即开始你的ChatGLM-6B三元组抽取模型训练之旅！**