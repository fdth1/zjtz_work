# ChatGLM-6B QLoRA 三元组抽取 - 快速开始

## 🚀 一键开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 生成训练数据
```bash
cd data && python generate_triplet_data.py && cd ..
```

### 3. 开始训练 (选择一种方式)

#### 方式1: 基础训练 (推荐新手)
```bash
python train_pure_python.py
```

#### 方式2: 自动配置训练 (推荐)
```bash
python train_with_config.py --config auto
```

#### 方式3: 交互式训练
```bash
python example_train.py
```

### 4. 测试模型
```bash
python inference_triplet.py
```

## 📋 训练配置选择

根据你的GPU显存选择合适的配置：

| GPU显存 | 命令 | 预估时间 |
|---------|------|----------|
| 8-12GB  | `python train_with_config.py --config small` | 4-6小时 |
| 12-24GB | `python train_with_config.py --config medium` | 2-3小时 |
| 24GB+   | `python train_with_config.py --config large` | 1-2小时 |
| 测试用  | `python train_with_config.py --config fast` | 30分钟 |

## 🔧 自定义参数

```bash
# 自定义训练参数
python train_with_config.py \
    --config medium \
    --epochs 5 \
    --batch_size 2 \
    --lora_r 16 \
    --learning_rate 1e-4
```

## 📁 项目结构

```
zjtz_work/
├── train_pure_python.py      # 基础训练脚本
├── train_with_config.py      # 配置化训练脚本
├── example_train.py          # 示例训练脚本
├── inference_triplet.py      # 推理脚本
├── evaluate_triplet.py       # 评估脚本
├── config/
│   └── train_config_simple.py  # 训练配置
├── data/
│   ├── generate_triplet_data.py # 数据生成
│   ├── train_triplet.jsonl      # 训练数据
│   └── val_triplet.jsonl        # 验证数据
└── output/                    # 模型输出目录
```

## 🎯 训练完成后

1. **测试模型**: `python inference_triplet.py`
2. **批量评估**: `python evaluate_triplet.py`
3. **查看模型**: `ls output/chatglm-6b-triplet-qlora/`

## ❓ 常见问题

### Q: 显存不足怎么办？
A: 使用更小的配置：`python train_with_config.py --config small --batch_size 1`

### Q: 训练数据不存在？
A: 运行：`cd data && python generate_triplet_data.py`

### Q: 模型下载失败？
A: 设置镜像：`export HF_ENDPOINT=https://hf-mirror.com`

### Q: 想要更快的训练？
A: 使用快速配置：`python train_with_config.py --config fast`

## 📖 详细文档

- [完整训练指南](PYTHON_TRAINING_GUIDE.md)
- [项目说明](README.md)
- [故障排除](TROUBLESHOOTING.md)

---

**开始训练只需要3个命令！** 🎉