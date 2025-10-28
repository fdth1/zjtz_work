# 🚀 ChatGLM-6B QLoRA 三元组抽取 - 快速开始

## 📋 环境要求

- **GPU**: 20GB显存（推荐RTX 3090/4090, A100等）
- **内存**: 32GB以上
- **Python**: 3.8+
- **CUDA**: 11.7+

## ⚡ 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 一键启动训练

```bash
./start_training.sh
```

或者手动运行：

```bash
# 测试环境
python test_basic.py

# 开始训练
python train.py
```

### 3. 监控训练

```bash
# 监控GPU使用情况
nvidia-smi -l 1

# 查看训练日志（如果使用后台训练）
tail -f training.log
```

### 4. 推理测试

```bash
# 交互式推理
python inference_triplet.py --interactive

# 单次推理
python inference_triplet.py --text "《娘家的故事第二部》是张玲执导，林在培、何赛飞等主演的电视剧。"
```

## 🔧 配置调整

如果遇到显存不足，可以调整 `glm_config.py` 中的参数：

```python
# 减少序列长度
self.max_source_seq_len = 200  # 从256减少
self.max_target_seq_len = 100  # 从128减少

# 减少LoRA rank
self.lora_rank = 4  # 从8减少

# 增加梯度累积步数
self.gradient_accumulation_steps = 16  # 从8增加
```

## 📊 训练配置（20GB显存优化）

| 参数 | 值 | 说明 |
|------|----|----|
| 批次大小 | 1 | 最小批次以节省显存 |
| 梯度累积 | 8 | 模拟8倍批次大小 |
| 序列长度 | 256+128 | 输入+输出总长度384 |
| LoRA rank | 8 | 平衡效果和显存 |
| 混合精度 | FP16 | 减少显存占用 |
| 学习率 | 2e-4 | 适合小批次训练 |

## 🎯 预期效果

训练完成后，模型能够从文本中抽取三元组，格式如下：

**输入**：
```
《娘家的故事第二部》是张玲执导，林在培、何赛飞等主演的电视剧。
```

**输出**：
```json
[
  {
    "predicate": "导演",
    "object_type": "人物", 
    "subject_type": "影视作品",
    "object": "张玲",
    "subject": "娘家的故事第二部"
  }
]
```

## 🆘 常见问题

### Q: CUDA Out of Memory
**A**: 减少序列长度或LoRA rank，增加梯度累积步数

### Q: 训练速度慢
**A**: 确保使用GPU，检查数据加载器设置

### Q: 模型效果不好
**A**: 增加训练轮数，调整学习率，检查数据质量

## 📞 技术支持

1. 运行 `python test_basic.py` 检查环境
2. 查看 `README.md` 获取详细文档
3. 检查训练日志中的错误信息