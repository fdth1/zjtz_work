# 🚀 ChatGLM-6B QLoRA 三元组抽取 - 20GB GPU 部署指南

本指南专门针对20GB GPU显存环境的部署和使用。

## 📋 环境要求

- **GPU**: 20GB+ 显存（如RTX 3090, RTX 4090, A100等）
- **系统内存**: 32GB+ 推荐
- **Python**: 3.8+
- **CUDA**: 11.7+
- **存储空间**: 50GB+ 可用空间

## 🔧 部署步骤

### 1. 确认模型位置

代码已配置为使用本地模型路径：
```
/root/.cache/modelscope/hub/models/ZhipuAI/ChatGLM-6B
```

**检查模型是否存在：**
```bash
ls -la /root/.cache/modelscope/hub/models/ZhipuAI/ChatGLM-6B/
```

如果模型不在此路径，请：
1. 移动模型到指定路径，或
2. 修改 `glm_config.py` 中的 `self.pre_model` 参数

### 2. 克隆项目

```bash
git clone https://github.com/fdth1/zjtz_work.git
cd zjtz_work
git checkout chatglm-6b-qlora-optimization
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 验证环境

```bash
# 检查模型路径
python check_model_path.py

# 运行基础测试
python test_basic.py
```

### 5. 开始训练

```bash
# 一键启动（推荐）
chmod +x start_training.sh
./start_training.sh

# 或手动启动
python train.py
```

## 📊 性能优化配置

项目已针对20GB显存进行优化：

### 内存优化设置
- **批次大小**: 1（最小化显存占用）
- **梯度累积**: 8步（保证训练效果）
- **有效批次大小**: 8
- **混合精度**: FP16（节省50%显存）
- **LoRA rank**: 8（平衡效果与显存）

### QLoRA配置
```python
# LoRA配置
lora_rank = 8
lora_alpha = 32
lora_dropout = 0.1

# 量化配置
load_in_4bit = True
bnb_4bit_compute_dtype = torch.float16
```

## 🔍 监控和调试

### 显存监控
```bash
# 实时监控GPU使用情况
nvidia-smi -l 1

# 查看详细显存使用
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1
```

### 训练监控
```bash
# 查看训练日志
tail -f training.log

# 监控训练进度
watch -n 1 'ls -la output/chatglm-6b-triplet-qlora/'
```

## 🚨 常见问题解决

### 1. 显存不足 (CUDA Out of Memory)

**解决方案：**
```python
# 在 glm_config.py 中进一步降低批次大小
self.batch_size = 1
self.gradient_accumulation_steps = 16  # 增加梯度累积

# 或启用更激进的优化
self.dataloader_num_workers = 0  # 减少数据加载器工作进程
```

### 2. 模型加载失败

**检查步骤：**
```bash
# 1. 验证模型路径
python check_model_path.py

# 2. 检查模型文件完整性
ls -la /root/.cache/modelscope/hub/models/ZhipuAI/ChatGLM-6B/

# 3. 测试模型加载
python -c "from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('/root/.cache/modelscope/hub/models/ZhipuAI/ChatGLM-6B', trust_remote_code=True); print('模型加载成功')"
```

### 3. 训练速度慢

**优化建议：**
- 确保使用SSD存储
- 增加系统内存
- 检查CPU使用率
- 优化数据预处理

## 📈 训练效果评估

### 训练过程监控
```bash
# 查看损失曲线
python -c "
import json
with open('output/chatglm-6b-triplet-qlora/trainer_state.json', 'r') as f:
    state = json.load(f)
    for log in state['log_history'][-10:]:
        if 'train_loss' in log:
            print(f'Step {log[\"step\"]}: Loss = {log[\"train_loss\"]:.4f}')
"
```

### 推理测试
```bash
# 交互式测试
python inference_triplet.py --interactive

# 批量测试
python inference_triplet.py --input_file test_data.txt --output_file results.json
```

## 🎯 生产环境部署

### 1. 模型保存
训练完成后，模型保存在：
```
output/chatglm-6b-triplet-qlora/model_best/
```

### 2. 推理服务
```bash
# 启动推理服务
python inference_triplet.py --server --port 8000

# 测试API
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "《娘家的故事第二部》是张玲执导的电视剧。"}'
```

### 3. 性能基准
在20GB GPU环境下的预期性能：
- **训练速度**: ~2-3 samples/second
- **推理速度**: ~10-15 samples/second
- **显存占用**: ~18-19GB（训练时）
- **显存占用**: ~8-10GB（推理时）

## 📝 日志和调试

### 重要日志文件
- `training.log`: 训练日志
- `output/chatglm-6b-triplet-qlora/trainer_state.json`: 训练状态
- `output/chatglm-6b-triplet-qlora/training_args.bin`: 训练参数

### 调试模式
```bash
# 启用详细日志
export TRANSFORMERS_VERBOSITY=debug
python train.py

# 启用CUDA调试
export CUDA_LAUNCH_BLOCKING=1
python train.py
```

## 🔄 更新和维护

### 更新代码
```bash
git pull origin chatglm-6b-qlora-optimization
```

### 清理缓存
```bash
# 清理PyTorch缓存
python -c "import torch; torch.cuda.empty_cache()"

# 清理Transformers缓存
rm -rf ~/.cache/huggingface/transformers/
```

## 📞 技术支持

如遇到问题，请提供以下信息：
1. GPU型号和显存大小
2. CUDA版本：`nvcc --version`
3. PyTorch版本：`python -c "import torch; print(torch.__version__)"`
4. 错误日志和堆栈跟踪
5. 系统配置：`nvidia-smi`

---

**注意**: 本配置已针对20GB显存环境优化，如果您的GPU显存更大或更小，可能需要调整批次大小和其他参数。