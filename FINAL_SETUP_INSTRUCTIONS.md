# 🎯 最终部署说明 - 本地ChatGLM-6B模型配置

## 📋 当前配置状态

✅ **项目已完全优化并配置为使用您的本地ChatGLM-6B模型**

- **模型路径**: `/root/.cache/modelscope/hub/models/ZhipuAI/ChatGLM-6B`
- **GPU显存**: 针对20GB显存优化
- **训练配置**: QLoRA + FP16 + 梯度累积
- **项目分支**: `chatglm-6b-qlora-optimization`

## 🚀 立即开始使用

### 1. 克隆并切换到优化分支
```bash
git clone https://github.com/fdth1/zjtz_work.git
cd zjtz_work
git checkout chatglm-6b-qlora-optimization
```

### 2. 确认模型路径（重要！）
```bash
# 检查您的模型是否在预期位置
ls -la /root/.cache/modelscope/hub/models/ZhipuAI/ChatGLM-6B/

# 如果模型不在此路径，运行配置助手
python setup_model_path.py
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 一键启动训练
```bash
chmod +x start_training.sh
./start_training.sh
```

## 🔧 如果模型路径不同

如果您的ChatGLM-6B模型在不同位置，有三种解决方案：

### 方案1: 使用配置助手（推荐）
```bash
python setup_model_path.py
```
脚本会自动搜索系统中的ChatGLM模型并帮您配置。

### 方案2: 手动修改配置
编辑 `glm_config.py` 文件，修改第45行：
```python
# 将这行
self.pre_model = '/root/.cache/modelscope/hub/models/ZhipuAI/ChatGLM-6B'

# 改为您的实际路径
self.pre_model = '/your/actual/model/path'
```

### 方案3: 使用在线模型
如果您希望使用在线模型（需要网络下载）：
```python
self.pre_model = 'THUDM/chatglm-6b'
```

## 📊 预期性能表现

在20GB GPU环境下：
- **训练速度**: ~2-3 samples/second
- **显存占用**: ~18-19GB（训练时）
- **推理速度**: ~10-15 samples/second
- **显存占用**: ~8-10GB（推理时）

## 🧪 验证部署

### 检查环境
```bash
python test_basic.py
```

### 检查模型路径
```bash
python check_model_path.py
```

### 测试训练功能
```bash
python test_training.py
```

## 🎯 三元组抽取测试

训练完成后，测试三元组抽取：

```bash
# 交互式测试
python inference_triplet.py --interactive

# 输入示例文本
《娘家的故事第二部》是张玲执导，林在培、何赛飞等主演的电视剧。
```

预期输出：
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

## 🚨 故障排除

### 问题1: 模型路径不存在
```bash
# 运行配置助手
python setup_model_path.py
```

### 问题2: 显存不足
编辑 `glm_config.py`，降低批次大小：
```python
self.batch_size = 1
self.gradient_accumulation_steps = 16  # 增加梯度累积
```

### 问题3: 依赖包问题
```bash
# 重新安装依赖
pip install -r requirements.txt --force-reinstall
```

### 问题4: CUDA版本不匹配
```bash
# 检查CUDA版本
nvcc --version
nvidia-smi

# 根据CUDA版本安装对应的PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

## 📁 重要文件说明

| 文件 | 用途 |
|------|------|
| `glm_config.py` | 主配置文件，包含模型路径 |
| `train.py` | 主训练脚本 |
| `inference_triplet.py` | 三元组抽取推理 |
| `start_training.sh` | 一键启动脚本 |
| `setup_model_path.py` | 模型路径配置助手 |
| `check_model_path.py` | 模型路径验证 |

## 🎉 成功标志

当您看到以下输出时，说明部署成功：

```
✅ 所有检查通过，可以开始训练！
🚀 开始训练 ChatGLM-6B QLoRA 三元组抽取模型...
📊 训练配置:
   - 模型: /root/.cache/modelscope/hub/models/ZhipuAI/ChatGLM-6B
   - 批次大小: 1
   - 梯度累积: 8
   - 显存优化: 启用
```

## 📞 需要帮助？

1. **查看日志**: 检查 `training.log` 文件
2. **运行测试**: `python test_basic.py`
3. **检查配置**: `python check_model_path.py`
4. **重新配置**: `python setup_model_path.py`

---

**🎯 您的项目已经完全准备就绪！**

所有代码都已针对您的20GB GPU环境和本地ChatGLM-6B模型进行了优化。只需按照上述步骤操作，即可开始训练您的三元组抽取模型。