# 🚨 模型路径问题解决方案

## 问题描述
```
OSError: /root/.cache/modelscope/hub/models/ZhipuAI/ChatGLM-6B does not appear to have a file named tokenization_chatglm.py
```

## 原因分析
本地模型路径不存在或模型文件不完整。

## 🔧 解决方案

### 方案1: 使用在线模型（推荐，快速解决）

1. **临时切换到在线模型**：
```bash
cd /root/rag_test/openhead
python setup_model_path.py
```
选择 "使用在线模型 (THUDM/chatglm-6b)" 选项。

2. **或者手动修改配置**：
编辑 `glm_config.py` 文件：
```python
# 将第45行改为：
self.pre_model = 'THUDM/chatglm-6b'
```

### 方案2: 下载完整的本地模型

#### 使用ModelScope下载：
```bash
# 安装ModelScope
pip install modelscope

# 下载ChatGLM-6B模型
python -c "
from modelscope import snapshot_download
model_dir = snapshot_download('ZhipuAI/ChatGLM-6B', cache_dir='/root/.cache/modelscope')
print(f'模型下载到: {model_dir}')
"
```

#### 使用Hugging Face下载：
```bash
# 安装git-lfs
apt-get update && apt-get install -y git-lfs

# 克隆模型
mkdir -p /root/.cache/modelscope/hub/models/ZhipuAI/
cd /root/.cache/modelscope/hub/models/ZhipuAI/
git lfs clone https://huggingface.co/THUDM/chatglm-6b ChatGLM-6B
```

### 方案3: 使用现有模型路径

如果您已经有ChatGLM-6B模型在其他位置：

1. **查找现有模型**：
```bash
find / -name "config.json" -path "*/chatglm*" 2>/dev/null
find / -name "tokenization_chatglm.py" 2>/dev/null
```

2. **使用配置助手**：
```bash
python setup_model_path.py
```

3. **手动配置路径**：
编辑 `glm_config.py`，将 `self.pre_model` 设置为实际路径。

## 🚀 快速测试

### 测试在线模型配置：
```bash
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm-6b', trust_remote_code=True)
print('✅ 在线模型配置成功')
"
```

### 测试本地模型配置：
```bash
python check_model_path.py
```

## 📊 模型大小参考

- **ChatGLM-6B**: ~13GB (完整模型)
- **下载时间**: 取决于网络速度，通常10-30分钟
- **存储空间**: 确保有至少20GB可用空间

## 🎯 推荐流程

1. **立即解决**: 使用在线模型开始训练
2. **后台下载**: 同时下载本地模型到指定路径
3. **切换配置**: 下载完成后切换到本地模型

## 💡 注意事项

- 在线模型首次使用时会自动下载到缓存
- 本地模型可以避免网络依赖，推理速度更快
- 确保有足够的磁盘空间和网络带宽

## 🔄 验证步骤

配置完成后运行：
```bash
python test_basic.py
python check_model_path.py
```

看到 "✅ 模型路径验证通过" 即表示配置成功。