# ChatGLM-6B QLoRA 三元组抽取微调

基于ChatGLM-6B的QLoRA微调实现，专门用于三元组抽取任务，针对20GB显存服务器优化。

## 🚀 特性

- ✅ **20GB显存优化**: 针对20GB显存服务器进行了专门优化
- ✅ **QLoRA微调**: 使用QLoRA技术，大幅减少显存占用
- ✅ **三元组抽取**: 专门针对三元组抽取任务优化的数据格式和提示词
- ✅ **混合精度训练**: 支持FP16混合精度训练，进一步节省显存
- ✅ **梯度累积**: 通过梯度累积模拟更大的批次大小
- ✅ **显存监控**: 实时监控显存使用情况
- ✅ **跨平台兼容**: 支持Linux服务器环境

## 📋 环境要求

### 硬件要求
- GPU: 20GB显存（如RTX 3090, RTX 4090, A100等）
- 内存: 32GB以上推荐
- 存储: 50GB以上可用空间

### 软件要求
- Python 3.8+
- CUDA 11.7+
- PyTorch 1.13+

## 🛠️ 安装

1. **克隆项目**
```bash
git clone <repository_url>
cd zjtz_work
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **测试环境**
```bash
python test_training.py
```

## 📊 数据格式

训练数据应为JSONL格式，每行包含一个样本：

```json
{
  "context": "Instruction: 你现在是一个很厉害的阅读理解器，严格按照人类指令进行回答。\nInput: 帮我抽取出下面句子中的三元组信息，返回JSON：\n\n《娘家的故事第二部》是张玲执导，林在培、何赛飞等主演的电视剧。\nAnswer: ",
  "target": "```json\n[{\"predicate\": \"导演\", \"object_type\": \"人物\", \"subject_type\": \"影视作品\", \"object\": \"张玲\", \"subject\": \"娘家的故事第二部\"}]\n```"
}
```

## 🎯 使用方法

### 1. 配置参数

编辑 `glm_config.py` 中的配置：

```python
class ProjectConfig(object):
    def __init__(self):
        # 模型配置
        self.pre_model = 'THUDM/chatglm-6b'  # 预训练模型
        
        # 训练配置 - 20GB显存优化
        self.batch_size = 1                    # 批次大小
        self.gradient_accumulation_steps = 8   # 梯度累积步数
        self.epochs = 3                        # 训练轮数
        self.learning_rate = 2e-4              # 学习率
        
        # 序列长度配置 - 显存优化
        self.max_source_seq_len = 256          # 输入序列长度
        self.max_target_seq_len = 128          # 输出序列长度
        
        # LoRA配置
        self.lora_rank = 8                     # LoRA rank
        self.lora_alpha = 32                   # LoRA alpha
        
        # 显存优化
        self.fp16 = True                       # 混合精度训练
```

### 2. 开始训练

```bash
# 基础训练
python train.py

# 或者使用nohup在后台运行
nohup python train.py > training.log 2>&1 &
```

### 3. 监控训练

```bash
# 查看训练日志
tail -f training.log

# 查看GPU使用情况
nvidia-smi -l 1
```

### 4. 推理测试

```bash
# 交互式推理
python inference_triplet.py --interactive

# 单次推理
python inference_triplet.py --text "《娘家的故事第二部》是张玲执导，林在培、何赛飞等主演的电视剧。"

# 演示模式
python inference_triplet.py
```

## 📈 训练配置说明

### 显存优化策略

1. **小批次 + 梯度累积**: `batch_size=1` + `gradient_accumulation_steps=8`
2. **序列长度限制**: 输入256 + 输出128 = 总长度384
3. **混合精度训练**: FP16减少显存占用
4. **梯度检查点**: 用计算换显存
5. **LoRA微调**: 只训练少量参数

### 关键参数调整

| 参数 | 默认值 | 说明 | 调整建议 |
|------|--------|------|----------|
| `batch_size` | 1 | 批次大小 | 显存不足时保持1 |
| `gradient_accumulation_steps` | 8 | 梯度累积 | 可调整为4-16 |
| `max_source_seq_len` | 256 | 输入长度 | 根据数据调整 |
| `max_target_seq_len` | 128 | 输出长度 | 根据三元组复杂度调整 |
| `lora_rank` | 8 | LoRA秩 | 4-16，越大效果越好但显存越多 |
| `learning_rate` | 2e-4 | 学习率 | 可尝试1e-4到5e-4 |

## 🔧 故障排除

### 显存不足 (CUDA Out of Memory)

1. **减少批次大小**:
   ```python
   self.batch_size = 1  # 已经是最小值
   ```

2. **减少序列长度**:
   ```python
   self.max_source_seq_len = 200  # 从256减少到200
   self.max_target_seq_len = 100  # 从128减少到100
   ```

3. **减少LoRA rank**:
   ```python
   self.lora_rank = 4  # 从8减少到4
   ```

4. **启用更多优化**:
   ```python
   self.fp16 = True  # 确保启用混合精度
   ```

### 训练速度慢

1. **增加梯度累积步数**:
   ```python
   self.gradient_accumulation_steps = 16  # 从8增加到16
   ```

2. **调整数据加载器**:
   ```python
   self.dataloader_num_workers = 8  # 增加工作进程
   ```

### 模型效果不好

1. **增加训练轮数**:
   ```python
   self.epochs = 5  # 从3增加到5
   ```

2. **调整学习率**:
   ```python
   self.learning_rate = 1e-4  # 降低学习率
   ```

3. **增加LoRA rank**:
   ```python
   self.lora_rank = 16  # 从8增加到16（需要更多显存）
   ```

## 📁 项目结构

```
zjtz_work/
├── train.py                    # 主训练脚本
├── inference_triplet.py        # 推理脚本
├── test_training.py            # 测试脚本
├── glm_config.py              # 配置文件
├── requirements.txt           # 依赖列表
├── README.md                  # 使用说明
├── data/                      # 数据目录
│   ├── mixed_train_dataset.jsonl
│   └── mixed_dev_dataset.jsonl
├── data_handle/               # 数据处理模块
│   ├── data_loader.py
│   └── data_preprocess.py
├── utils/                     # 工具模块
│   └── common_utils.py
└── output/                    # 输出目录
    └── chatglm-6b-triplet-qlora/
        ├── model_best/        # 最佳模型
        └── checkpoint-*/      # 训练检查点
```

## 🎯 三元组抽取示例

**输入文本**:
```
《娘家的故事第二部》是张玲执导，林在培、何赛飞等主演的电视剧。
```

**输出结果**:
```json
[
  {
    "predicate": "导演",
    "object_type": "人物",
    "subject_type": "影视作品",
    "object": "张玲",
    "subject": "娘家的故事第二部"
  },
  {
    "predicate": "主演",
    "object_type": "人物",
    "subject_type": "影视作品",
    "object": "林在培",
    "subject": "娘家的故事第二部"
  },
  {
    "predicate": "主演",
    "object_type": "人物",
    "subject_type": "影视作品",
    "object": "何赛飞",
    "subject": "娘家的故事第二部"
  }
]
```

## 📞 技术支持

如果遇到问题，请：

1. 首先运行 `python test_training.py` 检查环境
2. 查看训练日志中的错误信息
3. 检查显存使用情况 `nvidia-smi`
4. 根据错误信息调整配置参数

## 📄 许可证

本项目基于MIT许可证开源。