# ChatGLM-6B QLoRA 三元组抽取微调

本项目使用QLoRA技术对ChatGLM-6B模型进行微调，训练一个专门用于三元组抽取的模型。

## 项目结构

```
zjtz_work/
├── data/                           # 数据目录
│   ├── generate_triplet_data.py   # 训练数据生成脚本
│   ├── train_triplet.jsonl        # 训练数据
│   └── val_triplet.jsonl          # 验证数据
├── src/                           # 源代码目录
│   ├── train_qlora.py            # QLoRA微调训练脚本
│   ├── inference.py              # 推理脚本
│   └── evaluate.py               # 评估脚本
├── config/                        # 配置文件目录
│   └── train_config.yaml         # 训练配置文件
├── scripts/                       # 脚本目录
│   ├── train.sh                  # 训练脚本
│   ├── inference.sh              # 推理脚本
│   └── evaluate.sh               # 评估脚本
├── models/                        # 模型目录（用于存放下载的模型）
├── output/                        # 输出目录（训练后的模型）
├── requirements.txt               # 依赖包列表
└── README.md                      # 项目说明文档
```

## 环境要求

- Python 3.8+
- CUDA 11.0+
- GPU内存 >= 12GB (推荐24GB+)

## 安装依赖

```bash
pip install -r requirements.txt
```

## 主要依赖包

- `torch>=2.0.0`: PyTorch深度学习框架
- `transformers>=4.30.0`: Hugging Face Transformers库
- `peft>=0.4.0`: Parameter-Efficient Fine-Tuning库
- `bitsandbytes>=0.39.0`: 量化训练库
- `datasets>=2.12.0`: 数据集处理库
- `accelerate>=0.20.0`: 分布式训练加速库

## 快速开始

### 1. 生成训练数据

```bash
cd data
python generate_triplet_data.py
```

这将生成约1000个三元组抽取的训练样本，包括：
- 训练集：903个样本 (`train_triplet.jsonl`)
- 验证集：101个样本 (`val_triplet.jsonl`)

### 2. 开始训练

```bash
# 使用脚本训练
bash scripts/train.sh

# 或者直接使用Python命令
python src/train_qlora.py \
    --model_name_or_path THUDM/chatglm-6b \
    --train_file data/train_triplet.jsonl \
    --validation_file data/val_triplet.jsonl \
    --output_dir output/chatglm-6b-triplet-qlora \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-4
```

### 3. 模型推理

```bash
# 交互式推理
bash scripts/inference.sh

# 单次推理
python src/inference.py \
    --base_model THUDM/chatglm-6b \
    --lora_model output/chatglm-6b-triplet-qlora \
    --text "马云是阿里巴巴的创始人，阿里巴巴总部位于杭州。"
```

### 4. 模型评估

```bash
# 评估模型性能
bash scripts/evaluate.sh

# 或者直接使用Python命令
python src/evaluate.py \
    --base_model THUDM/chatglm-6b \
    --lora_model output/chatglm-6b-triplet-qlora \
    --test_file data/val_triplet.jsonl
```

## 训练配置说明

### QLoRA配置
- `lora_r`: LoRA的秩，默认为8
- `lora_alpha`: LoRA的缩放参数，默认为32
- `lora_dropout`: LoRA的dropout率，默认为0.1
- `target_modules`: 目标模块，默认为["query_key_value"]

### 训练参数
- `num_train_epochs`: 训练轮数，默认为3
- `per_device_train_batch_size`: 每个设备的训练批次大小，默认为4
- `gradient_accumulation_steps`: 梯度累积步数，默认为4
- `learning_rate`: 学习率，默认为2e-4
- `warmup_steps`: 预热步数，默认为100

### 量化配置
- `load_in_4bit`: 使用4位量化，默认为True
- `bnb_4bit_compute_dtype`: 计算数据类型，默认为float16
- `bnb_4bit_use_double_quant`: 使用双重量化，默认为True
- `bnb_4bit_quant_type`: 量化类型，默认为"nf4"

## 数据格式

训练数据采用JSONL格式，每行一个JSON对象：

```json
{
    "instruction": "请从以下文本中抽取所有的三元组，格式为(主体, 关系, 客体)：",
    "input": "马云是阿里巴巴的创始人，阿里巴巴总部位于杭州。",
    "output": "(马云, 创立, 阿里巴巴)\n(阿里巴巴, 总部在, 杭州)"
}
```

## 支持的关系类型

当前数据集包含以下关系类型：
- 工作于：人员与公司的雇佣关系
- 担任：人员与职位的关系
- 位于：实体与地理位置的关系
- 毕业于：人员与教育机构的关系
- 创立：人员与公司的创建关系
- 总部在：公司与地理位置的关系
- 任职：人员与职位的关系

## 模型性能

评估指标包括：
- **精确率 (Precision)**: 预测正确的三元组占所有预测三元组的比例
- **召回率 (Recall)**: 预测正确的三元组占所有真实三元组的比例
- **F1分数**: 精确率和召回率的调和平均数
- **完全匹配准确率**: 完全正确预测所有三元组的样本比例

## 使用示例

### Python API使用

```python
from src.inference import TripletExtractor

# 初始化抽取器
extractor = TripletExtractor(
    base_model_path="THUDM/chatglm-6b",
    lora_model_path="output/chatglm-6b-triplet-qlora"
)

# 抽取三元组
text = "张三毕业于清华大学，现在在腾讯担任CTO职位。"
result = extractor.extract_triplets(text)
print(result)
# 输出: (张三, 毕业于, 清华大学)
#      (张三, 工作于, 腾讯)
#      (张三, 担任, CTO)
```

### 批量处理

```python
texts = [
    "马云是阿里巴巴的创始人。",
    "李四在小米担任产品经理。",
    "腾讯总部位于深圳。"
]

results = extractor.batch_extract(texts)
for text, result in zip(texts, results):
    print(f"输入: {text}")
    print(f"输出: {result}")
    print("-" * 50)
```

## 注意事项

1. **GPU内存要求**: 训练需要至少12GB GPU内存，推荐24GB+
2. **模型下载**: 首次运行会自动下载ChatGLM-6B模型（约12GB）
3. **训练时间**: 在单张V100上训练3个epoch大约需要1-2小时
4. **存储空间**: 确保有足够的磁盘空间存储模型和数据

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小batch_size
   - 增加gradient_accumulation_steps
   - 使用更小的max_length

2. **模型下载失败**
   - 检查网络连接
   - 使用镜像源或手动下载模型

3. **训练过程中断**
   - 检查磁盘空间
   - 查看日志文件定位问题

## 扩展功能

### 添加新的关系类型

1. 修改 `data/generate_triplet_data.py` 中的关系定义
2. 重新生成训练数据
3. 重新训练模型

### 使用自定义数据

1. 准备符合格式要求的JSONL文件
2. 修改训练脚本中的数据路径
3. 开始训练

## 许可证

本项目采用MIT许可证。

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 联系方式

如有问题，请通过GitHub Issue联系。
