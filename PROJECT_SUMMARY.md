# ChatGLM-6B QLoRA 三元组抽取微调项目

## 项目概述

本项目实现了使用QLoRA技术对ChatGLM-6B模型进行微调，将其训练成专门的三元组抽取模型。项目包含完整的训练、推理和评估流程。

## 🎯 项目特点

- **高效微调**: 使用QLoRA (4-bit量化) 技术，大幅降低显存需求
- **专业任务**: 专门针对三元组抽取任务优化
- **完整流程**: 包含数据生成、训练、推理、评估的完整pipeline
- **易于使用**: 提供多种运行方式 (Shell脚本、Python脚本、配置文件)
- **问题修复**: 解决了Windows换行符和参数解析等常见问题

## 📁 项目结构

```
zjtz_work/
├── src/                          # 核心代码
│   ├── train_qlora.py           # QLoRA微调主程序
│   ├── inference.py             # 推理脚本
│   ├── evaluate.py              # 评估脚本
│   └── generate_triplet_data.py # 训练数据生成
├── scripts/                      # Shell脚本
│   ├── train.sh                 # 训练脚本
│   ├── inference.sh             # 推理脚本
│   └── evaluate.sh              # 评估脚本
├── data/                        # 数据目录
│   ├── train_triplet.jsonl      # 训练数据 (903条)
│   └── val_triplet.jsonl        # 验证数据 (101条)
├── config/                      # 配置文件
│   └── train_config.yaml        # 训练配置
├── output/                      # 输出目录
├── pure_python/                 # 纯Python版本
│   ├── train_triplet.py         # 训练脚本
│   ├── inference_triplet.py     # 推理脚本
│   └── evaluate_triplet.py      # 评估脚本
├── demo_simple.py               # 项目演示脚本
├── requirements.txt             # 依赖列表
├── README.md                    # 项目说明
├── TROUBLESHOOTING.md           # 故障排除指南
└── PROJECT_SUMMARY.md           # 项目总结 (本文件)
```

## 🚀 快速开始

### 1. 环境准备
```bash
# 安装依赖
pip install -r requirements.txt

# 或手动安装核心依赖
pip install torch transformers peft bitsandbytes datasets accelerate
```

### 2. 数据准备
```bash
# 生成训练数据
python src/generate_triplet_data.py
```

### 3. 模型训练

**方式1: 使用Shell脚本 (推荐)**
```bash
chmod +x scripts/train.sh
./scripts/train.sh
```

**方式2: 使用纯Python脚本**
```bash
python pure_python/train_triplet.py
```

**方式3: 直接运行**
```bash
python src/train_qlora.py \
    --model_name_or_path THUDM/chatglm-6b \
    --train_file data/train_triplet.jsonl \
    --validation_file data/val_triplet.jsonl \
    --output_dir output/chatglm-6b-triplet-qlora
```

### 4. 模型推理
```bash
# 使用Shell脚本
./scripts/inference.sh

# 或使用Python脚本
python pure_python/inference_triplet.py
```

### 5. 模型评估
```bash
# 使用Shell脚本
./scripts/evaluate.sh

# 或使用Python脚本
python pure_python/evaluate_triplet.py
```

## 🔧 技术特性

### QLoRA配置
- **量化**: 4-bit量化 (NF4)
- **LoRA参数**: r=8, alpha=32, dropout=0.1
- **目标模块**: query_key_value, dense, dense_h_to_4h, dense_4h_to_h

### 训练配置
- **批次大小**: 4 (可调整)
- **学习率**: 2e-4
- **训练轮数**: 3
- **梯度累积**: 4步
- **优化器**: AdamW
- **调度器**: 线性预热

### 数据格式
```json
{
    "instruction": "请从以下文本中抽取三元组，格式为(主体, 关系, 客体):",
    "input": "苹果公司是一家美国的科技公司，总部位于加利福尼亚州库比蒂诺。",
    "output": "(苹果公司, 是, 科技公司)\n(苹果公司, 总部位于, 加利福尼亚州库比蒂诺)\n(苹果公司, 国籍, 美国)"
}
```

## 🛠️ 问题修复

### 已解决的问题

1. **Windows换行符问题**
   - 错误: `$'\r': command not found`
   - 修复: 重新创建Shell脚本，使用Unix换行符

2. **参数解析问题**
   - 错误: `'THUDM/chatglm-6b\n'` 包含换行符
   - 修复: 对所有字符串参数使用 `.strip()` 方法

3. **Shell脚本长行问题**
   - 修复: 使用多行格式和反斜杠连接

4. **参数传递问题**
   - 修复: 使用引号包围所有变量

### 故障排除
详细的故障排除指南请参考 `TROUBLESHOOTING.md`

## 📊 性能指标

### 训练数据统计
- **训练集**: 903条样本
- **验证集**: 101条样本
- **数据类型**: 多领域三元组抽取任务

### 硬件要求
- **最低配置**: 12GB GPU显存
- **推荐配置**: 24GB+ GPU显存
- **内存**: 16GB+ RAM
- **存储**: 50GB+ 可用空间

### 训练时间估算
- **12GB GPU**: ~4-6小时 (batch_size=1)
- **24GB GPU**: ~2-3小时 (batch_size=4)
- **40GB+ GPU**: ~1-2小时 (batch_size=8)

## 🎮 使用示例

### 训练示例
```python
from src.train_qlora import main
import sys

# 设置参数
sys.argv = [
    'train_qlora.py',
    '--model_name_or_path', 'THUDM/chatglm-6b',
    '--train_file', 'data/train_triplet.jsonl',
    '--validation_file', 'data/val_triplet.jsonl',
    '--output_dir', 'output/my-triplet-model',
    '--num_train_epochs', '3'
]

# 开始训练
main()
```

### 推理示例
```python
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

# 加载模型
base_model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, "output/chatglm-6b-triplet-qlora")
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)

# 推理
text = "苹果公司是一家美国的科技公司。"
prompt = f"请从以下文本中抽取三元组，格式为(主体, 关系, 客体):\n{text}"

response, history = model.chat(tokenizer, prompt, history=[])
print(response)
```

## 📈 扩展建议

### 数据扩展
1. 增加更多领域的训练数据
2. 使用数据增强技术
3. 添加负样本训练

### 模型优化
1. 尝试不同的LoRA参数组合
2. 使用更大的基础模型 (ChatGLM2-6B, ChatGLM3-6B)
3. 实验不同的量化策略

### 功能扩展
1. 添加Web界面
2. 支持批量处理
3. 添加模型部署脚本
4. 集成到API服务

## 📝 更新日志

### v1.2 (最新)
- ✅ 修复Windows换行符问题
- ✅ 修复参数解析问题
- ✅ 添加故障排除指南
- ✅ 优化Shell脚本格式
- ✅ 添加参数验证测试

### v1.1
- ✅ 添加纯Python脚本版本
- ✅ 创建项目演示脚本
- ✅ 完善文档和README

### v1.0
- ✅ 实现QLoRA微调功能
- ✅ 生成三元组训练数据
- ✅ 创建推理和评估脚本
- ✅ 添加Shell脚本支持

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) - 基础模型
- [PEFT](https://github.com/huggingface/peft) - LoRA实现
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) - 量化支持
- [Transformers](https://github.com/huggingface/transformers) - 模型框架

---

**项目状态**: ✅ 完成并可用  
**最后更新**: 2025-10-27  
**维护者**: OpenHands AI Assistant