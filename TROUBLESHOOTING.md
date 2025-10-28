# 故障排除指南

## 常见问题及解决方案

### 1. Shell脚本换行符问题

**错误现象:**
```bash
scripts/train.sh: line 2: $'\r': command not found
scripts/train.sh: line 4: $'\r': command not found
```

**原因分析:**
- Windows系统使用CRLF (`\r\n`) 作为换行符
- Linux/Unix系统使用LF (`\n`) 作为换行符
- 在Linux环境下运行Windows格式的脚本会出现 `$'\r'` 错误

**解决方案:**
1. **已修复**: 重新创建了所有shell脚本，使用Unix换行符
2. **预防措施**: 使用多行格式编写长命令，避免单行过长
3. **手动修复**: 如果遇到类似问题，可以使用以下命令转换:
   ```bash
   # 方法1: 使用dos2unix (如果可用)
   dos2unix scripts/*.sh
   
   # 方法2: 使用sed
   sed -i 's/\r$//' scripts/*.sh
   
   # 方法3: 使用tr
   tr -d '\r' < scripts/train.sh > scripts/train_fixed.sh
   ```

### 2. 模型名称参数包含换行符

**错误现象:**
```
huggingface_hub.errors.HFValidationError: Repo id must use alphanumeric chars, '-', '_' or '.'. The name cannot start or end with '-' o'.'.' and the maximum length is 96: 'THUDM/chatglm-6b
```

**原因分析:**
- 参数传递过程中引入了换行符或空白字符
- 模型名称变成了 `'THUDM/chatglm-6b\n'` 而不是 `'THUDM/chatglm-6b'`

**解决方案:**
1. **已修复**: 在Python脚本中对所有字符串参数使用 `.strip()` 方法
2. **修复位置**: `src/train_qlora.py` 中的参数处理部分
   ```python
   # 修复前
   model_args = ModelArguments(model_name_or_path=args.model_name_or_path)
   
   # 修复后
   model_args = ModelArguments(model_name_or_path=args.model_name_or_path.strip())
   ```

### 3. Shell脚本参数传递问题

**最佳实践:**
1. **使用引号包围变量**:
   ```bash
   # 正确
   python src/train_qlora.py --model_name_or_path "$MODEL_NAME"
   
   # 错误 (可能导致空白字符问题)
   python src/train_qlora.py --model_name_or_path $MODEL_NAME
   ```

2. **使用多行格式编写长命令**:
   ```bash
   # 正确 - 多行格式
   python src/train_qlora.py \
       --model_name_or_path "$MODEL_NAME" \
       --train_file "$TRAIN_FILE" \
       --output_dir "$OUTPUT_DIR"
   
   # 避免 - 单行过长
   python src/train_qlora.py --model_name_or_path "$MODEL_NAME" --train_file "$TRAIN_FILE" --output_dir "$OUTPUT_DIR" --num_train_epochs $NUM_EPOCHS --per_device_train_batch_size $BATCH_SIZE
   ```

### 4. 依赖安装问题

**错误现象:**
```
ModuleNotFoundError: No module named 'torch'
```

**解决方案:**
```bash
# 安装基础依赖
pip install torch transformers peft bitsandbytes datasets accelerate

# 或使用requirements.txt
pip install -r requirements.txt
```

### 5. GPU内存不足

**错误现象:**
```
RuntimeError: CUDA out of memory
```

**解决方案:**
1. **减小批次大小**:
   ```bash
   # 在train.sh中修改
   BATCH_SIZE=2  # 从4减少到2
   GRAD_ACCUM=8  # 相应增加梯度累积步数
   ```

2. **使用更小的LoRA参数**:
   ```bash
   LORA_R=4      # 从8减少到4
   LORA_ALPHA=16 # 相应调整
   ```

3. **启用梯度检查点** (已默认启用):
   ```python
   gradient_checkpointing=True
   ```

### 6. 模型下载问题

**错误现象:**
```
ConnectionError: Failed to download model
```

**解决方案:**
1. **设置镜像源**:
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```

2. **手动下载模型**:
   ```bash
   # 使用git lfs下载
   git lfs install
   git clone https://huggingface.co/THUDM/chatglm-6b
   ```

3. **使用本地模型路径**:
   ```bash
   # 修改train.sh中的MODEL_NAME
   MODEL_NAME="/path/to/local/chatglm-6b"
   ```

## 使用建议

### 推荐的运行环境
- **GPU**: NVIDIA GPU with 24GB+ VRAM (如 RTX 3090, V100, A100)
- **内存**: 32GB+ RAM
- **存储**: 50GB+ 可用空间
- **CUDA**: 11.8+

### 训练参数调优
```bash
# 小显存GPU (12GB)
BATCH_SIZE=1
GRAD_ACCUM=16
LORA_R=4

# 中等显存GPU (24GB)
BATCH_SIZE=4
GRAD_ACCUM=4
LORA_R=8

# 大显存GPU (40GB+)
BATCH_SIZE=8
GRAD_ACCUM=2
LORA_R=16
```

### 快速测试
```bash
# 使用纯Python脚本进行快速测试
python train_triplet.py --help

# 使用演示脚本查看项目概览
python demo_simple.py

# 快速评估 (不需要训练)
python evaluate_triplet.py quick
```

## 联系支持

如果遇到其他问题，请检查:
1. Python版本 (推荐 3.8+)
2. CUDA版本兼容性
3. 依赖包版本冲突
4. 磁盘空间是否充足

更多问题可以查看项目的GitHub Issues或创建新的Issue。