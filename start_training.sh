#!/bin/bash

# ChatGLM-6B QLoRA 三元组抽取训练启动脚本

echo "🚀 ChatGLM-6B QLoRA 三元组抽取训练"
echo "=================================="

# 检查Python环境
echo "📋 检查Python环境..."
python --version
if [ $? -ne 0 ]; then
    echo "❌ Python未安装或不在PATH中"
    exit 1
fi

# 检查CUDA
echo "🔍 检查CUDA环境..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "⚠️ nvidia-smi未找到，可能没有CUDA环境"
fi

# 检查模型路径
echo ""
echo "🔍 检查模型路径..."
python check_model_path.py
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 模型路径检查失败"
    echo "💡 是否要运行模型路径配置助手? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        python setup_model_path.py
        if [ $? -ne 0 ]; then
            echo "❌ 模型路径配置失败"
            exit 1
        fi
        # 重新检查模型路径
        python check_model_path.py
        if [ $? -ne 0 ]; then
            echo "❌ 模型路径仍然无效"
            exit 1
        fi
    else
        echo "❌ 请手动配置模型路径后重试"
        exit 1
    fi
fi

# 运行基础测试
echo ""
echo "🧪 运行基础测试..."
python test_basic.py
if [ $? -ne 0 ]; then
    echo "❌ 基础测试失败，请检查环境配置"
    exit 1
fi

echo ""
echo "✅ 基础测试通过！"
echo ""

# 询问是否开始训练
read -p "是否开始训练？(y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🎓 开始训练..."
    echo "💡 提示: 使用 Ctrl+C 可以停止训练"
    echo "💡 提示: 可以在另一个终端运行 'nvidia-smi -l 1' 监控GPU使用情况"
    echo "💡 提示: 训练日志会实时显示，也可以重定向到文件: python train.py > training.log 2>&1"
    echo ""
    
    # 开始训练
    python train.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 训练完成！"
        echo "📁 模型保存在: output/chatglm-6b-triplet-qlora/"
        echo "🔍 可以使用以下命令进行推理测试:"
        echo "   python inference_triplet.py --interactive"
    else
        echo ""
        echo "❌ 训练过程中出现错误"
        exit 1
    fi
else
    echo "取消训练"
    echo ""
    echo "📝 手动训练命令:"
    echo "   python train.py"
    echo ""
    echo "📝 后台训练命令:"
    echo "   nohup python train.py > training.log 2>&1 &"
    echo "   tail -f training.log  # 查看训练日志"
fi