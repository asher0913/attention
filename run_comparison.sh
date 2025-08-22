#!/bin/bash

# CEM架构对比实验启动脚本
# 确保在CIFAR-10数据集上对比CEM-main和CEM-mix的性能

echo "🎯 CEM架构对比实验"
echo "========================================"
echo "📋 实验配置:"
echo "   - 数据集: CIFAR-10"  
echo "   - λ (lambda): 16"
echo "   - 正则化强度: 0.025"
echo "   - 训练轮数: 240"
echo "   - 对比: CEM-main (GMM) vs CEM-mix (GMM+Attention)"
echo "========================================"

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ Python未安装或不在PATH中"
    exit 1
fi

# 检查必要的项目目录
if [ ! -d "CEM-main" ]; then
    echo "❌ CEM-main目录不存在"
    exit 1
fi

if [ ! -d "CEM-mix" ]; then
    echo "❌ CEM-mix目录不存在"
    exit 1
fi

echo "✅ 环境检查通过"
echo ""

# 运行对比实验
echo "🚀 开始对比实验..."
python compare_cem_architectures.py

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 对比实验完成！"
    echo "📊 请查看生成的Markdown报告了解详细结果"
    echo ""
    echo "生成的文件:"
    echo "   📝 报告: CEM_Comparison_Report_*.md"
    echo "   📊 数据: cem_comparison_results_*.json"
    echo "   📋 日志: *_experiment_log.txt"
else
    echo ""
    echo "❌ 对比实验失败"
    echo "请检查错误信息和日志文件"
fi
