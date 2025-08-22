#!/bin/bash

# CEM直接损失融合对比实验启动脚本

set -e  # 遇到错误立即退出

echo "🚀 启动CEM直接损失融合对比实验"
echo "=================================================="

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📁 当前工作目录: $SCRIPT_DIR"

# 检查必要的项目文件夹
for dir in "CEM-main" "CEM-mix" "CEM-direct"; do
    if [ ! -d "$dir" ]; then
        echo "❌ 错误: 缺少必要的项目文件夹 $dir"
        exit 1
    fi
done

echo "✅ 项目文件夹检查完成"

# 设置Python环境
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# 检查Python脚本是否存在
if [ ! -f "compare_cem_direct_fusion.py" ]; then
    echo "❌ 错误: 找不到比较脚本 compare_cem_direct_fusion.py"
    exit 1
fi

echo "🧪 开始运行实验..."
echo "⚠️  注意: 完整实验可能需要几小时时间"
echo "💡 提示: 可以用 Ctrl+C 中断实验"
echo ""

# 运行比较脚本
python compare_cem_direct_fusion.py

echo ""
echo "🎉 实验完成！查看生成的Markdown报告了解详细结果。"
