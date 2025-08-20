#!/bin/bash

# CEM-att 使用示例脚本
# 在Linux服务器上运行此脚本进行训练

echo "🚀 CEM-att 训练示例"
echo "=================================="

# 1. Attention分类器训练（新方法）
echo "1️⃣ 训练 CEM + Attention 分类器"
python main_MIA.py \
    --filename cem_attention_cifar10 \
    --arch vgg11_bn \
    --cutlayer 4 \
    --batch_size 128 \
    --num_epochs 50 \
    --learning_rate 0.01 \
    --lambd 1.0 \
    --dataset cifar10 \
    --use_attention_classifier \
    --num_slots 8 \
    --attention_heads 8 \
    --attention_dropout 0.1 \
    --log_entropy 1

echo "=================================="

# 2. 基线GMM分类器训练（对比）
echo "2️⃣ 训练 CEM + GMM 分类器（基线对比）"
python main_MIA.py \
    --filename cem_baseline_cifar10 \
    --arch vgg11_bn \
    --cutlayer 4 \
    --batch_size 128 \
    --num_epochs 50 \
    --learning_rate 0.01 \
    --lambd 1.0 \
    --dataset cifar10 \
    --log_entropy 1

echo "🎉 训练完成！检查 saves/ 文件夹查看结果"
