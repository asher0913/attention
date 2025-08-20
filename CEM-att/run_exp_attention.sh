#!/bin/bash

# CEM-att 实验脚本 - 支持Attention机制
# 基于原始run_exp.sh，添加attention参数支持

GPU_id=0
arch=vgg11_bn  # 使用标准vgg11_bn（不用sgm版本）
batch_size=128
random_seed=125
cutlayer_list="4"
num_client=1

# 原始CEM参数
regularization='None'  # 简化，专注于attention vs GMM对比
var_threshold=0.1
learning_rate=0.01
local_lr=-1
num_epochs=50  # 减少epoch数进行快速测试
lambd_list="1 16"  # 测试不同的条件熵权重
log_entropy=1

# Attention相关参数
use_attention=true  # 设置为true使用attention，false使用GMM
num_slots=8
attention_heads=8
attention_dropout=0.1

dataset_list="cifar10"
scheme=V2_epoch
folder_name="saves/attention_vs_gmm_comparison"

echo "🚀 开始CEM-att实验"
echo "=================================="
echo "使用Attention机制: $use_attention"
echo "数据集: $dataset_list"
echo "Lambda值: $lambd_list"
echo "Epochs: $num_epochs"
echo "=================================="

for dataset in $dataset_list; do
    for lambd in $lambd_list; do
        for cutlayer in $cutlayer_list; do
            
            if [ "$use_attention" = "true" ]; then
                # 使用Attention分类器
                method="attention"
                filename=attention_lambd_${lambd}_epoch_${num_epochs}_slots_${num_slots}_heads_${attention_heads}
                
                echo "🎯 训练 CEM + Attention (lambd=$lambd)"
                CUDA_VISIBLE_DEVICES=${GPU_id} python main_MIA.py \
                    --arch=${arch} \
                    --cutlayer=$cutlayer \
                    --batch_size=${batch_size} \
                    --filename=$filename \
                    --num_client=$num_client \
                    --num_epochs=$num_epochs \
                    --dataset=$dataset \
                    --scheme=$scheme \
                    --regularization=${regularization} \
                    --log_entropy=${log_entropy} \
                    --random_seed=$random_seed \
                    --learning_rate=$learning_rate \
                    --lambd=$lambd \
                    --local_lr $local_lr \
                    --folder ${folder_name} \
                    --var_threshold ${var_threshold} \
                    --use_attention_classifier \
                    --num_slots ${num_slots} \
                    --attention_heads ${attention_heads} \
                    --attention_dropout ${attention_dropout}
            else
                # 使用原始GMM分类器（基线对比）
                method="gmm"
                filename=baseline_gmm_lambd_${lambd}_epoch_${num_epochs}
                
                echo "📊 训练 CEM + GMM 基线 (lambd=$lambd)"
                CUDA_VISIBLE_DEVICES=${GPU_id} python main_MIA.py \
                    --arch=${arch} \
                    --cutlayer=$cutlayer \
                    --batch_size=${batch_size} \
                    --filename=$filename \
                    --num_client=$num_client \
                    --num_epochs=$num_epochs \
                    --dataset=$dataset \
                    --scheme=$scheme \
                    --regularization=${regularization} \
                    --log_entropy=${log_entropy} \
                    --random_seed=$random_seed \
                    --learning_rate=$learning_rate \
                    --lambd=$lambd \
                    --local_lr $local_lr \
                    --folder ${folder_name} \
                    --var_threshold ${var_threshold}
                    # 注意：没有--use_attention_classifier参数
            fi
            
            echo "✅ 完成 $method 训练 (lambd=$lambd)"
            echo "--------------------------------"
            
        done
    done
done

echo "🎉 所有实验完成！"
echo "📁 结果保存在: $folder_name"
echo "🔍 检查logs对比attention vs GMM性能"
