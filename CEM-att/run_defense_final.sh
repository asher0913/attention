#!/bin/bash

# 🛡️ 最终修复版防御效果测试脚本
# 使用昨晚训练好的模型进行攻击测试

echo "🛡️ CEM-att 防御效果测试 (最终修复版)"
echo "=================================================="

# 设备配置
device="cuda"
GPU_id=0

# 模型目录 (基础路径)
model_base_folder="saves/cifar10/SCA_new_attention_fixed_lg1_thre0.125/saves/cifar10/SCA_new_attention_fixed_lg1_thre0.125/CEM_log_entropy1_cifar10_cutlayer4_archvgg11_bn_sgm_schemeV2_epoch_n_epochs240_batch_size128_lr0.05_regulastr0.025_bottlenecknoRELU_C8S1_SCA_new0.3_randomseed125_ssim0.5_lambd16"

# 检查可用的checkpoint文件
echo "🔍 检查可用的checkpoint文件..."
if [ -f "$model_base_folder/checkpoint_f_best.tar" ]; then
    checkpoint_type="best"
    echo "✅ 找到 checkpoint_f_best.tar"
elif [ -f "$model_base_folder/checkpoint_f_240.tar" ]; then
    checkpoint_type="240"
    echo "✅ 找到 checkpoint_f_240.tar，使用最后一个epoch"
else
    echo "❌ 错误: 找不到任何checkpoint文件!"
    echo "📋 可用文件:"
    ls -la "$model_base_folder"/checkpoint_f_*.tar 2>/dev/null || echo "   无checkpoint文件"
    exit 1
fi

echo "📁 使用模型目录: $model_base_folder"
echo "🎯 使用checkpoint: $checkpoint_type"

# 检查并生成测试数据
echo ""
echo "🔍 检查攻击测试数据..."
if [ ! -f "./test_cifar10_image.pt" ] || [ ! -f "./test_cifar10_label.pt" ]; then
    echo "⚠️  测试数据不存在，正在生成..."
    python generate_test_data.py
    if [ $? -ne 0 ]; then
        echo "❌ 测试数据生成失败"
        exit 1
    fi
    echo "✅ 测试数据生成完成"
else
    echo "✅ 测试数据已存在"
fi

echo ""
echo "🚀 开始防御效果测试..."
echo "=================================================="

# 构建正确的filename路径 (不要重复嵌套)
filename_path="CEM_log_entropy1_cifar10_cutlayer4_archvgg11_bn_sgm_schemeV2_epoch_n_epochs240_batch_size128_lr0.05_regulastr0.025_bottlenecknoRELU_C8S1_SCA_new0.3_randomseed125_ssim0.5_lambd16"

# 构建命令参数
cmd_args=(
    "--arch" "vgg11_bn_sgm"
    "--cutlayer" "4"
    "--batch_size" "128"
    "--filename" "$filename_path"
    "--num_client" "1"
    "--num_epochs" "240"
    "--dataset" "cifar10"
    "--scheme" "V2_epoch"
    "--regularization" "Gaussian_kl"
    "--regularization_strength" "0.025"
    "--log_entropy" "1"
    "--AT_regularization" "SCA_new"
    "--AT_regularization_strength" "0.3"
    "--random_seed" "125"
    "--gan_AE_type" "res_normN4C64"
    "--gan_loss_type" "SSIM"
    "--attack_epochs" "50"
    "--bottleneck_option" "noRELU_C8S1"
    "--folder" "$model_base_folder"
    "--var_threshold" "0.125"
    "--average_time" "20"
    "--lambd" "16"
    "--use_attention_classifier"
    "--num_slots" "8"
    "--attention_heads" "8"
    "--attention_dropout" "0.1"
)

# 根据checkpoint类型添加参数
if [ "$checkpoint_type" = "best" ]; then
    cmd_args+=("--test_best")
fi

echo "📋 执行命令:"
echo "CUDA_VISIBLE_DEVICES=$GPU_id python main_test_MIA.py ${cmd_args[*]}"

# 运行攻击测试
CUDA_VISIBLE_DEVICES=$GPU_id python main_test_MIA.py "${cmd_args[@]}"

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 防御效果测试完成!"
    echo "=================================================="
    echo "📊 关键防御指标:"
    echo "   ✅ MSE (均方误差) - 越低防御越好"
    echo "   ✅ SSIM (结构相似度) - 越低攻击质量越差"  
    echo "   ✅ PSNR (峰值信噪比) - 越高隐私保护越好"
    echo ""
    echo "🎯 Attention vs GMM防御效果对比:"
    echo "   如果MSE↓, SSIM↓, PSNR↑ → Attention防御更强!"
    echo ""
    echo "📁 使用的模型: $checkpoint_type epoch"
else
    echo ""
    echo "❌ 防御效果测试失败!"
    echo "💡 调试信息:"
    echo "   模型目录: $model_base_folder"
    echo "   Checkpoint: $checkpoint_type"
    echo "   可用文件:"
    ls -la "$model_base_folder"/checkpoint_f_*.tar 2>/dev/null || echo "   无checkpoint文件"
fi

echo ""
echo "📋 最终修复版防御测试脚本执行完毕"
