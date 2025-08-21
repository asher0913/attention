#!/bin/bash

# 🛡️ 修复版防御效果测试脚本
# 使用昨晚训练好的模型进行攻击测试

echo "🛡️ CEM-att 防御效果测试 (使用已训练模型) - 修复版"
echo "=================================================="

# 设备配置
device="cuda"
GPU_id=0

# 模型路径 (根据实际发现的路径)
model_folder="saves/cifar10/SCA_new_attention_fixed_lg1_thre0.125/saves/cifar10/SCA_new_attention_fixed_lg1_thre0.125/CEM_log_entropy1_cifar10_cutlayer4_archvgg11_bn_sgm_schemeV2_epoch_n_epochs240_batch_size128_lr0.05_regulastr0.025_bottlenecknoRELU_C8S1_SCA_new0.3_randomseed125_ssim0.5_lambd16"

echo "📁 使用模型路径: $model_folder"

# 检查模型是否存在
if [ ! -d "$model_folder" ]; then
    echo "❌ 错误: 模型目录不存在!"
    echo "💡 检查可用模型:"
    find saves/ -name "*.pth" -o -name "*.pt" -o -name "*.log" | head -5
    exit 1
fi

echo "✅ 模型目录存在"

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

# 运行攻击测试 (移除了不认识的参数)
CUDA_VISIBLE_DEVICES=$GPU_id python main_test_MIA.py \
  --arch vgg11_bn_sgm \
  --cutlayer 4 \
  --batch_size 128 \
  --filename "${model_folder}/CEM_log_entropy1_cifar10_cutlayer4_archvgg11_bn_sgm_schemeV2_epoch_n_epochs240_batch_size128_lr0.05_regulastr0.025_bottlenecknoRELU_C8S1_SCA_new0.3_randomseed125_ssim0.5_lambd16" \
  --num_client 1 \
  --num_epochs 240 \
  --dataset cifar10 \
  --scheme V2_epoch \
  --regularization Gaussian_kl \
  --regularization_strength 0.025 \
  --log_entropy 1 \
  --AT_regularization SCA_new \
  --AT_regularization_strength 0.3 \
  --random_seed 125 \
  --gan_AE_type res_normN4C64 \
  --gan_loss_type SSIM \
  --attack_epochs 50 \
  --bottleneck_option noRELU_C8S1 \
  --folder "$model_folder" \
  --var_threshold 0.125 \
  --average_time 20 \
  --lambd 16 \
  --use_attention_classifier \
  --num_slots 8 \
  --attention_heads 8 \
  --attention_dropout 0.1 \
  --test_best

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
else
    echo ""
    echo "❌ 防御效果测试失败!"
    echo "💡 可能的解决方案:"
    echo "   1. 检查GPU: nvidia-smi"
    echo "   2. 检查模型文件完整性"
    echo "   3. 查看详细错误信息"
fi

echo ""
echo "📋 修复版防御测试脚本执行完毕"
