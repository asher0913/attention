#!/bin/bash

# 🛡️ CEM-att 防御效果测试脚本 (仅攻击测试阶段)
# 跳过训练，直接使用已保存的模型进行攻击测试

echo "🛡️ CEM-att 防御效果测试 (跳过训练阶段)"
echo "=================================================="

# 设备配置
device="cuda"  # 在Linux服务器上使用CUDA
GPU_id=0

# 数据集配置
dataset="cifar10"

# 模型配置
arch="vgg11_bn_sgm"
cutlayer=4
bottleneck_option="noRELU_C8S1"

# 正则化配置
regularization="Gaussian_kl"
regularization_strength=0.025
AT_regularization="SCA_new"
AT_regularization_strength=0.3

# 训练配置 (用于定位保存的模型)
num_epochs=240
batch_size=128
learning_rate=0.001
random_seed=125
log_entropy=1

# GAN配置
gan_AE_type="res_normN4C64"
gan_loss_type="SSIM"
ssim_threshold=0.5

# Attention配置
num_slots=8
attention_heads=8
attention_dropout=0.1
var_threshold=0.125

# Lambda值配置
lambd=16

# 攻击测试配置
attack_epochs=50
average_time=20

# 构建保存路径 (与训练时保持一致)
folder_name="saves/cifar10/SCA_new_attention_fixed_lg1_thre0.125"

echo "📁 使用已保存的模型: $folder_name"

# 检查模型文件是否存在
if [ ! -d "$folder_name" ]; then
    echo "❌ 错误: 模型目录不存在: $folder_name"
    echo "💡 请确保训练已完成并且模型已保存"
    echo "📋 可能的模型目录:"
    ls -la saves/cifar10/ 2>/dev/null || echo "   saves/cifar10/ 目录不存在"
    exit 1
fi

echo "✅ 找到模型目录: $folder_name"
echo "📂 目录内容:"
ls -la "$folder_name/" | head -10

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

# 运行攻击测试
CUDA_VISIBLE_DEVICES=$GPU_id python main_test_MIA.py \
  --arch $arch \
  --cutlayer $cutlayer \
  --batch_size $batch_size \
  --filename "${folder_name}/CEM_log_entropy${log_entropy}_${dataset}_cutlayer${cutlayer}_arch${arch}_scheme${scheme}_n_epochs${num_epochs}_batch_size${batch_size}_lr${learning_rate}_regulastr${regularization_strength}_bottleneck${bottleneck_option}_${AT_regularization}${AT_regularization_strength}_randomseed${random_seed}_ssim${ssim_threshold}_lambd${lambd}" \
  --num_client 1 \
  --num_epochs $num_epochs \
  --dataset $dataset \
  --scheme "V2_epoch" \
  --regularization $regularization \
  --regularization_strength $regularization_strength \
  --log_entropy $log_entropy \
  --AT_regularization $AT_regularization \
  --AT_regularization_strength $AT_regularization_strength \
  --random_seed $random_seed \
  --gan_AE_type $gan_AE_type \
  --gan_loss_type $gan_loss_type \
  --attack_epochs $attack_epochs \
  --bottleneck_option $bottleneck_option \
  --folder $folder_name \
  --var_threshold $var_threshold \
  --average_time $average_time \
  --lambd $lambd \
  --use_attention_classifier \
  --num_slots $num_slots \
  --attention_heads $attention_heads \
  --attention_dropout $attention_dropout \
  --test_best

# 检查测试结果
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 防御效果测试完成!"
    echo "=================================================="
    echo "📊 攻击测试结果:"
    echo "   - 模型路径: $folder_name"
    echo "   - Lambda值: $lambd"
    echo "   - 正则化强度: $regularization_strength"
    echo ""
    echo "🔍 请检查输出日志中的关键指标:"
    echo "   ✅ MSE (均方误差) - 越低越好"
    echo "   ✅ SSIM (结构相似度) - 越低越好"  
    echo "   ✅ PSNR (峰值信噪比) - 越高越好"
    echo ""
    echo "📁 详细结果保存在: $folder_name/"
else
    echo ""
    echo "❌ 防御效果测试失败!"
    echo "💡 可能的原因:"
    echo "   1. 模型文件损坏或不完整"
    echo "   2. 测试数据问题"
    echo "   3. GPU内存不足"
    echo "   4. 参数不匹配"
    echo ""
    echo "🔧 建议解决方案:"
    echo "   1. 检查模型目录: ls -la $folder_name/"
    echo "   2. 重新生成测试数据: python generate_test_data.py"
    echo "   3. 检查GPU状态: nvidia-smi"
fi

echo ""
echo "📋 防御效果测试脚本执行完毕"
echo "=================================================="
