#!/bin/bash

# 🛡️ 完全匹配原始CEM-main的防御测试脚本
# 保证与原始版本完全一致，只是GMM换成了Attention

echo "🛡️ CEM-att 防御测试 (完全匹配原始CEM-main)"
echo "=================================================="

# 完全匹配原始CEM-main的参数设置
GPU_id=0
arch=vgg11_bn_sgm
batch_size=128
random_seed=125
cutlayer=4
num_client=1

AT_regularization=SCA_new
AT_regularization_strength=0.3
ssim_threshold=0.5
train_gan_AE_type=res_normN4C64
gan_loss_type=SSIM

dataset=cifar10
scheme=V2_epoch
regularization='Gaussian_kl'
var_threshold=0.125
learning_rate=0.05
num_epochs=240
regularization_strength=0.025
lambd=16
log_entropy=1

# 关键：使用原始CEM-main的filename格式
filename="pretrain_False_lambd_${lambd}_noise_${regularization_strength}_epoch_${num_epochs}_bottleneck_noRELU_C8S1_log_${log_entropy}_ATstrength_${AT_regularization_strength}_lr_${learning_rate}_varthres_${var_threshold}"

# 原始的folder格式 
folder_name="saves/cifar10/${AT_regularization}_attention_fixed_lg${log_entropy}_thre${var_threshold}"

# 攻击测试参数
target_client=0
attack_scheme=MIA
attack_epochs=50
average_time=20
internal_C=64
N=8
test_gan_AE_type=res_normN${N}C${internal_C}

echo "📋 测试参数:"
echo "   架构: $arch"
echo "   数据集: $dataset"
echo "   Lambda: $lambd"
echo "   正则化强度: $regularization_strength"
echo "   文件名: $filename"
echo "   文件夹: $folder_name"

# 检查模型文件是否存在
model_path="${folder_name}/${filename}"
if [ ! -d "$model_path" ]; then
    echo "❌ 错误: 模型目录不存在: $model_path"
    echo ""
    echo "🔍 查找可用的模型目录..."
    find saves/ -type d -name "*attention*" | head -10
    echo ""
    echo "💡 如果路径不匹配，请检查训练时使用的folder_name"
    exit 1
fi

echo "✅ 模型目录存在: $model_path"

# 检查checkpoint文件
if [ -f "$model_path/checkpoint_f_best.tar" ]; then
    test_best_flag="--test_best"
    echo "✅ 使用 checkpoint_f_best.tar"
elif [ -f "$model_path/checkpoint_f_240.tar" ]; then
    test_best_flag=""
    echo "✅ 使用 checkpoint_f_240.tar (num_epochs=240)"
else
    echo "❌ 错误: 找不到checkpoint文件"
    echo "📂 可用文件:"
    ls -la "$model_path"/checkpoint_f_*.tar 2>/dev/null || echo "   无checkpoint文件"
    exit 1
fi

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

# 完全匹配原始CEM-main的main_test_MIA.py调用
CUDA_VISIBLE_DEVICES=${GPU_id} python main_test_MIA.py \
    --arch=${arch} \
    --cutlayer=$cutlayer \
    --batch_size=${batch_size} \
    --filename=$filename \
    --num_client=$num_client \
    --num_epochs=$num_epochs \
    --dataset=$dataset \
    --scheme=$scheme \
    --regularization=${regularization} \
    --regularization_strength=${regularization_strength} \
    --log_entropy=${log_entropy} \
    --AT_regularization=${AT_regularization} \
    --AT_regularization_strength=${AT_regularization_strength} \
    --random_seed=$random_seed \
    --gan_AE_type ${test_gan_AE_type} \
    --gan_loss_type ${gan_loss_type} \
    --attack_epochs=$attack_epochs \
    --bottleneck_option noRELU_C8S1 \
    --folder ${folder_name} \
    --var_threshold ${var_threshold} \
    --average_time=$average_time \
    --lambd=${lambd} \
    --use_attention_classifier \
    --num_slots=8 \
    --attention_heads=8 \
    --attention_dropout=0.1 \
    $test_best_flag

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
    echo "🎯 与原始CEM-main对比:"
    echo "   - 此脚本完全匹配原始CEM-main的测试流程"
    echo "   - 唯一区别：GMM → Attention机制"
    echo "   - 可直接对比防御性能差异"
    echo ""
    echo "📁 模型路径: $model_path"
else
    echo ""
    echo "❌ 防御效果测试失败!"
    echo "💡 调试信息:"
    echo "   模型路径: $model_path"
    echo "   文件名: $filename"
    echo "   检查是否与训练时的参数完全一致"
fi

echo ""
echo "📋 完全匹配原始CEM-main的防御测试脚本执行完毕"
