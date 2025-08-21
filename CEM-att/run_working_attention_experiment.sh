#!/bin/bash

# =======================================================================
# 🎯 修复后的 CEM + Attention 完整实验脚本
# 解决了所有已知问题：
# 1. ✅ 修复了attention模块维度不匹配 (feature_dim=8 for bottleneck)
# 2. ✅ 修复了main_test_MIA.py导入错误
# 3. ✅ 确保所有参数正确传递
# =======================================================================

echo "🎯 开始运行修复后的 CEM + Attention 实验..."

# =============================================
# 🎯 ATTENTION设置 - 固定启用attention
# =============================================
USE_ATTENTION=true
NUM_SLOTS=8
ATTENTION_HEADS=8
ATTENTION_DROPOUT=0.1
echo "✅ Attention参数: Slots=$NUM_SLOTS, Heads=$ATTENTION_HEADS, Dropout=$ATTENTION_DROPOUT"
echo "✅ 特征维度: 8 (VGG11+bottleneck后的维度)"
# =============================================

GPU_id=0
arch=vgg11_bn_sgm
batch_size=128
random_seed=125
cutlayer_list="4"
num_client=1

AT_regularization=SCA_new
AT_regularization_strength=0.3
ssim_threshold=0.5
train_gan_AE_type=res_normN4C64
gan_loss_type=SSIM

dataset_list="cifar10"
scheme=V2_epoch
random_seed_list="125"

regularization='Gaussian_kl'
var_threshold=0.125
learning_rate=0.05
local_lr=-1
num_epochs=240

# 简化参数用于测试 (用户已修改)
regularization_strength_list="0.025"
lambd_list="16"
log_entropy=1

# Attention专用文件夹名称
folder_name="saves/cifar10/${AT_regularization}_attention_fixed_lg${log_entropy}_thre${var_threshold}"
echo "📁 实验结果保存到: $folder_name"

bottleneck_option_list="noRELU_C8S1"

for random_seed in $random_seed_list
do
  for bottleneck_option in $bottleneck_option_list
  do
    for dataset in $dataset_list
    do
      for cutlayer in $cutlayer_list
      do
        for regularization_strength in $regularization_strength_list
        do
          for lambd in $lambd_list
          do
            echo "🚀 开始训练..."
            echo "   - 数据集: $dataset"
            echo "   - Lambda: $lambd" 
            echo "   - 正则化强度: $regularization_strength"
            echo "   - Cutlayer: $cutlayer"
            echo "   - 使用Attention分类器: $USE_ATTENTION"
            echo "   - 特征维度: 8 (bottleneck压缩后)"
            
            # 构建attention相关参数
            if [ "$USE_ATTENTION" = "true" ]; then
                attention_args="--use_attention_classifier --num_slots $NUM_SLOTS --attention_heads $ATTENTION_HEADS --attention_dropout $ATTENTION_DROPOUT"
                echo "   - ✅ Attention参数已启用"
            else
                attention_args=""
                echo "   - 使用原始GMM分类器"
            fi

            echo "📊 开始训练阶段..."
            CUDA_VISIBLE_DEVICES=$GPU_id python main_MIA.py \
              --arch $arch \
              --cutlayer $cutlayer \
              --batch_size $batch_size \
              --filename "${folder_name}/CEM_log_entropy${log_entropy}_${dataset}_cutlayer${cutlayer}_arch${arch}_scheme${scheme}_n_epochs${num_epochs}_batch_size${batch_size}_lr${learning_rate}_regulastr${regularization_strength}_bottleneck${bottleneck_option}_${AT_regularization}${AT_regularization_strength}_randomseed${random_seed}_ssim${ssim_threshold}_lambd${lambd}" \
              --num_client $num_client \
              --num_epochs $num_epochs \
              --dataset $dataset \
              --scheme $scheme \
              --regularization $regularization \
              --regularization_strength $regularization_strength \
              --log_entropy $log_entropy \
              --AT_regularization $AT_regularization \
              --AT_regularization_strength $AT_regularization_strength \
              --random_seed $random_seed \
              --learning_rate $learning_rate \
              --lambd $lambd \
              --gan_AE_type $train_gan_AE_type \
              --gan_loss_type $gan_loss_type \
              --local_lr $local_lr \
              --bottleneck_option $bottleneck_option \
              --folder $folder_name \
              --ssim_threshold $ssim_threshold \
              --var_threshold $var_threshold \
              $attention_args

            if [ $? -eq 0 ]; then
                echo "✅ 训练完成: Lambda=$lambd, 正则化=$regularization_strength"
            else
                echo "❌ 训练失败: Lambda=$lambd, 正则化=$regularization_strength"
                continue
            fi
            echo ""

            echo "🔍 开始攻击测试阶段..."
            # 设置攻击参数
            target_client=0
            attack_scheme=MIA
            attack_epochs=50
            average_time=20
            test_gan_AE_type=$train_gan_AE_type
            
            # 检查并生成测试数据
            if [ ! -f "./test_cifar10_image.pt" ] || [ ! -f "./test_cifar10_label.pt" ]; then
              echo "⚠️  测试数据不存在，正在生成..."
              python generate_test_data.py
              if [ $? -ne 0 ]; then
                echo "❌ 测试数据生成失败"
                exit 1
              fi
              echo "✅ 测试数据生成完成"
            fi
            
            CUDA_VISIBLE_DEVICES=$GPU_id python main_test_MIA.py \
              --arch $arch \
              --cutlayer $cutlayer \
              --batch_size $batch_size \
              --filename "${folder_name}/CEM_log_entropy${log_entropy}_${dataset}_cutlayer${cutlayer}_arch${arch}_scheme${scheme}_n_epochs${num_epochs}_batch_size${batch_size}_lr${learning_rate}_regulastr${regularization_strength}_bottleneck${bottleneck_option}_${AT_regularization}${AT_regularization_strength}_randomseed${random_seed}_ssim${ssim_threshold}_lambd${lambd}" \
              --num_client $num_client \
              --num_epochs $num_epochs \
              --dataset $dataset \
              --scheme $scheme \
              --regularization $regularization \
              --regularization_strength $regularization_strength \
              --log_entropy $log_entropy \
              --AT_regularization $AT_regularization \
              --AT_regularization_strength $AT_regularization_strength \
              --random_seed $random_seed \
              --gan_AE_type $test_gan_AE_type \
              --gan_loss_type $gan_loss_type \
              --attack_epochs $attack_epochs \
              --bottleneck_option $bottleneck_option \
              --folder $folder_name \
              --var_threshold $var_threshold \
              --average_time $average_time \
              --test_best

            if [ $? -eq 0 ]; then
                echo "✅ 攻击测试完成: Lambda=$lambd, 正则化=$regularization_strength"
            else
                echo "❌ 攻击测试失败: Lambda=$lambd, 正则化=$regularization_strength"
            fi
            echo "=================================================="
          done
        done
      done
    done
  done
done

echo "🎉 修复后的 CEM + Attention 实验结束！"
echo "📊 结果保存在: $folder_name"
echo ""
echo "🔍 如果实验成功完成，您应该看到："
echo "   ✅ 训练准确度和损失日志"
echo "   ✅ 攻击测试的MSE、SSIM、PSNR指标"
echo "   ✅ 模型权重保存在checkpoint文件中"
