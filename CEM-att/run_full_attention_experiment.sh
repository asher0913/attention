#!/bin/bash

# =======================================================================
# 🎯 CEM + Attention 完整实验脚本 
# 这个脚本专门用于运行attention分类器的CEM实验
# =======================================================================

echo "🎯 开始运行 CEM + Attention 完整实验..."

# =============================================
# 🎯 ATTENTION设置 - 固定启用attention
# =============================================
USE_ATTENTION=true
NUM_SLOTS=8
ATTENTION_HEADS=8
ATTENTION_DROPOUT=0.1
echo "✅ Attention参数: Slots=$NUM_SLOTS, Heads=$ATTENTION_HEADS, Dropout=$ATTENTION_DROPOUT"
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
# regularization_strength_list="0.01 0.025 0.05 0.1 0.15"
# lambd_list="0 8 16"
regularization_strength_list="0.025"
lambd_list="16"
log_entropy=1

# Attention专用文件夹名称
folder_name="saves/cifar10/${AT_regularization}_attention_lg${log_entropy}_thre${var_threshold}"
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
            
            # 构建attention相关参数
            if [ "$USE_ATTENTION" = "true" ]; then
                attention_args="--use_attention_classifier --num_slots $NUM_SLOTS --attention_heads $ATTENTION_HEADS --attention_dropout $ATTENTION_DROPOUT"
                echo "   - Attention参数已启用"
            else
                attention_args=""
                echo "   - 使用原始GMM分类器"
            fi

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

            echo "✅ 训练完成: Lambda=$lambd, 正则化=$regularization_strength"
            echo ""

            echo "🔍 开始攻击测试..."
            # 设置攻击参数
            target_client=0
            attack_scheme=MIA
            attack_epochs=50
            average_time=20
            test_gan_AE_type=$train_gan_AE_type
            
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

            echo "✅ 攻击测试完成: Lambda=$lambd, 正则化=$regularization_strength"
            echo "=================================================="
          done
        done
      done
    done
  done
done

echo "🎉 CEM + Attention 完整实验结束！"
echo "📊 结果保存在: $folder_name"
