#!/bin/bash

# 修改版run_exp.sh - 添加attention选项
# 在脚本顶部设置 USE_ATTENTION=true 来启用attention分类器

# =============================================
# 🎯 ATTENTION设置 - 修改这里来启用/禁用attention
# =============================================
USE_ATTENTION=true  # 改为true启用attention，false使用原始GMM
NUM_SLOTS=8
ATTENTION_HEADS=8
ATTENTION_DROPOUT=0.1
# =============================================

GPU_id=0
arch=vgg11_bn_sgm ##if sgm, pls change this term vgg11_bn_sgm
batch_size=128
random_seed=125
cutlayer_list="4"
num_client=1

AT_regularization=SCA_new #"gan_adv_step1;dropout0.2;gan_adv_step1_pruning180 nopeek"
AT_regularization_strength=0.3
ssim_threshold=0.5
train_gan_AE_type=res_normN4C64
gan_loss_type=SSIM

dataset_list="cifar10" # "svhn facescrub mnist"
scheme=V2_epoch
random_seed_list="125"

regularization='Gaussian_kl' #'Gaussian_Nonekl'
var_threshold=0.125
learning_rate=0.05
local_lr=-1
num_epochs=240
regularization_strength_list="0.01 0.025 0.05 0.1 0.15"
lambd_list="0 8 16" #
log_entropy=1

# 根据是否使用attention调整文件夹名称
if [ "$USE_ATTENTION" = "true" ]; then
    folder_name="saves/cifar10/${AT_regularization}_attention_lg${log_entropy}_thre${var_threshold}"
    echo "🎯 使用 Attention 分类器"
else
    folder_name="saves/cifar10/${AT_regularization}_infocons_sgm_lg${log_entropy}_thre${var_threshold}"
    echo "📊 使用 GMM 分类器 (原始方法)"
fi

bottleneck_option_list="noRELU_C8S1"
pretrain="False"

for dataset in $dataset_list; do
    for lambd in $lambd_list; do
        for regularization_strength in $regularization_strength_list; do
            for cutlayer in $cutlayer_list; do
                for bottleneck_option in $bottleneck_option_list; do

                    # 根据是否使用attention调整文件名
                    if [ "$USE_ATTENTION" = "true" ]; then
                        filename=attention_pretrain_${pretrain}_lambd_${lambd}_noise_${regularization_strength}_epoch_${num_epochs}_bottleneck_${bottleneck_option}_log_${log_entropy}_ATstrength_${AT_regularization_strength}_lr_${learning_rate}_varthres_${var_threshold}_slots_${NUM_SLOTS}
                    else
                        filename=pretrain_${pretrain}_lambd_${lambd}_noise_${regularization_strength}_epoch_${num_epochs}_bottleneck_${bottleneck_option}_log_${log_entropy}_ATstrength_${AT_regularization_strength}_lr_${learning_rate}_varthres_${var_threshold}
                    fi
                  
                    ########################### training the model ###########################
                    if [ "$pretrain" = "True" ]; then
                        num_epochs=80
                        learning_rate=0.0001
                        
                        # 构建基础命令
                        base_cmd="CUDA_VISIBLE_DEVICES=${GPU_id} python main_MIA.py --arch=${arch}  --cutlayer=$cutlayer --batch_size=${batch_size} \
                            --filename=$filename --num_client=$num_client --num_epochs=$num_epochs \
                            --dataset=$dataset --scheme=$scheme --regularization=${regularization} --regularization_strength=${regularization_strength} --log_entropy=${log_entropy} --AT_regularization=${AT_regularization} --AT_regularization_strength=${AT_regularization_strength} \
                            --random_seed=$random_seed --learning_rate=$learning_rate --lambd=${lambd}  --gan_AE_type ${train_gan_AE_type} --gan_loss_type ${gan_loss_type} \
                            --local_lr $local_lr --bottleneck_option ${bottleneck_option} --folder ${folder_name} --ssim_threshold ${ssim_threshold} --var_threshold ${var_threshold} --load_from_checkpoint --load_from_checkpoint_server"
                        
                        # 如果使用attention，添加attention参数
                        if [ "$USE_ATTENTION" = "true" ]; then
                            eval "$base_cmd --use_attention_classifier --num_slots ${NUM_SLOTS} --attention_heads ${ATTENTION_HEADS} --attention_dropout ${ATTENTION_DROPOUT}"
                        else
                            eval "$base_cmd"
                        fi
                    else
                        num_epochs=240
                        learning_rate=0.05
                        
                        # 构建基础命令
                        base_cmd="CUDA_VISIBLE_DEVICES=${GPU_id} python main_MIA.py --arch=${arch}  --cutlayer=$cutlayer --batch_size=${batch_size} \
                            --filename=$filename --num_client=$num_client --num_epochs=$num_epochs \
                            --dataset=$dataset --scheme=$scheme --regularization=${regularization} --regularization_strength=${regularization_strength} --log_entropy=${log_entropy} --AT_regularization=${AT_regularization} --AT_regularization_strength=${AT_regularization_strength} \
                            --random_seed=$random_seed --learning_rate=$learning_rate --lambd=$lambd  --gan_AE_type ${train_gan_AE_type} --gan_loss_type ${gan_loss_type} \
                            --local_lr $local_lr --bottleneck_option ${bottleneck_option} --folder ${folder_name} --ssim_threshold ${ssim_threshold} --var_threshold ${var_threshold}"
                        
                        # 如果使用attention，添加attention参数
                        if [ "$USE_ATTENTION" = "true" ]; then
                            eval "$base_cmd --use_attention_classifier --num_slots ${NUM_SLOTS} --attention_heads ${ATTENTION_HEADS} --attention_dropout ${ATTENTION_DROPOUT}"
                        else
                            eval "$base_cmd"
                        fi
                    fi
                    
                    ########################### model inversion attack  ###########################
                    target_client=0
                    attack_scheme=MIA
                    attack_epochs=50
                    average_time=1
                    internal_C=64
                    N=8
                    test_gan_AE_type=res_normN${N}C${internal_C}

                    # 构建测试命令（测试阶段不需要attention参数）
                    CUDA_VISIBLE_DEVICES=${GPU_id} python main_test_MIA.py --arch=${arch}  --cutlayer=$cutlayer --batch_size=${batch_size} \
                        --filename=$filename --num_client=$num_client --num_epochs=$num_epochs \
                        --dataset=$dataset --scheme=$scheme --regularization=${regularization} --regularization_strength=${regularization_strength} --log_entropy=${log_entropy} --AT_regularization=${AT_regularization} --AT_regularization_strength=${AT_regularization_strength} \
                        --random_seed=$random_seed --gan_AE_type ${train_gan_AE_type} --gan_loss_type ${gan_loss_type} \
                        --attack_epochs=$attack_epochs --bottleneck_option ${bottleneck_option} --folder ${folder_name} --var_threshold ${var_threshold} \
                        --average_time=$average_time --gan_AE_type ${test_gan_AE_type} --test_best
                                                                        
                done
            done
        done
    done
done
