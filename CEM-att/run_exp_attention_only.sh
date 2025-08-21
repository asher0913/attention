#!/bin/bash
# 完全复制CEM-main的run_exp.sh，只把GMM换成Attention
# cd "$(dirname "$0")"
# cd ../../
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
#Extra argement (store_true): --collude_use_public, --initialize_different  --collude_not_regularize  --collude_not_regularize --num_client_regularize ${num_client_regularize}

regularization='Gaussian_kl' #'Gaussian_Nonekl'
var_threshold=0.125
learning_rate=0.05
local_lr=-1
num_epochs=240
regularization_strength_list="0.025"  # 只测试一个参数
lambd_list="16" # 只测试lambda=16
log_entropy=1
folder_name="saves/cifar10/${AT_regularization}_attention_ONLY_lg${log_entropy}_thre${var_threshold}" ##the folder to save the model
bottleneck_option_list="noRELU_C8S1" #"noRELU_C8S1"
pretrain="False"

echo "🚨 运行CEM with Attention (替代GMM) - 完全相同的其他设置"
echo "📋 参数: lambda=${lambd_list}, reg_strength=${regularization_strength_list}"

for dataset in $dataset_list; do
        for lambd in $lambd_list; do
                for regularization_strength in $regularization_strength_list; do
                        for cutlayer in $cutlayer_list; do
                                for bottleneck_option in $bottleneck_option_list; do

                                        filename=pretrain_${pretrain}_lambd_${lambd}_noise_${regularization_strength}_epoch_${num_epochs}_bottleneck_${bottleneck_option}_log_${log_entropy}_ATstrength_${AT_regularization_strength}_lr_${learning_rate}_varthres_${var_threshold}
                                      
########################### training the model ########################### 
                                        echo "🚀 开始训练: Attention替代GMM的CEM算法"
                                        
                                        num_epochs=240
                                        learning_rate=0.05
                                        CUDA_VISIBLE_DEVICES=${GPU_id} python main_MIA.py --arch=${arch}  --cutlayer=$cutlayer --batch_size=${batch_size} \
                                        --filename=$filename --num_client=$num_client --num_epochs=$num_epochs \
                                        --dataset=$dataset --scheme=$scheme --regularization=${regularization} --regularization_strength=${regularization_strength} --log_entropy=${log_entropy} --AT_regularization=${AT_regularization} --AT_regularization_strength=${AT_regularization_strength}\
                                        --random_seed=$random_seed --learning_rate=$learning_rate --lambd=$lambd  --gan_AE_type ${train_gan_AE_type} --gan_loss_type ${gan_loss_type}\
                                        --local_lr $local_lr --bottleneck_option ${bottleneck_option} --folder ${folder_name} --ssim_threshold ${ssim_threshold} --var_threshold ${var_threshold}

########################### model inversion attack  ###########################
                                        echo "🛡️ 开始防御测试: 评估Attention vs GMM的防御效果"
                                        
                                        target_client=0
                                        attack_scheme=MIA
                                        attack_epochs=50
                                        average_time=1
                                        internal_C=64
                                        N=8
                                        test_gan_AE_type=res_normN${N}C${internal_C}

  
                                        CUDA_VISIBLE_DEVICES=${GPU_id} python main_test_MIA.py --arch=${arch}  --cutlayer=$cutlayer --batch_size=${batch_size} \
                                                --filename=$filename --num_client=$num_client --num_epochs=$num_epochs \
                                                --dataset=$dataset --scheme=$scheme --regularization=${regularization} --regularization_strength=${regularization_strength} --log_entropy=${log_entropy} --AT_regularization=${AT_regularization} --AT_regularization_strength=${AT_regularization_strength}\
                                                --random_seed=$random_seed --gan_AE_type ${train_gan_AE_type} --gan_loss_type ${gan_loss_type}\
                                                --attack_epochs=$attack_epochs --bottleneck_option ${bottleneck_option} --folder ${folder_name} --var_threshold ${var_threshold}\
                                                --average_time=$average_time --gan_AE_type ${test_gan_AE_type} --test_best
                                                                                
                                done
                        done
                done
        done
done

echo "✅ 完成! Attention替代GMM的CEM实验结束"
echo "📊 请对比结果中的MSE、SSIM、PSNR指标"
