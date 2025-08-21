#!/usr/bin/env python3

"""
简单有效的防御测试脚本
直接基于实际的模型路径结构
完全匹配原始CEM-main的测试流程
"""

import os
import subprocess
import sys
from pathlib import Path

def main():
    print("🛡️ CEM-att 简单防御测试 (完全匹配原始CEM-main)")
    print("=" * 60)
    
    # 直接使用实际存在的路径
    actual_folder = "saves/cifar10/SCA_new_attention_fixed_lg1_thre0.125"
    actual_filename = "CEM_log_entropy1_cifar10_cutlayer4_archvgg11_bn_sgm_schemeV2_epoch_n_epochs240_batch_size128_lr0.05_regulastr0.025_bottlenecknoRELU_C8S1_SCA_new0.3_randomseed125_ssim0.5_lambd16"
    
    # 构建实际的模型目录路径
    model_dir = f"{actual_folder}/saves/cifar10/SCA_new_attention_fixed_lg1_thre0.125/{actual_filename}"
    
    print(f"📁 实际模型目录: {model_dir}")
    
    # 检查目录是否存在
    if not os.path.exists(model_dir):
        print(f"❌ 错误: 模型目录不存在: {model_dir}")
        return False
    
    # 检查checkpoint文件
    checkpoint_files = {
        'best': f"{model_dir}/checkpoint_f_best.tar",
        '240': f"{model_dir}/checkpoint_f_240.tar"
    }
    
    test_best = False
    if os.path.exists(checkpoint_files['best']):
        print("✅ 使用 checkpoint_f_best.tar")
        test_best = True
    elif os.path.exists(checkpoint_files['240']):
        print("✅ 使用 checkpoint_f_240.tar")
        test_best = False
    else:
        print("❌ 错误: 找不到checkpoint文件")
        print(f"   查找位置: {model_dir}")
        available = [f for f in os.listdir(model_dir) if f.startswith('checkpoint_f_') and f.endswith('.tar')]
        print(f"   可用文件: {available}")
        return False
    
    # 检查测试数据
    print("\n🔍 检查测试数据...")
    test_files = ["test_cifar10_image.pt", "test_cifar10_label.pt"]
    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"⚠️  {test_file} 不存在，正在生成...")
            try:
                subprocess.run(["python", "generate_test_data.py"], check=True)
                print("✅ 测试数据生成完成")
                break
            except Exception as e:
                print(f"❌ 测试数据生成失败: {e}")
                return False
    else:
        print("✅ 测试数据已存在")
    
    # 构建命令 - 完全匹配原始CEM-main的格式
    cmd = [
        "python", "main_test_MIA.py",
        "--arch", "vgg11_bn_sgm",
        "--cutlayer", "4",
        "--batch_size", "128",
        "--filename", actual_filename,  # 这是关键！只传递文件名，不是完整路径
        "--num_client", "1", 
        "--num_epochs", "240",
        "--dataset", "cifar10",
        "--scheme", "V2_epoch",
        "--regularization", "Gaussian_kl",
        "--regularization_strength", "0.025",
        "--log_entropy", "1",
        "--AT_regularization", "SCA_new",
        "--AT_regularization_strength", "0.3",
        "--random_seed", "125",
        "--gan_AE_type", "res_normN8C64",
        "--gan_loss_type", "SSIM",
        "--attack_epochs", "50",
        "--bottleneck_option", "noRELU_C8S1",
        "--folder", actual_folder,  # 这是基础文件夹
        "--var_threshold", "0.125",
        "--average_time", "20",
        "--lambd", "16",
        "--use_attention_classifier",
        "--num_slots", "8",
        "--attention_heads", "8",
        "--attention_dropout", "0.1"
    ]
    
    if test_best:
        cmd.append("--test_best")
    
    # 设置环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    
    print(f"\n📋 执行命令:")
    print(f"CUDA_VISIBLE_DEVICES=0 {' '.join(cmd)}")
    
    print(f"\n🚀 开始防御效果测试...")
    print("=" * 60)
    
    try:
        # 运行测试
        result = subprocess.run(cmd, env=env)
        
        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("🎉 防御效果测试成功完成!")
            print("📊 请查看上方输出中的关键指标:")
            print("   ✅ MSE (均方误差) - 越低防御越好")
            print("   ✅ SSIM (结构相似度) - 越低攻击质量越差")
            print("   ✅ PSNR (峰值信噪比) - 越高隐私保护越好")
            print("\n🎯 与原始CEM-main (GMM) 对比这些数值即可评估Attention机制的防御效果!")
            return True
        else:
            print(f"\n❌ 防御效果测试失败 (返回码: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"\n❌ 运行错误: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n💡 如果仍然失败，请检查:")
        print("   1. GPU状态: nvidia-smi")
        print("   2. Python环境和PyTorch安装")
        print("   3. 模型文件完整性")
        sys.exit(1)
