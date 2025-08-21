#!/usr/bin/env python3

"""
带GPU监控的防御测试脚本
实时显示显存占用和攻击进度
"""

import os
import subprocess
import sys
import time
import threading

def monitor_gpu():
    """实时监控GPU使用情况"""
    print("🔍 开始GPU监控线程...")
    while True:
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    used, total = line.split(', ')
                    used_gb = int(used) / 1024
                    total_gb = int(total) / 1024
                    usage_percent = (int(used) / int(total)) * 100
                    print(f"📊 GPU {i}: {used_gb:.1f}GB / {total_gb:.1f}GB ({usage_percent:.1f}%)")
                print("-" * 50)
            time.sleep(10)  # 每10秒检查一次
        except:
            time.sleep(10)
            continue

def main():
    print("🛡️ CEM-att 防御测试 (带GPU监控)")
    print("=" * 60)
    
    # 检测并生成测试数据
    print("🔍 检查测试数据...")
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
    
    # 使用与原始CEM-main完全一致的参数
    folder = "saves/cifar10/SCA_new_attention_fixed_lg1_thre0.125"
    filename = "CEM_log_entropy1_cifar10_cutlayer4_archvgg11_bn_sgm_schemeV2_epoch_n_epochs240_batch_size128_lr0.05_regulastr0.025_bottlenecknoRELU_C8S1_SCA_new0.3_randomseed125_ssim0.5_lambd16"
    
    # 验证路径存在
    expected_path = f"{folder}/saves/cifar10/SCA_new_attention_fixed_lg1_thre0.125/{filename}"
    if not os.path.exists(expected_path):
        print(f"❌ 错误: 模型目录不存在: {expected_path}")
        return False
    
    print(f"✅ 模型目录确认存在")
    
    # 检查checkpoint文件
    checkpoint_best = f"{expected_path}/checkpoint_f_best.tar"
    if os.path.exists(checkpoint_best):
        print("✅ 使用 checkpoint_f_best.tar")
        test_best = True
    else:
        print("❌ 错误: 找不到checkpoint_f_best.tar")
        return False
    
    # 构建命令 - 使用原始CEM-main的确切参数
    cmd = [
        "python", "main_test_MIA.py",
        "--arch", "vgg11_bn_sgm",
        "--cutlayer", "4",
        "--batch_size", "128",
        "--filename", filename,
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
        "--gan_AE_type", "res_normN8C64",  # 重要：使用高显存的GAN
        "--gan_loss_type", "SSIM",
        "--attack_epochs", "50",           # 完整的50个攻击epoch
        "--bottleneck_option", "noRELU_C8S1",
        "--folder", folder,
        "--var_threshold", "0.125",
        "--average_time", "1",             # 修复：使用原始的1而不是20
        "--lambd", "16",
        "--use_attention_classifier",
        "--num_slots", "8",
        "--attention_heads", "8",
        "--attention_dropout", "0.1",
        "--test_best"
    ]
    
    # 设置环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    
    print(f"\n📋 执行命令:")
    print(f"CUDA_VISIBLE_DEVICES=0 {' '.join(cmd)}")
    
    print(f"\n🚀 开始防御效果测试...")
    print("⚠️  注意观察下方的GPU内存使用情况!")
    print("⚠️  如果显存占用很低(<5GB)，说明攻击没有正常运行!")
    print("=" * 60)
    
    # 启动GPU监控线程
    monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
    monitor_thread.start()
    
    try:
        # 运行测试
        result = subprocess.run(cmd, env=env)
        
        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("🎉 防御效果测试成功完成!")
            print("📊 如果看到了MSE、SSIM、PSNR结果，说明攻击正常运行")
            print("📊 如果显存峰值超过10GB，说明大型GAN网络正确加载")
            return True
        else:
            print(f"\n❌ 防御效果测试失败 (返回码: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"\n❌ 运行错误: {e}")
        return False

if __name__ == "__main__":
    print("🚨 重要提示:")
    print("1. 此脚本会实时显示GPU内存使用情况")
    print("2. 正常的攻击过程应该使用15-25GB显存")
    print("3. 如果显存占用很低，说明攻击提前退出或失败")
    print("4. 请观察显存变化来判断攻击是否正常运行")
    print("")
    
    success = main()
    
    if not success:
        print("\n💡 显存占用异常的可能原因:")
        print("   1. GAN生成器网络加载失败")
        print("   2. 攻击过程中出现错误提前退出")
        print("   3. 批处理大小不正确")
        print("   4. 某些模块在CPU而非GPU上运行")
        sys.exit(1)
    else:
        print("\n✅ 测试完成! 请检查显存峰值是否达到15GB+")
        print("📊 如果显存峰值很低，说明还有问题需要解决")
