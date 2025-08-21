#!/usr/bin/env python3

"""
完美最终版防御测试脚本
修复了所有已知问题：
1. 路径问题 ✅
2. 返回值解包问题 ✅  
3. 参数设置问题 ✅
4. GPU监控功能 ✅
"""

import os
import subprocess
import sys
import time
import threading

def monitor_gpu():
    """实时监控GPU使用情况"""
    print("🔍 GPU监控线程启动...")
    while True:
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        used, total, util = parts[0], parts[1], parts[2]
                        used_gb = int(used) / 1024
                        total_gb = int(total) / 1024
                        usage_percent = (int(used) / int(total)) * 100
                        print(f"📊 GPU {i}: {used_gb:.1f}GB/{total_gb:.1f}GB ({usage_percent:.1f}%) 利用率:{util}%")
                print("-" * 60)
            time.sleep(15)  # 每15秒检查一次
        except:
            time.sleep(15)
            continue

def main():
    print("🛡️ CEM-att 完美最终防御测试")
    print("🔧 已修复所有已知问题 (路径+返回值+参数)")
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
    
    # 路径设置
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
    
    # 构建命令 - 完全匹配原始CEM-main
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
        "--gan_AE_type", "res_normN8C64",        # 大型GAN网络
        "--gan_loss_type", "SSIM",
        "--attack_epochs", "50",                 # 50个攻击epoch
        "--bottleneck_option", "noRELU_C8S1",
        "--folder", folder,
        "--var_threshold", "0.125",
        "--average_time", "1",                   # 重要：1次不是20次
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
    print(f"\n🚨 重要提示:")
    print(f"   1. 攻击过程应该消耗15-25GB显存")
    print(f"   2. 会看到50个epoch的GAN训练过程")
    print(f"   3. 最后输出MSE、SSIM、PSNR三个指标")
    
    print(f"\n🚀 开始防御效果测试...")
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
            print("📊 请查看上方的最终结果:")
            print("   ✅ MSE (均方误差) - 越低防御越好")
            print("   ✅ SSIM (结构相似度) - 越低攻击质量越差")
            print("   ✅ PSNR (峰值信噪比) - 越高隐私保护越好")
            print("")
            print("🎯 与原始CEM-main (GMM) 对比这些数值:")
            print("   - MSE更低 → Attention防御更强")
            print("   - SSIM更低 → 攻击重建质量更差")  
            print("   - PSNR更高 → 隐私保护更好")
            return True
        else:
            print(f"\n❌ 防御效果测试失败 (返回码: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"\n❌ 运行错误: {e}")
        return False

if __name__ == "__main__":
    print("🚨 完美最终版本 - 已修复所有已知问题!")
    print("🔧 修复内容:")
    print("   ✅ 路径拼接问题")
    print("   ✅ 返回值解包问题") 
    print("   ✅ 参数设置问题")
    print("   ✅ GPU监控功能")
    print("")
    
    success = main()
    
    if not success:
        print("\n💡 如果仍然失败，请检查:")
        print("   1. CUDA环境: nvidia-smi")
        print("   2. PyTorch CUDA版本")
        print("   3. 依赖包完整性")
        sys.exit(1)
    else:
        print("\n✅ 完美成功! 您已获得Attention vs GMM的完整对比数据!")
        print("📊 现在可以写论文分析Attention机制的防御优势了!")
