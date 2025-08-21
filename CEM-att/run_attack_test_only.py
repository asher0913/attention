#!/usr/bin/env python3

"""
仅运行攻击测试部分
适用于已经训练好模型，只需要测试攻击效果的情况
"""

import subprocess
import sys
import os
import argparse

def run_attack_test(checkpoint_path="./saves/cifar10/SCA_new_attention_fixed_lg1_thre0.125/", 
                   dataset="cifar10", use_attention=True):
    """
    运行攻击测试
    
    Args:
        checkpoint_path: 模型检查点路径
        dataset: 数据集名称
        use_attention: 是否使用attention分类器
    """
    
    print(f"🚀 开始攻击测试...")
    print(f"   📁 检查点路径: {checkpoint_path}")
    print(f"   📊 数据集: {dataset}")
    print(f"   🎯 使用Attention: {use_attention}")
    
    # 构建命令
    cmd = [
        "python", "main_test_MIA.py",
        "--model", "vgg11_bn_sgm",
        "--dataset", dataset,
        "--cutting_layer", "4",
        "--bottleneck_option", "noRELU_C8S1",
        "--regularization_option", "Gaussian_kl",
        "--regularization_strength", "0.025",
        "--AT_regularization_option", "SCA_new",
        "--AT_regularization_strength", "0.3",
        "--gan_AE_type", "res_normN4C64",
        "--gan_loss_type", "SSIM",
        "--checkpoint", checkpoint_path,
        "--device", "cpu",  # 本地测试用CPU
        "--batch_size", "128",
        "--lambd", "16",
        "--var_threshold", "0.125"
    ]
    
    # 添加attention相关参数
    if use_attention:
        cmd.extend([
            "--use_attention_classifier",
            "--num_slots", "8",
            "--attention_heads", "8",
            "--attention_dropout", "0.1"
        ])
    
    print(f"\n📋 执行命令:")
    print(f"   {' '.join(cmd)}")
    
    # 检查必要文件
    required_files = [
        "test_cifar10_image.pt",
        "test_cifar10_label.pt",
        "main_test_MIA.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ 缺少必要文件:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    # 检查检查点目录
    if not os.path.exists(checkpoint_path):
        print(f"\n⚠️  警告: 检查点目录不存在: {checkpoint_path}")
        print(f"   请确保训练已完成并保存了模型")
        return False
    
    try:
        print(f"\n🔄 正在运行攻击测试...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5分钟超时
        
        print(f"\n📊 攻击测试结果:")
        print(f"   返回码: {result.returncode}")
        
        if result.stdout:
            print(f"\n✅ 标准输出:")
            print(result.stdout)
        
        if result.stderr:
            print(f"\n⚠️  错误输出:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"\n🎉 攻击测试成功完成!")
            return True
        else:
            print(f"\n❌ 攻击测试失败!")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n⏰ 攻击测试超时 (5分钟)")
        return False
    except Exception as e:
        print(f"\n❌ 运行错误: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="运行CEM攻击测试")
    parser.add_argument("--checkpoint", default="./saves/cifar10/SCA_new_attention_fixed_lg1_thre0.125/", 
                       help="模型检查点路径")
    parser.add_argument("--dataset", default="cifar10", help="数据集")
    parser.add_argument("--no_attention", action="store_true", help="不使用attention分类器")
    
    args = parser.parse_args()
    
    print("🔍 CEM攻击测试工具")
    print("=" * 50)
    
    # 确保在正确目录
    if not os.path.exists("main_test_MIA.py"):
        print("❌ 错误: 请在CEM-att目录中运行此脚本")
        sys.exit(1)
    
    # 检查测试数据
    if not os.path.exists("test_cifar10_image.pt"):
        print("⚠️  测试数据不存在，正在生成...")
        try:
            subprocess.run(["python", "generate_test_data.py"], check=True)
            print("✅ 测试数据生成完成")
        except Exception as e:
            print(f"❌ 测试数据生成失败: {e}")
            sys.exit(1)
    
    # 运行攻击测试
    success = run_attack_test(
        checkpoint_path=args.checkpoint,
        dataset=args.dataset,
        use_attention=not args.no_attention
    )
    
    if success:
        print(f"\n🎯 攻击测试完成!")
        print(f"📊 请检查输出中的MSE、SSIM、PSNR指标")
    else:
        print(f"\n💡 如果遇到问题，请:")
        print(f"   1. 确保训练已完成")
        print(f"   2. 检查checkpoint路径是否正确")
        print(f"   3. 在Linux服务器上运行完整测试")

if __name__ == "__main__":
    main()
