#!/usr/bin/env python3
"""
CEM-direct 快速验证脚本
测试直接损失融合方法是否能正常运行（仅运行少量epoch进行验证）
"""

import subprocess
import sys
import os

def quick_test():
    """运行快速测试"""
    print("🚀 CEM-direct 快速验证测试")
    print("=" * 40)
    
    # 测试训练（仅5个epoch）
    print("📚 测试训练阶段...")
    train_cmd = [
        "python", "main_MIA.py",
        "--arch", "vgg11_bn_sgm",
        "--cutlayer", "4",
        "--batch_size", "32",  # 小批量快速测试
        "--num_epochs", "5",   # 仅5个epoch
        "--dataset", "cifar10",
        "--scheme", "V2_epoch",
        "--regularization", "Gaussian_kl",
        "--regularization_strength", "0.025",
        "--log_entropy", "1",
        "--AT_regularization", "SCA_new",
        "--AT_regularization_strength", "0.3",
        "--random_seed", "125",
        "--learning_rate", "0.05",
        "--lambd", "16",
        "--gan_AE_type", "res_normN4C64",
        "--gan_loss_type", "SSIM",
        "--local_lr", "-1",
        "--bottleneck_option", "noRELU_C8S1",
        "--folder", "saves/cifar10/test_direct_fusion",
        "--ssim_threshold", "0.5",
        "--var_threshold", "0.125"
    ]
    
    try:
        result = subprocess.run(
            train_cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10分钟超时
        )
        
        if result.returncode == 0:
            print("✅ 训练测试成功！")
            print("📋 输出片段:")
            # 显示最后几行输出
            output_lines = result.stdout.split('\n')
            for line in output_lines[-10:]:
                if line.strip():
                    print(f"  {line}")
                    
            return True
        else:
            print("❌ 训练测试失败")
            print("错误信息:")
            print(result.stderr[-1000:])  # 最后1000字符
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ 训练测试超时")
        return False
    except Exception as e:
        print(f"💥 训练测试出错: {str(e)}")
        return False

def main():
    # 确保在正确的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"📁 工作目录: {script_dir}")
    
    # 运行快速测试
    success = quick_test()
    
    if success:
        print("\n🎉 CEM-direct 验证成功！")
        print("💡 现在可以运行完整实验:")
        print("   bash run_exp.sh")
    else:
        print("\n❌ CEM-direct 验证失败，请检查错误信息")
        sys.exit(1)

if __name__ == "__main__":
    main()
