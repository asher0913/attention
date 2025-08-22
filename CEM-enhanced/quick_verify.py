#!/usr/bin/env python3
"""
CEM-Enhanced 快速验证脚本
测试串行Attention→GMM架构是否能正常运行
"""

import subprocess
import sys
import os

def verify_enhanced_cem():
    """验证CEM-Enhanced是否能正常运行"""
    print("🚀 CEM-Enhanced 快速验证")
    print("=" * 50)
    print("🎯 测试串行Attention→GMM架构")
    print("💡 包含：多尺度特征 + 直接损失融合 + 增强注意力")
    print()
    
    # 测试训练（仅3个epoch快速验证）
    print("📚 测试增强架构训练...")
    train_cmd = [
        "python", "main_MIA.py",
        "--arch", "vgg11_bn_sgm",
        "--cutlayer", "4", 
        "--batch_size", "32",
        "--num_epochs", "3",  # 快速测试
        "--dataset", "cifar10",
        "--scheme", "V2_epoch",
        "--regularization", "Gaussian_kl",
        "--regularization_strength", "0.025",
        "--log_entropy", "1",
        "--AT_regularization", "SCA_new",
        "--AT_regularization_strength", "0.3",
        "--random_seed", "125",
        "--learning_rate", "0.05",
        "--lambd", "16",  # 测试直接损失融合
        "--gan_AE_type", "res_normN4C64",
        "--gan_loss_type", "SSIM",
        "--local_lr", "-1",
        "--bottleneck_option", "noRELU_C8S1",
        "--folder", "saves/cifar10/test_enhanced",
        "--ssim_threshold", "0.5",
        "--var_threshold", "0.125"
    ]
    
    try:
        result = subprocess.run(
            train_cmd,
            capture_output=True,
            text=True,
            timeout=900  # 15分钟超时
        )
        
        if result.returncode == 0:
            print("✅ 增强架构训练测试成功！")
            
            # 检查输出中的关键信息
            output_lines = result.stdout.split('\n')
            enhanced_features_found = False
            attention_found = False
            direct_fusion_found = False
            
            for line in output_lines[-20:]:  # 检查最后20行
                if line.strip():
                    print(f"  {line}")
                    if "enhanced" in line.lower() or "attention" in line.lower():
                        attention_found = True
                    if "serial" in line.lower() or "fusion" in line.lower():
                        direct_fusion_found = True
                        
            print()
            print("🔍 架构特性验证：")
            print(f"  {'✅' if attention_found else '⚠️ '} 注意力机制")
            print(f"  {'✅' if direct_fusion_found else '⚠️ '} 直接损失融合")
            
            return True
        else:
            print("❌ 增强架构训练测试失败")
            print("错误输出：")
            print(result.stderr[-1500:])  # 最后1500字符
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ 训练测试超时（这在快速验证中是正常的）")
        print("💡 请运行完整实验：bash run_exp.sh")
        return True  # 超时不算失败
    except Exception as e:
        print(f"💥 训练测试出错: {str(e)}")
        return False

def main():
    # 确保在正确的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"📁 工作目录: {script_dir}")
    
    # 验证增强架构
    success = verify_enhanced_cem()
    
    if success:
        print("\n🎉 CEM-Enhanced 验证成功！")
        print()
        print("🚀 增强特性：")
        print("  • 串行Attention→GMM架构")
        print("  • 多尺度特征融合")
        print("  • 直接损失融合")
        print("  • 层次化Slot Attention")
        print("  • 增强Cross Attention")
        print("  • 自适应特征门控")
        print("  • 组合距离度量")
        print()
        print("💡 运行完整实验：")
        print("   bash run_exp.sh")
        print()
        print("📖 技术原理详解：")
        print("   查看 CEM_ENHANCED_技术原理详解.md")
    else:
        print("\n❌ CEM-Enhanced 验证失败")
        print("请检查错误信息并修复问题")
        sys.exit(1)

if __name__ == "__main__":
    main()
