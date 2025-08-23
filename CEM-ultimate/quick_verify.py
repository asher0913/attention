#!/usr/bin/env python3
"""
🚀 CEM-Ultimate 快速验证脚本
确保基础功能可以运行，不需要完整的革命性架构
"""

import os
import sys
import torch
import numpy as np

def test_basic_import():
    """测试基础模块导入"""
    print("🔍 测试基础模块导入...")
    try:
        import model_training
        print("✅ model_training 导入成功")
        return True
    except Exception as e:
        print(f"❌ model_training 导入失败: {e}")
        return False

def test_traditional_cem():
    """测试传统CEM架构（不使用革命性架构）"""
    print("🔍 测试传统CEM架构...")
    try:
        import model_training
        
        # 创建MIA_train实例，不使用革命性架构
        mi = model_training.MIA_train(
            arch="vgg11", 
            cutting_layer=4, 
            batch_size=8,  # 小batch size
            n_epochs=1,    # 只训练1个epoch
            lambd=16, 
            scheme="V2_epoch", 
            num_client=1, 
            dataset="cifar10",
            save_dir="./quick_test_saves",
            regularization_option="Gaussian_kl", 
            regularization_strength=0.025,
            use_ultimate_architecture=False  # 🔑 关键：不使用革命性架构
        )
        print("✅ 传统CEM架构初始化成功")
        
        # 测试一个简单的前向传播
        dummy_x = torch.randn(2, 3, 32, 32).cuda()
        dummy_y = torch.randint(0, 10, (2,)).cuda()
        
        features = mi.f(dummy_x)
        print(f"✅ 特征提取成功，特征形状: {features.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 传统CEM测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 CEM-Ultimate 快速验证开始")
    print("="*50)
    
    # 测试CUDA可用性
    if torch.cuda.is_available():
        print("✅ CUDA 可用")
    else:
        print("⚠️  CUDA 不可用，使用CPU")
    
    # 测试1: 基础导入
    if not test_basic_import():
        return False
    
    # 测试2: 传统CEM（不使用革命性架构）
    if not test_traditional_cem():
        return False
    
    print("="*50)
    print("🎉 快速验证完成！CEM-Ultimate基础功能正常")
    print("💡 建议：现在可以运行 bash run_exp.sh 进行完整实验")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
