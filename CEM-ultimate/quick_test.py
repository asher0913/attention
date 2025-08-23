#!/usr/bin/env python3
"""
🚀 CEM-Enhanced 快速测试脚本
验证增强版CEM算法的基础功能
"""

import torch
import numpy as np
import sys
import os

def test_enhanced_cem():
    """测试增强版CEM的基础功能"""
    print("🔍 测试增强版CEM算法...")
    
    try:
        # 导入模块
        import model_training
        print("✅ 模块导入成功")
        
        # 创建MIA_train实例
        mi = model_training.MIA_train(
            arch="vgg11", 
            cutting_layer=4, 
            batch_size=4,  # 小batch size快速测试
            n_epochs=1,    # 只训练1个epoch
            lambd=16, 
            scheme="V2_epoch", 
            num_client=1, 
            dataset="cifar10",
            save_dir="./quick_test_saves",
            regularization_option="Gaussian_kl", 
            regularization_strength=0.025
        )
        print("✅ CEM-Enhanced 初始化成功")
        
        # 测试增强的条件熵计算
        dummy_features = torch.randn(4, 128, 4, 4).cuda()  # VGG特征形状
        dummy_labels = torch.randint(0, 10, (4,)).cuda()
        unique_labels = torch.unique(dummy_labels)
        
        # 创建虚拟centroids
        centroids_list = {}
        for label in unique_labels:
            centroids_list[label.item()] = torch.randn(3, 2048).cuda()
        
        # 测试增强版条件熵计算
        rob_loss, intra_class_mse = mi.compute_class_means_enhanced(
            dummy_features, dummy_labels, unique_labels, centroids_list
        )
        
        print(f"✅ 增强版条件熵计算成功")
        print(f"   - 条件熵损失: {rob_loss.item():.6f}")
        print(f"   - 特征融合权重: {mi.feature_fusion_weights.data}")
        
        # 测试传统版本（作为对比）
        rob_loss_original, _ = mi.compute_class_means(
            dummy_features, dummy_labels, unique_labels, centroids_list
        )
        print(f"✅ 传统条件熵计算: {rob_loss_original.item():.6f}")
        
        improvement = ((rob_loss_original - rob_loss) / rob_loss_original * 100).item()
        print(f"🚀 条件熵改进: {improvement:+.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 CEM-Enhanced 快速测试")
    print("="*50)
    
    # 检查CUDA
    if torch.cuda.is_available():
        print("✅ CUDA 可用")
        device = "cuda"
    else:
        print("⚠️  CUDA 不可用，使用CPU")
        device = "cpu"
    
    # 运行测试
    success = test_enhanced_cem()
    
    print("="*50)
    if success:
        print("🎉 CEM-Enhanced 基础功能测试通过！")
        print("💡 现在可以运行: bash run_exp.sh")
        print("🚀 期待比CEM-main更好的性能！")
    else:
        print("❌ 测试失败，请检查错误")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
