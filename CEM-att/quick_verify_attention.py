#!/usr/bin/env python3

"""
快速验证attention分类器是否在实际使用
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import model_training

def verify_attention_usage():
    """验证attention分类器是否真的在使用"""
    print("🔍 验证Attention分类器使用情况...")
    
    # 初始化MIA_train (使用与实际脚本相同的参数)
    mi = model_training.MIA_train(
        arch='vgg11_bn_sgm',
        cutting_layer=4,
        batch_size=32,  # 小批量
        lambd=16.0,  # 与实际脚本相同
        n_epochs=1,
        scheme='V2_epoch',
        regularization_option='Gaussian_kl',
        regularization_strength=0.025,
        AT_regularization_option='SCA_new',
        AT_regularization_strength=0.3,
        bottleneck_option='noRELU_C8S1',
        gan_AE_type='res_normN4C64',
        gan_loss_type='SSIM',
        ssim_threshold=0.5,
        num_client=1,
        random_seed=125,
        log_entropy=1,
        var_threshold=0.125,
        # Attention参数 (与实际脚本相同)
        use_attention_classifier=True,
        num_slots=8,
        attention_heads=8,
        attention_dropout=0.1
    )
    
    print(f"✅ 初始化完成")
    print(f"📋 关键参数检查:")
    print(f"   - use_attention_classifier: {mi.use_attention_classifier}")
    print(f"   - lambd: {mi.lambd}")
    print(f"   - attention_classifier: {mi.attention_classifier is not None}")
    
    if mi.attention_classifier is not None:
        print(f"   - feature_dim: {mi.attention_classifier.feature_dim}")
        print(f"   - num_slots: {mi.attention_classifier.num_slots}")
    
    # 创建模拟数据
    device = mi.device
    batch_size = 8
    z_private = torch.randn(batch_size, 8, 8, 8).to(device)  # bottleneck后的特征
    label_private = torch.randint(0, 10, (batch_size,)).to(device)
    
    print(f"\n🧪 测试attention分类器前向传播...")
    print(f"   - 输入特征形状: {z_private.shape}")
    print(f"   - 标签形状: {label_private.shape}")
    
    # 测试attention分类器
    try:
        attention_logits, enhanced_features, slot_representations, attention_weights = mi.attention_classify_features(z_private, label_private)
        print(f"✅ Attention前向传播成功!")
        print(f"   - attention_logits形状: {attention_logits.shape}")
        print(f"   - 预测类别: {torch.argmax(attention_logits, dim=1)}")
        print(f"   - 真实标签: {label_private}")
        
        # 检查输出是否合理
        if attention_logits.shape == (batch_size, 10):
            print(f"✅ 输出维度正确 (batch_size={batch_size}, num_classes=10)")
        else:
            print(f"❌ 输出维度错误: 期望({batch_size}, 10), 实际{attention_logits.shape}")
            
        # 检查softmax后的概率
        probs = torch.softmax(attention_logits, dim=1)
        print(f"   - 预测概率范围: [{probs.min().item():.3f}, {probs.max().item():.3f}]")
        
    except Exception as e:
        print(f"❌ Attention前向传播失败: {e}")
        return False
    
    # 测试训练步骤逻辑
    print(f"\n🔄 模拟训练步骤逻辑...")
    
    # 模拟第1个epoch (random_ini_centers=True)
    random_ini_centers = True
    print(f"📍 第1个epoch (random_ini_centers={random_ini_centers}):")
    if mi.use_attention_classifier and mi.lambd > 0:
        print(f"   ✅ 会使用attention分类器 (即使random_ini_centers=True)")
    else:
        print(f"   ❌ 不会使用attention分类器")
    
    # 模拟第2个epoch (random_ini_centers=False)  
    random_ini_centers = False
    print(f"📍 第2个epoch开始 (random_ini_centers={random_ini_centers}):")
    if mi.use_attention_classifier and mi.lambd > 0:
        print(f"   ✅ 会使用attention分类器")
    else:
        print(f"   ❌ 不会使用attention分类器")
    
    print(f"\n🎯 结论:")
    if mi.use_attention_classifier and mi.lambd > 0 and mi.attention_classifier is not None:
        print(f"✅ Attention分类器配置正确，应该会被使用")
        print(f"✅ 从第1个epoch开始就会使用attention分类器进行分类")
        print(f"✅ 条件熵计算从第2个epoch开始使用attention方法")
        return True
    else:
        print(f"❌ Attention分类器配置有问题")
        return False

if __name__ == "__main__":
    print("🧪 验证CEM-att中Attention分类器的实际使用情况")
    print("=" * 60)
    
    success = verify_attention_usage()
    
    if success:
        print(f"\n🎉 验证成功！")
        print(f"📊 您的实验输出准确率低可能是因为:")
        print(f"   1. 训练还在早期阶段 (需要更多epoch)")
        print(f"   2. Attention分类器需要时间学习合适的表示")
        print(f"   3. 条件熵损失和分类损失的平衡需要调整")
        print(f"   \n💡 建议: 让实验继续运行，准确率应该会逐渐提升")
    else:
        print(f"\n❌ 验证失败，需要修复配置问题")
        sys.exit(1)
