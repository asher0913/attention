#!/usr/bin/env python3

"""
生成用于攻击测试的固定测试数据集
该脚本会创建 test_cifar10_image.pt 和 test_cifar10_label.pt 文件
"""

import torch
import torchvision
import torchvision.transforms as transforms
import os
import sys

def generate_cifar10_test_data(num_samples=128, save_dir="./"):
    """
    生成CIFAR10测试数据
    
    Args:
        num_samples: 生成的样本数量 (默认128，与main_test_MIA.py中的batch_size一致)
        save_dir: 保存目录
    """
    print(f"🔄 正在生成CIFAR10测试数据...")
    print(f"   📊 样本数量: {num_samples}")
    print(f"   📁 保存目录: {save_dir}")
    
    # CIFAR10数据预处理 (与main_MIA.py保持一致)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 加载CIFAR10测试集
    testset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # 创建数据加载器
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=num_samples, 
        shuffle=False,  # 使用固定顺序确保复现性
        num_workers=0
    )
    
    # 获取第一批数据作为测试数据
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    
    print(f"   ✅ 生成数据形状:")
    print(f"      - images: {images.shape}")
    print(f"      - labels: {labels.shape}")
    print(f"      - 标签分布: {torch.bincount(labels)}")
    
    # 保存测试数据
    image_path = os.path.join(save_dir, "test_cifar10_image.pt")
    label_path = os.path.join(save_dir, "test_cifar10_label.pt")
    
    torch.save(images, image_path)
    torch.save(labels, label_path)
    
    print(f"   ✅ 测试数据已保存:")
    print(f"      - 图像: {image_path}")
    print(f"      - 标签: {label_path}")
    
    # 验证保存的数据
    verify_saved_data(image_path, label_path)
    
    return images, labels

def verify_saved_data(image_path, label_path):
    """验证保存的数据"""
    print(f"\n🔍 验证保存的测试数据...")
    
    try:
        images = torch.load(image_path)
        labels = torch.load(label_path)
        
        print(f"   ✅ 数据加载成功:")
        print(f"      - images: {images.shape}, dtype: {images.dtype}")
        print(f"      - labels: {labels.shape}, dtype: {labels.dtype}")
        print(f"      - 图像值范围: [{images.min():.3f}, {images.max():.3f}]")
        print(f"      - 标签范围: [{labels.min()}, {labels.max()}]")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 数据验证失败: {e}")
        return False

def generate_other_datasets():
    """为其他数据集生成测试数据 (可选)"""
    print(f"\n📋 其他可生成的数据集:")
    print(f"   - MNIST: test_mnist_image.pt, test_mnist_label.pt")
    print(f"   - Fashion-MNIST: test_fmnist_image.pt, test_fmnist_label.pt")
    print(f"   - CIFAR100: test_cifar100_image.pt, test_cifar100_label.pt")
    print(f"   - SVHN: test_svhn_image.pt, test_svhn_label.pt")
    print(f"\n💡 如需生成其他数据集，可扩展此脚本")

if __name__ == "__main__":
    print("🚀 CIFAR10攻击测试数据生成器")
    print("=" * 50)
    
    # 确保在正确的目录中
    if not os.path.exists("main_test_MIA.py"):
        print("❌ 错误: 请在CEM-att目录中运行此脚本")
        sys.exit(1)
    
    # 生成CIFAR10测试数据
    try:
        images, labels = generate_cifar10_test_data()
        print(f"\n🎉 测试数据生成成功!")
        print(f"📌 现在可以运行 python main_test_MIA.py 进行攻击测试")
        
        generate_other_datasets()
        
    except Exception as e:
        print(f"\n❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
