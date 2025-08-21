#!/usr/bin/env python3

"""
检查攻击过程是否真的在运行
以及为什么显存占用这么低
"""

import torch
import os
import subprocess
import sys

def check_gpu_memory():
    """检查GPU显存使用情况"""
    if torch.cuda.is_available():
        print(f"🔍 CUDA可用，设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            mem_cached = torch.cuda.memory_reserved(i) / 1024**3      # GB
            print(f"   GPU {i}: 已分配 {mem_allocated:.2f}GB, 已缓存 {mem_cached:.2f}GB")
    else:
        print("❌ CUDA不可用")

def test_model_loading():
    """测试模型加载是否成功"""
    print("\n🧪 测试模型加载...")
    
    try:
        # 模拟真实的模型加载过程
        import model_training
        
        # 构建MIA_train对象
        mi = model_training.MIA_train(
            arch="vgg11_bn_sgm",
            cutting_layer=4,
            batch_size=128,
            n_epochs=240,
            scheme="V2_epoch",
            num_client=1,
            dataset="cifar10",
            save_dir="test_dir",
            random_seed=125,
            regularization_option="Gaussian_kl",
            regularization_strength=0.025,
            AT_regularization_option="SCA_new", 
            AT_regularization_strength=0.3,
            log_entropy=1,
            gan_AE_type="res_normN8C64",
            bottleneck_option="noRELU_C8S1",
            gan_loss_type="SSIM",
            use_attention_classifier=True,
            num_slots=8,
            attention_heads=8,
            attention_dropout=0.1,
            var_threshold=0.125
        )
        
        print("✅ MIA_train对象创建成功")
        
        # 检查GPU内存
        check_gpu_memory()
        
        # 尝试加载模型
        checkpoint_path = "saves/cifar10/SCA_new_attention_fixed_lg1_thre0.125/saves/cifar10/SCA_new_attention_fixed_lg1_thre0.125/CEM_log_entropy1_cifar10_cutlayer4_archvgg11_bn_sgm_schemeV2_epoch_n_epochs240_batch_size128_lr0.05_regulastr0.025_bottlenecknoRELU_C8S1_SCA_new0.3_randomseed125_ssim0.5_lambd16/checkpoint_f_best.tar"
        
        if os.path.exists(checkpoint_path):
            print(f"✅ 找到checkpoint: {checkpoint_path}")
            mi.resume(checkpoint_path)
            print("✅ 模型加载成功")
            
            # 检查加载后的GPU内存
            print("\n📊 模型加载后的GPU内存:")
            check_gpu_memory()
            
        else:
            print(f"❌ checkpoint不存在: {checkpoint_path}")
            
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()

def check_attack_function():
    """检查攻击函数是否会被调用"""
    print("\n🔍 检查攻击函数...")
    
    # 检查测试数据是否存在
    test_files = ["test_cifar10_image.pt", "test_cifar10_label.pt"]
    for test_file in test_files:
        if os.path.exists(test_file):
            data = torch.load(test_file)
            print(f"✅ {test_file}: shape={data.shape if hasattr(data, 'shape') else len(data)}")
        else:
            print(f"❌ {test_file} 不存在")
    
    # 检查main_test_MIA.py中的关键行
    try:
        with open("main_test_MIA.py", "r") as f:
            lines = f.readlines()
            
        # 查找关键的攻击调用行
        for i, line in enumerate(lines):
            if "mi.MIA_attack" in line:
                print(f"✅ 找到攻击调用 (第{i+1}行): {line.strip()}")
            if "mi(" in line and "verbose" in line:
                print(f"✅ 找到验证调用 (第{i+1}行): {line.strip()}")
                
    except Exception as e:
        print(f"❌ 检查main_test_MIA.py失败: {e}")

def main():
    print("🔍 诊断防御测试显存占用异常问题")
    print("=" * 50)
    
    print("\n💾 初始GPU状态:")
    check_gpu_memory()
    
    test_model_loading()
    
    check_attack_function()
    
    print("\n💡 可能的问题:")
    print("1. 攻击过程提前退出或报错")
    print("2. 生成器网络没有正确加载到GPU")
    print("3. batch_size设置过小")
    print("4. 某些网络组件在CPU上运行")
    print("5. 攻击epoch数设置错误")
    
    print("\n🔧 建议检查:")
    print("1. 运行完整的防御测试，观察显存变化")
    print("2. 检查攻击日志文件")
    print("3. 对比原始CEM-main的参数设置")

if __name__ == "__main__":
    main()
