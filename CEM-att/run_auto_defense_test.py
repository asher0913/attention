#!/usr/bin/env python3

"""
自动防御效果测试脚本
智能检测模型路径和checkpoint文件
"""

import os
import glob
import subprocess
import sys
from pathlib import Path

def find_checkpoint_files():
    """查找所有checkpoint文件"""
    checkpoint_files = []
    
    # 查找所有checkpoint文件
    for pattern in ["**/checkpoint_f_*.tar", "**/checkpoint_*.tar"]:
        files = glob.glob(f"saves/{pattern}", recursive=True)
        checkpoint_files.extend(files)
    
    return checkpoint_files

def get_model_info(checkpoint_path):
    """从checkpoint路径提取模型信息"""
    checkpoint_path = Path(checkpoint_path)
    
    # 获取模型目录（checkpoint文件的父目录）
    model_dir = checkpoint_path.parent
    
    # 获取checkpoint类型
    filename = checkpoint_path.name
    if "best" in filename:
        checkpoint_type = "best"
    elif "240" in filename:
        checkpoint_type = "240"
    else:
        # 提取数字
        import re
        match = re.search(r'checkpoint_f_(\d+)\.tar', filename)
        if match:
            checkpoint_type = match.group(1)
        else:
            checkpoint_type = "unknown"
    
    # 构建filename参数（相对于model_dir的路径）
    filename_param = model_dir.name
    
    return {
        'model_dir': str(model_dir),
        'checkpoint_type': checkpoint_type,
        'filename_param': filename_param,
        'checkpoint_file': str(checkpoint_path)
    }

def run_defense_test(model_info, use_cuda=True):
    """运行防御测试"""
    print(f"🚀 开始防御效果测试...")
    print(f"📁 模型目录: {model_info['model_dir']}")
    print(f"🎯 Checkpoint: {model_info['checkpoint_type']}")
    print(f"📄 文件名参数: {model_info['filename_param']}")
    
    # 检查并生成测试数据
    print(f"\n🔍 检查测试数据...")
    if not os.path.exists("test_cifar10_image.pt") or not os.path.exists("test_cifar10_label.pt"):
        print("⚠️  测试数据不存在，正在生成...")
        try:
            subprocess.run(["python", "generate_test_data.py"], check=True)
            print("✅ 测试数据生成完成")
        except Exception as e:
            print(f"❌ 测试数据生成失败: {e}")
            return False
    else:
        print("✅ 测试数据已存在")
    
    # 构建攻击测试命令
    cmd = [
        "python", "main_test_MIA.py",
        "--arch", "vgg11_bn_sgm",
        "--cutlayer", "4", 
        "--batch_size", "128",
        "--filename", model_info['filename_param'],
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
        "--gan_AE_type", "res_normN4C64",
        "--gan_loss_type", "SSIM",
        "--attack_epochs", "50",
        "--bottleneck_option", "noRELU_C8S1",
        "--folder", model_info['model_dir'],
        "--var_threshold", "0.125",
        "--average_time", "20",
        "--lambd", "16",
        "--use_attention_classifier",
        "--num_slots", "8",
        "--attention_heads", "8",
        "--attention_dropout", "0.1"
    ]
    
    # 如果是best checkpoint，添加test_best参数
    if model_info['checkpoint_type'] == "best":
        cmd.append("--test_best")
    
    # 设置GPU环境变量
    env = os.environ.copy()
    if use_cuda:
        env["CUDA_VISIBLE_DEVICES"] = "0"
    
    print(f"\n📋 执行命令:")
    print(f"   {'CUDA_VISIBLE_DEVICES=0 ' if use_cuda else ''}{' '.join(cmd)}")
    
    try:
        print(f"\n🔄 正在运行防御测试...")
        result = subprocess.run(cmd, env=env, capture_output=False, text=True)
        
        if result.returncode == 0:
            print(f"\n🎉 防御效果测试成功完成!")
            print(f"📊 请查看上方输出中的MSE、SSIM、PSNR指标")
            print(f"🎯 使用的checkpoint: {model_info['checkpoint_type']}")
            return True
        else:
            print(f"\n❌ 防御效果测试失败 (返回码: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"\n❌ 运行错误: {e}")
        return False

def main():
    print("🛡️ CEM-att 自动防御效果测试")
    print("=" * 50)
    
    # 查找checkpoint文件
    checkpoint_files = find_checkpoint_files()
    
    if not checkpoint_files:
        print("❌ 未找到任何checkpoint文件")
        print("💡 请确保训练已完成并保存了模型")
        return
    
    print(f"🔍 找到 {len(checkpoint_files)} 个checkpoint文件:")
    models = []
    for i, checkpoint_file in enumerate(checkpoint_files, 1):
        model_info = get_model_info(checkpoint_file)
        models.append(model_info)
        print(f"   {i}. {checkpoint_file} (type: {model_info['checkpoint_type']})")
    
    # 选择最佳模型（优先best，然后240，最后最大数字）
    best_model = None
    for model in models:
        if model['checkpoint_type'] == "best":
            best_model = model
            break
    
    if not best_model:
        # 查找最大epoch数
        numeric_models = []
        for model in models:
            try:
                epoch_num = int(model['checkpoint_type'])
                numeric_models.append((epoch_num, model))
            except ValueError:
                pass
        
        if numeric_models:
            numeric_models.sort(reverse=True)  # 按epoch数降序
            best_model = numeric_models[0][1]
    
    if not best_model:
        best_model = models[0]  # fallback到第一个
    
    print(f"\n✅ 自动选择模型: {best_model['checkpoint_file']}")
    print(f"🎯 Checkpoint类型: {best_model['checkpoint_type']}")
    
    # 运行测试
    success = run_defense_test(best_model, use_cuda=True)
    
    if success:
        print(f"\n🎯 防御效果测试完成!")
        print(f"📊 关键指标说明:")
        print(f"   ✅ MSE ↓ - 攻击重建误差越大越好")
        print(f"   ✅ SSIM ↓ - 攻击质量越差越好")
        print(f"   ✅ PSNR ↑ - 隐私保护越强越好")
    else:
        print(f"\n💡 如果遇到问题，请检查:")
        print(f"   1. GPU状态: nvidia-smi")
        print(f"   2. 模型文件完整性")
        print(f"   3. Python环境和依赖")

if __name__ == "__main__":
    main()
