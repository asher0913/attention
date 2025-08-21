#!/usr/bin/env python3

"""
智能防御效果测试脚本
自动检测已保存的模型并运行攻击测试
"""

import os
import glob
import subprocess
import sys
import argparse
from pathlib import Path

def find_model_directories():
    """查找所有可能的模型目录"""
    model_dirs = []
    
    # 查找所有包含.pth或.pt文件的目录
    for root, dirs, files in os.walk("saves/"):
        has_model = any(f.endswith(('.pth', '.pt')) for f in files)
        has_log = any(f.endswith('.log') for f in files)
        
        if has_model or has_log:
            model_dirs.append(root)
    
    return model_dirs

def extract_params_from_path(model_path):
    """从路径中提取参数"""
    params = {
        'lambd': 16,
        'regularization_strength': 0.025,
        'AT_regularization_strength': 0.3,
        'num_epochs': 240,
        'batch_size': 128,
        'learning_rate': 0.05,  # 从日志路径看是0.05
        'random_seed': 125,
        'ssim_threshold': 0.5
    }
    
    # 从路径中提取参数
    path_str = str(model_path)
    
    # 提取lambda值
    if 'lambd' in path_str:
        import re
        lambd_match = re.search(r'lambd(\d+)', path_str)
        if lambd_match:
            params['lambd'] = int(lambd_match.group(1))
    
    # 提取正则化强度
    if 'regulastr' in path_str:
        import re
        reg_match = re.search(r'regulastr([\d.]+)', path_str)
        if reg_match:
            params['regularization_strength'] = float(reg_match.group(1))
    
    return params

def run_defense_test(model_path, use_cuda=True):
    """运行防御测试"""
    print(f"🚀 开始防御效果测试...")
    print(f"📁 模型路径: {model_path}")
    
    # 提取参数
    params = extract_params_from_path(model_path)
    print(f"📋 提取的参数:")
    for key, value in params.items():
        print(f"   {key}: {value}")
    
    # 设备配置
    device = "cuda" if use_cuda else "cpu"
    gpu_id = "0" if use_cuda else ""
    
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
        "--batch_size", str(params['batch_size']),
        "--filename", f"{model_path}/CEM_log_entropy1_cifar10_cutlayer4_archvgg11_bn_sgm_schemeV2_epoch_n_epochs{params['num_epochs']}_batch_size{params['batch_size']}_lr{params['learning_rate']}_regulastr{params['regularization_strength']}_bottlenecknoRELU_C8S1_SCA_new{params['AT_regularization_strength']}_randomseed{params['random_seed']}_ssim{params['ssim_threshold']}_lambd{params['lambd']}",
        "--num_client", "1",
        "--num_epochs", str(params['num_epochs']),
        "--dataset", "cifar10",
        "--scheme", "V2_epoch",
        "--regularization", "Gaussian_kl",
        "--regularization_strength", str(params['regularization_strength']),
        "--log_entropy", "1",
        "--AT_regularization", "SCA_new",
        "--AT_regularization_strength", str(params['AT_regularization_strength']),
        "--random_seed", str(params['random_seed']),
        "--gan_AE_type", "res_normN4C64",
        "--gan_loss_type", "SSIM",
        "--attack_epochs", "50",
        "--bottleneck_option", "noRELU_C8S1",
        "--folder", model_path,
        "--var_threshold", "0.125",
        "--average_time", "20",
        "--lambd", str(params['lambd']),
        "--use_attention_classifier",
        "--num_slots", "8",
        "--attention_heads", "8", 
        "--attention_dropout", "0.1",
        "--test_best"
    ]
    
    # 设置GPU环境变量
    env = os.environ.copy()
    if use_cuda and gpu_id:
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    print(f"\n📋 执行命令:")
    print(f"   {'CUDA_VISIBLE_DEVICES=' + gpu_id + ' ' if use_cuda and gpu_id else ''}{' '.join(cmd)}")
    
    try:
        print(f"\n🔄 正在运行攻击测试...")
        result = subprocess.run(cmd, env=env, capture_output=False, text=True)
        
        if result.returncode == 0:
            print(f"\n🎉 防御效果测试成功完成!")
            print(f"📊 请检查输出中的MSE、SSIM、PSNR指标")
            return True
        else:
            print(f"\n❌ 防御效果测试失败 (返回码: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"\n❌ 运行错误: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="智能防御效果测试")
    parser.add_argument("--model_path", help="指定模型路径")
    parser.add_argument("--cpu", action="store_true", help="使用CPU而不是CUDA")
    parser.add_argument("--list", action="store_true", help="仅列出可用的模型")
    
    args = parser.parse_args()
    
    print("🛡️ CEM-att 智能防御效果测试")
    print("=" * 50)
    
    # 查找模型目录
    model_dirs = find_model_directories()
    
    if not model_dirs:
        print("❌ 未找到任何模型目录")
        print("💡 请确保训练已完成并保存了模型")
        return
    
    print(f"🔍 找到 {len(model_dirs)} 个模型目录:")
    for i, model_dir in enumerate(model_dirs, 1):
        print(f"   {i}. {model_dir}")
    
    if args.list:
        return
    
    # 选择模型
    if args.model_path:
        if os.path.exists(args.model_path):
            selected_path = args.model_path
        else:
            print(f"❌ 指定的路径不存在: {args.model_path}")
            return
    else:
        # 使用最新的模型 (最长路径通常是最深层的)
        selected_path = max(model_dirs, key=len)
        print(f"\n✅ 自动选择模型: {selected_path}")
    
    # 运行测试
    success = run_defense_test(selected_path, use_cuda=not args.cpu)
    
    if success:
        print(f"\n🎯 防御效果测试完成!")
        print(f"📁 结果保存在: {selected_path}")
    else:
        print(f"\n💡 如果遇到问题，请尝试:")
        print(f"   1. 使用 --cpu 参数在CPU上测试")
        print(f"   2. 使用 --list 查看所有可用模型")
        print(f"   3. 手动指定模型路径: --model_path <路径>")

if __name__ == "__main__":
    main()
