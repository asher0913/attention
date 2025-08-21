#!/usr/bin/env python3

"""
分析CEM实验结果
比较GMM vs Attention的性能差异
"""

import os
import json
import re
import glob
from pathlib import Path

def extract_accuracy_from_log(log_content):
    """从日志中提取准确率"""
    # 寻找最佳准确率
    best_acc_pattern = r"Best Average Validation Accuracy is ([\d.]+)"
    match = re.search(best_acc_pattern, log_content)
    if match:
        return float(match.group(1))
    
    # 寻找最后的准确率
    prec_pattern = r"\* Prec@1 ([\d.]+)"
    matches = re.findall(prec_pattern, log_content)
    if matches:
        return float(matches[-1])
    
    return None

def extract_attack_metrics(log_content):
    """从日志中提取攻击指标"""
    metrics = {}
    
    # MSE指标
    mse_pattern = r"MSE:\s*([\d.]+)"
    mse_match = re.search(mse_pattern, log_content)
    if mse_match:
        metrics['MSE'] = float(mse_match.group(1))
    
    # SSIM指标
    ssim_pattern = r"SSIM:\s*([\d.]+)"
    ssim_match = re.search(ssim_pattern, log_content)
    if ssim_match:
        metrics['SSIM'] = float(ssim_match.group(1))
    
    # PSNR指标
    psnr_pattern = r"PSNR:\s*([\d.]+)"
    psnr_match = re.search(psnr_pattern, log_content)
    if psnr_match:
        metrics['PSNR'] = float(psnr_match.group(1))
    
    return metrics

def analyze_experiment_results():
    """分析实验结果"""
    print("📊 CEM实验结果分析")
    print("=" * 60)
    
    results = {}
    
    # 查找所有实验目录
    save_dirs = glob.glob("./saves/cifar10/*/")
    
    for save_dir in save_dirs:
        dir_name = os.path.basename(save_dir.rstrip('/'))
        print(f"\n🔍 分析目录: {dir_name}")
        
        # 查找日志文件
        log_files = glob.glob(os.path.join(save_dir, "*.log"))
        if not log_files:
            print(f"   ❌ 未找到日志文件")
            continue
        
        log_file = log_files[0]  # 取第一个日志文件
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            # 提取指标
            accuracy = extract_accuracy_from_log(log_content)
            attack_metrics = extract_attack_metrics(log_content)
            
            results[dir_name] = {
                'accuracy': accuracy,
                'attack_metrics': attack_metrics,
                'log_file': log_file
            }
            
            print(f"   ✅ 准确率: {accuracy}%")
            if attack_metrics:
                for metric, value in attack_metrics.items():
                    print(f"   ✅ {metric}: {value}")
            else:
                print(f"   ⚠️  未找到攻击指标")
                
        except Exception as e:
            print(f"   ❌ 读取日志失败: {e}")
    
    return results

def compare_gmm_vs_attention(results):
    """比较GMM vs Attention的结果"""
    print(f"\n🎯 GMM vs Attention 性能对比")
    print("=" * 60)
    
    gmm_results = []
    attention_results = []
    
    for dir_name, data in results.items():
        if 'attention' in dir_name.lower():
            attention_results.append((dir_name, data))
        else:
            gmm_results.append((dir_name, data))
    
    print(f"\n📋 GMM结果 ({len(gmm_results)}个):")
    for dir_name, data in gmm_results:
        acc = data['accuracy']
        print(f"   - {dir_name}: {acc}%")
        for metric, value in data['attack_metrics'].items():
            print(f"     └─ {metric}: {value}")
    
    print(f"\n📋 Attention结果 ({len(attention_results)}个):")
    for dir_name, data in attention_results:
        acc = data['accuracy']
        print(f"   - {dir_name}: {acc}%")
        for metric, value in data['attack_metrics'].items():
            print(f"     └─ {metric}: {value}")
    
    # 计算平均值对比
    if gmm_results and attention_results:
        gmm_accs = [data['accuracy'] for _, data in gmm_results if data['accuracy'] is not None]
        att_accs = [data['accuracy'] for _, data in attention_results if data['accuracy'] is not None]
        
        if gmm_accs and att_accs:
            gmm_avg = sum(gmm_accs) / len(gmm_accs)
            att_avg = sum(att_accs) / len(att_accs)
            
            print(f"\n📊 平均准确率对比:")
            print(f"   GMM平均:      {gmm_avg:.2f}%")
            print(f"   Attention平均: {att_avg:.2f}%")
            print(f"   差异:         {att_avg - gmm_avg:+.2f}%")

def check_current_experiment():
    """检查当前实验状态"""
    print(f"\n🔍 当前实验状态检查")
    print("=" * 40)
    
    # 检查关键文件
    files_to_check = [
        "test_cifar10_image.pt",
        "test_cifar10_label.pt",
        "generate_test_data.py",
        "run_attack_test_only.py"
    ]
    
    print(f"📁 关键文件检查:")
    for file in files_to_check:
        status = "✅" if os.path.exists(file) else "❌"
        print(f"   {status} {file}")
    
    # 检查保存目录
    save_dir = "./saves/cifar10/SCA_new_attention_fixed_lg1_thre0.125/"
    if os.path.exists(save_dir):
        print(f"\n📂 最新实验目录: {save_dir}")
        files = os.listdir(save_dir)
        print(f"   包含文件: {len(files)}个")
        for file in sorted(files)[:5]:  # 显示前5个文件
            print(f"   - {file}")
        if len(files) > 5:
            print(f"   ... 还有{len(files)-5}个文件")
    else:
        print(f"\n❌ 最新实验目录不存在: {save_dir}")

def main():
    print("🚀 CEM实验结果分析工具")
    print("=" * 60)
    
    # 确保在正确目录
    if not os.path.exists("main_test_MIA.py"):
        print("❌ 错误: 请在CEM-att目录中运行此脚本")
        return
    
    # 检查当前实验状态
    check_current_experiment()
    
    # 分析所有结果
    results = analyze_experiment_results()
    
    if results:
        # 比较GMM vs Attention
        compare_gmm_vs_attention(results)
        
        print(f"\n💡 结果解读:")
        print(f"   🎯 准确率差异 1-2% 是正常的")
        print(f"   🎯 重点关注攻击防御指标 (MSE↓, SSIM↓, PSNR↑)")
        print(f"   🎯 Attention的优势通常体现在:")
        print(f"      - 更好的特征表示学习")
        print(f"      - 更强的攻击防御能力")
        print(f"      - 更稳定的训练收敛")
    else:
        print(f"\n⚠️  未找到可分析的结果")
        print(f"💡 建议:")
        print(f"   1. 确保实验已完成")
        print(f"   2. 检查日志文件是否存在")
        print(f"   3. 运行完整的攻击测试")

if __name__ == "__main__":
    main()
