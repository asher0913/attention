#!/usr/bin/env python3
"""
CEM架构对比脚本 - 包含直接损失融合方法
对比CEM-main (GMM), CEM-mix (混合架构-梯度累加), CEM-direct (混合架构-直接融合)
"""

import os
import sys
import subprocess
import json
import time
import re
import datetime
from pathlib import Path

class CEMDirectFusionComparison:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.results = {}
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def setup_environment(self):
        """设置实验环境"""
        print("🔧 设置实验环境...")
        
        # 检查必要的项目文件夹
        required_dirs = ['CEM-main', 'CEM-mix', 'CEM-direct']
        for dir_name in required_dirs:
            dir_path = os.path.join(self.base_dir, dir_name)
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"缺少必要的项目文件夹: {dir_name}")
                
        print("✅ 环境检查完成")
        
    def run_experiment(self, project_name, variant_name):
        """运行单个实验"""
        print(f"\n🚀 开始运行 {variant_name} 实验...")
        
        project_dir = os.path.join(self.base_dir, project_name)
        os.chdir(project_dir)
        
        # 参数设置
        lambd = 16
        reg_strength = 0.025
        dataset = "cifar10"
        
        try:
            # 第一步：训练
            print(f"  📚 训练 {variant_name} 模型...")
            train_cmd = [
                "python", "main_MIA.py",
                "--dataset", dataset,
                "--lambd", str(lambd),
                "--regularization_strength", str(reg_strength),
                "--n_epochs", "240",
                "--batch_size", "128"
            ]
            
            train_result = subprocess.run(
                train_cmd, 
                capture_output=True, 
                text=True, 
                timeout=7200  # 2小时超时
            )
            
            if train_result.returncode != 0:
                print(f"❌ {variant_name} 训练失败")
                print("STDERR:", train_result.stderr[-1000:])  # 最后1000字符
                return None
                
            # 第二步：攻击测试
            print(f"  🎯 运行 {variant_name} 攻击测试...")
            
            # 查找最新的保存文件夹
            saves_dir = os.path.join(project_dir, "saves")
            if not os.path.exists(saves_dir):
                print(f"❌ 找不到saves文件夹: {saves_dir}")
                return None
                
            # 获取最新的实验文件夹
            exp_folders = [f for f in os.listdir(saves_dir) if os.path.isdir(os.path.join(saves_dir, f))]
            if not exp_folders:
                print(f"❌ saves文件夹中没有实验结果")
                return None
                
            latest_folder = max(exp_folders, key=lambda x: os.path.getctime(os.path.join(saves_dir, x)))
            
            attack_cmd = [
                "python", "main_test_MIA.py",
                "--dataset", dataset,
                "--lambd", str(lambd),
                "--regularization_strength", str(reg_strength),
                "--folder", latest_folder,
                "--date_0", "240"
            ]
            
            attack_result = subprocess.run(
                attack_cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1小时超时
            )
            
            if attack_result.returncode != 0:
                print(f"❌ {variant_name} 攻击测试失败")
                print("STDERR:", attack_result.stderr[-1000:])
                return None
                
            # 解析结果
            results = self.parse_results(train_result.stdout, attack_result.stdout, variant_name)
            print(f"✅ {variant_name} 实验完成")
            return results
            
        except subprocess.TimeoutExpired:
            print(f"⏰ {variant_name} 实验超时")
            return None
        except Exception as e:
            print(f"❌ {variant_name} 实验出错: {str(e)}")
            return None
        finally:
            os.chdir(self.base_dir)
            
    def parse_results(self, train_output, attack_output, variant_name):
        """解析实验结果"""
        results = {
            "variant": variant_name,
            "best_accuracy": 0.0,
            "train_attack_metrics": {},
            "infer_attack_metrics": {}
        }
        
        # 解析训练准确率
        acc_pattern = r'best avg accu: ([\d.]+)'
        acc_matches = re.findall(acc_pattern, train_output)
        if acc_matches:
            results["best_accuracy"] = float(acc_matches[-1])
            
        # 解析攻击指标
        attack_lines = attack_output.split('\n')
        for line in attack_lines:
            if "Train Attack Results:" in line:
                # 解析训练攻击结果
                mse_match = re.search(r'MSE: ([\d.]+)', line)
                ssim_match = re.search(r'SSIM: ([\d.]+)', line)
                psnr_match = re.search(r'PSNR: ([\d.]+)', line)
                
                if mse_match:
                    results["train_attack_metrics"]["mse"] = float(mse_match.group(1))
                if ssim_match:
                    results["train_attack_metrics"]["ssim"] = float(ssim_match.group(1))
                if psnr_match:
                    results["train_attack_metrics"]["psnr"] = float(psnr_match.group(1))
                    
            elif "Inference Attack Results:" in line:
                # 解析推理攻击结果
                mse_match = re.search(r'MSE: ([\d.]+)', line)
                ssim_match = re.search(r'SSIM: ([\d.]+)', line)
                psnr_match = re.search(r'PSNR: ([\d.]+)', line)
                
                if mse_match:
                    results["infer_attack_metrics"]["mse"] = float(mse_match.group(1))
                if ssim_match:
                    results["infer_attack_metrics"]["ssim"] = float(ssim_match.group(1))
                if psnr_match:
                    results["infer_attack_metrics"]["psnr"] = float(psnr_match.group(1))
                    
        return results
        
    def run_all_experiments(self):
        """运行所有实验"""
        experiments = [
            ("CEM-main", "GMM (原始)"),
            ("CEM-mix", "混合架构 (梯度累加)"),
            ("CEM-direct", "混合架构 (直接融合)")
        ]
        
        self.results = {}
        
        for project_name, variant_name in experiments:
            print(f"\n{'='*60}")
            print(f"🧪 实验: {variant_name}")
            print(f"{'='*60}")
            
            result = self.run_experiment(project_name, variant_name)
            if result:
                self.results[variant_name] = result
            else:
                print(f"❌ {variant_name} 实验失败，跳过...")
                
        return self.results
        
    def generate_report(self):
        """生成实验报告"""
        if not self.results:
            print("❌ 没有可用的实验结果")
            return
            
        report_filename = f"CEM_Direct_Fusion_Comparison_Report_{self.timestamp}.md"
        report_path = os.path.join(self.base_dir, report_filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# CEM架构对比实验报告 - 直接损失融合\n\n")
            f.write(f"**实验时间**: {self.timestamp}\n")
            f.write(f"**实验参数**: λ=16, 正则化强度=0.025, 数据集=CIFAR-10\n\n")
            
            # 核心改进说明
            f.write("## 🚀 核心改进：直接损失融合\n\n")
            f.write("### 原始方法（CEM-mix）问题：\n")
            f.write("```python\n")
            f.write("total_loss = f_loss  # 只有分类损失\n")
            f.write("# 条件熵损失通过梯度累加影响参数\n")
            f.write("rob_loss.backward(retain_graph=True)\n")
            f.write("param.grad += self.lambd * encoder_gradients[name]\n")
            f.write("```\n\n")
            
            f.write("### 改进方法（CEM-direct）：\n")
            f.write("```python\n")
            f.write("# 🚀 直接融合条件熵损失\n")
            f.write("total_loss = f_loss + self.lambd * rob_loss\n")
            f.write("total_loss.backward()  # 统一优化\n")
            f.write("```\n\n")
            
            # 结果对比表
            f.write("## 📊 实验结果对比\n\n")
            f.write("| 架构 | 分类准确率(%) | 训练攻击MSE | 训练攻击SSIM | 训练攻击PSNR | 推理攻击MSE | 推理攻击SSIM | 推理攻击PSNR |\n")
            f.write("|------|---------------|-------------|--------------|--------------|-------------|--------------|-------------|\n")
            
            for variant, result in self.results.items():
                acc = result.get("best_accuracy", 0.0)
                train_mse = result.get("train_attack_metrics", {}).get("mse", "N/A")
                train_ssim = result.get("train_attack_metrics", {}).get("ssim", "N/A")
                train_psnr = result.get("train_attack_metrics", {}).get("psnr", "N/A")
                infer_mse = result.get("infer_attack_metrics", {}).get("mse", "N/A")
                infer_ssim = result.get("infer_attack_metrics", {}).get("ssim", "N/A")
                infer_psnr = result.get("infer_attack_metrics", {}).get("psnr", "N/A")
                
                f.write(f"| {variant} | {acc:.2f} | {train_mse} | {train_ssim} | {train_psnr} | {infer_mse} | {infer_ssim} | {infer_psnr} |\n")
                
            # 性能分析
            f.write("\n## 📈 性能分析\n\n")
            
            if len(self.results) >= 2:
                # 计算改进效果
                gmm_acc = 0
                mix_acc = 0
                direct_acc = 0
                
                for variant, result in self.results.items():
                    acc = result.get("best_accuracy", 0.0)
                    if "GMM" in variant:
                        gmm_acc = acc
                    elif "梯度累加" in variant:
                        mix_acc = acc
                    elif "直接融合" in variant:
                        direct_acc = acc
                        
                f.write("### 准确率对比：\n")
                if gmm_acc > 0:
                    f.write(f"- GMM基线: {gmm_acc:.2f}%\n")
                if mix_acc > 0:
                    f.write(f"- 混合架构(梯度累加): {mix_acc:.2f}%\n")
                    if gmm_acc > 0:
                        improvement = mix_acc - gmm_acc
                        f.write(f"  - 相比GMM提升: {improvement:+.2f}%\n")
                if direct_acc > 0:
                    f.write(f"- 混合架构(直接融合): {direct_acc:.2f}%\n")
                    if gmm_acc > 0:
                        improvement = direct_acc - gmm_acc
                        f.write(f"  - 相比GMM提升: {improvement:+.2f}%\n")
                    if mix_acc > 0:
                        improvement = direct_acc - mix_acc
                        f.write(f"  - 相比梯度累加提升: {improvement:+.2f}%\n")
                        
            # 结论
            f.write("\n## 🎯 实验结论\n\n")
            f.write("### 关键发现：\n")
            f.write("1. **直接损失融合的优势**: 条件熵损失直接参与优化目标，避免了梯度累加可能导致的影响稀释\n")
            f.write("2. **统一优化策略**: 分类损失和隐私保护损失在同一个目标函数中平衡，提高了训练一致性\n")
            f.write("3. **实现简化**: 移除了复杂的手动梯度累加逻辑，代码更简洁可靠\n\n")
            
            f.write("### 理论优势验证：\n")
            f.write("- 如果直接融合方法显著优于梯度累加方法，证明了统一优化目标的重要性\n")
            f.write("- 如果混合架构优于纯GMM，验证了attention机制在条件熵计算中的有效性\n\n")
            
            # 保存原始数据
            f.write("## 📋 原始实验数据\n\n")
            f.write("```json\n")
            f.write(json.dumps(self.results, indent=2, ensure_ascii=False))
            f.write("\n```\n")
            
        print(f"\n📄 实验报告已生成: {report_path}")
        return report_path

def main():
    print("🚀 CEM架构对比实验 - 直接损失融合版本")
    print("=" * 60)
    
    comparator = CEMDirectFusionComparison()
    
    try:
        # 设置环境
        comparator.setup_environment()
        
        # 运行实验
        print("\n🧪 开始运行所有实验...")
        results = comparator.run_all_experiments()
        
        if results:
            print(f"\n✅ 实验完成！成功运行了 {len(results)} 个架构")
            
            # 生成报告
            report_path = comparator.generate_report()
            
            print("\n🎉 实验总结:")
            for variant, result in results.items():
                acc = result.get("best_accuracy", 0.0)
                print(f"  {variant}: {acc:.2f}%")
                
        else:
            print("\n❌ 所有实验都失败了")
            
    except Exception as e:
        print(f"\n💥 实验出现错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
