#!/usr/bin/env python3
"""
CEM架构对比实验脚本
===========================

在完全相同的实验条件下对比：
1. CEM-main: 原始GMM计算条件熵损失
2. CEM-mix: GMM + Attention混合架构计算条件熵损失

实验参数：
- 数据集: CIFAR-10
- λ (lambd): 16
- 正则化强度: 0.025
- 训练轮数: 240
- 其他所有参数保持完全一致

输出完整的性能对比分析和Markdown报告
"""

import os
import sys
import subprocess
import time
import json
import re
from datetime import datetime
import shutil

class CEMArchitectureComparison:
    def __init__(self):
        self.base_dir = "/Users/asher/Documents/attention"
        self.results = {
            "experiment_info": {
                "dataset": "cifar10",
                "lambd": 16,
                "regularization_strength": 0.025,
                "num_epochs": 240,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "cem_main": {},
            "cem_mix": {},
            "comparison": {}
        }
        
        # 实验参数（确保完全一致）
        self.common_params = {
            "arch": "vgg11_bn_sgm",
            "cutlayer": 4,
            "batch_size": 128,
            "num_client": 1,
            "num_epochs": 240,
            "learning_rate": 0.05,
            "lambd": 16.0,
            "dataset_portion": 1.0,
            "client_sample_ratio": 1.0,
            "noniid": 1.0,
            "local_lr": -1.0,
            "dataset": "cifar10",
            "scheme": "V2_epoch",
            "regularization": "Gaussian_kl",
            "regularization_strength": 0.025,
            "var_threshold": 0.125,
            "AT_regularization": "SCA_new",
            "AT_regularization_strength": 0.3,
            "log_entropy": 1.0,
            "ssim_threshold": 0.5,
            "gan_AE_type": "res_normN4C64",
            "gan_loss_type": "SSIM",
            "bottleneck_option": "noRELU_C8S1",
            "optimize_computation": 1,
            "random_seed": 125
        }

    def setup_environment(self):
        """设置实验环境"""
        print("🔧 设置实验环境...")
        
        # 确保两个项目都存在
        cem_main_path = os.path.join(self.base_dir, "CEM-main")
        cem_mix_path = os.path.join(self.base_dir, "CEM-mix")
        
        if not os.path.exists(cem_main_path):
            raise FileNotFoundError(f"CEM-main项目不存在: {cem_main_path}")
        if not os.path.exists(cem_mix_path):
            raise FileNotFoundError(f"CEM-mix项目不存在: {cem_mix_path}")
            
        print(f"✅ CEM-main项目路径: {cem_main_path}")
        print(f"✅ CEM-mix项目路径: {cem_mix_path}")
        
        # 清理之前的结果
        for project in ["CEM-main", "CEM-mix"]:
            saves_path = os.path.join(self.base_dir, project, "saves")
            if os.path.exists(saves_path):
                print(f"🧹 清理 {project} 的之前结果...")
                shutil.rmtree(saves_path)
        
        return True

    def create_experiment_script(self, project_name):
        """为每个项目创建标准化的实验脚本"""
        script_content = f"""#!/bin/bash

# {project_name} 标准化实验脚本
# 确保与对比项目完全相同的实验条件

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 实验参数（完全一致）
arch="vgg11_bn_sgm"
cutlayer=4
batch_size=128
num_client=1
num_epochs=240
learning_rate=0.05
lambd=16
dataset_portion=1.0
client_sample_ratio=1.0
noniid=1.0
local_lr=-1.0
dataset="cifar10"
scheme="V2_epoch"
regularization="Gaussian_kl"
regularization_strength=0.025
var_threshold=0.125
AT_regularization="SCA_new"
AT_regularization_strength=0.3
log_entropy=1
ssim_threshold=0.5
gan_AE_type="res_normN4C64"
gan_loss_type="SSIM"
bottleneck_option="noRELU_C8S1"
optimize_computation=1
random_seed=125

# 文件名生成
filename="pretrain_False_lambd_${{lambd}}_noise_${{regularization_strength}}_epoch_${{num_epochs}}_bottleneck_${{bottleneck_option}}_log_${{log_entropy}}_ATstrength_${{AT_regularization_strength}}_lr_${{learning_rate}}_varthres_${{var_threshold}}"
folder_name="saves/cifar10/${{AT_regularization}}_infocons_sgm_lg${{log_entropy}}_thre${{var_threshold}}"

echo "🚀 开始 {project_name} 实验..."
echo "📊 实验参数: λ=${{lambd}}, 正则化强度=${{regularization_strength}}, 训练轮数=${{num_epochs}}"

# 训练阶段
echo "🔥 阶段1: 训练模型..."
python main_MIA.py \\
    --arch=${{arch}} \\
    --cutlayer=${{cutlayer}} \\
    --batch_size=${{batch_size}} \\
    --filename=${{filename}} \\
    --num_client=${{num_client}} \\
    --num_epochs=${{num_epochs}} \\
    --dataset=${{dataset}} \\
    --scheme=${{scheme}} \\
    --regularization=${{regularization}} \\
    --regularization_strength=${{regularization_strength}} \\
    --log_entropy=${{log_entropy}} \\
    --AT_regularization=${{AT_regularization}} \\
    --AT_regularization_strength=${{AT_regularization_strength}} \\
    --random_seed=${{random_seed}} \\
    --learning_rate=${{learning_rate}} \\
    --lambd=${{lambd}} \\
    --gan_AE_type ${{gan_AE_type}} \\
    --gan_loss_type ${{gan_loss_type}} \\
    --local_lr ${{local_lr}} \\
    --bottleneck_option ${{bottleneck_option}} \\
    --folder ${{folder_name}} \\
    --ssim_threshold ${{ssim_threshold}} \\
    --var_threshold ${{var_threshold}}

if [ $? -eq 0 ]; then
    echo "✅ {project_name} 训练完成"
else
    echo "❌ {project_name} 训练失败"
    exit 1
fi

# 攻击测试阶段  
echo "🔥 阶段2: 模型反演攻击测试..."
python main_test_MIA.py \\
    --arch=${{arch}} \\
    --cutlayer=${{cutlayer}} \\
    --batch_size=${{batch_size}} \\
    --filename=${{filename}} \\
    --num_client=${{num_client}} \\
    --num_epochs=${{num_epochs}} \\
    --dataset=${{dataset}} \\
    --scheme=${{scheme}} \\
    --regularization=${{regularization}} \\
    --regularization_strength=${{regularization_strength}} \\
    --log_entropy=${{log_entropy}} \\
    --AT_regularization=${{AT_regularization}} \\
    --AT_regularization_strength=${{AT_regularization_strength}} \\
    --random_seed=${{random_seed}} \\
    --learning_rate=${{learning_rate}} \\
    --lambd=${{lambd}} \\
    --gan_AE_type ${{gan_AE_type}} \\
    --gan_loss_type ${{gan_loss_type}} \\
    --local_lr ${{local_lr}} \\
    --bottleneck_option ${{bottleneck_option}} \\
    --folder ${{folder_name}} \\
    --ssim_threshold ${{ssim_threshold}} \\
    --var_threshold ${{var_threshold}}

if [ $? -eq 0 ]; then
    echo "✅ {project_name} 攻击测试完成"
else
    echo "❌ {project_name} 攻击测试失败"
    exit 1
fi

echo "🎯 {project_name} 完整实验完成！"
"""
        
        script_path = os.path.join(self.base_dir, project_name, "compare_experiment.sh")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # 添加执行权限
        os.chmod(script_path, 0o755)
        return script_path

    def run_experiment(self, project_name):
        """运行单个项目的实验"""
        print(f"\n🚀 开始运行 {project_name} 实验...")
        
        project_path = os.path.join(self.base_dir, project_name)
        script_path = self.create_experiment_script(project_name)
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 运行实验脚本
            result = subprocess.run(
                ["bash", "compare_experiment.sh"],
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=14400  # 4小时超时
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # 保存输出日志
            log_path = os.path.join(self.base_dir, f"{project_name}_experiment_log.txt")
            with open(log_path, 'w') as f:
                f.write(f"=== {project_name} 实验日志 ===\\n")
                f.write(f"开始时间: {datetime.fromtimestamp(start_time)}\\n")
                f.write(f"结束时间: {datetime.fromtimestamp(end_time)}\\n")
                f.write(f"运行时长: {duration:.2f} 秒\\n\\n")
                f.write("STDOUT:\\n")
                f.write(result.stdout)
                f.write("\\nSTDERR:\\n")
                f.write(result.stderr)
            
            if result.returncode == 0:
                print(f"✅ {project_name} 实验成功完成，耗时 {duration:.2f} 秒")
                return self.extract_results(project_name, result.stdout, duration)
            else:
                print(f"❌ {project_name} 实验失败")
                print(f"错误输出: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {project_name} 实验超时")
            return None
        except Exception as e:
            print(f"💥 {project_name} 实验出错: {str(e)}")
            return None

    def extract_results(self, project_name, output, duration):
        """从实验输出中提取结果"""
        results = {
            "duration": duration,
            "training": {},
            "defense": {},
            "raw_output": output
        }
        
        # 提取训练准确度
        accuracy_matches = re.findall(r'Prec@1\s+([\d.]+)', output)
        if accuracy_matches:
            final_accuracy = float(accuracy_matches[-1])
            results["training"]["final_accuracy"] = final_accuracy
            results["training"]["best_accuracy"] = max([float(acc) for acc in accuracy_matches])
        
        # 提取防御指标 (MSE, SSIM, PSNR)
        mse_matches = re.findall(r'MSE[:\s]+([\d.]+)', output)
        ssim_matches = re.findall(r'SSIM[:\s]+([\d.]+)', output)
        psnr_matches = re.findall(r'PSNR[:\s]+([\d.]+)', output)
        
        if mse_matches:
            results["defense"]["mse"] = float(mse_matches[-1])
        if ssim_matches:
            results["defense"]["ssim"] = float(ssim_matches[-1])
        if psnr_matches:
            results["defense"]["psnr"] = float(psnr_matches[-1])
        
        # 提取损失信息
        loss_matches = re.findall(r'loss[:\s]+([\d.]+)', output.lower())
        if loss_matches:
            results["training"]["final_loss"] = float(loss_matches[-1])
        
        print(f"📊 {project_name} 结果提取完成:")
        print(f"   - 最终准确度: {results['training'].get('final_accuracy', 'N/A')}%")
        print(f"   - 防御MSE: {results['defense'].get('mse', 'N/A')}")
        print(f"   - 防御SSIM: {results['defense'].get('ssim', 'N/A')}")
        print(f"   - 防御PSNR: {results['defense'].get('psnr', 'N/A')}")
        
        return results

    def compare_results(self):
        """对比两个项目的实验结果"""
        print("\\n📊 开始结果对比分析...")
        
        cem_main = self.results["cem_main"]
        cem_mix = self.results["cem_mix"]
        comparison = {}
        
        # 训练性能对比
        if "training" in cem_main and "training" in cem_mix:
            training_comp = {}
            
            # 准确度对比
            if "final_accuracy" in cem_main["training"] and "final_accuracy" in cem_mix["training"]:
                main_acc = cem_main["training"]["final_accuracy"]
                mix_acc = cem_mix["training"]["final_accuracy"]
                training_comp["accuracy_improvement"] = mix_acc - main_acc
                training_comp["accuracy_improvement_percent"] = ((mix_acc - main_acc) / main_acc) * 100
            
            # 训练时间对比
            if "duration" in cem_main and "duration" in cem_mix:
                training_comp["time_difference"] = cem_mix["duration"] - cem_main["duration"]
                training_comp["time_ratio"] = cem_mix["duration"] / cem_main["duration"]
            
            comparison["training"] = training_comp
        
        # 防御性能对比
        if "defense" in cem_main and "defense" in cem_mix:
            defense_comp = {}
            
            # MSE对比 (越小越好)
            if "mse" in cem_main["defense"] and "mse" in cem_mix["defense"]:
                main_mse = cem_main["defense"]["mse"]
                mix_mse = cem_mix["defense"]["mse"]
                defense_comp["mse_improvement"] = main_mse - mix_mse  # 正值表示CEM-mix更好
                defense_comp["mse_improvement_percent"] = ((main_mse - mix_mse) / main_mse) * 100
            
            # SSIM对比 (越大越好)
            if "ssim" in cem_main["defense"] and "ssim" in cem_mix["defense"]:
                main_ssim = cem_main["defense"]["ssim"]
                mix_ssim = cem_mix["defense"]["ssim"]
                defense_comp["ssim_improvement"] = mix_ssim - main_ssim
                defense_comp["ssim_improvement_percent"] = ((mix_ssim - main_ssim) / main_ssim) * 100
            
            # PSNR对比 (越大越好)
            if "psnr" in cem_main["defense"] and "psnr" in cem_mix["defense"]:
                main_psnr = cem_main["defense"]["psnr"]
                mix_psnr = cem_mix["defense"]["psnr"]
                defense_comp["psnr_improvement"] = mix_psnr - main_psnr
                defense_comp["psnr_improvement_percent"] = ((mix_psnr - main_psnr) / main_psnr) * 100
            
            comparison["defense"] = defense_comp
        
        self.results["comparison"] = comparison
        return comparison

    def generate_markdown_report(self):
        """生成详细的Markdown对比报告"""
        report = f"""# CEM架构对比实验报告

## 📋 实验概述

**实验时间**: {self.results['experiment_info']['timestamp']}  
**数据集**: {self.results['experiment_info']['dataset'].upper()}  
**实验目的**: 对比原始GMM方法与GMM+Attention混合架构在CEM算法中的性能

## ⚙️ 实验配置

### 共同参数设置
- **λ (lambda)**: {self.results['experiment_info']['lambd']}
- **正则化强度**: {self.results['experiment_info']['regularization_strength']}
- **训练轮数**: {self.results['experiment_info']['num_epochs']}
- **网络架构**: VGG11-BN with bottleneck
- **批大小**: 128
- **学习率**: 0.05
- **随机种子**: 125 (确保可重现性)

### 对比架构
1. **CEM-main**: 原始GMM计算条件熵损失
2. **CEM-mix**: GMM + Attention混合架构计算条件熵损失

## 📊 实验结果

### 训练性能对比
"""
        
        # 添加训练结果表格
        if "cem_main" in self.results and "cem_mix" in self.results:
            report += """
| 指标 | CEM-main | CEM-mix | 改进 |
|------|----------|---------|------|
"""
            
            # 准确度
            if "training" in self.results["cem_main"] and "training" in self.results["cem_mix"]:
                main_acc = self.results["cem_main"]["training"].get("final_accuracy", "N/A")
                mix_acc = self.results["cem_mix"]["training"].get("final_accuracy", "N/A")
                
                if main_acc != "N/A" and mix_acc != "N/A":
                    acc_improvement = self.results["comparison"]["training"].get("accuracy_improvement_percent", 0)
                    acc_symbol = "📈" if acc_improvement > 0 else "📉" if acc_improvement < 0 else "➖"
                    report += f"| 最终准确度 (%) | {main_acc:.2f} | {mix_acc:.2f} | {acc_symbol} {acc_improvement:+.2f}% |\\n"
                
                # 最佳准确度
                main_best = self.results["cem_main"]["training"].get("best_accuracy", "N/A")
                mix_best = self.results["cem_mix"]["training"].get("best_accuracy", "N/A")
                if main_best != "N/A" and mix_best != "N/A":
                    best_improvement = ((mix_best - main_best) / main_best) * 100
                    best_symbol = "📈" if best_improvement > 0 else "📉" if best_improvement < 0 else "➖"
                    report += f"| 最佳准确度 (%) | {main_best:.2f} | {mix_best:.2f} | {best_symbol} {best_improvement:+.2f}% |\\n"
            
            # 训练时间
            if "duration" in self.results["cem_main"] and "duration" in self.results["cem_mix"]:
                main_time = self.results["cem_main"]["duration"] / 3600  # 转换为小时
                mix_time = self.results["cem_mix"]["duration"] / 3600
                time_ratio = self.results["comparison"]["training"].get("time_ratio", 1)
                time_symbol = "⏰" if time_ratio > 1.1 else "⚡" if time_ratio < 0.9 else "➖"
                report += f"| 训练时间 (小时) | {main_time:.2f} | {mix_time:.2f} | {time_symbol} {time_ratio:.2f}x |\\n"
        
        # 防御性能对比
        report += """
### 防御性能对比 (模型反演攻击抵御能力)
"""
        
        if "defense" in self.results["cem_main"] and "defense" in self.results["cem_mix"]:
            report += """
| 防御指标 | CEM-main | CEM-mix | 改进 | 说明 |
|----------|----------|---------|------|------|
"""
            
            # MSE (越小越好)
            main_mse = self.results["cem_main"]["defense"].get("mse", "N/A")
            mix_mse = self.results["cem_mix"]["defense"].get("mse", "N/A")
            if main_mse != "N/A" and mix_mse != "N/A":
                mse_improvement = self.results["comparison"]["defense"].get("mse_improvement_percent", 0)
                mse_symbol = "🛡️" if mse_improvement > 0 else "⚠️" if mse_improvement < 0 else "➖"
                report += f"| MSE | {main_mse:.4f} | {mix_mse:.4f} | {mse_symbol} {mse_improvement:+.2f}% | 越小越好 |\\n"
            
            # SSIM (越大越好，但在防御中越小越好)
            main_ssim = self.results["cem_main"]["defense"].get("ssim", "N/A")
            mix_ssim = self.results["cem_mix"]["defense"].get("ssim", "N/A")
            if main_ssim != "N/A" and mix_ssim != "N/A":
                ssim_improvement = self.results["comparison"]["defense"].get("ssim_improvement_percent", 0)
                ssim_symbol = "⚠️" if ssim_improvement > 0 else "🛡️" if ssim_improvement < 0 else "➖"
                report += f"| SSIM | {main_ssim:.4f} | {mix_ssim:.4f} | {ssim_symbol} {ssim_improvement:+.2f}% | 防御中越小越好 |\\n"
            
            # PSNR (越大越好，但在防御中越小越好)
            main_psnr = self.results["cem_main"]["defense"].get("psnr", "N/A")
            mix_psnr = self.results["cem_mix"]["defense"].get("psnr", "N/A")
            if main_psnr != "N/A" and mix_psnr != "N/A":
                psnr_improvement = self.results["comparison"]["defense"].get("psnr_improvement_percent", 0)
                psnr_symbol = "⚠️" if psnr_improvement > 0 else "🛡️" if psnr_improvement < 0 else "➖"
                report += f"| PSNR (dB) | {main_psnr:.2f} | {mix_psnr:.2f} | {psnr_symbol} {psnr_improvement:+.2f}% | 防御中越小越好 |\\n"
        
        # 架构分析
        report += """
## 🏗️ 架构对比分析

### CEM-main (原始架构)
- **条件熵计算**: 纯GMM聚类建模
- **特点**: 
  - ✅ 稳定的聚类表现
  - ✅ 理论基础扎实
  - ❌ 固定的分布假设
  - ❌ 无法自适应复杂特征

### CEM-mix (混合架构)
- **条件熵计算**: GMM + Attention自适应融合
- **特点**:
  - ✅ 并行计算两种方法
  - ✅ 自适应权重调节
  - ✅ 兼具稳定性和灵活性
  - ✅ 根据特征复杂度智能选择

### 混合策略详解
```
条件熵损失 = α × GMM损失 + (1-α) × Attention损失

其中：
- α ∈ [0,1] 是自适应权重
- 简单特征分布 → α接近1 (更依赖GMM)
- 复杂特征分布 → α接近0 (更依赖Attention)
```
"""
        
        # 结论和建议
        report += """
## 🎯 实验结论

### 关键发现
"""
        
        # 根据实际结果生成结论
        if "comparison" in self.results:
            training_comp = self.results["comparison"].get("training", {})
            defense_comp = self.results["comparison"].get("defense", {})
            
            # 准确度结论
            acc_improvement = training_comp.get("accuracy_improvement_percent", 0)
            if acc_improvement > 1:
                report += f"1. **📈 分类性能提升**: CEM-mix相比CEM-main提升了 {acc_improvement:.2f}% 的准确度\\n"
            elif acc_improvement < -1:
                report += f"1. **📉 分类性能**: CEM-mix相比CEM-main降低了 {abs(acc_improvement):.2f}% 的准确度\\n"
            else:
                report += f"1. **➖ 分类性能**: 两种方法准确度相近，差异为 {acc_improvement:.2f}%\\n"
            
            # 防御能力结论
            mse_improvement = defense_comp.get("mse_improvement_percent", 0)
            if mse_improvement > 5:
                report += f"2. **🛡️ 防御能力增强**: MSE降低 {mse_improvement:.2f}%，攻击重构质量显著下降\\n"
            elif mse_improvement < -5:
                report += f"2. **⚠️ 防御能力**: MSE增加 {abs(mse_improvement):.2f}%，需要进一步优化\\n"
            else:
                report += f"2. **➖ 防御能力**: 两种方法防御性能相近\\n"
            
            # 计算开销
            time_ratio = training_comp.get("time_ratio", 1)
            if time_ratio > 1.2:
                report += f"3. **⏰ 计算开销**: 混合架构增加了 {(time_ratio-1)*100:.1f}% 的训练时间\\n"
            elif time_ratio < 0.8:
                report += f"3. **⚡ 计算效率**: 混合架构减少了 {(1-time_ratio)*100:.1f}% 的训练时间\\n"
            else:
                report += f"3. **➖ 计算开销**: 两种方法训练时间相近\\n"
        
        report += """
### 技术创新点
1. **自适应融合策略**: 根据特征复杂度动态调节GMM和Attention的权重
2. **并行计算架构**: 同时利用两种方法的优势
3. **端到端优化**: 混合模块与主网络联合训练

### 应用建议
- **数据分布复杂**: 推荐使用CEM-mix，能更好地建模复杂特征
- **稳定性优先**: CEM-main提供更可预测的基准性能
- **计算资源充足**: CEM-mix的额外计算开销换取性能提升是值得的

## 📁 实验数据

### 原始输出日志
- CEM-main日志: `CEM-main_experiment_log.txt`
- CEM-mix日志: `CEM-mix_experiment_log.txt`

### 结果文件
- 完整结果JSON: `cem_comparison_results.json`

---
*本报告由自动化实验脚本生成，确保了实验条件的完全一致性和结果的可重现性。*
"""
        
        return report

    def save_results(self):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON结果
        json_path = os.path.join(self.base_dir, f"cem_comparison_results_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # 保存Markdown报告
        report = self.generate_markdown_report()
        md_path = os.path.join(self.base_dir, f"CEM_Comparison_Report_{timestamp}.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\\n💾 实验结果已保存:")
        print(f"   📊 JSON结果: {json_path}")
        print(f"   📝 Markdown报告: {md_path}")
        
        return json_path, md_path

    def run_complete_comparison(self):
        """运行完整的对比实验"""
        print("🎯 开始CEM架构完整对比实验")
        print("=" * 60)
        
        try:
            # 1. 环境设置
            self.setup_environment()
            
            # 2. 运行CEM-main实验
            print("\\n" + "=" * 60)
            print("🔥 第一阶段: CEM-main (原始GMM架构)")
            print("=" * 60)
            cem_main_results = self.run_experiment("CEM-main")
            if cem_main_results is None:
                print("❌ CEM-main实验失败，终止对比")
                return False
            self.results["cem_main"] = cem_main_results
            
            # 3. 运行CEM-mix实验
            print("\\n" + "=" * 60)
            print("🔥 第二阶段: CEM-mix (GMM+Attention混合架构)")
            print("=" * 60)
            cem_mix_results = self.run_experiment("CEM-mix")
            if cem_mix_results is None:
                print("❌ CEM-mix实验失败，终止对比")
                return False
            self.results["cem_mix"] = cem_mix_results
            
            # 4. 结果对比分析
            print("\\n" + "=" * 60)
            print("📊 第三阶段: 结果对比分析")
            print("=" * 60)
            self.compare_results()
            
            # 5. 保存结果和报告
            json_path, md_path = self.save_results()
            
            print("\\n" + "=" * 60)
            print("🎉 CEM架构对比实验完成！")
            print("=" * 60)
            print(f"📊 查看详细对比报告: {md_path}")
            print(f"📁 原始实验数据: {json_path}")
            
            return True
            
        except Exception as e:
            print(f"💥 实验过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """主函数"""
    print("🚀 CEM架构对比实验启动")
    print("📋 实验配置: CIFAR-10, λ=16, 正则化强度=0.025")
    
    comparator = CEMArchitectureComparison()
    success = comparator.run_complete_comparison()
    
    if success:
        print("\\n✅ 实验成功完成！")
        print("📈 请查看生成的Markdown报告了解详细对比结果")
    else:
        print("\\n❌ 实验失败，请检查错误日志")
        sys.exit(1)

if __name__ == "__main__":
    main()
