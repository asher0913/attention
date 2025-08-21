# 🛡️ CEM-att 防御效果测试指南

## 📋 概述

您已经完成了模型训练（准确率84.55%），现在只需要测试防御效果（攻击测试阶段）。

## 🚀 快速使用

### **方法1: 简单脚本（推荐）**
```bash
cd CEM-att
bash run_defense_only.sh
```

### **方法2: 智能脚本**
```bash
cd CEM-att
python run_smart_defense_test.py
```

### **方法3: 查看所有可用模型**
```bash
python run_smart_defense_test.py --list
```

## 📂 已检测到的模型

```
saves/cifar10/SCA_new_attention_fixed_lg1_thre0.125/saves/cifar10/SCA_new_attention_fixed_lg1_thre0.125/CEM_log_entropy1_cifar10_cutlayer4_archvgg11_bn_sgm_schemeV2_epoch_n_epochs240_batch_size128_lr0.05_regulastr0.025_bottlenecknoRELU_C8S1_SCA_new0.3_randomseed125_ssim0.5_lambd16
```

**训练参数**:
- Lambda: 16
- 正则化强度: 0.025  
- 训练轮数: 240
- 批大小: 128
- 学习率: 0.05

## 📊 预期输出指标

### **关键防御指标**:

1. **MSE (均方误差)**
   - 数值越低 = 防御效果越好
   - 表示攻击重建图像与原图的差异

2. **SSIM (结构相似度)**
   - 数值越低 = 防御效果越好
   - 表示攻击重建图像的结构相似程度

3. **PSNR (峰值信噪比)**
   - 数值越高 = 防御效果越好
   - 表示信号质量和噪声比

### **输出示例**:
```
Attack Test Results:
MSE: 0.1234
SSIM: 0.2345  
PSNR: 28.567
```

## 🎯 结果分析

### **与原始CEM-main (GMM)对比**:

| 指标 | GMM基线 | Attention | 改进方向 |
|------|---------|-----------|----------|
| MSE  | X.XXX   | Y.YYY     | ↓ 更低更好 |
| SSIM | X.XXX   | Y.YYY     | ↓ 更低更好 |
| PSNR | X.XXX   | Y.YYY     | ↑ 更高更好 |

### **成功标准**:
- ✅ **MSE降低** → Attention防御更强
- ✅ **SSIM降低** → 攻击重建质量更差  
- ✅ **PSNR提升** → 隐私保护更好

## 🔧 故障排除

### **如果测试失败**:

1. **检查GPU状态**:
   ```bash
   nvidia-smi
   ```

2. **使用CPU测试**:
   ```bash
   python run_smart_defense_test.py --cpu
   ```

3. **重新生成测试数据**:
   ```bash
   python generate_test_data.py
   ```

4. **查看可用模型**:
   ```bash
   python run_smart_defense_test.py --list
   ```

## 📁 文件说明

- `run_defense_only.sh` - 简单防御测试脚本
- `run_smart_defense_test.py` - 智能防御测试脚本
- `generate_test_data.py` - 生成测试数据
- `test_cifar10_image.pt` - 攻击测试图像
- `test_cifar10_label.pt` - 攻击测试标签

## ⏱️ 预计运行时间

- **攻击测试**: 约10-30分钟（取决于GPU性能）
- **总输出**: 包含详细的MSE、SSIM、PSNR指标

## 🎉 成功完成后

您将获得：
1. ✅ **完整的防御效果评估**
2. ✅ **Attention vs GMM的对比数据**  
3. ✅ **论文实验所需的关键指标**
4. ✅ **验证Attention机制的防御优势**

---

**直接在Linux服务器上运行 `bash run_defense_only.sh` 即可！** 🚀
