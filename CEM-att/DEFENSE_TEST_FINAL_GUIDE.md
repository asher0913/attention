# 🛡️ 防御测试最终解决方案

## 🔧 问题诊断
从错误信息看出两个主要问题：
1. **路径重复嵌套**: `--filename` 参数导致路径被重复添加
2. **Checkpoint文件**: 没有 `checkpoint_f_best.tar`，但有 `checkpoint_f_240.tar`

## ✅ 解决方案

### **方法1: 自动智能脚本（最推荐）**
```bash
python run_auto_defense_test.py
```
- ✅ 自动检测所有checkpoint文件
- ✅ 智能选择最佳模型（best > 240 > 最大epoch）
- ✅ 自动处理路径问题
- ✅ 详细错误诊断

### **方法2: 修复版脚本**
```bash 
bash run_defense_final.sh
```
- ✅ 智能检测best或240 checkpoint
- ✅ 修复路径重复问题
- ✅ 详细日志输出

## 📊 预期输出

成功运行后应该看到：
```
Attack Test Results:
Average MSE: 0.xxxx
Average SSIM: 0.xxxx  
Average PSNR: xx.xx
```

## 🎯 关键指标含义

| 指标 | 含义 | 理想值 |
|------|------|---------|
| **MSE** | 攻击重建误差 | ↓ 越低越好 |
| **SSIM** | 结构相似度 | ↓ 越低越好 |
| **PSNR** | 峰值信噪比 | ↑ 越高越好 |

## 🔍 如果仍然失败

1. **检查模型文件**:
   ```bash
   find saves/ -name "checkpoint_f_*.tar"
   ```

2. **手动指定checkpoint**:
   修改脚本中的 `--test_best` 为 `--num_epochs 240`

3. **查看详细错误**:
   ```bash
   python run_auto_defense_test.py 2>&1 | tee defense_test.log
   ```

## 🚀 推荐执行步骤

1. **首选自动脚本**:
   ```bash
   cd CEM-att
   python run_auto_defense_test.py
   ```

2. **如果失败，尝试备用脚本**:
   ```bash
   bash run_defense_final.sh
   ```

3. **检查结果**:
   - 查看MSE、SSIM、PSNR数值
   - 与原始GMM版本对比
   - 验证Attention机制的防御效果

---

**现在在Linux服务器上运行 `python run_auto_defense_test.py` 即可！** 🎯
