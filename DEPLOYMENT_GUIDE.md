# CEM架构对比实验部署指南

## 🚀 Linux服务器部署步骤

### 1. 上传项目文件
```bash
# 在您的Linux服务器上创建目录
mkdir ~/cem_comparison
cd ~/cem_comparison

# 上传以下文件到服务器 (使用scp或rsync)
# - CEM-main/ (整个文件夹)
# - CEM-mix/ (整个文件夹)  
# - compare_cem_architectures.py
# - run_comparison.sh
# - COMPARISON_GUIDE.md
```

### 2. 检查环境依赖
```bash
# 检查Python环境
python --version
pip list | grep torch

# 检查CUDA环境
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 3. 设置执行权限
```bash
chmod +x run_comparison.sh
```

### 4. 运行对比实验
```bash
# 方法1: 使用启动脚本 (推荐)
bash run_comparison.sh

# 方法2: 直接运行Python脚本
python compare_cem_architectures.py
```

## 📁 目录结构要求

```
cem_comparison/
├── CEM-main/                    # 原始GMM架构项目
│   ├── model_training.py
│   ├── main_MIA.py
│   ├── main_test_MIA.py
│   └── ...
├── CEM-mix/                     # 混合架构项目
│   ├── model_training.py
│   ├── main_MIA.py
│   ├── main_test_MIA.py
│   └── ...
├── compare_cem_architectures.py  # 对比实验脚本
├── run_comparison.sh            # 启动脚本
└── COMPARISON_GUIDE.md          # 使用指南
```

## ⚙️ 环境要求

### 硬件要求
- **GPU**: NVIDIA GPU (推荐RTX A5000或更高)
- **内存**: 至少16GB RAM
- **存储**: 至少50GB可用空间

### 软件要求
- **Python**: 3.7+
- **PyTorch**: 1.8+ with CUDA support
- **CUDA**: 11.0+
- **其他依赖**: 见各项目的requirements.txt

## 🔧 故障排除

### 常见问题
1. **路径问题**: 脚本会自动检测当前目录，确保在正确位置运行
2. **权限问题**: 确保有执行权限 `chmod +x run_comparison.sh`
3. **CUDA问题**: 检查GPU可用性和显存大小
4. **依赖问题**: 安装缺失的Python包

### 检查脚本
```bash
# 测试脚本基本功能
python -c "
import compare_cem_architectures
print('脚本导入成功')
"
```

## 📊 监控实验进度

### 实时查看日志
```bash
# 查看CEM-main实验进度
tail -f CEM-main_experiment_log.txt

# 查看CEM-mix实验进度  
tail -f CEM-mix_experiment_log.txt
```

### 检查GPU使用情况
```bash
# 监控GPU使用
watch -n 1 nvidia-smi
```

## 🎯 预期结果

实验完成后会生成：
- `CEM_Comparison_Report_*.md` - 详细对比报告
- `cem_comparison_results_*.json` - 原始实验数据
- `*_experiment_log.txt` - 实验日志

## 💡 使用建议

1. **服务器环境**: 建议在后台运行，使用screen或tmux
2. **监控资源**: 定期检查GPU显存和系统资源
3. **备份结果**: 实验完成后及时备份生成的文件
4. **参数调整**: 如需修改实验参数，编辑compare_cem_architectures.py中的common_params

---

**注意**: 脚本已优化为在任何Linux环境中运行，无需修改路径配置。
