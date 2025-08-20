# CEM-att 脚本使用指南

## 🎯 部署到Linux NVIDIA服务器 - 完整实验运行指南

### 推荐运行方案

**最佳选择：`run_exp_with_attention_option.sh`** - 最灵活，支持一键切换attention/GMM

## 📋 所有脚本详细说明

### 1. `run_exp_with_attention_option.sh` ⭐ **推荐**

**用途**：完整的CEM实验 + 可选择attention/GMM分类器
**实验内容**：
- 模型训练（CEM + attention/GMM）
- 模型反演攻击测试（MIA）
- 多参数网格搜索

**关键控制参数**：
```bash
# 第8行：选择分类器类型
USE_ATTENTION=true   # true=使用attention, false=使用GMM

# 第10-12行：Attention参数
NUM_SLOTS=8          # Slot数量
ATTENTION_HEADS=8    # 注意力头数
ATTENTION_DROPOUT=0.1 # Dropout率

# 第15-20行：基础设置
GPU_id=0             # GPU编号
arch=vgg11_bn_sgm    # 网络架构
batch_size=128       # 批大小
cutlayer_list="4"    # 切割层（VGG第4层）
num_client=1         # 客户端数量

# 第28-39行：实验参数
dataset_list="cifar10"                    # 数据集
num_epochs=240                           # 训练轮数
learning_rate=0.05                       # 学习率
lambd_list="0 8 16"                      # 条件熵权重（核心参数）
regularization_strength_list="0.01 0.025 0.05 0.1 0.15"  # 正则化强度
log_entropy=1                            # 使用对数熵
```

**运行命令**：
```bash
# 1. 编辑脚本选择attention/GMM
nano run_exp_with_attention_option.sh
# 修改第8行: USE_ATTENTION=true

# 2. 运行完整实验
bash run_exp_with_attention_option.sh
```

---

### 2. `run_exp_attention.sh` 

**用途**：专门的attention vs GMM对比实验（简化版）
**实验内容**：
- 只做训练，不做攻击测试
- 减少epoch数量（50轮）用于快速对比

**关键参数**：
```bash
use_attention=true        # 控制是否使用attention
num_epochs=50            # 轮数较少，适合快速测试
lambd_list="1 16"        # 只测试两个lambda值
batch_size=128
```

**运行命令**：
```bash
bash run_exp_attention.sh
```

---

### 3. `run_exp.sh` 

**用途**：原始CEM实验脚本（只支持GMM）
**实验内容**：
- 完整的原始CEM实验
- 使用GMM分类器
- 包含训练 + 攻击测试

**特点**：
- 🔴 **不支持attention参数**
- ✅ 完全原版实验，用于baseline对比
- ✅ 最复杂的参数设置（5种正则化强度 × 3种lambda值）

**运行命令**：
```bash
bash run_exp.sh
```

---

### 4. `example_usage.sh`

**用途**：简单的使用示例（教学用）
**实验内容**：
- 演示基本训练命令
- 包含attention和GMM两种方法

**运行命令**：
```bash
bash example_usage.sh
```

---

## 🔧 关键参数详解

### 核心CEM参数
- **`lambd`**: 条件熵权重，控制条件熵损失在总损失中的比重
  - `0`: 只用分类损失（退化为普通分类）
  - `8-16`: 平衡分类和条件熵
  - 更大值: 更强调条件熵最小化

- **`cutlayer`**: VGG网络切割层
  - `4`: 在第4层切割，前4层做特征提取

- **`log_entropy`**: 是否使用对数熵
  - `1`: 使用log形式的条件熵计算
  - `0`: 使用原始方差形式

### Attention专有参数
- **`num_slots`**: Slot Attention的slot数量（替代GMM的高斯组件数）
- **`attention_heads`**: 多头注意力的头数
- **`attention_dropout`**: 注意力模块的dropout率

### 防御/攻击参数
- **`AT_regularization`**: 激活防御类型
  - `SCA_new`: 使用SCA防御
  - `None`: 无防御

- **`regularization_strength`**: 防御强度
- **`ssim_threshold`**: SSIM阈值（用于攻击评估）

## 🚀 推荐实验流程

### 阶段1：快速验证（推荐先运行）
```bash
# 使用简化版本快速测试
bash run_exp_attention.sh
```

### 阶段2：完整对比实验
```bash
# 1. 运行attention版本
# 编辑 run_exp_with_attention_option.sh，设置 USE_ATTENTION=true
bash run_exp_with_attention_option.sh

# 2. 运行GMM baseline
# 编辑 run_exp_with_attention_option.sh，设置 USE_ATTENTION=false  
bash run_exp_with_attention_option.sh
```

## 📊 实验结果

所有结果保存在：
- **训练日志**: `saves/[实验名]/tensorboard/`
- **模型权重**: `saves/[实验名]/checkpoint_*.tar`
- **攻击结果**: `saves/[实验名]/`目录下的图片和日志

## ⚡ 性能建议

**GPU内存优化**：
- 如果GPU内存不足，减少`batch_size`（128→64→32）
- 减少`num_slots`（8→4）

**时间优化**：
- 减少`num_epochs`（240→100→50）
- 减少参数搜索范围（只测试lambd=16）

**调试模式**：
```bash
# 快速调试：只跑1个epoch
num_epochs=1
lambd_list="16"
regularization_strength_list="0.05"
```
