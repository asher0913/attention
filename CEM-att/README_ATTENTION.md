# CEM-att: CEM with Attention Mechanism

这是基于原始CEM-main项目，将GMM分类器替换为Attention分类器的实现。

## 主要改动

### 1. 新增文件
- `attention_modules.py`: Slot Attention和Cross Attention实现
- `quick_test.py`: 快速测试脚本
- `README_ATTENTION.md`: 本说明文件

### 2. 修改文件
- `model_training.py`: 添加attention分类器支持
- `main_MIA.py`: 添加attention相关参数
- `model_architectures/vgg.py`: 修复bottleneck相关bug

## 使用方法

### 🚀 一键运行脚本（推荐）

#### 方法1：使用修改版run_exp脚本
```bash
# 编辑 run_exp_with_attention_option.sh 
# 将第8行改为: USE_ATTENTION=true  (启用attention)
# 或保持: USE_ATTENTION=false (使用原始GMM)
bash run_exp_with_attention_option.sh
```

#### 方法2：使用专门的attention脚本
```bash
# 编辑 run_exp_attention.sh
# 将第20行改为: use_attention=true/false
bash run_exp_attention.sh
```

#### 方法3：使用原始run_exp.sh
```bash
# 原始脚本会使用GMM方法（不支持attention）
bash run_exp.sh
```

### 基本训练命令

#### 使用Attention分类器（新功能）:
```bash
python main_MIA.py \
    --filename cem_attention_exp \
    --arch vgg11_bn \
    --cutlayer 4 \
    --batch_size 128 \
    --num_epochs 100 \
    --learning_rate 0.01 \
    --lambd 1.0 \
    --dataset cifar10 \
    --use_attention_classifier \
    --num_slots 8 \
    --attention_heads 8 \
    --attention_dropout 0.1 \
    --log_entropy 1
```

#### 使用原始GMM分类器（基线）:
```bash
python main_MIA.py \
    --filename cem_baseline_exp \
    --arch vgg11_bn \
    --cutlayer 4 \
    --batch_size 128 \
    --num_epochs 100 \
    --learning_rate 0.01 \
    --lambd 1.0 \
    --dataset cifar10 \
    --log_entropy 1
```

### 新增参数说明

- `--use_attention_classifier`: 启用attention分类器（不加此参数则使用原始GMM）
- `--num_slots`: Slot Attention的slot数量（默认8）
- `--attention_heads`: 注意力头数（默认8）
- `--attention_dropout`: Attention模块的dropout率（默认0.1）

## 技术细节

### Attention机制替代GMM的原理

1. **原始GMM方法**:
   - 为每个类别训练独立的高斯混合模型（3个高斯组件）
   - 使用固定的高斯中心进行聚类
   - 计算特征到高斯中心的距离来估计条件熵

2. **Attention方法**:
   - 使用Slot Attention学习动态的slot表示（8个slots）
   - Slot表示作为动态聚类中心
   - 使用Cross Attention增强特征表示
   - 计算特征到slot中心的距离来估计条件熵

### 关键函数

- `attention_classify_features()`: 执行attention前向传播
- `compute_attention_conditional_entropy()`: 计算基于attention的条件熵损失

## 设备支持

代码自动检测并使用可用设备：
- Linux + NVIDIA GPU: 自动使用CUDA
- macOS: 使用CPU或MPS（如果可用）
- 其他: 使用CPU

## 文件结构

```
CEM-att/
├── attention_modules.py          # Attention机制实现
├── model_training.py            # 主训练类（已修改）
├── main_MIA.py                  # 训练脚本（已修改）
├── quick_test.py                # 快速测试
├── README_ATTENTION.md          # 本文件
├── model_architectures/         # 网络架构
├── datasets_torch.py            # 数据加载
├── utils.py                     # 工具函数
└── [其他原始CEM文件]            # 完全保持不变
```

## 验证

在部署前可运行快速测试：
```bash
python quick_test.py
```

## 预期改进

相比原始GMM方法，Attention机制预期提供：
1. **动态学习**: slot表示可自适应调整
2. **端到端优化**: 整个系统联合训练
3. **更强表达能力**: 突破高斯分布假设限制
4. **更好的条件熵估计**: 基于学习的聚类中心

## 兼容性

- 完全向后兼容原始CEM-main
- 不使用`--use_attention_classifier`时行为与原始版本完全相同
- 所有原始参数和功能保持不变
