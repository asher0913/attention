# 🔍 CEM项目代码改动详细分析

## 📋 项目概览

| 项目 | 主要改动 | 文件数量 | 核心思路 |
|------|---------|---------|---------|
| CEM-att | 用Attention替换GMM | ~8个文件 | Slot + Cross Attention |
| CEM-mix | GMM + Attention混合 | ~6个文件 | 并行混合架构 |
| CEM-enhanced | 串行增强架构 | ~5个文件 | 串行 + 多尺度特征 |

---

## 🎯 **CEM-att 项目改动详情**

### 📁 **新增文件**
无新增文件，都是修改原有文件

### 📝 **修改的文件**

#### **1. model_training.py** (主要改动)

**新增代码块1: Attention模块定义 (第41-137行)**
```python
class SlotAttention(nn.Module):
    """
    Slot Attention module as requested by user:
    1. 先对feature做一遍slot attention
    2. 然后把这个slot attention作为一个KV输到一个cross attention里面
    3. 然后Q就是原feature
    """
    def __init__(self, feature_dim, num_slots=8, num_iterations=3):
        super().__init__()
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.feature_dim = feature_dim
        
        # Slot initialization parameters
        self.slot_mu = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, feature_dim))
        
        # Attention layers
        self.to_q = nn.Linear(feature_dim, feature_dim, bias=False)
        self.to_k = nn.Linear(feature_dim, feature_dim, bias=False) 
        self.to_v = nn.Linear(feature_dim, feature_dim, bias=False)
        
        # Slot update MLP
        self.slot_mlp = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
    def forward(self, features):
        # features: [batch_size, feature_dim]
        batch_size = features.size(0)
        
        # Initialize slots
        mu = self.slot_mu.expand(batch_size, self.num_slots, -1)
        sigma = self.slot_log_sigma.exp().expand(batch_size, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(sigma)
        
        # Iterative attention updates
        for _ in range(self.num_iterations):
            # Attention mechanism
            q = self.to_q(slots)  # [batch_size, num_slots, feature_dim]
            k = self.to_k(features.unsqueeze(1))  # [batch_size, 1, feature_dim]
            v = self.to_v(features.unsqueeze(1))  # [batch_size, 1, feature_dim]
            
            # Compute attention weights
            attn = torch.softmax(torch.sum(q * k, dim=-1, keepdim=True) / (self.feature_dim ** 0.5), dim=1)
            
            # Update slots
            updates = torch.sum(attn * v, dim=1, keepdim=True)  # [batch_size, 1, feature_dim]
            slots = slots + self.slot_mlp(updates.expand(-1, self.num_slots, -1))
        
        return slots

class CrossAttention(nn.Module):
    """
    Cross Attention module where:
    - Q comes from original features
    - K,V come from slot attention outputs
    """
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.head_dim = feature_dim // num_heads
        
        self.to_q = nn.Linear(feature_dim, feature_dim, bias=False)
        self.to_k = nn.Linear(feature_dim, feature_dim, bias=False)
        self.to_v = nn.Linear(feature_dim, feature_dim, bias=False)
        self.to_out = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, query_features, slot_features):
        batch_size = query_features.size(0)
        
        # Generate Q, K, V
        q = self.to_q(query_features).view(batch_size, 1, self.num_heads, self.head_dim)
        k = self.to_k(slot_features).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.to_v(slot_features).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Attention computation
        attn = torch.softmax(torch.sum(q * k, dim=-1) / (self.head_dim ** 0.5), dim=1)
        out = torch.sum(attn.unsqueeze(-1) * v, dim=1)
        out = out.view(batch_size, self.feature_dim)
        
        return self.to_out(out)

class SlotCrossAttentionCEM(nn.Module):
    """
    Complete Slot + Cross Attention module for CEM conditional entropy calculation
    Replaces GMM in compute_class_means function
    """
    def __init__(self, feature_dim=128, num_slots=8):
        super().__init__()
        self.slot_attention = SlotAttention(feature_dim, num_slots)
        self.cross_attention = CrossAttention(feature_dim)
        
    def forward(self, features, labels, unique_labels):
        """
        Replace GMM-based conditional entropy calculation with attention mechanism
        """
        total_conditional_entropy = 0.0
        
        for label in unique_labels:
            # Get features for this class
            class_mask = (labels == label.item())
            if class_mask.sum() == 0:
                continue
                
            class_features = features[class_mask]
            
            if class_features.size(0) == 1:
                # Single sample, entropy is 0
                continue
                
            # Apply slot attention
            slot_representations = self.slot_attention(class_features)
            
            # Apply cross attention for each feature
            enhanced_features = []
            for i in range(class_features.size(0)):
                enhanced_feat = self.cross_attention(
                    class_features[i:i+1], 
                    slot_representations[i:i+1]
                )
                enhanced_features.append(enhanced_feat)
            
            enhanced_features = torch.cat(enhanced_features, dim=0)
            
            # Calculate conditional entropy based on enhanced features
            feature_variance = torch.var(enhanced_features, dim=0)
            conditional_entropy = torch.mean(torch.log(feature_variance + 1e-8))
            
            # Weight by class frequency
            class_weight = class_mask.sum().float() / labels.size(0)
            total_conditional_entropy += class_weight * conditional_entropy
        
        return total_conditional_entropy
```

**修改代码块2: compute_class_means函数 (约第950-985行)**
```python
def compute_class_means(self, features, labels, unique_labels, centroids_list):
    """
    🚀 ATTENTION-BASED: 使用Attention机制替代GMM计算条件熵
    """
    # 初始化attention模块
    if not hasattr(self, 'attention_cem'):
        # 动态确定feature_dim
        if len(features.shape) == 4:  # [batch, channel, height, width]
            feature_dim = features.shape[1] * features.shape[2] * features.shape[3]
        else:  # [batch, feature_dim]
            feature_dim = features.shape[1]
            
        self.attention_cem = SlotCrossAttentionCEM(feature_dim=feature_dim).to(features.device)
    
    # 展平特征
    if len(features.shape) == 4:
        features_flat = features.view(features.size(0), -1)
    else:
        features_flat = features
    
    # 使用attention计算条件熵
    rob_loss = self.attention_cem(features_flat, labels, unique_labels)
    intra_class_mse = torch.tensor(0.0)
    
    return rob_loss, intra_class_mse
```

#### **2. main_MIA.py** (修改)
**修改内容**:
```python
# 第4-5行: 添加matplotlib后端设置
import matplotlib
matplotlib.use('Agg')  # 设置无显示后端，避免Qt错误

# 第8行: 修复导入
import model_training,model_training_paral_pruning  # 移除了model_training_paral
```

#### **3. main_test_MIA.py** (修改)
**修改内容**:
```python
# 第4-5行: 添加matplotlib后端设置
import matplotlib
matplotlib.use('Agg')  # 设置无显示后端，避免Qt错误

# 第11行: 修复导入
import model_training,model_training_paral_pruning  # 移除了model_training_paral
```

#### **4. run_exp.sh** (修改)
**修改内容**:
```bash
# 第27-28行: 修改参数范围
regularization_strength_list="0.025"  # 原来是 "0.01 0.025 0.05 0.1 0.15"
lambd_list="16"  # 原来是 "0 8 16"
```

---

## 🎯 **CEM-mix 项目改动详情**

### 📝 **修改的文件**

#### **1. model_training.py** (主要改动)

**新增代码块1: Attention模块 (第39-125行)**
```python
class SlotAttention(nn.Module):
    """Slot Attention module for hybrid architecture"""
    def __init__(self, feature_dim, num_slots=8, num_iterations=3):
        super().__init__()
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.feature_dim = feature_dim
        
        # Slot initialization parameters
        self.slot_mu = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, feature_dim))
        
        # Attention layers
        self.to_q = nn.Linear(feature_dim, feature_dim, bias=False)
        self.to_k = nn.Linear(feature_dim, feature_dim, bias=False) 
        self.to_v = nn.Linear(feature_dim, feature_dim, bias=False)
        
        # 简化的更新机制，避免GRU问题
        self.slot_norm = nn.LayerNorm(feature_dim)
        self.slot_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
    def forward(self, features):
        batch_size = features.size(0)
        
        # Initialize slots
        mu = self.slot_mu.expand(batch_size, self.num_slots, -1)
        sigma = self.slot_log_sigma.exp().expand(batch_size, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(sigma)
        
        # Iterative attention updates
        for _ in range(self.num_iterations):
            # Attention mechanism
            q = self.to_q(slots)
            k = self.to_k(features.unsqueeze(1))
            v = self.to_v(features.unsqueeze(1))
            
            # Compute attention weights
            attn = torch.softmax(torch.sum(q * k, dim=-1, keepdim=True) / (self.feature_dim ** 0.5), dim=1)
            
            # Update slots using LayerNorm + MLP instead of GRU
            updates = torch.sum(attn * v, dim=1, keepdim=True)
            slots = self.slot_norm(slots + self.slot_mlp(updates.expand(-1, self.num_slots, -1)))
        
        return slots

class CrossAttention(nn.Module):
    """Cross Attention module for hybrid architecture"""
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.head_dim = feature_dim // num_heads
        
        self.to_q = nn.Linear(feature_dim, feature_dim, bias=False)
        self.to_k = nn.Linear(feature_dim, feature_dim, bias=False)
        self.to_v = nn.Linear(feature_dim, feature_dim, bias=False)
        self.to_out = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, query_features, slot_features):
        batch_size = query_features.size(0)
        
        q = self.to_q(query_features).view(batch_size, 1, self.num_heads, self.head_dim)
        k = self.to_k(slot_features).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.to_v(slot_features).view(batch_size, -1, self.num_heads, self.head_dim)
        
        attn = torch.softmax(torch.sum(q * k, dim=-1) / (self.head_dim ** 0.5), dim=1)
        out = torch.sum(attn.unsqueeze(-1) * v, dim=1)
        out = out.view(batch_size, self.feature_dim)
        
        return self.to_out(out)
```

**新增代码块2: 自适应权重模块 (第127-168行)**
```python
class AdaptiveWeightModule(nn.Module):
    """动态调节GMM和Attention权重的模块"""
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.weight_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, feature_dim // 8),
            nn.ReLU(),
            nn.Linear(feature_dim // 8, 2),  # 输出GMM和Attention的权重
            nn.Softmax(dim=-1)
        )
        
    def forward(self, features):
        """
        根据特征复杂度预测GMM和Attention的融合权重
        """
        # 计算特征的统计信息
        feature_mean = torch.mean(features, dim=0)
        feature_var = torch.var(features, dim=0)
        feature_std = torch.std(features, dim=0)
        
        # 特征复杂度指标
        complexity_features = torch.cat([
            feature_mean,
            feature_var, 
            feature_std,
            torch.mean(torch.abs(features), dim=0),  # 平均绝对值
            torch.max(features, dim=0)[0],  # 最大值
            torch.min(features, dim=0)[0]   # 最小值
        ])
        
        # 预测权重
        weights = self.weight_predictor(complexity_features)
        return weights[0], weights[1]  # gmm_weight, attention_weight
```

**新增代码块3: 混合架构核心 (第170-321行)**
```python
class HybridGMMAttentionCEM(nn.Module):
    """混合GMM和Attention的CEM条件熵计算模块"""
    def __init__(self, feature_dim, num_slots=8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_slots = num_slots
        
        # Attention分支
        self.slot_attention = SlotAttention(feature_dim, num_slots)
        self.cross_attention = CrossAttention(feature_dim)
        
        # 自适应权重模块
        self.adaptive_weight = AdaptiveWeightModule(feature_dim)
        
    def forward(self, features, labels, unique_labels, centroids_list):
        """
        混合GMM和Attention计算条件熵
        """
        # 1. GMM分支计算
        gmm_loss = self.compute_gmm_branch(features, labels, unique_labels, centroids_list)
        
        # 2. Attention分支计算
        attention_loss = self.compute_attention_branch(features, labels, unique_labels)
        
        # 3. 自适应权重融合
        gmm_weight, attention_weight = self.adaptive_weight(features)
        
        # 4. 加权融合
        hybrid_loss = gmm_weight * gmm_loss + attention_weight * attention_loss
        
        return hybrid_loss
    
    def compute_gmm_branch(self, features, labels, unique_labels, centroids_list):
        """GMM分支：保持原始GMM计算逻辑"""
        # 这里保持原始的GMM计算逻辑
        # [原始GMM代码逻辑]
        pass
    
    def compute_attention_branch(self, features, labels, unique_labels):
        """Attention分支：使用Slot+Cross Attention"""
        total_conditional_entropy = 0.0
        total_samples = 0
        
        for label in unique_labels:
            class_mask = (labels == label.item())
            if class_mask.sum() <= 1:
                continue
                
            class_features = features[class_mask]
            
            # Slot Attention
            slot_representations = self.slot_attention(class_features)
            
            # Cross Attention
            enhanced_features = []
            for i in range(class_features.size(0)):
                enhanced_feat = self.cross_attention(
                    class_features[i:i+1], 
                    slot_representations[i:i+1]
                )
                enhanced_features.append(enhanced_feat)
            
            enhanced_features = torch.cat(enhanced_features, dim=0)
            
            # 条件熵计算
            if enhanced_features.shape[0] > 1:
                feature_variance = torch.var(enhanced_features, dim=0)
                conditional_entropy = torch.mean(torch.log(feature_variance + 1e-8))
            else:
                conditional_entropy = torch.tensor(0.0).to(features.device)
            
            total_conditional_entropy += conditional_entropy * class_mask.sum().float()
            total_samples += class_mask.sum().float()
        
        if total_samples > 0:
            return total_conditional_entropy / total_samples
        else:
            return torch.tensor(0.0).to(features.device)
```

**修改代码块4: compute_class_means函数 (第951-985行)**
```python
def compute_class_means(self, features, labels, unique_labels, centroids_list):
    """
    🚀 HYBRID GMM + ATTENTION: 混合架构计算条件熵
    """
    # 展平特征
    if len(features.shape) == 4:
        batch_size, channels, height, width = features.shape
        features_flat = features.view(batch_size, -1)
        feature_dim = channels * height * width
    else:
        features_flat = features
        feature_dim = features.shape[1]
    
    # 初始化混合模块
    if not hasattr(self, 'hybrid_cem'):
        self.hybrid_cem = HybridGMMAttentionCEM(feature_dim=feature_dim).to(features.device)
    
    # 使用混合架构计算条件熵
    rob_loss, intra_class_mse = self.hybrid_cem(features_flat, labels, unique_labels, centroids_list)
    
    return rob_loss, intra_class_mse
```

#### **2-4. 其他文件修改**
与CEM-att相同的matplotlib和导入修复。

---

## 🎯 **CEM-enhanced 项目改动详情**

### 📝 **修改的文件**

#### **1. model_training.py** (主要改动)

**新增代码块1: 增强Attention模块 (第42-194行)**
```python
class EnhancedSlotAttention(nn.Module):
    """增强版Slot Attention - 多层级联 + 残差连接"""
    def __init__(self, feature_dim, num_slots=12, num_iterations=4, num_layers=2):
        super().__init__()
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        
        # 多层attention
        self.attention_layers = nn.ModuleList([
            nn.ModuleDict({
                'to_q': nn.Linear(feature_dim, feature_dim, bias=False),
                'to_k': nn.Linear(feature_dim, feature_dim, bias=False),
                'to_v': nn.Linear(feature_dim, feature_dim, bias=False),
                'norm': nn.LayerNorm(feature_dim),
                'mlp': nn.Sequential(
                    nn.Linear(feature_dim, feature_dim * 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(feature_dim * 2, feature_dim)
                )
            }) for _ in range(num_layers)
        ])
        
        # Slot初始化
        self.slot_mu = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, feature_dim))
        
        # 温度退火参数
        self.register_buffer('temperature', torch.tensor(1.0))
        
    def forward(self, features):
        batch_size = features.size(0)
        
        # 初始化slots
        mu = self.slot_mu.expand(batch_size, self.num_slots, -1)
        sigma = self.slot_log_sigma.exp().expand(batch_size, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(sigma)
        
        # 多层级联attention
        for layer_idx in range(self.num_layers):
            layer = self.attention_layers[layer_idx]
            
            for iter_idx in range(self.num_iterations):
                # 温度退火
                temp = self.temperature * (0.9 ** iter_idx)
                
                # Attention计算
                q = layer['to_q'](slots)
                k = layer['to_k'](features.unsqueeze(1))
                v = layer['to_v'](features.unsqueeze(1))
                
                # 注意力权重 + 温度退火
                attn = torch.softmax(torch.sum(q * k, dim=-1, keepdim=True) / (self.feature_dim ** 0.5 * temp), dim=1)
                
                # 更新slots (残差连接)
                updates = torch.sum(attn * v, dim=1, keepdim=True)
                slots_residual = layer['norm'](slots + layer['mlp'](updates.expand(-1, self.num_slots, -1)))
                slots = slots + slots_residual  # 残差连接
        
        # Slot竞争机制
        slot_importance = torch.norm(slots, dim=-1, keepdim=True)
        slots = slots * torch.softmax(slot_importance / 0.1, dim=1)
        
        return slots

class EnhancedCrossAttention(nn.Module):
    """增强版Cross Attention - 更多头 + LayerNorm + GELU"""
    def __init__(self, feature_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.head_dim = feature_dim // num_heads
        
        self.to_q = nn.Linear(feature_dim, feature_dim, bias=False)
        self.to_k = nn.Linear(feature_dim, feature_dim, bias=False)
        self.to_v = nn.Linear(feature_dim, feature_dim, bias=False)
        
        self.norm_q = nn.LayerNorm(feature_dim)
        self.norm_k = nn.LayerNorm(feature_dim)
        self.norm_v = nn.LayerNorm(feature_dim)
        
        self.to_out = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, query_features, slot_features):
        batch_size = query_features.size(0)
        
        # 归一化 + 生成Q,K,V
        q = self.to_q(self.norm_q(query_features)).view(batch_size, 1, self.num_heads, self.head_dim)
        k = self.to_k(self.norm_k(slot_features)).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.to_v(self.norm_v(slot_features)).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Multi-head attention
        attn = torch.softmax(torch.sum(q * k, dim=-1) / (self.head_dim ** 0.5), dim=1)
        out = torch.sum(attn.unsqueeze(-1) * v, dim=1)
        out = out.view(batch_size, self.feature_dim)
        
        # 残差连接
        return query_features + self.to_out(out)
```

**新增代码块2: 串行架构核心 (第239-449行)**
```python
class SerialAttentionGMMCEM(nn.Module):
    """串行Attention→GMM架构 + 多尺度特征融合"""
    def __init__(self, feature_dim, num_classes=10, num_slots=12):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.num_slots = num_slots
        
        # 增强的Attention模块
        self.slot_attention = EnhancedSlotAttention(feature_dim, num_slots, num_iterations=4)
        self.cross_attention = EnhancedCrossAttention(feature_dim, num_heads=8)
        
        # 多尺度特征处理器
        self.multi_scale_processor = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim // 2),  # 下采样
            nn.Linear(feature_dim, feature_dim * 2),   # 上采样
            nn.Identity()  # 原始尺度
        ])
        
        # 自适应特征门控
        self.feature_gate = nn.Sequential(
            nn.Linear(feature_dim * 4, feature_dim * 2),  # 4 = 0.5 + 2 + 1 + 原始
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
        
        # 特征重构模块
        self.feature_reconstruction = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # 增强的自适应权重
        self.enhanced_adaptive_weight = EnhancedAdaptiveWeight(feature_dim)
        
    def forward(self, features, labels, unique_labels, centroids_list):
        """串行架构：多尺度特征 → Attention增强 → GMM处理"""
        
        # 1. 多尺度特征处理
        multi_scale_features = []
        for processor in self.multi_scale_processor:
            if isinstance(processor, nn.Identity):
                multi_scale_features.append(features)
            else:
                processed = processor(features)
                # 维度对齐
                if processed.size(1) != features.size(1):
                    if processed.size(1) < features.size(1):
                        processed = F.pad(processed, (0, features.size(1) - processed.size(1)))
                    else:
                        processed = processed[:, :features.size(1)]
                multi_scale_features.append(processed)
        
        # 2. 自适应特征门控
        concat_features = torch.cat(multi_scale_features + [features], dim=1)
        gate_weights = self.feature_gate(concat_features)
        gated_features = features * gate_weights
        
        # 3. 串行Attention处理
        attention_enhanced_features = self.process_attention_serial(gated_features, labels, unique_labels)
        
        # 4. 特征重构
        reconstructed_features = self.feature_reconstruction(attention_enhanced_features)
        
        # 5. 增强的GMM处理
        rob_loss = self.compute_enhanced_gmm_branch(reconstructed_features, labels, unique_labels, centroids_list)
        
        intra_class_mse = torch.tensor(0.0)
        return rob_loss, intra_class_mse
    
    def process_attention_serial(self, features, labels, unique_labels):
        """串行处理：先Slot Attention，再Cross Attention"""
        enhanced_features = []
        
        for label in unique_labels:
            class_mask = (labels == label.item())
            if class_mask.sum() <= 1:
                enhanced_features.append(features[class_mask])
                continue
                
            class_features = features[class_mask]
            
            # 步骤1: Slot Attention
            slot_representations = self.slot_attention(class_features)
            
            # 步骤2: Cross Attention (逐个处理)
            class_enhanced = []
            for i in range(class_features.size(0)):
                enhanced_feat = self.cross_attention(
                    class_features[i:i+1], 
                    slot_representations[i:i+1]
                )
                class_enhanced.append(enhanced_feat)
            
            class_enhanced = torch.cat(class_enhanced, dim=0)
            
            # 残差连接 + LayerNorm
            class_enhanced = F.layer_norm(class_features + class_enhanced, class_enhanced.shape[1:])
            enhanced_features.append(class_enhanced)
        
        # 重新组合
        result = torch.zeros_like(features)
        start_idx = 0
        for label, enhanced_feat in zip(unique_labels, enhanced_features):
            class_mask = (labels == label.item())
            result[class_mask] = enhanced_feat
        
        return result
    
    def compute_enhanced_gmm_branch(self, features, labels, unique_labels, centroids_list):
        """增强的GMM分支：处理attention增强后的特征"""
        total_reg_mutual_infor = 0.0
        total_weight = 0.0
        
        for i in unique_labels:
            i_item = i.item() if torch.is_tensor(i) else i
            centroids = centroids_list[i_item]
            class_mask = (labels == i_item)
            class_features = features[class_mask.squeeze(), :]
            
            if class_features.size(0) <= 1:
                continue
            
            # 🚀 增强距离度量：欧氏距离 + 余弦相似度
            # 欧氏距离
            euclidean_distances = torch.cdist(class_features, centroids)
            # 余弦相似度
            cos_sim = F.cosine_similarity(class_features.unsqueeze(1), centroids.unsqueeze(0), dim=2)
            cos_distances = 1 - cos_sim
            
            # 动态权重组合
            feature_var = torch.var(class_features, dim=0).mean()
            euclidean_weight = torch.sigmoid(feature_var)
            cosine_weight = 1 - euclidean_weight
            
            combined_distances = (euclidean_weight * euclidean_distances + 
                                cosine_weight * cos_distances)
            
            cluster_assignments = torch.argmin(combined_distances, dim=1).cpu().numpy()
            unique_cluster_assignments = np.unique(cluster_assignments)
            
            for j in unique_cluster_assignments:
                indice_cluster = cluster_assignments == j
                weight = sum(indice_cluster) / sum(class_mask.cpu().numpy())
                
                if sum(indice_cluster) <= 1:
                    continue
                
                cluster_features = class_features[indice_cluster]
                centroid = centroids[j]
                
                # 🚀 改进的方差计算
                variances = torch.mean((cluster_features - centroid)**2, dim=0)
                
                # 🚀 自适应正则化
                adaptive_reg = 0.001 * (1 + feature_var)
                reg_variances = variances + adaptive_reg
                
                # 🚀 改进的互信息计算
                epsilon = 1e-8
                mutual_infor = F.relu(torch.log(reg_variances + epsilon) - 
                                    torch.log(torch.tensor(adaptive_reg + epsilon)))
                
                reg_mutual_infor = mutual_infor.mean() * torch.tensor(weight)
                total_reg_mutual_infor += reg_mutual_infor
                total_weight += weight
        
        if total_weight > 0:
            return total_reg_mutual_infor / total_weight
        else:
            return torch.tensor(0.0).to(features.device)
```

**修改代码块3: 直接损失融合 (第1410-1461行)**
```python
# 🚀 CEM-ENHANCED: 直接损失融合 - 关键改进！
# 条件熵损失直接参与总损失优化，而非梯度累加
if not random_ini_centers and self.lambd > 0:
    total_loss = f_loss + self.lambd * rob_loss  # 🚀 直接融合条件熵损失
else:    
    total_loss = f_loss

# 简化训练流程：直接反向传播，移除复杂梯度累加
total_loss.backward()
```

#### **2-5. 其他文件修改**
与前两个项目相同的修复。

---

## 📊 **总结对比**

| 项目 | 核心改动 | 代码行数 | 主要新增类 |
|------|---------|---------|-----------|
| **CEM-att** | 用Attention完全替换GMM | ~200行 | `SlotAttention`, `CrossAttention`, `SlotCrossAttentionCEM` |
| **CEM-mix** | GMM+Attention并行混合 | ~300行 | 上述3个 + `AdaptiveWeightModule`, `HybridGMMAttentionCEM` |
| **CEM-enhanced** | 串行架构+多尺度特征 | ~500行 | `EnhancedSlotAttention`, `EnhancedCrossAttention`, `SerialAttentionGMMCEM`, `EnhancedAdaptiveWeight` |

**共同修改**:
- 所有项目都修复了matplotlib后端问题
- 所有项目都修复了导入错误
- 所有项目都调整了实验参数范围
- CEM-enhanced还加入了直接损失融合改进

这就是三个项目相对于CEM-main的所有代码改动详情！
