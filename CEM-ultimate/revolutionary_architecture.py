# ============================================================================
# CEM-ULTIMATE: 革命性突破架构
# 5大核心突破：
# 1. 替换VGG为ResNet-18预训练特征提取器
# 2. 变分自编码器(VAE)替代传统条件熵计算  
# 3. 对抗训练增强隐私保护
# 4. 动态λ调节和多目标优化
# 5. 知识蒸馏和特征增强
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.distributions import Normal, kl_divergence
import numpy as np

class PretrainedFeatureExtractor(nn.Module):
    """突破1：使用预训练ResNet-18替代VGG"""
    def __init__(self, output_dim=128):
        super().__init__()
        # 使用预训练ResNet-18
        self.backbone = models.resnet18(pretrained=True)
        
        # 移除最后的分类层
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # 自适应特征映射
        self.feature_adapter = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # 冻结前几层，只微调后面层
        self._freeze_early_layers()
        
    def _freeze_early_layers(self):
        """冻结前6层，只训练后面的层"""
        for i, child in enumerate(self.backbone.children()):
            if i < 6:
                for param in child.parameters():
                    param.requires_grad = False
                    
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        adapted_features = self.feature_adapter(features)
        return adapted_features

class VariationalEncoder(nn.Module):
    """突破2：变分自编码器替代传统条件熵"""
    def __init__(self, input_dim=128, latent_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # 编码器网络
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        
        # 均值和方差预测
        self.mu_layer = nn.Linear(128, latent_dim)
        self.logvar_layer = nn.Linear(128, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def compute_vae_loss(self, x, recon_x, mu, logvar, beta=1.0):
        """VAE损失：重构损失 + KL散度"""
        # 重构损失
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL散度
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 总VAE损失
        vae_loss = recon_loss + beta * kl_loss
        return vae_loss / x.size(0)  # 平均到batch size

class AdversarialPrivacyModule(nn.Module):
    """突破3：对抗训练增强隐私保护"""
    def __init__(self, feature_dim=128, num_classes=10):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # 隐私攻击器（判别器）
        self.privacy_attacker = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )
        
        # 特征混淆器（生成器）
        self.feature_obfuscator = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
            nn.Tanh()
        )
        
    def forward(self, features, labels):
        # 生成混淆特征
        obfuscated_features = features + 0.1 * self.feature_obfuscator(features)
        
        # 隐私攻击预测
        privacy_pred = self.privacy_attacker(obfuscated_features)
        
        # 对抗损失：攻击器尽量预测正确，混淆器尽量让攻击失败
        attack_loss = F.cross_entropy(privacy_pred, labels)
        
        return obfuscated_features, attack_loss

class DynamicLambdaController(nn.Module):
    """突破4：动态λ调节和多目标优化"""
    def __init__(self, initial_lambda=16.0):
        super().__init__()
        self.log_lambda = nn.Parameter(torch.log(torch.tensor(initial_lambda)))
        self.accuracy_history = []
        self.privacy_history = []
        
    def update_lambda(self, accuracy, privacy_loss, target_accuracy=0.85):
        """根据准确率和隐私损失动态调节λ"""
        self.accuracy_history.append(accuracy)
        self.privacy_history.append(privacy_loss.item())
        
        # 保持最近10个epoch的历史
        if len(self.accuracy_history) > 10:
            self.accuracy_history = self.accuracy_history[-10:]
            self.privacy_history = self.privacy_history[-10:]
        
        if len(self.accuracy_history) >= 3:
            recent_acc = np.mean(self.accuracy_history[-3:])
            recent_privacy = np.mean(self.privacy_history[-3:])
            
            # 自适应调节策略
            if recent_acc < target_accuracy:
                # 准确率不足，降低λ
                self.log_lambda.data *= 0.95
            elif recent_privacy < 0.01:
                # 隐私保护不足，增加λ
                self.log_lambda.data *= 1.05
                
        # 限制λ范围
        self.log_lambda.data = torch.clamp(self.log_lambda.data, 
                                         torch.log(torch.tensor(1.0)), 
                                         torch.log(torch.tensor(100.0)))
        
    def get_lambda(self):
        return torch.exp(self.log_lambda)

class KnowledgeDistillationModule(nn.Module):
    """突破5：知识蒸馏增强特征学习"""
    def __init__(self, student_dim=128, teacher_dim=512, temperature=4.0):
        super().__init__()
        self.temperature = temperature
        
        # 教师网络（预训练大模型）
        teacher_backbone = models.resnet50(pretrained=True)
        self.teacher = nn.Sequential(*list(teacher_backbone.children())[:-1])
        
        # 冻结教师网络
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        # 特征对齐网络
        self.feature_alignment = nn.Sequential(
            nn.Linear(teacher_dim, 256),
            nn.ReLU(),
            nn.Linear(256, student_dim)
        )
        
    def forward(self, student_features, input_images):
        """计算知识蒸馏损失"""
        with torch.no_grad():
            teacher_features = self.teacher(input_images)
            teacher_features = teacher_features.view(teacher_features.size(0), -1)
            
        # 对齐教师特征
        aligned_teacher_features = self.feature_alignment(teacher_features)
        
        # 知识蒸馏损失
        kd_loss = F.mse_loss(student_features, aligned_teacher_features.detach())
        
        return kd_loss, aligned_teacher_features

class UltimateConditionalEntropyCalculator(nn.Module):
    """终极条件熵计算器：集成所有突破性技术"""
    def __init__(self, feature_dim=128, num_classes=10):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # 集成所有模块
        self.vae = VariationalEncoder(feature_dim, latent_dim=32)
        self.adversarial = AdversarialPrivacyModule(feature_dim, num_classes)
        self.lambda_controller = DynamicLambdaController(initial_lambda=16.0)
        self.kd_module = KnowledgeDistillationModule(feature_dim, 2048)
        
        # 多尺度特征融合
        self.multi_scale_fusion = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Linear(feature_dim, feature_dim * 2),
            nn.Identity()
        ])
        
        self.fusion_gate = nn.Sequential(
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Sigmoid()
        )
        
    def forward(self, features, labels, input_images, epoch, accuracy):
        """终极条件熵计算"""
        
        # 1. 多尺度特征处理
        multi_features = []
        for fusion_layer in self.multi_scale_fusion:
            if isinstance(fusion_layer, nn.Identity):
                multi_features.append(features)
            else:
                scaled_feat = fusion_layer(features)
                if scaled_feat.size(1) != features.size(1):
                    # 调整维度匹配
                    if scaled_feat.size(1) < features.size(1):
                        scaled_feat = F.pad(scaled_feat, (0, features.size(1) - scaled_feat.size(1)))
                    else:
                        scaled_feat = scaled_feat[:, :features.size(1)]
                multi_features.append(scaled_feat)
        
        fused_features = torch.cat(multi_features, dim=1)
        gate_weights = self.fusion_gate(fused_features)
        enhanced_features = features * gate_weights
        
        # 2. VAE处理
        recon_features, mu, logvar = self.vae(enhanced_features)
        vae_loss = self.vae.compute_vae_loss(enhanced_features, recon_features, mu, logvar)
        
        # 3. 对抗训练
        obfuscated_features, adversarial_loss = self.adversarial(enhanced_features, labels)
        
        # 4. 知识蒸馏
        kd_loss, teacher_features = self.kd_module(enhanced_features, input_images)
        
        # 5. 动态λ调节
        self.lambda_controller.update_lambda(accuracy, vae_loss)
        current_lambda = self.lambda_controller.get_lambda()
        
        # 6. 多目标损失融合
        total_privacy_loss = (
            0.4 * vae_loss +           # VAE重构损失
            0.3 * adversarial_loss +   # 对抗损失  
            0.2 * kd_loss +           # 知识蒸馏损失
            0.1 * torch.mean(torch.var(obfuscated_features, dim=0))  # 传统条件熵
        )
        
        # 返回增强特征和损失
        return total_privacy_loss, current_lambda, {
            'vae_loss': vae_loss,
            'adversarial_loss': adversarial_loss, 
            'kd_loss': kd_loss,
            'current_lambda': current_lambda
        }

class UltimateEnhancedCEM(nn.Module):
    """终极增强CEM架构：集成所有突破性改进"""
    def __init__(self, num_classes=10, feature_dim=128):
        super().__init__()
        
        # 突破1：预训练特征提取器
        self.feature_extractor = PretrainedFeatureExtractor(feature_dim)
        
        # 突破2-5：终极条件熵计算器
        self.ultimate_calculator = UltimateConditionalEntropyCalculator(feature_dim, num_classes)
        
        # 增强分类器
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x, labels, epoch, accuracy):
        # 提取特征
        features = self.feature_extractor(x)
        
        # 分类预测
        logits = self.classifier(features)
        
        # 计算终极条件熵损失
        privacy_loss, current_lambda, loss_details = self.ultimate_calculator(
            features, labels, x, epoch, accuracy
        )
        
        return logits, privacy_loss, current_lambda, loss_details
