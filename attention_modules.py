import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Scaled dot-product attention"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(output)
        
        return output, attention_weights


class SlotAttention(nn.Module):
    """Slot Attention module for learning object-centric representations"""
    
    def __init__(self, num_slots, d_model, num_heads=8, num_iterations=3, mlp_hidden_size=128, dropout=0.1):
        super(SlotAttention, self).__init__()
        
        self.num_slots = num_slots
        self.d_model = d_model
        self.num_iterations = num_iterations
        
        # Slot initialization
        self.slots_mu = nn.Parameter(torch.randn(1, 1, d_model))
        self.slots_log_sigma = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Attention mechanism
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # GRU for slot updates
        self.gru = nn.GRUCell(d_model, d_model)
        
        # MLP for slot refinement
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, d_model)
        )
        
        # Layer normalization
        self.norm_inputs = nn.LayerNorm(d_model)
        self.norm_slots = nn.LayerNorm(d_model)
        self.norm_mlp = nn.LayerNorm(d_model)
        
    def forward(self, inputs, mask=None):
        """
        Args:
            inputs: (batch_size, num_inputs, d_model)
            mask: (batch_size, num_inputs) - optional mask for inputs
        Returns:
            slots: (batch_size, num_slots, d_model)
        """
        batch_size, num_inputs, d_model = inputs.shape
        
        # Initialize slots
        slots = self.slots_mu.expand(batch_size, self.num_slots, -1) + \
                torch.exp(self.slots_log_sigma).expand(batch_size, self.num_slots, -1) * \
                torch.randn(batch_size, self.num_slots, d_model, device=inputs.device)
        
        # Normalize inputs
        inputs = self.norm_inputs(inputs)
        
        # Iterative slot attention
        for _ in range(self.num_iterations):
            # Normalize slots
            slots_prev = self.norm_slots(slots)
            
            # Attention from slots to inputs
            slots_attended, _ = self.attention(slots_prev, inputs, inputs, mask)
            
            # GRU update
            slots = self.gru(
                slots_attended.reshape(-1, d_model),
                slots.reshape(-1, d_model)
            ).reshape(batch_size, self.num_slots, d_model)
            
            # MLP refinement
            slots = slots + self.mlp(self.norm_mlp(slots))
        
        return slots


class CrossAttentionModule(nn.Module):
    """Cross Attention module using Slot Attention output as KV and original features as Q"""
    
    def __init__(self, d_model, num_slots, num_heads=8, mlp_hidden_size=128, dropout=0.1):
        super(CrossAttentionModule, self).__init__()
        
        self.d_model = d_model
        self.num_slots = num_slots
        
        # Slot Attention module
        self.slot_attention = SlotAttention(num_slots, d_model, num_heads, dropout=dropout)
        
        # Cross Attention
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, features, mask=None):
        """
        Args:
            features: (batch_size, channels, height, width) or (batch_size, num_features, d_model) - original features
            mask: (batch_size, num_features) - optional mask for features
        Returns:
            enhanced_features: (batch_size, num_features, d_model) - enhanced features
            slot_representations: (batch_size, num_slots, d_model) - slot representations
        """
        # Handle different input shapes
        if len(features.shape) == 4:
            # Input is (batch_size, channels, height, width)
            batch_size, channels, height, width = features.shape
            features = features.view(batch_size, height * width, channels)
        elif len(features.shape) == 3:
            # Input is already (batch_size, num_features, d_model)
            pass
        else:
            raise ValueError(f"Expected 3D or 4D input, got {len(features.shape)}D")
        
        # Apply Slot Attention to get slot representations
        slot_representations = self.slot_attention(features, mask)
        
        # Cross Attention: Q=features, K=V=slot_representations
        enhanced_features, attention_weights = self.cross_attention(
            features, slot_representations, slot_representations, mask
        )
        
        # Residual connection and normalization
        enhanced_features = self.norm1(features + enhanced_features)
        
        # Output projection
        enhanced_features = self.output_projection(enhanced_features)
        enhanced_features = self.norm2(enhanced_features)
        
        return enhanced_features, slot_representations, attention_weights


class FeatureClassificationModule(nn.Module):
    """Module for feature classification using Slot Attention + Cross Attention"""
    
    def __init__(self, feature_dim, num_slots, num_classes, num_heads=8, 
                 mlp_hidden_size=128, dropout=0.1, use_slot_classification=True):
        super(FeatureClassificationModule, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_slots = num_slots
        self.num_classes = num_classes
        self.use_slot_classification = use_slot_classification
        
        # Cross Attention module
        self.cross_attention = CrossAttentionModule(
            feature_dim, num_slots, num_heads, mlp_hidden_size, dropout
        )
        
        # Classification head
        if use_slot_classification:
            # Classify based on slot representations
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, mlp_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden_size, num_classes)
            )
        else:
            # Classify based on enhanced features
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, mlp_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden_size, num_classes)
            )
        
        # Slot to class mapping (optional)
        self.slot_to_class = nn.Linear(feature_dim, num_classes)
        
    def forward(self, features, mask=None):
        """
        Args:
            features: (batch_size, channels, height, width) or (batch_size, num_features, feature_dim)
            mask: (batch_size, num_features) - optional mask
        Returns:
            logits: (batch_size, num_classes)
            enhanced_features: (batch_size, num_features, feature_dim)
            slot_representations: (batch_size, num_slots, feature_dim)
            attention_weights: attention weights from cross attention
        """
        # Apply Cross Attention (handles both 3D and 4D inputs)
        enhanced_features, slot_representations, attention_weights = self.cross_attention(features, mask)
        
        if self.use_slot_classification:
            # Use slot representations for classification
            # Average across slots
            slot_avg = slot_representations.mean(dim=1)  # (batch_size, feature_dim)
            logits = self.classifier(slot_avg)
        else:
            # Use enhanced features for classification
            # Average across features
            feature_avg = enhanced_features.mean(dim=1)  # (batch_size, feature_dim)
            logits = self.classifier(feature_avg)
        
        return logits, enhanced_features, slot_representations, attention_weights
    
    def get_slot_class_predictions(self, slot_representations):
        """Get class predictions for each slot"""
        return self.slot_to_class(slot_representations)  # (batch_size, num_slots, num_classes)


def create_attention_classifier(feature_dim, num_slots, num_classes, **kwargs):
    """Factory function to create attention-based classifier"""
    return FeatureClassificationModule(feature_dim, num_slots, num_classes, **kwargs)
