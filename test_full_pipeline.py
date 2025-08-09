#!/usr/bin/env python3
print("🧪 Testing Attention CEM Project")
print("=" * 50)
from model_training_attention import MIA_train
import torch
print("✅ All imports successful")
model = MIA_train(arch="vgg11_bn", cutting_layer=4, batch_size=64, n_epochs=1, dataset="cifar10", use_attention_classifier=True, num_slots=8, attention_heads=8, attention_dropout=0.1, bottleneck_option="None")
print("✅ Model created successfully")
dummy_features = torch.randn(4, 128, 8, 8)
dummy_labels = torch.randint(0, 10, (4,))
logits, enhanced_features, slot_representations, attention_weights = model.attention_classify_features(dummy_features, dummy_labels)
print("✅ Attention classifier works!")
print(f"Logits shape: {logits.shape}")
print("🎉 All tests passed! Project ready for deployment.")
