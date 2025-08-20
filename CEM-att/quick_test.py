#!/usr/bin/env python3

"""
Quick test for CEM-att - just verify imports and basic functionality
NO FULL TRAINING - just check for bugs
"""

import torch
import numpy as np

def quick_test():
    print("🧪 CEM-att Quick Test (No Training)")
    print("=" * 50)
    
    # Test imports
    try:
        from attention_modules import FeatureClassificationModule
        from model_training import MIA_train
        print("✅ Imports successful")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test device detection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Device detected: {device}")
    
    # Test attention module creation
    try:
        attention_clf = FeatureClassificationModule(
            feature_dim=128,
            num_slots=8, 
            num_classes=10,
            num_heads=8,
            dropout=0.1
        ).to(device)
        print("✅ Attention classifier created")
        
        # Test forward pass
        dummy_input = torch.randn(2, 128, 8, 8).to(device)
        logits, enhanced_features, slot_representations, attention_weights = attention_clf(dummy_input)
        print(f"✅ Forward pass works: logits {logits.shape}")
        
    except Exception as e:
        print(f"❌ Attention module error: {e}")
        return False
    
    # Test MIA_train creation (minimal test)
    try:
        model = MIA_train(
            arch="vgg11_bn",
            cutting_layer=4,
            batch_size=8,  # Small batch
            n_epochs=1,    # Single epoch
            dataset="cifar10",
            save_dir="./test_temp",
            use_attention_classifier=True,
            num_slots=8,
            attention_heads=8,
        )
        print("✅ MIA_train with attention created")
        
    except Exception as e:
        print(f"❌ MIA_train error: {e}")
        return False
    
    print("\n🎉 All quick tests passed!")
    print("✅ CEM-att is ready for Linux NVIDIA deployment")
    return True

if __name__ == "__main__":
    success = quick_test()
    
    if success:
        print("\n🚀 CEM-att Setup Complete!")
        print("📋 Usage on Linux server:")
        print("python main_MIA.py --filename your_exp --use_attention_classifier --lambd 1.0")
        print("\n📁 CEM-att folder ready for deployment!")
    else:
        print("\n💥 Issues found - need fixing before deployment")
