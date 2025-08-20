#!/usr/bin/env python3

"""
Test script for CEM-att: CEM with Attention classifier
This script tests the attention integration in the CEM framework
"""

import torch
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append('.')

def test_attention_cem():
    """Test the attention CEM implementation"""
    
    print("🧪 Testing CEM-att: CEM with Attention classifier")
    print("=" * 60)
    
    try:
        # Test imports
        print("✅ Testing imports...")
        from attention_modules import FeatureClassificationModule
        from model_training import MIA_train
        from datasets_torch import get_cifar10_trainloader, get_cifar10_testloader
        print("✅ All imports successful")
        
        # Test model creation with attention
        print("✅ Testing model creation with attention...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model = MIA_train(
            arch="vgg11_bn",
            cutting_layer=4,
            batch_size=32,
            n_epochs=1,
            lambd=1.0,
            dataset="cifar10",
            save_dir="./test_saves",
            use_attention_classifier=True,
            num_slots=8,
            attention_heads=8,
            attention_dropout=0.1,
            var_threshold=0.1
        )
        print("✅ Model created successfully with attention classifier")
        
        # Test attention classifier
        print("✅ Testing attention classifier...")
        if model.attention_classifier is not None:
            print("✅ Attention classifier initialized")
            
            # Create dummy input
            dummy_input = torch.randn(4, 128, 8, 8).to(device)
            
            # Test forward pass
            logits, enhanced_features, slot_representations, attention_weights = model.attention_classifier(dummy_input)
            
            print(f"✅ Attention classifier forward pass successful")
            print(f"   Logits shape: {logits.shape}")
            print(f"   Enhanced features shape: {enhanced_features.shape}")
            print(f"   Slot representations shape: {slot_representations.shape}")
            
            # Test attention conditional entropy
            dummy_labels = torch.randint(0, 10, (4,)).to(device)
            unique_labels = torch.unique(dummy_labels)
            
            dummy_features = torch.randn(4, 128, 8, 8).to(device)
            rob_loss, intra_class_mse = model.compute_attention_conditional_entropy(
                dummy_features, dummy_labels, unique_labels, slot_representations
            )
            
            print(f"✅ Attention conditional entropy computed successfully")
            print(f"   Rob loss: {rob_loss.item():.6f}")
            print(f"   Intra-class MSE: {intra_class_mse.item():.6f}")
            
        else:
            print("❌ ERROR: Attention classifier not initialized!")
            return False
            
        # Test data loading
        print("✅ Testing data loading...")
        train_loader = get_cifar10_trainloader(batch_size=32, num_workers=1)
        test_loader = get_cifar10_testloader(batch_size=32, num_workers=1)
        print("✅ Data loaders created successfully")
        
        # Test one training step
        print("✅ Testing one training step...")
        data_iter = iter(train_loader[0])  # train_loader is a list
        images, labels = next(data_iter)
        
        # Prepare dummy centroids (required for original CEM framework)
        centroids_list = {}
        for i in range(10):  # CIFAR-10 has 10 classes
            centroids_list[i] = torch.randn(3, 8192).to(device)  # 3 centroids per class, flattened VGG features
        
        # Test training step
        model.train_target_step(
            x_private=images,
            label_private=labels,
            adding_noise=False,
            random_ini_centers=False,
            centroids_list=centroids_list,
            client_id=0
        )
        print("✅ Training step completed successfully")
        
        print("🎉 All tests passed! CEM-att is ready for training.")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_attention_cem()
    if success:
        print("\n🚀 CEM-att implementation is working correctly!")
        print("You can now train with attention by using:")
        print("python main_MIA.py --filename test_attention --use_attention_classifier --lambd 1.0")
    else:
        print("\n💥 CEM-att implementation has issues. Please check the errors above.")
    
    sys.exit(0 if success else 1)
