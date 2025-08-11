#!/usr/bin/env python3
"""
Test script to verify that attention conditional entropy loss is working correctly
"""

import torch
import numpy as np
from model_training_attention import MIA_train
from datasets_torch import get_cifar10_trainloader

def test_attention_conditional_entropy():
    print("=== Testing Attention Conditional Entropy Loss ===")
    
    # Set random seeds
    torch.manual_seed(123)
    np.random.seed(123)
    
    # Configuration
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test dataset
    print("Loading test dataset...")
    train_loader, _, _ = get_cifar10_trainloader(batch_size=batch_size, num_workers=0)
    
    # Build attention model
    print("Building attention model...")
    attn_model = MIA_train(
        arch="vgg11_bn",
        cutting_layer=4,
        batch_size=batch_size,
        n_epochs=1,
        lambd=1,
        dataset="cifar10",
        scheme="V2_epoch",
        num_client=1,
        save_dir="./test_saves",
        use_attention_classifier=True,
        num_slots=8,
        attention_heads=8,
        attention_dropout=0.1,
        regularization_strength=0.1,
        learning_rate=0.01,
        random_seed=123,
    )
    
    # Test with a single batch
    print("Testing with a single batch...")
    batch = next(iter(train_loader[0]))
    images, labels = batch
    images = images.to(device)
    labels = labels.to(device)
    
    # Test attention method
    print("Testing attention method...")
    attn_model.f.train()
    attn_model.attention_classifier.train()
    
    z_private = attn_model.f(images)
    unique_labels = torch.unique(labels)
    
    # Get attention outputs
    attention_logits, enhanced_features, slot_representations, attention_weights = attn_model.attention_classify_features(z_private, labels)
    
    # Compute conditional entropy loss
    rob_loss_attn, intra_class_mse_attn = attn_model.compute_attention_conditional_entropy(
        z_private, labels, unique_labels, slot_representations
    )
    
    print(f"Attention conditional entropy loss: {rob_loss_attn.item():.6f}")
    print(f"Attention intra-class MSE: {intra_class_mse_attn.item():.6f}")
    
    # Test gradient computation
    print("Testing gradient computation...")
    rob_loss_attn.backward()
    
    # Check if gradients were computed
    has_gradients = False
    for param in attn_model.attention_classifier.parameters():
        if param.grad is not None:
            has_gradients = True
            break
    
    if has_gradients:
        print("✅ Attention loss is differentiable")
    else:
        print("❌ ERROR: Attention loss gradients not computed!")
        return False
    
    print("✅ Test completed successfully!")
    return True

if __name__ == "__main__":
    test_attention_conditional_entropy()
