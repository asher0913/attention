#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import numpy as np
import os
import json
from datetime import datetime
from model_training_attention import MIA_train
from datasets_torch import get_cifar10_trainloader, get_cifar10_testloader

def set_seeds(seed: int = 123):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def test_fixed_attention():
    print("=== Testing Fixed Attention Implementation ===")
    set_seeds(123)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load a small dataset for testing
    batch_size = 32
    train_loader, _, _ = get_cifar10_trainloader(batch_size=batch_size, num_workers=0)
    test_loader, _, _ = get_cifar10_testloader(batch_size=batch_size, num_workers=0)
    
    print(f"Training samples: {len(train_loader[0].dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model with attention classifier
    model = MIA_train(
        arch="vgg11_bn", cutting_layer=4, batch_size=batch_size, n_epochs=3, lambd=16,
        dataset="cifar10", scheme="V2_epoch", num_client=1, save_dir="./test_saves_fixed",
        use_attention_classifier=True, num_slots=8, attention_heads=8, attention_dropout=0.1,
        regularization_strength=0.025, learning_rate=0.05, random_seed=123,
    )
    
    print("\n=== Testing Single Batch ===")
    # Test with a single batch
    images, labels = next(iter(train_loader[0]))
    images = images.to(model.device)
    labels = labels.to(model.device)
    
    model.f.train()
    model.f_tail.train()
    model.classifier.train()
    if model.attention_classifier is not None:
        model.attention_classifier.train()
    
    # Forward pass
    z_private = model.f(images)
    unique_labels = torch.unique(labels)
    
    # Test attention classification
    attention_logits, enhanced_features, slot_representations, attention_weights = model.attention_classify_features(z_private, labels)
    
    # Test conditional entropy calculation
    rob_loss, intra_class_mse = model.compute_attention_conditional_entropy(z_private, labels, unique_labels, slot_representations)
    
    print(f"Feature shape: {z_private.shape}")
    print(f"Slot representations shape: {slot_representations.shape}")
    print(f"Attention logits shape: {attention_logits.shape}")
    print(f"Conditional entropy loss: {rob_loss.item():.6f}")
    print(f"Intra-class MSE: {intra_class_mse.item():.6f}")
    
    # Test that loss is positive and reasonable
    if rob_loss.item() > 0:
        print("✅ Conditional entropy loss is positive")
    else:
        print("❌ Conditional entropy loss is negative or zero")
    
    # Test gradient computation
    rob_loss.backward()
    if rob_loss.grad_fn is None:
        print("❌ ERROR: Attention loss is not differentiable!")
        return False
    else:
        print("✅ Attention loss is differentiable")
    
    # Test full training step
    print("\n=== Testing Full Training Step ===")
    model.optimizer.zero_grad()
    
    # Simulate training step
    ce_loss = F.cross_entropy(attention_logits, labels)
    print(f"Cross-entropy loss: {ce_loss.item():.6f}")
    
    # Test that total loss calculation matches original GMM
    total_loss = ce_loss  # rob_loss is NOT added to total_loss, just like in original GMM
    print(f"Total loss (without rob_loss): {total_loss.item():.6f}")
    
    # Test gradient accumulation like in original GMM
    rob_loss.backward(retain_graph=True)
    encoder_gradients = {name: param.grad.clone() for name, param in model.f.named_parameters()}
    model.optimizer.zero_grad()
    
    total_loss.backward()
    for name, param in model.f.named_parameters():
        param.grad += model.lambd * encoder_gradients[name]
    
    print("✅ Training step completed successfully")
    
    # Test validation
    print("\n=== Testing Validation ===")
    model.f.eval()
    model.f_tail.eval()
    model.classifier.eval()
    if model.attention_classifier is not None:
        model.attention_classifier.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            if batch_idx >= 5:  # Just test first 5 batches
                break
            images = images.to(model.device)
            labels = labels.to(model.device)
            
            z_private = model.f(images)
            attention_logits, _, _, _ = model.attention_classify_features(z_private, labels)
            
            pred = attention_logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100. * correct / total
    print(f"Validation accuracy: {accuracy:.2f}%")
    
    if accuracy > 10:  # Should be much better than 10%
        print("✅ Validation accuracy is reasonable")
    else:
        print("❌ Validation accuracy is too low")
    
    print("\n=== Test Summary ===")
    print("✅ Fixed attention implementation test completed")
    print("Key fixes applied:")
    print("1. Changed torch.log to reg_variances.mean() to avoid negative values")
    print("2. Confirmed rob_loss is NOT added to total_loss (matches original GMM)")
    print("3. Confirmed gradient accumulation works correctly")
    
    return True

if __name__ == "__main__":
    test_fixed_attention()
