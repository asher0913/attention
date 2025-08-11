#!/usr/bin/env python3
"""
Run training with attention mechanism and conditional entropy loss
This script demonstrates the complete pipeline using attention instead of GMM
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from datetime import datetime

from model_training_attention import MIA_train
from datasets_torch import get_cifar10_trainloader, get_cifar10_testloader


def set_seeds(seed: int = 123):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_attention_with_conditional_entropy():
    """Train model using attention mechanism with conditional entropy loss"""
    
    print("=== Training with Attention + Conditional Entropy Loss ===")
    
    # Set random seeds
    set_seeds(123)
    
    # Configuration
    epochs = 5  # Reduced for testing
    batch_size = 64
    
    # Check device availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load datasets
    print("\nLoading CIFAR-10 datasets...")
    train_loader, _, _ = get_cifar10_trainloader(batch_size=batch_size, num_workers=2)
    test_loader, _, _ = get_cifar10_testloader(batch_size=batch_size, num_workers=2)
    
    print(f"Training samples: {len(train_loader[0].dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Build attention model with conditional entropy loss enabled
    print("\nBuilding attention model with conditional entropy loss...")
    model = MIA_train(
        arch="vgg11_bn",
        cutting_layer=4,
        batch_size=batch_size,
        n_epochs=epochs,
        lambd=1,  # Enable conditional entropy loss
        dataset="cifar10",
        scheme="V2_epoch",
        num_client=1,
        save_dir="./saves/attention_conditional_entropy",
        use_attention_classifier=True,
        num_slots=8,
        attention_heads=8,
        attention_dropout=0.1,
        bottleneck_option="None",
        regularization_option="None",
        AT_regularization_option="None",
        regularization_strength=0.1,  # Set regularization strength for conditional entropy
        AT_regularization_strength=0.0,
        learning_rate=0.01,
        local_lr=-1,
        random_seed=123,
    )
    
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    t0 = time.time()
    
    for epoch in range(epochs):
        model.f.train()
        model.f_tail.train()
        model.classifier.train()
        if model.attention_classifier is not None:
            model.attention_classifier.train()
        
        total_loss = 0.0
        total_ce_loss = 0.0
        total_conditional_entropy_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader[0]):
            images = images.to(model.device)
            labels = labels.to(model.device)
            
            # Forward pass with conditional entropy loss
            z_private = model.f(images)
            unique_labels = torch.unique(labels)
            
            # Get attention outputs
            attention_logits, enhanced_features, slot_representations, attention_weights = model.attention_classify_features(z_private, labels)
            
            # Compute conditional entropy loss
            rob_loss, intra_class_mse = model.compute_attention_conditional_entropy(
                z_private, labels, unique_labels, slot_representations
            )
            
            # Classification loss
            ce_loss = F.cross_entropy(attention_logits, labels)
            
            # Total loss (classification + conditional entropy)
            loss = ce_loss + model.lambd * rob_loss
            
            # Backward pass
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_conditional_entropy_loss += rob_loss.item()
            pred = attention_logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader[0])}, "
                      f"Total Loss: {loss.item():.4f}, CE: {ce_loss.item():.4f}, "
                      f"Cond Entropy: {rob_loss.item():.4f}, Acc: {100.*correct/total:.2f}%")
        
        # Epoch summary
        epoch_loss = total_loss / len(train_loader[0])
        epoch_ce_loss = total_ce_loss / len(train_loader[0])
        epoch_conditional_entropy_loss = total_conditional_entropy_loss / len(train_loader[0])
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs} completed - "
              f"Total Loss: {epoch_loss:.4f}, CE: {epoch_ce_loss:.4f}, "
              f"Cond Entropy: {epoch_conditional_entropy_loss:.4f}, Acc: {epoch_acc:.2f}%")
    
    t1 = time.time()
    training_time = t1 - t0
    print(f"Training completed in {training_time:.1f} seconds")
    
    # Evaluation
    print("\n=== Evaluating trained model ===")
    model.f.eval()
    model.attention_classifier.eval()
    
    test_correct = 0
    test_total = 0
    test_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(model.device)
            labels = labels.to(model.device)
            
            z_private = model.f(images)
            attention_logits, _, _, _ = model.attention_classify_features(z_private, labels)
            
            loss = F.cross_entropy(attention_logits, labels)
            test_loss += loss.item()
            
            pred = attention_logits.argmax(dim=1)
            test_correct += (pred == labels).sum().item()
            test_total += labels.size(0)
            
            if batch_idx % 20 == 0:
                print(f"Evaluation batch {batch_idx}/{len(test_loader)}")
    
    test_accuracy = 100. * test_correct / test_total
    avg_test_loss = test_loss / len(test_loader)
    
    print(f"\n=== Final Results ===")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Training Time: {training_time:.1f} seconds")
    
    # Save results
    results = {
        'test_accuracy': test_accuracy,
        'test_loss': avg_test_loss,
        'training_time': training_time,
        'epochs': epochs,
        'batch_size': batch_size,
        'lambd': model.lambd,
        'regularization_strength': model.regularization_strength,
        'num_slots': model.num_slots,
        'attention_heads': model.attention_heads,
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"attention_conditional_entropy_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print("Training with attention + conditional entropy loss completed successfully!")


if __name__ == "__main__":
    train_attention_with_conditional_entropy()
