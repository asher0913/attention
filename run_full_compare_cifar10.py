#!/usr/bin/env python3
"""
Run a full training and evaluation comparison between Attention classifier and GMM-based
classifier on CIFAR-10. This script implements a complete pipeline for comparing the two
feature classification methods.

Features:
- Complete training pipeline for both GMM and Attention methods
- Proper feature extraction using VGG11_bn backbone
- Comprehensive evaluation on test set
- Automatic CUDA support
- Detailed results output

Usage:
  python run_full_compare_cifar10.py

Expected runtime: ~10-20 minutes on CUDA GPU
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import os
import json
from datetime import datetime

from model_training_attention import MIA_train
from datasets_torch import get_cifar10_trainloader, get_cifar10_testloader
from GMM import fit_gmm_torch


def set_seeds(seed: int = 123):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def copy_shared_weights(src: MIA_train, dst: MIA_train):
    """Copy shared weights between models for fair comparison"""
    dst.f.load_state_dict(src.f.state_dict())
    dst.f_tail.load_state_dict(src.f_tail.state_dict())
    dst.classifier.load_state_dict(src.classifier.state_dict())


@torch.no_grad()
def forward_logits(model: MIA_train, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Forward pass to get logits from model"""
    device = model.device
    images = images.to(device)
    labels = labels.to(device)

    z_private = model.f(images)

    if getattr(model, 'use_attention_classifier', False) and model.attention_classifier is not None:
        logits, _, _, _ = model.attention_classify_features(z_private, labels)
        return logits
    else:
        out = model.f_tail(z_private)
        arch = model.arch if isinstance(model.arch, str) else str(model.arch)
        if "mobilenetv2" in arch:
            out = torch.nn.functional.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            logits = model.classifier(out)
        elif arch == "resnet20" or arch == "resnet32":
            out = torch.nn.functional.avg_pool2d(out, 8)
            out = out.view(out.size(0), -1)
            logits = model.classifier(out)
        else:
            out = out.view(out.size(0), -1)
            logits = model.classifier(out)
        return logits


def train_model(model: MIA_train, n_epochs: int, train_loader, test_loader, method_name: str):
    """Complete training function for a model"""
    print(f"\n=== Training {method_name} Model ===")
    
    # Set training parameters
    model.n_epochs = int(n_epochs)
    model.pub_dataloader = test_loader  # Set test loader for evaluation
    
    # Initialize GMM components if needed
    if not getattr(model, 'use_attention_classifier', False):
        print("Initializing GMM components...")
        model.centroids_list = []
        model.random_ini_centers = True
        
        # Get feature dimensions
        sample_batch = next(iter(train_loader))
        sample_images = sample_batch[0].to(model.device)
        sample_features = model.f(sample_images)
        feature_dim = sample_features.size(1) * sample_features.size(2) * sample_features.size(3)
        
        # Initialize centroids for each class
        for cls in range(model.num_class):
            centroids = torch.randn(3, feature_dim).to(model.device)  # 3 components per class
            model.centroids_list.append(centroids)
    
    # Training loop
    print(f"Starting training for {n_epochs} epochs...")
    t0 = time.time()
    
    for epoch in range(n_epochs):
        model.f.train()
        model.f_tail.train()
        model.classifier.train()
        if getattr(model, 'attention_classifier', None) is not None:
            model.attention_classifier.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(model.device)
            labels = labels.to(model.device)
            
            # Forward pass
            if getattr(model, 'use_attention_classifier', False):
                # Attention method
                z_private = model.f(images)
                logits, _, _, _ = model.attention_classify_features(z_private, labels)
                loss = F.cross_entropy(logits, labels)
            else:
                # GMM method
                z_private = model.f(images)
                out = model.f_tail(z_private)
                out = out.view(out.size(0), -1)
                logits = model.classifier(out)
                loss = F.cross_entropy(logits, labels)
            
            # Backward pass
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%")
        
        # Epoch summary
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{n_epochs} completed - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
    
    t1 = time.time()
    training_time = t1 - t0
    print(f"{method_name} training completed in {training_time:.1f} seconds")
    
    return training_time


@torch.no_grad()
def evaluate_model(model: MIA_train, test_loader, method_name: str) -> tuple[float, float]:
    """Evaluate model on test set"""
    print(f"\n=== Evaluating {method_name} Model ===")
    
    device = model.device
    model.f.eval()
    model.f_tail.eval()
    model.classifier.eval()
    if getattr(model, 'attention_classifier', None) is not None:
        model.attention_classifier.eval()

    crit = nn.CrossEntropyLoss().to(device)
    
    ce_sum = 0.0
    n_samples = 0
    correct = 0
    
    print("Running evaluation on test set...")
    for batch_idx, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        logits = forward_logits(model, images, labels)
        loss = crit(logits, labels)
        
        ce_sum += float(loss.item()) * images.size(0)
        n_samples += int(images.size(0))
        pred = logits.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        
        if batch_idx % 50 == 0:
            print(f"Evaluation batch {batch_idx}/{len(test_loader)}")

    avg_ce = ce_sum / max(1, n_samples)
    top1 = correct / max(1, n_samples)
    
    print(f"{method_name} Evaluation Results:")
    print(f"  Cross Entropy Loss: {avg_ce:.4f}")
    print(f"  Top-1 Accuracy: {top1*100:.2f}%")
    
    return avg_ce, top1


def build_model(use_attention: bool, batch_size: int, save_dir_suffix: str) -> MIA_train:
    """Build model with specified configuration"""
    model = MIA_train(
        arch="vgg11_bn",
        cutting_layer=4,
        batch_size=batch_size,
        n_epochs=1,  # Will be set during training
        lambd=0 if use_attention else 1,  # Disable robustness for attention
        dataset="cifar10",
        scheme="V2_epoch",
        num_client=1,
        save_dir=f"./saves/full_compare_{save_dir_suffix}",
        use_attention_classifier=use_attention,
        num_slots=8,
        attention_heads=8,
        attention_dropout=0.1,
        bottleneck_option="None",
        regularization_option="None",
        AT_regularization_option="None",
        regularization_strength=0.0,
        AT_regularization_strength=0.0,
        learning_rate=0.01,
        local_lr=-1,
        random_seed=123,
    )
    return model


def main():
    """Main function to run the comparison experiment"""
    print("=== CIFAR-10 GMM vs Attention Comparison Experiment ===")
    
    # Set random seeds
    set_seeds(123)
    
    # Configuration
    epochs = 10
    batch_size = 128
    
    # Check device availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load datasets
    print("\nLoading CIFAR-10 datasets...")
    try:
        train_loader, _, _ = get_cifar10_trainloader(batch_size=batch_size, num_workers=2)
        test_loader, _, _ = get_cifar10_testloader(batch_size=batch_size, num_workers=2)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("Attempting to download CIFAR-10 dataset manually...")
        
        # Manual download attempt
        import urllib.request
        import tarfile
        
        cifar_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        data_dir = "./data"
        os.makedirs(data_dir, exist_ok=True)
        
        print("Downloading CIFAR-10 dataset...")
        urllib.request.urlretrieve(cifar_url, os.path.join(data_dir, "cifar-10-python.tar.gz"))
        
        print("Extracting dataset...")
        with tarfile.open(os.path.join(data_dir, "cifar-10-python.tar.gz"), 'r:gz') as tar:
            tar.extractall(data_dir)
        
        print("Retrying dataset loading...")
        train_loader, _, _ = get_cifar10_trainloader(batch_size=batch_size, num_workers=2)
        test_loader, _, _ = get_cifar10_testloader(batch_size=batch_size, num_workers=2)
    
    print(f"Training samples: {len(train_loader[0].dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Build models
    print("\nBuilding models...")
    attn_model = build_model(True, batch_size, "attention")
    gmm_model = build_model(False, batch_size, "gmm")
    
    # Align initial weights for fair comparison
    print("Aligning initial weights...")
    copy_shared_weights(attn_model, gmm_model)
    
    # Training and evaluation results
    results = {
        'attention': {},
        'gmm': {},
        'comparison': {}
    }
    
    # Train and evaluate Attention model
    attn_train_time = train_model(attn_model, epochs, train_loader[0], test_loader, "Attention")
    attn_ce, attn_top1 = evaluate_model(attn_model, test_loader, "Attention")
    
    results['attention'] = {
        'training_time': attn_train_time,
        'cross_entropy': attn_ce,
        'top1_accuracy': attn_top1
    }
    
    # Train and evaluate GMM model
    gmm_train_time = train_model(gmm_model, epochs, train_loader[0], test_loader, "GMM")
    gmm_ce, gmm_top1 = evaluate_model(gmm_model, test_loader, "GMM")
    
    results['gmm'] = {
        'training_time': gmm_train_time,
        'cross_entropy': gmm_ce,
        'top1_accuracy': gmm_top1
    }
    
    # Comparison analysis
    accuracy_diff = attn_top1 - gmm_top1
    ce_diff = gmm_ce - attn_ce  # Lower CE is better
    time_diff = attn_train_time - gmm_train_time
    
    results['comparison'] = {
        'accuracy_difference': accuracy_diff,
        'ce_difference': ce_diff,
        'time_difference': time_diff,
        'attention_better_accuracy': accuracy_diff > 0,
        'attention_better_ce': ce_diff > 0,
        'attention_faster': time_diff < 0
    }
    
    # Print final results
    print("\n" + "="*60)
    print("FINAL COMPARISON RESULTS")
    print("="*60)
    print(f"{'Metric':<20} {'Attention':<15} {'GMM':<15} {'Difference':<15}")
    print("-"*60)
    print(f"{'Training Time (s)':<20} {attn_train_time:<15.1f} {gmm_train_time:<15.1f} {time_diff:<15.1f}")
    print(f"{'Cross Entropy':<20} {attn_ce:<15.4f} {gmm_ce:<15.4f} {ce_diff:<15.4f}")
    print(f"{'Top-1 Accuracy (%)':<20} {attn_top1*100:<15.2f} {gmm_top1*100:<15.2f} {accuracy_diff*100:<15.2f}")
    print("-"*60)
    
    # Winner analysis
    print("\nWINNER ANALYSIS:")
    if accuracy_diff > 0:
        print(f"✓ Attention method achieves {accuracy_diff*100:.2f}% higher accuracy")
    else:
        print(f"✓ GMM method achieves {abs(accuracy_diff)*100:.2f}% higher accuracy")
    
    if ce_diff > 0:
        print(f"✓ Attention method has {ce_diff:.4f} lower cross-entropy loss")
    else:
        print(f"✓ GMM method has {abs(ce_diff):.4f} lower cross-entropy loss")
    
    if time_diff < 0:
        print(f"✓ Attention method is {abs(time_diff):.1f}s faster")
    else:
        print(f"✓ GMM method is {time_diff:.1f}s faster")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"comparison_results_{timestamp}.json"
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Recursively convert numpy types
    def convert_dict(d):
        for key, value in d.items():
            if isinstance(value, dict):
                convert_dict(value)
            else:
                d[key] = convert_numpy(value)
    
    convert_dict(results)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()
