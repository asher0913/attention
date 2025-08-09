#!/usr/bin/env python3
"""
Run a full training and evaluation comparison between Attention classifier and GMM-based
classifier on CIFAR-10 without any CLI arguments. Just run:

  python run_full_compare_cifar10.py

Defaults:
- Architecture: vgg11_bn with cutting_layer=4 (as used in the project)
- Epochs: 10 for each model
- Batch size: 128
- Full test-set evaluation and summary

On a Tesla P100, expect roughly 30–60s per epoch per model (varies by dataloader
throughput and exact environment). So ~10–20 minutes total for 10 epochs x 2 models.
"""

import time
import torch
import torch.nn as nn

from model_training_attention import MIA_train


def set_seeds(seed: int = 123):
    torch.manual_seed(seed)


def copy_shared_weights(src: MIA_train, dst: MIA_train):
    dst.f.load_state_dict(src.f.state_dict())
    dst.f_tail.load_state_dict(src.f_tail.state_dict())
    dst.classifier.load_state_dict(src.classifier.state_dict())


@torch.no_grad()
def forward_logits(model: MIA_train, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
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


def epoch_train(model: MIA_train, n_epochs: int) -> None:
    model.n_epochs = int(n_epochs)
    model(log_frequency=100, verbose=False, progress_bar=False)


@torch.no_grad()
def full_eval(model: MIA_train) -> tuple[float, float]:
    device = model.device
    model.f.eval(); model.f_tail.eval(); model.classifier.eval()
    if getattr(model, 'attention_classifier', None) is not None:
        model.attention_classifier.eval()

    crit = nn.CrossEntropyLoss().to(device)
    test_loader = model.pub_dataloader

    ce_sum = 0.0
    n_samples = 0
    correct = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = forward_logits(model, images, labels)
        loss = crit(logits, labels)
        ce_sum += float(loss.item()) * images.size(0)
        n_samples += int(images.size(0))
        pred = logits.argmax(dim=1)
        correct += int((pred == labels).sum().item())

    avg_ce = ce_sum / max(1, n_samples)
    top1 = correct / max(1, n_samples)
    return avg_ce, top1


def build_model(use_attention: bool, batch_size: int, save_dir_suffix: str) -> MIA_train:
    model = MIA_train(
        arch="vgg11_bn",
        cutting_layer=4,
        batch_size=batch_size,
        n_epochs=1,
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
    set_seeds(123)

    epochs = 10
    batch_size = 128

    print("== Full CIFAR-10 Comparison: Attention vs GMM ==")
    has_mps = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    print(f"MPS available: {has_mps}; CUDA available: {torch.cuda.is_available()}")

    # Build both models
    attn_model = build_model(True, batch_size, "attn")
    gmm_model = build_model(False, batch_size, "gmm")

    # Align initial weights for a fair start
    copy_shared_weights(attn_model, gmm_model)

    # Train attention
    print(f"[Epoch] Training Attention for {epochs} epoch(s)...")
    t0 = time.time()
    epoch_train(attn_model, epochs)
    t1 = time.time()
    print(f"Attention training time: {t1 - t0:.1f}s")

    # Train gmm
    print(f"[Epoch] Training GMM for {epochs} epoch(s)...")
    t0 = time.time()
    epoch_train(gmm_model, epochs)
    t1 = time.time()
    print(f"GMM training time:       {t1 - t0:.1f}s")

    # Full evaluations
    print("[Eval] Full eval on test set (Attention)...")
    attn_ce, attn_top1 = full_eval(attn_model)
    print(f"Attention eval: CE={attn_ce:.4f}, Top1={attn_top1*100:.2f}%")

    print("[Eval] Full eval on test set (GMM)...")
    gmm_ce, gmm_top1 = full_eval(gmm_model)
    print(f"GMM eval:       CE={gmm_ce:.4f}, Top1={gmm_top1*100:.2f}%")

    # Summary
    print("\n== Summary (Epoch x Full Eval) ==")
    print(f"Attention: evalCE={attn_ce:.4f}, evalTop1={attn_top1*100:.2f}%")
    print(f"GMM:       evalCE={gmm_ce:.4f}, evalTop1={gmm_top1*100:.2f}%")


if __name__ == "__main__":
    main()

