#!/usr/bin/env python3
"""
Quick comparison of Attention classifier vs. GMM-based classifier on CIFAR-10.

Default为“快速对比模式”（仅若干 step），因此运行时间短、准确率低，只适合粗略 sanity check。
如果需要真正训练，请使用 epoch 模式：

Examples:
  # 快速对比（默认）：
  python compare_attention_vs_gmm_cifar10.py --mode quick --train_steps 200 --eval_batches 200

  # 真实训练若干个 epoch（全量训练集 + 可选全量评估）：
  python compare_attention_vs_gmm_cifar10.py --mode epoch --epochs 10 --full_eval
"""

import argparse
import sys
import time
import traceback

import torch
import torch.nn as nn

from model_training_attention import MIA_train


def set_seeds(seed: int = 123):
    torch.manual_seed(seed)


def copy_shared_weights(src: MIA_train, dst: MIA_train):
    """Copy shared module weights (encoder f, cloud f_tail, classifier) from src to dst for fair init."""
    # Encoder
    dst.f.load_state_dict(src.f.state_dict())
    # Cloud tail and classifier
    dst.f_tail.load_state_dict(src.f_tail.state_dict())
    dst.classifier.load_state_dict(src.classifier.state_dict())


@torch.no_grad()
def forward_logits(model: MIA_train, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Produce logits for a batch according to model's current classification path."""
    device = model.device
    images = images.to(device)
    labels = labels.to(device)

    # Local encoder to smashed features
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


def quick_train(model: MIA_train, max_steps: int = 50) -> float:
    """A short training loop over the client train loader of the model."""
    device = model.device
    model.f.train(); model.f_tail.train(); model.classifier.train()
    if getattr(model, 'attention_classifier', None) is not None:
        model.attention_classifier.train()

    crit = nn.CrossEntropyLoss().to(device)
    total_loss = 0.0
    steps = 0

    # Single client assumed (id 0)
    train_loader = model.client_dataloader[0]
    it = iter(train_loader)
    while steps < max_steps:
        try:
            images, labels = next(it)
        except StopIteration:
            it = iter(train_loader)
            continue

        images = images.to(device)
        labels = labels.to(device)

        model.optimizer_zero_grad()
        # Use the project's single-step training to ensure consistency
        # For attention path, centroids_list is unused; pass None and random_ini_centers=True
        try:
            _, _, _ = model.train_target_step(
                images, labels,
                adding_noise=False,
                random_ini_centers=True,
                centroids_list=None,
                client_id=0,
            )
        except Exception as e:
            # Fallback: do a direct forward/backward on CE if the above fails
            logits = forward_logits(model, images, labels)
            loss = crit(logits, labels)
            loss.backward()

        model.optimizer_step()

        # Track current CE loss (forward in no-grad for speed)
        with torch.no_grad():
            logits = forward_logits(model, images, labels)
            loss = crit(logits, labels)
            total_loss += float(loss.item())

        steps += 1

    return total_loss / max(1, steps)


@torch.no_grad()
def quick_eval(model: MIA_train, max_batches: int = 50) -> tuple[float, float]:
    """Evaluate on a few batches from the public/test loader; return (avg_ce, top1_acc)."""
    device = model.device
    model.f.eval(); model.f_tail.eval(); model.classifier.eval()
    if getattr(model, 'attention_classifier', None) is not None:
        model.attention_classifier.eval()

    crit = nn.CrossEntropyLoss().to(device)
    test_loader = model.pub_dataloader

    ce_sum = 0.0
    n_samples = 0
    correct = 0
    batches = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = forward_logits(model, images, labels)
        loss = crit(logits, labels)

        ce_sum += float(loss.item()) * images.size(0)
        n_samples += int(images.size(0))
        pred = logits.argmax(dim=1)
        correct += int((pred == labels).sum().item())

        batches += 1
        if batches >= max_batches:
            break

    avg_ce = ce_sum / max(1, n_samples)
    top1 = correct / max(1, n_samples)
    return avg_ce, top1


def epoch_train(model: MIA_train, n_epochs: int) -> None:
    """Train using the model's full epoch loop over the entire train loader."""
    # Override epochs if needed
    model.n_epochs = int(n_epochs)
    # Run main training loop (full dataset each epoch)
    # Disable verbose logs and progress bar for a cleaner run here
    model(log_frequency=100, verbose=False, progress_bar=False)


@torch.no_grad()
def full_eval(model: MIA_train) -> tuple[float, float]:
    """Evaluate on the entire test/public loader; return (avg_ce, top1_acc)."""
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
        save_dir=f"./saves/quick_compare_{save_dir_suffix}",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "epoch"], default="quick",
                        help="quick: 少量step快速对比；epoch: 按epoch完整训练")
    parser.add_argument("--train_steps", type=int, default=50, help="quick模式每个模型训练的steps数")
    parser.add_argument("--eval_batches", type=int, default=50, help="quick模式评估时的batch数")
    parser.add_argument("--epochs", type=int, default=1, help="epoch模式训练的epoch数")
    parser.add_argument("--full_eval", action="store_true", help="epoch模式下对全量测试集评估")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    set_seeds(123)

    print("== Quick CIFAR-10 Comparison: Attention vs GMM ==")
    # Inform about MPS
    has_mps = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    print(f"MPS available: {has_mps}; CUDA available: {torch.cuda.is_available()}")
    print("Note: The existing trainer prefers CUDA else CPU. On macOS, it will likely use CPU.")

    # Build models
    try:
        attn_model = build_model(True, args.batch_size, "attn")
        gmm_model = build_model(False, args.batch_size, "gmm")
    except Exception as e:
        print("[Error] Failed to initialize models. If this is due to dataset download restrictions,\n"
              "please ensure CIFAR-10 is present under ./data or allow network for torchvision datasets.")
        traceback.print_exc()
        sys.exit(1)

    # Align initial weights for a fair start
    copy_shared_weights(attn_model, gmm_model)

    if args.mode == "quick":
        print(f"[Quick] Training Attention for {args.train_steps} steps...")
        t0 = time.time()
        attn_train_loss = quick_train(attn_model, args.train_steps)
        t1 = time.time()
        print(f"Attention train CE: {attn_train_loss:.4f}  (time: {t1-t0:.1f}s)")

        print(f"[Quick] Training GMM for {args.train_steps} steps...")
        t0 = time.time()
        gmm_train_loss = quick_train(gmm_model, args.train_steps)
        t1 = time.time()
        print(f"GMM train CE: {gmm_train_loss:.4f}       (time: {t1-t0:.1f}s)")

        print(f"[Quick] Evaluating Attention on {args.eval_batches} batches...")
        attn_ce, attn_top1 = quick_eval(attn_model, args.eval_batches)
        print(f"Attention eval: CE={attn_ce:.4f}, Top1={attn_top1*100:.2f}%")

        print(f"[Quick] Evaluating GMM on {args.eval_batches} batches...")
        gmm_ce, gmm_top1 = quick_eval(gmm_model, args.eval_batches)
        print(f"GMM eval:       CE={gmm_ce:.4f}, Top1={gmm_top1*100:.2f}%")

        print("\n== Summary (Quick) ==")
        print(f"Attention: trainCE={attn_train_loss:.4f}, evalCE={attn_ce:.4f}, evalTop1={attn_top1*100:.2f}%")
        print(f"GMM:       trainCE={gmm_train_loss:.4f}, evalCE={gmm_ce:.4f}, evalTop1={gmm_top1*100:.2f}%")

    else:  # epoch mode
        print(f"[Epoch] Training Attention for {args.epochs} epoch(s) on full train set...")
        t0 = time.time()
        epoch_train(attn_model, args.epochs)
        t1 = time.time()
        print(f"Attention training done. (time: {t1-t0:.1f}s)")

        print(f"[Epoch] Training GMM for {args.epochs} epoch(s) on full train set...")
        t0 = time.time()
        epoch_train(gmm_model, args.epochs)
        t1 = time.time()
        print(f"GMM training done.       (time: {t1-t0:.1f}s)")

        if args.full_eval:
            print("[Epoch] Full eval on entire test set (Attention)...")
            attn_ce, attn_top1 = full_eval(attn_model)
            print(f"Attention eval: CE={attn_ce:.4f}, Top1={attn_top1*100:.2f}%")

            print("[Epoch] Full eval on entire test set (GMM)...")
            gmm_ce, gmm_top1 = full_eval(gmm_model)
            print(f"GMM eval:       CE={gmm_ce:.4f}, Top1={gmm_top1*100:.2f}%")
        else:
            print(f"[Epoch] Quick eval on {args.eval_batches} batches (Attention)...")
            attn_ce, attn_top1 = quick_eval(attn_model, args.eval_batches)
            print(f"Attention eval: CE={attn_ce:.4f}, Top1={attn_top1*100:.2f}%")

            print(f"[Epoch] Quick eval on {args.eval_batches} batches (GMM)...")
            gmm_ce, gmm_top1 = quick_eval(gmm_model, args.eval_batches)
            print(f"GMM eval:       CE={gmm_ce:.4f}, Top1={gmm_top1*100:.2f}%")

        print("\n== Summary (Epoch) ==")
        print(f"Attention: evalCE={attn_ce:.4f}, evalTop1={attn_top1*100:.2f}%")
        print(f"GMM:       evalCE={gmm_ce:.4f}, evalTop1={gmm_top1*100:.2f}%")


if __name__ == "__main__":
    main()
