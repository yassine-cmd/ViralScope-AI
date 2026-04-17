import os
import sys
import argparse
import yaml
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from torchvision import transforms
from transformers import CLIPTokenizer

from models.multimodal import ViralScopeModel
from data.dataset import (
    ViralScopeDataset,
    build_train_transform,
    build_eval_transform,
    build_weighted_sampler,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(preds, targets, threshold=0.5):
    preds_binary = (preds >= threshold).float()

    tp = ((preds_binary == 1) & (targets == 1)).sum().item()
    tn = ((preds_binary == 0) & (targets == 0)).sum().item()
    fp = ((preds_binary == 1) & (targets == 0)).sum().item()
    fn = ((preds_binary == 0) & (targets == 1)).sum().item()

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc_roc = roc_auc_score(targets.numpy(), preds.numpy())
        pr_auc = average_precision_score(targets.numpy(), preds.numpy())
    except (ImportError, ValueError):
        auc_roc = 0.0
        pr_auc = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc_roc,
        "pr_auc": pr_auc,
    }


def train_epoch(model, dataloader, criterion, optimizer, device, clip_norm=1.0):
    """Train one epoch with gradient clipping."""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []

    for batch in dataloader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(images, input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()

        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

        optimizer.step()

        total_loss += loss.item()
        probs = torch.sigmoid(logits).detach()
        all_preds.extend(probs.cpu())
        all_targets.extend(labels.cpu())

    all_preds = torch.stack(all_preds)
    all_targets = torch.stack(all_targets)
    metrics = compute_metrics(all_preds, all_targets)
    metrics["loss"] = total_loss / len(dataloader)

    return metrics


@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device):
    """Validate one epoch."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    for batch in dataloader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(images, input_ids, attention_mask)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        probs = torch.sigmoid(logits)
        all_preds.extend(probs.cpu())
        all_targets.extend(labels.cpu())

    all_preds = torch.stack(all_preds)
    all_targets = torch.stack(all_targets)
    metrics = compute_metrics(all_preds, all_targets)
    metrics["loss"] = total_loss / len(dataloader)

    return metrics


def unfreeze_backbone(model, backbone_name, lr, optimizer):
    """Unfreeze a backbone and add its params to the optimizer with a low LR."""
    backbone = getattr(model, backbone_name)
    for param in backbone.parameters():
        param.requires_grad = True

    optimizer.add_param_group({
        "params": [p for p in backbone.parameters() if p.requires_grad],
        "lr": lr,
    })
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"  Unfroze {backbone_name}: {trainable:,} params at lr={lr}")


def build_optimizer(model, config, stage="head"):
    """Build AdamW optimizer with appropriate LR for each stage.

    Stage 'head':  Only fusion MLP is trainable, lr = lr_head (1e-3)
    Stage 'full':  After unfreezing, head lr drops to lr_head_stage2 (1e-4),
                   backbones get lr_backbone (5e-6)
    """
    training_cfg = config["training"]
    head_params = [p for p in model.fusion.parameters() if p.requires_grad]

    if stage == "head":
        lr = training_cfg["lr_head"]
    else:
        lr = training_cfg.get("lr_head_stage2", training_cfg["lr_head"] * 0.1)

    optimizer = AdamW(
        head_params,
        lr=lr,
        weight_decay=training_cfg["weight_decay"],
        betas=tuple(training_cfg.get("betas", [0.9, 0.999])),
    )
    return optimizer


def build_scheduler(optimizer, config):
    """Build CosineAnnealingWarmRestarts scheduler."""
    training_cfg = config["training"]
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=training_cfg.get("scheduler_T_0", 10),
        T_mult=training_cfg.get("scheduler_T_mult", 2),
        eta_min=training_cfg.get("scheduler_eta_min", 1e-6),
    )
    return scheduler


def build_criterion(config):
    """Build loss function — BCEWithLogitsLoss by default.

    WeightedRandomSampler handles class imbalance,
    so we do NOT add class weights to the loss.
    """
    training_cfg = config["training"]
    loss_name = training_cfg.get("loss_function", "BCEWithLogitsLoss")

    if loss_name == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss()
    elif loss_name == "FocalLoss":
        from models.losses import FocalLoss
        return FocalLoss(
            gamma=training_cfg.get("focal_loss_gamma", 2.0),
            alpha=None,  # No alpha — sampler handles imbalance
        )
    else:
        return nn.BCEWithLogitsLoss()


def main(config_path="config.yaml", epochs=None, batch_size=None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["project"]["seed"])

    device_cfg = config["project"]["device"]
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)
    print(f"Using device: {device}")

    training_cfg = config["training"]
    batch_size = batch_size or training_cfg["batch_size"]
    epochs = epochs or training_cfg["epochs"]
    clip_norm = training_cfg.get("gradient_clip_norm", 1.0)
    two_stage = training_cfg.get("two_stage", {})

    print("Loading model (CLIP backbone)...")
    model = ViralScopeModel(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,} (fusion head only)")

    print("Loading datasets...")
    clip_checkpoint = config["model"]["clip"]["checkpoint"]
    tokenizer = CLIPTokenizer.from_pretrained(clip_checkpoint)
    train_transform = build_train_transform(config)
    eval_transform = build_eval_transform(config)

    train_dataset = ViralScopeDataset("train", train_transform, tokenizer, config)
    val_dataset = ViralScopeDataset("val", eval_transform, tokenizer, config)

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")

    train_sampler = build_weighted_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=training_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=training_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    criterion = build_criterion(config)
    optimizer = build_optimizer(model, config, stage="head")
    scheduler = build_scheduler(optimizer, config)

    checkpoint_dir = config["paths"]["checkpoints"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = config["paths"]["best_model"]
    log_path = config["paths"]["training_log"]

    early_stopping_patience = training_cfg["early_stopping_patience"]
    early_stopping_counter = 0
    best_metric = 0.0
    best_metrics = {}
    training_log = []

    head_warmup_epochs = two_stage.get("head_warmup_epochs", epochs) if two_stage.get("enabled") else epochs
    backbones_unfrozen = False

    print(f"\n{'='*100}")
    print(f"Starting {epochs} epochs of two-stage training...")
    print(f"  Stage 1: Head warmup for {head_warmup_epochs} epochs (backbones frozen)")
    if two_stage.get("enabled"):
        print(f"  Stage 2: Full fine-tuning (backbones unfrozen at epoch {head_warmup_epochs + 1})")
    print(f"{'='*100}")
    print()
    print(f"{'Epoch':>5} | {'Stage':>7} | {'Train Loss':>10} | {'Train Acc':>8} | "
          f"{'Val Loss':>10} | {'Val Acc':>8} | {'Val F1':>8} | {'Val PR-AUC':>10}")
    print("-" * 100)

    for epoch in range(epochs):
        # ------- Stage 2: Unfreeze backbones -------
        if two_stage.get("enabled") and epoch == head_warmup_epochs and not backbones_unfrozen:
            print(f"\n{'='*100}")
            print(f"  STAGE 2: Unfreezing backbones at epoch {epoch + 1}")
            print(f"{'='*100}")

            # Rebuild optimizer with staggered LRs:
            #   fusion head → lr_head_stage2 (1e-4)
            #   backbones   → lr_backbone    (5e-6)
            optimizer = build_optimizer(model, config, stage="full")

            backbone_lr = training_cfg["lr_backbone"]
            if two_stage.get("unfreeze_cv", True):
                unfreeze_backbone(model, "cv_extractor", backbone_lr, optimizer)
            if two_stage.get("unfreeze_nlp", True):
                unfreeze_backbone(model, "nlp_extractor", backbone_lr, optimizer)
            backbones_unfrozen = True

            trainable_now = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Trainable params now: {trainable_now:,}")

            # Reset early stopping and scheduler for stage 2
            early_stopping_counter = 0
            scheduler = build_scheduler(optimizer, config)
            print()

        stage = "head" if not backbones_unfrozen else "full"
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, clip_norm)
        val_metrics = validate_epoch(model, val_loader, criterion, device)

        scheduler.step(epoch)

        log_entry = {
            "epoch": epoch + 1,
            "stage": stage,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "train_pr_auc": train_metrics["pr_auc"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_auc_roc": val_metrics["auc_roc"],
            "val_pr_auc": val_metrics["pr_auc"],
        }
        training_log.append(log_entry)

        print(
            f"{epoch+1:>5} | {stage:>7} | {train_metrics['loss']:>10.4f} | {train_metrics['accuracy']:>8.4f} | "
            f"{val_metrics['loss']:>10.4f} | {val_metrics['accuracy']:>8.4f} | "
            f"{val_metrics['f1']:>8.4f} | {val_metrics['pr_auc']:>10.4f}"
        )

        if val_metrics["pr_auc"] > best_metric:
            best_metric = val_metrics["pr_auc"]
            best_metrics = val_metrics.copy()
            early_stopping_counter = 0

            torch.save({
                "epoch": epoch,
                "stage": stage,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_metric": best_metric,
                "config": config,
            }, best_model_path)
            print(f"  -> Saved best model (PR-AUC: {best_metric:.4f})")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

    pd.DataFrame(training_log).to_csv(log_path, index=False)
    print(f"\nTraining complete. Best PR-AUC: {best_metric:.4f}")
    print(f"Model saved to: {best_model_path}")
    print(f"Training log saved to: {log_path}")

    return best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViralScope AI model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    args = parser.parse_args()

    main(args.config, args.epochs, args.batch_size)
