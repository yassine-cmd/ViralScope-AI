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
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchvision import transforms
from transformers import AutoTokenizer

from models.multimodal import ViralScopeModel
from models.losses import FocalLoss
from data.dataset import ViralScopeDataset, DummyViralScopeDataset


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
    except ImportError:
        auc_roc = 0.0
        pr_auc = 0.0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc_roc,
        "pr_auc": pr_auc
    }


def train_epoch(model, dataloader, criterion, optimizer, device, freeze_backbone=True):
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


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
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


def main(config_path="config.yaml", epochs=None, batch_size=None, use_dummy_data=False):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    set_seed(config["project"]["seed"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config["project"]["device"] == "cpu":
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    batch_size = batch_size or config["training"]["batch_size"]
    epochs = epochs or config["training"]["epochs"]
    
    print("Loading model...")
    model = ViralScopeModel(config).to(device)
    
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "fusion" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    
    optimizer = AdamW([
        {"params": backbone_params, "lr": config["model"]["lr_backbone_cv"]},
        {"params": head_params, "lr": config["model"]["lr_head"]}
    ], weight_decay=config["model"]["weight_decay"])
    
    if config["training"]["loss_function"] == "FocalLoss":
        class_weights = config["training"].get("focal_loss_alpha")
        criterion = FocalLoss(
            alpha=class_weights,
            gamma=config["training"]["focal_loss_gamma"]
        )
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config["training"]["scheduler_factor"],
        patience=config["training"]["scheduler_patience"],
        verbose=True
    )
    
    if use_dummy_data:
        print("Using dummy data for training...")
        train_dataset = DummyViralScopeDataset(num_samples=200)
        val_dataset = DummyViralScopeDataset(num_samples=50)
    else:
        print("Loading datasets...")
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["nlp"]["checkpoint"])
        
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = ViralScopeDataset(
            split="train",
            transform=train_transform,
            tokenizer=tokenizer,
            config=config
        )
        val_dataset = ViralScopeDataset(
            split="val",
            transform=train_transform,
            tokenizer=tokenizer,
            config=config
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=torch.cuda.is_available()
    )
    
    checkpoint_dir = config["paths"]["checkpoints"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = config["paths"]["best_model"]
    log_path = config["paths"]["training_log"]
    
    early_stopping_patience = config["training"]["early_stopping_patience"]
    early_stopping_counter = 0
    best_metric = 0.0
    best_metrics = {}
    
    training_log = []
    
    print(f"\nStarting training for {epochs} epochs...\n")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>8} | {'Val Loss':>10} | {'Val Acc':>8} | {'Val F1':>8} | {'Val PR-AUC':>10}")
    print("-" * 80)
    
    for epoch in range(epochs):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        scheduler.step(val_metrics["pr_auc"])
        
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "train_auc_roc": train_metrics["auc_roc"],
            "train_pr_auc": train_metrics["pr_auc"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_auc_roc": val_metrics["auc_roc"],
            "val_pr_auc": val_metrics["pr_auc"]
        }
        training_log.append(log_entry)
        
        print(f"{epoch+1:>5} | {train_metrics['loss']:>10.4f} | {train_metrics['accuracy']:>8.4f} | "
              f"{val_metrics['loss']:>10.4f} | {val_metrics['accuracy']:>8.4f} | "
              f"{val_metrics['f1']:>8.4f} | {val_metrics['pr_auc']:>10.4f}")
        
        if val_metrics["pr_auc"] > best_metric:
            best_metric = val_metrics["pr_auc"]
            best_metrics = val_metrics.copy()
            early_stopping_counter = 0
            
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_metric": best_metric,
                "config": config
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
    parser.add_argument("--dummy-data", action="store_true", help="Use dummy data for testing")
    args = parser.parse_args()
    
    main(args.config, args.epochs, args.batch_size, args.dummy_data)
