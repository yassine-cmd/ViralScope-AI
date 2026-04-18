import torch
import yaml
import json
import os
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
)

from models.multimodal import ViralScopeModel
from data.dataset import create_dataloaders


def load_model(config, device):
    """Load model from checkpoint, handling both raw state_dict and full checkpoint formats."""
    model = ViralScopeModel(config).to(device)

    model_path = config["paths"]["best_model"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}, "
              f"stage={checkpoint.get('stage', '?')}, "
              f"best_metric={checkpoint.get('best_metric', '?')}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded raw state_dict")

    model.eval()
    return model


@torch.no_grad()
def collect_predictions(model, dataloader, device):
    """Collect all predictions and labels from a dataloader."""
    all_probs = []
    all_labels = []

    for batch in dataloader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"]

        logits = model(images, input_ids, attention_mask)
        probs = torch.sigmoid(logits).cpu()
        all_probs.append(probs)
        all_labels.append(labels)

    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()


def find_optimal_threshold(labels, probs):
    """Find threshold that maximizes F1 on the given set."""
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    best_f1 = float(f1_scores[best_idx])
    return best_threshold, best_f1


def evaluate_with_threshold(labels, probs, threshold):
    """Compute all metrics using a given threshold."""
    preds = (probs > threshold).astype(int)

    return {
        "threshold": threshold,
        "accuracy": float((preds == labels).mean()),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auc_roc": float(roc_auc_score(labels, probs)),
        "pr_auc": float(average_precision_score(labels, probs)),
        "precision": float(
            classification_report(labels, preds, output_dict=True, zero_division=0).get("1", {}).get("precision", 0)
        ),
        "recall": float(
            classification_report(labels, preds, output_dict=True, zero_division=0).get("1", {}).get("recall", 0)
        ),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "classification_report": classification_report(labels, preds, output_dict=True, zero_division=0),
    }


def evaluate(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(config, device)

    print("Loading datasets...")
    _, val_loader, test_loader = create_dataloaders(config)

    print("\nFinding optimal threshold on validation set...")
    val_probs, val_labels = collect_predictions(model, val_loader, device)
    optimal_threshold, val_best_f1 = find_optimal_threshold(val_labels, val_probs)
    print(f"Optimal threshold: {optimal_threshold:.4f} (Val F1: {val_best_f1:.4f})")

    if optimal_threshold < 0.2:
        print(f"Note: Threshold {optimal_threshold:.4f} is below 0.2 - this can be normal for imbalanced data")

    print("\nEvaluating on test set...")
    test_probs, test_labels = collect_predictions(model, test_loader, device)

    metrics = evaluate_with_threshold(test_labels, test_probs, optimal_threshold)
    metrics_fixed = evaluate_with_threshold(test_labels, test_probs, 0.5)

    os.makedirs("results", exist_ok=True)
    results = {
        "optimal_threshold": metrics,
        "fixed_threshold_0.5": metrics_fixed,
        "validation_threshold": optimal_threshold,
        "validation_f1_at_threshold": val_best_f1,
    }
    with open("results/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("TEST SET EVALUATION (Threshold from Validation)")
    print("=" * 60)

    print(f"\n--- With optimal threshold ({optimal_threshold:.4f}) ---")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"{metrics['confusion_matrix']}")

    print(f"\n--- With fixed threshold (0.50) ---")
    print(f"Accuracy:  {metrics_fixed['accuracy']:.4f}")
    print(f"F1-Score:  {metrics_fixed['f1']:.4f}")
    print(f"Confusion Matrix:")
    print(f"{metrics_fixed['confusion_matrix']}")
    print("=" * 60)

    print(f"\nMetrics saved to: results/metrics.json")

    return results


if __name__ == "__main__":
    evaluate()
