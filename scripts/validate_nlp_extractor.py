"""
Task 3.2: Validate NLP Extractor with Real Data
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.nlp_extractor import NLPExtractor


def validate_nlp_extractor():
    print("=" * 60)
    print("Task 3.2: Validate NLP Extractor with Real Data")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load pre-tokenized tensors
    input_ids_path = 'data/tensors/input_ids.pt'
    attention_masks_path = 'data/tensors/attention_masks.pt'
    train_indices_path = 'data/splits/train_indices.pt'
    
    for path in [input_ids_path, attention_masks_path, train_indices_path]:
        if not os.path.exists(path):
            print(f"[ERROR] Required file not found: {path}")
            return False
    
    print("\nLoading pre-tokenized tensors...")
    input_ids = torch.load(input_ids_path, weights_only=True)
    attention_masks = torch.load(attention_masks_path, weights_only=True)
    train_indices = torch.load(train_indices_path)
    
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Attention Masks shape: {attention_masks.shape}")
    print(f"Training indices: {len(train_indices)}")
    
    # Initialize NLP extractor
    print("\nInitializing NLP Extractor (DistilBERT)...")
    model = NLPExtractor(checkpoint="distilbert-base-uncased", freeze=True)
    model = model.to(device)
    model.eval()
    
    # Get first 64 training samples
    print("\nExtracting features from first 64 training samples...")
    batch_input_ids = input_ids[train_indices[:64]].to(device)
    batch_attention_masks = attention_masks[train_indices[:64]].to(device)
    
    print(f"Batch input_ids shape: {batch_input_ids.shape}")
    print(f"Batch attention_masks shape: {batch_attention_masks.shape}")
    
    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        features = model(batch_input_ids, batch_attention_masks)
    
    # Validate output
    print(f"Output shape: {features.shape}")
    expected_shape = (64, 768)
    
    if features.shape != expected_shape:
        print(f"[ERROR] Expected shape {expected_shape}, got {features.shape}")
        return False
    
    # Check for NaN/Inf
    if torch.isnan(features).any():
        print("[ERROR] Output contains NaN values")
        return False
    
    if torch.isinf(features).any():
        print("[ERROR] Output contains Inf values")
        return False
    
    # Check variance
    variance = features.var().item()
    print(f"Feature variance: {variance:.4f}")
    if variance < 0.01:
        print("[WARNING] Very low variance - model may be outputting constants")
    
    # Save validation checkpoint
    checkpoint_path = 'models/checkpoints/nlp_extractor_validation.pt'
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({
        'features': features.cpu(),
        'train_indices': train_indices[:64]
    }, checkpoint_path)
    print(f"\nValidation checkpoint saved: {checkpoint_path}")
    
    print("\n" + "=" * 60)
    print("Task 3.2: NLP Extractor Validation PASSED")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = validate_nlp_extractor()
    sys.exit(0 if success else 1)
