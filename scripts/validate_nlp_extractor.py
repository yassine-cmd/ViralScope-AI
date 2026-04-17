"""
Task 3.2: Validate NLP Extractor (CLIP Text) with Real Data
"""
import os
import sys
import yaml
import torch
from transformers import CLIPModel, CLIPTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.nlp_extractor import NLPExtractor


def validate_nlp_extractor():
    print("=" * 60)
    print("Task 3.2: Validate NLP Extractor (CLIP Text) with Real Data")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    clip_cfg = config['model']['clip']
    checkpoint = clip_cfg['checkpoint']
    max_seq_length = clip_cfg.get('max_seq_length', 77)

    # Initialize NLP extractor (CLIP Text)
    print(f"\nInitializing NLP Extractor (CLIP Text: {checkpoint})...")
    clip_model = CLIPModel.from_pretrained(checkpoint)
    model = NLPExtractor(
        text_model=clip_model.text_model,
        text_projection=clip_model.text_projection,
        freeze=True,
    )
    model = model.to(device)
    model.eval()

    # Tokenize sample titles
    print("\nTokenizing sample titles...")
    tokenizer = CLIPTokenizer.from_pretrained(checkpoint)

    sample_titles = [
        "How I Built $1,000,000 Business in 30 Days",
        "I Tried 100 Days of EXTREME CHALLENGE",
        "WORLD RECORD: Fastest Time to Beat All Bosses",
        "Simple cooking tutorial for beginners",
        "My morning routine 2024 (nothing special)",
    ]

    encoded = tokenizer(
        sample_titles,
        max_length=max_seq_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )

    batch_input_ids = encoded['input_ids'].to(device)
    batch_attention_masks = encoded['attention_mask'].to(device)

    print(f"Batch input_ids shape: {batch_input_ids.shape}")
    print(f"Batch attention_masks shape: {batch_attention_masks.shape}")

    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        features = model(batch_input_ids, batch_attention_masks)

    # Validate output
    feature_dim = clip_cfg.get('feature_dim', 512)
    print(f"Output shape: {features.shape}")
    expected_shape = (len(sample_titles), feature_dim)

    if features.shape != expected_shape:
        print(f"[ERROR] Expected shape {expected_shape}, got {features.shape}")
        return False

    # Check L2 normalization
    norms = features.norm(dim=1)
    print(f"Feature norms (should be ~1.0): mean={norms.mean():.4f}, std={norms.std():.6f}")

    # Check for NaN/Inf
    if torch.isnan(features).any():
        print("[ERROR] Output contains NaN values")
        return False

    if torch.isinf(features).any():
        print("[ERROR] Output contains Inf values")
        return False

    # Check variance (different titles should produce different embeddings)
    variance = features.var().item()
    print(f"Feature variance: {variance:.4f}")
    if variance < 0.001:
        print("[WARNING] Very low variance - model may be outputting constants")

    # Compute pairwise cosine similarities
    print("\nPairwise cosine similarities:")
    for i in range(len(sample_titles)):
        for j in range(i + 1, len(sample_titles)):
            sim = (features[i] * features[j]).sum().item()
            print(f"  '{sample_titles[i][:40]}...' vs '{sample_titles[j][:40]}...': {sim:.4f}")

    # Save validation checkpoint
    checkpoint_path = 'models/checkpoints/nlp_extractor_validation.pt'
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({
        'features': features.cpu(),
        'titles': sample_titles,
    }, checkpoint_path)
    print(f"\nValidation checkpoint saved: {checkpoint_path}")

    print("\n" + "=" * 60)
    print("Task 3.2: NLP Extractor (CLIP Text) Validation PASSED")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = validate_nlp_extractor()
    sys.exit(0 if success else 1)
