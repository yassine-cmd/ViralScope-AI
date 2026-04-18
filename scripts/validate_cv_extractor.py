"""
Task 2.2: Validate CV Extractor (CLIP Vision) with Real Data
"""
import os
import sys
import yaml
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from transformers import CLIPModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cv_extractor import CVExtractor


def validate_cv_extractor():
    print("=" * 60)
    print("Task 2.2: Validate CV Extractor (CLIP Vision) with Real Data")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    clip_cfg = config['model']['clip']
    aug = config.get('augmentation', {})

    # Load dataset info
    csv_path = 'data/processed/labeled_dataset.csv'
    train_indices_path = 'data/splits/train_indices.pt'
    thumbnail_dir = 'data/raw/thumbnails'

    if not os.path.exists(csv_path):
        print(f"[ERROR] labeled_dataset.csv not found at {csv_path}")
        return False

    if not os.path.exists(train_indices_path):
        print(f"[ERROR] train_indices.pt not found at {train_indices_path}")
        return False

    # Load dataset
    df = pd.read_csv(csv_path)
    train_indices = torch.load(train_indices_path)

    print(f"Dataset size: {len(df)}")
    print(f"Training samples: {len(train_indices)}")

    # Initialize CV extractor (CLIP Vision)
    print("\nInitializing CV Extractor (CLIP Vision)...")
    clip_model = CLIPModel.from_pretrained(clip_cfg['checkpoint'])
    model = CVExtractor(
        vision_model=clip_model.vision_model,
        visual_projection=clip_model.visual_projection,
        freeze=True,
    )
    model = model.to(device)
    model.eval()

    # CLIP image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=aug.get('normalize_mean', [0.48145466, 0.4578275, 0.40821073]),
            std=aug.get('normalize_std', [0.26862954, 0.26130258, 0.27577711]),
        ),
    ])

    # Load first 64 training images
    print("\nLoading first 64 training images...")
    batch_images = []
    batch_video_ids = []

    for idx in train_indices[:64]:
        video_id = df.iloc[idx]['video_id']
        img_path = os.path.join(thumbnail_dir, f"{video_id}.jpg")

        if os.path.exists(img_path):
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image)
                batch_images.append(image_tensor)
                batch_video_ids.append(video_id)
            except Exception as e:
                print(f"  Warning: Failed to load {img_path}: {e}")
        else:
            print(f"  Warning: Thumbnail not found: {img_path}")

    if len(batch_images) == 0:
        print("[ERROR] No images could be loaded")
        return False

    print(f"Successfully loaded {len(batch_images)} images")

    # Create batch
    batch_tensor = torch.stack(batch_images).to(device)
    print(f"Batch shape: {batch_tensor.shape}")

    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        features = model(batch_tensor)

    # Validate output
    print(f"Output shape: {features.shape}")
    feature_dim = clip_cfg.get('feature_dim', 512)
    expected_shape = (len(batch_images), feature_dim)

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

    # Check variance
    variance = features.var().item()
    print(f"Feature variance: {variance:.4f}")
    if variance < 0.001:
        print("[WARNING] Very low variance - model may be outputting constants")

    # Save validation checkpoint
    checkpoint_path = 'models/checkpoints/cv_extractor_validation.pt'
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({
        'features': features.cpu(),
        'video_ids': batch_video_ids,
        'model_state': model.state_dict()
    }, checkpoint_path)
    print(f"\nValidation checkpoint saved: {checkpoint_path}")

    print("\n" + "=" * 60)
    print("Task 2.2: CV Extractor (CLIP Vision) Validation PASSED")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = validate_cv_extractor()
    sys.exit(0 if success else 1)
