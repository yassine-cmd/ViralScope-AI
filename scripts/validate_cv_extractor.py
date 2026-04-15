"""
Task 2.2: Validate CV Extractor with Real Data
"""
import os
import sys
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cv_extractor import CVExtractor


def validate_cv_extractor():
    print("=" * 60)
    print("Task 2.2: Validate CV Extractor with Real Data")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
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
    
    # Initialize CV extractor
    print("\nInitializing CV Extractor...")
    model = CVExtractor(pretrained=True, freeze=True)
    model = model.to(device)
    model.eval()
    
    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
    expected_shape = (len(batch_images), 1280)
    
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
    checkpoint_path = 'models/checkpoints/cv_extractor_validation.pt'
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({
        'features': features.cpu(),
        'video_ids': batch_video_ids,
        'model_state': model.state_dict()
    }, checkpoint_path)
    print(f"\nValidation checkpoint saved: {checkpoint_path}")
    
    print("\n" + "=" * 60)
    print("Task 2.2: CV Extractor Validation PASSED")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = validate_cv_extractor()
    sys.exit(0 if success else 1)
