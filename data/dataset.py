import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import pandas as pd


class ViralScopeDataset(Dataset):
    def __init__(self, split="train", transform=None, tokenizer=None, config=None):
        """
        Args:
            split: "train", "val", or "test"
            transform: torchvision transforms for images
            tokenizer: CLIPTokenizer instance
            config: config dict
        """
        self.config = config or {}
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer

        raw_dir = self.config.get("data", {}).get("raw_dir", "data/raw")
        self.thumbnail_dir = os.path.join(raw_dir, "thumbnails")

        self.data_dir = self.config.get("data", {}).get("processed_dir", "data/processed")
        self.splits_dir = self.config.get("data", {}).get("splits_dir", "data/splits")
        self.max_seq_length = self.config.get("model", {}).get("clip", {}).get("max_seq_length", 77)

        self.df = self._load_split()

    def _load_split(self):
        df = pd.read_csv(os.path.join(self.data_dir, "labeled_dataset.csv"))

        split_file = os.path.join(self.splits_dir, f"{self.split}_indices.pt")
        if os.path.exists(split_file):
            indices = torch.load(split_file, weights_only=True).numpy()
            df = df.iloc[indices].reset_index(drop=True)
        else:
            raise FileNotFoundError(f"Split indices not found: {split_file}")

        # Verify thumbnails exist (upstream should have filtered, but safety check)
        valid_mask = df["video_id"].apply(
            lambda vid: os.path.exists(os.path.join(self.thumbnail_dir, f"{vid}.jpg"))
        )
        n_missing = (~valid_mask).sum()
        if n_missing > 0:
            print(f"  [{self.split}] Dropped {n_missing}/{len(df)} rows with missing thumbnails")
        df = df[valid_mask].reset_index(drop=True)

        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load thumbnail — all should exist after upstream filtering
        image_path = os.path.join(self.thumbnail_dir, f"{row['video_id']}.jpg")
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Tokenize title with CLIP tokenizer
        title = str(row.get("title", "Untitled"))
        if self.tokenizer:
            encoding = self.tokenizer(
                title,
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].squeeze()
            attention_mask = encoding["attention_mask"].squeeze()
        else:
            input_ids = torch.zeros(self.max_seq_length, dtype=torch.long)
            attention_mask = torch.zeros(self.max_seq_length, dtype=torch.long)

        label = torch.tensor(row["is_viral"], dtype=torch.float32)

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label,
        }

    def get_labels(self):
        """Return all labels for sampler construction."""
        return self.df["is_viral"].values


def build_train_transform(config):
    """Build augmented transform for training images using CLIP normalization."""
    aug = config.get("augmentation", {})
    size = tuple(aug.get("resize_size", [224, 224]))
    mean = aug.get("normalize_mean", [0.48145466, 0.4578275, 0.40821073])
    std = aug.get("normalize_std", [0.26862954, 0.26130258, 0.27577711])

    return transforms.Compose([
        transforms.Resize(size),
        transforms.RandomRotation(degrees=aug.get("rotation_range", [-10, 10])[1]),
        transforms.ColorJitter(
            brightness=aug.get("color_jitter_brightness", 0.2),
            contrast=aug.get("color_jitter_contrast", 0.2),
            saturation=aug.get("color_jitter_saturation", 0.1),
        ),
        transforms.RandomHorizontalFlip(p=aug.get("horizontal_flip_prob", 0.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def build_eval_transform(config):
    """Build clean transform for validation/test images (no augmentation, CLIP normalization)."""
    aug = config.get("augmentation", {})
    size = tuple(aug.get("resize_size", [224, 224]))
    mean = aug.get("normalize_mean", [0.48145466, 0.4578275, 0.40821073])
    std = aug.get("normalize_std", [0.26862954, 0.26130258, 0.27577711])

    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def build_weighted_sampler(dataset):
    """Build a WeightedRandomSampler to balance classes in each batch."""
    labels = dataset.get_labels()
    class_counts = np.bincount(labels.astype(int))
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels.astype(int)]
    sample_weights = torch.from_numpy(sample_weights).double()
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def create_dataloaders(config, batch_size=None, num_workers=None):
    """Create train/val/test dataloaders with proper transforms and class balancing."""
    from transformers import CLIPTokenizer

    clip_cfg = config.get("model", {}).get("clip", {})
    batch_size = batch_size or config.get("training", {}).get("batch_size", 32)
    num_workers = num_workers or config.get("training", {}).get("num_workers", 2)

    tokenizer = CLIPTokenizer.from_pretrained(clip_cfg.get("checkpoint", "openai/clip-vit-base-patch32"))
    train_transform = build_train_transform(config)
    eval_transform = build_eval_transform(config)

    train_ds = ViralScopeDataset(split="train", transform=train_transform, tokenizer=tokenizer, config=config)
    val_ds = ViralScopeDataset(split="val", transform=eval_transform, tokenizer=tokenizer, config=config)
    test_ds = ViralScopeDataset(split="test", transform=eval_transform, tokenizer=tokenizer, config=config)

    train_sampler = build_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader
