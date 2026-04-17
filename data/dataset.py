import os
import torch
import numpy as np
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
            tokenizer: HuggingFace tokenizer
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
        self.max_seq_length = self.config.get("model", {}).get("nlp", {}).get("max_seq_length", 64)

        self.df = self._load_split()

    def _load_split(self):
        df = pd.read_csv(os.path.join(self.data_dir, "labeled_dataset.csv"))

        split_file = os.path.join(self.splits_dir, f"{self.split}_indices.pt")
        if os.path.exists(split_file):
            indices = torch.load(split_file, weights_only=True).numpy()
            df = df.iloc[indices].reset_index(drop=True)
        else:
            raise FileNotFoundError(f"Split indices not found: {split_file}")

        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = os.path.join(self.thumbnail_dir, f"{row['video_id']}.jpg")
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            self.missing_count = getattr(self, 'missing_count', 0) + 1
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        if self.transform:
            image = self.transform(image)

        title = str(row.get("title", "Untitled"))
        if self.tokenizer:
            encoding = self.tokenizer(
                title,
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
                return_tensors="pt"
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
            "label": label
        }

    def get_labels(self):
        """Return all labels for sampler construction."""
        return self.df["is_viral"].values


def build_train_transform(config):
    """Build augmented transform for training images."""
    aug = config.get("augmentation", {})
    size = tuple(aug.get("resize_size", [224, 224]))
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_eval_transform(config):
    """Build clean transform for validation/test images (no augmentation)."""
    aug = config.get("augmentation", {})
    size = tuple(aug.get("resize_size", [224, 224]))
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    from transformers import AutoTokenizer

    batch_size = batch_size or config.get("training", {}).get("batch_size", 32)
    num_workers = num_workers or config.get("training", {}).get("num_workers", 2)

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["nlp"]["checkpoint"])
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


class DummyViralScopeDataset(Dataset):
    def __init__(self, num_samples=100, seq_len=64):
        self.num_samples = num_samples
        self.seq_len = seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "image": torch.randn(3, 224, 224),
            "input_ids": torch.randint(0, 30522, (self.seq_len,)),
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "label": torch.tensor(idx % 2, dtype=torch.float32)
        }

    def get_labels(self):
        return np.array([i % 2 for i in range(self.num_samples)])
