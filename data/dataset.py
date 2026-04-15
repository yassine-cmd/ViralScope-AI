import os
import torch
from torch.utils.data import Dataset
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
        
        self.data_dir = self.config.get("data", {}).get("processed_dir", "data/processed")
        self.thumbnail_dir = os.path.join(self.data_dir, "thumbnails")
        self.splits_dir = self.config.get("data", {}).get("splits_dir", "data/splits")
        
        self.df = self._load_split()
        
    def _load_split(self):
        df = pd.read_csv(os.path.join(self.data_dir, "labeled_dataset.csv"))
        
        split_file = os.path.join(self.splits_dir, f"{self.split}_indices.pt")
        if os.path.exists(split_file):
            indices = torch.load(split_file).numpy()
            df = df.iloc[indices].reset_index(drop=True)
        else:
            raise FileNotFoundError(f"Split indices not found: {split_file}")
        
        return df.reset_index(drop=True)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        image_path = os.path.join(self.thumbnail_dir, f"{row['video_id']}.jpg")
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        title = str(row.get("title", ""))
        if self.tokenizer:
            encoding = self.tokenizer(
                title,
                truncation=True,
                max_length=64,
                padding="max_length",
                return_tensors="pt"
            )
            input_ids = encoding["input_ids"].squeeze()
            attention_mask = encoding["attention_mask"].squeeze()
        else:
            input_ids = torch.zeros(64, dtype=torch.long)
            attention_mask = torch.zeros(64, dtype=torch.long)
        
        label = torch.tensor(row["is_viral"], dtype=torch.float32)
        
        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label
        }


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
            "attention_mask": torch.ones(self.seq_len),
            "label": torch.tensor(idx % 2, dtype=torch.float32)
        }
