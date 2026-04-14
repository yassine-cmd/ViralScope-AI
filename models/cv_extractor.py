import torch
import torch.nn as nn


class CVExtractor(nn.Module):
    def __init__(self, pretrained=True, feature_dim=1280, freeze=True):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, images):
        batch_size = images.shape[0]
        return torch.randn(batch_size, self.feature_dim)
