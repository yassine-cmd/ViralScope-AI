import torch
import torch.nn as nn


class CVExtractor(nn.Module):
    def __init__(self, pretrained=True, feature_dim=1280, freeze=True):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, images):
        batch_size = images.shape[0]
        return torch.randn(batch_size, self.feature_dim)
import torch
import torch.nn as nn
import torchvision.models as models


class CVExtractor(nn.Module):
    def __init__(self, pretrained=True, feature_dim=1280, freeze=True):
        super().__init__()
        backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        )

        self.features = nn.Sequential(
            backbone.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """
        Args:
            x: image tensor, shape (batch, 3, 224, 224)
        Returns:
            features: shape (batch, 1280)
        """
        return self.features(x)
