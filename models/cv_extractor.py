import torch
import torch.nn as nn
import torch.nn.functional as F


class CVExtractor(nn.Module):
    """CLIP Vision Encoder — extracts visual features from thumbnails.

    Wraps CLIP's vision transformer and visual projection to produce
    L2-normalized 512-dim embeddings aligned with CLIP's text space.
    """

    def __init__(self, vision_model, visual_projection, freeze=True):
        """
        Args:
            vision_model: CLIPVisionModel (from CLIPModel.vision_model)
            visual_projection: nn.Linear (from CLIPModel.visual_projection)
            freeze: if True, freeze all backbone parameters
        """
        super().__init__()
        self.vision_model = vision_model
        self.visual_projection = visual_projection

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, pixel_values):
        """
        Args:
            pixel_values: image tensor, shape (batch, 3, 224, 224)
        Returns:
            features: L2-normalized, shape (batch, 512)
        """
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled = outputs.pooler_output                   # (batch, 768)
        projected = self.visual_projection(pooled)       # (batch, 512)
        return F.normalize(projected, p=2, dim=1)
