import torch
import torch.nn as nn
import torch.nn.functional as F


class NLPExtractor(nn.Module):
    """CLIP Text Encoder — extracts text features from video titles.

    Wraps CLIP's text transformer and text projection to produce
    L2-normalized 512-dim embeddings aligned with CLIP's vision space.
    """

    def __init__(self, text_model, text_projection, freeze=True):
        """
        Args:
            text_model: CLIPTextModel (from CLIPModel.text_model)
            text_projection: nn.Linear (from CLIPModel.text_projection)
            freeze: if True, freeze all backbone parameters
        """
        super().__init__()
        self.text_model = text_model
        self.text_projection = text_projection

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: shape (batch, seq_len)
            attention_mask: shape (batch, seq_len)
        Returns:
            features: L2-normalized, shape (batch, 512)
        """
        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled = outputs.pooler_output                   # (batch, 512)
        projected = self.text_projection(pooled)         # (batch, 512)
        return F.normalize(projected, p=2, dim=1)
