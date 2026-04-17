import torch
import torch.nn as nn
from transformers import CLIPModel

from models.cv_extractor import CVExtractor
from models.nlp_extractor import NLPExtractor
from models.fusion_model import FusionMLP


class ViralScopeModel(nn.Module):
    """Multi-modal viral content classifier using CLIP.

    Loads a single CLIP model and splits it into vision and text encoders
    to avoid duplicate weights. Both encoders produce L2-normalized
    embeddings in a shared latent space, which are then fused with
    cross-modal interaction features for binary classification.
    """

    def __init__(self, config):
        super().__init__()

        clip_cfg = config["model"]["clip"]
        fusion_cfg = config["model"]["fusion"]

        # Load CLIP once, then split into vision / text components
        clip_model = CLIPModel.from_pretrained(clip_cfg["checkpoint"])

        self.cv_extractor = CVExtractor(
            vision_model=clip_model.vision_model,
            visual_projection=clip_model.visual_projection,
            freeze=clip_cfg["freeze_backbone"],
        )
        self.nlp_extractor = NLPExtractor(
            text_model=clip_model.text_model,
            text_projection=clip_model.text_projection,
            freeze=clip_cfg["freeze_backbone"],
        )

        self.fusion = FusionMLP(
            feature_dim=clip_cfg["feature_dim"],
            hidden_layers=fusion_cfg["hidden_layers"],
            dropout=fusion_cfg["dropout"],
            activation=fusion_cfg["activation"],
        )

    def forward(self, images, input_ids, attention_mask):
        """
        Args:
            images:         shape (batch, 3, 224, 224)
            input_ids:      shape (batch, seq_len)
            attention_mask: shape (batch, seq_len)
        Returns:
            logits: shape (batch,)
        """
        cv_features = self.cv_extractor(images)                       # (batch, 512)
        nlp_features = self.nlp_extractor(input_ids, attention_mask)  # (batch, 512)
        logits = self.fusion(cv_features, nlp_features)
        return logits

    def predict_proba(self, images, input_ids, attention_mask):
        """Returns probability instead of logit."""
        logits = self.forward(images, input_ids, attention_mask)
        return torch.sigmoid(logits)
