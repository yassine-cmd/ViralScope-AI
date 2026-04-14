import torch
import torch.nn as nn
from models.cv_extractor import CVExtractor
from models.nlp_extractor import NLPExtractor
from models.fusion_model import FusionMLP


class ViralScopeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cv_extractor = CVExtractor(
            pretrained=config['model']['cv']['pretrained'],
            feature_dim=config['model']['cv']['feature_dim'],
            freeze=config['model']['cv']['freeze_backbone']
        )
        self.nlp_extractor = NLPExtractor(
            checkpoint=config['model']['nlp']['checkpoint'],
            feature_dim=config['model']['nlp']['feature_dim'],
            freeze=config['model']['nlp']['freeze_backbone']
        )
        self.fusion = FusionMLP(
            cv_dim=config['model']['cv']['feature_dim'],
            nlp_dim=config['model']['nlp']['feature_dim'],
            hidden_layers=config['model']['fusion']['hidden_layers'],
            dropout=config['model']['fusion']['dropout'],
            activation=config['model']['fusion']['activation']
        )
    
    def forward(self, images, input_ids, attention_mask):
        """
        Args:
            images: shape (batch, 3, 224, 224)
            input_ids: shape (batch, seq_len)
            attention_mask: shape (batch, seq_len)
        Returns:
            logits: shape (batch,)
        """
        cv_features = self.cv_extractor(images)
        nlp_features = self.nlp_extractor(input_ids, attention_mask)
        logits = self.fusion(cv_features, nlp_features)
        return logits
    
    def predict_proba(self, images, input_ids, attention_mask):
        """Returns probability instead of logit."""
        logits = self.forward(images, input_ids, attention_mask)
        return torch.sigmoid(logits)
