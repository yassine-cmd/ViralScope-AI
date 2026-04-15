import torch
import torch.nn as nn
from transformers import DistilBertModel


class NLPExtractor(nn.Module):
    def __init__(self, checkpoint="distilbert-base-uncased", feature_dim=768, freeze=True):
        super().__init__()
        self.backbone = DistilBertModel.from_pretrained(checkpoint)
        
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: shape (batch, seq_len)
            attention_mask: shape (batch, seq_len)
        Returns:
            cls_features: shape (batch, 768)
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_features = outputs.last_hidden_state[:, 0, :]
        return cls_features
