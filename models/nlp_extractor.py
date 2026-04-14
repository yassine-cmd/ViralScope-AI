import torch
import torch.nn as nn


class NLPExtractor(nn.Module):
    def __init__(self, checkpoint='distilbert-base-uncased', feature_dim=768, freeze=True):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        return torch.randn(batch_size, self.feature_dim)
