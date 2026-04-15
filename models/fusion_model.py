import torch
import torch.nn as nn


class FusionMLP(nn.Module):
    def __init__(self, cv_dim=1280, nlp_dim=768, hidden_layers=[512, 128], dropout=0.4, activation='ReLU'):
        super().__init__()

        activation_fn = getattr(nn, activation)

        layers = []
        input_dim = cv_dim + nlp_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation_fn())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, cv_features, nlp_features):
        """
        Args:
            cv_features: shape (batch, 1280)
            nlp_features: shape (batch, 768)
        Returns:
            logits: shape (batch,)
        """
        combined = torch.cat([cv_features, nlp_features], dim=1)
        return self.network(combined).squeeze(-1)
