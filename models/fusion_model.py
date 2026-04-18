import torch
import torch.nn as nn


class FusionMLP(nn.Module):
    """Cross-modal fusion head using CLIP's shared latent space.

    Input: 512 (img) + 512 (txt) + 1 (cos_sim) = 1025
    Note: Removed redundant |img-txt| and img*txt features that add no 
    independent information and increase overfitting risk on small datasets.
    """

    def __init__(self, feature_dim=512, hidden_layers=None, dropout=0.2, activation="GELU"):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [256, 64]

        activation_fn = getattr(nn, activation)

        input_dim = 2 * feature_dim + 1  # img + txt + cos_sim

        layers = []
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation_fn())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, img_emb, txt_emb):
        """
        Args:
            img_emb: L2-normalized CLIP image embeddings, shape (batch, 512)
            txt_emb: L2-normalized CLIP text embeddings,  shape (batch, 512)
        Returns:
            logits: shape (batch,)
        """
        cos_sim = (img_emb * txt_emb).sum(dim=1, keepdim=True)

        combined = torch.cat([img_emb, txt_emb, cos_sim], dim=1)

        return self.network(combined).squeeze(-1)
