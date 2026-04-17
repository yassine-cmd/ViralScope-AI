import torch
import torch.nn as nn


class FusionMLP(nn.Module):
    """Cross-modal fusion head exploiting CLIP's shared latent space.

    Instead of naively concatenating image and text embeddings, this module
    computes explicit interaction features (element-wise difference, product,
    and cosine similarity) that help the MLP learn alignment and contradiction
    patterns between the thumbnail and title — a core virality signal.

    Input dimensionality:
        512 (img) + 512 (txt) + 512 (|diff|) + 512 (prod) + 1 (cos_sim) = 2049
    """

    def __init__(self, feature_dim=512, hidden_layers=None, dropout=0.2, activation="GELU"):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [256, 64]

        activation_fn = getattr(nn, activation)

        # 4 * feature_dim (raw + interactions) + 1 (cosine similarity)
        input_dim = 4 * feature_dim + 1

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
        # Cross-modal interaction features
        cos_sim = (img_emb * txt_emb).sum(dim=1, keepdim=True)   # (batch, 1)
        element_diff = torch.abs(img_emb - txt_emb)              # (batch, 512)
        element_prod = img_emb * txt_emb                         # (batch, 512)

        combined = torch.cat(
            [img_emb, txt_emb, element_diff, element_prod, cos_sim], dim=1
        )  # (batch, 2049)

        return self.network(combined).squeeze(-1)
