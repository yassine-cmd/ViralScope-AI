"""FeatureBuilder — Fuse embeddings + probes + title + visual → X_df."""

import numpy as np
import pandas as pd

from src.visual_stats_extractor import VisualStatsExtractor


class FeatureBuilder:
    """Fuse all feature sources into a single DataFrame."""

    def __init__(self, config):
        self.config = config
        self.emb_dim = config["model"]["embedding"]["image_dim"]
        self.feature_cols = None

    def build(self, labeled_df, img_embs, txt_embs, probe_feats,
             title_features, visual_features):
        """Build fused DataFrame X with named columns."""
        assert len(labeled_df) == len(img_embs) == len(txt_embs) == len(probe_feats)

        img_norm = img_embs / (np.linalg.norm(img_embs, axis=1, keepdims=True) + 1e-5)
        txt_norm = txt_embs / (np.linalg.norm(txt_embs, axis=1, keepdims=True) + 1e-5)

        diff = np.abs(img_norm - txt_norm)
        prod = img_norm * txt_norm
        cos_sim = (img_norm * txt_norm).sum(axis=1, keepdims=True)

        n_emb = img_norm.shape[1]
        emb_cols = [f"emb_img_{i}" for i in range(n_emb)]
        txt_cols = [f"emb_txt_{i}" for i in range(n_emb)]
        diff_cols = [f"emb_diff_{i}" for i in range(n_emb)]
        prod_cols = [f"emb_prod_{i}" for i in range(n_emb)]

        emb_df = pd.DataFrame(
            np.hstack([img_norm, txt_norm, diff, prod, cos_sim]),
            columns=emb_cols + txt_cols + diff_cols + prod_cols + ["cos_sim"]
        )
        print(f"[Features] Embedding features: {emb_df.shape}")

        probe_cols = [f"probe_{i}" for i in range(probe_feats.shape[1])]
        probe_df = pd.DataFrame(probe_feats, columns=probe_cols)

        visual_df = pd.DataFrame(visual_features, columns=VisualStatsExtractor.COLUMNS)

        df = labeled_df.copy()
        df["channel_log_power"] = np.log10(df["channel_ref_views"] + 1)

        cat_dummies = pd.get_dummies(df["category_id"], prefix="cat")
        if cat_dummies.shape[1] > 20:
            cat_dummies = cat_dummies.iloc[:, :20]

        tab_df = pd.DataFrame({
            "hour_of_day": df["hour_of_day"].values,
            "day_of_week": df["day_of_week"].values,
            "is_weekend": df["is_weekend"].values,
        })

        probe_sum = probe_feats.sum(axis=1, keepdims=True)
        df["probe_x_channel"] = probe_sum.flatten() * df["channel_log_power"].values
        df["cos_x_weekend"] = cos_sim.flatten() * df["is_weekend"].values

        interaction_df = pd.DataFrame({
            "channel_log_power": df["channel_log_power"].values,
            "probe_x_channel": df["probe_x_channel"].values,
            "cos_x_weekend": df["cos_x_weekend"].values,
        })

        X_df = pd.concat([
            emb_df, probe_df, visual_df,
            cat_dummies.reset_index(drop=True),
            tab_df, interaction_df
        ], axis=1)

        self.feature_cols = X_df.columns.tolist()
        print(f"[Features] Total DataFrame: {X_df.shape}")

        y = df["is_viral"].values.astype(np.float32)
        return X_df, y

    def get_feature_names(self):
        return self.feature_cols
