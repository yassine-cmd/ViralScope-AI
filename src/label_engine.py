"""LabelEngine — Compute LOO viral labels with proper train/test leakage prevention."""

import numpy as np
import pandas as pd


class LabelEngine:
    """Compute LOO viral labels with proper train/test leakage prevention."""

    def __init__(self, config):
        self.config = config
        self.seed = config["project"]["seed"]
        self.labeled_df = None

    def compute_labels(self, df, train_idx=None):
        """LOO viral labelling — train-only stats to prevent leakage.

        When train_idx is supplied, channel means are computed from
        training rows only. Val/test rows use the plain training mean.
        When called without train_idx, all rows act as training data.
        """
        df_for_stats = df.iloc[train_idx].copy() if train_idx is not None else df.copy()
        train_stats = (df_for_stats.groupby("channel_id")["views"]
                         .agg(["sum", "count", "mean"]).reset_index())
        train_stats.columns = ["channel_id", "channel_sum", "video_count", "channel_avg"]
        global_mean = df_for_stats["views"].mean()

        result = df.merge(train_stats, on="channel_id", how="left").reset_index(drop=True)

        is_train = (result.index.isin(train_idx)
                     if train_idx is not None
                     else pd.Series([True] * len(result), index=result.index))

        result["channel_ref_views"] = global_mean
        reliable = result["video_count"].fillna(0) >= 2

        loo = is_train & reliable
        result.loc[loo, "channel_ref_views"] = (
            (result.loc[loo, "channel_sum"] - result.loc[loo, "views"]) /
            (result.loc[loo, "video_count"] - 1).clip(lower=1))

        vt = (~is_train) & reliable
        result.loc[vt, "channel_ref_views"] = result.loc[vt, "channel_avg"]

        result["multiplier"] = result["views"] / (result["channel_ref_views"] + 1e-5)
        result["log_multiplier"] = np.log1p(result["multiplier"])
        thresh = self.config["data"]["target_threshold"]
        result["is_viral"] = (result["multiplier"] > thresh).astype(float)

        minority = min(result["is_viral"].mean(), 1 - result["is_viral"].mean())
        if minority < 0.10:
            raise ValueError(
                "[Labels] Class imbalance < 10% detected. "
                "Update config.yaml: set target_threshold explicitly (e.g., 1.2) "
                "or increase max_per_class to improve label balance."
            )

        cols = ["video_id", "title", "views", "channel_ref_views",
                "multiplier", "is_viral", "channel_id",
                "category_id", "hour_of_day", "day_of_week", "is_weekend"]
        return result[[c for c in cols if c in result.columns]].reset_index(drop=True)

    def apply_sampling_strategy(self, df, buffer=False):
        """Apply sampling strategy from config: balanced/imbalanced/all."""
        strategy = self.config["data"].get("sampling_strategy", "imbalanced")

        if strategy == "all":
            df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
            print(f'[Labels] Strategy: "all" | Total rows: {len(df):,}')
            return df

        if strategy == "imbalanced":
            viral_df = df[df["is_viral"] == 1].copy()
            viral_channels = viral_df["channel_id"].unique()
            neg_ratio = self.config["data"].get("negative_ratio", 3)

            potential_hard_negs = df[(df["is_viral"] == 0) & (df["channel_id"].isin(viral_channels))].copy()
            other_negs = df[(df["is_viral"] == 0) & (~df["channel_id"].isin(viral_channels))].copy()

            n_neg_needed = int(len(viral_df) * neg_ratio)
            hard_negatives = potential_hard_negs.sample(n=min(len(potential_hard_negs), n_neg_needed), random_state=self.seed)
            remaining = max(0, n_neg_needed - len(hard_negatives))

            if remaining > 0 and len(other_negs) > 0:
                fill = other_negs.sample(n=min(remaining, len(other_negs)), random_state=self.seed)
                neg_samples = pd.concat([hard_negatives, fill])
            else:
                neg_samples = hard_negatives

            if len(neg_samples) < n_neg_needed:
                print(f"[Labels] WARNING: only {len(neg_samples):,} non-viral samples available for target ratio {neg_ratio}.")

            neg_samples = neg_samples.sample(n=min(len(neg_samples), n_neg_needed), random_state=self.seed)
            df = pd.concat([viral_df, neg_samples]).sample(frac=1, random_state=self.seed).reset_index(drop=True)
            print(f'[Labels] Strategy: "Hard Negatives" | Viral: {len(viral_df):,} | Non-Viral: {len(neg_samples):,}')
            return df

        if buffer:
            max_per_class = int(self.config["data"].get("max_per_class", 1000) *
                              self.config["data"].get("buffer_multiplier", 1.5))
        else:
            max_per_class = self.config["data"].get("max_per_class", 1000)

        viral_df = df[df["is_viral"] == 1]
        nonviral_df = df[df["is_viral"] == 0]
        n_available = min(max_per_class, len(viral_df), len(nonviral_df))

        if n_available < max_per_class:
            print(f"[Labels] WARNING: only {n_available:,}/class available (target {max_per_class:,}).")

        df = pd.concat([
            viral_df.sample(n=n_available, random_state=self.seed),
            nonviral_df.sample(n=n_available, random_state=self.seed),
        ]).sample(frac=1, random_state=self.seed).reset_index(drop=True)

        print(f'[Labels] Strategy: "balanced" | {n_available:,}/class x 2 = {len(df):,} rows')
        return df

    def fit_transform(self, df, buffer=True):
        """Full labeling pipeline: compute labels then apply sampling."""
        self.labeled_df = self.compute_labels(df)
        print(f"[Labels] Full dataset labelled: {len(self.labeled_df):,} rows")
        print(f'[Labels]   Viral     : {int(self.labeled_df["is_viral"].sum()):,}')
        print(f'[Labels]   Non-viral : {int((self.labeled_df["is_viral"]==0).sum()):,}')

        self.labeled_df = self.apply_sampling_strategy(self.labeled_df, buffer=buffer)

        self.labeled_df.to_csv(f"{self.config['data']['processed_dir']}/labeled_dataset.csv", index=False)
        print("[Labels] Saved labeled_dataset.csv")
        return self.labeled_df
