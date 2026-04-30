"""DataSplitter — Random stratified train/val/test splits."""

import numpy as np
from sklearn.model_selection import train_test_split


class DataSplitter:
    """Random stratified train/val/test splits."""

    def __init__(self, config):
        self.config = config
        self.seed = config["project"]["seed"]
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None

    def split(self, df):
        """Create random stratified splits."""
        indices = np.arange(len(df))
        y = df["is_viral"].values

        train_idx, temp_idx = train_test_split(
            indices, test_size=0.3, stratify=y, random_state=self.seed, shuffle=True
        )

        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, stratify=y[temp_idx], random_state=self.seed, shuffle=True
        )

        self.train_idx, self.val_idx, self.test_idx = train_idx, val_idx, test_idx

        print("[Split] Random Stratified Split Applied.")
        print(f"[Split] Train: {len(self.train_idx):,}  |  Val: {len(self.val_idx):,}  |  Test: {len(self.test_idx):,}")
        return self.train_idx, self.val_idx, self.test_idx

    def recompute_labels_train_only(self, df, label_engine):
        """Recompute LOO labels using ONLY training data stats to prevent leakage."""
        df_new = label_engine.compute_labels(df, train_idx=self.train_idx)
        print("[Split] Labels re-computed with train-only channel stats (LOO fix applied)")

        y = df_new["is_viral"].values
        for name, idx in [("Train", self.train_idx), ("Val", self.val_idx), ("Test", self.test_idx)]:
            rate = y[idx].mean()
            print(f"[Split] {name}: {len(idx):,} rows | viral rate: {rate:.3f}")

        print(f"[Split] Final class counts:")
        print(f"[Split]   Viral     : {int(df_new['is_viral'].sum()):,}")
        print(f"[Split]   Non-viral : {int((df_new['is_viral']==0).sum()):,}")
        print(f"[Split]   Total     : {len(df_new):,}")

        df_new.to_csv(f"{self.config['data']['processed_dir']}/labeled_dataset.csv", index=False)
        return df_new
