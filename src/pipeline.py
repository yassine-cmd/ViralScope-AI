"""ViralScopePipeline — Orchestrator that ties all modules together."""

import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.config_loader import ConfigLoader
from src.data_loader import DataLoader
from src.label_engine import LabelEngine
from src.thumbnail_manager import ThumbnailManager
from src.embedding_extractor import EmbeddingExtractor
from src.data_splitter import DataSplitter
from src.title_feature_extractor import TitleFeatureExtractor
from src.visual_stats_extractor import VisualStatsExtractor
from src.feature_builder import FeatureBuilder
from src.stacking_trainer import StackingTrainer
from src.model_evaluator import ModelEvaluator
from src.model_persistence import ModelPersistence


class ViralScopePipeline:
    """Orchestrator that runs the full pipeline."""

    def __init__(self, config_path="config.yaml"):
        self.config = None
        self.data_loader = None
        self.label_engine = None
        self.tm = None
        self.extractor = None
        self.splitter = None
        self.title_extractor = None
        self.visual_extractor = None
        self.feature_builder = None
        self.trainer = None
        self.evaluator = None
        self.persistence = None

        self.labeled_df = None
        self.ensemble = None
        self.optimal_thr = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None

    def run(self):
        """Execute full pipeline."""
        loader = ConfigLoader("config.yaml")
        loader.load().validate().create_directories().print_summary()
        self.config = loader.config

        self.data_loader = DataLoader(self.config)
        self.data_loader.load_and_clean()
        clean_df = self.data_loader.clean_df

        self.label_engine = LabelEngine(self.config)
        self.labeled_df = self.label_engine.fit_transform(clean_df, buffer=True)

        self.tm = ThumbnailManager(self.config)
        failed = self.tm.download(self.labeled_df["video_id"].tolist())
        loader.failed_video_ids = failed

        self.labeled_df = self.labeled_df[
            ~self.labeled_df["video_id"].isin(failed)
        ].reset_index(drop=True)
        print(f"[Cap] Rows after removing {len(failed):,} failed thumbnails: {len(self.labeled_df):,}")

        strategy = self.config["data"].get("sampling_strategy", "balanced")

        if strategy == "balanced":
            max_per_class = self.config["data"].get("max_per_class", 999999)
            viral_df = self.labeled_df[self.labeled_df["is_viral"] == 1]
            nonviral_df = self.labeled_df[self.labeled_df["is_viral"] == 0]
            n_final = min(max_per_class, len(viral_df), len(nonviral_df))

            if n_final < max_per_class:
                print(f"[Cap] WARNING: only {n_final:,}/class available (target {max_per_class:,}).")

            self.labeled_df = pd.concat([
                viral_df.sample(n=n_final, random_state=self.config["project"]["seed"]),
                nonviral_df.sample(n=n_final, random_state=self.config["project"]["seed"]),
            ]).sample(frac=1, random_state=self.config["project"]["seed"]).reset_index(drop=True)

            print(f"[Cap] Final dataset : {n_final:,} viral + {n_final:,} non-viral = {len(self.labeled_df):,} total")
        else:
            print(f'[Cap] Strategy is "{strategy}" — no balanced capping applied.')

        self.labeled_df = self.labeled_df.sample(frac=1, random_state=self.config["project"]["seed"]).reset_index(drop=True)
        print("[Cap] Dataset shuffled.")

        self.labeled_df.to_csv(f"{self.config['data']['processed_dir']}/labeled_dataset.csv", index=False)
        print("[Cap] Saved labeled_dataset.csv")

        self.extractor = EmbeddingExtractor(self.config, self.tm)
        img_embs, txt_embs, probe_feats = self.extractor.extract(self.labeled_df)

        vids_out = np.load(self.extractor.ids_path, allow_pickle=True).astype(str)
        id_to_pos = {vid: i for i, vid in enumerate(vids_out)}
        self.labeled_df = self.labeled_df[self.labeled_df["video_id"].isin(id_to_pos)].reset_index(drop=True)
        indices = [id_to_pos[vid] for vid in self.labeled_df["video_id"]]
        img_embs = img_embs[indices]
        txt_embs = txt_embs[indices]
        probe_feats = probe_feats[indices]

        _viral_rate = self.labeled_df["is_viral"].mean()
        print(f"[Sync] Rows: {len(self.labeled_df):,} | viral rate: {_viral_rate:.3f}")
        if self.config["data"].get("sampling_strategy", "balanced") == "balanced":
            assert abs(_viral_rate - 0.5) < 0.05, f"Class balance broken: {_viral_rate:.3f}"

        self.splitter = DataSplitter(self.config)
        self.train_idx, self.val_idx, self.test_idx = self.splitter.split(self.labeled_df)
        self.labeled_df = self.splitter.recompute_labels_train_only(self.labeled_df, self.label_engine)

        self.title_extractor = TitleFeatureExtractor(self.config)
        title_features = None  # Intentionally excluded: introduced noise in ablation tests.
                               # TitleFeatureExtractor is kept for reference / future experiments.

        self.visual_extractor = VisualStatsExtractor(self.tm, resize=160)
        visual_features = self.visual_extractor.compute(self.labeled_df["video_id"].tolist())

        self.feature_builder = FeatureBuilder(self.config)
        X, y = self.feature_builder.build(
            self.labeled_df, img_embs, txt_embs, probe_feats,
            title_features, visual_features
        )

        self.X_train, self.y_train = X.iloc[self.train_idx], y[self.train_idx]
        self.X_val, self.y_val = X.iloc[self.val_idx], y[self.val_idx]
        self.X_test, self.y_test = X.iloc[self.test_idx], y[self.test_idx]

        print(f"[Features] Splits created: Train={len(self.X_train)}, Val={len(self.X_val)}, Test={len(self.X_test)}")

        self.trainer = StackingTrainer(self.config)
        multiplier_train = self.labeled_df.iloc[self.train_idx]["multiplier"].values
        self.ensemble, val_pr_auc, val_f1 = self.trainer.fit(
            self.X_train, self.y_train, multiplier_train, self.X_val, self.y_val, self.X_test, self.y_test
        )

        self.ensemble._iso = IsotonicRegression(out_of_bounds="clip").fit(
            self.ensemble.predict_proba(self.X_val)[:, 1], self.y_val
        )

        self.evaluator = ModelEvaluator(self.config)
        val_probs = self.ensemble.predict_proba(self.X_val)[:, 1]
        self.optimal_thr = self.evaluator.find_optimal_threshold(self.y_val, val_probs)
        test_probs, test_preds_opt, test_preds_fixed = self.evaluator.evaluate_test_set(
            self.ensemble, self.X_test, self.y_test, self.optimal_thr
        )

        self.persistence = ModelPersistence(self.config)
        self.persistence.save_model(self.ensemble, self.optimal_thr)
        self.persistence.save_results(self.y_test, test_probs, test_preds_opt, test_preds_fixed, self.optimal_thr)
        self.persistence.save_training_log(val_pr_auc, val_f1)

        try:
            import google.colab
            self.persistence.download_if_colab()
        except ImportError:
            pass

        import joblib
        joblib.dump(self.feature_builder.get_feature_names(), "models/feature_names.joblib")
        print("[Save] feature_names.joblib saved to models/ for Streamlit app")
        print("[Save] Pipeline complete.")

        return self
