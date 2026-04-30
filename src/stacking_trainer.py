"""StackingTrainer — 5-fold OOF stacking ensemble training."""

import numpy as np
import pandas as pd
from sklearn.base import clone as sklearn_clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score


class SoftVotingEnsemble:
    """Weighted soft-voting combiner.

    Weights are each base learner's OOF PR-AUC, normalised to sum to 1.
    """

    def __init__(self, base_learners, weights, feature_names=None):
        self.base_learners = base_learners
        self.weights = weights
        self.feature_names = feature_names
        self.learner_names = list(base_learners.keys())

    def _to_df(self, X):
        if not isinstance(X, pd.DataFrame) and self.feature_names is not None:
            return pd.DataFrame(X, columns=self.feature_names)
        return X

    def predict_proba(self, X):
        X = self._to_df(X)
        prob1 = sum(
            self.weights[n] * self.base_learners[n].predict_proba(X)[:, 1]
            for n in self.learner_names
        )
        prob1 = np.clip(prob1, 0.0, 1.0)
        if hasattr(self, "_iso"):
            prob1 = self._iso.predict(prob1)
        return np.column_stack([1 - prob1, prob1])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


class StackingTrainer:
    """5-fold OOF trainer for stacking ensemble."""

    def __init__(self, config, use_meta_learner=False):
        self.config = config
        self.seed = config["project"]["seed"]
        self.n_folds = config["model"]["stacking"]["n_folds"]
        self.use_meta_learner = use_meta_learner
        self.trained_base = {}
        self.fold_metrics = {}
        self.oof_prauc = {}
        self.weights = {}
        self.ensemble = None
        self.feature_names = None

    def _get_base_learners(self, n_features, y_train):
        cfg_emb = self.config["model"]["embedding"]
        emb_dims = cfg_emb["image_dim"] * 4 + 1
        emb_cols = list(range(emb_dims))
        tab_cols = list(range(emb_dims, n_features))

        spw = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)

        full_scale_prep = ColumnTransformer([
            ("scale_embs", StandardScaler(), emb_cols),
            ("scale_tab", StandardScaler(), tab_cols),
        ], verbose_feature_names_out=False)

        pca_prep = ColumnTransformer([
            ("pca_embs", PCA(n_components=64, random_state=self.seed), emb_cols),
            ("pass_tab", "passthrough", tab_cols),
        ], verbose_feature_names_out=False)

        return {
            "logreg": Pipeline([
                ("prep", full_scale_prep),
                ("clf", LogisticRegression(
                    C=1.0, max_iter=2000,
                    class_weight="balanced",
                    random_state=self.seed,
                )),
            ]),
            "xgb": Pipeline([
                ("prep", pca_prep),
                ("clf", XGBClassifier(
                    n_estimators=500, max_depth=6, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=spw,
                    random_state=self.seed,
                    eval_metric="aucpr",
                )),
            ]),
            "lgbm": Pipeline([
                ("prep", pca_prep),
                ("clf", LGBMClassifier(
                    n_estimators=500, learning_rate=0.03,
                    num_leaves=63,
                    scale_pos_weight=spw,
                    random_state=self.seed, verbose=-1,
                )),
            ]),
            "rf": Pipeline([
                ("prep", full_scale_prep),
                ("clf", RandomForestClassifier(
                    n_estimators=300, max_depth=10, max_features="sqrt",
                    class_weight="balanced",
                    random_state=self.seed, n_jobs=-1,
                )),
            ]),
        }

    def _train_single(self, name, learner, X_tr, y_tr, sw_tr, X_v, X_t, kf):
        oof = np.zeros(len(X_tr))
        val_p = np.zeros(len(X_v))
        test_p = np.zeros(len(X_t))
        fold_f1s = []

        for fold, (tr_idx, oof_idx) in enumerate(kf.split(X_tr, y_tr)):
            m = sklearn_clone(learner)
            m.fit(X_tr.iloc[tr_idx], y_tr[tr_idx],
                  clf__sample_weight=sw_tr[tr_idx])
            oof[oof_idx] = m.predict_proba(X_tr.iloc[oof_idx])[:, 1]
            fold_f1s.append(f1_score(y_tr[oof_idx],
                                     (oof[oof_idx] >= 0.5).astype(int),
                                     zero_division=0))
            val_p += m.predict_proba(X_v)[:, 1] / kf.n_splits
            test_p += m.predict_proba(X_t)[:, 1] / kf.n_splits

        final = sklearn_clone(learner)
        final.fit(X_tr, y_tr, clf__sample_weight=sw_tr)
        return name, oof, val_p, test_p, final, fold_f1s

    def fit(self, X_train, y_train, multiplier_train, X_val, y_val, X_test, y_test):
        self.feature_names = list(X_train.columns) if hasattr(X_train, "columns") else None
        sample_weights = np.log1p(multiplier_train).astype(np.float32)

        n_features = X_train.shape[1]
        base_learners = self._get_base_learners(n_features, y_train)
        n_base = len(base_learners)

        oof_preds = np.zeros((len(X_train), n_base), dtype=np.float32)
        val_preds = np.zeros((len(X_val), n_base), dtype=np.float32)
        test_preds = np.zeros((len(X_test), n_base), dtype=np.float32)

        kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                             random_state=self.seed)

        print(f"[Stack] Training {n_base} base learners with "
              f"{self.n_folds}-fold OOF (shuffle=True)...\n")

        results = Parallel(n_jobs=-1, verbose=0)(
            delayed(self._train_single)(
                name, learner, X_train, y_train, sample_weights, X_val, X_test, kf
            )
            for name, learner in base_learners.items()
        )

        trained_base = {}
        for li, (name, oof, val_p, test_p, model, fold_f1s) in enumerate(results):
            oof_preds[:, li] = oof
            val_preds[:, li] = val_p
            test_preds[:, li] = test_p
            trained_base[name] = model
            oof_pr = average_precision_score(y_train, oof)
            self.oof_prauc[name] = oof_pr
            self.fold_metrics[name] = fold_f1s
            print(f"  {name:8s}  OOF PR-AUC: {oof_pr:.4f}  "
                  f"fold F1s: {[round(f, 3) for f in fold_f1s]}")

        self.trained_base = trained_base

        raw_w = np.array([self.oof_prauc[n] for n in trained_base], dtype=np.float64)
        raw_w = np.clip(raw_w, 0, None)
        norm_w = raw_w / raw_w.sum()
        self.weights = {n: float(w) for n, w in zip(trained_base, norm_w)}

        print("\n[Stack] Soft-voting weights (OOF PR-AUC normalised):")
        for n, w in self.weights.items():
            print(f"  {n:8s}  {w:.4f}")

        val_soft = sum(
            self.weights[n] * val_preds[:, li]
            for li, n in enumerate(trained_base)
        )
        val_pr_auc = average_precision_score(y_val, val_soft)
        prec, rec, thr = precision_recall_curve(y_val, val_soft)
        f1s = 2 * prec * rec / (prec + rec + 1e-10)
        opt_idx = int(f1s.argmax())
        opt_thr = float(thr[opt_idx]) if opt_idx < len(thr) else 0.5
        val_f1 = f1_score(y_val, (val_soft >= opt_thr).astype(int), zero_division=0)
        print(f"\n[Stack] Soft-vote val  PR-AUC: {val_pr_auc:.4f}  "
              f"F1: {val_f1:.4f}  (thr={opt_thr:.3f})")

        if self.use_meta_learner:
            meta_scaler = StandardScaler()
            meta_train = meta_scaler.fit_transform(oof_preds)
            meta_val = meta_scaler.transform(val_preds)
            meta_clf = LogisticRegression(
                C=0.1, class_weight=None, solver="lbfgs",
                max_iter=1000, random_state=self.seed,
            )
            meta_clf.fit(meta_train, y_train)
            val_meta = meta_clf.predict_proba(meta_val)[:, 1]
            meta_pr = average_precision_score(y_val, val_meta)
            print(f"[Stack] LogReg meta  val PR-AUC: {meta_pr:.4f}  "
                  f"({'soft vote wins' if val_pr_auc >= meta_pr else 'meta wins'})")

        self.ensemble = SoftVotingEnsemble(
            base_learners=trained_base,
            weights=self.weights,
            feature_names=self.feature_names,
        )
        print("\n[Stack] Ensemble ready.")
        return self.ensemble, val_pr_auc, val_f1

    def plot_oof_curves(self, save_path="results/oof_fold_f1.pdf"):
        import matplotlib.pyplot as plt
        if not self.fold_metrics:
            print("[Plot] No data — run fit() first."); return
        learners = list(self.fold_metrics.keys())
        n_folds = len(next(iter(self.fold_metrics.values())))
        folds = list(range(1, n_folds + 1))
        fig, ax = plt.subplots(figsize=(8, 4))
        for name in learners:
            ax.plot(folds, self.fold_metrics[name], marker="o", lw=2,
                    label=f"{name} (w={self.weights.get(name, 0):.3f})")
        ax.set_title("F1 per OOF fold — base learners", fontsize=13, fontweight="bold")
        ax.set_xlabel("Fold"); ax.set_ylabel("F1 Score (OOF)")
        ax.set_xticks(folds); ax.legend(fontsize=9); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.show()
        print(f"[Plot] Saved → {save_path}")
        return self
