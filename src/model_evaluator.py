"""ModelEvaluator — Threshold tuning, metrics, test set evaluation."""

import numpy as np
from sklearn.metrics import (
    average_precision_score, confusion_matrix,
    f1_score, precision_recall_curve, roc_auc_score
)


class ModelEvaluator:
    """Evaluate model performance with threshold tuning."""

    def __init__(self, config):
        self.config = config

    def find_optimal_threshold(self, y_true, probs):
        """Find threshold that maximizes F1 on validation set."""
        prec, rec, thr = precision_recall_curve(y_true, probs)
        f1s = 2 * prec * rec / (prec + rec + 1e-10)
        best_idx = int(f1s.argmax())
        optimal_thr = float(thr[best_idx]) if best_idx < len(thr) else 0.5

        print(f"[Eval] Optimal threshold (val F1-max): {optimal_thr:.4f}")
        print(f"[Eval] Val F1 at optimal threshold: {f1s[best_idx]:.4f}")

        target_recall = 0.90
        if len(np.where(rec >= target_recall)[0]) > 0:
            p_at_90 = prec[np.where(rec >= target_recall)[0][-1]]
            print(f"[Eval] Precision at 90% Recall: {p_at_90:.4f}")

        return optimal_thr

    def evaluate_test_set(self, ensemble, X_test, y_test, optimal_thr):
        """Evaluate ensemble on test set at optimal and fixed thresholds."""
        test_probs = ensemble.predict_proba(X_test)[:, 1]

        preds_opt = (test_probs >= optimal_thr).astype(int)
        preds_fixed = (test_probs >= 0.5).astype(int)

        print(f"\n{'=' * 60}")
        print("TEST SET RESULTS")
        print(f"{'=' * 60}")

        print(f"\nWith optimal threshold ({optimal_thr:.4f}):")
        print(f"  Accuracy  : {(preds_opt == y_test).mean():.4f}")
        print(f"  F1        : {f1_score(y_test, preds_opt, zero_division=0):.4f}")
        print(f"  AUC-ROC   : {roc_auc_score(y_test, test_probs):.4f}")
        print(f"  PR-AUC    : {average_precision_score(y_test, test_probs):.4f}")
        print(f"  Confusion :\n{confusion_matrix(y_test, preds_opt).tolist()}")

        print(f"\nWith fixed threshold (0.50):")
        print(f"  Accuracy  : {(preds_fixed == y_test).mean():.4f}")
        print(f"  F1        : {f1_score(y_test, preds_fixed, zero_division=0):.4f}")
        print(f"  Confusion :\n{confusion_matrix(y_test, preds_fixed).tolist()}")
        print(f"{'=' * 60}")

        return test_probs, preds_opt, preds_fixed
