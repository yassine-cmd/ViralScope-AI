"""ModelPersistence — Save/load models, results CSV."""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, f1_score, roc_auc_score
)


class ModelPersistence:
    """Save and load models, results, and generate reports."""

    def __init__(self, config):
        self.config = config
        self.model_path = config["paths"]["best_model"]
        self.log_path = config["paths"]["training_log"]
        self.results_path = f"{config['paths']['results']}/eval_results.csv"

    def save_model(self, ensemble, threshold):
        """Save ensemble + threshold together."""
        joblib.dump({"model": ensemble, "threshold": threshold}, self.model_path)
        print(f"[Save] Model + threshold -> {os.path.abspath(self.model_path)}")

    def save_results(self, y_test, test_probs, test_preds_opt, test_preds_fixed, optimal_thr):
        """Save test results to CSV."""
        results = pd.DataFrame([{
            "classifier": "stacked_ensemble",
            "optimal_thr": round(optimal_thr, 4),
            "test_pr_auc": round(average_precision_score(y_test, test_probs), 4),
            "test_auc_roc": round(roc_auc_score(y_test, test_probs), 4),
            "test_f1_optimal": round(f1_score(y_test, test_preds_opt, zero_division=0), 4),
            "test_f1_fixed": round(f1_score(y_test, test_preds_fixed, zero_division=0), 4),
            "test_accuracy": round((test_preds_opt == y_test).mean(), 4),
        }])
        results.to_csv(self.results_path, index=False)
        print(f"[Save] Test results -> {os.path.abspath(self.results_path)}")

    def save_training_log(self, val_pr_auc, val_f1):
        """Save classifier summary."""
        summary = pd.DataFrame([{
            "classifier": "stacked_ensemble",
            "val_pr_auc": round(val_pr_auc, 4),
            "val_f1": round(val_f1, 4),
            "selected": True,
        }])
        summary.to_csv(self.log_path, index=False)
        print(f"[Save] Classifier summary -> {os.path.abspath(self.log_path)}")

    def download_if_colab(self):
        """Auto-download outputs on Colab."""
        try:
            import google.colab
            from google.colab import files as _cf
            for p in [self.model_path, self.log_path, self.results_path]:
                if os.path.exists(p):
                    _cf.download(p)
        except ImportError:
            pass

    @staticmethod
    def load_model(model_path):
        """Load ensemble + threshold from disk."""
        data = joblib.load(model_path)
        return data["model"], data["threshold"]
