"""
XGBoost binary classifier for predictive maintenance.
Predicts whether a miner will fail within the next 24 hours.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

from ..config import MODELS_DIR
from .evaluation import (
    compute_classification_metrics,
    select_threshold_by_recall,
    select_threshold_f1_max,
    select_threshold_precision_floor,
)


class MinerFailureClassifier:
    """XGBoost-based failure prediction for ASIC miners."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        scale_pos_weight: Optional[float] = None,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.threshold_ = 0.5
        self.model_ = None
        self.feature_names_ = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 20,
    ) -> "MinerFailureClassifier":
        """Train the XGBoost model."""
        self.feature_names_ = list(X_train.columns)

        # Auto-compute class weight if not provided.
        #
        # Naive n_neg/n_pos produces absurd values under extreme imbalance.
        # In our case (~0.17% positive) it gives ~580, which destroys
        # probability calibration: the model pushes almost every row above
        # 0.5 and everything becomes "alert". Instead we use sqrt-scaling,
        # which keeps the model sensitive to the minority class without
        # flattening the decision surface, and cap at 50 as a safety belt.
        spw = self.scale_pos_weight
        if spw is None:
            n_neg = int((y_train == 0).sum())
            n_pos = max(int((y_train == 1).sum()), 1)
            raw_ratio = n_neg / n_pos
            spw = float(min(np.sqrt(raw_ratio), 50.0))
            print(
                f"  scale_pos_weight: raw n_neg/n_pos={raw_ratio:.1f} "
                f"-> using sqrt-capped {spw:.1f}"
            )

        self.model_ = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=spw,
            random_state=self.random_state,
            eval_metric="logloss",
            verbosity=0,
            # tree_method="hist" is the histogram-based split finder.
            # For ~2M rows × ~50 features, it's 10-100x faster than the
            # default "exact" (ColMaker) method and has near-identical
            # accuracy. Without this, step 7 takes 30-60+ minutes on CPU
            # and was the real reason previous runs appeared "frozen".
            tree_method="hist",
            n_jobs=-1,
        )

        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model_.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )

        print(f"Trained XGBoost: {self.n_estimators} trees, "
              f"scale_pos_weight={spw:.1f}, "
              f"features={len(self.feature_names_)}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Binary predictions using stored threshold."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold_).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Raw probability predictions. Shape (n, 2)."""
        return self.model_.predict_proba(X[self.feature_names_])

    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, float]:
        """Run prediction and return metrics."""
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)[:, 1]
        metrics = compute_classification_metrics(y_test, y_pred, y_prob)
        print(f"Evaluation: precision={metrics['precision']:.3f} "
              f"recall={metrics['recall']:.3f} "
              f"f1={metrics['f1']:.3f} "
              f"auc_roc={metrics.get('auc_roc', 0):.3f}")
        return metrics

    def get_feature_importance(
        self,
        importance_type: str = "gain",
        top_n: int = 20,
    ) -> pd.DataFrame:
        """Feature importance ranked by gain."""
        importance = self.model_.get_booster().get_score(
            importance_type=importance_type
        )
        df = pd.DataFrame([
            {"feature": k, "importance": v}
            for k, v in importance.items()
        ]).sort_values("importance", ascending=False)
        return df.head(top_n).reset_index(drop=True)

    def optimize_threshold(
        self,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        strategy: str = "f1_max",
        target_recall: float = 0.85,
        min_precision: float = 0.10,
    ) -> float:
        """
        Pick a decision threshold from validation scores.

        strategy="f1_max" (default, recommended): maximize F1. Balances
            precision and recall. Won't collapse to 0 under extreme imbalance.

        strategy="precision_floor": pick the lowest threshold that still
            achieves min_precision, maximizing recall subject to that floor.
            Use when you have a hard false-alarm budget.

        strategy="recall_target": legacy behavior — highest threshold still
            achieving target_recall. Use only when false negatives are far
            more expensive than false positives AND the model is well-
            calibrated; otherwise it degenerates under class imbalance.
        """
        y_prob = self.predict_proba(X_val)[:, 1]
        if strategy == "f1_max":
            self.threshold_ = select_threshold_f1_max(y_val, y_prob)
            detail = "maximize F1"
        elif strategy == "precision_floor":
            self.threshold_ = select_threshold_precision_floor(
                y_val, y_prob, min_precision=min_precision,
            )
            detail = f"precision >= {min_precision}"
        elif strategy == "recall_target":
            self.threshold_ = select_threshold_by_recall(
                y_val, y_prob, target_recall=target_recall,
            )
            detail = f"recall >= {target_recall}"
        else:
            raise ValueError(
                f"Unknown strategy {strategy!r}; use f1_max, precision_floor, or recall_target"
            )
        print(
            f"  Threshold ({strategy}, {detail}): {self.threshold_:.4f} "
            f"[prob range: {y_prob.min():.4f}..{y_prob.max():.4f}]"
        )
        return self.threshold_

    def save(self, path: Optional[Path] = None) -> Path:
        if path is None:
            path = MODELS_DIR / "xgboost_failure.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model_,
            "threshold": self.threshold_,
            "feature_names": self.feature_names_,
        }, path)
        print(f"Saved model to {path}")
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "MinerFailureClassifier":
        if path is None:
            path = MODELS_DIR / "xgboost_failure.joblib"
        data = joblib.load(path)
        instance = cls()
        instance.model_ = data["model"]
        instance.threshold_ = data["threshold"]
        instance.feature_names_ = data["feature_names"]
        print(f"Loaded model from {path}")
        return instance
