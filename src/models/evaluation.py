"""
Shared evaluation utilities for all models.
No dependency on other src/models/ files to avoid circular imports.
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute standard classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
        metrics["auc_pr"] = average_precision_score(y_true, y_prob)

    return metrics


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


def select_threshold_by_recall(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    target_recall: float = 0.95,
) -> float:
    """
    Find HIGHEST decision threshold still achieving at least target_recall.
    Useful when the caller cares about guaranteed coverage and can tolerate
    low precision. In heavily imbalanced settings this can collapse to 0;
    prefer select_threshold_f1_max or select_threshold_precision_floor.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    valid = recall[:-1] >= target_recall
    if valid.any():
        # thresholds is sorted ascending → take the last valid = highest threshold
        return float(thresholds[valid][-1])
    return 0.5


def select_threshold_f1_max(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> float:
    """
    Pick the threshold that maximizes F1 on the given validation scores.
    This is the right default for imbalanced classification: it balances
    precision and recall and won't collapse to 0 like recall-only targeting.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    # precision_recall_curve returns arrays of length N+1 for precision/recall
    # and N for thresholds. Drop the final (precision=1, recall=0) sentinel.
    p, r = precision[:-1], recall[:-1]
    denom = p + r
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denom > 0, 2 * p * r / denom, 0.0)
    if len(f1) == 0 or not np.isfinite(f1).any():
        return 0.5
    best_idx = int(np.nanargmax(f1))
    return float(thresholds[best_idx])


def select_threshold_precision_floor(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    min_precision: float = 0.10,
) -> float:
    """
    Pick the LOWEST threshold that achieves at least min_precision. Among
    all thresholds meeting the floor, this maximizes recall. Good for
    operator-facing alerts where a precision SLO matters (e.g. "< 1 false
    alarm per 10 correct detections" → min_precision=0.1).
    Falls back to F1-max if no threshold can hit the floor.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    p, r = precision[:-1], recall[:-1]
    valid = p >= min_precision
    if not valid.any():
        return select_threshold_f1_max(y_true, y_scores)
    # thresholds is ascending → first valid = lowest threshold meeting floor
    first_idx = int(np.argmax(valid))
    return float(thresholds[first_idx])


def select_threshold_f1_with_floor(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    min_precision: float = 0.05,
) -> float:
    """
    F1-max with a precision-floor safety net.

    Try F1-max first. If the resulting precision on y_true/y_scores falls
    below min_precision, fall back to precision-floor selection instead.

    This exists because F1-max can mathematically "collapse" to a very
    low threshold under severe class imbalance. The collapse isn't a
    bug — when the positive-class probability distribution is compressed
    toward zero, "flag everything" really does maximize F1 on paper —
    but it produces an alert rule that cries wolf on every row and is
    useless operationally. The precision floor gives the operator a
    principled fallback they can reason about: "never below this
    false-alarm rate".

    Default min_precision=0.05 translates to "no more than 19 false
    alarms per correct detection". Reasonable for predictive
    maintenance where an operator will triage the flagged window, not
    auto-shutdown the miner.
    """
    f1_thresh = select_threshold_f1_max(y_true, y_scores)
    y_pred = (y_scores >= f1_thresh).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    precision_at_f1 = tp / max(tp + fp, 1)
    if precision_at_f1 >= min_precision:
        return f1_thresh
    return select_threshold_precision_floor(y_true, y_scores, min_precision)


def select_anomaly_threshold(
    reconstruction_errors: np.ndarray,
    percentile: float = 95.0,
) -> float:
    """Set anomaly threshold at given percentile of healthy reconstruction errors."""
    return float(np.percentile(reconstruction_errors, percentile))


def compare_models(metrics_a: dict, metrics_b: dict, names: tuple = ("XGBoost", "LSTM-AE")) -> pd.DataFrame:
    """Side-by-side model comparison."""
    all_keys = sorted(set(list(metrics_a.keys()) + list(metrics_b.keys())))
    return pd.DataFrame({
        names[0]: [metrics_a.get(k, np.nan) for k in all_keys],
        names[1]: [metrics_b.get(k, np.nan) for k in all_keys],
    }, index=all_keys)


def detection_timeline(
    df: pd.DataFrame,
    predictions: np.ndarray,
    timestamp_col: str = "timestamp",
    miner_col: str = "miner_id",
    failure_col: str = "failure_type",
    label_col: str = "is_pre_failure",
) -> pd.DataFrame:
    """
    Per-failure lead-time analysis.

    For each distinct pre-failure window in the data (one per failing miner),
    report whether the model raised a flag at any point inside the window,
    and how many minutes of lead time were won (time between first flag and
    end of window = start of cascade).

    This is the RIGHT granularity for predictive maintenance: dashboards and
    operators care about "did we catch failure X with enough runway", not
    "what fraction of rows got labeled correctly".

    Previous implementation used failure_type != "none" as the onset anchor,
    which mislabeled failures whose pre-failure window straddled a temporal
    train/test split as "missed" even when the model had flagged every row
    correctly. That bug is fixed here by using is_pre_failure directly.
    """
    df = df.copy()
    df["predicted"] = np.asarray(predictions).astype(int)

    results = []
    for miner_id in df[miner_col].unique():
        miner_df = df[df[miner_col] == miner_id].sort_values(timestamp_col)

        pf_rows = miner_df[miner_df[label_col].astype(bool)]
        if len(pf_rows) == 0:
            continue

        window_start = pf_rows.iloc[0][timestamp_col]
        window_end = pf_rows.iloc[-1][timestamp_col]
        failure_type = pf_rows.iloc[0][failure_col]
        if failure_type == "none":
            # Fallback — some pre-failure rows may still carry "none" if the
            # generator labels the window slightly before failure_type flips.
            later = miner_df[miner_df[failure_col] != "none"]
            if len(later) > 0:
                failure_type = later.iloc[0][failure_col]

        flagged = pf_rows[pf_rows["predicted"] == 1]
        if len(flagged) > 0:
            first_flag = flagged.iloc[0][timestamp_col]
            lead_time_minutes = (window_end - first_flag).total_seconds() / 60
            detected = True
        else:
            first_flag = None
            lead_time_minutes = 0.0
            detected = False

        results.append({
            "miner_id": miner_id,
            "failure_type": failure_type,
            "window_start": window_start,
            "window_end": window_end,
            "first_flag": first_flag,
            "lead_time_minutes": lead_time_minutes,
            "lead_time_hours": lead_time_minutes / 60,
            "detected": detected,
            "n_pre_failure_rows": len(pf_rows),
            "n_flagged_in_window": len(flagged),
        })

    return pd.DataFrame(results)
