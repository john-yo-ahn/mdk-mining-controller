"""
Model metadata sidecar.

Writes a .metadata.json file next to each saved model recording the
git commit, training date, dataset size, validation metrics, feature
hash, and training hyperparameters. Purely informational — nothing
depends on the sidecar at load time — but it makes every saved model
reproducible and auditable without requiring the caller to remember
context.

Usage:
    from src.models.metadata import save_model_metadata
    save_model_metadata(
        model_path=MODELS_DIR / "xgboost_failure.joblib",
        model_type="xgboost_binary",
        n_train_rows=1_673_280,
        n_val_rows=270_540,
        val_metrics={"auc_roc": 0.801, "f1": 0.163, ...},
        feature_names=feature_cols,
        training_config={"n_estimators": 400, ...},
    )
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _current_git_commit() -> Optional[str]:
    """Return the short git commit hash of HEAD, or None if not a repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode().strip()
    except Exception:
        return None


def _git_is_dirty() -> Optional[bool]:
    """True if the working tree has uncommitted changes, None if unknown."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return bool(out.strip())
    except Exception:
        return None


def save_model_metadata(
    model_path: Path,
    *,
    model_type: str,
    n_train_rows: int,
    val_metrics: dict[str, Any],
    feature_names: Optional[list[str]] = None,
    training_config: Optional[dict[str, Any]] = None,
    n_val_rows: Optional[int] = None,
    n_test_rows: Optional[int] = None,
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    """
    Write a sidecar JSON next to model_path describing the training run.

    The sidecar lives at `{model_path}.metadata.json`. If the directory
    doesn't exist, it's created. Write is atomic-ish via rename-on-complete.
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    feature_hash = None
    if feature_names:
        feature_hash = hashlib.sha256(
            ",".join(feature_names).encode("utf-8")
        ).hexdigest()[:16]

    meta: dict[str, Any] = {
        "schema_version": 1,
        "model_file": model_path.name,
        "model_type": model_type,
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _current_git_commit(),
        "git_dirty": _git_is_dirty(),
        "n_train_rows": int(n_train_rows),
        "n_val_rows": int(n_val_rows) if n_val_rows is not None else None,
        "n_test_rows": int(n_test_rows) if n_test_rows is not None else None,
        "val_metrics": {k: float(v) if isinstance(v, (int, float)) else v
                        for k, v in val_metrics.items()},
        "feature_count": len(feature_names) if feature_names else None,
        "feature_names_hash": feature_hash,
        "training_config": training_config or {},
    }
    if extra:
        meta["extra"] = extra

    sidecar = model_path.with_suffix(model_path.suffix + ".metadata.json")
    tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True))
    tmp.replace(sidecar)
    return sidecar


def load_model_metadata(model_path: Path) -> Optional[dict[str, Any]]:
    """Load the sidecar JSON if present, else return None."""
    model_path = Path(model_path)
    sidecar = model_path.with_suffix(model_path.suffix + ".metadata.json")
    if not sidecar.exists():
        return None
    try:
        return json.loads(sidecar.read_text())
    except Exception:
        return None
