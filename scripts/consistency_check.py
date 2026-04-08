"""
Full-pipeline consistency check.

Runs a battery of tests against the currently-committed state to
verify everything hangs together after the Phase A + Phase B fixes.
Does NOT retrain — relies on the already-saved models in
data/models/ and the cached feature matrix in data/processed/.

Exits 0 if all checks pass, 1 otherwise. Intended to be run before
every commit or release as a smoke test.

Usage:
    uv run python -m scripts.consistency_check
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*fragmented.*")

import numpy as np
import pandas as pd


# ── Test reporting helpers ────────────────────────────────────────

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str):
    """Decorator: register a test, catch exceptions, record pass/fail."""
    def decorator(fn):
        def wrapper():
            print(f"\n── {name} ──")
            try:
                t0 = time.time()
                detail = fn()
                elapsed = time.time() - t0
                RESULTS.append((name, True, detail or f"ok ({elapsed:.1f}s)"))
                print(f"  PASS  ({elapsed:.1f}s)  {detail or ''}")
                return True
            except AssertionError as e:
                RESULTS.append((name, False, f"ASSERTION: {e}"))
                print(f"  FAIL  {e}")
                return False
            except Exception as e:
                RESULTS.append((name, False, f"EXCEPTION: {type(e).__name__}: {e}"))
                print(f"  ERROR {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                return False
        return wrapper
    return decorator


# ── Individual checks ─────────────────────────────────────────────

@check("1. Imports: every src.* and scripts.* module loads")
def check_imports():
    import src.config
    import src.run_pipeline
    import src.validate
    import src.synthetic.generator
    import src.synthetic.scenarios
    import src.synthetic.physics
    import src.synthetic.failures
    import src.synthetic.metadata
    import src.pipeline.features
    import src.pipeline.preprocessing
    import src.pipeline.ingestion
    import src.kpi.true_efficiency
    import src.models.xgboost_classifier
    import src.models.lstm_autoencoder
    import src.models.evaluation
    import src.models.metadata
    import src.optimizer.rules
    import src.optimizer.safety
    import src.storage.backend
    import src.cli.ai_bridge
    import src.cli.simulation
    import src.cli.app
    import scripts.train_lstm_only
    import scripts.test_live_feature_parity
    return "24 modules imported"


@check("2. Feature cache exists and has expected shape")
def check_feature_cache():
    from src.config import PROCESSED_DIR
    from src.pipeline.features import FEATURES_VERSION

    cache = PROCESSED_DIR / f"features.v{FEATURES_VERSION}.parquet"
    assert cache.exists(), f"missing {cache}"

    df = pd.read_parquet(cache)
    assert len(df) > 1_000_000, f"suspiciously small: {len(df)} rows"
    assert "is_pre_failure" in df.columns
    assert "degradation_phase" in df.columns
    assert "miner_id" in df.columns
    assert "timestamp" in df.columns

    # Verify the label fix: no pre-failure rows on healthy phases
    pf_on_healthy = df[
        df["is_pre_failure"] & (df["degradation_phase"] == "healthy")
    ]
    assert len(pf_on_healthy) == 0, f"label bug: {len(pf_on_healthy)} pre-failure rows on healthy phase"

    return f"{len(df):,} rows, {len(df.columns)} cols, labels clean"


@check("3. Trained models load cleanly")
def check_model_load():
    from src.models.xgboost_classifier import MinerFailureClassifier
    from src.models.lstm_autoencoder import AnomalyDetector

    xgb = MinerFailureClassifier.load()
    assert xgb.model_ is not None
    assert xgb.threshold_ is not None
    assert 0 < xgb.threshold_ < 1, f"threshold {xgb.threshold_} out of [0,1]"
    assert xgb.feature_names_ is not None
    assert len(xgb.feature_names_) == 152, f"expected 152 features, got {len(xgb.feature_names_)}"

    lstm = AnomalyDetector.load()
    assert lstm.model_ is not None
    assert lstm.threshold_ is not None
    assert lstm.threshold_ > 0
    assert lstm.feature_mean_ is not None, "persistent scaler missing"
    assert lstm.feature_std_ is not None
    assert lstm.feature_mean_.shape == (6,), f"scaler mean shape {lstm.feature_mean_.shape}"
    assert lstm.feature_names_ is not None
    assert len(lstm.feature_names_) == 6

    return f"XGB threshold={xgb.threshold_:.4f}, LSTM threshold={lstm.threshold_:.6f}, scaler present"


@check("4. Metadata sidecars exist and parse")
def check_metadata_sidecars():
    from src.config import MODELS_DIR
    from src.models.metadata import load_model_metadata

    xgb_meta = load_model_metadata(MODELS_DIR / "xgboost_failure.joblib")
    assert xgb_meta is not None, "XGBoost sidecar missing"
    assert xgb_meta["schema_version"] == 1
    assert xgb_meta["model_type"] == "xgboost_binary"
    assert xgb_meta["feature_count"] == 152
    assert "git_commit" in xgb_meta
    assert "val_metrics" in xgb_meta
    assert "auc_roc" in xgb_meta["val_metrics"]

    lstm_meta = load_model_metadata(MODELS_DIR / "lstm_ae.pt")
    assert lstm_meta is not None, "LSTM sidecar missing"
    assert lstm_meta["schema_version"] == 1
    assert lstm_meta["model_type"] == "lstm_autoencoder"
    assert lstm_meta["feature_count"] == 6

    return f"XGB AUC recorded: {xgb_meta['val_metrics'].get('auc_roc', '?'):.3f}"


@check("5. Metadata feature hash matches loaded model")
def check_metadata_hash():
    import hashlib
    from src.config import MODELS_DIR
    from src.models.xgboost_classifier import MinerFailureClassifier
    from src.models.metadata import load_model_metadata

    xgb = MinerFailureClassifier.load()
    expected_hash = hashlib.sha256(
        ",".join(xgb.feature_names_).encode("utf-8")
    ).hexdigest()[:16]

    meta = load_model_metadata(MODELS_DIR / "xgboost_failure.joblib")
    recorded = meta.get("feature_names_hash")
    assert recorded == expected_hash, (
        f"hash mismatch: model has {expected_hash}, sidecar has {recorded}"
    )
    return f"hash {expected_hash} matches"


@check("6. Train/val/test split is deterministic")
def check_split_determinism():
    from src.config import PROCESSED_DIR
    from src.pipeline.features import split_temporal_tvt, FEATURES_VERSION

    df = pd.read_parquet(
        PROCESSED_DIR / f"features.v{FEATURES_VERSION}.parquet",
        columns=["timestamp", "miner_id", "failure_type", "is_pre_failure"],
    )

    a_train, a_val, a_test = split_temporal_tvt(df, 0.55, 0.15)
    b_train, b_val, b_test = split_temporal_tvt(df, 0.55, 0.15)

    assert len(a_train) == len(b_train), "train size drift"
    assert len(a_val) == len(b_val), "val size drift"
    assert len(a_test) == len(b_test), "test size drift"
    assert a_train["timestamp"].max() == b_train["timestamp"].max()
    assert a_test["timestamp"].min() == b_test["timestamp"].min()

    return f"train={len(a_train):,} val={len(a_val):,} test={len(a_test):,}"


@check("7. XGBoost predictions reproduce metadata metrics")
def check_xgboost_reproducibility():
    from src.config import PROCESSED_DIR, MODELS_DIR
    from src.pipeline.features import split_temporal_tvt, get_feature_columns, FEATURES_VERSION
    from src.models.xgboost_classifier import MinerFailureClassifier
    from src.models.evaluation import compute_classification_metrics
    from src.models.metadata import load_model_metadata

    df = pd.read_parquet(PROCESSED_DIR / f"features.v{FEATURES_VERSION}.parquet")
    _, _, test_df = split_temporal_tvt(df, 0.55, 0.15)
    feature_cols = get_feature_columns(df)
    X_test = test_df[feature_cols]
    y_test = test_df["is_pre_failure"].astype(int).values

    xgb = MinerFailureClassifier.load()
    y_prob = xgb.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= xgb.threshold_).astype(int)
    metrics = compute_classification_metrics(y_test, y_pred, y_prob)

    meta = load_model_metadata(MODELS_DIR / "xgboost_failure.joblib")
    recorded_auc = meta["val_metrics"].get("auc_roc", 0)
    recorded_f1 = meta["val_metrics"].get("f1", 0)

    auc_drift = abs(metrics["auc_roc"] - recorded_auc)
    f1_drift = abs(metrics["f1"] - recorded_f1)
    assert auc_drift < 0.001, f"AUC drift: fresh={metrics['auc_roc']:.4f} meta={recorded_auc:.4f}"
    assert f1_drift < 0.001, f"F1 drift: fresh={metrics['f1']:.4f} meta={recorded_f1:.4f}"

    return f"AUC={metrics['auc_roc']:.3f} F1={metrics['f1']:.3f} matches sidecar"


@check("8. XGBoost detection timeline reproduces 3/6 catches")
def check_detection_timeline():
    from src.config import PROCESSED_DIR
    from src.pipeline.features import split_temporal_tvt, get_feature_columns, FEATURES_VERSION
    from src.models.xgboost_classifier import MinerFailureClassifier
    from src.models.evaluation import detection_timeline

    df = pd.read_parquet(PROCESSED_DIR / f"features.v{FEATURES_VERSION}.parquet")
    _, _, test_df = split_temporal_tvt(df, 0.55, 0.15)
    feature_cols = get_feature_columns(df)
    X_test = test_df[feature_cols]

    xgb = MinerFailureClassifier.load()
    y_prob = xgb.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= xgb.threshold_).astype(int)

    timeline = detection_timeline(test_df, y_pred)
    n_total = len(timeline)
    n_detected = int(timeline["detected"].sum())
    assert n_total == 6, f"expected 6 test failures, got {n_total}"
    assert n_detected == 3, f"expected 3/6 detected, got {n_detected}/{n_total}"

    if n_detected > 0:
        avg_lead = timeline[timeline["detected"]]["lead_time_hours"].mean()
        # Report reports 182.6h; allow ±1h drift
        assert 180 < avg_lead < 185, f"avg lead time {avg_lead:.1f}h not near 182.6h"

    return f"{n_detected}/{n_total} detected, avg lead {avg_lead:.1f}h"


@check("9. LSTM separation ratio and detection rate reproduce")
def check_lstm_reproducibility():
    from src.config import PROCESSED_DIR, MODELS_DIR
    from src.pipeline.features import split_temporal_tvt, FEATURES_VERSION
    from src.models.lstm_autoencoder import AnomalyDetector
    from src.models.metadata import load_model_metadata

    df = pd.read_parquet(PROCESSED_DIR / f"features.v{FEATURES_VERSION}.parquet")
    _, _, test_df = split_temporal_tvt(df, 0.55, 0.15)

    lstm = AnomalyDetector.load()
    healthy_test = test_df[test_df["failure_type"] == "none"]
    failure_test = test_df[test_df["failure_type"] != "none"]

    X_fail = lstm.prepare_sequences(failure_test, stride=5)
    X_healthy = lstm.prepare_sequences(healthy_test, stride=5)
    assert len(X_fail) > 0 and len(X_healthy) > 0

    fail_err = lstm.compute_reconstruction_error(X_fail)
    health_err = lstm.compute_reconstruction_error(X_healthy)
    sep = float(fail_err.mean() / max(health_err.mean(), 1e-10))
    far = float((health_err > lstm.threshold_).mean())
    det_rate = float((fail_err > lstm.threshold_).mean())

    # Compare against sidecar values (should be identical up to float)
    meta = load_model_metadata(MODELS_DIR / "lstm_ae.pt")
    recorded_sep = meta["val_metrics"].get("test_separation_ratio", 0)
    if recorded_sep:
        drift = abs(sep - recorded_sep)
        assert drift < 0.01, f"separation drift: fresh={sep:.3f} meta={recorded_sep:.3f}"

    # Sanity bounds from the report
    assert sep > 2.0, f"separation ratio {sep:.2f}x below 2.0"
    assert far < 0.10, f"healthy false-alarm rate {far:.1%} above 10%"

    return f"sep={sep:.2f}x FAR={far:.1%} det={det_rate:.1%}"


@check("10. Live feature parity still passes (scripts/test_live_feature_parity)")
def check_live_parity():
    import subprocess
    result = subprocess.run(
        [".venv/bin/python", "-u", "-m", "scripts.test_live_feature_parity"],
        capture_output=True, text=True, timeout=180,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"parity test failed: returncode={result.returncode}"
    # Pull out the match rate line
    for line in output.splitlines():
        if "Match rate" in line:
            return line.strip()
    return "passed"


@check("11. build_feature_matrix emits zero warnings on a 3-miner slice")
def check_no_warnings():
    import warnings as warn_module
    from src.config import PROCESSED_DIR
    from src.pipeline.features import build_feature_matrix, FeatureConfig, FEATURES_VERSION
    from src.kpi.true_efficiency import compute_all_te_variants

    df = pd.read_parquet(
        PROCESSED_DIR / f"features.v{FEATURES_VERSION}.parquet",
        columns=[
            "timestamp", "miner_id", "model", "container_id", "position",
            "clock_frequency_mhz", "voltage_v", "hashrate_th",
            "temperature_c", "power_w", "ambient_temperature_c",
            "operating_mode", "failure_type", "is_pre_failure",
            "degradation_phase",
        ],
    )
    first3 = df["miner_id"].unique()[:3]
    df = df[df["miner_id"].isin(first3)].reset_index(drop=True)
    df = compute_all_te_variants(df)

    captured: list[str] = []
    def showwarning(message, category, *args, **kwargs):
        # Ignore PerformanceWarning (DataFrame fragmentation, F7 scope)
        name = category.__name__
        if "PerformanceWarning" in name:
            return
        captured.append(f"{name}: {message}")

    old_showwarning = warn_module.showwarning
    warn_module.showwarning = showwarning
    try:
        warn_module.resetwarnings()
        warn_module.simplefilter("always")
        cfg = FeatureConfig(
            rolling_windows_minutes=[2, 15, 60, 360, 1440, 10080],
        )
        out = build_feature_matrix(df, cfg, drop_warmup=False, verbose=False)
    finally:
        warn_module.showwarning = old_showwarning

    assert len(out) > 0, "feature matrix is empty"
    if captured:
        print("  Captured non-PerformanceWarning messages:")
        for w in captured[:5]:
            print(f"    {w}")
    assert len(captured) == 0, f"{len(captured)} warnings emitted"

    return f"0 warnings on {len(out):,} rows × {len(out.columns)} cols"


@check("12. AIBridge live inference end-to-end smoke test")
def check_ai_bridge_smoke():
    from src.cli.ai_bridge import AIBridge, BUFFER_SIZE, LSTM_SEQ_LEN

    assert BUFFER_SIZE == 10080, f"BUFFER_SIZE={BUFFER_SIZE}, expected 10080"
    assert LSTM_SEQ_LEN == 60, f"LSTM_SEQ_LEN={LSTM_SEQ_LEN}, expected 60"

    ai = AIBridge()
    assert ai.load_models(), "load_models returned False"
    assert len(ai.xgb_features) == 152
    assert ai.lstm_model is not None
    assert ai.lstm_model.feature_mean_ is not None

    ai.register_miner("MNR-TEST", "Antminer S21 Pro", 234.0)
    buf = ai.buffers["MNR-TEST"]
    assert buf._size == 10080
    assert buf.lstm_scaler_mean is not None, "scaler not propagated on register_miner"

    # Fill with deterministic telemetry (5000 readings - enough for
    # MIN_BUFFER_FOR_FEATURES=360 but not full 7 days)
    rng = np.random.default_rng(42)
    for _ in range(5000):
        ai.push_telemetry(
            "MNR-TEST",
            freq=500.0 + rng.normal(0, 5),
            volt=0.38 + rng.normal(0, 0.002),
            hr=234.0 + rng.normal(0, 3),
            temp=72.0 + rng.normal(0, 1.5),
            pwr=3510.0 + rng.normal(0, 30),
            ambient=30.0 + rng.normal(0, 1),
        )

    score, predicted = ai.predict("MNR-TEST", health_score=0.95)
    assert 0.0 <= score <= 1.0
    assert isinstance(predicted, bool)
    assert not ai._xgb_error_logged, "XGBoost live path logged an error"
    assert not ai._lstm_error_logged, "LSTM live path logged an error"

    return f"score={score:.4f} predicted={predicted}"


@check("13. git working tree is clean (no uncommitted changes)")
def check_git_clean():
    import subprocess
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, timeout=5,
    )
    dirty = result.stdout.strip()
    if dirty:
        print(f"  Uncommitted: {dirty[:200]}")
    assert not dirty, "working tree is dirty"
    return "working tree clean"


# ── Main ──────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 70)
    print("  MDK AI Mining Controller — Consistency Check")
    print("=" * 70)
    total_start = time.time()

    check_imports()
    check_feature_cache()
    check_model_load()
    check_metadata_sidecars()
    check_metadata_hash()
    check_split_determinism()
    check_xgboost_reproducibility()
    check_detection_timeline()
    check_lstm_reproducibility()
    check_live_parity()
    check_no_warnings()
    check_ai_bridge_smoke()
    check_git_clean()

    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    for name, ok, detail in RESULTS:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}")
        if not ok:
            print(f"         {detail}")
    print()
    print(f"  Total: {n_pass} passed, {n_fail} failed in {time.time()-total_start:.1f}s")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
