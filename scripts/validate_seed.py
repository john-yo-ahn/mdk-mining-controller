"""
Run the full pipeline on a specific random seed with an isolated
data directory. Used for multi-seed validation — proving that the
TE separation, XGBoost detection, and LSTM separation hold across
different random realizations of the synthetic data, not just the
one seed the project was developed against.

Usage:
    uv run python -m scripts.validate_seed --seed 42 --data-dir /tmp/mdk_seed_42
    uv run python -m scripts.validate_seed --seed 123 --data-dir /tmp/mdk_seed_123

The script:
  1. Patches src.config paths to use the given --data-dir
  2. Generates fresh synthetic telemetry with the given seed
  3. Preprocesses and computes TE KPI (assignment-compliant formula)
  4. Builds the v3 feature matrix (175 features including te_health suite)
  5. Trains XGBoost and evaluates on the held-out test set
  6. Trains LSTM via the Phase A-D path (per-model scalers, alive filter,
     9 features, burn-in threshold calibration)
  7. Writes a JSON summary to --data-dir/seed_results.json

Each run is fully isolated — no shared state, no file contention.
Multiple seeds can run concurrently as long as the machine has
enough RAM (~8 GB peak per run during feature building).
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

# ── Monkey-patch config paths BEFORE any other src import ──────────
def _patch_config(data_dir: Path):
    """Override src.config paths so this run uses an isolated directory."""
    import src.config as cfg
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg.DATA_DIR = data_dir
    cfg.RAW_DIR = data_dir / "raw"
    cfg.PROCESSED_DIR = data_dir / "processed"
    cfg.MODELS_DIR = data_dir / "models"
    cfg.BATCH_DB_PATH = cfg.RAW_DIR / "mdk.duckdb"
    cfg.LIVE_DB_PATH = cfg.RAW_DIR / "mdk_live.duckdb"
    for d in [cfg.RAW_DIR, cfg.PROCESSED_DIR, cfg.MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    seed = args.seed
    total_start = time.time()

    print(f"\n{'='*70}")
    print(f"  MULTI-SEED VALIDATION — seed={seed}")
    print(f"  data_dir={data_dir}")
    print(f"{'='*70}\n")

    # Patch BEFORE importing pipeline modules
    _patch_config(data_dir)

    import numpy as np
    import pandas as pd
    from src.config import (
        RAW_DIR, PROCESSED_DIR, MODELS_DIR,
        SimulationConfig, BATCH_DB_PATH,
    )
    from src.synthetic.generator import MiningDataGenerator
    from src.pipeline.preprocessing import preprocess_pipeline
    from src.kpi.true_efficiency import compute_all_te_variants, compute_fleet_te_summary
    from src.pipeline.features import (
        build_feature_matrix, get_feature_columns, split_temporal_tvt,
        FEATURES_VERSION,
    )
    from src.models.xgboost_classifier import MinerFailureClassifier
    from src.models.evaluation import (
        compute_classification_metrics, detection_timeline,
    )
    from src.models.lstm_autoencoder import AnomalyDetector, filter_alive_rows

    results = {"seed": seed, "data_dir": str(data_dir)}

    raw_path = RAW_DIR / "mining_telemetry.parquet"
    cache_path = PROCESSED_DIR / f"features.v{FEATURES_VERSION}.parquet"

    # ── Step 1: Generate synthetic data (or reuse if cached) ───────
    if raw_path.exists():
        print(f"STEP 1: Reusing existing raw data at {raw_path}")
        df_raw = pd.read_parquet(raw_path)
    else:
        print("STEP 1: Generating synthetic data...")
        t0 = time.time()
        cfg = SimulationConfig(random_seed=seed)
        gen = MiningDataGenerator(cfg)
        df_raw = gen.generate()
        df_raw.to_parquet(raw_path, index=False, engine="pyarrow")
        print(f"  Generated {len(df_raw):,} rows in {time.time()-t0:.1f}s")

    results["n_rows"] = len(df_raw)
    results["n_miners"] = int(df_raw["miner_id"].nunique())

    # ── Steps 2-3: Preprocess + TE + features (or reuse cache) ─────
    if cache_path.exists():
        print(f"STEPS 2-3: Reusing feature cache at {cache_path}")
        df_features = pd.read_parquet(cache_path)
        # Still compute TE summary on raw for the separation metric
        df_clean = preprocess_pipeline(df_raw, verbose=False)
        df_te = compute_all_te_variants(df_clean)
        df_te["efficiency_jth"] = np.where(
            df_te["hashrate_th"] > 0,
            df_te["power_w"] / df_te["hashrate_th"],
            np.nan,
        )
        del df_clean
    else:
        print("STEP 2: Preprocessing + TE KPI...")
        df_clean = preprocess_pipeline(df_raw, verbose=False)
        df_te = compute_all_te_variants(df_clean)
        df_te["efficiency_jth"] = np.where(
            df_te["hashrate_th"] > 0,
            df_te["power_w"] / df_te["hashrate_th"],
            np.nan,
        )
        del df_clean

        print("STEP 3: Building feature matrix...")
        t0 = time.time()
        df_features = build_feature_matrix(df_te, verbose=True)
        print(f"  {len(df_features):,} rows, {len(get_feature_columns(df_features))} features in {time.time()-t0:.1f}s")
        df_features.to_parquet(cache_path, index=False, engine="pyarrow")

    del df_raw
    gc.collect()

    # TE separation (per-miner)
    summary = compute_fleet_te_summary(df_te)
    healthy_m = summary[summary["failure_type"] == "none"]
    failing_m = summary[summary["failure_type"] != "none"]
    if len(healthy_m) > 0 and len(failing_m) > 0:
        te_sep = (
            healthy_m["te_health_mean"].mean()
            / max(failing_m["te_health_mean"].mean(), 1e-10)
            - 1
        ) * 100
        jth_sep = (
            failing_m["jth_mean"].mean()
            / max(healthy_m["jth_mean"].mean(), 1e-10)
            - 1
        ) * 100
    else:
        te_sep = float("nan")
        jth_sep = float("nan")
    results["te_health_per_miner_sep_pct"] = round(te_sep, 1)
    results["jth_per_miner_sep_pct"] = round(jth_sep, 1)
    results["n_healthy_miners"] = int(len(healthy_m))
    results["n_failing_miners"] = int(len(failing_m))
    print(f"  TE per-miner sep: te_health={te_sep:+.1f}%, jth={jth_sep:+.1f}%")
    del df_te

    feature_cols = get_feature_columns(df_features)
    results["n_features"] = len(feature_cols)

    # ── Step 4: XGBoost ────────────────────────────────────────────
    print("STEP 4: Training XGBoost...")
    train_df, val_df, test_df = split_temporal_tvt(
        df_features, train_pos_fraction=0.55, val_pos_fraction=0.15,
    )
    X_train = train_df[feature_cols]
    y_train = train_df["is_pre_failure"].astype(int).values
    X_val = val_df[feature_cols]
    y_val = val_df["is_pre_failure"].astype(int).values
    X_test = test_df[feature_cols]
    y_test = test_df["is_pre_failure"].astype(int).values

    t0 = time.time()
    xgb = MinerFailureClassifier()
    xgb.fit(X_train, y_train)
    xgb.optimize_threshold(X_val, y_val, strategy="f1_with_floor")
    xgb_time = time.time() - t0

    y_prob = xgb.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= xgb.threshold_).astype(int)
    metrics = compute_classification_metrics(y_test, y_pred, y_prob)
    tl = detection_timeline(test_df, y_pred)
    n_detected = int(tl["detected"].sum()) if len(tl) else 0
    n_total = len(tl) if len(tl) else 0
    avg_lead = (
        float(tl[tl["detected"]]["lead_time_hours"].mean())
        if n_detected > 0 else float("nan")
    )

    results["xgb_auc"] = round(metrics["auc_roc"], 3)
    results["xgb_f1"] = round(metrics["f1"], 3)
    results["xgb_precision"] = round(metrics["precision"], 3)
    results["xgb_recall"] = round(metrics["recall"], 3)
    results["xgb_detection"] = f"{n_detected}/{n_total}"
    results["xgb_avg_lead_hours"] = round(avg_lead, 1)
    results["xgb_train_time_s"] = round(xgb_time, 1)

    # Check if te_health features are in top 20
    importances = xgb.model_.get_booster().get_score(importance_type="gain")
    top_20 = sorted(importances.items(), key=lambda x: -x[1])[:20]
    te_in_top20 = [name for name, _ in top_20 if "te_health" in name]
    results["te_health_in_top20_features"] = te_in_top20
    results["te_health_top20_count"] = len(te_in_top20)

    print(
        f"  XGBoost: AUC={metrics['auc_roc']:.3f} F1={metrics['f1']:.3f} "
        f"detect={n_detected}/{n_total} lead={avg_lead:.1f}h "
        f"te_in_top20={len(te_in_top20)} ({xgb_time:.1f}s)"
    )

    # Save model
    xgb.save(MODELS_DIR / "xgboost_failure.joblib")

    # ── Step 5: LSTM ───────────────────────────────────────────────
    print("STEP 5: Training LSTM-AE (Phase A-D path)...")

    # Needed columns for LSTM
    needed = [
        "timestamp", "miner_id", "model", "failure_type", "is_pre_failure",
        "clock_frequency_mhz", "voltage_v", "hashrate_th",
        "temperature_c", "power_w", "ambient_temperature_c",
    ]
    df_lstm = pd.read_parquet(cache_path, columns=needed)
    train_l, val_l, test_l = split_temporal_tvt(
        df_lstm, train_pos_fraction=0.55, val_pos_fraction=0.15,
    )
    del df_lstm
    gc.collect()

    healthy_train = filter_alive_rows(train_l[train_l["failure_type"] == "none"])
    healthy_val = filter_alive_rows(val_l[val_l["failure_type"] == "none"])
    healthy_test = filter_alive_rows(test_l[test_l["failure_type"] == "none"])
    failure_test = test_l[test_l["failure_type"] != "none"].reset_index(drop=True)
    fail_alive_mask = (
        (failure_test["hashrate_th"] > 1.0) & (failure_test["voltage_v"] > 0.05)
    )
    failure_alive = failure_test[fail_alive_mask].reset_index(drop=True)

    lstm = AnomalyDetector(
        input_dim=9, seq_len=60, hidden_dim=64, latent_dim=32,
        n_layers=2, n_epochs=30, batch_size=256,
        early_stopping_patience=4,
    )
    lstm.fit_scaler(healthy_train)

    t0 = time.time()
    X_train_l = lstm.prepare_sequences(healthy_train, stride=5)
    X_val_l = lstm.prepare_sequences(healthy_val, stride=5)
    X_h_test = lstm.prepare_sequences(healthy_test, stride=5)
    X_f_alive = lstm.prepare_sequences(failure_alive, stride=5) \
        if len(failure_alive) else np.empty((0, 60, 9), dtype=np.float32)

    del train_l, val_l, test_l
    del healthy_train, healthy_val, healthy_test, failure_test, failure_alive
    gc.collect()

    lstm.fit(X_train_l, X_val_l)

    # Burn-in threshold calibration
    n_burn = max(len(X_h_test) // 5, 1000)
    n_burn = min(n_burn, len(X_h_test))
    burn_errs = lstm.compute_reconstruction_error(X_h_test[:n_burn])
    lstm.set_threshold(burn_errs, percentile=95.0)
    X_h_eval = X_h_test[n_burn:]

    h_err = lstm.compute_reconstruction_error(X_h_eval)
    fa_err = lstm.compute_reconstruction_error(X_f_alive) if len(X_f_alive) else np.array([])
    lstm_time = time.time() - t0

    h_mean = float(h_err.mean()) if len(h_err) else 0.0
    fa_mean = float(fa_err.mean()) if len(fa_err) else 0.0
    sep_alive = fa_mean / max(h_mean, 1e-10) if len(fa_err) else float("nan")
    far = float((h_err > lstm.threshold_).mean()) if len(h_err) else float("nan")
    det_alive = float((fa_err > lstm.threshold_).mean()) if len(fa_err) else float("nan")

    results["lstm_sep_alive"] = round(sep_alive, 2)
    results["lstm_detection_alive"] = round(det_alive * 100, 1) if not np.isnan(det_alive) else "nan"
    results["lstm_far"] = round(far * 100, 1) if not np.isnan(far) else "nan"
    results["lstm_train_time_s"] = round(lstm_time, 1)

    lstm.save(MODELS_DIR / "lstm_ae.pt")

    print(
        f"  LSTM: sep_alive={sep_alive:.2f}x det={det_alive:.1%} "
        f"FAR={far:.1%} ({lstm_time:.1f}s)"
    )

    # ── Summary ────────────────────────────────────────────────────
    results["total_time_s"] = round(time.time() - total_start, 1)
    results_path = data_dir / "seed_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  SEED {seed} COMPLETE in {results['total_time_s']:.0f}s")
    print(f"{'='*70}")
    print(json.dumps(results, indent=2, default=str))
    print(f"\nResults written to {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
