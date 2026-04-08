"""
End-to-end pipeline: generate data → preprocess → features → train models → evaluate.
Run with: uv run python -m src.run_pipeline
"""

import time
import sys
import numpy as np
import pandas as pd

from .config import SimulationConfig, FeatureConfig, MODELS_DIR, PROCESSED_DIR
from .synthetic.generator import MiningDataGenerator
from .pipeline.ingestion import load_telemetry
from .pipeline.preprocessing import preprocess_pipeline
from .pipeline.features import (
    build_feature_matrix,
    get_feature_columns,
    split_temporal_tvt,
    FEATURES_VERSION,
)
from .kpi.true_efficiency import compute_all_te_variants, compute_fleet_te_summary
from .storage.backend import DuckDBStore
from .models.xgboost_classifier import MinerFailureClassifier
from .models.lstm_autoencoder import AnomalyDetector
from .models.evaluation import (
    compute_classification_metrics, compute_confusion_matrix,
    detection_timeline, compare_models,
)


def main():
    total_start = time.time()

    # ── Step 1: Generate synthetic data ─────────────────────────────
    print("=" * 70)
    print("STEP 1: Generating synthetic mining telemetry")
    print("=" * 70)

    # Use config.py defaults (30 miners × 120 days). The override that
    # used to live here (n_miners=20, n_days=90) was left over from
    # early development when we wanted fast iteration; it's now the
    # job of config.py to hold the canonical scale.
    config = SimulationConfig()
    generator = MiningDataGenerator(config)

    # Load from cache if parquet exists and matches expected size
    from .config import RAW_DIR
    cache_path = RAW_DIR / "mining_telemetry.parquet"
    expected_rows = config.n_miners * config.n_steps
    regenerate = True
    if cache_path.exists():
        try:
            df_raw = MiningDataGenerator.load()
            if len(df_raw) == expected_rows:
                print(f"  Loaded cached dataset: {len(df_raw):,} rows")
                regenerate = False
        except Exception:
            pass

    if regenerate:
        t0 = time.time()
        df_raw = generator.generate()
        path = generator.save(df_raw)
        print(f"  Generated in {time.time()-t0:.1f}s: {len(df_raw):,} rows")
    print(f"  Failure distribution:")
    print(df_raw[df_raw["failure_type"] != "none"]["failure_type"].value_counts().to_string())
    print(f"  Pre-failure labels: {df_raw['is_pre_failure'].sum():,} "
          f"({df_raw['is_pre_failure'].mean():.1%})")

    print(f"  Phase distribution:")
    phase_counts = df_raw["degradation_phase"].value_counts()
    for phase, count in phase_counts.items():
        print(f"    {phase:15s}: {count:>10,}")

    # ── Step 2: Load into DuckDB ────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: Loading into DuckDB")
    print("=" * 70)

    # Hold the writer lock only for the ingest + summary query, then
    # release immediately. The training steps below don't touch the DB,
    # so there's no reason to keep it locked through XGBoost / LSTM.
    with DuckDBStore() as store:
        n_loaded = store.ingest(df_raw, mode="replace")
        print(f"  Loaded {n_loaded:,} rows into DuckDB ({store.db_path})")

        summary = store.fleet_summary()
        print(f"  Fleet summary: {len(summary)} miners")
        print(f"  Avg J/TH across fleet: {summary['avg_jth'].mean():.1f}")
        db_path = store.db_path

    # ── Step 3: Preprocess ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3: Preprocessing")
    print("=" * 70)

    df_clean = preprocess_pipeline(df_raw)

    # ── Step 4: KPI computation ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4: Computing True Efficiency KPIs")
    print("=" * 70)

    df_te = compute_all_te_variants(df_clean)
    te_summary = compute_fleet_te_summary(df_te)
    print(f"  Fleet avg TE_health: {te_summary['te_health_mean'].mean():.6f}")
    print(f"  Fleet avg J/TH: {te_summary['jth_mean'].mean():.1f}")

    # Show TE for healthy vs failing miners
    healthy = te_summary[te_summary["failure_type"] == "none"]
    failing = te_summary[te_summary["failure_type"] != "none"]
    if len(healthy) > 0 and len(failing) > 0:
        print(f"  TE_health (healthy miners): {healthy['te_health_mean'].mean():.6f}")
        print(f"  TE_health (failing miners): {failing['te_health_mean'].mean():.6f}")
        print(f"  -> TE_health separates healthy from failing by "
              f"{(healthy['te_health_mean'].mean() / max(failing['te_health_mean'].mean(), 1e-10) - 1)*100:.1f}%")

    # ── Step 5: Feature engineering ─────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 5: Feature engineering")
    print("=" * 70)

    # Cache the engineered feature matrix. The build is a pure function of
    # the raw parquet, so we invalidate the cache whenever the raw parquet
    # is newer than the cached features. The version number in the filename
    # comes from FEATURES_VERSION in src/pipeline/features.py — bump it
    # whenever you change feature code to force a fresh rebuild.
    features_cache = PROCESSED_DIR / f"features.v{FEATURES_VERSION}.parquet"
    raw_parquet = RAW_DIR / "mining_telemetry.parquet"
    cache_fresh = (
        features_cache.exists()
        and raw_parquet.exists()
        and features_cache.stat().st_mtime >= raw_parquet.stat().st_mtime
    )

    df_features = None
    if cache_fresh:
        try:
            df_features = pd.read_parquet(features_cache)
            print(
                f"  Loaded cached features: {len(df_features):,} rows, "
                f"{len(df_features.columns)} cols from {features_cache}"
            )
        except Exception as e:
            print(f"  Cache read failed ({e}), rebuilding")
            df_features = None

    if df_features is None:
        t0 = time.time()
        df_features = build_feature_matrix(df_te)
        build_secs = time.time() - t0
        print(f"  Feature matrix built in {build_secs:.1f}s")
        try:
            df_features.to_parquet(features_cache, index=False, engine="pyarrow")
            print(f"  Cached features to {features_cache}")
        except Exception as e:
            print(f"  WARNING: failed to cache features: {e}")

    feature_cols = get_feature_columns(df_features)

    # ── Step 6: Train/val/test split ────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 6: Train/val/test split")
    print("=" * 70)

    # Three-way temporal split. Model fits on train, thresholds are tuned
    # on val, final metrics are reported on test. Without the val split,
    # tuning the decision threshold on the same data you evaluate on is
    # textbook leakage — every precision/recall number is biased upward.
    train_df, val_df, test_df = split_temporal_tvt(
        df_features, train_pos_fraction=0.55, val_pos_fraction=0.15,
    )

    X_train = train_df[feature_cols]
    y_train = train_df["is_pre_failure"].astype(int).values
    X_val = val_df[feature_cols]
    y_val = val_df["is_pre_failure"].astype(int).values
    X_test = test_df[feature_cols]
    y_test = test_df["is_pre_failure"].astype(int).values

    print(f"  Features: {len(feature_cols)}")

    # ── Step 7: Train XGBoost ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 7: Training XGBoost classifier")
    print("=" * 70)

    t0 = time.time()
    xgb_model = MinerFailureClassifier(n_estimators=400, max_depth=6, learning_rate=0.05)
    # Fit on train, early-stop on val. XGBoost gets to see val internally
    # via its own eval_set, but that's fine — val is used symmetrically
    # for the threshold tuning below, and test is touched only once at
    # the very end.
    xgb_model.fit(X_train, y_train, X_val, y_val)
    print(f"  Training time: {time.time()-t0:.1f}s")

    # Threshold tuning uses VALIDATION scores, not test. This is the key
    # correctness fix for the Apr 8 data-leakage issue. The f1_with_floor
    # strategy falls back to precision >= 0.05 if F1-max would collapse
    # to a threshold that flags everything under extreme class imbalance.
    xgb_model.optimize_threshold(
        X_val, y_val, strategy="f1_with_floor", min_precision=0.05,
    )

    # Final metrics: test set, untouched until this moment.
    print("\n  Final test-set metrics (model + threshold never saw this data):")
    xgb_metrics = xgb_model.evaluate(X_test, y_test)

    # Feature importance
    importance = xgb_model.get_feature_importance(top_n=10)
    print(f"\n  Top 10 features:")
    for _, row in importance.iterrows():
        print(f"    {row['feature']:40s} {row['importance']:.1f}")

    # Detection timeline
    xgb_preds = xgb_model.predict(X_test)
    timeline = detection_timeline(test_df, xgb_preds)
    if len(timeline) > 0:
        detected = timeline[timeline["detected"]]
        print(f"\n  Detection timeline:")
        print(f"    Failures in test set: {len(timeline)}")
        print(f"    Detected: {len(detected)}/{len(timeline)}")
        if len(detected) > 0:
            print(f"    Avg lead time: {detected['lead_time_hours'].mean():.1f} hours")

    # Confusion matrix
    cm = compute_confusion_matrix(y_test, xgb_preds)
    print(f"\n  Confusion matrix:")
    print(f"    TN={cm[0][0]:,}  FP={cm[0][1]:,}")
    print(f"    FN={cm[1][0]:,}  TP={cm[1][1]:,}")

    # Save model + metadata sidecar
    xgb_model.save()
    from .models.metadata import save_model_metadata
    save_model_metadata(
        model_path=MODELS_DIR / "xgboost_failure.joblib",
        model_type="xgboost_binary",
        n_train_rows=len(X_train),
        n_val_rows=len(X_val),
        n_test_rows=len(X_test),
        val_metrics=xgb_metrics,
        feature_names=feature_cols,
        training_config={
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.05,
            "tree_method": "hist",
            "scale_pos_weight": "sqrt-capped",
            "threshold_strategy": "f1_with_floor",
            "threshold_value": float(xgb_model.threshold_),
            "split": "split_temporal_tvt(0.55, 0.15)",
        },
        extra={
            "features_version": FEATURES_VERSION,
            "test_failures_detected": int(timeline["detected"].sum()) if len(timeline) else 0,
            "test_failures_total": int(len(timeline)),
        },
    )

    # ── Step 8: Train LSTM-Autoencoder ──────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 8: Training LSTM-Autoencoder")
    print("=" * 70)

    lstm_model = AnomalyDetector(
        input_dim=9, seq_len=60, hidden_dim=64, latent_dim=32,
        n_layers=2, n_epochs=30, batch_size=256,
        early_stopping_patience=4,
    )

    # Pull healthy rows from each split. The LSTM-AE is trained on
    # healthy-only data; every failure row (in any split) is treated as
    # held-out for evaluation only.
    healthy_train = train_df[train_df["failure_type"] == "none"]
    healthy_val = val_df[val_df["failure_type"] == "none"]
    healthy_test = test_df[test_df["failure_type"] == "none"]
    failure_test = test_df[test_df["failure_type"] != "none"]

    print(
        f"  Healthy rows — train: {len(healthy_train):,} "
        f"val: {len(healthy_val):,} test: {len(healthy_test):,}"
    )

    # Fit the persistent normalization scaler on the training set ONLY.
    # Every later prepare_sequences call reuses the same mean/std so
    # reconstruction errors are directly comparable across splits.
    lstm_model.fit_scaler(healthy_train)

    t0 = time.time()
    X_train_seq = lstm_model.prepare_sequences(healthy_train, stride=5)
    X_val_seq = lstm_model.prepare_sequences(healthy_val, stride=5)
    print(
        f"  Prepared {len(X_train_seq):,} train + {len(X_val_seq):,} val "
        f"sequences in {time.time()-t0:.1f}s"
    )

    if len(X_train_seq) > 100 and len(X_val_seq) > 50:
        t0 = time.time()
        # Train on training sequences, early-stop on val sequences.
        lstm_model.fit(X_train_seq, X_val_seq)
        print(f"  Training time: {time.time()-t0:.1f}s")

        # Calibrate the anomaly threshold on VAL reconstruction errors.
        # Previously this used a carve-out of the training set, which
        # produced a threshold calibrated on data the model had literally
        # seen during fitting. Val is held out symmetrically across
        # XGBoost and LSTM-AE.
        val_errors = lstm_model.compute_reconstruction_error(X_val_seq)
        lstm_model.set_threshold(val_errors, percentile=95.0)

        # Final evaluation on test data. These numbers are honest:
        # neither the weights nor the threshold ever saw these rows.
        lstm_val_metrics = {"best_val_loss": float(min(lstm_model.val_losses_) if lstm_model.val_losses_ else 0.0)}
        lstm_n_test = 0
        X_fail_seq = None
        X_health_test_seq = None
        if len(failure_test) > 0:
            X_fail_seq = lstm_model.prepare_sequences(failure_test, stride=5)
            X_health_test_seq = lstm_model.prepare_sequences(healthy_test, stride=5)
            lstm_n_test = len(X_fail_seq) + len(X_health_test_seq)

            if len(X_fail_seq) > 0 and len(X_health_test_seq) > 0:
                fail_errors = lstm_model.compute_reconstruction_error(X_fail_seq)
                health_errors = lstm_model.compute_reconstruction_error(X_health_test_seq)

                fail_preds = (fail_errors > lstm_model.threshold_).astype(int)
                health_preds = (health_errors > lstm_model.threshold_).astype(int)
                sep = float(fail_errors.mean() / max(health_errors.mean(), 1e-10))

                print(f"\n  LSTM-AE Test Results (val-calibrated threshold):")
                print(f"    Healthy sequences flagged:  {health_preds.sum()}/{len(health_preds)} "
                      f"({health_preds.mean():.1%})")
                print(f"    Failure sequences flagged:  {fail_preds.sum()}/{len(fail_preds)} "
                      f"({fail_preds.mean():.1%})")
                print(f"    Mean error (healthy): {health_errors.mean():.6f}")
                print(f"    Mean error (failure): {fail_errors.mean():.6f}")
                print(f"    Separation ratio:     {sep:.2f}x")

                lstm_val_metrics.update({
                    "test_healthy_far": float(health_preds.mean()),
                    "test_failure_detection_rate": float(fail_preds.mean()),
                    "test_mean_error_healthy": float(health_errors.mean()),
                    "test_mean_error_failure": float(fail_errors.mean()),
                    "test_separation_ratio": sep,
                })

        lstm_model.save()
        save_model_metadata(
            model_path=MODELS_DIR / "lstm_ae.pt",
            model_type="lstm_autoencoder",
            n_train_rows=len(X_train_seq),
            n_val_rows=len(X_val_seq),
            n_test_rows=lstm_n_test,
            val_metrics=lstm_val_metrics,
            feature_names=lstm_model.feature_names_,
            training_config={
                "input_dim": 6,
                "seq_len": 60,
                "hidden_dim": 64,
                "latent_dim": 32,
                "n_layers": 2,
                "n_epochs": 30,
                "batch_size": 256,
                "early_stopping_patience": 4,
                "threshold_percentile": 95.0,
                "threshold_value": float(lstm_model.threshold_) if lstm_model.threshold_ else 0.0,
                "device": lstm_model.device,
            },
            extra={"features_version": FEATURES_VERSION},
        )
    else:
        print("  Not enough healthy sequences to train. Skipping LSTM-AE.")

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Total time: {time.time()-total_start:.1f}s")
    print(f"  Dataset: {len(df_raw):,} rows, {config.n_miners} miners, {config.n_days} days")
    print(f"  Features: {len(feature_cols)}")
    print(f"  XGBoost: precision={xgb_metrics['precision']:.3f} recall={xgb_metrics['recall']:.3f}")
    print(f"  Models saved to: {MODELS_DIR}")
    print(f"  Database: {db_path}")

    return xgb_metrics


if __name__ == "__main__":
    main()
