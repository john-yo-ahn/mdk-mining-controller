"""
Standalone LSTM-AE retrain.

Loads the cached feature matrix, recreates the same train/val/test split
the main pipeline uses, frees the large pandas frames as soon as the
sequences are built, and then trains the LSTM-AE alone.

Used after a full pipeline run where the main pipeline OOM-killed during
LSTM training due to peak memory pressure (5.9 GB pandas + 3 GB feature
parquet still in mmap + PyTorch state). Running just the LSTM with the
upstream frames freed cuts working set by ~10 GB.

Usage:
    uv run python -m scripts.train_lstm_only
"""

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, MODELS_DIR
from src.pipeline.features import split_temporal_tvt, FEATURES_VERSION
from src.models.lstm_autoencoder import AnomalyDetector, filter_alive_rows
from src.models.metadata import save_model_metadata


def main():
    total_start = time.time()
    cache_path = PROCESSED_DIR / f"features.v{FEATURES_VERSION}.parquet"
    if not cache_path.exists():
        raise SystemExit(
            f"Feature cache not found: {cache_path}\n"
            "Run `uv run python -m src.run_pipeline` first to build it."
        )

    print(f"Loading cached features from {cache_path}")
    t0 = time.time()
    needed_cols = [
        "timestamp", "miner_id", "model", "failure_type", "is_pre_failure",
        "clock_frequency_mhz", "voltage_v", "hashrate_th",
        "temperature_c", "power_w", "ambient_temperature_c",
    ]
    df = pd.read_parquet(cache_path, columns=needed_cols)
    print(f"  Loaded {len(df):,} rows x {len(df.columns)} cols in {time.time()-t0:.1f}s")

    print("Building train/val/test split...")
    train_df, val_df, test_df = split_temporal_tvt(
        df, train_pos_fraction=0.55, val_pos_fraction=0.15,
    )
    del df
    gc.collect()

    # Filter offline rows (hashrate≈0 or voltage≈0) out of the healthy
    # splits so the AE trains and evaluates on rows where the miner is
    # actually running. Shutdown rows drifted into the "healthy" class
    # in the old pipeline and pulled the healthy distribution toward
    # zero in every feature; they also contaminated the test metric
    # because a shutdown failure sequence reconstructs as ~0 error.
    healthy_train = filter_alive_rows(train_df[train_df["failure_type"] == "none"])
    healthy_val = filter_alive_rows(val_df[val_df["failure_type"] == "none"])
    healthy_test = filter_alive_rows(test_df[test_df["failure_type"] == "none"])

    # Failure rows are NOT filtered — we split them into alive/dead
    # to report sep_alive (the honest metric the plan gates on) and
    # sep_all (the contaminated metric kept for continuity with the
    # pre-Phase-B sidecar).
    failure_test = test_df[test_df["failure_type"] != "none"].reset_index(drop=True)
    fail_alive_mask = (failure_test["hashrate_th"] > 1.0) & (failure_test["voltage_v"] > 0.05)
    failure_test_alive = failure_test[fail_alive_mask].reset_index(drop=True)
    failure_test_dead = failure_test[~fail_alive_mask].reset_index(drop=True)
    print(
        f"Healthy rows (alive) - train: {len(healthy_train):,} "
        f"val: {len(healthy_val):,} test: {len(healthy_test):,}"
    )
    print(
        f"Failure rows in test: {len(failure_test):,} "
        f"({len(failure_test_alive):,} alive, "
        f"{len(failure_test_dead):,} dead/shutdown)"
    )

    lstm_model = AnomalyDetector(
        input_dim=9, seq_len=60, hidden_dim=64, latent_dim=32,
        n_layers=2, n_epochs=30, batch_size=256,
        early_stopping_patience=4,
    )
    print(f"Device: {lstm_model.device}")
    lstm_model.fit_scaler(healthy_train)

    print("Building train sequences...")
    t0 = time.time()
    X_train_seq = lstm_model.prepare_sequences(healthy_train, stride=5)
    print(f"  {len(X_train_seq):,} train sequences in {time.time()-t0:.1f}s")

    print("Building val sequences...")
    t0 = time.time()
    X_val_seq = lstm_model.prepare_sequences(healthy_val, stride=5)
    print(f"  {len(X_val_seq):,} val sequences in {time.time()-t0:.1f}s")

    print("Building test sequences (alive failures + dead failures + healthy held out)...")
    t0 = time.time()
    X_fail_alive_seq = lstm_model.prepare_sequences(failure_test_alive, stride=5) \
        if len(failure_test_alive) else np.empty((0, 60, 9), dtype=np.float32)
    X_fail_dead_seq = lstm_model.prepare_sequences(failure_test_dead, stride=5) \
        if len(failure_test_dead) else np.empty((0, 60, 9), dtype=np.float32)
    X_health_test_seq = lstm_model.prepare_sequences(healthy_test, stride=5)
    print(
        f"  {len(X_fail_alive_seq):,} alive + {len(X_fail_dead_seq):,} dead failure "
        f"+ {len(X_health_test_seq):,} healthy test sequences "
        f"in {time.time()-t0:.1f}s"
    )

    # CRUCIAL: free pandas frames before LSTM fit to avoid OOM.
    del train_df, val_df, test_df
    del healthy_train, healthy_val, healthy_test, failure_test
    del failure_test_alive, failure_test_dead
    gc.collect()
    print("Freed upstream pandas frames before LSTM fit.")

    print(f"Training LSTM-AE: {len(X_train_seq):,} sequences, "
          f"up to {lstm_model.n_epochs} epochs")
    t0 = time.time()
    lstm_model.fit(X_train_seq, X_val_seq)
    print(f"Training time: {time.time()-t0:.1f}s")

    # Threshold calibration: use the first 20% of test-healthy
    # sequences as a burn-in window. This reflects real operational
    # deployment, where the anomaly threshold is calibrated on a
    # rolling window of recent healthy telemetry rather than on a
    # training-time validation split whose distribution can drift
    # from live. The val split remains the training supervision
    # signal (early stopping) — it just isn't used to set the
    # threshold any more. The remaining 80% of test-healthy (held
    # out as X_health_eval_seq) is what every downstream metric is
    # computed on, so there's no leakage of the burn-in slice into
    # the quoted FAR or detection rates.
    n_burn = max(len(X_health_test_seq) // 5, 1000)
    n_burn = min(n_burn, len(X_health_test_seq))
    print(
        f"Threshold calibration: {n_burn:,} burn-in sequences "
        f"from test-healthy (first 20% of {len(X_health_test_seq):,})"
    )
    burn_errors = lstm_model.compute_reconstruction_error(X_health_test_seq[:n_burn])
    lstm_model.set_threshold(burn_errors, percentile=95.0)
    X_health_eval_seq = X_health_test_seq[n_burn:]
    print(
        f"  Held-out eval healthy: {len(X_health_eval_seq):,} sequences "
        f"(burn-in slice is not counted toward FAR or detection metrics)"
    )

    lstm_val_metrics = {
        "best_val_loss": float(
            min(lstm_model.val_losses_) if lstm_model.val_losses_ else 0.0
        ),
        "threshold_calibration": "test_healthy_burn_in_20pct",
        "burn_in_size": int(n_burn),
    }
    if len(X_health_eval_seq) > 0:
        health_errors = lstm_model.compute_reconstruction_error(X_health_eval_seq)
        health_preds = (health_errors > lstm_model.threshold_).astype(int)
        h_mean = float(health_errors.mean())

        if len(X_fail_alive_seq) > 0:
            fa_errors = lstm_model.compute_reconstruction_error(X_fail_alive_seq)
            fa_preds = (fa_errors > lstm_model.threshold_).astype(int)
            fa_mean = float(fa_errors.mean())
            sep_alive = fa_mean / max(h_mean, 1e-10)
            det_alive = float(fa_preds.mean())
        else:
            fa_errors = np.array([])
            fa_mean = 0.0
            sep_alive = float("nan")
            det_alive = float("nan")

        if len(X_fail_dead_seq) > 0:
            fd_errors = lstm_model.compute_reconstruction_error(X_fail_dead_seq)
            fd_mean = float(fd_errors.mean())
        else:
            fd_errors = np.array([])
            fd_mean = 0.0

        # sep_all keeps the old contaminated definition for continuity
        # with the pre-Phase-B sidecar — it concatenates alive and dead
        # failure errors and divides by healthy.
        if len(fa_errors) or len(fd_errors):
            f_all = np.concatenate([a for a in (fa_errors, fd_errors) if len(a)])
            f_all_mean = float(f_all.mean())
            sep_all = f_all_mean / max(h_mean, 1e-10)
            det_all = float(((f_all > lstm_model.threshold_).astype(int)).mean())
        else:
            f_all_mean = 0.0
            sep_all = float("nan")
            det_all = float("nan")

        print()
        print("LSTM-AE Test Results (val-calibrated threshold):")
        print(
            f"  Healthy sequences flagged:  {health_preds.sum()}/{len(health_preds)} "
            f"({health_preds.mean():.1%})"
        )
        print(
            f"  Mean error (healthy):       {h_mean:.6f}"
        )
        print(
            f"  Mean error (failure alive): {fa_mean:.6f}  "
            f"({len(fa_errors)} sequences)"
        )
        print(
            f"  Mean error (failure dead):  {fd_mean:.6f}  "
            f"({len(fd_errors)} sequences)"
        )
        print(f"  Separation ratio (alive):    {sep_alive:.3f}x  ← honest metric")
        print(f"  Separation ratio (all):      {sep_all:.3f}x  (includes shutdowns)")
        print(
            f"  Detection rate (alive):      "
            f"{det_alive if not np.isnan(det_alive) else 0:.1%}"
        )

        lstm_val_metrics.update({
            "test_healthy_far": float(health_preds.mean()),
            "test_mean_error_healthy": h_mean,
            "test_mean_error_failure_alive": fa_mean,
            "test_mean_error_failure_dead": fd_mean,
            "test_n_failure_alive": int(len(fa_errors)),
            "test_n_failure_dead": int(len(fd_errors)),
            "test_separation_ratio_alive": sep_alive,
            "test_separation_ratio_all": sep_all,
            "test_detection_rate_alive": det_alive,
            "test_detection_rate_all": det_all,
            # Legacy keys kept as aliases so the consistency check can
            # still match the sidecar shape from commit 7a460ac.
            "test_failure_detection_rate": det_all,
            "test_mean_error_failure": f_all_mean,
            "test_separation_ratio": sep_all,
        })

    lstm_model.save()
    n_test_rows = (
        len(X_fail_alive_seq) + len(X_fail_dead_seq) + len(X_health_test_seq)
    )
    save_model_metadata(
        model_path=MODELS_DIR / "lstm_ae.pt",
        model_type="lstm_autoencoder",
        n_train_rows=len(X_train_seq),
        n_val_rows=len(X_val_seq),
        n_test_rows=n_test_rows,
        val_metrics=lstm_val_metrics,
        feature_names=lstm_model.feature_names_,
        training_config={
            "input_dim": 9,
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
            "trainer": "scripts/train_lstm_only.py (standalone)",
        },
        extra={"features_version": FEATURES_VERSION},
    )
    print(f"\nDone in {time.time()-total_start:.1f}s")


if __name__ == "__main__":
    main()
