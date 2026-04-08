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
from src.models.lstm_autoencoder import AnomalyDetector
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
        "timestamp", "miner_id", "failure_type", "is_pre_failure",
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

    healthy_train = train_df[train_df["failure_type"] == "none"]
    healthy_val = val_df[val_df["failure_type"] == "none"]
    healthy_test = test_df[test_df["failure_type"] == "none"]
    failure_test = test_df[test_df["failure_type"] != "none"]
    print(
        f"Healthy rows - train: {len(healthy_train):,} "
        f"val: {len(healthy_val):,} test: {len(healthy_test):,}"
    )
    print(f"Failure rows in test: {len(failure_test):,}")

    lstm_model = AnomalyDetector(
        input_dim=6, seq_len=60, hidden_dim=64, latent_dim=32,
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

    print("Building test sequences (failure + healthy held out)...")
    t0 = time.time()
    X_fail_seq = lstm_model.prepare_sequences(failure_test, stride=5)
    X_health_test_seq = lstm_model.prepare_sequences(healthy_test, stride=5)
    print(
        f"  {len(X_fail_seq):,} failure + {len(X_health_test_seq):,} healthy "
        f"test sequences in {time.time()-t0:.1f}s"
    )

    # CRUCIAL: free pandas frames before LSTM fit to avoid OOM.
    del train_df, val_df, test_df
    del healthy_train, healthy_val, healthy_test, failure_test
    gc.collect()
    print("Freed upstream pandas frames before LSTM fit.")

    print(f"Training LSTM-AE: {len(X_train_seq):,} sequences, "
          f"up to {lstm_model.n_epochs} epochs")
    t0 = time.time()
    lstm_model.fit(X_train_seq, X_val_seq)
    print(f"Training time: {time.time()-t0:.1f}s")

    val_errors = lstm_model.compute_reconstruction_error(X_val_seq)
    lstm_model.set_threshold(val_errors, percentile=95.0)

    lstm_val_metrics = {"best_val_loss": float(min(lstm_model.val_losses_) if lstm_model.val_losses_ else 0.0)}
    if len(X_fail_seq) > 0 and len(X_health_test_seq) > 0:
        fail_errors = lstm_model.compute_reconstruction_error(X_fail_seq)
        health_errors = lstm_model.compute_reconstruction_error(X_health_test_seq)
        fail_preds = (fail_errors > lstm_model.threshold_).astype(int)
        health_preds = (health_errors > lstm_model.threshold_).astype(int)
        sep = float(fail_errors.mean() / max(health_errors.mean(), 1e-10))

        print()
        print("LSTM-AE Test Results (val-calibrated threshold):")
        print(f"  Healthy sequences flagged:  {health_preds.sum()}/{len(health_preds)} "
              f"({health_preds.mean():.1%})")
        print(f"  Failure sequences flagged:  {fail_preds.sum()}/{len(fail_preds)} "
              f"({fail_preds.mean():.1%})")
        print(f"  Mean error (healthy): {health_errors.mean():.6f}")
        print(f"  Mean error (failure): {fail_errors.mean():.6f}")
        print(f"  Separation ratio:     {sep:.2f}x")

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
        n_test_rows=(len(X_fail_seq) + len(X_health_test_seq)) if len(X_fail_seq) > 0 else 0,
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
            "trainer": "scripts/train_lstm_only.py (standalone)",
        },
        extra={"features_version": FEATURES_VERSION},
    )
    print(f"\nDone in {time.time()-total_start:.1f}s")


if __name__ == "__main__":
    main()
