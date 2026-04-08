"""
LSTM-AE experiment harness.

One entry point, three modes. Use this for iteration; use
scripts/train_lstm_only.py for the canonical production run whose
sidecar the consistency_check measures against.

Modes:
    toy   ~60 s   5 miners, stride=20, hidden=16, 5 epochs, batch=256.
                  Smoke test only. Catches syntax/shape/scaler bugs.
    fast  ~3 min  full fleet, stride=20, hidden=64, 8 epochs, batch=512.
                  Used for A/B comparison between experiments. Metrics
                  are computed on a 20% sample of the test set to
                  further speed up the inference step.
    full  ~25 min full fleet, stride=5, hidden=64, 30 epochs, batch=256.
                  Same knobs as scripts/train_lstm_only.py but does not
                  overwrite data/models/lstm_ae.pt or its sidecar — it
                  writes to /tmp so iteration runs don't stomp the
                  canonical checkpoint.

Usage:
    uv run python -m scripts.lstm_experiment_harness --mode=toy
    uv run python -m scripts.lstm_experiment_harness --mode=fast
    uv run python -m scripts.lstm_experiment_harness --mode=full

The output is a JSON summary printed to stdout with the same shape
as the sidecar val_metrics dict, plus a `mode`, `wall_clock_s` and
a short `gate_status` line that compares the result against the
phase gate described in
/Users/john/.claude/plans/frolicking-stirring-flask.md.
"""

from __future__ import annotations

import argparse
import gc
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR
from src.pipeline.features import FEATURES_VERSION, split_temporal_tvt
from src.models.lstm_autoencoder import AnomalyDetector, filter_alive_rows


@dataclass(frozen=True)
class ModeConfig:
    name: str
    n_miners: Optional[int]   # None → full fleet
    stride: int
    hidden_dim: int
    latent_dim: int
    n_epochs: int
    batch_size: int
    patience: int
    test_sample_frac: float   # 1.0 → evaluate on all test sequences
    # Phase gate this mode targets (printed at the end for orientation).
    # sep_alive threshold; the harness never hard-fails on this, it just
    # reports pass/fail so the caller can decide whether to proceed.
    sep_alive_gate: Optional[float]


MODES = {
    "toy": ModeConfig(
        name="toy",
        n_miners=5,
        stride=20,
        hidden_dim=16,
        latent_dim=8,
        n_epochs=5,
        batch_size=256,
        patience=2,
        test_sample_frac=0.2,
        sep_alive_gate=None,  # toy is smoke test only
    ),
    "fast": ModeConfig(
        name="fast",
        n_miners=None,
        stride=20,
        hidden_dim=64,
        latent_dim=32,
        n_epochs=8,
        batch_size=512,
        patience=2,
        test_sample_frac=0.2,
        sep_alive_gate=2.0,  # the Phase C gate; earlier phases set lower
    ),
    "full": ModeConfig(
        name="full",
        n_miners=None,
        stride=5,
        hidden_dim=64,
        latent_dim=32,
        n_epochs=30,
        batch_size=256,
        patience=4,
        test_sample_frac=1.0,
        sep_alive_gate=2.0,
    ),
}


# ── Data loading ──────────────────────────────────────────────────

NEEDED_COLS = [
    "timestamp", "miner_id", "model", "failure_type", "is_pre_failure",
    "clock_frequency_mhz", "voltage_v", "hashrate_th",
    "temperature_c", "power_w", "ambient_temperature_c",
]


def load_and_split():
    cache = PROCESSED_DIR / f"features.v{FEATURES_VERSION}.parquet"
    if not cache.exists():
        raise SystemExit(
            f"Feature cache missing: {cache}\n"
            "Run `uv run python -m src.run_pipeline` first."
        )
    print(f"Loading features from {cache}")
    df = pd.read_parquet(cache, columns=NEEDED_COLS)
    print(f"  {len(df):,} rows × {len(df.columns)} cols, "
          f"{df['miner_id'].nunique()} miners, "
          f"{df['model'].nunique()} hardware models")

    train_df, val_df, test_df = split_temporal_tvt(
        df, train_pos_fraction=0.55, val_pos_fraction=0.15,
    )
    del df
    gc.collect()
    return train_df, val_df, test_df


def subsample_miners(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n: int,
) -> list[str]:
    """Return a list of n miner IDs with two guarantees:

    1. At least 1 miner from each hardware model (so per-model scalers
       built in Phase A have something to test against).
    2. At least 2 miners that actually fail during the test window (so
       the separation metric has real failure sequences to compare).

    Picks are deterministic (sorted inputs, no random state)."""
    all_miners = sorted(train_df["miner_id"].unique())
    failing_in_test = sorted(
        test_df[test_df["failure_type"] != "none"]["miner_id"].unique()
    )
    print(
        f"  Fleet: {len(all_miners)} total miners, "
        f"{len(failing_in_test)} fail in test window"
    )

    # Pick at least 2 failing miners from different hardware models when
    # possible. Falls back to whatever failing miners exist if the fleet
    # is smaller.
    chosen: list[str] = []
    chosen_models: set[str] = set()
    for mid in failing_in_test:
        if len(chosen) >= 2:
            break
        m = train_df.loc[train_df["miner_id"] == mid, "model"].iloc[0]
        if m not in chosen_models:
            chosen.append(mid)
            chosen_models.add(m)
    # If we still need failing miners after model-diversity pass, take any.
    for mid in failing_in_test:
        if len(chosen) >= 2:
            break
        if mid not in chosen:
            chosen.append(mid)

    # Fill up to n with healthy miners from models we haven't seen yet.
    model_to_miners: dict[str, list[str]] = {}
    for mid in all_miners:
        m = train_df.loc[train_df["miner_id"] == mid, "model"].iloc[0]
        model_to_miners.setdefault(m, []).append(mid)
    for m, miners in sorted(model_to_miners.items()):
        if len(chosen) >= n:
            break
        if m in chosen_models:
            continue
        # Pick the first miner of this model that isn't already chosen
        for mid in miners:
            if mid not in chosen:
                chosen.append(mid)
                chosen_models.add(m)
                break
    # Any remaining slots go to any miner not yet picked.
    for mid in all_miners:
        if len(chosen) >= n:
            break
        if mid not in chosen:
            chosen.append(mid)

    return chosen[:n]


# ── Metric computation ───────────────────────────────────────────

def compute_metrics(
    lstm: AnomalyDetector,
    X_healthy_eval: np.ndarray,
    X_failure_alive: np.ndarray,
    X_failure_dead: np.ndarray,
) -> dict:
    """Produce sep_all, sep_alive, detection_rate_alive, healthy_far."""
    he = lstm.compute_reconstruction_error(X_healthy_eval)
    fa = lstm.compute_reconstruction_error(X_failure_alive) if len(X_failure_alive) else np.array([])
    fd = lstm.compute_reconstruction_error(X_failure_dead) if len(X_failure_dead) else np.array([])

    h_mean = float(he.mean()) if len(he) else 0.0
    fa_mean = float(fa.mean()) if len(fa) else 0.0
    fd_mean = float(fd.mean()) if len(fd) else 0.0

    # "all" = alive + dead concatenated; matches the old single-metric view
    if len(fa) and len(fd):
        f_all = np.concatenate([fa, fd])
    elif len(fa):
        f_all = fa
    else:
        f_all = fd
    f_all_mean = float(f_all.mean()) if len(f_all) else 0.0

    eps = 1e-10
    sep_all = f_all_mean / max(h_mean, eps)
    sep_alive = fa_mean / max(h_mean, eps) if len(fa) else float("nan")

    thr = float(lstm.threshold_) if lstm.threshold_ is not None else float("nan")
    far = float((he > thr).mean()) if len(he) and not np.isnan(thr) else float("nan")
    det_rate_alive = (
        float((fa > thr).mean()) if len(fa) and not np.isnan(thr) else float("nan")
    )
    det_rate_all = (
        float((f_all > thr).mean()) if len(f_all) and not np.isnan(thr) else float("nan")
    )

    return {
        "threshold": thr,
        "n_healthy_eval": int(len(he)),
        "n_failure_alive": int(len(fa)),
        "n_failure_dead": int(len(fd)),
        "healthy_mean_error": h_mean,
        "failure_alive_mean_error": fa_mean,
        "failure_dead_mean_error": fd_mean,
        "sep_all": sep_all,
        "sep_alive": sep_alive,
        "healthy_far": far,
        "detection_rate_alive": det_rate_alive,
        "detection_rate_all": det_rate_all,
    }


# ── Main run ──────────────────────────────────────────────────────

def run(cfg: ModeConfig) -> dict:
    total_start = time.time()
    print(f"\n=== Running mode: {cfg.name} ===")
    print(f"  stride={cfg.stride}  hidden_dim={cfg.hidden_dim}  "
          f"latent_dim={cfg.latent_dim}  batch_size={cfg.batch_size}  "
          f"epochs={cfg.n_epochs}  patience={cfg.patience}")

    train_df, val_df, test_df = load_and_split()

    if cfg.n_miners is not None:
        kept_ids = set(subsample_miners(train_df, test_df, cfg.n_miners))
        train_df = train_df[train_df["miner_id"].isin(kept_ids)].reset_index(drop=True)
        val_df   = val_df[val_df["miner_id"].isin(kept_ids)].reset_index(drop=True)
        test_df  = test_df[test_df["miner_id"].isin(kept_ids)].reset_index(drop=True)
        print(f"  Subsampled to {len(kept_ids)} miners: {sorted(kept_ids)}")

    # Apply the offline-row filter to the healthy splits so the AE
    # trains only on "actually healthy" telemetry. Failure sequences
    # are NOT filtered here — the split into alive vs dead happens
    # below and determines sep_alive vs sep_all.
    healthy_train = filter_alive_rows(train_df[train_df["failure_type"] == "none"])
    healthy_val   = filter_alive_rows(val_df[val_df["failure_type"] == "none"])
    healthy_test  = filter_alive_rows(test_df[test_df["failure_type"] == "none"])
    failure_test  = test_df[test_df["failure_type"] != "none"].reset_index(drop=True)

    # Split failure_test into alive vs dead so the harness can report
    # the honest separation metric (sep_alive) separately from the
    # trivial metric (sep_all, which lumps shutdown sequences in).
    fail_alive_mask = (failure_test["hashrate_th"] > 1.0) & (failure_test["voltage_v"] > 0.05)
    failure_alive = failure_test[fail_alive_mask].reset_index(drop=True)
    failure_dead  = failure_test[~fail_alive_mask].reset_index(drop=True)
    print(
        f"  failure_test: {len(failure_test):,} rows, "
        f"{len(failure_alive):,} alive ({100*len(failure_alive)/max(len(failure_test),1):.1f}%), "
        f"{len(failure_dead):,} dead (shutdown)"
    )

    # Sample test set for faster inference during iteration.
    # Full mode leaves test_sample_frac=1.0 for canonical numbers.
    if cfg.test_sample_frac < 1.0:
        rng = np.random.default_rng(42)
        def _sample(df):
            if len(df) == 0:
                return df
            n = max(int(len(df) * cfg.test_sample_frac), 1)
            idx = rng.choice(len(df), size=n, replace=False)
            return df.iloc[np.sort(idx)].reset_index(drop=True)
        healthy_test_eval  = _sample(healthy_test)
        failure_alive_eval = _sample(failure_alive)
        failure_dead_eval  = _sample(failure_dead)
    else:
        healthy_test_eval  = healthy_test
        failure_alive_eval = failure_alive
        failure_dead_eval  = failure_dead

    print(
        f"  Healthy train: {len(healthy_train):,} | val: {len(healthy_val):,} | "
        f"test (eval): {len(healthy_test_eval):,}  |  "
        f"Failure alive (eval): {len(failure_alive_eval):,}  |  "
        f"Failure dead (eval): {len(failure_dead_eval):,}"
    )

    lstm = AnomalyDetector(
        input_dim=6, seq_len=60,
        hidden_dim=cfg.hidden_dim, latent_dim=cfg.latent_dim,
        n_layers=2, n_epochs=cfg.n_epochs, batch_size=cfg.batch_size,
        early_stopping_patience=cfg.patience,
    )
    print(f"  Device: {lstm.device}")

    lstm.fit_scaler(healthy_train)

    t0 = time.time()
    X_train = lstm.prepare_sequences(healthy_train, stride=cfg.stride)
    X_val = lstm.prepare_sequences(healthy_val, stride=cfg.stride)
    X_health_test = lstm.prepare_sequences(healthy_test_eval, stride=cfg.stride)
    X_fail_alive = lstm.prepare_sequences(failure_alive_eval, stride=cfg.stride) \
        if len(failure_alive_eval) else np.empty((0, 60, 6), dtype=np.float32)
    X_fail_dead = lstm.prepare_sequences(failure_dead_eval, stride=cfg.stride) \
        if len(failure_dead_eval) else np.empty((0, 60, 6), dtype=np.float32)
    print(
        f"  Sequences built in {time.time()-t0:.1f}s: "
        f"train={len(X_train):,} val={len(X_val):,} "
        f"h_test={len(X_health_test):,} "
        f"f_alive={len(X_fail_alive):,} f_dead={len(X_fail_dead):,}"
    )

    if len(X_train) == 0 or len(X_val) == 0:
        raise SystemExit(
            "Empty sequences — not enough rows per miner for seq_len=60. "
            "Try a smaller subsample or smaller stride."
        )

    # Free the upstream DataFrames before fit to keep RSS down in fast/full modes.
    del train_df, val_df, test_df
    del healthy_train, healthy_val, healthy_test, failure_test
    del failure_alive, failure_dead
    del healthy_test_eval, failure_alive_eval, failure_dead_eval
    gc.collect()

    print(f"  Training on {len(X_train):,} sequences...")
    t0 = time.time()
    lstm.fit(X_train, X_val)
    print(f"  Training wall-clock: {time.time()-t0:.1f}s")

    val_errs = lstm.compute_reconstruction_error(X_val)
    lstm.set_threshold(val_errs, percentile=95.0)

    metrics = compute_metrics(
        lstm,
        X_healthy_eval=X_health_test,
        X_failure_alive=X_fail_alive,
        X_failure_dead=X_fail_dead,
    )

    # Determinism spot-check: save to /tmp, reload, compare one metric.
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td) / "lstm_exp.pt"
        lstm.save(tmp_path)
        reloaded = AnomalyDetector.load(tmp_path)
        reload_metrics = compute_metrics(
            reloaded,
            X_healthy_eval=X_health_test,
            X_failure_alive=X_fail_alive,
            X_failure_dead=X_fail_dead,
        )
        # Compare sep_all because sep_alive can be NaN when there are
        # no failing miners in the sample.
        drift_a = abs(metrics["sep_all"] - reload_metrics["sep_all"])
        metrics["reload_sep_drift"] = float(drift_a)
        if drift_a > 1e-3:
            print(f"  WARNING: reload drift {drift_a:.6e} — determinism broken")

    metrics["mode"] = cfg.name
    metrics["wall_clock_s"] = round(time.time() - total_start, 1)
    metrics["n_train_sequences"] = int(len(X_train))
    metrics["n_val_sequences"] = int(len(X_val))
    if cfg.sep_alive_gate is not None:
        passed = (
            not np.isnan(metrics["sep_alive"])
            and metrics["sep_alive"] >= cfg.sep_alive_gate
        )
        metrics["gate_status"] = (
            f"{'PASS' if passed else 'FAIL'} "
            f"(sep_alive={metrics['sep_alive']:.3f} "
            f"vs gate {cfg.sep_alive_gate:.2f})"
        )
    else:
        metrics["gate_status"] = "no-gate (smoke test)"

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=sorted(MODES.keys()), default="toy",
        help="toy / fast / full",
    )
    args = parser.parse_args()

    cfg = MODES[args.mode]
    metrics = run(cfg)

    print("\n=== Summary ===")
    print(json.dumps(metrics, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
