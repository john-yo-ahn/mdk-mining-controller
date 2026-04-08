"""
Clean, validate, and normalize raw telemetry.
"""

import numpy as np
import pandas as pd
from ..config import SENSOR_COLUMNS


# Physical bounds for plausibility checks
# Note: voltage lower bound is 0 to allow shutdown state (v=0)
# Frequency lower bound is 0 to allow shutdown state
BOUNDS = {
    "temperature_c": (0, 130),
    "voltage_v": (0.0, 1.0),
    "hashrate_th": (0, 600),
    "power_w": (0, 15000),
    "clock_frequency_mhz": (0, 1000),
    "ambient_temperature_c": (-20, 60),
}


def handle_missing_values(df: pd.DataFrame, max_gap_minutes: int = 5) -> pd.DataFrame:
    """Forward-fill sensor columns for gaps up to max_gap_minutes."""
    df = df.copy()
    sensor_cols = [c for c in SENSOR_COLUMNS if c in df.columns]

    for miner_id in df["miner_id"].unique():
        mask = df["miner_id"] == miner_id
        df.loc[mask, sensor_cols] = (
            df.loc[mask, sensor_cols].ffill(limit=max_gap_minutes)
        )

    df = df.dropna(subset=["hashrate_th", "temperature_c", "power_w"])
    return df


def apply_plausibility_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Clamp values to physical bounds. Flag out-of-bounds readings."""
    df = df.copy()
    n_violations = 0

    for col, (lo, hi) in BOUNDS.items():
        if col not in df.columns:
            continue
        oob = (df[col] < lo) | (df[col] > hi)
        n_violations += oob.sum()
        df[col] = df[col].clip(lo, hi)

    if n_violations > 0:
        print(f"  Plausibility: clamped {n_violations:,} out-of-bounds values")

    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add computed columns: efficiency, temperature delta."""
    df = df.copy()
    df["efficiency_jth"] = np.where(
        df["hashrate_th"] > 0,
        df["power_w"] / df["hashrate_th"],
        np.nan,
    )
    if "ambient_temperature_c" in df.columns:
        df["temp_delta_c"] = df["temperature_c"] - df["ambient_temperature_c"]

    df["hashrate_realization"] = np.nan
    # Will be filled per-model in the feature step
    return df


def normalize_per_device(
    df: pd.DataFrame,
    columns: list = None,
) -> pd.DataFrame:
    """Z-score normalization grouped by miner_id."""
    df = df.copy()
    if columns is None:
        columns = ["temperature_c", "hashrate_th", "power_w", "efficiency_jth"]
    columns = [c for c in columns if c in df.columns]

    for col in columns:
        stats = df.groupby("miner_id")[col].transform
        mean = stats("mean")
        std = stats("std").replace(0, 1)
        df[f"{col}_zscore"] = (df[col] - mean) / std

    return df


def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing: missing values, plausibility, derived, normalize."""
    print("Preprocessing pipeline:")
    print(f"  Input: {len(df):,} rows")

    df = handle_missing_values(df)
    print(f"  After missing value handling: {len(df):,} rows")

    df = apply_plausibility_checks(df)
    df = add_derived_columns(df)
    df = normalize_per_device(df)

    print(f"  Output: {len(df):,} rows, {len(df.columns)} columns")
    return df
