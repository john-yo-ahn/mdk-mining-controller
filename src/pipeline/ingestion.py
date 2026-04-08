"""
Load and validate raw telemetry data from various sources.
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd

from ..config import RAW_DIR, TELEMETRY_COLUMNS


def load_telemetry(
    filepath: Optional[Path] = None,
    fmt: str = "parquet",
) -> pd.DataFrame:
    """Load raw telemetry from file. Default: data/raw/mining_telemetry.parquet"""
    if filepath is None:
        filepath = RAW_DIR / f"mining_telemetry.{fmt}"

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Telemetry file not found: {filepath}")

    if fmt == "parquet" or filepath.suffix == ".parquet":
        df = pd.read_parquet(filepath, engine="pyarrow")
    elif fmt == "csv" or filepath.suffix == ".csv":
        df = pd.read_csv(filepath, parse_dates=["timestamp"])
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    validate_schema(df)
    df = df.sort_values(["miner_id", "timestamp"]).reset_index(drop=True)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Verify required columns exist."""
    required = {"timestamp", "miner_id", "hashrate_th", "temperature_c", "power_w"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def get_miner_ids(df: pd.DataFrame) -> List[str]:
    return sorted(df["miner_id"].unique())


def get_time_range(df: pd.DataFrame) -> tuple:
    return df["timestamp"].min(), df["timestamp"].max()


def filter_by_miners(df: pd.DataFrame, miner_ids: List[str]) -> pd.DataFrame:
    return df[df["miner_id"].isin(miner_ids)].copy()


def filter_by_time(
    df: pd.DataFrame,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if start:
        mask &= df["timestamp"] >= start
    if end:
        mask &= df["timestamp"] <= end
    return df[mask].copy()
