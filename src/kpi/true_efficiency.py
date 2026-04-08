"""
True Efficiency KPI — goes beyond simple J/TH.
Incorporates cooling overhead, environmental conditions, and degradation.
"""

import numpy as np
import pandas as pd
from ..config import TEConfig, DEFAULT_MINER_SPECS


def compute_te_base(
    hashrate_th: pd.Series,
    power_chip_w: pd.Series,
    config: TEConfig = TEConfig(),
) -> pd.Series:
    """
    TE = hashrate / (P_chip + P_cooling*alpha + P_infra*beta)
    Units: TH/s per Watt (higher is better).
    """
    total_power = (
        power_chip_w
        * (1 + config.alpha_cooling + config.beta_infra)
    )
    return np.where(total_power > 0, hashrate_th / total_power, 0.0)


def compute_te_adjusted(
    te_base: pd.Series,
    ambient_temp_c: pd.Series,
    config: TEConfig = TEConfig(),
) -> pd.Series:
    """Environmental adjustment: penalize hot conditions."""
    penalty = 1 - config.delta_temp * np.maximum(0, ambient_temp_c - config.temp_baseline_c)
    return te_base * penalty


def compute_te_health(
    te_adjusted: pd.Series,
    hashrate_actual_th: pd.Series,
    hashrate_nameplate_th: pd.Series,
) -> pd.Series:
    """Degradation-aware TE using hashrate realization."""
    realization = np.where(
        hashrate_nameplate_th > 0,
        np.clip(hashrate_actual_th / hashrate_nameplate_th, 0, 1),
        0,
    )
    return te_adjusted * realization


def compute_all_te_variants(
    df: pd.DataFrame,
    config: TEConfig = TEConfig(),
) -> pd.DataFrame:
    """Compute and append all TE variants to a telemetry DataFrame."""
    df = df.copy()

    # Map nameplate hashrate per model
    nameplate_map = {
        k: v.hashrate_nameplate_th for k, v in DEFAULT_MINER_SPECS.items()
    }
    df["hashrate_nameplate_th"] = df["model"].map(nameplate_map)
    df["hashrate_realization"] = np.where(
        df["hashrate_nameplate_th"] > 0,
        np.clip(df["hashrate_th"] / df["hashrate_nameplate_th"], 0, 1),
        0,
    )

    df["te_base"] = compute_te_base(df["hashrate_th"], df["power_w"], config)
    df["te_adjusted"] = compute_te_adjusted(
        df["te_base"], df["ambient_temperature_c"], config,
    )
    df["te_health"] = compute_te_health(
        df["te_adjusted"], df["hashrate_th"], df["hashrate_nameplate_th"],
    )
    return df


def compute_fleet_te_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-miner TE aggregates: mean, std, min, trend slope."""
    active = df[df["operating_mode"] != "Shutdown"].copy()

    summary = active.groupby("miner_id").agg(
        model=("model", "first"),
        te_health_mean=("te_health", "mean"),
        te_health_std=("te_health", "std"),
        te_health_min=("te_health", "min"),
        jth_mean=("efficiency_jth", "mean"),
        avg_temp=("temperature_c", "mean"),
        max_temp=("temperature_c", "max"),
        hashrate_realization_mean=("hashrate_realization", "mean"),
        failure_type=("failure_type", lambda x: x[x != "none"].iloc[-1] if (x != "none").any() else "none"),
        n_readings=("te_health", "count"),
    ).reset_index()

    return summary
