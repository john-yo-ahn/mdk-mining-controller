"""
True Efficiency (TE) KPI — goes beyond simple J/TH by layering
four operational variables required by the assignment §3.1.b:

  1. cooling system power consumption  →  α_cooling + β_infra in the denominator
  2. chip voltage                        →  voltage_stability factor in the numerator
  3. environmental conditions            →  ambient temperature penalty on te_adjusted
  4. device operating mode               →  operating_mode_factor in the numerator

The three layers are:

  te_base     : hashrate × voltage_stability × mode_factor
                divided by site-scaled power (chip × 1.20 by default)
  te_adjusted : te_base × environmental temperature penalty
  te_health   : te_adjusted × hashrate realization (0..1 of nameplate)

te_health is the headline operator-facing metric because it folds
all four §3.1.b variables and hardware degradation into a single
number. te_base and te_adjusted are exposed for introspection
(debugging / feature engineering) rather than as stand-alone
operator KPIs.
"""

from typing import Dict

import numpy as np
import pandas as pd

from ..config import TEConfig, DEFAULT_MINER_SPECS


# ─── Core formula pieces (shared by vector + scalar paths) ──────────────

def _voltage_stability_factor(
    voltage_v,
    voltage_default_v,
    penalty_coefficient: float,
):
    """
    Penalty for deviating from the chip's spec voltage.

    Formula:  clip(1 - k × |V - V_default| / V_default, 0, 1)

    At spec voltage the factor is 1.0; deviation in either direction
    (over- or under-volting) reduces the factor linearly. k defaults
    to 0.5 via `TEConfig.voltage_penalty_coefficient`. V_default is
    looked up per hardware model from DEFAULT_MINER_SPECS.

    Works for both pd.Series and plain floats.
    """
    safe_default = np.where(
        np.asarray(voltage_default_v) > 0,
        voltage_default_v,
        1.0,
    )
    deviation = np.abs(np.asarray(voltage_v) - np.asarray(voltage_default_v)) / safe_default
    factor = 1.0 - penalty_coefficient * deviation
    return np.clip(factor, 0.0, 1.0)


def _operating_mode_factor(
    operating_mode,
    weights: dict,
):
    """
    Lookup factor for device operating mode (Normal / Idle / Shutdown).

    Unknown modes fall back to 1.0 so a future mode label never
    silently zeros out the KPI for every row in the fleet.

    Works for both pd.Series (uses vectorized .map) and plain strings
    (uses dict.get).
    """
    if isinstance(operating_mode, pd.Series):
        return operating_mode.map(weights).fillna(1.0).to_numpy()
    return float(weights.get(operating_mode, 1.0))


# ─── Vectorized public API (per-row computation over a DataFrame) ──────

def compute_te_base(
    hashrate_th: pd.Series,
    power_chip_w: pd.Series,
    voltage_v: pd.Series,
    voltage_default_v: pd.Series,
    operating_mode: pd.Series,
    config: TEConfig = TEConfig(),
) -> pd.Series:
    """
    True-efficiency base layer, incorporating:
      1. cooling + infrastructure overhead (α_cooling + β_infra)
      2. chip-voltage stability factor
      4. device operating mode factor

    Formula:
        te_base = hashrate × voltage_stability × mode_factor
                  / (power_chip × (1 + α + β))

    Units: TH/s per W of site-scaled power (higher is better).
    """
    v_factor = _voltage_stability_factor(
        voltage_v, voltage_default_v, config.voltage_penalty_coefficient,
    )
    mode_factor = _operating_mode_factor(operating_mode, config.operating_mode_weights)

    total_power = power_chip_w * (1.0 + config.alpha_cooling + config.beta_infra)
    effective_hashrate = hashrate_th * v_factor * mode_factor
    return np.where(
        total_power > 0,
        effective_hashrate / total_power,
        0.0,
    )


def compute_te_adjusted(
    te_base: pd.Series,
    ambient_temp_c: pd.Series,
    config: TEConfig = TEConfig(),
) -> pd.Series:
    """
    Environmental adjustment (§3.1.b variable #3): penalize hot
    ambient conditions. A miner at 35°C ambient with temp_baseline=25
    and delta_temp=0.008 gets a 8% haircut vs the same miner at 25°C.
    """
    penalty = 1.0 - config.delta_temp * np.maximum(0, ambient_temp_c - config.temp_baseline_c)
    return te_base * penalty


def compute_te_health(
    te_adjusted: pd.Series,
    hashrate_actual_th: pd.Series,
    hashrate_nameplate_th: pd.Series,
) -> pd.Series:
    """
    Degradation awareness: te_health = te_adjusted × hashrate_realization.

    realization = clip(hashrate_actual / nameplate, 0, 1).
    A chip delivering 80% of its spec hashrate gets its TE scaled to
    80% of its te_adjusted value, regardless of how clean its
    instantaneous power draw looks.
    """
    realization = np.where(
        hashrate_nameplate_th > 0,
        np.clip(hashrate_actual_th / hashrate_nameplate_th, 0.0, 1.0),
        0.0,
    )
    return te_adjusted * realization


def compute_all_te_variants(
    df: pd.DataFrame,
    config: TEConfig = TEConfig(),
) -> pd.DataFrame:
    """
    Compute and append all TE variants to a telemetry DataFrame.

    Expects the DataFrame to carry (at minimum):
      - hashrate_th, power_w, voltage_v, ambient_temperature_c
      - operating_mode (string)
      - model (the hardware-family short token, e.g. "Pro", "M56S",
        "M63", "XP" — looked up in DEFAULT_MINER_SPECS to resolve
        voltage_default_v and hashrate_nameplate_th)

    Adds these columns:
      hashrate_nameplate_th, hashrate_realization, voltage_default_v,
      te_base, te_adjusted, te_health
    """
    df = df.copy()

    # Per-model nameplate + default voltage lookup.
    #
    # The synthetic generator writes the short token ("Pro", "M56S",
    # "M63", "XP") into the `model` column of batch telemetry, so
    # _spec_short_key() gives us the right key for that path.
    #
    # But the live CLI bridge (src/cli/ai_bridge.py:MinerBuffer.to_dataframe)
    # sets model to the full spec name ("Antminer S21 Pro"), and
    # src/cli/simulation.py passes the same full name into register_miner.
    # To make both paths resolve without requiring the caller to know
    # which convention to use, we build each lookup map with BOTH keys:
    # the short token AND the full spec.model_name pointing at the
    # same value.
    nameplate_map: Dict[str, float] = {}
    voltage_default_map: Dict[str, float] = {}
    for k, spec in DEFAULT_MINER_SPECS.items():
        short = _spec_short_key(k, spec)
        full = str(spec.model_name)
        nameplate_map[short] = spec.hashrate_nameplate_th
        nameplate_map[full] = spec.hashrate_nameplate_th
        voltage_default_map[short] = spec.voltage_default_v
        voltage_default_map[full] = spec.voltage_default_v

    df["hashrate_nameplate_th"] = df["model"].map(nameplate_map)
    df["voltage_default_v"] = df["model"].map(voltage_default_map)

    # Realization is a precomputed column so downstream code that reads
    # it (feature builder, reporting) doesn't have to recompute.
    df["hashrate_realization"] = np.where(
        df["hashrate_nameplate_th"] > 0,
        np.clip(df["hashrate_th"] / df["hashrate_nameplate_th"], 0.0, 1.0),
        0.0,
    )

    df["te_base"] = compute_te_base(
        df["hashrate_th"],
        df["power_w"],
        df["voltage_v"],
        df["voltage_default_v"],
        df["operating_mode"],
        config,
    )
    df["te_adjusted"] = compute_te_adjusted(
        df["te_base"], df["ambient_temperature_c"], config,
    )
    df["te_health"] = compute_te_health(
        df["te_adjusted"], df["hashrate_th"], df["hashrate_nameplate_th"],
    )
    return df


def _spec_short_key(full_key: str, spec) -> str:
    """
    DEFAULT_MINER_SPECS is keyed by labels like "S21_Pro", but the
    synthetic generator writes the trailing token of spec.model_name
    (e.g. "Antminer S21 Pro" → "Pro") into the DataFrame's `model`
    column. Resolve both to the same short token here so
    compute_all_te_variants' lookup map matches what's in the data.
    """
    parts = str(spec.model_name).split()
    return parts[-1] if len(parts) > 1 else str(spec.model_name)


def compute_fleet_te_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-miner TE aggregates: mean, std, min, trend slope.

    Note: we no longer filter out Shutdown rows here — that used to
    be a pre-Level-1 workaround because the old TE formula didn't
    know about operating_mode and would compute nonsense TE values
    for shutdown miners. After Level 1, Shutdown/Idle rows naturally
    have te_health=0 via the operating_mode_factor, so they are
    excluded automatically from any threshold-based reporting
    downstream without a pre-filter.
    """
    summary = df.groupby("miner_id").agg(
        model=("model", "first"),
        te_base_mean=("te_base", "mean"),
        te_adjusted_mean=("te_adjusted", "mean"),
        te_health_mean=("te_health", "mean"),
        te_health_std=("te_health", "std"),
        te_health_min=("te_health", "min"),
        jth_mean=("efficiency_jth", "mean"),
        avg_temp=("temperature_c", "mean"),
        max_temp=("temperature_c", "max"),
        hashrate_realization_mean=("hashrate_realization", "mean"),
        failure_type=(
            "failure_type",
            lambda x: x[x != "none"].iloc[-1] if (x != "none").any() else "none",
        ),
        n_readings=("te_health", "count"),
    ).reset_index()

    return summary


# ─── Scalar helpers (for the live simulator, which operates per-tick) ──

def compute_te_base_scalar(
    hashrate_th: float,
    power_chip_w: float,
    voltage_v: float,
    voltage_default_v: float,
    operating_mode: str,
    config: TEConfig = TEConfig(),
) -> float:
    """
    Single-row / single-miner variant of compute_te_base. Used by
    the CLI simulator (`src/cli/simulation.py`) so the dashboard's
    TE numbers use the same math as the batch pipeline — before
    this helper existed, simulation.py reimplemented the formula
    inline and drifted (it was missing β_infra, giving TE values
    ~4.3% higher than the batch pipeline for the same telemetry).
    """
    if power_chip_w <= 0:
        return 0.0

    v_factor = float(_voltage_stability_factor(
        voltage_v, voltage_default_v, config.voltage_penalty_coefficient,
    ))
    mode_factor = float(_operating_mode_factor(
        operating_mode, config.operating_mode_weights,
    ))
    total_power = power_chip_w * (1.0 + config.alpha_cooling + config.beta_infra)
    return (hashrate_th * v_factor * mode_factor) / total_power


def compute_te_adjusted_scalar(
    te_base: float,
    ambient_temp_c: float,
    config: TEConfig = TEConfig(),
) -> float:
    """Scalar variant of compute_te_adjusted."""
    penalty = 1.0 - config.delta_temp * max(0.0, ambient_temp_c - config.temp_baseline_c)
    return te_base * penalty


def compute_te_health_scalar(
    te_adjusted: float,
    hashrate_actual_th: float,
    hashrate_nameplate_th: float,
) -> float:
    """Scalar variant of compute_te_health."""
    if hashrate_nameplate_th <= 0:
        return 0.0
    realization = max(0.0, min(1.0, hashrate_actual_th / hashrate_nameplate_th))
    return te_adjusted * realization
