"""
Feature engineering for ML models.

Produces the full feature matrix consumed by XGBoost and LSTM.

Features computed:
- Cross-signal ratios (J/TH, temp/watt, etc.)
- Multi-timescale rolling stats (2m, 15m, 1h, 6h, 1d, 7d) — mean, std, min, max
- Rate of change (first derivative)
- Degradation slope (rolling OLS on J/TH)
- Trend features (7-day OLS slope)
- Variance trends (is std growing?)
- Cross-signal correlations (temp-power, hashrate-voltage)
- Autocorrelation lag-1 (signal smoothness)
- Peak counts (intermittent fault detection)
- Diurnal amplitude (24h cycle amplitude)
- Cross-miner features (neighbor deltas, container rank)
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import FeatureConfig, DEFAULT_MINER_SPECS, FEATURE_EXCLUDE_COLUMNS


# Cache-invalidation marker. BUMP THIS any time you change the feature
# computation code below — adding a column, renaming one, changing a
# window size, fixing a bug. The pipeline looks for
#   data/processed/features.v{FEATURES_VERSION}.parquet
# so a version bump guarantees a fresh rebuild even when the raw parquet
# is unchanged. Without this, stale caches cause silent train/inference
# drift that is extremely hard to debug.
#
# Version history:
#   v1 — initial feature set, 152 columns
#   v2 — Apr 8: label fix in synthetic generator (is_pre_failure now
#        derived from degradation_phase instead of scheduled phase
#        durations). Same feature columns, but labels differ, so cached
#        features from v1 will train a different model.
#   v3 — Apr 9: te_health promoted to a first-class feature-engineering
#        citizen. The True Efficiency KPI formula was rewritten in
#        commit 0605590 to incorporate all 4 assignment §3.1.b variables
#        (cooling, voltage, environmental, operating_mode); this cache
#        version runs the rolling-stats / trend / correlation / diurnal
#        suite over te_health alongside jth, so XGBoost actually gets
#        the new TE signal as inputs instead of a single raw per-row
#        value. Feature count rises from 152 to ~181. Also fixes a
#        silent bug in the v2 cache where Pro and XP miners had
#        hashrate_nameplate_th = 0 (see commit 0605590 for details),
#        so te_health is now nonzero for all four hardware models.
FEATURES_VERSION = 3


# ─── Cross-signal ratios ─────────────────────────────────────────────

def compute_cross_signal_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Derived ratios that capture cross-signal relationships."""
    df = df.copy()

    df["jth"] = np.where(df["hashrate_th"] > 0, df["power_w"] / df["hashrate_th"], np.nan)
    df["temp_per_watt"] = np.where(df["power_w"] > 0, df["temperature_c"] / df["power_w"] * 1000, np.nan)
    df["hashrate_per_mhz"] = np.where(
        df["clock_frequency_mhz"] > 0,
        df["hashrate_th"] / df["clock_frequency_mhz"],
        np.nan,
    )
    df["power_per_mhz"] = np.where(
        df["clock_frequency_mhz"] > 0,
        df["power_w"] / df["clock_frequency_mhz"],
        np.nan,
    )
    # Current estimate
    df["current_a"] = np.where(df["voltage_v"] > 0.05, df["power_w"] / df["voltage_v"], np.nan)

    return df


# ─── Multi-timescale rolling statistics ──────────────────────────────

def compute_rolling_stats(
    df: pd.DataFrame,
    columns: List[str],
    windows_minutes: List[int],
) -> pd.DataFrame:
    """
    Rolling mean/std/min/max per column per window.
    For long windows (>=1440 min), only compute mean and std to keep feature count manageable.
    """
    df = df.sort_values(["miner_id", "timestamp"]).reset_index(drop=True)
    grouped = df.groupby("miner_id", sort=False)

    new_cols = {}
    for col in columns:
        if col not in df.columns:
            continue
        for win in windows_minutes:
            prefix = f"{col}_roll_{win}m"
            min_periods = max(1, win // 4)
            rolling = grouped[col].rolling(win, min_periods=min_periods)

            # mean and std for all windows
            new_cols[f"{prefix}_mean"] = rolling.mean().reset_index(level=0, drop=True)
            new_cols[f"{prefix}_std"] = rolling.std().reset_index(level=0, drop=True).fillna(0)

            # min and max only for short windows (< 1440 min)
            if win < 1440:
                new_cols[f"{prefix}_min"] = rolling.min().reset_index(level=0, drop=True)
                new_cols[f"{prefix}_max"] = rolling.max().reset_index(level=0, drop=True)

    for name, col in new_cols.items():
        df[name] = col.values

    return df


# ─── Rate of change ──────────────────────────────────────────────────

def compute_rate_of_change(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """First-order finite difference per minute. Grouped by miner_id."""
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        df[f"{col}_rate"] = df.groupby("miner_id")[col].diff().fillna(0)
    return df


# ─── 7-day trend slope (vectorized via cumsum trick) ─────────────────

def compute_trend_features(
    df: pd.DataFrame,
    columns: List[str],
    window_hours: int = 168,
) -> pd.DataFrame:
    """
    Rolling OLS slope of each column over the trend window.
    Uses cumsum trick: O(n) per column vs O(n*w) for naive per-window OLS.

    Adds {col}_trend_7d for each col.
    """
    df = df.sort_values(["miner_id", "timestamp"]).reset_index(drop=True)
    window = window_hours * 60  # minutes

    for col in columns:
        if col not in df.columns:
            continue

        def _rolling_slope(s: pd.Series) -> pd.Series:
            """Rolling OLS slope via sum-of-products formulation."""
            y = s.values.astype(np.float64)
            n = len(y)
            if n < 10:
                return pd.Series(np.zeros(n), index=s.index)

            # For a window of size w:
            # slope = (w * Σ(xy) - Σx * Σy) / (w * Σ(x²) - (Σx)²)
            # Where x = 0, 1, ..., w-1
            # Σx = w*(w-1)/2, Σ(x²) = w*(w-1)*(2w-1)/6

            # Build rolling sums of y and xy where x = index within window
            result = np.full(n, np.nan)
            # Use a simple approach: rolling sum of y*i where i is minute-index
            # For efficiency use a sliding window with pandas
            rolling = s.rolling(window, min_periods=max(60, window // 4))
            sum_y = rolling.sum().values
            sum_y_times_idx = (s * np.arange(n)).rolling(window, min_periods=max(60, window // 4)).sum().values
            sum_idx = pd.Series(np.arange(n, dtype=np.float64)).rolling(window, min_periods=max(60, window // 4)).sum().values
            sum_idx_sq = pd.Series(np.arange(n, dtype=np.float64) ** 2).rolling(window, min_periods=max(60, window // 4)).sum().values
            counts = rolling.count().values

            valid = counts > 10
            denom = counts[valid] * sum_idx_sq[valid] - sum_idx[valid] ** 2
            denom_safe = np.where(denom != 0, denom, 1)
            slopes = (counts[valid] * sum_y_times_idx[valid] - sum_idx[valid] * sum_y[valid]) / denom_safe
            result[valid] = slopes
            return pd.Series(result, index=s.index)

        slopes = df.groupby("miner_id")[col].apply(_rolling_slope).reset_index(level=0, drop=True)
        df[f"{col}_trend_7d"] = slopes.fillna(0).values

    return df


# ─── Variance trend (is std growing?) ────────────────────────────────

def compute_variance_trend(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Compare recent 24h std to prior 7d std.
    A growing ratio = degradation signal.
    Adds {col}_std_trend for each col.
    """
    df = df.sort_values(["miner_id", "timestamp"]).reset_index(drop=True)

    for col in columns:
        if col not in df.columns:
            continue
        rolling_24h = df.groupby("miner_id")[col].rolling(1440, min_periods=60).std().reset_index(level=0, drop=True)
        rolling_7d = df.groupby("miner_id")[col].rolling(10080, min_periods=1440).std().reset_index(level=0, drop=True)
        # Ratio: recent / long-term. np.where evaluates both branches
        # numerically so the division is computed even when the guard
        # rejects it; wrap in errstate to silence the cosmetic
        # RuntimeWarning from divide-by-zero inside the dead branch.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                rolling_7d.values > 1e-6,
                rolling_24h.values / rolling_7d.values,
                1.0
            )
        df[f"{col}_std_trend"] = ratio

    return df


# ─── Cross-signal rolling correlation ────────────────────────────────

def compute_cross_signal_correlations(
    df: pd.DataFrame,
    pairs: List[Tuple[str, str]],
    window_minutes: int = 360,
) -> pd.DataFrame:
    """
    Rolling Pearson correlation between signal pairs.
    Uses closed form: r = (E[XY] - E[X]E[Y]) / (σx × σy)
    """
    df = df.sort_values(["miner_id", "timestamp"]).reset_index(drop=True)

    # Single-miner fast path: skip the groupby entirely. This matters
    # for streaming live inference (one miner per call) where the
    # groupby-apply return-shape edge case would otherwise blow up.
    n_miners = df["miner_id"].nunique()

    for a, b in pairs:
        if a not in df.columns or b not in df.columns:
            continue

        col_name = f"corr_{a}_{b}_6h"
        df[col_name] = 0.0

        if n_miners == 1:
            ga = df[a].astype(np.float64)
            gb = df[b].astype(np.float64)
            r = ga.rolling(window_minutes, min_periods=60).corr(gb)
            df[col_name] = r.fillna(0).values
            continue

        def _rolling_corr(group):
            ga = group[a].astype(np.float64)
            gb = group[b].astype(np.float64)
            r = ga.rolling(window_minutes, min_periods=60).corr(gb)
            return r

        # pandas 2.x deprecation warning about implicit group-column
        # handling in apply(). The fix (include_groups=False or
        # column subset) reshapes the returned Series unpredictably
        # for this use case, so we silence the specific warning
        # instead.
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message=".*DataFrameGroupBy.apply operated on the grouping columns.*",
            )
            corrs = df.groupby("miner_id").apply(_rolling_corr).reset_index(level=0, drop=True)
        df[col_name] = corrs.fillna(0).values

    return df


# ─── Lag-1 autocorrelation ───────────────────────────────────────────

def compute_autocorrelation_lag1(
    df: pd.DataFrame,
    columns: List[str],
    window_minutes: int = 60,
) -> pd.DataFrame:
    """
    Rolling lag-1 autocorrelation.
    High autocorr = smooth signal. Drops as noise/chaos increases.
    """
    df = df.sort_values(["miner_id", "timestamp"]).reset_index(drop=True)

    for col in columns:
        if col not in df.columns:
            continue
        # Rolling correlation of the signal with its own lag-1
        lagged = df.groupby("miner_id")[col].shift(1)

        def _rolling_acf(g):
            vals = g.dropna().astype(np.float64)
            lag = vals.shift(1)
            # Rolling corr via closed form
            r = vals.rolling(window_minutes, min_periods=20).corr(lag)
            return r

        acf = df.groupby("miner_id")[col].apply(_rolling_acf).reset_index(level=0, drop=True)
        df[f"{col}_acf1"] = acf.fillna(0).values

    return df


# ─── Peak counts ─────────────────────────────────────────────────────

def compute_peak_count(
    df: pd.DataFrame,
    column: str,
    window_minutes: int = 60,
    sigma_threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Count values exceeding (rolling_mean + sigma × rolling_std) within a window.
    Intermittent fault detection.
    """
    df = df.sort_values(["miner_id", "timestamp"]).reset_index(drop=True)
    if column not in df.columns:
        return df

    rolling_mean = df.groupby("miner_id")[column].rolling(window_minutes, min_periods=30).mean().reset_index(level=0, drop=True)
    rolling_std = df.groupby("miner_id")[column].rolling(window_minutes, min_periods=30).std().reset_index(level=0, drop=True)
    threshold = rolling_mean.values + sigma_threshold * rolling_std.values
    exceeds = (df[column].values > threshold).astype(np.int32)

    # Rolling sum of exceedances
    exceeds_series = pd.Series(exceeds, index=df.index)
    df[f"{column}_peaks_per_hour"] = (
        df.assign(_ex=exceeds_series).groupby("miner_id")["_ex"]
        .rolling(window_minutes, min_periods=1).sum()
        .reset_index(level=0, drop=True).values
    )

    return df


# ─── Diurnal amplitude ───────────────────────────────────────────────

def compute_diurnal_amplitude(
    df: pd.DataFrame,
    column: str,
    window_hours: int = 48,
) -> pd.DataFrame:
    """
    Amplitude of the 24h cycle via (rolling_max - rolling_min) over window.
    A growing amplitude indicates degradation (harder to recover at night).
    """
    df = df.sort_values(["miner_id", "timestamp"]).reset_index(drop=True)
    if column not in df.columns:
        return df

    window_min = window_hours * 60
    roll_max = df.groupby("miner_id")[column].rolling(window_min, min_periods=720).max().reset_index(level=0, drop=True)
    roll_min = df.groupby("miner_id")[column].rolling(window_min, min_periods=720).min().reset_index(level=0, drop=True)
    df[f"{column}_diurnal_amp"] = (roll_max.values - roll_min.values)
    df[f"{column}_diurnal_amp"] = np.nan_to_num(df[f"{column}_diurnal_amp"], nan=0.0)

    return df


# ─── Degradation slope (legacy) ──────────────────────────────────────

def compute_degradation_slope(
    df: pd.DataFrame,
    window_hours: int = 6,
) -> pd.DataFrame:
    """Rolling OLS slope of J/TH over short window (6h). Quick degradation signal."""
    df = df.sort_values(["miner_id", "timestamp"]).reset_index(drop=True)
    if "jth" not in df.columns:
        df["jth"] = np.where(df["hashrate_th"] > 0, df["power_w"] / df["hashrate_th"], np.nan)

    window = window_hours * 60

    def _slope(s: pd.Series) -> pd.Series:
        y = s.values.astype(np.float64)
        n = len(y)
        result = np.full(n, np.nan)

        rolling = s.rolling(window, min_periods=max(60, window // 4))
        sum_y = rolling.sum().values
        sum_y_times_idx = (s * np.arange(n)).rolling(window, min_periods=max(60, window // 4)).sum().values
        sum_idx = pd.Series(np.arange(n, dtype=np.float64)).rolling(window, min_periods=max(60, window // 4)).sum().values
        sum_idx_sq = pd.Series(np.arange(n, dtype=np.float64) ** 2).rolling(window, min_periods=max(60, window // 4)).sum().values
        counts = rolling.count().values

        valid = counts > 10
        denom = counts[valid] * sum_idx_sq[valid] - sum_idx[valid] ** 2
        denom_safe = np.where(denom != 0, denom, 1)
        result[valid] = (counts[valid] * sum_y_times_idx[valid] - sum_idx[valid] * sum_y[valid]) / denom_safe
        return pd.Series(result, index=s.index)

    slopes = df.groupby("miner_id")["jth"].apply(_slope).reset_index(level=0, drop=True)
    df["jth_degradation_slope"] = slopes.fillna(0).values
    return df


# ─── Cross-miner features ────────────────────────────────────────────

def compute_cross_miner_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Container-level peer comparisons.
    Adds:
      neighbor_temp_delta — miner's temp minus container mean
      container_temp_rank — 0-1 rank within container
      efficiency_deviation — jth minus container median jth
      container_avg_temp, container_max_temp — informational
    """
    df = df.sort_values(["miner_id", "timestamp"]).reset_index(drop=True)

    if "container_id" not in df.columns:
        return df

    # Container stats per timestamp
    group = df.groupby(["container_id", "timestamp"])

    container_mean_temp = group["temperature_c"].transform("mean")
    container_max_temp = group["temperature_c"].transform("max")
    container_min_temp = group["temperature_c"].transform("min")
    container_med_jth = group["jth"].transform("median") if "jth" in df.columns else pd.Series(np.zeros(len(df)))

    df["container_avg_temp"] = container_mean_temp.values
    df["container_max_temp"] = container_max_temp.values
    df["neighbor_temp_delta"] = (df["temperature_c"] - container_mean_temp).values

    # Temp rank within container
    df["container_temp_rank"] = group["temperature_c"].rank(pct=True).values

    if "jth" in df.columns:
        df["efficiency_deviation"] = (df["jth"] - container_med_jth).values
    else:
        df["efficiency_deviation"] = 0.0

    df = df.fillna({
        "neighbor_temp_delta": 0.0,
        "container_temp_rank": 0.5,
        "efficiency_deviation": 0.0,
    })

    return df


# ─── Full pipeline ───────────────────────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame,
    config: FeatureConfig = FeatureConfig(),
    include_labels: bool = True,
    drop_warmup: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    drop_warmup: if True (default, batch behavior), drops the first
        ~7 days per miner because the longest rolling windows can't
        populate until then. Streaming callers pass False and handle
        warmup by returning None while the buffer is short.
    verbose: if True (default), prints per-stage column counts.
        Streaming callers pass False to avoid log spam on every tick.
    """
    if verbose:
        print("Building feature matrix...")

    sensor_cols = ["temperature_c", "hashrate_th", "power_w", "voltage_v"]
    # te_health joins the trend list alongside jth as of FEATURES_VERSION=3.
    # te_base and te_adjusted deliberately do NOT get the full treatment
    # because their per-row separation is near zero on this synthetic data
    # (see docs/TECHNICAL_REPORT.md §4 for the audit); te_health is where
    # the signal lives because of the hashrate_realization multiplier.
    trend_cols = ["temperature_c", "hashrate_th", "power_w",
                  "jth", "voltage_v", "te_health"]
    variance_cols = ["hashrate_th", "voltage_v", "power_w"]

    # 1. Cross-signal ratios
    df = compute_cross_signal_ratios(df)
    if verbose: print(f"  After ratios:          {len(df.columns)} cols")

    # 2. Rolling statistics (multi-timescale)
    # te_health added alongside jth as of FEATURES_VERSION=3 — the KPI
    # gets the same 6-window mean/std/min/max treatment as the raw
    # efficiency signal, so XGBoost can actually learn trends in the
    # composite metric instead of only seeing the raw per-row value.
    df = compute_rolling_stats(df, sensor_cols + ["jth", "te_health"],
                               config.rolling_windows_minutes)
    if verbose: print(f"  After rolling stats:   {len(df.columns)} cols")

    # 3. Rate of change
    df = compute_rate_of_change(df, config.rate_of_change_columns)
    if verbose: print(f"  After rate of change:  {len(df.columns)} cols")

    # 4. Short degradation slope (6h)
    df = compute_degradation_slope(df, config.degradation_slope_window_hours)
    if verbose: print(f"  After degr slope:      {len(df.columns)} cols")

    # 5. Long trend features (7d)
    df = compute_trend_features(df, trend_cols, config.trend_window_hours)
    if verbose: print(f"  After trend features:  {len(df.columns)} cols")

    # 6. Variance trends
    df = compute_variance_trend(df, variance_cols)
    if verbose: print(f"  After variance trends: {len(df.columns)} cols")

    # 7. Cross-signal correlations
    # te_health × temperature_c pair added in v3: a degrading chip will
    # typically show te_health dropping while its temperature rises (or
    # drops if throttling is engaging), so their rolling correlation
    # carries an early-warning signal the raw stats miss.
    pairs = [
        ("temperature_c", "power_w"),
        ("hashrate_th", "voltage_v"),
        ("jth", "temperature_c"),
        ("power_w", "clock_frequency_mhz"),
        ("te_health", "temperature_c"),
    ]
    df = compute_cross_signal_correlations(df, pairs, config.correlation_window_minutes)
    if verbose: print(f"  After correlations:    {len(df.columns)} cols")

    # 8. Autocorrelation lag-1
    df = compute_autocorrelation_lag1(df, ["hashrate_th", "temperature_c", "power_w"], config.autocorr_window_minutes)
    if verbose: print(f"  After autocorr:        {len(df.columns)} cols")

    # 9. Peak counts (intermittent faults)
    df = compute_peak_count(df, "power_w", config.peak_window_minutes, config.peak_sigma_threshold)
    df = compute_peak_count(df, "voltage_v", config.peak_window_minutes, config.peak_sigma_threshold)
    if verbose: print(f"  After peak counts:     {len(df.columns)} cols")

    # 10. Diurnal amplitude
    # te_health diurnal added in v3 — a healthy miner's TE follows a
    # daily thermal cycle as ambient swings, and the amplitude of that
    # cycle is itself a health indicator (too large → chip is
    # over-sensitive to environment, too small → something is suppressing
    # the normal daily variation, e.g. throttled + stuck).
    df = compute_diurnal_amplitude(df, "jth", config.diurnal_amplitude_window_hours)
    df = compute_diurnal_amplitude(df, "temperature_c", config.diurnal_amplitude_window_hours)
    df = compute_diurnal_amplitude(df, "te_health", config.diurnal_amplitude_window_hours)
    if verbose: print(f"  After diurnal:         {len(df.columns)} cols")

    # 11. Cross-miner features
    if config.cross_miner_features_enabled:
        df = compute_cross_miner_features(df)
        if verbose: print(f"  After cross-miner:     {len(df.columns)} cols")

    # Drop warm-up rows (longest window needs data to fill).
    # Keep rows after day 7 so the 7-day rolling windows have data.
    # Uses cumcount() instead of groupby-apply to avoid the pandas
    # FutureWarning about implicit group-column handling.
    warmup_steps = min(10080, config.rolling_windows_minutes[-1])  # 7 days
    if drop_warmup:
        row_num = df.groupby("miner_id").cumcount()
        df = df[row_num >= warmup_steps].reset_index(drop=True)

    # Fill remaining NaN with 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    df = df.replace([np.inf, -np.inf], 0)

    feature_cols = [c for c in df.columns if c not in FEATURE_EXCLUDE_COLUMNS]
    if verbose: print(f"  FINAL: {len(df):,} rows, {len(feature_cols)} features")
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return list of engineered feature column names (excludes metadata/labels/physics traces)."""
    return [c for c in df.columns if c not in FEATURE_EXCLUDE_COLUMNS]


def split_temporal(
    df: pd.DataFrame,
    train_fraction: float = 0.7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Two-way temporal train/test split (no shuffle). Retained for callers
    that don't need a validation set. New code should prefer
    split_temporal_tvt so threshold/hyperparameter tuning happens on data
    the final metrics never see.
    """
    timestamps = sorted(df["timestamp"].unique())
    split_idx = int(len(timestamps) * train_fraction)
    split_time = timestamps[split_idx]

    train = df[df["timestamp"] < split_time].copy()
    test = df[df["timestamp"] >= split_time].copy()

    print(f"Temporal split at {split_time}:")
    print(f"  Train: {len(train):,} rows ({train['is_pre_failure'].mean()*100:.2f}% positive)")
    print(f"  Test:  {len(test):,} rows ({test['is_pre_failure'].mean()*100:.2f}% positive)")

    return train, test


def split_temporal_tvt(
    df: pd.DataFrame,
    train_pos_fraction: float = 0.55,
    val_pos_fraction: float = 0.15,
    label_col: str = "is_pre_failure",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Three-way temporal split: train / validation / test (no shuffle).

    The validation set exists so decision thresholds and anomaly cutoffs
    can be tuned on data the test set has never seen. Without it, every
    reported precision/recall number is slightly optimistic because the
    threshold was fit on the same rows the metric is computed over.

    Splits are ordered in time (train first, then val, then test) so
    evaluation mirrors the production case of training on historical
    data and predicting the future.

    IMPORTANT — adaptive boundary selection:
    Because the synthetic dataset has very sparse, clustered positive
    labels (each failing miner contributes a single ~24-hour pre-failure
    window), a naive fixed-fraction split (e.g. 60% train / 15% val /
    25% test by time) will frequently produce a val set with zero
    positives, making F1-based threshold tuning degenerate.

    Instead we place boundaries by CUMULATIVE POSITIVE COUNT:
    - train contains the first `train_pos_fraction` of positives
    - val contains the next `val_pos_fraction`
    - test contains the rest

    This guarantees every split has positives (assuming enough exist)
    without violating temporal ordering. Row counts per split will vary
    with failure distribution — the train slice is typically still the
    largest because healthy rows dominate even in the early period.
    """
    if train_pos_fraction + val_pos_fraction >= 1.0:
        raise ValueError(
            f"train_pos_fraction + val_pos_fraction must be < 1 "
            f"(got {train_pos_fraction}+{val_pos_fraction})"
        )

    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    labels = df_sorted[label_col].astype(bool).to_numpy()
    total_pos = int(labels.sum())
    if total_pos < 3:
        raise ValueError(
            f"Too few positive rows ({total_pos}) to form a three-way split. "
            f"Increase failure_fraction or n_miners in SimulationConfig."
        )

    # Walk the rows in temporal order and place boundaries as soon as the
    # cumulative positive count crosses the requested fraction.
    cum_pos = np.cumsum(labels)
    train_target = int(round(total_pos * train_pos_fraction))
    val_target = int(round(total_pos * (train_pos_fraction + val_pos_fraction)))
    # searchsorted gives the first index i such that cum_pos[i] >= target.
    # Advance one past that index so the row with the target positive
    # belongs to the EARLIER split, not the later one.
    train_end_row = int(np.searchsorted(cum_pos, train_target, side="right"))
    val_end_row = int(np.searchsorted(cum_pos, val_target, side="right"))

    # Snap boundaries to the next timestamp change so no single minute
    # ever straddles two splits.
    timestamps = df_sorted["timestamp"].to_numpy()
    def _snap_to_next_timestamp(row_idx: int) -> int:
        if row_idx >= len(timestamps):
            return len(timestamps)
        boundary_ts = timestamps[row_idx]
        # Advance past all rows sharing this timestamp.
        while row_idx < len(timestamps) and timestamps[row_idx] == boundary_ts:
            row_idx += 1
        return row_idx

    train_end_row = _snap_to_next_timestamp(train_end_row)
    val_end_row = _snap_to_next_timestamp(val_end_row)
    if val_end_row <= train_end_row:
        val_end_row = _snap_to_next_timestamp(train_end_row + 1)

    train = df_sorted.iloc[:train_end_row].copy()
    val = df_sorted.iloc[train_end_row:val_end_row].copy()
    test = df_sorted.iloc[val_end_row:].copy()

    def _desc(name: str, part: pd.DataFrame) -> str:
        pos = int(part[label_col].sum())
        pct = 100.0 * pos / max(len(part), 1)
        first = part["timestamp"].iloc[0] if len(part) else None
        last = part["timestamp"].iloc[-1] if len(part) else None
        return (
            f"  {name:<6}{len(part):>10,} rows  "
            f"{pos:>6,} positive ({pct:.2f}%)  "
            f"{first} → {last}"
        )

    print("Temporal split (adaptive, boundary by cumulative positive count):")
    print(_desc("Train:", train))
    print(_desc("Val:",   val))
    print(_desc("Test:",  test))

    return train, val, test
