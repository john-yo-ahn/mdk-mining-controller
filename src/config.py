"""
Central configuration for the MDK AI Mining Controller.
All constants, specs, paths, and tunable parameters live here.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


# ── Paths ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

for _d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ── Database paths (single source of truth) ───────────────────────────
# BATCH_DB_PATH:  written by `src.run_pipeline` from synthetic data, holds
#                 the full historical telemetry table used for training.
# LIVE_DB_PATH:   written by `src.cli.ai_bridge` during the live simulator,
#                 holds rolling per-tick telemetry from the running fleet.
# They are intentionally separate files so the simulator can run while a
# training pipeline is in flight without fighting over DuckDB's writer lock.

BATCH_DB_PATH = RAW_DIR / "mdk.duckdb"
LIVE_DB_PATH = RAW_DIR / "mdk_live.duckdb"


# ── Physics parameter names (canonical list) ─────────────────────────

PHYSICS_PARAM_NAMES = [
    "thermal_resistance",
    "leakage_coefficient",
    "chip_efficiency",
    "psu_internal_resistance",
    "solder_resistance",
    "fan_capacity",
    "voltage_regulation_gain",
]


# ── Hardware Specs ────────────────────────────────────────────────────

@dataclass(frozen=True)
class MinerSpec:
    """Hardware specification for one ASIC miner model (nameplate)."""
    model_name: str
    hashrate_nameplate_th: float
    power_nameplate_w: float
    freq_default_mhz: float
    freq_min_mhz: float
    freq_max_mhz: float
    voltage_default_v: float
    voltage_min_v: float = 0.28
    voltage_max_v: float = 0.50
    temp_max_c: float = 95.0
    temp_throttle_c: float = 85.0
    thermal_resistance_cw: float = 0.012
    thermal_capacitance_jc: float = 120.0

    # Extended physics parameters (nameplate baselines)
    leakage_base_w: float = 70.0            # base leakage power at 40C
    leakage_alpha: float = 0.02             # Arrhenius exponent slope
    chip_efficiency_nominal: float = 1.0    # solder/route quality (1.0 = ideal)
    psu_internal_resistance_base_ohm: float = 0.0008
    voltage_regulation_gain: float = 1.0    # closed-loop tightness
    fan_capacity_nominal: float = 1.0       # cooling effort multiplier
    solder_resistance_base_ohm: float = 0.0

    @property
    def efficiency_jth(self) -> float:
        return self.power_nameplate_w / self.hashrate_nameplate_th


# Thermal resistance tuned so equilibrium temperature at full load is ~70-75C
# Formula: T_eq = T_ambient + P * Rth  →  Rth ≈ 40 / P_nameplate for T_eq near 70C at 30C ambient
DEFAULT_MINER_SPECS: Dict[str, MinerSpec] = {
    "S21_Pro": MinerSpec(
        model_name="Antminer S21 Pro",
        hashrate_nameplate_th=234.0,
        power_nameplate_w=3510.0,
        freq_default_mhz=500,
        freq_min_mhz=200,
        freq_max_mhz=700,
        voltage_default_v=0.38,
        thermal_resistance_cw=0.0115,   # 3510W × 0.0115 = 40.4C rise → T_eq ≈ 70C
        thermal_capacitance_jc=200.0,
    ),
    "M56S": MinerSpec(
        model_name="Whatsminer M56S",
        hashrate_nameplate_th=212.0,
        power_nameplate_w=5400.0,
        freq_default_mhz=480,
        freq_min_mhz=200,
        freq_max_mhz=650,
        voltage_default_v=0.40,
        thermal_resistance_cw=0.0075,   # 5400W × 0.0075 = 40.5C rise
        thermal_capacitance_jc=250.0,
    ),
    "M63": MinerSpec(
        model_name="Whatsminer M63",
        hashrate_nameplate_th=390.0,
        power_nameplate_w=7215.0,
        freq_default_mhz=520,
        freq_min_mhz=200,
        freq_max_mhz=700,
        voltage_default_v=0.36,
        thermal_resistance_cw=0.0055,   # 7215W × 0.0055 = 39.7C rise
        thermal_capacitance_jc=300.0,
    ),
    "S19_XP": MinerSpec(
        model_name="Antminer S19 XP",
        hashrate_nameplate_th=141.0,
        power_nameplate_w=3010.0,
        freq_default_mhz=450,
        freq_min_mhz=200,
        freq_max_mhz=600,
        voltage_default_v=0.42,
        thermal_resistance_cw=0.0135,   # 3010W × 0.0135 = 40.6C rise
        thermal_capacitance_jc=180.0,
    ),
}


# ── Per-miner physics state (mutable, drifted by scenarios) ──────────

@dataclass
class MinerPhysicsParams:
    """
    Mutable per-miner physics state.
    Baselines are set at init (with randomized personality).
    Current values drift via scenario phases.
    """
    # Per-miner randomized baselines (constant over life)
    thermal_resistance_base: float
    leakage_coefficient_base: float
    chip_efficiency_base: float
    psu_internal_resistance_base: float
    solder_resistance_base: float
    fan_capacity_base: float
    voltage_regulation_gain_base: float
    efficiency_offset: float  # binning bonus/penalty
    install_days_ago: int     # for age-based aging

    # Current drifted values (scenarios mutate these)
    thermal_resistance: float
    leakage_coefficient: float
    chip_efficiency: float
    psu_internal_resistance: float
    solder_resistance: float
    fan_capacity: float
    voltage_regulation_gain: float

    def clamp(self):
        """Keep physics parameters in realistic ranges."""
        self.thermal_resistance = max(0.001, min(self.thermal_resistance, 0.1))
        self.leakage_coefficient = max(0.5, min(self.leakage_coefficient, 10.0))
        self.chip_efficiency = max(0.1, min(self.chip_efficiency, 1.2))
        self.psu_internal_resistance = max(0.0, min(self.psu_internal_resistance, 0.05))
        self.solder_resistance = max(0.0, min(self.solder_resistance, 0.01))
        self.fan_capacity = max(0.1, min(self.fan_capacity, 1.5))
        self.voltage_regulation_gain = max(0.3, min(self.voltage_regulation_gain, 1.2))


def make_physics_params(spec: MinerSpec, rng: np.random.Generator) -> MinerPhysicsParams:
    """Create randomized per-miner physics params from a spec."""
    thermal_resistance_base = float(np.clip(
        rng.normal(spec.thermal_resistance_cw, 0.05 * spec.thermal_resistance_cw),
        0.7 * spec.thermal_resistance_cw,
        1.4 * spec.thermal_resistance_cw,
    ))
    leakage_coefficient_base = float(np.clip(rng.normal(1.0, 0.03), 0.9, 1.1))
    chip_efficiency_base = float(np.clip(rng.normal(1.0, 0.02), 0.92, 1.05))
    psu_internal_resistance_base = float(np.clip(
        rng.normal(spec.psu_internal_resistance_base_ohm, 0.10 * spec.psu_internal_resistance_base_ohm),
        0.5 * spec.psu_internal_resistance_base_ohm,
        2.0 * spec.psu_internal_resistance_base_ohm,
    ))
    solder_resistance_base = float(max(0.0, rng.normal(0.0, 0.0005)))
    fan_capacity_base = float(np.clip(rng.normal(1.0, 0.04), 0.85, 1.10))
    voltage_regulation_gain_base = float(np.clip(rng.normal(1.0, 0.02), 0.93, 1.05))
    efficiency_offset = float(rng.normal(0.0, 0.02))
    install_days_ago = int(rng.integers(0, 3 * 365))

    return MinerPhysicsParams(
        thermal_resistance_base=thermal_resistance_base,
        leakage_coefficient_base=leakage_coefficient_base,
        chip_efficiency_base=chip_efficiency_base,
        psu_internal_resistance_base=psu_internal_resistance_base,
        solder_resistance_base=solder_resistance_base,
        fan_capacity_base=fan_capacity_base,
        voltage_regulation_gain_base=voltage_regulation_gain_base,
        efficiency_offset=efficiency_offset,
        install_days_ago=install_days_ago,
        # Current = baseline at init
        thermal_resistance=thermal_resistance_base,
        leakage_coefficient=leakage_coefficient_base,
        chip_efficiency=chip_efficiency_base,
        psu_internal_resistance=psu_internal_resistance_base,
        solder_resistance=solder_resistance_base,
        fan_capacity=fan_capacity_base,
        voltage_regulation_gain=voltage_regulation_gain_base,
    )


# ── Simulation Config ─────────────────────────────────────────────────

@dataclass
class SimulationConfig:
    # Fleet size and simulation length.
    #
    # The defaults (30 miners × 120 days × 1 min sampling) produce ~5.2M
    # raw telemetry rows and enough failure events to give statistically
    # stable validation/test metrics. The previous defaults (20 × 90)
    # produced only ~3 test failures per run, making recall numbers
    # bounce ±33% on every retrain.
    #
    # Bump n_miners further if you want more headroom for the val/test
    # splits (each failing miner contributes one ~24h pre-failure
    # window, and the adaptive splitter needs several windows per
    # partition to compute a meaningful F1 curve).
    n_miners: int = 30
    n_days: int = 120
    sample_interval_seconds: int = 60
    random_seed: int = 42

    # Ambient temperature
    ambient_temp_mean_c: float = 30.0
    ambient_temp_amplitude_c: float = 8.0
    weekly_ambient_amplitude_c: float = 2.0
    seasonal_drift_amplitude_c: float = 3.0

    # Noise
    noise_level: float = 0.02

    # Failure distribution.
    # 0.55 means ~16 of 30 miners develop a failure over the 120-day
    # simulation, giving ~16 pre-failure windows of ~24h each (≈23k
    # positive rows out of 5.2M). The adaptive splitter places 55% of
    # positives in train, 15% in val, 30% in test.
    failure_fraction: float = 0.55

    # Workload / operator events
    workload_change_prob_per_day: float = 0.5
    operator_event_prob_per_day: float = 0.1

    # Container layout
    container_size: int = 10  # miners per container

    # Thermal coupling
    neighbor_thermal_coupling: float = 0.05
    container_supply_temp_coupling: float = 0.02  # C rise per kW shared

    # Feature flag — use new physics drift mechanism
    enable_physics_drift: bool = True

    @property
    def n_steps(self) -> int:
        return self.n_days * 24 * 60 * 60 // self.sample_interval_seconds


# ── Feature Engineering Config ────────────────────────────────────────

@dataclass
class FeatureConfig:
    # Multi-timescale rolling windows (minutes)
    # 2m, 15m, 1h, 6h, 1d, 7d
    rolling_windows_minutes: List[int] = field(
        default_factory=lambda: [2, 15, 60, 360, 1440, 10080]
    )
    rate_of_change_columns: List[str] = field(default_factory=lambda: [
        "temperature_c", "hashrate_th", "power_w", "efficiency_jth",
    ])
    degradation_slope_window_hours: int = 6
    label_horizon_hours: int = 24

    # New: multi-timescale trend features
    trend_window_hours: int = 168  # 7 days
    correlation_window_minutes: int = 360  # 6 hours
    autocorr_window_minutes: int = 60
    peak_window_minutes: int = 60
    peak_sigma_threshold: float = 3.0
    diurnal_amplitude_window_hours: int = 48
    cross_miner_features_enabled: bool = True

    # Drop warm-up rows (longest window needs this much warm-up)
    keep_minimum_history_minutes: int = 120


# ── KPI Config ────────────────────────────────────────────────────────

@dataclass
class TEConfig:
    # Cooling + infrastructure overhead on the denominator.
    # "How much site power is consumed per watt of chip power" —
    # cooling = alpha_cooling (fans + CRACs), infra = beta_infra
    # (PDUs, network, lighting). Combined 1 + alpha + beta = 1.20 by
    # default, i.e. facility draws 1.20 W per 1 W of chip work.
    alpha_cooling: float = 0.15
    beta_infra: float = 0.05

    # Environmental penalty on te_adjusted:
    #     te_adjusted = te_base × (1 - delta_temp × max(0, ambient - temp_baseline_c))
    # A miner in a 35°C environment gets a (1 - 0.008 × 10) = 0.92 ×
    # haircut versus the same hardware in a 25°C environment.
    delta_temp: float = 0.008
    temp_baseline_c: float = 25.0

    # Chip voltage stability penalty (Assignment §3.1.b variable #2).
    # Miners operating at their spec's default voltage get 1.0; each
    # percent deviation from default costs voltage_penalty_coefficient
    # percentage points off the factor. 0.5 was chosen so that:
    #   * the measured ±2% healthy voltage noise maps to ~1% penalty
    #     (negligible on healthy rows; adds no separation noise)
    #   * a 20% deviation (~a failing chip at 0.30 V vs 0.38 V spec)
    #     maps to a 10% penalty (meaningful, not dominant)
    # The clip [0, 1] below in compute_te_base guards against extreme
    # voltages producing negative factors.
    voltage_penalty_coefficient: float = 0.5

    # Device operating mode weights (Assignment §3.1.b variable #4).
    # Idle and Shutdown miners contribute zero true efficiency because
    # they are not producing useful work, regardless of how little
    # power they are drawing. Normal is 1.0. Unknown/novel modes
    # default to 1.0 in compute_te_base.
    operating_mode_weights: Dict[str, float] = field(default_factory=lambda: {
        "Normal": 1.0,
        "Idle": 0.0,
        "Shutdown": 0.0,
    })


# ── Optimizer Config ──────────────────────────────────────────────────

@dataclass
class OptimizerConfig:
    thermal_warning_c: float = 80.0
    thermal_critical_c: float = 90.0
    thermal_shutdown_c: float = 95.0
    thermal_hysteresis_c: float = 5.0
    freq_step_mhz: float = 25.0
    min_change_interval_seconds: int = 300
    energy_price_cheap_usd: float = 0.035
    energy_price_expensive_usd: float = 0.07
    degradation_te_threshold: float = 0.85


# ── Telemetry Schema ──────────────────────────────────────────────────

TELEMETRY_COLUMNS = [
    "timestamp",
    "miner_id",
    "model",
    "container_id",
    "position",
    "clock_frequency_mhz",
    "voltage_v",
    "hashrate_th",
    "temperature_c",
    "power_w",
    "ambient_temperature_c",
    "operating_mode",
    "failure_type",
    "is_pre_failure",
    # New columns from physics-first generator
    "degradation_phase",
    "days_to_failure",
    "scenario_name",
    "container_supply_temp_c",
    "freq_setpoint_mhz",
    "voltage_setpoint_v",
]

# Physics trace columns (kept for diagnostics, EXCLUDED from ML features)
PHYSICS_TRACE_COLUMNS = [
    "thermal_resistance",
    "leakage_coefficient",
    "chip_efficiency",
    "psu_internal_resistance",
    "solder_resistance",
    "fan_capacity",
    "voltage_regulation_gain",
]

SENSOR_COLUMNS = [
    "clock_frequency_mhz",
    "voltage_v",
    "hashrate_th",
    "temperature_c",
    "power_w",
    "ambient_temperature_c",
]

# Columns excluded from feature matrix (metadata + labels + physics traces)
FEATURE_EXCLUDE_COLUMNS = {
    "timestamp", "miner_id", "model", "container_id", "position",
    "operating_mode", "failure_type", "is_pre_failure",
    "degradation_phase", "days_to_failure", "scenario_name",
    "hashrate_nameplate_th", "hashrate_realization",
    # Exclude voltage_default_v — it's added by compute_all_te_variants
    # in FEATURES_VERSION=3 but is a constant per hardware model, so it
    # would act as a model-ID label leakage feature if left in the
    # feature matrix. The useful voltage signal is already captured by
    # the voltage_v column + its rolling stats.
    "voltage_default_v",
    *PHYSICS_TRACE_COLUMNS,
}
