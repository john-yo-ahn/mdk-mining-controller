"""
Physics-based models for ASIC miner behavior.

Pure functions — given operating conditions and physics parameters,
return physical outputs.

Physics-first approach: failures drift the UPSTREAM parameters
(thermal_resistance, leakage_coefficient, chip_efficiency, psu_internal_resistance,
solder_resistance, fan_capacity, voltage_regulation_gain) and these functions
produce the downstream telemetry naturally.
"""

import math
import numpy as np
from typing import Tuple, List

from ..config import MinerSpec


# ─── Power model ──────────────────────────────────────────────────────

def compute_power(
    frequency_mhz: np.ndarray,
    voltage_v: np.ndarray,
    temperature_c: np.ndarray,
    spec: MinerSpec,
    leakage_coefficient: float = 1.0,
    chip_efficiency: float = 1.0,
) -> np.ndarray:
    """
    CMOS power model: P = (C * V^2 * f) / chip_efficiency + leakage(T) * coefficient

    - Lower chip_efficiency = same compute draws more watts (solder/trace losses)
    - Higher leakage_coefficient = more leakage at same temperature (electromigration)
    """
    cap_factor = spec.power_nameplate_w / (
        spec.voltage_default_v ** 2 * spec.freq_default_mhz
    )
    # Protect against division by zero
    chip_eff = max(0.1, chip_efficiency)
    dynamic = (cap_factor * voltage_v ** 2 * frequency_mhz) / chip_eff
    leakage = compute_leakage_power(temperature_c, spec, coefficient=leakage_coefficient)
    return dynamic + leakage


def compute_leakage_power(
    temperature_c: np.ndarray,
    spec: MinerSpec,
    coefficient: float = 1.0,
) -> np.ndarray:
    """Arrhenius-based leakage: exponential with temperature, scaled by coefficient."""
    base_leakage = spec.power_nameplate_w * 0.02
    exponent = np.clip(0.02 * (temperature_c - 40.0), -10, 10)
    return base_leakage * np.exp(exponent) * coefficient


def compute_hashrate(
    frequency_mhz: np.ndarray,
    voltage_v: np.ndarray,
    temperature_c: np.ndarray,
    spec: MinerSpec,
    chip_efficiency: float = 1.0,
) -> np.ndarray:
    """
    Hashrate proportional to frequency, scaled by chip_efficiency, with thermal throttling.

    chip_efficiency < 1.0 → chip can't produce full hashrate for given frequency
    """
    freq_ratio = frequency_mhz / spec.freq_default_mhz
    base_hashrate = spec.hashrate_nameplate_th * freq_ratio * chip_efficiency

    throttle = np.where(
        temperature_c > spec.temp_throttle_c,
        np.clip(
            1.0 - (temperature_c - spec.temp_throttle_c)
            / (spec.temp_max_c - spec.temp_throttle_c),
            0.0, 1.0
        ),
        1.0,
    )
    return base_hashrate * throttle


# ─── Thermal model ────────────────────────────────────────────────────

def thermal_step(
    temperature_c: float,
    power_w: float,
    ambient_c: float,
    thermal_resistance: float,
    thermal_capacitance: float,
    dt_seconds: float = 60.0,
) -> float:
    """First-order RC thermal model: dT/dt = (P*Rth - (T - Ta)) / (Rth*Cth)"""
    heat_in = power_w * thermal_resistance
    heat_out = temperature_c - ambient_c
    dtemp = (heat_in - heat_out) / thermal_capacitance * dt_seconds
    new_temp = temperature_c + dtemp
    return max(ambient_c, min(new_temp, 150.0))


# ─── Voltage model (PSU dynamics) ─────────────────────────────────────

def compute_voltage_actual(
    voltage_setpoint: float,
    load_fraction: float,
    psu_internal_resistance: float,
    voltage_regulation_gain: float,
    rng: np.random.Generator,
) -> float:
    """
    PSU output voltage model (dimensionless):
      V_actual = V_set × (1 − load_fraction × R_norm / reg_gain) + ripple(t)

    psu_internal_resistance is a dimensionless "normalized resistance" (0-0.05 range)
    load_fraction is power/nameplate (0-1+)

    As capacitors age: normalized R rises, reg_gain drops, ripple grows.
    Healthy baseline (R=0.0008, gain=1.0) → ~0.08% droop at full load.
    Failing PSU (R=0.03, gain=0.7) → ~4% droop + large ripple.
    """
    gain = max(0.3, voltage_regulation_gain)
    droop_pct = (load_fraction * psu_internal_resistance) / gain
    # Ripple grows as regulation gain drops (0.1% to ~2% of nominal)
    ripple_amplitude = voltage_setpoint * (0.001 + 0.02 * (1.0 / gain - 1.0))
    ripple = float(rng.normal(0, ripple_amplitude))
    return voltage_setpoint * (1.0 - droop_pct) + ripple


# ─── Container-level thermal coupling ─────────────────────────────────

def compute_container_supply_temp(
    miner_powers_w: np.ndarray,
    base_ambient_c: float,
    coupling: float,
) -> float:
    """
    Container effective supply temperature.
    Shared cooling means the container warms up as total heat output rises.
    """
    total_kw = float(np.sum(miner_powers_w)) / 1000.0
    return base_ambient_c + coupling * total_kw


def apply_neighbor_coupling(
    temps: np.ndarray,
    coupling: float,
) -> np.ndarray:
    """
    Each miner's effective ambient bump from its neighbors.
    Returns array of temperature offsets (not the new temps — just the delta to add).
    Uses mean of adjacent neighbors (position-based).
    """
    n = len(temps)
    if n <= 1:
        return np.zeros(n)

    offsets = np.zeros(n)
    for i in range(n):
        neighbors = []
        if i > 0:
            neighbors.append(temps[i - 1])
        if i < n - 1:
            neighbors.append(temps[i + 1])
        if neighbors:
            mean_neighbor = sum(neighbors) / len(neighbors)
            # Excess neighbor heat above fleet mean
            offsets[i] = coupling * max(0.0, mean_neighbor - temps.mean())
    return offsets


# ─── Ambient temperature profile ──────────────────────────────────────

def ambient_temperature_profile(
    n_steps: int,
    dt_seconds: int = 60,
    mean_c: float = 30.0,
    amplitude_c: float = 8.0,
    seed: int = 42,
    weekly_amplitude_c: float = 2.0,
    seasonal_drift_amplitude_c: float = 3.0,
    n_days: int = 30,
) -> np.ndarray:
    """
    Multi-timescale ambient temperature profile:
      - Diurnal cycle (24h)
      - Weekly cycle (cooler weekends due to lower grid load)
      - Seasonal drift (slow rise/fall over whole simulation)
      - Gaussian noise
    """
    rng = np.random.default_rng(seed)
    t_seconds = np.arange(n_steps) * dt_seconds
    hours = t_seconds / 3600.0
    days = hours / 24.0

    # Diurnal: peak at 2pm (hour 14)
    diurnal = amplitude_c * np.sin(2 * np.pi * (hours - 8) / 24)

    # Weekly: cooler on weekends (days 5, 6 of week)
    day_of_week = days % 7
    weekly = weekly_amplitude_c * np.cos(2 * np.pi * (day_of_week - 3) / 7)

    # Seasonal: slow sine over simulation length
    if n_days > 1:
        seasonal = seasonal_drift_amplitude_c * np.sin(2 * np.pi * days / max(n_days, 1))
    else:
        seasonal = np.zeros(n_steps)

    noise = rng.normal(0, 0.5, n_steps)
    return mean_c + diurnal + weekly + seasonal + noise


# ─── Workload event generation ────────────────────────────────────────

def make_workload_events(
    n_steps: int,
    dt_seconds: int,
    n_days: int,
    spec: MinerSpec,
    rng: np.random.Generator,
    change_prob_per_day: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate workload setpoint timelines with occasional frequency/voltage changes.
    Returns (frequency_setpoint_mhz, voltage_setpoint_v) arrays of length n_steps.
    """
    freq_base = spec.freq_default_mhz
    volt_base = spec.voltage_default_v

    freq = np.full(n_steps, freq_base, dtype=np.float64)
    volt = np.full(n_steps, volt_base, dtype=np.float64)

    # Number of workload changes across the whole simulation
    n_changes = max(0, int(rng.poisson(n_days * change_prob_per_day)))
    if n_changes == 0:
        return freq, volt

    # Random change points
    change_steps = sorted(rng.integers(0, n_steps, size=n_changes).tolist())

    current_freq = freq_base
    current_volt = volt_base
    last_step = 0

    for step in change_steps:
        freq[last_step:step] = current_freq
        volt[last_step:step] = current_volt

        # Pick new setpoint
        freq_delta_pct = float(rng.uniform(-0.08, 0.08))  # ±8%
        current_freq = float(np.clip(
            freq_base * (1 + freq_delta_pct),
            spec.freq_min_mhz, spec.freq_max_mhz
        ))
        # Voltage typically tracks frequency slightly
        current_volt = float(np.clip(
            volt_base * (1 + freq_delta_pct * 0.3),
            spec.voltage_min_v, spec.voltage_max_v
        ))
        last_step = step

    freq[last_step:] = current_freq
    volt[last_step:] = current_volt
    return freq, volt


def make_operator_events(
    n_steps: int,
    dt_seconds: int,
    n_days: int,
    rng: np.random.Generator,
    prob_per_day: float = 0.1,
) -> List[Tuple[int, int, str]]:
    """
    Generate operator events: maintenance windows, pool switches.
    Returns list of (start_step, end_step, kind) tuples.
    """
    events = []
    n_events = int(rng.poisson(n_days * prob_per_day))

    for _ in range(n_events):
        start = int(rng.integers(0, max(1, n_steps - 200)))
        kind = str(rng.choice(["maintenance_window", "pool_switch"]))
        if kind == "maintenance_window":
            duration = int(rng.integers(30, 120))
        else:
            duration = int(rng.integers(3, 10))
        events.append((start, min(start + duration, n_steps - 1), kind))

    return sorted(events)


# ─── Energy price profile ─────────────────────────────────────────────

def energy_price_profile(
    n_steps: int,
    dt_seconds: int = 60,
    seed: int = 42,
) -> np.ndarray:
    """Synthetic energy price profile. Returns $/kWh."""
    rng = np.random.default_rng(seed)
    hours = np.arange(n_steps) * dt_seconds / 3600.0
    base = 0.04 + 0.03 * np.maximum(0, np.sin(2 * np.pi * (hours - 8) / 24))
    noise = rng.normal(0, 0.003, n_steps)
    return np.clip(base + noise, 0.02, 0.15)
