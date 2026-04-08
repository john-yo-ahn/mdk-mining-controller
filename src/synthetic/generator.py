"""
Physics-first synthetic mining telemetry generator.

Key differences from the legacy generator:
1. Failures drift upstream physics parameters, not output signals
2. 3-phase degradation lifecycle (incubation → acceleration → cascade)
3. Per-miner personality (randomized baseline params)
4. Container-level thermal coupling
5. Workload variation (frequency changes)
6. 90-day default simulation
7. Inner loop uses scalar float math (no np.array([x]) allocations)
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..config import (
    SimulationConfig, MinerSpec, DEFAULT_MINER_SPECS, RAW_DIR,
    MinerPhysicsParams, make_physics_params,
)
from .physics import (
    compute_leakage_power, ambient_temperature_profile,
    compute_container_supply_temp, apply_neighbor_coupling,
    make_workload_events, make_operator_events, compute_voltage_actual,
)
from .scenarios import (
    SCENARIOS, list_scenarios, get_scenario, FailureScenario,
    apply_scenario_phase, compute_phase_at_step, resolve_phase_durations,
)
from .metadata import generate_miner_metadata


# ─── Runtime state per miner ─────────────────────────────────────────

@dataclass
class MinerRuntime:
    """Per-miner state + preallocated output buffers."""
    miner_id: str
    spec: MinerSpec
    container_id: str
    position: int
    params: MinerPhysicsParams

    # Workload timelines (baked in at init)
    freq_setpoint: np.ndarray
    voltage_setpoint: np.ndarray

    # Operator events
    operator_events: List[Tuple[int, int, str]]

    # Failure scenario (optional)
    scenario: Optional[FailureScenario]
    onset_step: int
    phase_durations: List[int]
    failure_step: int

    # Output buffers (preallocated float arrays, length n_steps)
    temperature_c: np.ndarray
    power_w: np.ndarray
    hashrate_th: np.ndarray
    voltage_v: np.ndarray
    frequency_mhz: np.ndarray
    container_supply_temp_c: np.ndarray
    ambient_c: np.ndarray

    # Labels
    degradation_phase: np.ndarray   # object dtype ("healthy"/"incubation"/etc.)
    days_to_failure: np.ndarray     # float32
    failure_type: np.ndarray        # object
    scenario_name: np.ndarray       # object
    operating_mode: np.ndarray      # object

    # Physics traces
    thermal_resistance_trace: np.ndarray
    leakage_coefficient_trace: np.ndarray
    chip_efficiency_trace: np.ndarray
    psu_internal_resistance_trace: np.ndarray
    solder_resistance_trace: np.ndarray
    fan_capacity_trace: np.ndarray
    voltage_regulation_gain_trace: np.ndarray

    is_shutdown: bool = False

    @classmethod
    def create(
        cls,
        miner_id: str,
        spec: MinerSpec,
        container_id: str,
        position: int,
        params: MinerPhysicsParams,
        n_steps: int,
        freq_setpoint: np.ndarray,
        voltage_setpoint: np.ndarray,
        operator_events: List[Tuple[int, int, str]],
        scenario: Optional[FailureScenario],
        onset_step: int,
        phase_durations: List[int],
    ):
        failure_step = onset_step + sum(phase_durations) if phase_durations else -1
        return cls(
            miner_id=miner_id,
            spec=spec,
            container_id=container_id,
            position=position,
            params=params,
            freq_setpoint=freq_setpoint,
            voltage_setpoint=voltage_setpoint,
            operator_events=operator_events,
            scenario=scenario,
            onset_step=onset_step,
            phase_durations=phase_durations,
            failure_step=failure_step,
            temperature_c=np.zeros(n_steps, dtype=np.float32),
            power_w=np.zeros(n_steps, dtype=np.float32),
            hashrate_th=np.zeros(n_steps, dtype=np.float32),
            voltage_v=np.zeros(n_steps, dtype=np.float32),
            frequency_mhz=np.zeros(n_steps, dtype=np.float32),
            container_supply_temp_c=np.zeros(n_steps, dtype=np.float32),
            ambient_c=np.zeros(n_steps, dtype=np.float32),
            degradation_phase=np.full(n_steps, "healthy", dtype=object),
            days_to_failure=np.full(n_steps, np.nan, dtype=np.float32),
            failure_type=np.full(n_steps, "none", dtype=object),
            scenario_name=np.full(n_steps, "", dtype=object),
            operating_mode=np.full(n_steps, "Normal", dtype=object),
            thermal_resistance_trace=np.zeros(n_steps, dtype=np.float32),
            leakage_coefficient_trace=np.zeros(n_steps, dtype=np.float32),
            chip_efficiency_trace=np.zeros(n_steps, dtype=np.float32),
            psu_internal_resistance_trace=np.zeros(n_steps, dtype=np.float32),
            solder_resistance_trace=np.zeros(n_steps, dtype=np.float32),
            fan_capacity_trace=np.zeros(n_steps, dtype=np.float32),
            voltage_regulation_gain_trace=np.zeros(n_steps, dtype=np.float32),
        )


# ─── Main generator ──────────────────────────────────────────────────

class MiningDataGenerator:
    """Physics-first mining fleet telemetry generator."""

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.rng = np.random.default_rng(self.config.random_seed)

    def generate(self) -> pd.DataFrame:
        """Run full simulation. Returns DataFrame."""
        cfg = self.config
        n_steps = cfg.n_steps

        # Global ambient profile (shared across all miners)
        ambient = ambient_temperature_profile(
            n_steps=n_steps,
            dt_seconds=cfg.sample_interval_seconds,
            mean_c=cfg.ambient_temp_mean_c,
            amplitude_c=cfg.ambient_temp_amplitude_c,
            seed=cfg.random_seed,
            weekly_amplitude_c=cfg.weekly_ambient_amplitude_c,
            seasonal_drift_amplitude_c=cfg.seasonal_drift_amplitude_c,
            n_days=cfg.n_days,
        )

        # Fleet metadata + runtime init
        metadata = generate_miner_metadata(cfg)
        runtimes = self._build_runtimes(metadata, n_steps)

        # Group miners by container
        containers: Dict[str, List[MinerRuntime]] = {}
        for r in runtimes:
            containers.setdefault(r.container_id, []).append(r)

        # Simulate each container
        print(f"Simulating {len(runtimes)} miners across {len(containers)} containers...")
        start = time.time()
        for container_id, miners in tqdm(containers.items(), desc="Containers"):
            self._simulate_container(miners, ambient)
        print(f"  Simulation complete in {time.time()-start:.1f}s")

        # Assemble final dataframe
        return self._assemble_dataframe(runtimes)

    # ── Setup ──────────────────────────────────────────────────────

    def _build_runtimes(
        self,
        metadata: pd.DataFrame,
        n_steps: int,
    ) -> List[MinerRuntime]:
        cfg = self.config
        rng = self.rng
        runtimes = []

        # Pick which miners get failures
        n_failures = max(2, int(cfg.n_miners * cfg.failure_fraction))
        failure_miner_indices = set(rng.choice(
            cfg.n_miners, size=n_failures, replace=False
        ).tolist())

        # Distribute failure types across scenarios with phases
        available_scenarios = [
            s for s in list_scenarios()
            if get_scenario(s).phases is not None
        ]

        for i, row in metadata.iterrows():
            spec = DEFAULT_MINER_SPECS[row["model"]]
            params = make_physics_params(spec, rng)

            # Workload timelines
            per_miner_rng = np.random.default_rng(cfg.random_seed + i)
            freq_set, volt_set = make_workload_events(
                n_steps=n_steps,
                dt_seconds=cfg.sample_interval_seconds,
                n_days=cfg.n_days,
                spec=spec,
                rng=per_miner_rng,
                change_prob_per_day=cfg.workload_change_prob_per_day,
            )

            operator_events = make_operator_events(
                n_steps=n_steps,
                dt_seconds=cfg.sample_interval_seconds,
                n_days=cfg.n_days,
                rng=per_miner_rng,
                prob_per_day=cfg.operator_event_prob_per_day,
            )

            # Assign scenario if this miner is unlucky
            scenario = None
            onset_step = -1
            phase_durations = []

            if i in failure_miner_indices and available_scenarios:
                scenario_name = available_scenarios[
                    len(runtimes) % len(available_scenarios)
                ]
                scenario = get_scenario(scenario_name)
                phase_durations = resolve_phase_durations(scenario, per_miner_rng)
                total_dur = sum(phase_durations)
                # Onset: random position leaving room for full failure + buffer
                max_onset = max(100, n_steps - total_dur - 100)
                onset_step = int(per_miner_rng.integers(
                    n_steps // 20,  # at least 5% into sim
                    max(n_steps // 20 + 1, max_onset)
                ))

            runtime = MinerRuntime.create(
                miner_id=row["miner_id"],
                spec=spec,
                container_id=row["container_id"],
                position=int(row["position"]),
                params=params,
                n_steps=n_steps,
                freq_setpoint=freq_set,
                voltage_setpoint=volt_set,
                operator_events=operator_events,
                scenario=scenario,
                onset_step=onset_step,
                phase_durations=phase_durations,
            )
            runtimes.append(runtime)

        return runtimes

    # ── Container-level simulation loop ────────────────────────────

    def _simulate_container(
        self,
        miners: List[MinerRuntime],
        ambient: np.ndarray,
    ):
        """Process one container step by step with thermal coupling."""
        cfg = self.config
        n_steps = len(ambient)
        n_miners = len(miners)

        # Initial temperatures
        for m in miners:
            m.temperature_c[0] = 35.0 + self.rng.uniform(0, 5)

        # Shared state arrays (reused across steps)
        prev_powers = np.zeros(n_miners)
        prev_temps = np.zeros(n_miners)

        for t in range(n_steps):
            # Compute container supply temp from previous step's powers
            for j, m in enumerate(miners):
                prev_powers[j] = m.power_w[t - 1] if t > 0 else 0.0
                prev_temps[j] = m.temperature_c[t - 1] if t > 0 else m.temperature_c[0]

            supply_temp = compute_container_supply_temp(
                prev_powers, float(ambient[t]), cfg.container_supply_temp_coupling
            )
            neighbor_offsets = apply_neighbor_coupling(
                prev_temps, cfg.neighbor_thermal_coupling
            )

            for j, m in enumerate(miners):
                self._step_miner(
                    m, t, float(ambient[t]),
                    float(neighbor_offsets[j]),
                    supply_temp,
                )

    # ── Inner loop (hot path) — all scalar float math ──────────────

    def _step_miner(
        self,
        m: MinerRuntime,
        t: int,
        ambient_t: float,
        neighbor_offset: float,
        supply_temp: float,
    ):
        """
        The hot loop. Must be fast.
        All locals are plain Python floats (no numpy allocations).
        """
        spec = m.spec
        params = m.params
        rng = self.rng

        # ── Effective ambient (includes container + neighbor coupling) ──
        effective_ambient = ambient_t + 0.5 * (supply_temp - ambient_t) + neighbor_offset

        # ── Apply scenario phase drift (if active) ──
        if m.scenario and t >= m.onset_step and not m.is_shutdown:
            phase_name, dtf = apply_scenario_phase(
                m.scenario, t, m.onset_step, m.phase_durations, params, rng,
            )
            m.degradation_phase[t] = phase_name
            m.days_to_failure[t] = dtf
            m.failure_type[t] = m.scenario.name
            m.scenario_name[t] = m.scenario.name
        elif not m.scenario:
            m.degradation_phase[t] = "healthy"

        # ── Check operator events ──
        in_maintenance = False
        for (start, end, kind) in m.operator_events:
            if start <= t <= end:
                if kind == "maintenance_window":
                    in_maintenance = True
                    m.operating_mode[t] = "Idle"
                    break

        # ── Shutdown check ──
        if m.is_shutdown:
            m.hashrate_th[t] = 0.0
            m.power_w[t] = 50.0
            prev_t = m.temperature_c[t - 1] if t > 0 else 35.0
            m.temperature_c[t] = max(effective_ambient, prev_t - 0.5)
            m.voltage_v[t] = 0.0
            m.frequency_mhz[t] = 0.0
            m.ambient_c[t] = ambient_t
            m.container_supply_temp_c[t] = supply_temp
            m.operating_mode[t] = "Shutdown"
            self._write_traces(m, t)
            return

        if in_maintenance:
            m.hashrate_th[t] = 0.0
            m.power_w[t] = 100.0
            prev_t = m.temperature_c[t - 1] if t > 0 else 35.0
            m.temperature_c[t] = max(effective_ambient, prev_t - 0.2)
            m.voltage_v[t] = 0.0
            m.frequency_mhz[t] = 0.0
            m.ambient_c[t] = ambient_t
            m.container_supply_temp_c[t] = supply_temp
            self._write_traces(m, t)
            return

        # ── Setpoints for this step ──
        freq = float(m.freq_setpoint[t])
        v_set = float(m.voltage_setpoint[t])

        # ── Power computation (inline CMOS) ──
        prev_temp = float(m.temperature_c[t - 1]) if t > 0 else 35.0
        prev_power = float(m.power_w[t - 1]) if t > 0 else 0.0

        # Dynamic power: (C * V^2 * f) / chip_efficiency
        cap_factor = spec.power_nameplate_w / (
            spec.voltage_default_v ** 2 * spec.freq_default_mhz
        )
        chip_eff = max(0.1, params.chip_efficiency)

        # Compute v_actual based on load fraction
        load_fraction = prev_power / max(spec.power_nameplate_w, 1.0)
        v_actual = compute_voltage_actual(
            v_set, load_fraction,
            params.psu_internal_resistance,
            params.voltage_regulation_gain,
            rng,
        )

        dynamic = (cap_factor * v_actual * v_actual * freq) / chip_eff
        # Leakage with coefficient
        leakage_base = spec.power_nameplate_w * 0.02
        leak_exp = 0.02 * (prev_temp - 40.0)
        leak_exp = max(-10.0, min(leak_exp, 10.0))
        leakage = leakage_base * math.exp(leak_exp) * params.leakage_coefficient
        # Solder + PSU IR² losses
        # load_fraction² × nameplate power × resistance_pct
        ir_loss = (load_fraction * load_fraction * spec.power_nameplate_w * 0.01 *
                   (params.psu_internal_resistance + params.solder_resistance) /
                   max(spec.psu_internal_resistance_base_ohm, 1e-9))

        power = dynamic + leakage + ir_loss
        power += rng.normal(0.0, abs(power) * 0.005)
        power = max(0.0, power)

        # ── Thermal model (inline RC step) ──
        # Thermal resistance effectively worsens as fan_capacity drops
        effective_rth = params.thermal_resistance / max(0.1, params.fan_capacity)
        heat_in = power * effective_rth
        heat_out = prev_temp - effective_ambient
        dtemp = (heat_in - heat_out) / spec.thermal_capacitance_jc * 60.0
        new_temp = prev_temp + dtemp + rng.normal(0.0, 0.2)
        if new_temp < effective_ambient:
            new_temp = effective_ambient
        if new_temp > 130.0:
            new_temp = 130.0

        # Shutdown check
        if new_temp > spec.temp_max_c:
            m.is_shutdown = True
            m.hashrate_th[t] = 0.0
            m.power_w[t] = 50.0
            m.temperature_c[t] = new_temp
            m.voltage_v[t] = v_actual
            m.frequency_mhz[t] = freq
            m.ambient_c[t] = ambient_t
            m.container_supply_temp_c[t] = supply_temp
            m.operating_mode[t] = "Shutdown"
            self._write_traces(m, t)
            return

        # ── Hashrate model ──
        freq_ratio = freq / spec.freq_default_mhz
        base_hashrate = spec.hashrate_nameplate_th * freq_ratio * chip_eff

        # Thermal throttling
        if new_temp > spec.temp_throttle_c:
            throttle = 1.0 - (new_temp - spec.temp_throttle_c) / (spec.temp_max_c - spec.temp_throttle_c)
            if throttle < 0.0:
                throttle = 0.0
            base_hashrate *= throttle

        hashrate = base_hashrate + rng.normal(0.0, abs(base_hashrate) * 0.008)
        if hashrate < 0.0:
            hashrate = 0.0

        # Store results
        m.temperature_c[t] = new_temp
        m.power_w[t] = power
        m.hashrate_th[t] = hashrate
        m.voltage_v[t] = v_actual
        m.frequency_mhz[t] = freq
        m.ambient_c[t] = ambient_t
        m.container_supply_temp_c[t] = supply_temp
        self._write_traces(m, t)

    def _write_traces(self, m: MinerRuntime, t: int):
        """Record current physics parameter values for diagnostics."""
        p = m.params
        m.thermal_resistance_trace[t] = p.thermal_resistance
        m.leakage_coefficient_trace[t] = p.leakage_coefficient
        m.chip_efficiency_trace[t] = p.chip_efficiency
        m.psu_internal_resistance_trace[t] = p.psu_internal_resistance
        m.solder_resistance_trace[t] = p.solder_resistance
        m.fan_capacity_trace[t] = p.fan_capacity
        m.voltage_regulation_gain_trace[t] = p.voltage_regulation_gain

    # ── Assemble DataFrame ─────────────────────────────────────────

    def _assemble_dataframe(self, runtimes: List[MinerRuntime]) -> pd.DataFrame:
        cfg = self.config
        n_steps = cfg.n_steps
        timestamps = pd.date_range(
            start="2026-01-01",
            periods=n_steps,
            freq=f"{cfg.sample_interval_seconds}s",
        )

        frames = []
        for m in runtimes:
            df = pd.DataFrame({
                "timestamp": timestamps,
                "miner_id": m.miner_id,
                "model": m.spec.model_name.split()[-1] if len(m.spec.model_name.split()) > 1 else m.spec.model_name,
                "container_id": m.container_id,
                "position": m.position,
                "clock_frequency_mhz": m.frequency_mhz,
                "voltage_v": m.voltage_v,
                "hashrate_th": m.hashrate_th,
                "temperature_c": m.temperature_c,
                "power_w": m.power_w,
                "ambient_temperature_c": m.ambient_c,
                "container_supply_temp_c": m.container_supply_temp_c,
                "freq_setpoint_mhz": m.freq_setpoint,
                "voltage_setpoint_v": m.voltage_setpoint,
                "operating_mode": m.operating_mode,
                "failure_type": m.failure_type,
                "scenario_name": m.scenario_name,
                "degradation_phase": m.degradation_phase,
                "days_to_failure": m.days_to_failure,
                # Physics traces (for diagnostics, excluded from features)
                "thermal_resistance": m.thermal_resistance_trace,
                "leakage_coefficient": m.leakage_coefficient_trace,
                "chip_efficiency": m.chip_efficiency_trace,
                "psu_internal_resistance": m.psu_internal_resistance_trace,
                "solder_resistance": m.solder_resistance_trace,
                "fan_capacity": m.fan_capacity_trace,
                "voltage_regulation_gain": m.voltage_regulation_gain_trace,
                "is_pre_failure": False,  # filled by label step
            })
            frames.append(df)

        result = pd.concat(frames, ignore_index=True)
        result = self._label_pre_failure(result, runtimes)
        return result

    def _label_pre_failure(
        self,
        df: pd.DataFrame,
        runtimes: List[MinerRuntime],
    ) -> pd.DataFrame:
        """
        Label is_pre_failure=True for rows where the miner is actively
        degrading — i.e. degradation_phase is "incubation" or "acceleration".

        WHY THIS DEFINITION (fix from Apr 8):
        The previous implementation computed the pre-failure window from
        the scheduled phase_durations at scenario creation time, then
        stamped it as "24h before scheduled cascade start". This drifted
        badly from reality because:
          1. The actual degradation_phase evolves based on physics, and
             may progress faster or slower than the schedule.
          2. Some scenarios never reach their cascade phase within the
             simulation window — the label would still fire on scheduled
             time, producing pre-failure rows on miners that never fail.
          3. Some scheduled cascade points landed AFTER the real failure
             onset by up to 18 days, so "pre-failure" rows were actually
             post-failure rows.

        The net effect was that >50% of the supervised training positives
        had degradation_phase="healthy" and failure_type="none" — XGBoost
        had no signal to learn from because those rows genuinely were
        healthy telemetry. AUC collapsed to ~0.49.

        By anchoring the label directly to degradation_phase we get a
        label that is ALWAYS consistent with the telemetry signal the
        model is asked to predict from. Every positive row is now a
        row where the physics state reflects actual degradation.
        """
        df = df.copy()

        if "degradation_phase" in df.columns:
            df["is_pre_failure"] = df["degradation_phase"].isin(
                ("incubation", "acceleration")
            )
        else:
            # Safety fallback for any future caller that produces frames
            # without the degradation_phase column. Keeps column shape
            # consistent but labels nothing.
            df["is_pre_failure"] = False

        return df

    # ── Persistence ────────────────────────────────────────────────

    def save(self, df: pd.DataFrame, filename: str = "mining_telemetry.parquet"):
        path = RAW_DIR / filename
        df.to_parquet(path, index=False, engine="pyarrow")
        print(f"Saved {len(df):,} rows to {path}")
        return path

    @staticmethod
    def load(filename: str = "mining_telemetry.parquet") -> pd.DataFrame:
        path = RAW_DIR / filename
        return pd.read_parquet(path, engine="pyarrow")
