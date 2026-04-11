"""
Mining fleet simulation for the CLI dashboard.
Uses the scenario engine for data-driven failure injection.
Connects to trained AI models via the AI bridge.
"""

import collections
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict

import numpy as np

from .ai_bridge import AIBridge
from ..kpi.true_efficiency import (
    compute_te_base_scalar,
    compute_te_adjusted_scalar,
    compute_te_health_scalar,
)
from ..synthetic.scenarios import (
    SCENARIOS, list_scenarios, get_scenario,
    apply_scenario_effects, compute_progression,
)


class OperatingMode(Enum):
    HIGH = "High"
    NORMAL = "Normal"
    LOW = "Low"
    SLEEP = "Sleep"
    SHUTDOWN = "Shutdown"


class FailureType(Enum):
    NONE = "none"
    GRADUAL_DEGRADATION = "gradual_degradation"
    THERMAL_RUNAWAY = "thermal_runaway"
    FAN_STALL = "fan_stall"
    PSU_DEGRADATION = "psu_degradation"
    SUDDEN_CHIP_FAILURE = "sudden_chip_failure"
    COOLANT_RESTRICTION = "coolant_restriction"
    FIRMWARE_OSCILLATION = "firmware_oscillation"
    CONNECTOR_CORROSION = "connector_corrosion"


class AlertSeverity(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    INFO = "INFO"


@dataclass
class MinerSpec:
    model: str
    hashrate_nameplate_th: float
    power_nameplate_w: float
    freq_default_mhz: float
    freq_min_mhz: float
    freq_max_mhz: float
    voltage_default_v: float
    temp_max_c: float = 95.0
    temp_throttle_c: float = 85.0
    thermal_resistance: float = 0.012
    thermal_capacitance: float = 120.0
    efficiency_jth: float = 18.0


MINER_MODELS = {
    "Antminer S21 Pro": MinerSpec(
        model="Antminer S21 Pro",
        hashrate_nameplate_th=234.0, power_nameplate_w=3510.0,
        freq_default_mhz=500, freq_min_mhz=200, freq_max_mhz=700,
        voltage_default_v=0.38, efficiency_jth=15.0,
    ),
    "Whatsminer M56S": MinerSpec(
        model="Whatsminer M56S",
        hashrate_nameplate_th=212.0, power_nameplate_w=5400.0,
        freq_default_mhz=480, freq_min_mhz=200, freq_max_mhz=650,
        voltage_default_v=0.40, efficiency_jth=25.5,
    ),
    "Whatsminer M63": MinerSpec(
        model="Whatsminer M63",
        hashrate_nameplate_th=390.0, power_nameplate_w=7215.0,
        freq_default_mhz=520, freq_min_mhz=200, freq_max_mhz=700,
        voltage_default_v=0.36, efficiency_jth=18.5,
    ),
    "Antminer S19 XP": MinerSpec(
        model="Antminer S19 XP",
        hashrate_nameplate_th=141.0, power_nameplate_w=3010.0,
        freq_default_mhz=450, freq_min_mhz=200, freq_max_mhz=600,
        voltage_default_v=0.42, efficiency_jth=21.5,
    ),
}


@dataclass
class Alert:
    timestamp: float
    miner_id: str
    severity: AlertSeverity
    message: str
    failure_type: FailureType = FailureType.NONE


@dataclass
class OptimizerAction:
    timestamp: float
    miner_id: str
    action: str
    old_value: float
    new_value: float
    reason: str


@dataclass
class MinerState:
    miner_id: str
    spec: MinerSpec
    container_id: str
    position: int

    # Operating state
    mode: OperatingMode = OperatingMode.NORMAL
    frequency_mhz: float = 0.0
    voltage_v: float = 0.0
    temperature_c: float = 35.0
    hashrate_th: float = 0.0
    power_w: float = 0.0
    ambient_c: float = 30.0
    uptime_hours: float = 0.0

    # KPIs
    efficiency_jth: float = 0.0
    te_base: float = 0.0
    te_health: float = 0.0
    hashrate_realization: float = 1.0

    # Health
    health_score: float = 1.0
    failure_type: FailureType = FailureType.NONE
    failure_progress: float = 0.0
    is_flagged: bool = False
    predicted_failure: bool = False
    anomaly_score: float = 0.0

    # History deques for sparkline graphs in the TUI detail panel.
    # 360 points = 6 hours at 1-min sampling. Enough for the 60-point
    # sparkline window with room for future zoom/pan.
    te_health_history: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=360)
    )
    anomaly_score_history: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=360)
    )
    health_score_history: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=360)
    )

    # Scenario engine fields
    _step: int = 0
    _failure_onset_step: int = -1
    _failure_duration: int = 0
    _scenario_name: str = ""
    _degradation_factor: float = 1.0
    _base_thermal_resistance: float = 0.012

    def __post_init__(self):
        if self.frequency_mhz == 0:
            self.frequency_mhz = self.spec.freq_default_mhz
        if self.voltage_v == 0:
            self.voltage_v = self.spec.voltage_default_v
        self._base_thermal_resistance = self.spec.thermal_resistance


class MiningFleetSimulation:
    """Simulates a fleet of ASIC miners with scenario-driven failure injection."""

    def __init__(self, n_miners: int = 24, seed: int = 42, speed: float = 1.0):
        self.n_miners = n_miners
        self.seed = seed
        self.speed = speed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.miners: list[MinerState] = []
        self.alerts: list[Alert] = []
        self.actions: list[OptimizerAction] = []
        self.step = 0
        self.start_time = time.time()

        # Fleet aggregates
        self.total_hashrate_th = 0.0
        self.total_power_w = 0.0
        self.fleet_efficiency_jth = 0.0
        self.fleet_te_health = 0.0
        self.healthy_count = 0
        self.warning_count = 0
        self.critical_count = 0
        self.energy_price_usd = 0.055

        # AI Bridge
        self.ai = AIBridge()
        self._ai_ready = False

        self._init_fleet()
        self._init_ai()

    def _init_fleet(self):
        models = list(MINER_MODELS.keys())
        containers = ["Container-A", "Container-B", "Container-C"]

        for i in range(self.n_miners):
            model_name = models[i % len(models)]
            spec = MINER_MODELS[model_name]
            container = containers[i % len(containers)]

            miner = MinerState(
                miner_id=f"MNR-{i+1:03d}",
                spec=spec,
                container_id=container,
                position=i % 20 + 1,
                frequency_mhz=spec.freq_default_mhz + self.rng.uniform(-20, 20),
                voltage_v=spec.voltage_default_v + self.rng.uniform(-0.01, 0.01),
                temperature_c=30 + self.rng.uniform(0, 10),
                uptime_hours=self.rng.uniform(100, 5000),
            )
            self.miners.append(miner)

        # Assign failures to ~30% of fleet using ALL 8 scenario types
        n_failures = max(2, int(self.n_miners * 0.3))
        failure_miners = self.rng.sample(range(self.n_miners), n_failures)
        all_scenarios = list_scenarios()

        for i, idx in enumerate(failure_miners):
            scenario_name = all_scenarios[i % len(all_scenarios)]
            scenario = get_scenario(scenario_name)

            duration = self.rng.randint(scenario.duration_range[0], scenario.duration_range[1])
            onset = self.rng.randint(60, 600)

            miner = self.miners[idx]
            miner._scenario_name = scenario_name
            miner._failure_onset_step = onset
            miner._failure_duration = duration

            # Map scenario name to FailureType enum (or NONE if no match)
            try:
                miner.failure_type = FailureType(scenario_name)
            except ValueError:
                miner.failure_type = FailureType.NONE

    def _init_ai(self):
        """Load trained models and init storage."""
        print("Initializing AI bridge...")
        self._ai_ready = self.ai.load_models()
        self.ai.init_storage()

        for miner in self.miners:
            self.ai.register_miner(
                miner.miner_id, miner.spec.model, miner.spec.hashrate_nameplate_th,
            )

        if self._ai_ready:
            print("  AI bridge: REAL models active")
        else:
            print("  AI bridge: fallback heuristics (run 'uv run mdk train' first)")

    # ── Scenario injection (live, mid-run) ─────────────────────────

    def inject_scenario(self, miner_id: str, scenario_name: str) -> bool:
        """Inject a failure scenario into a running miner. Returns success."""
        miner = next((m for m in self.miners if m.miner_id == miner_id), None)
        if miner is None:
            return False
        if scenario_name not in SCENARIOS:
            return False

        scenario = get_scenario(scenario_name)
        miner._scenario_name = scenario_name
        miner._failure_onset_step = self.step  # starts NOW
        miner._failure_duration = self.rng.randint(
            scenario.duration_range[0], scenario.duration_range[1]
        )
        miner._degradation_factor = 1.0
        miner.spec.thermal_resistance = miner._base_thermal_resistance
        miner.is_flagged = False
        miner.failure_progress = 0.0

        try:
            miner.failure_type = FailureType(scenario_name)
        except ValueError:
            miner.failure_type = FailureType.NONE

        return True

    def get_active_failures(self) -> List[Dict]:
        """Return currently active failure scenarios with progress and AI detection status."""
        active = []
        for miner in self.miners:
            if not miner._scenario_name or miner._failure_onset_step < 0:
                continue
            if self.step < miner._failure_onset_step:
                # Not yet started
                active.append({
                    "miner_id": miner.miner_id,
                    "scenario": miner._scenario_name,
                    "progress": 0.0,
                    "ai_detected": False,
                    "onset_step": miner._failure_onset_step,
                    "status": "pending",
                })
                continue

            progress = compute_progression(
                self.step, miner._failure_onset_step, miner._failure_duration,
            )
            active.append({
                "miner_id": miner.miner_id,
                "scenario": miner._scenario_name,
                "progress": progress,
                "ai_detected": miner.predicted_failure or miner.is_flagged,
                "onset_step": miner._failure_onset_step,
                "anomaly_score": miner.anomaly_score,
                "status": "active" if progress < 1.0 else "complete",
            })
        return active

    # ── Main tick ──────────────────────────────────────────────────

    def tick(self) -> None:
        """Advance simulation by one step (~1 minute of simulated time)."""
        self.step += 1
        hour_of_day = (self.step / 60) % 24

        # Ambient temperature: diurnal cycle
        ambient = 28 + 8 * math.sin(2 * math.pi * (hour_of_day - 6) / 24)
        ambient += self.rng.gauss(0, 0.5)

        # Energy price
        base_price = 0.04 + 0.03 * max(0, math.sin(2 * math.pi * (hour_of_day - 8) / 24))
        self.energy_price_usd = max(0.02, base_price + self.rng.gauss(0, 0.003))

        new_alerts = []
        new_actions = []

        for miner in self.miners:
            miner._step = self.step
            miner.ambient_c = ambient + self.rng.gauss(0, 1)

            if miner.mode == OperatingMode.SHUTDOWN:
                miner.hashrate_th = 0
                miner.power_w = 50
                miner.temperature_c = max(miner.ambient_c, miner.temperature_c - 0.5)
                continue

            # Apply scenario-driven failure effects
            self._apply_failure(miner)

            # Physics: power = CV²f + leakage(T)
            cap_factor = miner.spec.power_nameplate_w / (
                miner.spec.voltage_default_v ** 2 * miner.spec.freq_default_mhz
            )
            dynamic_power = cap_factor * miner.voltage_v ** 2 * miner.frequency_mhz
            leakage = 50 * math.exp(0.02 * min(miner.temperature_c - 40, 60))
            miner.power_w = (dynamic_power + leakage) * miner._degradation_factor
            miner.power_w += self.rng.gauss(0, miner.power_w * 0.005)
            miner.power_w = max(0, miner.power_w)

            # Physics: thermal model
            heat_in = miner.power_w * miner.spec.thermal_resistance
            heat_out = miner.temperature_c - miner.ambient_c
            dtemp = (heat_in - heat_out) / miner.spec.thermal_capacitance * 60
            miner.temperature_c += dtemp
            miner.temperature_c += self.rng.gauss(0, 0.2)
            miner.temperature_c = max(miner.ambient_c, min(miner.temperature_c, 130))

            # Physics: hashrate with thermal throttling
            freq_ratio = miner.frequency_mhz / miner.spec.freq_default_mhz
            base_hashrate = miner.spec.hashrate_nameplate_th * freq_ratio

            if miner.temperature_c > miner.spec.temp_throttle_c:
                throttle = max(0, 1 - (
                    (miner.temperature_c - miner.spec.temp_throttle_c)
                    / (miner.spec.temp_max_c - miner.spec.temp_throttle_c)
                ))
                base_hashrate *= throttle

            miner.hashrate_th = base_hashrate * miner._degradation_factor
            miner.hashrate_th += self.rng.gauss(0, miner.hashrate_th * 0.008)
            miner.hashrate_th = max(0, miner.hashrate_th)

            # KPIs — delegate to the canonical TE formula so the
            # dashboard and the batch pipeline use identical math.
            # Pre-Level-1 this block reimplemented TE inline and drifted
            # (it was missing β_infra, giving TE values ~4.3% higher
            # than the batch pipeline for the same telemetry). Now both
            # code paths call the same helpers in src/kpi/true_efficiency.py.
            if miner.hashrate_th > 0:
                miner.efficiency_jth = miner.power_w / miner.hashrate_th
                miner.hashrate_realization = min(
                    1.0, miner.hashrate_th / miner.spec.hashrate_nameplate_th,
                )
                miner.te_base = compute_te_base_scalar(
                    hashrate_th=miner.hashrate_th,
                    power_chip_w=miner.power_w,
                    voltage_v=miner.voltage_v,
                    voltage_default_v=miner.spec.voltage_default_v,
                    operating_mode=miner.mode.value,
                )
                te_adjusted_val = compute_te_adjusted_scalar(
                    te_base=miner.te_base,
                    ambient_temp_c=miner.ambient_c,
                )
                miner.te_health = compute_te_health_scalar(
                    te_adjusted=te_adjusted_val,
                    hashrate_actual_th=miner.hashrate_th,
                    hashrate_nameplate_th=miner.spec.hashrate_nameplate_th,
                )
            else:
                miner.efficiency_jth = float("inf")
                miner.te_base = 0
                miner.te_health = 0
                miner.hashrate_realization = 0

            # Health scoring
            miner.health_score = self._compute_health(miner)
            miner.uptime_hours += 1 / 60

            # Push telemetry to AI bridge
            self.ai.push_telemetry(
                miner.miner_id,
                miner.frequency_mhz, miner.voltage_v, miner.hashrate_th,
                miner.temperature_c, miner.power_w, miner.ambient_c,
            )

            # AI predictions — staggered: ONE miner per tick.
            #
            # build_feature_matrix costs ~250ms per miner regardless
            # of buffer size (fixed overhead of rolling windows,
            # correlations, trends across 175 columns). Running all
            # 24 at once freezes the UI for 8 seconds. Staggering
            # means each tick spends ~250ms on ONE miner's prediction
            # and ~1ms on physics for the other 23.
            #
            # At 500ms tick interval, the ~250ms prediction leaves
            # ~250ms for Textual rendering — enough for smooth UI.
            # Each miner gets predicted once every n_miners ticks
            # (~12 seconds at 500ms × 24 miners). Between predictions,
            # scores hold their last value.
            #
            # Skip shutdown miners entirely — they have no useful
            # telemetry to predict on and waste the 250ms budget.
            if miner.mode == OperatingMode.SHUTDOWN:
                pass  # no prediction needed
            elif self.miners.index(miner) == (self.step % len(self.miners)):
                if self._ai_ready:
                    miner.anomaly_score, miner.predicted_failure = self.ai.predict(
                        miner.miner_id, health_score=miner.health_score,
                    )
                    risk = self.ai._risk_levels.get(miner.miner_id, "LOW")
                    if risk == "LOW" and miner.is_flagged:
                        miner.is_flagged = False
                else:
                    miner.anomaly_score = self._compute_anomaly_score(miner)
                    miner.predicted_failure = miner.anomaly_score > 0.7

            # Append to sparkline history deques (after all values finalized)
            miner.te_health_history.append(miner.te_health)
            miner.anomaly_score_history.append(miner.anomaly_score)
            miner.health_score_history.append(miner.health_score)

            # Optimizer + alerts
            new_actions.extend(self._run_optimizer(miner))
            new_alerts.extend(self._check_alerts(miner))

        self.alerts.extend(new_alerts)
        self.actions.extend(new_actions)
        self.alerts = self.alerts[-500:]
        self.actions = self.actions[-500:]

        # Fleet aggregates
        active = [m for m in self.miners if m.mode != OperatingMode.SHUTDOWN]
        self.total_hashrate_th = sum(m.hashrate_th for m in active)
        self.total_power_w = sum(m.power_w for m in active)
        if self.total_hashrate_th > 0:
            self.fleet_efficiency_jth = self.total_power_w / self.total_hashrate_th
        te_values = [m.te_health for m in active if m.te_health > 0]
        self.fleet_te_health = sum(te_values) / len(te_values) if te_values else 0

        self.healthy_count = sum(1 for m in self.miners if m.health_score > 0.8)
        self.warning_count = sum(1 for m in self.miners if 0.4 <= m.health_score <= 0.8)
        self.critical_count = sum(1 for m in self.miners if m.health_score < 0.4)

        self.ai.flush_to_db(self.step)

    # ── Scenario-driven failure effects ────────────────────────────

    def _apply_failure(self, miner: MinerState) -> None:
        """Apply failure effects using the scenario engine."""
        if not miner._scenario_name:
            return
        if self.step < miner._failure_onset_step:
            return

        scenario = SCENARIOS.get(miner._scenario_name)
        if not scenario:
            return

        # Compute progress
        miner.failure_progress = compute_progression(
            self.step, miner._failure_onset_step, miner._failure_duration,
        )

        # Build current values (includes new physics keys for extended scenarios)
        current = {
            "temperature": miner.temperature_c,
            "power": miner.power_w,
            "voltage": miner.voltage_v,
            "degradation_factor": miner._degradation_factor,
            "thermal_resistance": miner.spec.thermal_resistance,
            # New physics parameters (used by physics-first scenarios)
            "chip_efficiency": getattr(miner, "_chip_efficiency", 1.0),
            "leakage_coefficient": getattr(miner, "_leakage_coefficient", 1.0),
            "psu_internal_resistance": getattr(miner, "_psu_internal_resistance", 0.0008),
            "solder_resistance": getattr(miner, "_solder_resistance", 0.0),
            "fan_capacity": getattr(miner, "_fan_capacity", 1.0),
            "voltage_regulation_gain": getattr(miner, "_voltage_regulation_gain", 1.0),
        }

        # Apply scenario effects
        modified = apply_scenario_effects(
            scenario, self.step, miner._failure_onset_step,
            miner._failure_duration, current, self.np_rng,
        )

        # Apply back to miner state
        miner.temperature_c = modified.get("temperature", miner.temperature_c)
        miner.voltage_v = modified.get("voltage", miner.voltage_v)
        miner._degradation_factor = modified.get("degradation_factor", miner._degradation_factor)
        miner.spec.thermal_resistance = modified.get("thermal_resistance", miner.spec.thermal_resistance)
        # Write back new physics params if modified
        miner._chip_efficiency = modified.get("chip_efficiency", 1.0)
        miner._leakage_coefficient = modified.get("leakage_coefficient", 1.0)
        miner._psu_internal_resistance = modified.get("psu_internal_resistance", 0.0008)
        miner._solder_resistance = modified.get("solder_resistance", 0.0)
        miner._fan_capacity = modified.get("fan_capacity", 1.0)
        miner._voltage_regulation_gain = modified.get("voltage_regulation_gain", 1.0)

        # Clamp degradation
        miner._degradation_factor = max(0.0, min(1.0, miner._degradation_factor))

        # Shutdown if temperature too high
        if miner.temperature_c > miner.spec.temp_max_c:
            miner.mode = OperatingMode.SHUTDOWN

    # ── Heuristic fallback (when no trained model) ─────────────────

    def _compute_health(self, miner: MinerState) -> float:
        score = 1.0
        if miner.temperature_c > miner.spec.temp_throttle_c:
            score -= 0.3 * (
                (miner.temperature_c - miner.spec.temp_throttle_c)
                / (miner.spec.temp_max_c - miner.spec.temp_throttle_c)
            )
        score *= (0.3 + 0.7 * miner.hashrate_realization)
        score *= miner._degradation_factor
        return max(0.0, min(1.0, score))

    def _compute_anomaly_score(self, miner: MinerState) -> float:
        """Heuristic fallback when no trained model is available."""
        score = 0.0
        if miner.temperature_c > miner.spec.temp_throttle_c:
            score += 0.3 * min(1, (miner.temperature_c - miner.spec.temp_throttle_c) / 15)
        if miner.hashrate_realization < 0.9:
            score += 0.4 * (1 - miner.hashrate_realization)
        if miner._degradation_factor < 0.95:
            score += 0.3 * (1 - miner._degradation_factor)
        score += self.rng.gauss(0, 0.02)
        return max(0.0, min(1.0, score))

    # ── Optimizer ──────────────────────────────────────────────────

    def _run_optimizer(self, miner: MinerState) -> list[OptimizerAction]:
        actions = []
        ts = time.time()

        if miner.temperature_c > 90:
            new_freq = max(miner.spec.freq_min_mhz, miner.frequency_mhz - 50)
            if new_freq != miner.frequency_mhz:
                actions.append(OptimizerAction(
                    ts, miner.miner_id, "REDUCE_FREQ",
                    miner.frequency_mhz, new_freq,
                    f"Thermal critical: {miner.temperature_c:.1f}C"
                ))
                miner.frequency_mhz = new_freq
        elif miner.temperature_c > 82:
            new_freq = max(miner.spec.freq_min_mhz, miner.frequency_mhz - 20)
            if new_freq != miner.frequency_mhz:
                actions.append(OptimizerAction(
                    ts, miner.miner_id, "REDUCE_FREQ",
                    miner.frequency_mhz, new_freq,
                    f"Thermal warning: {miner.temperature_c:.1f}C"
                ))
                miner.frequency_mhz = new_freq

        if self.energy_price_usd < 0.035 and miner.temperature_c < 75:
            new_freq = min(miner.spec.freq_max_mhz, miner.frequency_mhz + 15)
            if new_freq != miner.frequency_mhz:
                actions.append(OptimizerAction(
                    ts, miner.miner_id, "BOOST_FREQ",
                    miner.frequency_mhz, new_freq,
                    f"Cheap energy: ${self.energy_price_usd:.3f}/kWh"
                ))
                miner.frequency_mhz = new_freq
        elif self.energy_price_usd > 0.07:
            new_freq = max(miner.spec.freq_min_mhz, miner.frequency_mhz - 10)
            if new_freq != miner.frequency_mhz:
                actions.append(OptimizerAction(
                    ts, miner.miner_id, "THROTTLE_FREQ",
                    miner.frequency_mhz, new_freq,
                    f"Expensive energy: ${self.energy_price_usd:.3f}/kWh"
                ))
                miner.frequency_mhz = new_freq

        if miner.predicted_failure and not miner.is_flagged:
            miner.is_flagged = True
            actions.append(OptimizerAction(
                ts, miner.miner_id, "FLAG_MAINTENANCE",
                0, 1,
                f"Anomaly score: {miner.anomaly_score:.4f}"
            ))

        return actions

    # ── Alert generation ───────────────────────────────────────────

    def _check_alerts(self, miner: MinerState) -> list[Alert]:
        alerts = []
        ts = time.time()

        if miner.temperature_c > miner.spec.temp_max_c:
            alerts.append(Alert(
                ts, miner.miner_id, AlertSeverity.CRITICAL,
                f"THERMAL SHUTDOWN {miner.temperature_c:.1f}C > {miner.spec.temp_max_c}C",
                miner.failure_type,
            ))
        elif miner.temperature_c > 90:
            alerts.append(Alert(
                ts, miner.miner_id, AlertSeverity.HIGH,
                f"Temperature critical: {miner.temperature_c:.1f}C",
                miner.failure_type,
            ))
        elif miner.temperature_c > miner.spec.temp_throttle_c:
            if self.step % 10 == 0:
                alerts.append(Alert(
                    ts, miner.miner_id, AlertSeverity.MEDIUM,
                    f"Temperature warning: {miner.temperature_c:.1f}C",
                    miner.failure_type,
                ))

        if miner.hashrate_realization < 0.7 and miner.mode != OperatingMode.SHUTDOWN:
            alerts.append(Alert(
                ts, miner.miner_id, AlertSeverity.HIGH,
                f"Hashrate drop: {miner.hashrate_realization:.0%} of nameplate",
                miner.failure_type,
            ))

        if self._ai_ready and self.step % 20 == 0:
            risk = self.ai._risk_levels.get(miner.miner_id, "LOW")
            sustained = self.ai._last_scores.get(miner.miner_id, {}).get("sustained_minutes", 0)
            if risk == "CRITICAL":
                alerts.append(Alert(
                    ts, miner.miner_id, AlertSeverity.CRITICAL,
                    f"AI Risk: CRITICAL — anomaly sustained {sustained} min, immediate action needed",
                    miner.failure_type,
                ))
            elif risk == "HIGH":
                alerts.append(Alert(
                    ts, miner.miner_id, AlertSeverity.HIGH,
                    f"AI Risk: HIGH — anomaly sustained {sustained} min, schedule inspection",
                    miner.failure_type,
                ))
            elif risk == "ELEVATED" and self.step % 60 == 0:  # less frequent for elevated
                alerts.append(Alert(
                    ts, miner.miner_id, AlertSeverity.MEDIUM,
                    f"AI Risk: ELEVATED — watching anomaly ({sustained} min)",
                    miner.failure_type,
                ))
        elif not self._ai_ready and miner.predicted_failure and self.step % 20 == 0:
            alerts.append(Alert(
                ts, miner.miner_id, AlertSeverity.HIGH,
                f"Heuristic: anomaly detected (score={miner.anomaly_score:.2f})",
                miner.failure_type,
            ))

        return alerts
