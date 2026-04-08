"""
Scenario Engine for failure injection.

Two parallel systems that share the same FailureScenario registry:

1. LEGACY (SignalEffect + apply_scenario_effects):
   - Used by the live dashboard (src/cli/simulation.py)
   - Modifies output signals directly (temperature += 5)
   - Kept for backward compatibility

2. PHYSICS-FIRST (ScenarioPhase + apply_scenario_phase):
   - Used by the offline generator (src/synthetic/generator.py)
   - Drifts upstream physics parameters naturally
   - 3-phase lifecycle: incubation → acceleration → cascade
   - Enables long-timescale predictive signals
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import MinerPhysicsParams


# ─── Signal effects (legacy path, for dashboard) ─────────────────────

@dataclass
class SignalEffect:
    """Describes how one output signal changes during a failure (legacy)."""
    signal: str
    mode: str          # "scale", "offset", "replace", "noise"
    curve: str         # "linear", "exponential", "step", "intermittent", "sine"
    magnitude: float
    phase: float = 0.0
    frequency: float = 1.0


# ─── Scenario phases (physics-first path, for generator) ─────────────

@dataclass
class ScenarioPhase:
    """
    One phase of a multi-phase degradation lifecycle.

    name:                  "incubation" | "acceleration" | "cascade"
    duration_minutes_range: (min, max) — concrete duration picked at runtime
    drift_rate_per_step:   fraction of total drift_target applied per minute
                           (so a phase lasting 10000 steps with rate 0.0001
                            applies the full drift over this phase)
    noise_multiplier:      scales noise/variance during this phase
    """
    name: str
    duration_minutes_range: Tuple[int, int]
    drift_rate_per_step: float
    noise_multiplier: float = 1.0


@dataclass
class FailureScenario:
    """
    A complete failure scenario definition.

    Two coexisting sets of fields:

    LEGACY:
      effects:        list of SignalEffect (used by dashboard)
      duration_range: legacy total duration

    PHYSICS-FIRST:
      phases:         list of ScenarioPhase (used by generator)
      primary_param:  which physics parameter is drifted
      drift_target:   final multiplier/absolute target for primary_param
    """
    name: str
    description: str
    effects: List[SignalEffect]
    duration_range: Tuple[int, int] = (500, 3000)
    detection_hint: str = ""

    # Physics-first fields (optional — fallback to legacy if absent)
    phases: Optional[List[ScenarioPhase]] = None
    primary_param: Optional[str] = None
    drift_target: float = 1.0
    drift_mode: str = "multiply"  # "multiply" or "absolute"


# ─── Legacy: signal-level effect curves ──────────────────────────────

def compute_progression(step: int, onset: int, duration: int) -> float:
    """Legacy progression. 0.0 at onset, 1.0 at full failure."""
    if step < onset:
        return 0.0
    return min(1.0, (step - onset) / max(duration, 1))


def apply_curve(progress: float, curve: str, step: int = 0,
                frequency: float = 1.0, phase: float = 0.0) -> float:
    """Map progress through a curve shape."""
    if curve == "linear":
        return progress
    elif curve == "exponential":
        return (math.exp(3 * progress) - 1) / (math.exp(3) - 1)
    elif curve == "step":
        return 1.0 if progress > 0 else 0.0
    elif curve == "intermittent":
        if progress < 0.1:
            return 0.0
        cycle = math.sin(step * 0.1 * frequency + phase)
        threshold = 1.0 - progress
        return 1.0 if cycle > threshold else 0.0
    elif curve == "sine":
        return abs(math.sin(step * 0.05 * frequency + phase)) * progress
    elif curve == "plateau":
        return min(1.0, progress * 3)
    else:
        return progress


def apply_scenario_effects(
    scenario: FailureScenario,
    step: int,
    onset: int,
    duration: int,
    current_values: Dict[str, float],
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Legacy: apply SignalEffects to current signal values."""
    progress = compute_progression(step, onset, duration)
    if progress <= 0:
        return current_values

    result = dict(current_values)

    for effect in scenario.effects:
        curve_val = apply_curve(progress, effect.curve, step, effect.frequency, effect.phase)
        strength = curve_val * effect.magnitude

        signal = effect.signal
        if signal not in result and signal not in ("thermal_resistance", "degradation_factor"):
            continue

        if effect.mode == "scale":
            if signal in result:
                result[signal] *= (1.0 + strength)
            elif signal == "degradation_factor":
                result["degradation_factor"] = result.get("degradation_factor", 1.0) * (1.0 + strength)
        elif effect.mode == "offset":
            if signal in result:
                result[signal] += strength
        elif effect.mode == "replace":
            result[signal] = strength
        elif effect.mode == "noise":
            if signal in result:
                result[signal] += rng.normal(0, abs(strength))

    return result


# ─── Physics-first: phase-based drift ────────────────────────────────

def resolve_phase_durations(
    scenario: FailureScenario,
    rng: np.random.Generator,
) -> List[int]:
    """Pick concrete durations from each phase's range."""
    if not scenario.phases:
        return []
    return [
        int(rng.integers(p.duration_minutes_range[0], p.duration_minutes_range[1] + 1))
        for p in scenario.phases
    ]


def compute_phase_at_step(
    step: int,
    onset: int,
    phase_durations: List[int],
) -> Tuple[str, int, int]:
    """
    Returns (phase_name, steps_into_this_phase, total_duration_covered_so_far).
    phase_name is "healthy" before onset, "failed" after all phases complete.
    """
    if step < onset:
        return ("healthy", 0, 0)

    elapsed = step - onset
    cumulative = 0
    phase_names = ["incubation", "acceleration", "cascade"]

    for i, dur in enumerate(phase_durations):
        phase_name = phase_names[i] if i < len(phase_names) else f"phase{i}"
        if elapsed < cumulative + dur:
            return (phase_name, elapsed - cumulative, cumulative)
        cumulative += dur

    return ("failed", elapsed - cumulative, cumulative)


def _apply_drift(
    current: float,
    base: float,
    target: float,
    rate: float,
    mode: str,
) -> float:
    """
    Apply one drift step.
    - multiply mode: current -> current * (1 + rate * sign)
    - absolute mode: current += rate * (target - base)
    """
    if mode == "multiply":
        # Drift factor moves current toward base * target
        final_val = base * target
        direction = 1.0 if final_val > current else -1.0
        return current * (1.0 + rate * direction)
    else:  # absolute
        final_val = base + target
        step = (final_val - base) * rate
        return current + step


def apply_scenario_phase(
    scenario: FailureScenario,
    step: int,
    onset: int,
    phase_durations: List[int],
    params: MinerPhysicsParams,
    rng: np.random.Generator,
) -> Tuple[str, float]:
    """
    Drift the miner's physics parameters in-place according to the current phase.
    Returns (phase_name, days_to_failure).

    days_to_failure is:
      - NaN before onset (healthy)
      - positive during incubation/acceleration/cascade
      - 0 at full failure
    """
    phase_name, steps_into, cumulative = compute_phase_at_step(step, onset, phase_durations)

    if phase_name == "healthy":
        return ("healthy", float("nan"))

    if phase_name == "failed":
        return ("failed", 0.0)

    # Get the phase object
    phase_idx = ["incubation", "acceleration", "cascade"].index(phase_name) \
        if phase_name in ["incubation", "acceleration", "cascade"] else 0
    if phase_idx >= len(scenario.phases):
        return ("failed", 0.0)

    phase = scenario.phases[phase_idx]
    primary_param = scenario.primary_param

    if primary_param and hasattr(params, primary_param):
        current_value = getattr(params, primary_param)
        base_value = getattr(params, f"{primary_param}_base")
        new_value = _apply_drift(
            current_value, base_value, scenario.drift_target,
            phase.drift_rate_per_step, scenario.drift_mode,
        )
        setattr(params, primary_param, new_value)
        params.clamp()

    # Compute days remaining
    total_duration = sum(phase_durations)
    remaining_steps = max(0, total_duration - (step - onset))
    days_to_failure = remaining_steps / (60 * 24)  # 1-min steps

    return (phase_name, days_to_failure)


# ─── Built-in scenarios ──────────────────────────────────────────────

SCENARIOS: Dict[str, FailureScenario] = {}


def register_scenario(scenario: FailureScenario):
    SCENARIOS[scenario.name] = scenario
    return scenario


# Helper for phase definitions
def _phases(incubation_minutes, acceleration_minutes, cascade_minutes,
            inc_rate=0.0001, acc_rate=0.001, cas_rate=0.01):
    """Create standard 3-phase lifecycle."""
    return [
        ScenarioPhase(
            name="incubation",
            duration_minutes_range=incubation_minutes,
            drift_rate_per_step=inc_rate,
            noise_multiplier=1.0,
        ),
        ScenarioPhase(
            name="acceleration",
            duration_minutes_range=acceleration_minutes,
            drift_rate_per_step=acc_rate,
            noise_multiplier=1.5,
        ),
        ScenarioPhase(
            name="cascade",
            duration_minutes_range=cascade_minutes,
            drift_rate_per_step=cas_rate,
            noise_multiplier=3.0,
        ),
    ]


# Time constants: 1 day = 1440 minutes, 1 hour = 60 minutes

# 1. Gradual chip efficiency decline (solder fatigue / electromigration)
register_scenario(FailureScenario(
    name="gradual_degradation",
    description="Slow hashrate decline from solder joint fatigue or electromigration. "
                "Chip efficiency drifts downward over weeks. Power stays constant; J/TH rises.",
    effects=[
        SignalEffect("degradation_factor", "scale", "linear", -0.35),
        SignalEffect("power", "noise", "linear", 50),
        SignalEffect("temperature", "offset", "exponential", 3.0),
    ],
    duration_range=(3000, 10000),
    detection_hint="Watch for J/TH trending upward while frequency is unchanged",
    phases=_phases(
        incubation_minutes=(14 * 1440, 14 * 1440),   # 14 days
        acceleration_minutes=(7 * 1440, 7 * 1440),    # 7 days
        cascade_minutes=(1 * 1440, 1 * 1440),         # 1 day
        inc_rate=0.00001,
        acc_rate=0.00012,
        cas_rate=0.0005,
    ),
    primary_param="chip_efficiency",
    drift_target=0.65,  # chip_efficiency * 0.65 = 35% loss
    drift_mode="multiply",
))

# 2. Thermal runaway (rapid cascade)
register_scenario(FailureScenario(
    name="thermal_runaway",
    description="Thermal feedback loop: rising temp increases leakage power, "
                "which generates more heat. Starts slow, accelerates exponentially.",
    effects=[
        SignalEffect("temperature", "offset", "exponential", 25.0),
        SignalEffect("power", "offset", "exponential", 500),
        SignalEffect("degradation_factor", "scale", "exponential", -0.5),
    ],
    duration_range=(100, 500),
    detection_hint="Temperature rising without frequency/workload change",
    phases=_phases(
        incubation_minutes=(5 * 1440, 5 * 1440),      # 5 days
        acceleration_minutes=(1 * 1440, 1 * 1440),    # 1 day
        cascade_minutes=(4 * 60, 4 * 60),             # 4 hours
        inc_rate=0.00002,
        acc_rate=0.00020,
        cas_rate=0.0020,
    ),
    primary_param="thermal_resistance",
    drift_target=1.5,  # Rth * 1.5 = 50% worse cooling
    drift_mode="multiply",
))

# 3. Fan bearing failure (gradual cooling loss)
register_scenario(FailureScenario(
    name="fan_stall",
    description="Fan bearing wears gradually. Cooling capacity drops, thermal resistance rises.",
    effects=[
        SignalEffect("thermal_resistance", "scale", "linear", 2.0),
        SignalEffect("temperature", "offset", "linear", 8.0),
        SignalEffect("temperature", "noise", "linear", 2.0),
    ],
    duration_range=(500, 3000),
    detection_hint="Temperature slowly climbing, noisier readings",
    phases=_phases(
        incubation_minutes=(10 * 1440, 10 * 1440),    # 10 days
        acceleration_minutes=(3 * 1440, 3 * 1440),    # 3 days
        cascade_minutes=(12 * 60, 12 * 60),           # 12 hours
        inc_rate=0.00001,
        acc_rate=0.00015,
        cas_rate=0.0008,
    ),
    primary_param="fan_capacity",
    drift_target=0.4,  # fan_capacity * 0.4
    drift_mode="multiply",
))

# 4. PSU capacitor aging
register_scenario(FailureScenario(
    name="psu_degradation",
    description="Electrolytic capacitors lose capacitance. Voltage ripple grows, "
                "causing intermittent compute errors and hashrate dips.",
    effects=[
        SignalEffect("voltage", "noise", "exponential", 0.02),
        SignalEffect("voltage", "sine", "sine", 0.015, frequency=2.0),
        SignalEffect("degradation_factor", "scale", "intermittent", -0.15),
        SignalEffect("power", "noise", "linear", 100),
    ],
    duration_range=(2000, 8000),
    detection_hint="Voltage variance increasing, intermittent hashrate dips",
    phases=_phases(
        incubation_minutes=(21 * 1440, 21 * 1440),    # 21 days
        acceleration_minutes=(10 * 1440, 10 * 1440),  # 10 days
        cascade_minutes=(1 * 1440, 1 * 1440),         # 1 day
        inc_rate=0.000008,
        acc_rate=0.00008,
        cas_rate=0.0004,
    ),
    primary_param="psu_internal_resistance",
    drift_target=4.0,  # 4× PSU resistance
    drift_mode="multiply",
))

# 5. Sudden hashboard failure (instant)
register_scenario(FailureScenario(
    name="sudden_chip_failure",
    description="Hashboards fail instantly. No warning signs. Hashrate drops 33-66% immediately.",
    effects=[
        SignalEffect("degradation_factor", "scale", "step", -0.66),
        SignalEffect("power", "scale", "step", -0.60),
    ],
    duration_range=(1, 1),
    detection_hint="Instant large hashrate drop",
    phases=[
        ScenarioPhase(
            name="incubation",
            duration_minutes_range=(1, 1),
            drift_rate_per_step=0.0,
            noise_multiplier=1.0,
        ),
        ScenarioPhase(
            name="acceleration",
            duration_minutes_range=(1, 1),
            drift_rate_per_step=0.0,
            noise_multiplier=1.0,
        ),
        ScenarioPhase(
            name="cascade",
            duration_minutes_range=(1, 1),
            drift_rate_per_step=1.0,  # full drift in 1 step
            noise_multiplier=1.0,
        ),
    ],
    primary_param="chip_efficiency",
    drift_target=0.34,  # lose 2/3
    drift_mode="multiply",
))

# 6. Coolant restriction (immersion mining)
register_scenario(FailureScenario(
    name="coolant_restriction",
    description="Partial blockage in coolant loop. Flow rate drops, causing uneven cooling.",
    effects=[
        SignalEffect("thermal_resistance", "scale", "exponential", 1.5),
        SignalEffect("temperature", "offset", "exponential", 12.0),
        SignalEffect("temperature", "noise", "linear", 3.0),
        SignalEffect("degradation_factor", "scale", "linear", -0.10),
    ],
    duration_range=(1000, 5000),
    detection_hint="Temperature variance increases, hotspots emerge",
    phases=_phases(
        incubation_minutes=(14 * 1440, 14 * 1440),    # 14 days
        acceleration_minutes=(5 * 1440, 5 * 1440),    # 5 days
        cascade_minutes=(6 * 60, 6 * 60),             # 6 hours
        inc_rate=0.000012,
        acc_rate=0.00018,
        cas_rate=0.0012,
    ),
    primary_param="thermal_resistance",
    drift_target=1.8,
    drift_mode="multiply",
))

# 7. Firmware oscillation bug
register_scenario(FailureScenario(
    name="firmware_oscillation",
    description="Firmware autotune bug causes frequency/voltage to oscillate. "
                "Not a hardware failure but looks like one to simple monitors.",
    effects=[
        SignalEffect("degradation_factor", "scale", "sine", -0.20, frequency=3.0),
        SignalEffect("power", "noise", "sine", 200, frequency=3.0),
        SignalEffect("temperature", "offset", "sine", 5.0, frequency=3.0, phase=1.5),
    ],
    duration_range=(500, 2000),
    detection_hint="Periodic hashrate oscillation with matching power/temp cycles",
    phases=_phases(
        incubation_minutes=(2 * 1440, 2 * 1440),      # 2 days
        acceleration_minutes=(1 * 1440, 1 * 1440),    # 1 day
        cascade_minutes=(6 * 60, 6 * 60),             # 6 hours
        inc_rate=0.00005,
        acc_rate=0.00030,
        cas_rate=0.0010,
    ),
    primary_param="voltage_regulation_gain",
    drift_target=0.7,
    drift_mode="multiply",
))

# 8. Connector corrosion
register_scenario(FailureScenario(
    name="connector_corrosion",
    description="Humidity causes corrosion on power connectors. "
                "Intermittent contact resistance grows. Voltage dips, power spikes.",
    effects=[
        SignalEffect("voltage", "noise", "exponential", 0.03),
        SignalEffect("power", "noise", "intermittent", 300, frequency=0.5),
        SignalEffect("degradation_factor", "scale", "intermittent", -0.10, frequency=0.7),
        SignalEffect("temperature", "offset", "linear", 2.0),
    ],
    duration_range=(2000, 6000),
    detection_hint="Intermittent voltage drops, sporadic power spikes",
    phases=_phases(
        incubation_minutes=(21 * 1440, 21 * 1440),    # 21 days
        acceleration_minutes=(7 * 1440, 7 * 1440),    # 7 days
        cascade_minutes=(1 * 1440, 1 * 1440),         # 1 day
        inc_rate=0.000008,
        acc_rate=0.0001,
        cas_rate=0.0006,
    ),
    primary_param="solder_resistance",
    drift_target=0.005,  # absolute target (+0.005 ohm)
    drift_mode="absolute",
))


def list_scenarios() -> List[str]:
    return list(SCENARIOS.keys())


def get_scenario(name: str) -> FailureScenario:
    return SCENARIOS[name]


def create_custom_scenario(
    name: str,
    description: str,
    effects: List[Dict],
    duration_range: tuple = (500, 3000),
    detection_hint: str = "",
) -> FailureScenario:
    scenario = FailureScenario(
        name=name,
        description=description,
        effects=[SignalEffect(**e) for e in effects],
        duration_range=duration_range,
        detection_hint=detection_hint,
    )
    register_scenario(scenario)
    return scenario
