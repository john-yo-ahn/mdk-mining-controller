"""
DEPRECATED: Legacy failure type enum kept for dashboard compatibility only.

The scenario engine in src/synthetic/scenarios.py has replaced this module.
The new physics-first generator uses `MinerPhysicsParams` and `apply_scenario_phase`
to drift parameters, rather than hardcoded failure-specific math.

This file only exists to provide the `FailureType` enum that
`src/cli/simulation.py` imports.
"""

from enum import Enum


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
