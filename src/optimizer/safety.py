"""
Safety constraint enforcement layer.
Every proposed action must pass through this guard before execution.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict
import time

from ..config import OptimizerConfig, MinerSpec


@dataclass
class ControlAction:
    miner_id: str
    action_type: str       # set_frequency, set_voltage, set_mode, flag_maintenance
    value: float
    reason: str
    priority: str = "medium"   # critical, high, medium, low
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class SafetyVerdict:
    approved: bool
    original_action: ControlAction
    modified_action: Optional[ControlAction] = None
    rejection_reason: Optional[str] = None


class SafetyGuard:
    """
    Firewall between optimizer decisions and hardware commands.
    Enforces thermal limits, rate limiting, and value bounds.
    """

    def __init__(self, config: OptimizerConfig = OptimizerConfig()):
        self.config = config
        self._last_action_time: Dict[str, float] = {}  # miner_id -> timestamp

    def check_action(
        self,
        action: ControlAction,
        current_temperature: float,
        spec: Optional[MinerSpec] = None,
    ) -> SafetyVerdict:
        """Validate action against safety constraints."""

        # 1. Thermal shutdown override — reject any non-shutdown action if too hot
        if current_temperature >= self.config.thermal_shutdown_c:
            if action.action_type != "flag_maintenance":
                return SafetyVerdict(
                    approved=False,
                    original_action=action,
                    rejection_reason=f"Thermal shutdown: {current_temperature:.1f}C >= {self.config.thermal_shutdown_c}C",
                )

        # 2. Rate limiting
        last_time = self._last_action_time.get(action.miner_id, 0)
        if action.action_type.startswith("set_") and \
           (action.timestamp - last_time) < self.config.min_change_interval_seconds:
            return SafetyVerdict(
                approved=False,
                original_action=action,
                rejection_reason=f"Rate limited: last action {action.timestamp - last_time:.0f}s ago",
            )

        # 3. Value bounds clamping
        modified = None
        if action.action_type == "set_frequency" and spec:
            clamped = max(spec.freq_min_mhz, min(action.value, spec.freq_max_mhz))
            if clamped != action.value:
                modified = ControlAction(
                    miner_id=action.miner_id,
                    action_type=action.action_type,
                    value=clamped,
                    reason=f"{action.reason} (clamped to [{spec.freq_min_mhz}, {spec.freq_max_mhz}])",
                    priority=action.priority,
                    timestamp=action.timestamp,
                )

        # Record action time
        self._last_action_time[action.miner_id] = action.timestamp

        return SafetyVerdict(
            approved=True,
            original_action=action,
            modified_action=modified,
        )

    def enforce_thermal_shutdown(
        self,
        miner_id: str,
        temperature_c: float,
    ) -> Optional[ControlAction]:
        """Emergency action if temperature exceeds hard limit."""
        if temperature_c >= self.config.thermal_shutdown_c:
            return ControlAction(
                miner_id=miner_id,
                action_type="set_mode",
                value=0,  # shutdown
                reason=f"EMERGENCY: {temperature_c:.1f}C >= {self.config.thermal_shutdown_c}C",
                priority="critical",
            )
        return None
