"""
Rule-based efficiency optimizer.
Implements condition-action policies for frequency, voltage, and maintenance.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from ..config import OptimizerConfig, DEFAULT_MINER_SPECS
from .safety import SafetyGuard, ControlAction


class RuleBasedOptimizer:
    """
    Evaluates rules per miner per timestep.
    All proposed actions pass through the SafetyGuard.
    """

    def __init__(
        self,
        config: OptimizerConfig = OptimizerConfig(),
        safety: Optional[SafetyGuard] = None,
    ):
        self.config = config
        self.safety = safety or SafetyGuard(config)

    def evaluate(
        self,
        miner_id: str,
        temperature_c: float,
        hashrate_th: float,
        power_w: float,
        te_health: float,
        energy_price: float,
        current_frequency_mhz: float,
        ambient_temp_c: float,
        model: str = "",
        anomaly_score: float = 0.0,
        predicted_failure: bool = False,
    ) -> List[ControlAction]:
        """Evaluate all rules for one miner at one timestep."""
        spec = DEFAULT_MINER_SPECS.get(model)
        actions = []

        # Thermal management
        actions.extend(self._rule_thermal(
            miner_id, temperature_c, current_frequency_mhz, spec,
        ))

        # Energy price optimization
        actions.extend(self._rule_energy_price(
            miner_id, energy_price, current_frequency_mhz, temperature_c, spec,
        ))

        # Degradation response
        actions.extend(self._rule_degradation(
            miner_id, te_health, anomaly_score, predicted_failure,
        ))

        # Filter through safety guard
        approved = []
        for action in actions:
            verdict = self.safety.check_action(action, temperature_c, spec)
            if verdict.approved:
                approved.append(verdict.modified_action or verdict.original_action)

        return approved

    def _rule_thermal(
        self,
        miner_id: str,
        temperature_c: float,
        current_freq: float,
        spec,
    ) -> List[ControlAction]:
        cfg = self.config
        actions = []

        if temperature_c > cfg.thermal_critical_c:
            new_freq = max(spec.freq_min_mhz if spec else 200,
                          current_freq - cfg.freq_step_mhz * 2)
            actions.append(ControlAction(
                miner_id, "set_frequency", new_freq,
                f"Thermal critical: {temperature_c:.1f}C", "high",
            ))
        elif temperature_c > cfg.thermal_warning_c:
            new_freq = max(spec.freq_min_mhz if spec else 200,
                          current_freq - cfg.freq_step_mhz)
            actions.append(ControlAction(
                miner_id, "set_frequency", new_freq,
                f"Thermal warning: {temperature_c:.1f}C", "medium",
            ))

        return actions

    def _rule_energy_price(
        self,
        miner_id: str,
        energy_price: float,
        current_freq: float,
        temperature_c: float,
        spec,
    ) -> List[ControlAction]:
        cfg = self.config
        actions = []

        if energy_price < cfg.energy_price_cheap_usd and temperature_c < cfg.thermal_warning_c - 10:
            new_freq = min(spec.freq_max_mhz if spec else 700,
                          current_freq + cfg.freq_step_mhz * 0.5)
            actions.append(ControlAction(
                miner_id, "set_frequency", new_freq,
                f"Cheap energy: ${energy_price:.3f}/kWh", "low",
            ))
        elif energy_price > cfg.energy_price_expensive_usd:
            new_freq = max(spec.freq_min_mhz if spec else 200,
                          current_freq - cfg.freq_step_mhz * 0.5)
            actions.append(ControlAction(
                miner_id, "set_frequency", new_freq,
                f"Expensive energy: ${energy_price:.3f}/kWh", "low",
            ))

        return actions

    def _rule_degradation(
        self,
        miner_id: str,
        te_health: float,
        anomaly_score: float,
        predicted_failure: bool,
    ) -> List[ControlAction]:
        actions = []

        if predicted_failure or anomaly_score > 0.7:
            actions.append(ControlAction(
                miner_id, "flag_maintenance", 1,
                f"AI prediction: anomaly_score={anomaly_score:.2f}", "high",
            ))

        return actions

    def evaluate_fleet(
        self,
        fleet_df: pd.DataFrame,
        energy_price: float,
    ) -> pd.DataFrame:
        """Evaluate rules across all miners. Returns DataFrame of actions."""
        all_actions = []

        for _, row in fleet_df.iterrows():
            actions = self.evaluate(
                miner_id=row.get("miner_id", ""),
                temperature_c=row.get("temperature_c", 0),
                hashrate_th=row.get("hashrate_th", 0),
                power_w=row.get("power_w", 0),
                te_health=row.get("te_health", 0),
                energy_price=energy_price,
                current_frequency_mhz=row.get("clock_frequency_mhz", 500),
                ambient_temp_c=row.get("ambient_temperature_c", 30),
                model=row.get("model", ""),
                anomaly_score=row.get("anomaly_score", 0),
                predicted_failure=row.get("predicted_failure", False),
            )
            for a in actions:
                all_actions.append({
                    "miner_id": a.miner_id,
                    "action": a.action_type,
                    "value": a.value,
                    "reason": a.reason,
                    "priority": a.priority,
                })

        return pd.DataFrame(all_actions) if all_actions else pd.DataFrame()
