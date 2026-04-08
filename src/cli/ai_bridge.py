"""
Bridge between the live simulation and the trained ML models.
Maintains a rolling telemetry buffer per miner, computes features on-the-fly,
and runs real XGBoost/LSTM inference instead of fake heuristics.
Also persists telemetry to DuckDB.
"""

import collections
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero")

from ..config import MODELS_DIR, RAW_DIR, LIVE_DB_PATH, TEConfig

# Feature config matching what the model was trained on
ROLLING_WINDOWS = [2, 15, 60]
SENSOR_COLS = ["temperature_c", "hashrate_th", "power_w", "voltage_v"]
RATE_COLS = ["temperature_c", "hashrate_th", "power_w", "efficiency_jth"]
BUFFER_SIZE = 120  # keep 2 hours of 1-min data per miner


class MinerBuffer:
    """Rolling ring buffer of raw telemetry for one miner."""

    __slots__ = [
        "miner_id", "model", "nameplate_hashrate",
        "clock_frequency_mhz", "voltage_v", "hashrate_th",
        "temperature_c", "power_w", "ambient_temperature_c",
        "_size", "_count",
    ]

    def __init__(self, miner_id: str, model: str, nameplate_hashrate: float, size: int = BUFFER_SIZE):
        self.miner_id = miner_id
        self.model = model
        self.nameplate_hashrate = nameplate_hashrate
        self._size = size
        self._count = 0

        # Ring buffers as numpy arrays (fast)
        self.clock_frequency_mhz = np.zeros(size)
        self.voltage_v = np.zeros(size)
        self.hashrate_th = np.zeros(size)
        self.temperature_c = np.zeros(size)
        self.power_w = np.zeros(size)
        self.ambient_temperature_c = np.zeros(size)

    def push(self, freq, volt, hr, temp, pwr, ambient):
        idx = self._count % self._size
        self.clock_frequency_mhz[idx] = freq
        self.voltage_v[idx] = volt
        self.hashrate_th[idx] = hr
        self.temperature_c[idx] = temp
        self.power_w[idx] = pwr
        self.ambient_temperature_c[idx] = ambient
        self._count += 1

    @property
    def length(self) -> int:
        return min(self._count, self._size)

    def _ordered(self, arr) -> np.ndarray:
        """Return the buffer contents in chronological order."""
        n = self.length
        if self._count <= self._size:
            return arr[:n]
        start = self._count % self._size
        return np.concatenate([arr[start:], arr[:start]])[:n]

    def get_ordered(self, field: str) -> np.ndarray:
        return self._ordered(getattr(self, field))

    def export_lstm_sequence(self, seq_len: int = 30) -> Optional[np.ndarray]:
        """Export the last seq_len readings as a normalized numpy array for LSTM.
        Returns shape (1, seq_len, 6) or None if not enough data."""
        if self.length < seq_len:
            return None

        freq = self._ordered(self.clock_frequency_mhz)[-seq_len:]
        volt = self._ordered(self.voltage_v)[-seq_len:]
        hr = self._ordered(self.hashrate_th)[-seq_len:]
        temp = self._ordered(self.temperature_c)[-seq_len:]
        pwr = self._ordered(self.power_w)[-seq_len:]
        amb = self._ordered(self.ambient_temperature_c)[-seq_len:]

        raw = np.stack([freq, volt, hr, temp, pwr, amb], axis=1)  # (seq_len, 6)

        # Per-miner normalization (same as training)
        mean = raw.mean(axis=0)
        std = raw.std(axis=0)
        std[std == 0] = 1
        normalized = (raw - mean) / std

        return normalized.reshape(1, seq_len, 6).astype(np.float32)

    def compute_features(self) -> Optional[Dict[str, float]]:
        """
        Compute all 84 features that the XGBoost model expects.
        Returns None if buffer has < 60 readings (need at least 60m of data).
        """
        n = self.length
        if n < 61:
            return None

        # Get ordered arrays
        freq = self._ordered(self.clock_frequency_mhz)
        volt = self._ordered(self.voltage_v)
        hr = self._ordered(self.hashrate_th)
        temp = self._ordered(self.temperature_c)
        pwr = self._ordered(self.power_w)
        amb = self._ordered(self.ambient_temperature_c)

        # Use latest values for base features
        f = {}
        f["clock_frequency_mhz"] = freq[-1]
        f["voltage_v"] = volt[-1]
        f["hashrate_th"] = hr[-1]
        f["temperature_c"] = temp[-1]
        f["power_w"] = pwr[-1]
        f["ambient_temperature_c"] = amb[-1]

        # Derived
        f["efficiency_jth"] = pwr[-1] / hr[-1] if hr[-1] > 0 else 0
        f["temp_delta_c"] = temp[-1] - amb[-1]

        # Z-scores (relative to this miner's own buffer)
        for name, arr in [("temperature_c", temp), ("hashrate_th", hr),
                          ("power_w", pwr), ("efficiency_jth", None)]:
            if arr is None:
                eff = np.where(hr > 0, pwr / hr, 0)
                arr = eff
            mean = arr.mean()
            std = arr.std() or 1.0
            f[f"{name}_zscore"] = (arr[-1] - mean) / std

        # TE KPIs
        cfg = TEConfig()
        total_power = pwr[-1] * (1 + cfg.alpha_cooling + cfg.beta_infra)
        f["te_base"] = hr[-1] / total_power if total_power > 0 else 0
        temp_penalty = 1 - cfg.delta_temp * max(0, amb[-1] - cfg.temp_baseline_c)
        realization = min(1.0, hr[-1] / self.nameplate_hashrate) if self.nameplate_hashrate > 0 else 0
        f["te_adjusted"] = f["te_base"] * temp_penalty
        f["te_health"] = f["te_adjusted"] * realization

        # Cross-signal ratios
        f["jth"] = f["efficiency_jth"]
        f["temp_per_watt"] = temp[-1] / pwr[-1] * 1000 if pwr[-1] > 0 else 0
        f["hashrate_per_mhz"] = hr[-1] / freq[-1] if freq[-1] > 0 else 0
        f["power_per_mhz"] = pwr[-1] / freq[-1] if freq[-1] > 0 else 0

        # Rolling stats
        jth_arr = np.where(hr > 0, pwr / hr, 0)

        for col_name, arr in [("temperature_c", temp), ("hashrate_th", hr),
                              ("power_w", pwr), ("voltage_v", volt), ("jth", jth_arr)]:
            for win in ROLLING_WINDOWS:
                window = arr[-win:] if n >= win else arr
                prefix = f"{col_name}_roll_{win}m"
                f[f"{prefix}_mean"] = window.mean()
                f[f"{prefix}_std"] = window.std()
                f[f"{prefix}_min"] = window.min()
                f[f"{prefix}_max"] = window.max()

        # Rate of change (latest diff)
        eff_arr = np.where(hr > 0, pwr / hr, 0)
        for col_name, arr in [("temperature_c", temp), ("hashrate_th", hr),
                              ("power_w", pwr), ("efficiency_jth", eff_arr)]:
            f[f"{col_name}_rate"] = arr[-1] - arr[-2] if n >= 2 else 0

        # Degradation slope (linear regression on last 60 points of J/TH)
        jth_window = jth_arr[-60:]
        if len(jth_window) >= 10:
            x = np.arange(len(jth_window), dtype=float)
            valid = jth_window > 0
            if valid.sum() > 5:
                xv, yv = x[valid], jth_window[valid]
                slope = np.polyfit(xv, yv, 1)[0]
                f["jth_degradation_slope"] = slope
            else:
                f["jth_degradation_slope"] = 0
        else:
            f["jth_degradation_slope"] = 0

        return f


class AIBridge:
    """
    Connects the simulation to trained models and DuckDB storage.
    Drop-in replacement for the fake heuristics in simulation.py.
    """

    def __init__(self):
        self.buffers: Dict[str, MinerBuffer] = {}
        self.xgb_model = None
        self.xgb_features: List[str] = []
        self.xgb_threshold: float = 0.5
        self.lstm_model = None
        self.db = None
        self._db_buffer: List[dict] = []
        self._db_flush_interval = 50
        self._step = 0
        self._models_loaded = False

        # Risk level system (replaces binary prediction)
        self._consecutive_above: Dict[str, int] = {}   # consecutive ticks above threshold
        self._consecutive_below: Dict[str, int] = {}   # consecutive ticks below threshold
        self._last_scores: Dict[str, Dict] = {}
        self._risk_levels: Dict[str, str] = {}          # current risk level per miner

        # Thresholds tuned for useful predictions
        # Healthy miners typically score 0.09-0.14 combined (LSTM baseline noise)
        # Failing miners score 0.20+ as degradation progresses
        self._score_floor = 0.25        # below this = LOW (healthy baseline is ~0.10-0.18)
        self._elevated_minutes = 10     # sustained above floor for 10 min = ELEVATED
        self._high_minutes = 30         # sustained for 30 min = HIGH
        self._critical_minutes = 60     # sustained for 60 min = CRITICAL
        self._clear_minutes = 15        # below floor for 15 min = reset to LOW

    def load_models(self) -> bool:
        """Load trained models from disk. Returns True if at least XGBoost loaded."""
        xgb_path = MODELS_DIR / "xgboost_failure.joblib"
        lstm_path = MODELS_DIR / "lstm_ae.pt"

        # XGBoost
        if xgb_path.exists():
            try:
                import joblib
                data = joblib.load(xgb_path)
                self.xgb_model = data["model"]
                self.xgb_features = data["feature_names"]
                self.xgb_threshold = data["threshold"]
                print(f"  Loaded XGBoost model ({len(self.xgb_features)} features, threshold={self.xgb_threshold:.4f})")
            except Exception as e:
                print(f"  Failed to load XGBoost: {e}")

        # LSTM
        if lstm_path.exists():
            try:
                from ..models.lstm_autoencoder import AnomalyDetector
                self.lstm_model = AnomalyDetector.load(lstm_path)
                print(f"  Loaded LSTM-AE (threshold={self.lstm_model.threshold_:.6f})")
            except Exception as e:
                print(f"  Failed to load LSTM-AE: {e}")

        self._models_loaded = self.xgb_model is not None
        return self._models_loaded

    def init_storage(self) -> bool:
        """Initialize DuckDB for live telemetry persistence."""
        try:
            from ..storage.backend import DuckDBStore
            # ensure_live_schema=True creates the telemetry table on first
            # open so the clear() / flush_to_db() calls below always work,
            # even on a brand-new database file.
            self.db = DuckDBStore(LIVE_DB_PATH, ensure_live_schema=True)
            self.db.clear("telemetry")  # discard prior session's live rows
            print(f"  DuckDB storage initialized: {self.db.db_path}")
            return True
        except Exception as e:
            print(f"  Failed to init DuckDB: {e}")
            self.db = None
            return False

    def register_miner(self, miner_id: str, model: str, nameplate_hashrate: float):
        """Register a miner's buffer."""
        if miner_id not in self.buffers:
            self.buffers[miner_id] = MinerBuffer(miner_id, model, nameplate_hashrate)

    def push_telemetry(
        self,
        miner_id: str,
        freq: float, volt: float, hr: float,
        temp: float, pwr: float, ambient: float,
    ):
        """Push a new reading into the miner's buffer."""
        buf = self.buffers.get(miner_id)
        if buf:
            buf.push(freq, volt, hr, temp, pwr, ambient)

        # Buffer for DB persistence
        if self.db:
            self._db_buffer.append({
                "miner_id": miner_id,
                "clock_frequency_mhz": freq,
                "voltage_v": volt,
                "hashrate_th": hr,
                "temperature_c": temp,
                "power_w": pwr,
                "ambient_temperature_c": ambient,
            })

    def predict(self, miner_id: str, health_score: float = 1.0) -> tuple:
        """
        Run XGBoost + LSTM ensemble prediction with risk-level system.
        Incorporates health_score as a direct escalation signal.
        Returns (combined_score: float, predicted_failure: bool).

        Risk is determined by THREE signals:
          1. ML model score (XGBoost + LSTM) — learned patterns
          2. Sustained duration above threshold — persistence filter
          3. Health score — direct measurement of current degradation

        Health directly escalates risk:
          health > 0.8  → no effect (healthy)
          health 0.5-0.8 → minimum ELEVATED
          health 0.3-0.5 → minimum HIGH
          health < 0.3  → minimum CRITICAL
        """
        if not self._models_loaded:
            return 0.0, False

        buf = self.buffers.get(miner_id)
        if buf is None:
            return 0.0, False

        features = buf.compute_features()
        if features is None:
            return 0.0, False

        xgb_score = 0.0
        lstm_score = 0.0

        # XGBoost
        try:
            feature_vector = np.array([[features.get(f, 0.0) for f in self.xgb_features]])
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            xgb_score = float(self.xgb_model.predict_proba(feature_vector)[0, 1])
        except Exception:
            pass

        # LSTM
        if self.lstm_model is not None and hasattr(self.lstm_model, 'model_') and self.lstm_model.model_ is not None:
            try:
                seq = buf.export_lstm_sequence(seq_len=30)
                if seq is not None:
                    error = self.lstm_model.compute_reconstruction_error(seq)
                    if len(error) > 0 and self.lstm_model.threshold_ and self.lstm_model.threshold_ > 0:
                        lstm_score = min(1.0, float(error[0]) / (self.lstm_model.threshold_ * 3))
            except Exception:
                pass

        # Combine ML scores
        if lstm_score > 0:
            combined = 0.65 * xgb_score + 0.35 * lstm_score
        else:
            combined = xgb_score

        # ── Risk from ML model (time-sustained) ──────────────────
        above_floor = combined >= self._score_floor
        consec_above = self._consecutive_above.get(miner_id, 0)
        consec_below = self._consecutive_below.get(miner_id, 0)

        if above_floor:
            self._consecutive_above[miner_id] = consec_above + 1
            self._consecutive_below[miner_id] = 0
        else:
            self._consecutive_below[miner_id] = consec_below + 1
            if self._consecutive_below[miner_id] >= self._clear_minutes:
                self._consecutive_above[miner_id] = 0

        sustained = self._consecutive_above.get(miner_id, 0)

        if sustained >= self._critical_minutes:
            ml_risk = "CRITICAL"
        elif sustained >= self._high_minutes:
            ml_risk = "HIGH"
        elif sustained >= self._elevated_minutes:
            ml_risk = "ELEVATED"
        else:
            ml_risk = "LOW"

        # ── Risk from health score (direct measurement) ──────────
        if health_score < 0.3:
            health_risk = "CRITICAL"
        elif health_score < 0.5:
            health_risk = "HIGH"
        elif health_score < 0.8:
            health_risk = "ELEVATED"
        else:
            health_risk = "LOW"

        # ── Final risk = worst of the two ────────────────────────
        risk_order = {"LOW": 0, "ELEVATED": 1, "HIGH": 2, "CRITICAL": 3}
        risk = max(ml_risk, health_risk, key=lambda r: risk_order[r])
        predicted = risk in ("HIGH", "CRITICAL")

        # Determine what's driving the risk
        if risk_order[health_risk] > risk_order[ml_risk]:
            risk_source = "health"
        elif risk_order[ml_risk] > risk_order[health_risk]:
            risk_source = "ai_model"
        else:
            risk_source = "both"

        self._risk_levels[miner_id] = risk
        self._last_scores[miner_id] = {
            "xgb_score": xgb_score,
            "lstm_score": lstm_score,
            "combined": combined,
            "sustained_minutes": sustained,
            "risk_level": risk,
            "ml_risk": ml_risk,
            "health_risk": health_risk,
            "risk_source": risk_source,
            "health_score": health_score,
        }

        return float(combined), predicted

    def flush_to_db(self, step: int):
        """Periodically flush buffered telemetry to DuckDB."""
        self._step = step
        if not self.db or not self._db_buffer:
            return
        if step % self._db_flush_interval != 0:
            return

        try:
            df = pd.DataFrame(self._db_buffer)
            # Build in exact schema column order
            now = pd.Timestamp.now()
            insert_df = pd.DataFrame({
                "timestamp": now,
                "miner_id": df["miner_id"],
                "model": "",
                "container_id": "",
                "position": 0,
                "clock_frequency_mhz": df["clock_frequency_mhz"],
                "voltage_v": df["voltage_v"],
                "hashrate_th": df["hashrate_th"],
                "temperature_c": df["temperature_c"],
                "power_w": df["power_w"],
                "ambient_temperature_c": df["ambient_temperature_c"],
                "operating_mode": "Normal",
                "failure_type": "none",
                "is_pre_failure": False,
            })
            self.db.con.execute("INSERT INTO telemetry SELECT * FROM insert_df")
            self._db_buffer.clear()
        except Exception as e:
            print(f"  DB flush error: {e}")
            self._db_buffer.clear()

    def get_db_count(self) -> int:
        """How many rows have been persisted to DuckDB."""
        if self.db:
            try:
                return self.db.count()
            except Exception:
                return 0
        return 0

    def get_detailed_scores(self, miner_id: str) -> Dict:
        """Get per-model score breakdown for a miner."""
        return self._last_scores.get(miner_id, {
            "xgb_score": 0.0, "lstm_score": 0.0, "combined": 0.0, "consecutive": 0,
        })

    def get_feature_contributions(self, miner_id: str, top_n: int = 5) -> List[tuple]:
        """Get top N features by absolute value for a miner."""
        buf = self.buffers.get(miner_id)
        if buf is None:
            return []
        features = buf.compute_features()
        if features is None:
            return []

        # Only include features the model uses
        relevant = {k: v for k, v in features.items() if k in self.xgb_features}
        sorted_feats = sorted(relevant.items(), key=lambda x: abs(x[1]), reverse=True)
        return sorted_feats[:top_n]

    def close(self):
        if self.db:
            self.db.close()
