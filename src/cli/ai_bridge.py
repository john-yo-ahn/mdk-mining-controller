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

# Buffer must cover the longest rolling window used at training time
# (10080 minutes = 7 days). Previously this was 120 (2 hours), which
# meant the 7-day rolling features used by the trained XGBoost model
# could never populate and were silently zero-filled at inference,
# producing a completely different feature distribution from training.
BUFFER_SIZE = 10080  # 7 days at 1-minute sampling
LSTM_SEQ_LEN = 60    # must match AnomalyDetector config at training


class MinerBuffer:
    """
    Rolling ring buffer of raw telemetry for one miner.

    On each tick the caller pushes a new minute of telemetry. The buffer
    stores the last BUFFER_SIZE minutes (7 days). compute_features()
    reconstructs a miniature DataFrame from the buffer contents and
    runs it through the EXACT same build_feature_matrix function the
    batch pipeline uses — guaranteeing training-inference feature
    parity for all non-cross-miner features.

    Cross-miner features (container rank, neighbor deltas) cannot be
    computed from a single miner's buffer in isolation. They are
    assembled by AIBridge.predict() which has access to all miners'
    buffers simultaneously — or, when running single-miner predict,
    they default to 0 as a documented approximation.
    """

    __slots__ = [
        "miner_id", "model", "container_id", "position", "nameplate_hashrate",
        "clock_frequency_mhz", "voltage_v", "hashrate_th",
        "temperature_c", "power_w", "ambient_temperature_c",
        "_size", "_count",
        # LSTM scaler, set by AIBridge.load_models after the LSTM loads.
        # These are the persistent global mean/std saved with the trained
        # model, not a per-buffer local normalization.
        "lstm_scaler_mean", "lstm_scaler_std",
    ]

    def __init__(
        self,
        miner_id: str,
        model: str,
        nameplate_hashrate: float,
        size: int = BUFFER_SIZE,
        container_id: str = "live",
        position: int = 0,
    ):
        self.miner_id = miner_id
        self.model = model
        self.container_id = container_id
        self.position = position
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

        # Populated later by AIBridge.load_models when the LSTM loads.
        self.lstm_scaler_mean = None
        self.lstm_scaler_std = None

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

    def to_dataframe(self) -> pd.DataFrame:
        """
        Reconstruct a DataFrame from the buffer contents, shaped exactly
        like a slice of the batch telemetry table. This is what the
        feature builder expects.
        """
        n = self.length
        freq = self._ordered(self.clock_frequency_mhz)
        volt = self._ordered(self.voltage_v)
        hr = self._ordered(self.hashrate_th)
        temp = self._ordered(self.temperature_c)
        pwr = self._ordered(self.power_w)
        amb = self._ordered(self.ambient_temperature_c)

        # Fake timestamps — the feature builder uses them only for
        # sort-ordering within a miner, and the rows are already in
        # chronological order from _ordered().
        ts = pd.date_range(
            start="2026-01-01",
            periods=n,
            freq="1min",
        )
        return pd.DataFrame({
            "timestamp": ts,
            "miner_id": self.miner_id,
            "model": self.model,
            "container_id": self.container_id,
            "position": self.position,
            "clock_frequency_mhz": freq,
            "voltage_v": volt,
            "hashrate_th": hr,
            "temperature_c": temp,
            "power_w": pwr,
            "ambient_temperature_c": amb,
            "operating_mode": "Normal",
            "failure_type": "none",
            "is_pre_failure": False,
            "degradation_phase": "healthy",
            "hashrate_nameplate_th": self.nameplate_hashrate,
            # Synthetic-generator metadata columns the batch feature
            # builder passes through. In live deployment these would
            # come from real setpoints; here we default to the current
            # observed values as a reasonable approximation.
            "container_supply_temp_c": amb,
            "freq_setpoint_mhz": freq,
            "voltage_setpoint_v": volt,
        })

    def export_lstm_sequence(self, seq_len: int = LSTM_SEQ_LEN) -> Optional[np.ndarray]:
        """
        Export the last seq_len readings as a normalized numpy array
        for LSTM-AE inference. Returns shape (1, seq_len, 9) or None
        if the buffer doesn't have enough data yet.

        Uses the PERSISTENT scaler (lstm_scaler_mean/std, populated by
        AIBridge at load time from the saved model) so reconstruction
        errors are directly comparable with the validation-set
        threshold the model was calibrated against. If the scaler
        hasn't been propagated yet, falls back to per-buffer
        normalization with a warning.

        The 9 features are the 6 raw sensor channels plus 3
        physics-derived columns (efficiency_jth, temp_delta_c,
        power_per_ghz), matching AnomalyDetector.DEFAULT_FEATURES
        order. The derived columns are computed inline here rather
        than stored in the buffer so the buffer schema remains
        purely raw telemetry and the derivation logic has exactly
        one owner (src/models/lstm_autoencoder.py:_ensure_derived_columns
        for the batch path, this method for the live path, with
        identical formulas).
        """
        if self.length < seq_len:
            return None

        freq = self._ordered(self.clock_frequency_mhz)[-seq_len:]
        volt = self._ordered(self.voltage_v)[-seq_len:]
        hr = self._ordered(self.hashrate_th)[-seq_len:]
        temp = self._ordered(self.temperature_c)[-seq_len:]
        pwr = self._ordered(self.power_w)[-seq_len:]
        amb = self._ordered(self.ambient_temperature_c)[-seq_len:]

        # Derived physics — must match _ensure_derived_columns in
        # src/models/lstm_autoencoder.py bit-for-bit. Any change
        # to either implementation must be mirrored in the other or
        # live inference will drift from batch training.
        efficiency_jth = pwr / np.maximum(hr.astype(np.float32), 1.0)
        temp_delta_c = temp.astype(np.float32) - amb.astype(np.float32)
        power_per_ghz = pwr / np.maximum(freq.astype(np.float32), 1.0)

        # Column order must match AnomalyDetector.DEFAULT_FEATURES:
        # [clock_frequency_mhz, voltage_v, hashrate_th, temperature_c,
        #  power_w, ambient_temperature_c,
        #  efficiency_jth, temp_delta_c, power_per_ghz]
        raw = np.stack(
            [freq, volt, hr, temp, pwr, amb,
             efficiency_jth, temp_delta_c, power_per_ghz],
            axis=1,
        )  # (seq_len, 9)

        if self.lstm_scaler_mean is not None and self.lstm_scaler_std is not None:
            normalized = (raw - self.lstm_scaler_mean) / self.lstm_scaler_std
        else:
            # Fallback only. Should not happen if load_models ran first.
            mean = raw.mean(axis=0)
            std = raw.std(axis=0)
            std[std == 0] = 1
            normalized = (raw - mean) / std

        return normalized.reshape(1, seq_len, 9).astype(np.float32)

    # Minimum buffer length before compute_features returns a result.
    # Needs to cover at least the 6-hour rolling window used by the
    # degradation slope + correlations. Cross-timescale features that
    # need more history (1d, 7d) will return less-informative values
    # until the buffer fills — that's acceptable during warmup.
    MIN_BUFFER_FOR_FEATURES = 360  # 6 hours

    def compute_features(self) -> Optional[Dict[str, float]]:
        """
        Compute the full 152-feature vector the trained XGBoost model
        expects by delegating to the batch feature builder.

        Returns None if the buffer has less than MIN_BUFFER_FOR_FEATURES
        rows — the caller should display "AI warming up" during that
        period and defer predictions.

        Parity contract: for any minute of telemetry, this returns
        values identical (to within float rounding) to what
        build_feature_matrix produces on a full DataFrame containing
        the same history. This is the key fix behind F1 in
        REMAINING_FIXES.md — previously ~80 of 152 features were
        silently missing at inference time.

        Cross-miner features (container rank, neighbor deltas) are
        computed from a single-miner DataFrame, so they degrade to
        defaults. AIBridge.predict handles this by collecting all
        miner buffers into one DataFrame for full cross-miner
        feature computation.
        """
        n = self.length
        if n < self.MIN_BUFFER_FOR_FEATURES:
            return None

        # Import here to avoid circular-import during module load
        from ..pipeline.preprocessing import preprocess_pipeline
        from ..pipeline.features import build_feature_matrix
        from ..kpi.true_efficiency import compute_all_te_variants

        # Reconstruct a single-miner DataFrame shaped like the batch
        # telemetry table and run it through the full pipeline. This
        # is the full pipeline the batch trainer uses: preprocess →
        # TE variants → feature builder. Parity-tested via
        # scripts/test_live_feature_parity.py.
        df = self.to_dataframe()
        df = preprocess_pipeline(df, verbose=False)
        df = compute_all_te_variants(df)
        feat_df = build_feature_matrix(df, drop_warmup=False, verbose=False)
        if len(feat_df) == 0:
            return None
        return feat_df.iloc[-1].to_dict()


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
        # Session-level error gate: we log streaming inference failures
        # once, then stay quiet. Not silent any more — just not spammy.
        self._xgb_error_logged = False
        self._lstm_error_logged = False

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
        """
        Load trained models from disk. Returns True if XGBoost loaded.

        After Apr 8 F1 fix: the live inference path is now architecturally
        equivalent to the batch pipeline for all non-cross-miner features.
        MinerBuffer holds 7 days of telemetry, compute_features() delegates
        to build_feature_matrix() for the full 152-feature vector, and the
        LSTM path uses the persistent scaler from the saved model with
        the correct seq_len=60.

        Remaining approximation: cross-miner features (container rank,
        neighbor deltas) are still computed from a single-miner buffer
        in isolation when running per-miner predict(). For full cross-
        miner feature accuracy, the simulator would need to pass all
        miner buffers to a bulk predict method — not required for this
        prototype.
        """
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
                # Propagate the per-hardware-model scaler to any
                # already-registered buffers so export_lstm_sequence
                # uses the correct scale for each miner's hardware
                # family. Schema v2: AnomalyDetector.lookup_scaler()
                # returns the right (mean, std) pair for the buffer's
                # model, falling back to the global scaler for unknown
                # families.
                if (
                    self.lstm_model.feature_scalers_
                    or self.lstm_model.global_fallback_mean_ is not None
                ):
                    for buf in self.buffers.values():
                        mean, std = self.lstm_model.lookup_scaler(buf.model)
                        buf.lstm_scaler_mean = mean
                        buf.lstm_scaler_std = std
            except Exception as e:
                print(f"  Failed to load LSTM-AE: {e}")

        self._models_loaded = self.xgb_model is not None
        if self._models_loaded:
            print(
                "  NOTE: live inference uses the full batch feature "
                "builder; cross-miner features remain approximate."
            )
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
            buf = MinerBuffer(miner_id, model, nameplate_hashrate)
            # If the LSTM model is already loaded, propagate its
            # per-hardware-model scaler so this new buffer normalizes
            # sequences the same way the model was trained.
            # lookup_scaler accepts either "Antminer S21 Pro" or "Pro"
            # and will fall back to the global scaler for unknown
            # families.
            if (
                self.lstm_model is not None
                and (
                    self.lstm_model.feature_scalers_
                    or self.lstm_model.global_fallback_mean_ is not None
                )
            ):
                mean, std = self.lstm_model.lookup_scaler(model)
                buf.lstm_scaler_mean = mean
                buf.lstm_scaler_std = std
            self.buffers[miner_id] = buf

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

        # XGBoost. After the F1 fix, compute_features returns the full
        # 152-feature vector that matches training. Missing columns
        # (cross-miner defaults) are still filled with 0.0, but the
        # set of zero-filled columns is now small and documented.
        try:
            feature_vector = np.array([[features.get(f, 0.0) for f in self.xgb_features]])
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            xgb_score = float(self.xgb_model.predict_proba(feature_vector)[0, 1])
        except Exception as e:
            if not self._xgb_error_logged:
                print(f"  WARN: XGBoost live inference failed: {e}")
                self._xgb_error_logged = True

        # LSTM-AE. seq_len=60 matches the AnomalyDetector the model was
        # trained with. The persistent scaler (feature_mean_/std_) was
        # propagated into MinerBuffer by load_models so normalization
        # is now consistent with training.
        if self.lstm_model is not None and self.lstm_model.model_ is not None:
            try:
                seq = buf.export_lstm_sequence(seq_len=LSTM_SEQ_LEN)
                if seq is not None:
                    error = self.lstm_model.compute_reconstruction_error(seq)
                    if len(error) > 0 and self.lstm_model.threshold_ and self.lstm_model.threshold_ > 0:
                        lstm_score = min(1.0, float(error[0]) / (self.lstm_model.threshold_ * 3))
            except Exception as e:
                if not self._lstm_error_logged:
                    print(f"  WARN: LSTM live inference failed: {e}")
                    self._lstm_error_logged = True

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
