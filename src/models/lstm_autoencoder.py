"""
LSTM-Autoencoder for unsupervised anomaly detection.
Trained on healthy-only data. Reconstruction error = anomaly score.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import MODELS_DIR, SENSOR_COLUMNS
from .evaluation import select_anomaly_threshold, compute_classification_metrics

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def filter_alive_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop telemetry rows where the miner is effectively offline.

    A row counts as offline if ``hashrate_th < 1.0`` OR
    ``voltage_v < 0.05``. Those rows are neither "healthy" nor
    "failing" in the sense the autoencoder is supposed to learn —
    they represent shutdown, idle, or post-failure states where
    every telemetry channel has collapsed to near-zero.

    We drop them from the healthy training set because they would
    pull the global healthy distribution toward zero in every
    feature (a fleet spending ~2% of its time in maintenance
    shutdowns would teach the AE that "near-zero everywhere" is a
    normal healthy pattern, which defeats the point). And on the
    evaluation side, shutdown sequences in the *failure* test set
    are trivially easy to reconstruct — a sequence of zeros feeds
    back as near-zero reconstruction error — which is why the
    original LSTM looked like it was inverted (failures
    reconstructing better than healthy). The right metric
    concerns only failure sequences where the miner is still
    alive and is producing telemetry that a useful detector
    should recognize as anomalous; those are what we call
    "alive failures" downstream.

    This helper is called on the healthy training / validation /
    test splits before prepare_sequences. Failure sequences are
    NOT filtered by this helper; callers that want alive-only
    failure slices should apply the same mask explicitly so the
    distinction is visible in code.
    """
    mask = (df["hashrate_th"] > 1.0) & (df["voltage_v"] > 0.05)
    dropped = int((~mask).sum())
    if dropped > 0:
        pct = 100.0 * dropped / max(len(df), 1)
        print(
            f"  filter_alive_rows: dropped {dropped:,} offline rows "
            f"({pct:.2f}%) from {len(df):,}"
        )
    return df[mask].reset_index(drop=True)


if HAS_TORCH:
    class LSTMEncoder(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 32,
                     n_layers: int = 2, dropout: float = 0.1):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers,
                                batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_dim, latent_dim)

        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.fc(h[-1])

    class LSTMDecoder(nn.Module):
        def __init__(self, latent_dim: int = 32, hidden_dim: int = 64,
                     output_dim: int = 6, n_layers: int = 2,
                     seq_len: int = 60, dropout: float = 0.1):
            super().__init__()
            self.seq_len = seq_len
            self.fc = nn.Linear(latent_dim, hidden_dim)
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, n_layers,
                                batch_first=True, dropout=dropout)
            self.output = nn.Linear(hidden_dim, output_dim)

        def forward(self, z):
            h = self.fc(z).unsqueeze(1).repeat(1, self.seq_len, 1)
            out, _ = self.lstm(h)
            return self.output(out)

    class LSTMAutoencoder(nn.Module):
        def __init__(self, input_dim: int = 6, hidden_dim: int = 64,
                     latent_dim: int = 32, n_layers: int = 2,
                     seq_len: int = 60, dropout: float = 0.1):
            super().__init__()
            self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim, n_layers, dropout)
            self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim, n_layers, seq_len, dropout)

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)


def _ensure_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the three physics-derived columns the LSTM-AE expects as
    part of its 9-feature input vector: ``efficiency_jth``,
    ``temp_delta_c``, ``power_per_ghz``. Idempotent — if a column
    already exists (because the feature cache already carries it)
    it is left alone. Returns a copy only if at least one column
    was added.

    Why these three:
      * efficiency_jth (power_w / hashrate_th) — the primary
        degradation signal. J/TH drift is what XGBoost's top
        features all measure, and it catches slow-building PSU
        and cooling failures well before the raw sensors show a
        discrete alarm.
      * temp_delta_c (temperature_c - ambient_temperature_c) —
        decouples chip self-heating from ambient-driven swings,
        which matters especially for coolant-restriction failures
        where the chip runs hot even though ambient is cool.
      * power_per_ghz (power_w / clock_frequency_mhz) — a per-chip
        work proxy that picks up firmware-oscillation / clock-
        glitch failures that leave hashrate intact but distort
        the power draw pattern.
    """
    out = df
    if "efficiency_jth" not in out.columns:
        out = out.copy()
        out["efficiency_jth"] = (
            out["power_w"].astype(np.float32)
            / np.maximum(out["hashrate_th"].astype(np.float32), 1.0)
        )
    if "temp_delta_c" not in out.columns:
        if out is df:
            out = out.copy()
        out["temp_delta_c"] = (
            out["temperature_c"].astype(np.float32)
            - out["ambient_temperature_c"].astype(np.float32)
        )
    if "power_per_ghz" not in out.columns:
        if out is df:
            out = out.copy()
        out["power_per_ghz"] = (
            out["power_w"].astype(np.float32)
            / np.maximum(out["clock_frequency_mhz"].astype(np.float32), 1.0)
        )
    return out


class AnomalyDetector:
    """High-level wrapper for LSTM-Autoencoder anomaly detection."""

    DEFAULT_FEATURES = [
        # Raw sensor channels
        "clock_frequency_mhz", "voltage_v", "hashrate_th",
        "temperature_c", "power_w", "ambient_temperature_c",
        # Physics-derived (computed inline by _ensure_derived_columns
        # so the feature cache doesn't need a schema bump).
        "efficiency_jth",
        "temp_delta_c",
        "power_per_ghz",
    ]

    def __init__(
        self,
        input_dim: int = 9,
        seq_len: int = 60,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        n_layers: int = 2,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        n_epochs: int = 50,
        device: str = "auto",
        early_stopping_patience: int = 5,
    ):
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.early_stopping_patience = early_stopping_patience
        self.threshold_ = None
        self.model_ = None
        self.train_losses_ = []
        self.val_losses_ = []
        # Persistent per-feature normalization. Fit once on training
        # data; used identically at training, validation, and inference
        # so reconstruction errors are comparable across calls.
        #
        # Scaler layout (schema v2): one (mean, std) pair per hardware
        # model, looked up by the `model` column on the input DataFrame.
        # Each hardware family (Antminer S21 Pro, Whatsminer M56S, etc.)
        # has a very different healthy operating envelope — mixing them
        # into a single global scaler smears the manifold and prevents
        # the autoencoder from tightening on any of them. A per-model
        # scaler tightens each family independently.
        #
        # global_fallback_* is used when inference encounters a hardware
        # model that did not appear at training time, with a clearly
        # logged warning.
        self.feature_names_: Optional[List[str]] = None
        self.feature_scalers_: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.global_fallback_mean_: Optional[np.ndarray] = None
        self.global_fallback_std_: Optional[np.ndarray] = None

        if not HAS_TORCH:
            print("WARNING: PyTorch not installed. LSTM-AE will not be available.")
            print("Install with: uv add torch")
            self.device = "cpu"
            return

        if device == "auto":
            # Prefer CUDA, then Apple Silicon MPS, then CPU.
            # MPS gives a 3-5x speedup on this LSTM on M-series Macs.
            if torch.cuda.is_available():
                self.device = "cuda"
            elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
                and torch.backends.mps.is_built()
            ):
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model_ = LSTMAutoencoder(
            input_dim, hidden_dim, latent_dim, n_layers, seq_len,
        ).to(self.device)

    # Backward-compat properties: the v1 scaler exposed `feature_mean_`
    # and `feature_std_` as flat numpy arrays. Under schema v2 those are
    # replaced by a per-hardware-model dict. Any code still reading the
    # old attributes must be updated to call lookup_scaler(model_name).
    # Raise loudly rather than returning a silently-wrong value.
    @property
    def feature_mean_(self) -> np.ndarray:
        raise AttributeError(
            "AnomalyDetector.feature_mean_ was removed in scaler schema v2. "
            "Use lookup_scaler(model_name)[0] for a per-hardware scaler, "
            "or global_fallback_mean_ for the unknown-model fallback."
        )

    @property
    def feature_std_(self) -> np.ndarray:
        raise AttributeError(
            "AnomalyDetector.feature_std_ was removed in scaler schema v2. "
            "Use lookup_scaler(model_name)[1] for a per-hardware scaler, "
            "or global_fallback_std_ for the unknown-model fallback."
        )

    @staticmethod
    def _normalize_model_name(model_name: Optional[str]) -> Optional[str]:
        """
        The synthetic generator writes the hardware family as the
        trailing token of the full spec name: ``"Antminer S21 Pro"``
        becomes ``"Pro"`` in the ``model`` column of the DataFrame.
        Callers (e.g. AIBridge.register_miner) may pass either the
        full spec name or the short token; normalize to the short
        token here so scaler lookups succeed either way.
        """
        if model_name is None:
            return None
        parts = str(model_name).split()
        return parts[-1] if len(parts) > 1 else str(model_name)

    def lookup_scaler(
        self, model_name: Optional[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (mean, std) for the given hardware model, falling back
        to the global scaler if the model is unknown or not provided.
        Accepts either the short token (``"Pro"``) or the full spec
        name (``"Antminer S21 Pro"``) — both map to the same entry.
        One lookup per miner (not per timestep) is the expected call
        frequency, so the dict lookup is fine as-is.
        """
        key = self._normalize_model_name(model_name)
        if key is not None and key in self.feature_scalers_:
            return self.feature_scalers_[key]
        if self.global_fallback_mean_ is None or self.global_fallback_std_ is None:
            raise RuntimeError(
                "LSTM scaler has not been fitted yet. Call fit_scaler() "
                "on training data first."
            )
        return self.global_fallback_mean_, self.global_fallback_std_

    # Minimum distinct miners per hardware model required to fit a
    # dedicated per-model scaler. Below this threshold the fleet for
    # that model is too small for the per-model mean/std to be stable
    # (one failing miner could dominate), so we fall back to the
    # global scaler for that family.
    MIN_MINERS_PER_MODEL_SCALER = 3

    def fit_scaler(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> None:
        """
        Fit per-hardware-model and global fallback scalers on a healthy
        DataFrame.

        Call this once on the training data BEFORE prepare_sequences. The
        stats are then persisted on the model and reused at training,
        validation, and inference time so reconstruction errors are
        comparable across all of them.

        Schema v2 change: instead of a single global mean/std pair the
        scaler is a dict keyed on hardware model (the `model` column
        on the input DataFrame). Each family's mean and std are
        computed separately, which tightens the healthy manifold for
        each family and unblocks useful reconstruction-error signals.
        Falls back to a global mean/std for any hardware model that
        has fewer than MIN_MINERS_PER_MODEL_SCALER distinct miners in
        training (the per-model stats are too noisy below that cutoff).

        The previous schema v1 used a single flat mean/std on all
        training rows, which in production smeared four hardware
        families together and gave the AE nothing to overfit cleanly.
        """
        df = _ensure_derived_columns(df)
        if feature_columns is None:
            feature_columns = [c for c in self.DEFAULT_FEATURES if c in df.columns]
        self.feature_names_ = list(feature_columns)

        all_values = df[feature_columns].to_numpy(dtype=np.float32)
        gf_mean = all_values.mean(axis=0)
        gf_std = all_values.std(axis=0)
        gf_std[gf_std == 0] = 1.0
        self.global_fallback_mean_ = gf_mean
        self.global_fallback_std_ = gf_std

        if "model" not in df.columns:
            raise ValueError(
                "fit_scaler expects a 'model' column on the training "
                "DataFrame so it can fit per-hardware-model scalers. "
                "Got columns: " + ", ".join(df.columns[:10]) + "..."
            )

        self.feature_scalers_ = {}
        model_miner_counts = df.groupby("model")["miner_id"].nunique()
        total_rows = len(all_values)
        print(
            f"  Fitting LSTM-AE scaler (schema v2) on "
            f"{total_rows:,} rows × {len(feature_columns)} features"
        )
        for model_name, grp in df.groupby("model"):
            n_miners = int(model_miner_counts[model_name])
            if n_miners < self.MIN_MINERS_PER_MODEL_SCALER:
                print(
                    f"    {model_name!s}: {len(grp):,} rows, {n_miners} miners "
                    f"— below min {self.MIN_MINERS_PER_MODEL_SCALER}, using "
                    f"global fallback"
                )
                continue
            v = grp[feature_columns].to_numpy(dtype=np.float32)
            m = v.mean(axis=0)
            s = v.std(axis=0)
            s[s == 0] = 1.0
            self.feature_scalers_[str(model_name)] = (m, s)
            print(
                f"    {model_name!s}: {len(grp):,} rows, {n_miners} miners "
                f"— per-model scaler fitted"
            )
        print(
            f"  Global fallback: mean/std computed on all "
            f"{total_rows:,} training rows"
        )

    def prepare_sequences(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        stride: int = 1,
    ) -> np.ndarray:
        """
        Convert DataFrame to sliding-window sequences.
        Groups by miner_id so no window ever spans two devices.
        Applies the per-hardware-model scaler fitted by fit_scaler():
        each miner's sequences are normalized using the mean/std of
        the hardware family that miner belongs to, falling back to the
        global scaler for unknown families.
        If the scaler has not been fitted yet, one is fitted here on
        this df as a convenience — but callers should prefer explicit
        fit_scaler on the training set first.
        Returns shape (n_sequences, seq_len, n_features).
        """
        df = _ensure_derived_columns(df)
        if feature_columns is None:
            feature_columns = list(
                self.feature_names_
                or [c for c in self.DEFAULT_FEATURES if c in df.columns]
            )

        if not self.feature_scalers_ and self.global_fallback_mean_ is None:
            self.fit_scaler(df, feature_columns)

        if feature_columns != self.feature_names_:
            raise ValueError(
                f"Feature columns {feature_columns} do not match those the "
                f"scaler was fitted on ({self.feature_names_})"
            )

        has_model_col = "model" in df.columns
        if not has_model_col:
            print(
                "  prepare_sequences: DataFrame has no 'model' column, "
                "falling back to global scaler for all miners"
            )

        sequences = []
        for miner_id in df["miner_id"].unique():
            miner_df = df[df["miner_id"] == miner_id].sort_values("timestamp")
            if has_model_col:
                model_name = str(miner_df["model"].iloc[0])
            else:
                model_name = None
            mean, std = self.lookup_scaler(model_name)
            values = miner_df[feature_columns].to_numpy(dtype=np.float32)
            values = (values - mean) / std
            for i in range(0, len(values) - self.seq_len, stride):
                sequences.append(values[i:i + self.seq_len])

        if not sequences:
            return np.empty((0, self.seq_len, len(feature_columns)), dtype=np.float32)
        return np.asarray(sequences, dtype=np.float32)

    def fit(
        self,
        X_healthy: np.ndarray,
        X_val: Optional[np.ndarray] = None,
    ) -> List[float]:
        """
        Train on healthy-only sequences using MSE loss.

        If X_val is provided, tracks validation loss every epoch and early-
        stops when it fails to improve for `self.early_stopping_patience`
        epochs in a row. The best weights (lowest val_loss) are restored at
        the end, so a late overfitting burst never reaches the saved model.
        """
        if not HAS_TORCH:
            print("PyTorch not available. Skipping LSTM training.")
            return []

        dataset = TensorDataset(torch.FloatTensor(X_healthy))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        self.train_losses_ = []
        self.val_losses_ = []

        best_val = float("inf")
        best_state = None
        epochs_since_improve = 0
        patience = max(1, int(self.early_stopping_patience))

        for epoch in range(self.n_epochs):
            self.model_.train()
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                output = self.model_(batch)
                loss = criterion(output, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch)

            avg_loss = epoch_loss / max(len(X_healthy), 1)
            self.train_losses_.append(avg_loss)

            val_loss = float("nan")
            if X_val is not None and len(X_val) > 0:
                val_loss = float(self.compute_reconstruction_error(X_val).mean())
                self.val_losses_.append(val_loss)

                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model_.state_dict().items()
                    }
                    epochs_since_improve = 0
                else:
                    epochs_since_improve += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                val_str = f" val_loss={val_loss:.6f}" if not np.isnan(val_loss) else ""
                print(
                    f"  Epoch {epoch+1}/{self.n_epochs}: loss={avg_loss:.6f}{val_str}"
                )

            if X_val is not None and epochs_since_improve >= patience:
                print(
                    f"  Early stop at epoch {epoch+1}: no val improvement "
                    f"for {patience} epochs (best val_loss={best_val:.6f})"
                )
                break

        if best_state is not None:
            # Rebuild the model from scratch on CPU, load the best
            # state into it, then move it back to the target device.
            # This sidesteps an observed PyTorch MPS issue where
            # load_state_dict on a live MPS model left subtle graph
            # state behind that caused save/reload to drift. Building
            # a fresh nn.Module guarantees a clean parameter layout.
            fresh = LSTMAutoencoder(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                latent_dim=self.latent_dim,
                n_layers=self.n_layers,
                seq_len=self.seq_len,
            )
            fresh.load_state_dict(best_state)
            self.model_ = fresh.to(self.device)
            self.model_.eval()
            print(f"  Restored best weights (val_loss={best_val:.6f})")

        return self.train_losses_

    def compute_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Per-sequence mean squared reconstruction error.

        Inference runs on CPU regardless of the training device. On Apple
        Silicon MPS we observed a kernel bug where specific combinations
        of batch size and this LSTM architecture return numerically wrong
        outputs (sometimes off by an order of magnitude) while the same
        model on CPU or with a different batch size is correct. The bug
        is silent — no error, no warning, just wrong numbers — and it
        used to make training-time separation metrics look great while
        reload metrics looked terrible on the identical weights. Forcing
        CPU here eliminates the risk entirely. The model is small enough
        (~60k params) that CPU inference is not a bottleneck for the
        dataset sizes we evaluate on.
        """
        if not HAS_TORCH:
            return np.zeros(len(X))

        # Saturate the CPU for the forward pass. torch's default is often
        # 1 thread when launched under certain launchers (uv in particular
        # has been observed to ship an OMP_NUM_THREADS=1 env to subprocs),
        # which leaves ~80% of the cores idle on this box. Pinning to the
        # physical core count roughly halves the wall-clock of the bigger
        # compute calls used by the experiment harness.
        import os
        torch.set_num_threads(max(1, os.cpu_count() or 1))

        self.model_.eval()
        # Snapshot then move to CPU for inference; restore after.
        was_on = next(self.model_.parameters()).device
        needs_restore = str(was_on) != "cpu"
        if needs_restore:
            self.model_ = self.model_.to("cpu")

        errors = []
        # Keep a safe CPU batch size that is not known to hit the MPS bug.
        # On CPU the value only affects throughput, not correctness.
        cpu_bs = min(max(self.batch_size, 1), 256)
        dataset = TensorDataset(torch.FloatTensor(X))
        loader = DataLoader(dataset, batch_size=cpu_bs, shuffle=False)

        try:
            with torch.no_grad():
                for (batch,) in loader:
                    output = self.model_(batch)
                    mse = ((output - batch) ** 2).mean(dim=(1, 2))
                    errors.extend(mse.numpy())
        finally:
            if needs_restore:
                self.model_ = self.model_.to(was_on)

        return np.array(errors)

    def set_threshold(self, errors_healthy: np.ndarray, percentile: float = 95.0) -> float:
        """Set anomaly threshold from healthy validation errors."""
        self.threshold_ = select_anomaly_threshold(errors_healthy, percentile)
        print(f"Anomaly threshold set at {percentile}th percentile: {self.threshold_:.6f}")
        return self.threshold_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary anomaly labels: 1=anomaly, 0=normal."""
        errors = self.compute_reconstruction_error(X)
        if self.threshold_ is None:
            raise ValueError("Threshold not set. Call set_threshold() first.")
        return (errors > self.threshold_).astype(int)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Raw anomaly scores (reconstruction errors)."""
        return self.compute_reconstruction_error(X)

    def save(self, path: Optional[Path] = None) -> Path:
        if path is None:
            path = MODELS_DIR / "lstm_ae.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        if HAS_TORCH and self.model_ is not None:
            # Move to CPU before saving. This sidesteps a PyTorch MPS
            # quirk where state_dict pulled off a long-trained MPS
            # model can round-trip bit-exactly (weights equal, forward
            # on a single input equal) but still produce subtly
            # different reconstruction-error distributions when
            # reloaded into a fresh MPS context. Copying to CPU first
            # forces a synchronous parameter materialization and
            # eliminates the drift.
            self.model_.eval()
            was_on = self.device
            cpu_model = self.model_.to("cpu")
            try:
                torch.save({
                    "model_state": cpu_model.state_dict(),
                    "threshold": self.threshold_,
                    "config": {
                        "input_dim": self.input_dim,
                        "seq_len": self.seq_len,
                        "hidden_dim": self.hidden_dim,
                        "latent_dim": self.latent_dim,
                        "n_layers": self.n_layers,
                    },
                    # Persist the per-model scalers alongside the weights
                    # so inference reproduces exactly the same feature
                    # scales as training. Schema v2 — one (mean, std) pair
                    # per hardware model + a global fallback, up from the
                    # single-global pair in v1. load() refuses a v1
                    # scaler (no "schema" key) with a clear error.
                    "scaler": {
                        "schema": 2,
                        "feature_names": self.feature_names_,
                        "global_mean": (
                            self.global_fallback_mean_.tolist()
                            if self.global_fallback_mean_ is not None
                            else None
                        ),
                        "global_std": (
                            self.global_fallback_std_.tolist()
                            if self.global_fallback_std_ is not None
                            else None
                        ),
                        "per_model": {
                            name: {"mean": m.tolist(), "std": s.tolist()}
                            for name, (m, s) in self.feature_scalers_.items()
                        },
                    },
                }, path)
            finally:
                # Restore to original device so in-memory usage after
                # save() still runs on MPS/CUDA.
                self.model_ = cpu_model.to(was_on)
        print(f"Saved LSTM-AE to {path}")
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AnomalyDetector":
        if path is None:
            path = MODELS_DIR / "lstm_ae.pt"
        if not HAS_TORCH:
            print("PyTorch not available. Cannot load LSTM model.")
            return cls()
        data = torch.load(path, map_location="cpu", weights_only=False)
        cfg = data["config"]
        instance = cls(**cfg)
        instance.model_.load_state_dict(data["model_state"])
        instance.threshold_ = data["threshold"]
        scaler = data.get("scaler") or {}
        schema = scaler.get("schema", 1)
        if schema == 1:
            raise ValueError(
                f"Model at {path} has a legacy v1 scaler (single global "
                f"mean/std). Retrain with scripts/train_lstm_only.py to "
                f"produce a v2 per-hardware-model scaler. The v1 layout "
                f"is incompatible with the current inference path."
            )
        if schema != 2:
            raise ValueError(
                f"Unknown scaler schema {schema} in {path}. This "
                f"AnomalyDetector only understands schema v2."
            )
        instance.feature_names_ = scaler.get("feature_names")
        gf_mean = scaler.get("global_mean")
        gf_std = scaler.get("global_std")
        instance.global_fallback_mean_ = (
            np.asarray(gf_mean, dtype=np.float32) if gf_mean is not None else None
        )
        instance.global_fallback_std_ = (
            np.asarray(gf_std, dtype=np.float32) if gf_std is not None else None
        )
        per_model = scaler.get("per_model") or {}
        instance.feature_scalers_ = {
            name: (
                np.asarray(entry["mean"], dtype=np.float32),
                np.asarray(entry["std"], dtype=np.float32),
            )
            for name, entry in per_model.items()
        }
        print(
            f"Loaded LSTM-AE from {path} "
            f"(scaler v2: {len(instance.feature_scalers_)} per-model, "
            f"global fallback present)"
        )
        return instance
