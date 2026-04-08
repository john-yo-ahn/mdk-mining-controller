"""
LSTM-Autoencoder for unsupervised anomaly detection.
Trained on healthy-only data. Reconstruction error = anomaly score.
"""

from pathlib import Path
from typing import List, Optional, Tuple

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


class AnomalyDetector:
    """High-level wrapper for LSTM-Autoencoder anomaly detection."""

    DEFAULT_FEATURES = [
        "clock_frequency_mhz", "voltage_v", "hashrate_th",
        "temperature_c", "power_w", "ambient_temperature_c",
    ]

    def __init__(
        self,
        input_dim: int = 6,
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
        # feature_names_ is the ordered column list used to build sequences.
        self.feature_names_: Optional[List[str]] = None
        self.feature_mean_: Optional[np.ndarray] = None   # shape (n_features,)
        self.feature_std_: Optional[np.ndarray] = None    # shape (n_features,)

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

    def fit_scaler(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> None:
        """
        Fit GLOBAL per-feature mean/std on a healthy DataFrame.

        Call this once on the training data BEFORE prepare_sequences. The
        stats are then persisted on the model and reused for every later
        prepare_sequences call — training, validation, and inference — so
        reconstruction errors are comparable across all of them.

        The previous implementation recomputed mean/std per-miner per-call,
        meaning a model trained on 90 days of stable data would see very
        different scales at inference time (where only a short rolling
        buffer is available) and the threshold became meaningless.
        """
        if feature_columns is None:
            feature_columns = [c for c in self.DEFAULT_FEATURES if c in df.columns]
        self.feature_names_ = list(feature_columns)
        values = df[feature_columns].to_numpy(dtype=np.float32)
        self.feature_mean_ = values.mean(axis=0)
        std = values.std(axis=0)
        std[std == 0] = 1.0  # avoid div-by-zero for constant columns
        self.feature_std_ = std
        print(
            f"  Fitted LSTM-AE scaler on {len(values):,} rows × "
            f"{len(feature_columns)} features"
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
        Applies the persistent scaler fitted by fit_scaler(). If the
        scaler has not been fitted yet, one is fitted here on this df
        as a convenience — but callers should prefer explicit fit_scaler
        on the training set first.
        Returns shape (n_sequences, seq_len, n_features).
        """
        if feature_columns is None:
            feature_columns = list(
                self.feature_names_
                or [c for c in self.DEFAULT_FEATURES if c in df.columns]
            )

        if self.feature_mean_ is None:
            self.fit_scaler(df, feature_columns)

        if feature_columns != self.feature_names_:
            raise ValueError(
                f"Feature columns {feature_columns} do not match those the "
                f"scaler was fitted on ({self.feature_names_})"
            )

        mean = self.feature_mean_
        std = self.feature_std_
        sequences = []
        for miner_id in df["miner_id"].unique():
            miner_df = df[df["miner_id"] == miner_id].sort_values("timestamp")
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
                    # Persist the scaler alongside the weights so inference
                    # reproduces exactly the same feature scales as training.
                    "scaler": {
                        "feature_names": self.feature_names_,
                        "mean": self.feature_mean_,
                        "std": self.feature_std_,
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
        scaler = data.get("scaler")
        if scaler is not None:
            instance.feature_names_ = scaler.get("feature_names")
            instance.feature_mean_ = scaler.get("mean")
            instance.feature_std_ = scaler.get("std")
        else:
            print(
                "  WARNING: loaded LSTM-AE has no persisted scaler. "
                "Reconstruction errors may be miscalibrated until fit_scaler "
                "is called on a representative healthy sample."
            )
        print(f"Loaded LSTM-AE from {path}")
        return instance
