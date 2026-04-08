"""
Generate per-miner fleet metadata deterministically.
"""

import numpy as np
import pandas as pd
from ..config import SimulationConfig, DEFAULT_MINER_SPECS


def generate_miner_metadata(config: SimulationConfig) -> pd.DataFrame:
    """
    Create metadata for each miner: ID, model, container, position, install date.
    """
    rng = np.random.default_rng(config.random_seed)
    model_keys = list(DEFAULT_MINER_SPECS.keys())
    container_size = getattr(config, "container_size", 20)
    n_containers = max(1, (config.n_miners + container_size - 1) // container_size)
    containers = [f"Container-{chr(65 + i)}" for i in range(n_containers)]

    records = []
    for i in range(config.n_miners):
        container_idx = i // container_size
        records.append({
            "miner_id": f"MNR-{i+1:03d}",
            "model": model_keys[i % len(model_keys)],
            "container_id": containers[container_idx],
            "position": (i % container_size) + 1,
            "install_date": pd.Timestamp("2025-06-01") + pd.Timedelta(days=int(rng.integers(0, 180))),
            "firmware_version": f"1.{rng.integers(0, 5)}.{rng.integers(0, 10)}",
        })

    return pd.DataFrame(records)
