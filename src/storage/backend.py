"""
Storage abstraction layer.
DuckDB for development, TimescaleDB for production (future).

Design notes
------------
DuckDB enforces a single writer lock per database file. If a previous run
crashed (or this process is started while another writer is alive), opening
the file fails with an opaque IO error. This module turns that into an
actionable RuntimeError that names the holder PID.

Two ingest modes are supported:

  * mode="replace"  drops the target table and recreates it from the
                    DataFrame's schema. Use this for batch reloads where
                    the schema is the source of truth (run_pipeline).

  * mode="append"   inserts into an existing table, creating it on first
                    write. Use this for live telemetry streams.

A read_only mode is also supported, so dashboards / notebooks can attach
to the same file while a writer is running.
"""

from __future__ import annotations

import atexit
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import duckdb
import pandas as pd

from ..config import BATCH_DB_PATH


# Live-stream telemetry schema. Used by ai_bridge.flush_to_db().
# The batch path uses CREATE TABLE AS SELECT and infers its own schema
# from the DataFrame, so this DDL only matters for the live store.
_LIVE_TELEMETRY_DDL = """
CREATE TABLE IF NOT EXISTS telemetry (
    timestamp             TIMESTAMP,
    miner_id              VARCHAR,
    model                 VARCHAR,
    container_id          VARCHAR,
    position              INTEGER,
    clock_frequency_mhz   DOUBLE,
    voltage_v             DOUBLE,
    hashrate_th           DOUBLE,
    temperature_c         DOUBLE,
    power_w               DOUBLE,
    ambient_temperature_c DOUBLE,
    operating_mode        VARCHAR,
    failure_type          VARCHAR,
    is_pre_failure        BOOLEAN
)
"""

_FEATURES_DDL = """
CREATE TABLE IF NOT EXISTS features (
    timestamp      TIMESTAMP,
    miner_id       VARCHAR,
    is_pre_failure BOOLEAN,
    PRIMARY KEY (timestamp, miner_id)
)
"""


class TelemetryStore(ABC):
    """Abstract base for telemetry storage backends."""

    @abstractmethod
    def ingest(self, df: pd.DataFrame, table: str = "telemetry", mode: str = "replace") -> int: ...

    @abstractmethod
    def query(self, sql: str) -> pd.DataFrame: ...

    @abstractmethod
    def latest(self, miner_id: str, table: str = "telemetry") -> dict: ...

    @abstractmethod
    def count(self, table: str = "telemetry") -> int: ...


def _format_lock_error(db_path: str) -> str:
    """Build a useful error message that names the process holding the lock."""
    import shutil
    import subprocess

    msg = (
        f"DuckDB at {db_path} is locked by another process.\n"
        "DuckDB only allows a single writer at a time."
    )

    if shutil.which("lsof"):
        try:
            out = subprocess.check_output(
                ["lsof", "--", db_path],
                stderr=subprocess.DEVNULL,
                timeout=2,
            ).decode().strip()
            if out:
                msg += f"\n\nHolders (lsof):\n{out}"
        except Exception:
            pass

    msg += (
        "\n\nResolutions:"
        "\n  • Wait for the other process to finish, or kill it."
        "\n  • Re-open with read_only=True if you only need to read."
    )
    return msg


class DuckDBStore(TelemetryStore):
    """
    DuckDB-backed store. Zero infrastructure, reads Parquet natively.
    Ideal for development and the assignment.

    Parameters
    ----------
    db_path : str | Path | None
        Path to the .duckdb file. Defaults to BATCH_DB_PATH from config.
    read_only : bool
        Open in read-only mode. Multiple readers can attach simultaneously
        and they will not block a writer.
    ensure_live_schema : bool
        If True, create the live telemetry table on init. Used by ai_bridge.
        The batch pipeline does NOT need this — its ingest() drops and
        recreates telemetry from the synthetic DataFrame.
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        *,
        read_only: bool = False,
        ensure_live_schema: bool = False,
    ):
        if db_path is None:
            db_path = BATCH_DB_PATH
        self.db_path = str(db_path)
        self.read_only = read_only
        self.con: Optional[duckdb.DuckDBPyConnection] = None

        try:
            self.con = duckdb.connect(self.db_path, read_only=read_only)
        except duckdb.IOException as e:
            if "lock" in str(e).lower() or "being used" in str(e).lower():
                raise RuntimeError(_format_lock_error(self.db_path)) from e
            raise

        if not read_only:
            self.con.execute(_FEATURES_DDL)
            if ensure_live_schema:
                self.con.execute(_LIVE_TELEMETRY_DDL)

        # Make sure the connection is released even if the caller forgets.
        atexit.register(self._safe_close)

    # ── Ingest ────────────────────────────────────────────────────────

    def ingest(
        self,
        df: pd.DataFrame,
        table: str = "telemetry",
        mode: str = "replace",
    ) -> int:
        """
        Insert DataFrame into table.

        mode="replace": drop and recreate table to match df's schema.
        mode="append":  insert into existing table; create from df if missing.
        """
        if mode not in ("replace", "append"):
            raise ValueError(f"mode must be 'replace' or 'append', got {mode!r}")
        if self.read_only:
            raise RuntimeError("Cannot ingest into a read-only DuckDBStore")

        # Register so DuckDB can find `df` regardless of caller frame depth.
        self.con.register("_ingest_df", df)
        try:
            if mode == "replace":
                self.con.execute(f"DROP TABLE IF EXISTS {table}")
                self.con.execute(f"CREATE TABLE {table} AS SELECT * FROM _ingest_df")
            else:  # append
                exists = self.con.execute(
                    "SELECT count(*) FROM information_schema.tables "
                    "WHERE table_schema = 'main' AND table_name = ?",
                    [table],
                ).fetchone()[0]
                if exists == 0:
                    self.con.execute(f"CREATE TABLE {table} AS SELECT * FROM _ingest_df")
                else:
                    self.con.execute(f"INSERT INTO {table} SELECT * FROM _ingest_df")
        finally:
            self.con.unregister("_ingest_df")
        return len(df)

    def ingest_parquet(self, path: str, table: str = "telemetry") -> int:
        """Load directly from Parquet (replace mode)."""
        if self.read_only:
            raise RuntimeError("Cannot ingest into a read-only DuckDBStore")
        self.con.execute(f"DROP TABLE IF EXISTS {table}")
        self.con.execute(
            f"CREATE TABLE {table} AS SELECT * FROM read_parquet(?)",
            [path],
        )
        return self.con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]

    def clear(self, table: str = "telemetry") -> None:
        """Empty a table without dropping its schema. No-op if it doesn't exist."""
        if self.read_only:
            raise RuntimeError("Cannot clear a read-only DuckDBStore")
        exists = self.con.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_name = ?",
            [table],
        ).fetchone()[0]
        if exists:
            self.con.execute(f"DELETE FROM {table}")

    # ── Read ──────────────────────────────────────────────────────────

    def query(self, sql: str) -> pd.DataFrame:
        return self.con.execute(sql).fetchdf()

    def latest(self, miner_id: str, table: str = "telemetry") -> dict:
        result = self.con.execute(
            f"SELECT * FROM {table} WHERE miner_id = ? "
            "ORDER BY timestamp DESC LIMIT 1",
            [miner_id],
        ).fetchdf()
        if len(result) == 0:
            return {}
        return result.iloc[0].to_dict()

    def count(self, table: str = "telemetry") -> int:
        return self.con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]

    def query_miners(self, sql_where: str = "1=1") -> pd.DataFrame:
        return self.query(f"SELECT * FROM telemetry WHERE {sql_where}")

    def fleet_summary(self) -> pd.DataFrame:
        return self.query("""
            SELECT
                miner_id,
                model,
                count(*) as n_readings,
                avg(hashrate_th) as avg_hashrate,
                avg(power_w) as avg_power,
                avg(temperature_c) as avg_temp,
                avg(power_w / NULLIF(hashrate_th, 0)) as avg_jth,
                max(temperature_c) as max_temp,
                sum(CASE WHEN is_pre_failure THEN 1 ELSE 0 END) as pre_failure_count,
                max(failure_type) as failure_type
            FROM telemetry
            WHERE operating_mode != 'Shutdown'
            GROUP BY miner_id, model
            ORDER BY miner_id
        """)

    # ── Lifecycle ─────────────────────────────────────────────────────

    def close(self) -> None:
        self._safe_close()

    def _safe_close(self) -> None:
        """Idempotent close. Safe to call from atexit and __exit__."""
        if self.con is not None:
            try:
                self.con.close()
            except Exception:
                pass
            self.con = None

    def __enter__(self) -> "DuckDBStore":
        return self

    def __exit__(self, *args) -> None:
        self.close()
