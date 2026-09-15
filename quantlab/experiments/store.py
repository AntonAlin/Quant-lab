"""Append-only experiment log.

Two files:
    ~/.quantlab/experiments/runs.parquet          one row per completed run
    ~/.quantlab/experiments/returns/<run_id>.parquet   that run's OOS per-bar returns

The second one exists because PBO needs the return *series* of every trial,
not just its Sharpe. Nothing here ever deletes or overwrites a row; if you want
to start again, delete the directory yourself and feel the guilt.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPERIMENTS_DIR = Path.home() / ".quantlab" / "experiments"


class ExperimentStore:
    def __init__(self, root: Path = EXPERIMENTS_DIR) -> None:
        self.root = Path(root)
        self.runs_path = self.root / "runs.parquet"
        self.returns_dir = self.root / "returns"

    # -- write --------------------------------------------------------------- #
    def log_run(self, record: dict[str, Any], oos_returns: pd.Series) -> str:
        if "run_id" not in record or not record["run_id"]:
            raise ValueError("record needs a run_id.")
        self.root.mkdir(parents=True, exist_ok=True)
        self.returns_dir.mkdir(parents=True, exist_ok=True)
        new = pd.DataFrame([record])
        existing = self.load_runs()
        if not existing.empty and (existing["run_id"] == record["run_id"]).any():
            raise ValueError(f"run_id {record['run_id']} already logged. The log is append-only and ids are unique.")
        combined = pd.concat([existing, new], ignore_index=True) if not existing.empty else new
        combined = _harmonise_dtypes(combined)
        # Write to a temp file then rename so a crash mid-write cannot corrupt the log.
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".parquet")
        os.close(fd)
        try:
            combined.to_parquet(tmp, index=False)
            os.replace(tmp, self.runs_path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        oos_returns.rename("ret").to_frame().to_parquet(self.returns_dir / f"{record['run_id']}.parquet")
        return str(record["run_id"])

    # -- read ---------------------------------------------------------------- #
    def load_runs(self) -> pd.DataFrame:
        if not self.runs_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.runs_path)

    def load_returns(self, run_id: str) -> pd.Series | None:
        p = self.returns_dir / f"{run_id}.parquet"
        if not p.exists():
            return None
        s = pd.read_parquet(p)["ret"]
        s.index = pd.DatetimeIndex(s.index)
        return s

    def returns_matrix(self, run_ids: list[str]) -> pd.DataFrame:
        cols = {}
        for rid in run_ids:
            s = self.load_returns(rid)
            if s is not None:
                cols[rid] = s
        if not cols:
            return pd.DataFrame()
        return pd.DataFrame(cols)

    def trials(self, ticker: str | None = None, interval: str | None = None, session_id: str | None = None) -> pd.DataFrame:
        """Rows that count as 'trials' for the DSR: same instrument, distinct config hash."""
        runs = self.load_runs()
        if runs.empty:
            return runs
        if ticker is not None:
            runs = runs[runs["ticker"] == ticker]
        if interval is not None:
            runs = runs[runs["interval"] == interval]
        if session_id is not None:
            runs = runs[runs["session_id"] == session_id]
        return runs.drop_duplicates("config_hash", keep="last")

    def n_trials(self, ticker: str | None = None, interval: str | None = None, session_id: str | None = None) -> int:
        return int(len(self.trials(ticker, interval, session_id)))

    def trial_sharpes(self, ticker: str | None = None, interval: str | None = None, session_id: str | None = None) -> np.ndarray:
        t = self.trials(ticker, interval, session_id)
        if t.empty or "sharpe_per_bar" not in t:
            return np.array([])
        return t["sharpe_per_bar"].dropna().to_numpy(dtype=float)


def _harmonise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Object columns with mixed None/float upset pyarrow. Coerce the obvious ones."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            try:
                out[c] = pd.to_numeric(out[c])
            except (ValueError, TypeError):
                out[c] = out[c].astype(str).where(out[c].notna(), None)
    return out
