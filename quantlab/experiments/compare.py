"""Cross-run leaderboard, with DSR recomputed against the *current* trial count.

The DSR stored on a run was computed with however many trials existed at the
time. Twenty runs later, that number is stale and flattering. The leaderboard
recomputes every run's DSR against the full current trial set so old runs do
not get to keep their early-bird significance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.backtest.metrics import PBOResult, expected_max_sharpe, pbo_cscv, probabilistic_sharpe
from quantlab.experiments.store import ExperimentStore

LEADERBOARD_COLS = [
    "run_id", "timestamp", "ticker", "interval", "model_family", "label_scheme", "label_horizon", "n_features",
    "wf_mode", "wf_train", "wf_test", "wf_embargo", "sizing", "direction",
    "m_sharpe", "dsr_now", "m_cagr", "m_max_drawdown", "m_calmar", "m_sortino", "m_hit_rate", "m_profit_factor",
    "m_n_trades", "m_turnover_annual", "m_avg_exposure", "oos_accuracy", "oos_logloss", "random_percentile",
    "m_bench_sharpe", "n_trials_at_run", "config_hash", "session_id",
]


def leaderboard(store: ExperimentStore, ticker: str | None = None, interval: str | None = None) -> pd.DataFrame:
    runs = store.load_runs()
    if runs.empty:
        return runs
    if ticker:
        runs = runs[runs["ticker"] == ticker]
    if interval:
        runs = runs[runs["interval"] == interval]
    if runs.empty:
        return runs
    runs = runs.copy()
    trials = runs.drop_duplicates("config_hash", keep="last")
    n = len(trials)
    var = float(trials["sharpe_per_bar"].var(ddof=1)) if n > 1 else 0.0
    sr0 = expected_max_sharpe(n, var)
    # Recompute DSR with today's trial count. Skew/kurt per run are not stored, so
    # we approximate with the normal case (skew 0, kurt 3): slightly generous on
    # fat tails, but consistent across rows, which is what a leaderboard needs.
    runs["dsr_now"] = [
        probabilistic_sharpe(sr, sr0, int(nb), 0.0, 3.0) if np.isfinite(sr) else np.nan
        for sr, nb in zip(runs["sharpe_per_bar"], runs["m_n_bars"])
    ]
    runs["n_trials_now"] = n
    cols = [c for c in LEADERBOARD_COLS if c in runs.columns] + ["n_trials_now"]
    return runs[cols].sort_values("m_sharpe", ascending=False).reset_index(drop=True)


def pbo_across_runs(store: ExperimentStore, run_ids: list[str], n_partitions: int = 10) -> PBOResult:
    M = store.returns_matrix(run_ids)
    if M.shape[1] < 2:
        return PBOResult(np.nan, np.array([]), int(M.shape[1]), n_partitions, 0, np.zeros((0, 2)))
    return pbo_cscv(M, n_partitions=n_partitions)


def parallel_coordinates_frame(lb: pd.DataFrame) -> pd.DataFrame:
    """Numeric parameter columns + OOS Sharpe, ready for a parallel-coordinates plot."""
    if lb.empty:
        return lb
    keep = ["wf_train", "wf_test", "wf_embargo", "label_horizon", "n_features", "m_turnover_annual", "m_avg_exposure", "oos_accuracy", "m_max_drawdown", "m_sharpe"]
    out = lb[[c for c in keep if c in lb.columns]].copy()
    out["model_code"] = pd.factorize(lb["model_family"])[0]
    out["label_code"] = pd.factorize(lb["label_scheme"])[0]
    return out.dropna(subset=["m_sharpe"])
