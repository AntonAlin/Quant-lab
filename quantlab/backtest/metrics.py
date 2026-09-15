"""Performance metrics and the overfitting police: deflated Sharpe, PBO, random-strategy benchmark.

All Sharpe ratios inside the DSR/PSR maths are *per-bar* (non-annualised). The
annualised numbers are for humans; the statistics want the raw ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329


# --------------------------------------------------------------------------- #
# Plain performance
# --------------------------------------------------------------------------- #
def sharpe(returns: pd.Series | np.ndarray, bars_per_year: float, rf: float = 0.0) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return np.nan
    excess = r - rf / bars_per_year
    sd = excess.std(ddof=1)
    if sd == 0:
        return np.nan
    return float(excess.mean() / sd * np.sqrt(bars_per_year))


def sortino(returns: pd.Series | np.ndarray, bars_per_year: float) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return np.nan
    downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    if downside == 0:
        return np.nan
    return float(r.mean() / downside * np.sqrt(bars_per_year))


def max_drawdown(equity: pd.Series) -> tuple[float, int, pd.Timestamp | None, pd.Timestamp | None]:
    """(max drawdown as negative fraction, longest drawdown duration in bars, peak, trough)."""
    eq = equity.astype(float)
    peak = eq.cummax()
    dd = eq / peak - 1.0
    mdd = float(dd.min()) if len(dd) else np.nan
    trough = dd.idxmin() if len(dd) else None
    peak_time = eq.loc[:trough].idxmax() if trough is not None else None
    # Duration: longest run of bars below the previous peak.
    underwater = (dd < 0).to_numpy()
    longest = run = 0
    for u in underwater:
        run = run + 1 if u else 0
        longest = max(longest, run)
    return mdd, int(longest), peak_time, trough


def cagr(equity: pd.Series, bars_per_year: float) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return np.nan
    years = len(equity) / bars_per_year
    total = float(equity.iloc[-1] / equity.iloc[0])
    if total <= 0:
        return -1.0
    return float(total ** (1.0 / years) - 1.0)


def alpha_beta(returns: pd.Series, benchmark: pd.Series, bars_per_year: float) -> tuple[float, float]:
    both = pd.concat([returns, benchmark], axis=1).dropna()
    if len(both) < 10 or both.iloc[:, 1].var() == 0:
        return np.nan, np.nan
    b = float(np.cov(both.iloc[:, 0], both.iloc[:, 1])[0, 1] / both.iloc[:, 1].var(ddof=1))
    a = float((both.iloc[:, 0].mean() - b * both.iloc[:, 1].mean()) * bars_per_year)
    return a, b


def worst_month(returns: pd.Series) -> float:
    if not isinstance(returns.index, pd.DatetimeIndex) or returns.empty:
        return np.nan
    monthly = (1.0 + returns).groupby(returns.index.to_period("M")).prod() - 1.0
    return float(monthly.min())


def performance_table(
    returns: pd.Series,
    benchmark: pd.Series,
    positions: pd.Series,
    trades: pd.DataFrame,
    bars_per_year: float,
) -> dict[str, float]:
    """The headline dictionary. Every number here is on whatever series you pass in, so pass OOS."""
    r = returns.dropna()
    eq = (1.0 + r).cumprod()
    mdd, dd_dur, _, _ = max_drawdown(eq)
    c = cagr(eq, bars_per_year)
    vol = float(r.std(ddof=1) * np.sqrt(bars_per_year)) if len(r) > 1 else np.nan
    a, b = alpha_beta(r, benchmark, bars_per_year)
    wins = trades.loc[trades["pnl"] > 0, "pnl"] if len(trades) else pd.Series(dtype=float)
    losses = trades.loc[trades["pnl"] <= 0, "pnl"] if len(trades) else pd.Series(dtype=float)
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    hit = float(len(wins) / len(trades)) if len(trades) else np.nan
    avg_w = float(wins.mean()) if len(wins) else np.nan
    avg_l = float(losses.mean()) if len(losses) else np.nan
    pf = gross_win / gross_loss if gross_loss > 0 else (np.inf if gross_win > 0 else np.nan)
    expectancy = float(trades["pnl"].mean()) if len(trades) else np.nan
    q = r.quantile([0.05, 0.95]) if len(r) > 20 else pd.Series([np.nan, np.nan], index=[0.05, 0.95])
    tail = float(abs(q.loc[0.95] / q.loc[0.05])) if q.loc[0.05] != 0 and np.isfinite(q.loc[0.05]) else np.nan
    turnover = float(positions.diff().abs().fillna(positions.abs()).sum() / len(positions) * bars_per_year) if len(positions) else np.nan
    bench_eq = (1.0 + benchmark.reindex(r.index).fillna(0.0)).cumprod()
    return {
        "cagr": c,
        "ann_vol": vol,
        "sharpe": sharpe(r, bars_per_year),
        "sortino": sortino(r, bars_per_year),
        "calmar": c / abs(mdd) if mdd < 0 else np.nan,
        "max_drawdown": mdd,
        "drawdown_duration_bars": dd_dur,
        "hit_rate": hit,
        "avg_win": avg_w,
        "avg_loss": avg_l,
        "profit_factor": pf,
        "expectancy": expectancy,
        "turnover_annual": turnover,
        "avg_exposure": float((positions != 0).mean()) if len(positions) else np.nan,
        "n_trades": int(len(trades)),
        "tail_ratio": tail,
        "worst_month": worst_month(r),
        "alpha_annual": a,
        "beta": b,
        "total_return": float(eq.iloc[-1] - 1.0) if len(eq) else np.nan,
        "bench_total_return": float(bench_eq.iloc[-1] - 1.0) if len(bench_eq) else np.nan,
        "bench_sharpe": sharpe(benchmark.reindex(r.index), bars_per_year),
        "bench_max_drawdown": max_drawdown(bench_eq)[0],
        "n_bars": int(len(r)),
    }


def rolling_sharpe(returns: pd.Series, window: int, bars_per_year: float) -> pd.Series:
    mu = returns.rolling(window, min_periods=window).mean()
    sd = returns.rolling(window, min_periods=window).std(ddof=1)
    return mu / sd.replace(0, np.nan) * np.sqrt(bars_per_year)


# --------------------------------------------------------------------------- #
# Deflated Sharpe ratio (Bailey & Lopez de Prado 2014)
# --------------------------------------------------------------------------- #
@dataclass
class DSRResult:
    sharpe_per_bar: float
    sharpe_annual: float
    n_trials: int
    expected_max_sharpe: float  # SR0, per bar
    psr: float  # probabilistic Sharpe vs 0
    dsr: float  # probabilistic Sharpe vs SR0
    significant: bool
    skew: float
    kurtosis: float
    n_obs: int

    def verdict(self) -> str:
        if not np.isfinite(self.dsr):
            return "Not enough data to compute a deflated Sharpe ratio."
        if self.significant:
            return (
                f"DSR = {self.dsr:.2f}: after accounting for {self.n_trials} trials, there is a {self.dsr:.0%} "
                "probability the true Sharpe is above what you would get from the best of that many random tries."
            )
        return (
            f"DSR = {self.dsr:.2f}: after accounting for {self.n_trials} trials, this Sharpe is NOT distinguishable "
            f"from the best of {self.n_trials} random strategies. The equity curve is probably selection bias."
        )


def probabilistic_sharpe(sr: float, sr_benchmark: float, n: int, skew: float, kurt: float) -> float:
    """PSR: P(true SR > sr_benchmark), per-bar SR, `kurt` is *non-excess* kurtosis."""
    if n < 3 or not np.isfinite(sr):
        return np.nan
    denom = np.sqrt(max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2))
    z = (sr - sr_benchmark) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """E[max SR] of n_trials i.i.d. draws with variance sr_variance, per bar."""
    if n_trials <= 1 or sr_variance <= 0 or not np.isfinite(sr_variance):
        return 0.0
    n = float(n_trials)
    z1 = stats.norm.ppf(1.0 - 1.0 / n)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(np.sqrt(sr_variance) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe(
    returns: pd.Series,
    n_trials: int,
    trial_sharpes_per_bar: np.ndarray | None,
    bars_per_year: float,
    alpha: float = 0.05,
) -> DSRResult:
    """Full DSR pipeline on a per-bar return series.

    `trial_sharpes_per_bar`: per-bar Sharpes of the other configurations tried
    (from the experiment log). Their variance sets how lucky the best trial can
    be. With fewer than 2 trials we fall back to the classic PSR against zero.
    """
    r = returns.dropna().to_numpy(dtype=float)
    n = len(r)
    if n < 3 or r.std(ddof=1) == 0:
        nan = float("nan")
        return DSRResult(nan, nan, n_trials, nan, nan, nan, False, nan, nan, n)
    sr = float(r.mean() / r.std(ddof=1))
    sk = float(stats.skew(r))
    ku = float(stats.kurtosis(r, fisher=False))
    if trial_sharpes_per_bar is not None and len(trial_sharpes_per_bar) >= 2:
        var = float(np.nanvar(trial_sharpes_per_bar, ddof=1))
    else:
        var = 0.0
    sr0 = expected_max_sharpe(max(n_trials, 1), var)
    psr = probabilistic_sharpe(sr, 0.0, n, sk, ku)
    dsr = probabilistic_sharpe(sr, sr0, n, sk, ku)
    return DSRResult(
        sharpe_per_bar=sr,
        sharpe_annual=sr * np.sqrt(bars_per_year),
        n_trials=int(n_trials),
        expected_max_sharpe=sr0,
        psr=psr,
        dsr=dsr,
        significant=bool(np.isfinite(dsr) and dsr >= 1.0 - alpha),
        skew=sk,
        kurtosis=ku,
        n_obs=n,
    )


# --------------------------------------------------------------------------- #
# Probability of backtest overfitting (Bailey, Borwein, Lopez de Prado, Zhu 2015)
# --------------------------------------------------------------------------- #
@dataclass
class PBOResult:
    pbo: float
    logits: np.ndarray
    n_configs: int
    n_partitions: int
    n_combinations: int
    is_oos_sharpe_pairs: np.ndarray  # (n_comb, 2): best-IS config's IS and OOS Sharpe

    def verdict(self) -> str:
        if not np.isfinite(self.pbo):
            return "PBO needs at least 2 logged runs with overlapping OOS periods."
        if self.pbo > 0.5:
            return f"PBO = {self.pbo:.0%}: picking the best-looking config is more likely than not to underperform out of sample."
        return f"PBO = {self.pbo:.0%}: selection on in-sample performance has a {self.pbo:.0%} chance of picking a loser."


def pbo_cscv(returns_matrix: pd.DataFrame, n_partitions: int = 10, max_combinations: int = 2000, seed: int = 0) -> PBOResult:
    """Combinatorially symmetric cross-validation over a T x N matrix of per-bar returns.

    Rows are bars, columns are configurations. The matrix is trimmed to bars all
    configurations share; NaNs from non-overlapping OOS windows are dropped.
    """
    M = returns_matrix.dropna(how="any")
    T, N = M.shape
    if N < 2 or T < 2 * n_partitions:
        return PBOResult(np.nan, np.array([]), N, n_partitions, 0, np.zeros((0, 2)))
    if n_partitions % 2:
        raise ValueError("n_partitions must be even.")
    blocks = np.array_split(np.arange(T), n_partitions)
    X = M.to_numpy()
    combos = list(combinations(range(n_partitions), n_partitions // 2))
    if len(combos) > max_combinations:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(combos), size=max_combinations, replace=False)
        combos = [combos[k] for k in keep]
    logits = []
    pairs = []
    all_blocks = set(range(n_partitions))
    for train_blocks in combos:
        tr = np.concatenate([blocks[b] for b in train_blocks])
        te = np.concatenate([blocks[b] for b in sorted(all_blocks - set(train_blocks))])
        sr_tr = _sharpe_cols(X[tr])
        sr_te = _sharpe_cols(X[te])
        best = int(np.nanargmax(sr_tr))
        # Rank of the IS-best config in the OOS ranking, as a fraction (0, 1).
        rank = (stats.rankdata(sr_te)[best]) / (N + 1)
        logits.append(np.log(rank / (1.0 - rank)))
        pairs.append((sr_tr[best], sr_te[best]))
    logits_arr = np.array(logits)
    pbo = float(np.mean(logits_arr < 0))
    return PBOResult(pbo, logits_arr, N, n_partitions, len(combos), np.array(pairs))


def _sharpe_cols(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(sd > 0, mu / sd, -np.inf)
    return out


# --------------------------------------------------------------------------- #
# Random strategy benchmark
# --------------------------------------------------------------------------- #
@dataclass
class RandomBenchmarkResult:
    strategy_sharpe: float
    random_sharpes: np.ndarray
    percentile: float  # share of random strategies the real one beats
    n: int

    def verdict(self) -> str:
        if not np.isfinite(self.percentile):
            return "Random benchmark could not be computed."
        if self.percentile < 0.95:
            return (
                f"The strategy beats only {self.percentile:.0%} of {self.n} random strategies with the same "
                "turnover and exposure. That is not evidence of skill."
            )
        return f"The strategy beats {self.percentile:.0%} of {self.n} turnover-matched random strategies."


def random_strategy_benchmark(
    positions: pd.Series,
    asset_returns: pd.Series,
    per_side_cost: float,
    bars_per_year: float,
    n: int = 1000,
    seed: int = 0,
) -> RandomBenchmarkResult:
    """Shuffle the strategy's own position *runs* to build random strategies with identical
    turnover, exposure and holding-period distribution, then compare Sharpes.

    Shuffling runs rather than bars is the whole point: a bar-level shuffle would
    produce a strategy that trades every bar and gets eaten by costs, which makes
    the real strategy look good for the wrong reason.
    """
    pos = positions.to_numpy(dtype=float)
    r = asset_returns.reindex(positions.index).fillna(0.0).to_numpy(dtype=float)
    runs = _runs(pos)
    if len(runs) < 2:
        return RandomBenchmarkResult(np.nan, np.array([]), np.nan, 0)
    real = _net_sharpe(pos, r, per_side_cost, bars_per_year)
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for k in range(n):
        order = rng.permutation(len(runs))
        p = np.concatenate([np.full(runs[i][1], runs[i][0]) for i in order])
        out[k] = _net_sharpe(p, r, per_side_cost, bars_per_year)
    pct = float(np.mean(out < real)) if np.isfinite(real) else np.nan
    return RandomBenchmarkResult(real, out, pct, n)


def _runs(x: np.ndarray) -> list[tuple[float, int]]:
    runs: list[tuple[float, int]] = []
    if len(x) == 0:
        return runs
    cur, length = x[0], 1
    for v in x[1:]:
        if v == cur:
            length += 1
        else:
            runs.append((cur, length))
            cur, length = v, 1
    runs.append((cur, length))
    return runs


def _net_sharpe(pos: np.ndarray, r: np.ndarray, per_side_cost: float, bars_per_year: float) -> float:
    cost = np.abs(np.diff(np.r_[0.0, pos])) * per_side_cost
    net = pos * r - cost
    return sharpe(net, bars_per_year)
