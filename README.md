# QuantLab

A local Streamlit laboratory for building directional ML trading strategies on a single instrument, and then doing everything possible to prove they are not real.

Download OHLCV from Yahoo Finance, pick features and a labeling scheme, train any of eight model families (or an ensemble, or a meta-labeled pair) under walk-forward validation with an embargo, backtest the concatenated out-of-sample predictions with non-zero costs, and get told, in a red box, whether the resulting Sharpe survives the number of things you have already tried.

No live trading, no broker, no cloud, no database. One instrument at a time. Everything lands in `~/.quantlab/`.

## Install

Python 3.11 or newer.

```bash
git clone <this repo> quant-lab
cd quant-lab
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The pinned `torch` wheel is the default PyPI build. On a machine without a GPU you can save a couple of gigabytes with:

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

Optional, makes `import quantlab` work from anywhere:

```bash
pip install -e .
```

## Run

```bash
streamlit run quantlab/app.py
```

Tests:

```bash
pytest
```

The suite runs on synthetic data and needs no network. It covers feature leakage (a future spike must not move any feature before it happens), walk-forward geometry and embargo, triple-barrier labels against a hand-computed series, engine P&L including costs, Sharpe and drawdown against hand-computed values, config round-tripping, and an end-to-end run through the store and leaderboard.

## Worked example: `^OMXS30`

1. **Sidebar.** Ticker `^OMXS30`, start `2005-01-01`, end today, interval `1d`. Click *Load data*. You get roughly 5,000 bars. Read the data-health box: the OMX index has zero volume on Yahoo for long stretches, which is normal for an index and the report says so. Anything about duplicate timestamps or spike-and-revert patterns is not normal.

2. **Data tab.** The ADF p-value on log price will be high (unit root) and near zero on returns. Open the fractional-differentiation expander; for OMXS30 the smallest `d` that passes ADF is typically around 0.3 to 0.4. Note it.

3. **Features tab.** Click *Trend + vol starter*. Add `frac_diff` with the `d` from step 2 and `hmm_state` if you are patient. Check the correlation heatmap: `sma_ratio` and `ema_ratio` at the same window are near-duplicates, drop one. Mutual information with the label will be tiny for everything. That is what daily equity index data looks like.

4. **Labels tab.** Triple barrier, horizon 10, profit-take 2.0 ATR, stop-loss 1.0 ATR, uniqueness weights. If more than 60% of labels hit the vertical barrier the multiples are too wide for the horizon. Watch the class balance: with asymmetric barriers you will get an imbalanced label and the tab will shout about accuracy.

5. **Model tab.** Start with `lightgbm`, 300 trees, depth via 15 leaves, learning rate 0.03. Walk-forward: rolling, train 1000, test 125, step 125, embargo 10 (it auto-fills to the label horizon). You get about 30 folds. Click *Train walk-forward*. Fold-by-fold test accuracy will bounce between 0.45 and 0.58. That is not noise in the code, that is the signal-to-noise ratio of the problem.

6. **Backtest tab.** Defaults: next-open execution, 2 bps commission, 5 bps spread, 2 bps slippage, long-short, fixed fractional 1x. Thresholds at 0.55. The first thing on the tab is the "Is this real?" block. Expect a raw Sharpe somewhere between -0.3 and 0.8 and a DSR well below 0.95 on your first try. The run is logged automatically.

7. **Diagnostics tab.** Permutation importance on OOS. On an index, most of what survives is volatility-family features, which mostly says the model has learned to shrink exposure in high-vol regimes. The per-regime table will confirm that. Calibration is usually poor: probabilities cluster around 0.5 and the thresholds decide everything.

8. **Compare tab.** After five or ten runs, the leaderboard recomputes every run's DSR against the current trial count, and PBO across the logged OOS return series tells you how often the in-sample winner is an out-of-sample loser. If PBO is above 0.5 and no row has DSR above 0.95, you have not found a strategy, you have found a search process.

## How to not fool yourself

The app is built around three mechanisms. They are not optional and the UI does not let you skip them quietly.

### The embargo

Every label looks forward. A fixed-horizon label at bar `t` is the return to `t+h`; a triple-barrier label resolves somewhere in `(t, t+h]`. If the training window ends at `t` and the test window starts at `t+1`, the last `h` training labels contain returns that overlap the first `h` test bars. The model is trained on the answer to the first part of the exam.

The walk-forward splitter inserts a gap of `embargo` bars between the end of train and the start of test, and additionally purges any training sample whose label resolution time lands inside the test window. The embargo defaults to the label horizon and the config refuses anything smaller unless you tick a box that says you know what you are doing. Features carry a one-bar shift on top of that, so a feature at row `t` describes information available at the close of `t-1`, which is when you would actually be deciding.

The leakage checks in `quantlab/validation/leakage.py` run before every walk-forward: chronological folds, no overlap, embargo respected, feature matrix shifted and index-aligned, no feature correlated 0.98+ with the forward return. If one of them fires, the fix is not to remove the check.

### Deflated Sharpe ratio (DSR)

If you try 50 configurations on the same data and keep the best one, its Sharpe is the maximum of 50 noisy draws, not an estimate of anything. Bailey and Lopez de Prado (2014) give the expected maximum Sharpe of `N` independent trials with a given cross-trial variance:

    SR0 = sqrt(V[SR]) * ((1 - γ) Φ⁻¹(1 - 1/N) + γ Φ⁻¹(1 - 1/(N e)))

with `γ` the Euler-Mascheroni constant. The DSR is then the probabilistic Sharpe ratio of your strategy against that benchmark instead of against zero, adjusted for the skewness and kurtosis of your returns and the number of observations. A DSR of 0.95 means a 95% probability the true Sharpe exceeds what the best of `N` random tries would have produced.

`N` and `V[SR]` come from the experiment log, which is why every evaluated backtest is written there whether you like it or not. The Backtest tab shows raw Sharpe and DSR side by side and lets you choose whether `N` counts all runs on this ticker or only this session's. The Compare tab recomputes each old run's DSR with today's `N`, so a run that looked significant at trial 3 stops looking significant at trial 40.

### Probability of backtest overfitting (PBO)

DSR asks whether one Sharpe is significant. PBO asks a different question: if you pick the best configuration by in-sample performance, how often does it underperform the median out of sample? It is estimated by combinatorially symmetric cross-validation (Bailey, Borwein, Lopez de Prado, Zhu, 2015): the OOS return series of every logged run are stacked into a matrix, time is cut into `S` blocks, and for every way of choosing `S/2` blocks as "train", the config with the best train Sharpe is looked up in the "test" ranking. The fraction of combinations where it lands below the median is the PBO. Above 0.5 means your selection procedure is worse than picking at random.

The Compare tab runs it across all logged runs on the current ticker. It needs at least two runs and, to mean anything, at least ten with overlapping OOS periods.

### And a fourth, because three was not enough

The Backtest tab also shuffles your strategy's own position runs 1,000 times, preserving turnover, exposure and holding-period distribution, and reports where your Sharpe sits in that distribution. A strategy in the 60th percentile of random strategies with the same trading pattern has not demonstrated skill, only a trading pattern.

## Project layout

```
quantlab/
├── app.py                 Streamlit entrypoint (layout only)
├── config.py              dataclasses for every config object, JSON round-trip, config hash
├── pipeline.py            orchestration: dataset -> walk-forward -> OOS -> backtest -> record
├── diagnostics.py         permutation importance, SHAP, calibration, regimes
├── seeding.py             one seed for python / numpy / torch
├── data/        loader (yfinance + parquet cache + integrity report), trading calendar
├── features/    registry (schema-driven), technical, statistical, regime + calendar
├── labeling/    fixed horizon, triple barrier, trend scanning, sample weights
├── models/      ModelAdapter base, classical (sklearn/xgb/lgbm), deep (torch), ensemble, meta-labeling
├── validation/  walk-forward splitter with embargo + purging, leakage assertions
├── backtest/    engine, sizing, metrics (incl. DSR, PBO, random benchmark)
├── experiments/ append-only parquet store, cross-run leaderboard
├── ui/          sidebar, one module per tab, Plotly figures
└── tests/       pytest suite
```

Two files are not in the original specification: `pipeline.py` and `diagnostics.py`. They exist because `app.py` was required to hold zero business logic and the orchestration had to live somewhere importable by both the UI and the tests.

## Things worth knowing

- **Caches.** Price data in `~/.quantlab/cache/`, runs in `~/.quantlab/experiments/runs.parquet`, per-run OOS returns in `~/.quantlab/experiments/returns/`. Delete the directory to start over; nothing in the app will do it for you.
- **Sequence models** build their training windows from the train fold only, so no training sequence ever contains a test bar. At prediction time they are handed the bars immediately before the test fold (train tail plus embargo) as history, exactly what a live model would see, so every test bar is scored. Calling `predict_proba` without that context leaves the first `seq_len - 1` rows honestly NaN rather than silently padded.
- **Hourly data.** Yahoo serves about 730 days of it. The 750-bar warning is calibrated for daily bars; on hourly data you will have the bars but not the years.
- **Re-tuning per fold** runs an Optuna TPE study per fold, minimising log-loss on the chronological tail of that fold with an embargo between the tuning-train and tuning-validation slices. Your configured hyperparameters are enqueued as trial 0, so tuning can never pick something worse on that tail than not tuning. Sampler is seeded, so it is reproducible. Ensembles are not re-tuned; tune the members individually first.
- **SHAP** is available for every family: `TreeExplainer` for the forests and boosters, `LinearExplainer` for logistic regression, `GradientExplainer` for the torch nets with the sequence axis summed out. Ensembles and meta-labeled models are explained one component at a time because members live on different output scales (probability vs log-odds) and averaging them would be meaningless.
- **The leaderboard DSR** is recomputed with each run's own stored return skew and kurtosis, not a normal approximation. Rows logged by older versions without those columns fall back to normal moments.
- **Determinism.** One seed drives python, numpy, torch and every estimator's `random_state`. Tree models with `n_jobs=-1` are still deterministic; LightGBM and XGBoost are deterministic on CPU with the pinned versions.
