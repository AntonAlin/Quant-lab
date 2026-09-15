"""Plotly figures. Dark theme, unified hover, shared x where it makes sense. No business logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TEMPLATE = "plotly_dark"
GREY = "#8a8f98"
GREEN = "#26a69a"
RED = "#ef5350"
BLUE = "#42a5f5"
AMBER = "#ffb74d"


def _base(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(template=TEMPLATE, title=title, height=height, hovermode="x unified", margin=dict(l=40, r=20, t=50, b=40), legend=dict(orientation="h", y=1.02, x=0))
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1)
    return fig


def price_chart(df: pd.DataFrame, show_raw: bool = True) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="OHLC (adjusted)"), row=1, col=1)
    if show_raw and "close_raw" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["close_raw"], name="close (raw)", line=dict(color=GREY, width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="volume", marker_color=GREY), row=2, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False)
    return _base(fig, "Price and volume", 560)


def returns_histogram(r: pd.Series) -> go.Figure:
    r = r.dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=r, nbinsx=120, name="log returns", marker_color=BLUE, histnorm="probability density"))
    xs = np.linspace(r.min(), r.max(), 300)
    pdf = np.exp(-0.5 * ((xs - r.mean()) / r.std()) ** 2) / (r.std() * np.sqrt(2 * np.pi))
    fig.add_trace(go.Scatter(x=xs, y=pdf, name="normal with same mean/std", line=dict(color=AMBER)))
    fig.update_layout(hovermode="closest")
    return _base(fig, "Return distribution vs normal", 360)


def acf_bars(acf: np.ndarray, pacf: np.ndarray, conf: float) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("ACF", "PACF"))
    lags = np.arange(1, len(acf) + 1)
    fig.add_trace(go.Bar(x=lags, y=acf, marker_color=BLUE, name="ACF"), row=1, col=1)
    fig.add_trace(go.Bar(x=np.arange(1, len(pacf) + 1), y=pacf, marker_color=GREEN, name="PACF"), row=1, col=2)
    for c in (1, 2):
        fig.add_hline(y=conf, line=dict(color=GREY, dash="dot"), row=1, col=c)
        fig.add_hline(y=-conf, line=dict(color=GREY, dash="dot"), row=1, col=c)
    fig.update_layout(hovermode="closest", showlegend=False)
    return _base(fig, "Autocorrelation of log returns", 340)


def feature_lines(X: pd.DataFrame, cols: list[str], close: pd.Series | None = None) -> go.Figure:
    n = len(cols) + (1 if close is not None else 0)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    row = 1
    if close is not None:
        fig.add_trace(go.Scatter(x=close.index, y=close, name="close", line=dict(color=GREY)), row=row, col=1)
        row += 1
    for c in cols:
        fig.add_trace(go.Scatter(x=X.index, y=X[c], name=c, line=dict(width=1)), row=row, col=1)
        row += 1
    return _base(fig, "Selected features (shifted by one bar)", 160 * n + 100)


def heatmap(m: pd.DataFrame, title: str, zmin: float = -1, zmax: float = 1, colorscale: str = "RdBu") -> go.Figure:
    fig = go.Figure(go.Heatmap(z=m.values, x=m.columns, y=m.index, zmin=zmin, zmax=zmax, colorscale=colorscale, colorbar=dict(len=0.8)))
    fig.update_layout(hovermode="closest")
    return _base(fig, title, max(400, 22 * len(m) + 120))


def bar(series: pd.Series, title: str, color: str = BLUE, horizontal: bool = True) -> go.Figure:
    fig = go.Figure()
    if horizontal:
        fig.add_trace(go.Bar(x=series.values, y=series.index.astype(str), orientation="h", marker_color=color))
    else:
        fig.add_trace(go.Bar(x=series.index.astype(str), y=series.values, marker_color=color))
    fig.update_layout(hovermode="closest")
    return _base(fig, title, max(320, 22 * len(series) + 120) if horizontal else 360)


def labels_over_time(labels: pd.DataFrame, close: pd.Series) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
    fig.add_trace(go.Scatter(x=close.index, y=close, name="close", line=dict(color=GREY)), row=1, col=1)
    lab = labels["label"]
    for val, color, name in ((1, GREEN, "label +1"), (-1, RED, "label -1"), (0, AMBER, "label 0")):
        m = lab == val
        if m.any():
            fig.add_trace(go.Scatter(x=close.index[m], y=close[m], mode="markers", name=name, marker=dict(color=color, size=4, opacity=0.6)), row=1, col=1)
    # Rolling share of +1 labels: shows label drift, which is regime drift in a hat.
    share = (lab == 1).astype(float).where(lab.notna()).rolling(60, min_periods=20).mean()
    fig.add_trace(go.Scatter(x=share.index, y=share, name="rolling share of +1 (60 bars)", line=dict(color=BLUE)), row=2, col=1)
    fig.add_hline(y=0.5, line=dict(color=GREY, dash="dot"), row=2, col=1)
    return _base(fig, "Labels over time", 520)


def equity_curves(equity: pd.Series, bench: pd.Series, in_sample_mask: pd.Series | None = None, folds: pd.DataFrame | None = None) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
    fig.add_trace(go.Scatter(x=equity.index, y=equity, name="strategy (OOS)", line=dict(color=GREEN, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=bench.index, y=bench, name="buy & hold", line=dict(color=GREY, width=1.5)), row=1, col=1)
    dd_s = equity / equity.cummax() - 1
    dd_b = bench / bench.cummax() - 1
    fig.add_trace(go.Scatter(x=dd_s.index, y=dd_s, name="strategy drawdown", fill="tozeroy", line=dict(color=RED, width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=dd_b.index, y=dd_b, name="b&h drawdown", line=dict(color=GREY, width=1, dash="dot")), row=2, col=1)
    if folds is not None and len(folds):
        for i, row in folds.iterrows():
            if i % 2 == 0:
                fig.add_vrect(x0=row["test_start"], x1=row["test_end"], fillcolor="white", opacity=0.04, line_width=0, row=1, col=1)
    fig.update_yaxes(title_text="equity (x)", row=1, col=1)
    fig.update_yaxes(title_text="drawdown", tickformat=".0%", row=2, col=1)
    return _base(fig, "Equity vs buy & hold (out-of-sample only)", 560)


def rolling_sharpe_plot(rs: pd.Series, window: int) -> go.Figure:
    fig = go.Figure(go.Scatter(x=rs.index, y=rs, name=f"rolling Sharpe ({window} bars)", line=dict(color=BLUE)))
    fig.add_hline(y=0, line=dict(color=GREY, dash="dot"))
    return _base(fig, "Rolling Sharpe (OOS)", 320)


def fold_bars(fold_metrics: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=fold_metrics["fold"], y=fold_metrics["strategy_return"], name="strategy", marker_color=GREEN))
    fig.add_trace(go.Bar(x=fold_metrics["fold"], y=fold_metrics["bench_return"], name="buy & hold", marker_color=GREY))
    fig.update_layout(barmode="group", hovermode="x unified", yaxis_tickformat=".1%")
    return _base(fig, "Return per walk-forward fold", 340)


def positions_plot(positions: pd.Series, close: pd.Series) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.03)
    fig.add_trace(go.Scatter(x=close.index, y=close, name="close", line=dict(color=GREY)), row=1, col=1)
    fig.add_trace(go.Scatter(x=positions.index, y=positions, name="position", line=dict(color=BLUE, shape="hv")), row=2, col=1)
    return _base(fig, "Executed positions", 440)


def confusion_heatmap(cm: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=cm.values, x=cm.columns, y=cm.index, colorscale="Blues", text=cm.values, texttemplate="%{text}"))
    fig.update_layout(hovermode="closest")
    return _base(fig, "Confusion matrix (OOS)", 360)


def calibration_plot(cal: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="perfect", line=dict(color=GREY, dash="dot")))
    if len(cal):
        fig.add_trace(go.Scatter(x=cal["mean_predicted"], y=cal["fraction_positive"], name="model", mode="lines+markers", line=dict(color=GREEN)))
    fig.update_layout(hovermode="closest", xaxis_title="mean predicted probability", yaxis_title="observed frequency")
    return _base(fig, "Calibration (OOS)", 380)


def prob_by_class(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for cls, color in ((1, GREEN), (-1, RED), (0, AMBER)):
        m = df["true_class"] == cls
        if m.any():
            fig.add_trace(go.Histogram(x=df.loc[m, "p_long"], name=f"true class {cls}", opacity=0.6, marker_color=color, nbinsx=40))
    fig.update_layout(barmode="overlay", hovermode="closest", xaxis_title="P(long)")
    return _base(fig, "Predicted P(long) by true class (OOS)", 360)


def random_benchmark_hist(random_sharpes: np.ndarray, strategy_sharpe: float) -> go.Figure:
    fig = go.Figure(go.Histogram(x=random_sharpes, nbinsx=60, marker_color=GREY, name="random strategies"))
    fig.add_vline(x=strategy_sharpe, line=dict(color=GREEN, width=3), annotation_text="this strategy", annotation_position="top")
    fig.update_layout(hovermode="closest", xaxis_title="Sharpe (net of costs)")
    return _base(fig, "Turnover-matched random strategies", 340)


def pbo_hist(logits: np.ndarray) -> go.Figure:
    fig = go.Figure(go.Histogram(x=logits, nbinsx=40, marker_color=BLUE))
    fig.add_vline(x=0, line=dict(color=RED, width=2))
    fig.update_layout(hovermode="closest", xaxis_title="logit of OOS rank of the IS-best config (left of 0 = overfit)")
    return _base(fig, "PBO: distribution of OOS rank logits", 340)


def parallel_coords(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _base(go.Figure(), "No runs yet")
    dims = [dict(label=c, values=df[c]) for c in df.columns if c != "m_sharpe"]
    fig = go.Figure(go.Parcoords(line=dict(color=df["m_sharpe"], colorscale="Viridis", showscale=True, cmin=df["m_sharpe"].min(), cmax=df["m_sharpe"].max()), dimensions=dims + [dict(label="OOS Sharpe", values=df["m_sharpe"])]))
    fig.update_layout(template=TEMPLATE, height=460, margin=dict(l=60, r=40, t=50, b=30), title="Parameters vs OOS Sharpe")
    return fig


def regime_bars(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _base(go.Figure(), "No regime data")
    d = df.copy()
    d["label"] = d["regime_type"] + ": " + d["regime"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["label"], y=d["strategy_sharpe"], name="strategy", marker_color=GREEN))
    fig.add_trace(go.Bar(x=d["label"], y=d["bench_sharpe"], name="buy & hold", marker_color=GREY))
    fig.update_layout(barmode="group", hovermode="x unified")
    return _base(fig, "Sharpe per regime (OOS)", 380)


def shap_beeswarm(shap_df: pd.DataFrame, X: pd.DataFrame, top: int = 15, scale: str = "model output") -> go.Figure:
    order = shap_df.abs().mean().sort_values(ascending=False).index[:top]
    fig = go.Figure()
    for i, c in enumerate(order[::-1]):
        xv = X[c]
        col = (xv - xv.min()) / (xv.max() - xv.min() + 1e-12)
        fig.add_trace(go.Scatter(x=shap_df[c], y=np.full(len(shap_df), i) + np.random.uniform(-0.25, 0.25, len(shap_df)), mode="markers", name=c, marker=dict(color=col, colorscale="RdBu_r", size=5, opacity=0.6, showscale=(i == 0)), showlegend=False))
    fig.update_yaxes(tickvals=list(range(len(order))), ticktext=list(order[::-1]))
    fig.update_layout(hovermode="closest", xaxis_title=f"SHAP value (impact on {scale})")
    return _base(fig, "SHAP (last fold, OOS)", max(380, 26 * len(order) + 120))


def scatter(x: pd.Series, y: pd.Series, title: str, xlab: str, ylab: str) -> go.Figure:
    fig = px.scatter(x=x, y=y, labels={"x": xlab, "y": ylab})
    fig.update_traces(marker=dict(color=BLUE))
    fig.update_layout(hovermode="closest")
    return _base(fig, title, 360)
