"""Tab 1: price chart, return distribution, ACF/PACF, stationarity, integrity report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from quantlab.features.statistical import min_stationary_d
from quantlab.ui import plots
from quantlab.ui.state import df


@st.cache_data(show_spinner=False)
def _stationarity(close: pd.Series) -> dict[str, float]:
    from statsmodels.tsa.stattools import adfuller, kpss

    lr = np.log(close).diff().dropna()
    out = {}
    out["adf_p_price"] = float(adfuller(np.log(close.dropna()), autolag="AIC")[1])
    out["adf_p_returns"] = float(adfuller(lr, autolag="AIC")[1])
    try:
        out["kpss_p_returns"] = float(kpss(lr, regression="c", nlags="auto")[1])
    except Exception:  # noqa: BLE001 - kpss throws InterpolationWarning as errors in some versions
        out["kpss_p_returns"] = np.nan
    return out


@st.cache_data(show_spinner=False)
def _acf(close: pd.Series, nlags: int) -> tuple[np.ndarray, np.ndarray, float]:
    from statsmodels.tsa.stattools import acf, pacf

    lr = np.log(close).diff().dropna()
    # Drop lag 0: it is 1 by definition and makes every other bar invisible.
    return acf(lr, nlags=nlags, fft=True)[1:], pacf(lr, nlags=nlags)[1:], 1.96 / np.sqrt(len(lr))


@st.cache_data(show_spinner=False)
def _frac_sweep(close: pd.Series) -> pd.DataFrame:
    return min_stationary_d(np.log(close))


def render() -> None:
    d = df()
    if d is None:
        st.info("Load data from the sidebar first.")
        return
    rep = st.session_state["report"]
    st.plotly_chart(plots.price_chart(d), use_container_width=True)
    if "adj_factor" in d and (d["adj_factor"].round(6) != 1).any():
        st.caption("Dotted line is the unadjusted close. Where it diverges, dividends/splits were folded into the adjusted series you are modelling.")

    c1, c2 = st.columns(2)
    lr = np.log(d["close"]).diff()
    with c1:
        st.plotly_chart(plots.returns_histogram(lr), use_container_width=True)
        from scipy import stats

        r = lr.dropna()
        st.caption(f"skew {stats.skew(r):.2f}, excess kurtosis {stats.kurtosis(r):.2f}, Jarque-Bera p = {stats.jarque_bera(r)[1]:.2e}. Fat tails are not optional.")
    with c2:
        nlags = st.slider("ACF lags", 5, 60, 20)
        a, p, conf = _acf(d["close"], nlags)
        st.plotly_chart(plots.acf_bars(a, p, conf), use_container_width=True)

    st.subheader("Stationarity")
    s = _stationarity(d["close"])
    m1, m2, m3 = st.columns(3)
    m1.metric("ADF p (log price)", f"{s['adf_p_price']:.3f}", help="High p = unit root = do not feed raw prices to a model.")
    m2.metric("ADF p (log returns)", f"{s['adf_p_returns']:.2e}")
    m3.metric("KPSS p (log returns)", f"{s['kpss_p_returns']:.3f}", help="Low p here means KPSS rejects stationarity. ADF and KPSS should disagree with each other for a stationary series.")
    with st.expander("Fractional differentiation: minimum d for stationarity"):
        sweep = _frac_sweep(d["close"])
        st.dataframe(sweep.style.format({"adf_pvalue": "{:.4f}", "corr_with_original": "{:.3f}"}), use_container_width=True, hide_index=True)
        sd = sweep.attrs.get("suggested_d")
        if sd is not None and np.isfinite(sd):
            st.success(f"Smallest d with ADF p < 0.05: **{sd:.1f}**. Use it in the frac_diff feature if you want a stationary series that still remembers where price was.")
        else:
            st.warning("No d in [0, 1] made the series stationary at 5%. That is unusual; check the data.")

    st.subheader("Integrity report")
    summary = pd.Series(rep.summary()).astype(str).to_frame("value")
    st.dataframe(summary, use_container_width=True)
    if len(rep.price_gaps):
        st.markdown("**Returns beyond the sigma threshold**")
        st.dataframe(rep.price_gaps, use_container_width=True)
    if len(rep.suspected_bad_ticks):
        st.markdown("**Suspected bad ticks (spike and full revert)**")
        st.dataframe(rep.suspected_bad_ticks, use_container_width=True)
    if rep.gaps.missing_business_days:
        st.markdown(f"**Missing business days in long gaps** ({len(rep.gaps.missing_business_days)})")
        st.dataframe(pd.Series(rep.gaps.missing_business_days, name="date"), use_container_width=True)
