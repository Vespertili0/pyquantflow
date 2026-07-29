"""
Feature Distribution & FFD Audit Diagnostics

Provides visual proof of stationarity (ADF tests) and feature distribution
shifts (CUSUM downsampling impact) using Fractional Differentiation (FFD).
"""

import numpy as np
import pandas as pd
import plotly.subplots
import plotly.graph_objects as go
from scipy.stats import gaussian_kde, entropy as scipy_entropy, wasserstein_distance
from typing import Optional, List

from ._renderer import DiagnosticResult, PALETTE
from pyquantflow.data.features.fractional_differentiation import _adf_test_stat


def plot_downsampling_shift(
    raw_df: pd.DataFrame,
    event_df: pd.DataFrame,
    feature_cols: List[str],
    n_bins: int = 50,
    divergence_metric: str = "kl",
) -> DiagnosticResult:
    """
    Audits how event-based downsampling reshapes feature distributions.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Continuous bar DataFrame containing the features.
    event_df : pd.DataFrame
        Event-sampled (e.g., CUSUM filtered) DataFrame containing the features.
    feature_cols : List[str]
        List of feature column names to evaluate.
    n_bins : int, default=50
        Number of points to evaluate the KDE over.
    divergence_metric : str, default="kl"
        Metric to compute divergence. Either "kl" (Kullback-Leibler) or "wasserstein".

    Returns
    -------
    DiagnosticResult
        Multi-subplot figure with KDE overlays and divergence scores.
    """
    if divergence_metric not in ["kl", "wasserstein"]:
        raise ValueError("divergence_metric must be 'kl' or 'wasserstein'")

    fig = plotly.subplots.make_subplots(
        rows=1,
        cols=len(feature_cols),
        subplot_titles=feature_cols,
        horizontal_spacing=0.05,
    )

    divergence_scores = {}
    flagged_features = []

    for i, col in enumerate(feature_cols):
        raw_vals = raw_df[col].dropna()
        event_vals = event_df[col].dropna()

        if len(raw_vals) < 2 or len(event_vals) < 2:
            divergence_scores[col] = np.nan
            continue

        raw_kde = gaussian_kde(raw_vals)
        event_kde = gaussian_kde(event_vals)

        # Shared evaluation grid
        min_val = min(raw_vals.min(), event_vals.min())
        max_val = max(raw_vals.max(), event_vals.max())
        x_grid = np.linspace(min_val, max_val, n_bins)

        raw_pdf = raw_kde(x_grid)
        event_pdf = event_kde(x_grid)

        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=raw_pdf,
                mode="lines",
                name="Raw",
                line=dict(color=PALETTE["accent_1"]),
                showlegend=(i == 0),
            ),
            row=1,
            col=i + 1,
        )

        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=event_pdf,
                mode="lines",
                name="Event",
                line=dict(color=PALETTE["accent_2"]),
                showlegend=(i == 0),
            ),
            row=1,
            col=i + 1,
        )

        if divergence_metric == "kl":
            score = scipy_entropy(event_pdf + 1e-10, raw_pdf + 1e-10)
        else:
            score = wasserstein_distance(event_vals, raw_vals)

        divergence_scores[col] = float(score)

    valid_scores = [s for s in divergence_scores.values() if not np.isnan(s)]
    if valid_scores:
        threshold = np.percentile(valid_scores, 75)
        for i, col in enumerate(feature_cols):
            score = divergence_scores.get(col, np.nan)
            if score > threshold:
                flagged_features.append(col)
                # Update subplot title with warning
                fig.layout.annotations[i].text = f"{col} ⚠"

    fig.update_layout(
        template="plotly_dark",
        font={"family": "Inter, DM Mono, monospace", "size": 13},
        paper_bgcolor="#0F0F13",
        plot_bgcolor="#0F0F13",
        title_text="Feature Distribution Shift (Raw vs Downsampled)",
    )

    metadata = {
        "divergence_scores": divergence_scores,
        "n_features_flagged": len(flagged_features),
        "divergence_metric": divergence_metric,
    }

    return DiagnosticResult(figure=fig, metadata=metadata)


def plot_stationarity_profile(
    raw_series: pd.Series,
    ffd_series: pd.Series,
    d_order: float,
    ticker: Optional[str] = None,
    max_lags: int = 40,
) -> DiagnosticResult:
    """
    Visualises the stationarity profile and memory preservation of an FFD series.

    Parameters
    ----------
    raw_series : pd.Series
        Original raw time series.
    ffd_series : pd.Series
        Fractionally differentiated time series.
    d_order : float
        The differencing order d* used for FFD.
    ticker : Optional[str], default=None
        Symbol/name of the ticker for metadata logging.
    max_lags : int, default=40
        Maximum number of lags to plot in the ACF chart.

    Returns
    -------
    DiagnosticResult
        Dual-panel figure with time series overlay and ACF bar chart.
    """
    fig = plotly.subplots.make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.1,
        specs=[[{"secondary_y": True}], [{}]],
        subplot_titles=("Series Overlay", "Autocorrelation Function (ACF)"),
    )

    # Top panel: Series Overlay
    fig.add_trace(
        go.Scatter(
            x=raw_series.index,
            y=raw_series.values,
            mode="lines",
            name="Raw",
            line=dict(color=PALETTE["accent_1"]),
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=ffd_series.index,
            y=ffd_series.values,
            mode="lines",
            name=f"FFD (d={d_order:.2f})",
            line=dict(color=PALETTE["accent_2"]),
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    # Bottom panel: ACF
    lags = list(range(max_lags + 1))

    # Calculate ACF manually avoiding statsmodels dependency
    def _calc_acf(series, k):
        if isinstance(series.index, pd.MultiIndex) and "ticker" in series.index.names:
            shifted = series.groupby(level="ticker").shift(k)
            return series.corr(shifted)
        return series.autocorr(lag=k)

    raw_acf = [_calc_acf(raw_series, k) for k in lags]
    ffd_acf = [_calc_acf(ffd_series.dropna(), k) for k in lags]

    fig.add_trace(
        go.Bar(
            x=lags,
            y=raw_acf,
            name="Raw ACF",
            marker_color=PALETTE["accent_1"],
            legendgroup="raw",
            offsetgroup=1,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=lags,
            y=ffd_acf,
            name="FFD ACF",
            marker_color=PALETTE["accent_2"],
            legendgroup="ffd",
            offsetgroup=2,
        ),
        row=2,
        col=1,
    )

    # Annotations & Layout
    def _calc_adf(series):
        if isinstance(series.index, pd.MultiIndex) and "ticker" in series.index.names:
            stats = series.groupby(level="ticker").apply(lambda x: _adf_test_stat(x.dropna()))
            return stats.mean()
        return _adf_test_stat(series.dropna())

    adf_stat = _calc_adf(ffd_series)
    if np.isnan(adf_stat):
        adf_stat = 0.0

    fig.add_annotation(
        text=f"ADF t={adf_stat:.3f} | d*={d_order:.2f}",
        xref="x domain",
        yref="y domain",
        x=0.98,
        y=0.95,
        row=2,
        col=1,
        showarrow=False,
        font=dict(size=13),
        bgcolor="rgba(15, 15, 19, 0.8)",
        bordercolor=PALETTE["accent_2"],
        borderwidth=1,
    )

    fig.update_layout(
        template="plotly_dark",
        font={"family": "Inter, DM Mono, monospace", "size": 13},
        paper_bgcolor="#0F0F13",
        plot_bgcolor="#0F0F13",
        barmode="group",
        bargap=0.15,
        bargroupgap=0.1,
    )

    # Ensure integer ticks for ACF lags
    fig.update_xaxes(tickmode="linear", dtick=5, row=2, col=1)

    metadata = {
        "adf_stat": float(adf_stat),
        "d_order": float(d_order),
        "lag1_acf_raw": float(raw_acf[1]) if len(raw_acf) > 1 else np.nan,
        "lag1_acf_ffd": float(ffd_acf[1]) if len(ffd_acf) > 1 else np.nan,
        "ticker": ticker,
    }

    return DiagnosticResult(figure=fig, metadata=metadata)
