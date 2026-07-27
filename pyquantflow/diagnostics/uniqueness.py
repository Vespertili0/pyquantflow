"""
Concurrency & Sample Uniqueness Diagnostics Module

Provides diagnostics for evaluating trade concurrency (c_t) and return-weighted sample uniqueness (u_i)
to detect hidden label redundancy and sample size collapse.
"""

import pandas as pd
import numpy as np
import plotly.subplots
import plotly.graph_objects as go
from typing import Optional
from ._renderer import DiagnosticResult, FigureFactory, PALETTE


def plot_sample_concurrency(
    t1_series: pd.Series,
    weight_series: Optional[pd.Series] = None,
    concurrency_threshold_pct: float = 0.75,
) -> DiagnosticResult:
    """
    Renders a dual-axis diagnostic chart showing active label concurrency (c_t) and sample uniqueness (u_i).

    Parameters
    ----------
    t1_series : pd.Series
        Event resolution timestamps where index represents event start time (t0) and values are end times (t1).
    weight_series : Optional[pd.Series], default=None
        Return-attributed sample uniqueness weights (u_i * |returns|) to overlay on secondary y-axis.
    concurrency_threshold_pct : float, default=0.75
        Percentile threshold of c_t above which contiguous time periods are shaded as warning bands.

    Returns
    -------
    DiagnosticResult
        Contains dual-axis Plotly figure and metadata (`mean_uniqueness`, `peak_concurrency`, `effective_sample_size`, `pct_high_concurrency`).
    """
    # 1. Build business-day date range
    t0_dt = pd.to_datetime(t1_series.dropna().index, utc=True)
    t1_dt = pd.to_datetime(t1_series.dropna().values, utc=True)
    
    if len(t0_dt) == 0:
        raise ValueError("t1_series is empty or contains only NaTs")
        
    t0_idx_naive = t0_dt.tz_convert(None)
    t1_vals_naive = t1_dt.tz_convert(None)
        
    date_range = pd.date_range(
        start=t0_idx_naive.min(), end=t1_vals_naive.max(), freq="B"
    )
    bars = date_range.values  # numpy datetime64 array

    # 2. Difference-array concurrency
    start_idx = np.searchsorted(bars, t0_idx_naive.values, side="left")
    end_idx   = np.searchsorted(bars, t1_vals_naive.values, side="right")

    diff = np.zeros(len(bars) + 1)
    for s, e in zip(start_idx, end_idx):
        diff[s] += 1
        diff[min(e, len(bars))] -= 1
    c_t = np.cumsum(diff)[:-1]  # shape: (len(date_range),)

    # 3. Point uniqueness and per-event average uniqueness (u_i)
    point_u = np.divide(1.0, c_t, out=np.zeros_like(c_t, dtype=float), where=c_t > 0)
    cum_u   = np.insert(np.cumsum(point_u), 0, 0)
    u_i = np.array([
        (cum_u[min(e, len(bars))] - cum_u[s]) / max(min(e, len(bars)) - s, 1)
        for s, e in zip(start_idx, end_idx)
    ])
    
    fig = plotly.subplots.make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        row_heights=[0.6, 0.4], 
        vertical_spacing=0.05,
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
    )
    
    plot_dates = date_range.tz_localize("UTC")
    
    fig.add_trace(go.Bar(
        x=plot_dates, 
        y=c_t, 
        name="Concurrency c_t", 
        marker_color=PALETTE["accent_1"]
    ), row=1, col=1)
    
    threshold_value = float(np.percentile(c_t, concurrency_threshold_pct * 100))
    
    # Add warning bands
    high_conc_mask = c_t > threshold_value
    runs = []
    in_run = False
    start_run = 0
    for i in range(len(high_conc_mask)):
        if high_conc_mask[i] and not in_run:
            start_run = i
            in_run = True
        elif not high_conc_mask[i] and in_run:
            runs.append((start_run, i - 1))
            in_run = False
    if in_run:
        runs.append((start_run, len(high_conc_mask) - 1))
        
    for s, e in runs:
        fig.add_vrect(
            x0=plot_dates[s], 
            x1=plot_dates[e], 
            fillcolor=PALETTE["warning"], 
            opacity=0.15, 
            layer="below", 
            line_width=0,
            row=1, col=1
        )
        
    fig.add_trace(go.Scatter(
        x=t0_dt, 
        y=u_i, 
        mode="markers", 
        name="Uniqueness u_i", 
        marker_color=PALETTE["accent_2"]
    ), row=2, col=1)
    
    if weight_series is not None:
        fig.add_trace(go.Scatter(
            x=weight_series.index, 
            y=weight_series.values, 
            mode="lines", 
            name="Sample Weight",
            line_color="rgba(255, 255, 255, 0.5)"
        ), row=2, col=1, secondary_y=True)
        
    fig.update_layout(
        template="plotly_dark",
        font={"family": "Inter, DM Mono, monospace", "size": 13},
        paper_bgcolor="#0F0F13",
        plot_bgcolor="#0F0F13",
        colorway=list(PALETTE.values()),
        showlegend=False
    )
    
    metadata = {
        "mean_uniqueness": float(u_i.mean()) if len(u_i) > 0 else 0.0,
        "peak_concurrency": int(c_t.max()) if len(c_t) > 0 else 0,
        "effective_sample_size": float(weight_series.sum()) if weight_series is not None else float(u_i.sum()),
        "pct_high_concurrency": float(high_conc_mask.sum() / len(c_t)) if len(c_t) > 0 else 0.0,
    }
    
    return DiagnosticResult(figure=fig, metadata=metadata)
