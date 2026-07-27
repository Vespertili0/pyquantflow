"""
Triple Barrier Trajectory Diagnostics Module

Provides visual tools for auditing Triple Barrier parameter calibration (take-profit, stop-loss, and horizon),
sampled price trajectories, exit type distributions, and holding period histograms.
"""

import warnings
import pandas as pd
import numpy as np
import plotly.subplots
import plotly.graph_objects as go
from ._renderer import DiagnosticResult, DiagnosticWarning, PALETTE


def plot_barrier_trajectories(
    df: pd.DataFrame,
    event_timestamps: pd.DatetimeIndex,
    pt_mult: float,
    sl_mult: float,
    horizon: int,
    n_events: int = 20,
    price_col: str = "Close",
    vol_col: str = "atr",
    seed: int = 42,
) -> DiagnosticResult:
    """
    Renders price trajectory paths for sampled event entries alongside TP/SL barriers and exit summary panels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data, `label`, `t1`, and volatility column (`vol_col`).
    event_timestamps : pd.DatetimeIndex
        Timestamps of event entry points to audit.
    pt_mult : float
        Profit-taking barrier multiplier applied to volatility.
    sl_mult : float
        Stop-loss barrier multiplier applied to volatility.
    horizon : int
        Vertical barrier maximum holding period in bars.
    n_events : int, default=20
        Number of events to randomly sample and render in the main trajectory panel.
    price_col : str, default="Close"
        Name of the price column.
    vol_col : str, default="atr"
        Name of the volatility/ATR column used to scale barrier width. Fallback to EWMA(20) if missing.
    seed : int, default=42
        Random seed for reproducible event sampling.

    Returns
    -------
    DiagnosticResult
        Contains the 3-panel Plotly figure and metadata (`pct_tp`, `pct_sl`, `pct_timeout`, `median_holding_bars`, `n_sampled`).
    """
    fig = plotly.subplots.make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}]],
        row_heights=[0.65, 0.35],
        vertical_spacing=0.08,
        subplot_titles=["Price Trajectories", "Exit Type Distribution", "Holding Period (bars)"],
    )
    
    if vol_col not in df.columns:
        warnings.warn(
            f"Column '{vol_col}' not found in df. Falling back to EWMA(span=20) volatility.",
            DiagnosticWarning,
            stacklevel=2
        )
        sigma = df[price_col].pct_change().ewm(span=20).std()
    else:
        sigma = df[vol_col]
        
    valid_events = event_timestamps.intersection(df.index)
    n_sampled = min(n_events, len(valid_events))
    
    if n_sampled > 0:
        rng = np.random.default_rng(seed)
        sampled_idx = rng.choice(len(valid_events), size=n_sampled, replace=False)
        sampled_ts = valid_events[np.sort(sampled_idx)]
        
        for ts in sampled_ts:
            label_val = df.loc[ts, "label"]
            t1_val = df.loc[ts, "t1"]
            
            if pd.isna(t1_val):
                continue
                
            colour = PALETTE["tp"] if label_val == 2 else PALETTE["sl"] if label_val == 0 else PALETTE["timeout"]
            
            entry_idx = df.index.get_loc(ts)
            end_idx_max = min(entry_idx + horizon, len(df) - 1)
            end_ts_max = df.index[end_idx_max]
            
            end_ts = min(t1_val, end_ts_max)
            
            path = df.loc[ts:end_ts, price_col]
            fig.add_trace(go.Scatter(
                x=path.index, 
                y=path.values, 
                mode="lines", 
                line={"color": colour, "width": 1.2}, 
                showlegend=False
            ), row=1, col=1)
            
            p_entry = df.loc[ts, price_col]
            sig = sigma.loc[ts]
            
            fig.add_trace(go.Scatter(
                x=[ts, end_ts],
                y=[p_entry * (1 + pt_mult * sig), p_entry * (1 + pt_mult * sig)],
                mode="lines",
                line={"color": PALETTE["tp"], "dash": "dash"},
                showlegend=False
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=[ts, end_ts],
                y=[p_entry * (1 - sl_mult * sig), p_entry * (1 - sl_mult * sig)],
                mode="lines",
                line={"color": PALETTE["sl"], "dash": "dash"},
                showlegend=False
            ), row=1, col=1)
            
            fig.add_vline(
                x=end_ts_max, 
                line_dash="dot", 
                line_color="grey", 
                row=1, col=1
            )
            
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line={"color": PALETTE["tp"]}, name="Take Profit"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line={"color": PALETTE["sl"]}, name="Stop Loss"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line={"color": PALETTE["timeout"]}, name="Timeout"), row=1, col=1)

    labels_full = df.loc[valid_events, "label"]
    pct_tp = (labels_full == 2).mean() if len(labels_full) > 0 else 0.0
    pct_sl = (labels_full == 0).mean() if len(labels_full) > 0 else 0.0
    pct_timeout = (labels_full == 1).mean() if len(labels_full) > 0 else 0.0
    
    fig.add_trace(go.Bar(
        y=["Exits"], x=[pct_tp * 100], orientation="h", name="Take Profit", marker_color=PALETTE["tp"], showlegend=False
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        y=["Exits"], x=[pct_timeout * 100], orientation="h", name="Timeout", marker_color=PALETTE["timeout"], showlegend=False
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        y=["Exits"], x=[pct_sl * 100], orientation="h", name="Stop Loss", marker_color=PALETTE["sl"], showlegend=False
    ), row=2, col=1)
    
    fig.update_layout(barmode="stack")
    
    t1_full = df.loc[valid_events, "t1"].dropna()
    holding_bars = []
    for ts, t1 in t1_full.items():
        if t1 in df.index:
            holding_bars.append(df.index.get_loc(t1) - df.index.get_loc(ts))
            
    if holding_bars:
        fig.add_trace(go.Histogram(
            x=holding_bars, 
            nbinsx=20, 
            marker_color=PALETTE["accent_1"],
            showlegend=False
        ), row=2, col=2)
        median_holding_bars = float(np.median(holding_bars))
    else:
        median_holding_bars = 0.0
        
    fig.update_layout(
        template="plotly_dark",
        font={"family": "Inter, DM Mono, monospace", "size": 13},
        paper_bgcolor="#0F0F13",
        plot_bgcolor="#0F0F13",
        colorway=list(PALETTE.values())
    )
        
    metadata = {
        "pct_tp": float(pct_tp),
        "pct_sl": float(pct_sl),
        "pct_timeout": float(pct_timeout),
        "median_holding_bars": median_holding_bars,
        "n_sampled": n_sampled,
    }
    
    return DiagnosticResult(figure=fig, metadata=metadata)
