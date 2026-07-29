"""
CPCV Path Fan Chart & PBO Visualiser

Measures and visualises the Probability of Backtest Overfitting (PBO) across all 
combinatorial paths from StrategyLab's out-of-sample combinations.
"""

import numpy as np
import pandas as pd
import plotly.subplots
import plotly.graph_objects as go
from typing import Any

from ._renderer import DiagnosticResult, FigureFactory, PALETTE


def plot_cpcv_paths(
    population: Any,  # skfolio.Population
    benchmark_sharpe: float = 0.0,
    annualisation_factor: float = 252.0,
) -> DiagnosticResult:
    """
    Renders an equity fan chart of CPCV paths and a Sharpe Ratio histogram for PBO.

    Parameters
    ----------
    population : skfolio.Population
        The population of portfolios from StrategyLab.evaluate_robustness_combinatorial().
    benchmark_sharpe : float, default=0.0
        The target Sharpe Ratio below which a strategy is considered overfit.
    annualisation_factor : float, default=252.0
        Periods per year used in the backtest (unused mathematically here since 
        the skfolio.Population handles annualisation natively, provided for API parity).

    Returns
    -------
    DiagnosticResult
        Dual-panel figure with equity paths fan chart and Sharpe distribution.
    """
    n_paths = len(population)
    if n_paths == 0:
        raise ValueError("Population is empty. Cannot compute PBO.")

    # Extract data from population
    # Assuming periodic returns; skfolio generally computes equity as (1 + r).cumprod()
    paths = [p.returns for p in population]
    sharpes = np.array([p.annualized_sharpe_ratio for p in population])

    # Compute cumulative equity paths
    cum_paths = [(1 + s).cumprod() for s in paths]

    # Align to common index
    # We forward fill because CPCV paths might drop out on different dates.
    aligned_df = pd.concat(cum_paths, axis=1).ffill()
    common_index = aligned_df.index
    path_matrix = aligned_df.values

    # Compute percentiles
    with np.errstate(invalid="ignore"):
        median_path = np.nanmedian(path_matrix, axis=1)
        p5 = np.nanpercentile(path_matrix, 5, axis=1)
        p95 = np.nanpercentile(path_matrix, 95, axis=1)

    fig = plotly.subplots.make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.1,
        subplot_titles=("OOS Equity Paths (Combinatorial CV)", "Distribution of OOS Sharpe Ratios"),
    )

    # Top panel: Fan chart
    # Band first (layer below)
    fig.add_trace(
        go.Scatter(
            x=common_index,
            y=p95,
            fill=None,
            mode="lines",
            line_color="rgba(0,0,0,0)",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    
    fig.add_trace(
        go.Scatter(
            x=common_index,
            y=p5,
            fill="tonexty",
            mode="lines",
            line_color="rgba(0,0,0,0)",
            fillcolor="rgba(167, 139, 250, 0.15)",  # faint PALETTE["accent_2"]
            name="5th–95th Pct",
        ),
        row=1,
        col=1,
    )

    # Plot all individual paths very faintly
    # Subsampling might be necessary if n_paths is huge (e.g. thousands), 
    # but Plotly handles hundreds fine.
    for i in range(n_paths):
        fig.add_trace(
            go.Scatter(
                x=common_index,
                y=path_matrix[:, i],
                mode="lines",
                line=dict(color="rgba(200, 200, 200, 0.15)", width=0.8),
                showlegend=False,
                name=f"Path {i}",
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    # Median path on top
    fig.add_trace(
        go.Scatter(
            x=common_index,
            y=median_path,
            mode="lines",
            line=dict(color=PALETTE["accent_1"], width=2.5),
            name="Median Path",
        ),
        row=1,
        col=1,
    )

    # Bottom panel: Sharpe Histogram
    fig.add_trace(
        go.Histogram(
            x=sharpes,
            marker_color=PALETTE["accent_2"],
            name="OOS Sharpe",
            nbinsx=30,
        ),
        row=2,
        col=1,
    )

    median_oos_sharpe = float(np.median(sharpes))
    pct5_oos_sharpe = float(np.percentile(sharpes, 5))
    pct95_oos_sharpe = float(np.percentile(sharpes, 95))

    fig.add_vline(
        x=median_oos_sharpe,
        row=2,
        col=1,
        line_dash="dash",
        line_color=PALETTE["accent_1"],
        annotation_text=f"Median={median_oos_sharpe:.2f}",
        annotation_position="top left",
    )

    fig.add_vline(
        x=pct5_oos_sharpe,
        row=2,
        col=1,
        line_dash="dot",
        line_color=PALETTE["warning"],
        annotation_text=f"5th Pct={pct5_oos_sharpe:.2f}",
        annotation_position="top left",
    )

    pbo_estimate = float((sharpes < benchmark_sharpe).mean())

    fig.add_annotation(
        text=f"PBO = {pbo_estimate:.1%}",
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.05,
        row=2,
        col=1,
        showarrow=False,
        font=dict(size=14, color=PALETTE["warning"]),
        bgcolor="rgba(15, 15, 19, 0.8)",
    )

    fig.update_layout(
        template="plotly_dark",
        font={"family": "Inter, DM Mono, monospace", "size": 13},
        paper_bgcolor="#0F0F13",
        plot_bgcolor="#0F0F13",
        xaxis2_title="Annualised Sharpe Ratio",
        yaxis_title="Cumulative Equity",
        yaxis2_title="Count",
    )
    
    # Use logarithmic y-axis if equity grows significantly (optional, leaving linear as default)
    # fig.update_yaxes(type="log", row=1, col=1)

    metadata = {
        "pbo_estimate": pbo_estimate,
        "median_oos_sharpe": median_oos_sharpe,
        "pct5_oos_sharpe": pct5_oos_sharpe,
        "pct95_oos_sharpe": pct95_oos_sharpe,
        "n_paths": n_paths,
    }

    return DiagnosticResult(figure=fig, metadata=metadata)
