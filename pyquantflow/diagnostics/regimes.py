"""
Explosive Bubble & SADF Regime Diagnostics

Audits whether event triggers and model signals fire in safe, stationary regimes
versus dangerous explosive bubble phases detected by the GSADF statistic.
"""

import numpy as np
import pandas as pd
import plotly.subplots
import plotly.graph_objects as go
from typing import Optional, Union

from ._renderer import DiagnosticResult, PALETTE


def plot_sadf_regimes(
    price_series: pd.Series,
    sadf_series: pd.Series,
    critical_value: float = 1.4,
    events: Optional[Union[pd.DatetimeIndex, pd.Series]] = None,
    title: Optional[str] = None,
) -> DiagnosticResult:
    """
    Renders price overlaid with GSADF explosive bubble regimes and event triggers.

    Parameters
    ----------
    price_series : pd.Series
        The original price series.
    sadf_series : pd.Series
        The GSADF statistic series matching the price index.
    critical_value : float, default=1.4
        The threshold above which a regime is considered explosive.
    events : Optional[Union[pd.DatetimeIndex, pd.Series]], default=None
        Optional event timestamps to overlay.
    title : Optional[str], default=None
        Custom title for the figure.

    Returns
    -------
    DiagnosticResult
        Dual-panel figure displaying price with bubble regions and the SADF statistic.
    """
    fig = plotly.subplots.make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.05,
    )

    # Top panel: Price
    fig.add_trace(
        go.Scatter(
            x=price_series.index,
            y=price_series.values,
            mode="lines",
            name="Price",
            line=dict(color=PALETTE["accent_1"]),
        ),
        row=1,
        col=1,
    )

    # Explosive vrects detection
    mask = sadf_series > critical_value
    if mask.any():
        # Estimate bar size for padding single-bar regimes
        if len(sadf_series) > 1:
            bar_size = sadf_series.index[1] - sadf_series.index[0]
        else:
            bar_size = (
                pd.Timedelta(days=1)
                if pd.api.types.is_datetime64_any_dtype(sadf_series.index)
                else 1
            )

        # Find contiguous blocks where mask is True
        changes = mask.ne(mask.shift()).cumsum()
        for _, grp in sadf_series[mask].groupby(changes[mask]):
            x0 = grp.index[0]
            x1 = grp.index[-1]
            if len(grp) == 1:
                x1 = x0 + bar_size

            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor=PALETTE["sl"],
                opacity=0.2,
                layer="below",
                line_width=0,
                row="all",  # PRD says both share x-axis, usually helpful on both panels
                col=1,
            )

    pct_explosive_regime = float(mask.mean())

    n_events_in_explosive = 0
    pct_events_in_explosive = 0.0

    # Event overlays
    if events is not None:
        if isinstance(events, pd.Series):
            events_ts = events.index
        else:
            events_ts = events

        # Align events to closest price bars for y-values
        price_at_events = price_series.reindex(events_ts, method="nearest")

        fig.add_trace(
            go.Scatter(
                x=events_ts,
                y=price_at_events,
                mode="markers",
                marker_color=PALETTE["accent_1"],
                marker_symbol="line-ns-open",
                marker_size=10,
                name="Events",
            ),
            row=1,
            col=1,
        )

        # Check how many events fall into explosive regimes
        sadf_at_events = sadf_series.reindex(events_ts, method="nearest")
        events_in_bubbles = sadf_at_events > critical_value
        n_events_in_explosive = int(events_in_bubbles.sum())

        if len(events_ts) > 0:
            pct_events_in_explosive = float(n_events_in_explosive / len(events_ts))

    # Bottom panel: GSADF
    fig.add_trace(
        go.Scatter(
            x=sadf_series.index,
            y=sadf_series.values,
            mode="lines",
            name="GSADF",
            line=dict(color=PALETTE["accent_2"]),
        ),
        row=2,
        col=1,
    )

    fig.add_hline(
        y=critical_value,
        row=2,
        col=1,
        line_dash="dash",
        line_color=PALETTE["warning"],
        annotation_text=f"Critical = {critical_value}",
    )

    max_sadf_stat = float(sadf_series.max()) if not sadf_series.empty else np.nan

    plot_title = title if title else "GSADF Explosive Bubble Regimes"
    fig.update_layout(
        template="plotly_dark",
        font={"family": "Inter, DM Mono, monospace", "size": 13},
        paper_bgcolor="#0F0F13",
        plot_bgcolor="#0F0F13",
        title=plot_title,
    )

    # Remove rangeslider
    fig.update_xaxes(rangeslider_visible=False)

    metadata = {
        "pct_explosive_regime": pct_explosive_regime,
        "n_events_in_explosive": n_events_in_explosive,
        "pct_events_in_explosive": pct_events_in_explosive,
        "max_sadf_stat": max_sadf_stat,
        "critical_value": float(critical_value),
    }

    return DiagnosticResult(figure=fig, metadata=metadata)
