"""
Renderer Core Module

Provides rendering abstractions, custom Plotly theme settings, diagnostic result structures,
and context-aware figure display helpers for marimo, IPython, and standard Python execution environments.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import sys
import plotly.graph_objects as go

PALETTE = {
    "tp":        "#06D6A0",   # green  — Take Profit
    "sl":        "#FF6B6B",   # red    — Stop Loss
    "timeout":   "#FFD166",   # orange — Timeout
    "accent_1":  "#00D9FF",   # cyan   — primary chart accent
    "accent_2":  "#A78BFA",   # violet — secondary accent
    "warning":   "#FF6B6B",   # red    — concurrency warning bands (same as SL)
}


class DiagnosticWarning(UserWarning):
    """Warning raised for non-fatal diagnostic issues, such as ticker truncation."""
    pass


@dataclass
class DiagnosticResult:
    """
    Encapsulates a Plotly figure alongside structured execution metadata.

    Attributes
    ----------
    figure : go.Figure
        The rendered Plotly figure object.
    metadata : Dict[str, Any]
        Plain-language dictionary containing scalar metrics and audit results.
    """
    figure: go.Figure
    metadata: Dict[str, Any]


class FigureFactory:
    """
    Factory for instantiating dark-themed Plotly figures and dispatching context-aware output.
    """

    @staticmethod
    def create(layout_overrides: Optional[Dict[str, Any]] = None) -> go.Figure:
        """
        Creates a pre-themed Plotly Figure adhering to project aesthetic guidelines.

        Parameters
        ----------
        layout_overrides : Optional[Dict[str, Any]], default=None
            Optional dictionary of Plotly layout overrides to update the figure layout.

        Returns
        -------
        go.Figure
            Configured dark-mode Plotly figure.
        """
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            font={"family": "Inter, DM Mono, monospace", "size": 13},
            paper_bgcolor="#0F0F13",
            plot_bgcolor="#0F0F13",
            colorway=list(PALETTE.values())
        )
        if layout_overrides:
            fig.update_layout(**layout_overrides)
        return fig

    @staticmethod
    def show(result: DiagnosticResult) -> None:
        """
        Dispatches figure rendering for marimo, Jupyter/IPython, or standard browser environments.

        Parameters
        ----------
        result : DiagnosticResult
            The diagnostic result containing the target figure.
        """
        fig = result.figure
        if "marimo" in sys.modules:
            mo = sys.modules["marimo"]
            if hasattr(mo, "as_html"):
                mo.as_html(fig)
                return
        
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                fig.show(renderer="notebook")
                return
        except ImportError:
            pass
            
        fig.show()
