"""
Diagnostics Package Initialization

This package provides visual diagnostic tools for Financial Machine Learning (FinML) workflows,
including CUSUM event overlays, sample concurrency and uniqueness profiling, and Triple Barrier
trajectory auditing.
"""

from ._renderer import DiagnosticResult, DiagnosticWarning, FigureFactory
from .events import plot_cusum_events, plot_multi_asset_events
from .uniqueness import plot_sample_concurrency
from .barriers import plot_barrier_trajectories
from . import _accessors  # noqa: F401  # side-effect: injects methods onto AssetOrganiser

__all__ = [
    "DiagnosticResult",
    "DiagnosticWarning",
    "FigureFactory",
    "plot_cusum_events",
    "plot_multi_asset_events",
    "plot_sample_concurrency",
    "plot_barrier_trajectories",
]
