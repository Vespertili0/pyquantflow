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
from .features import plot_downsampling_shift, plot_stationarity_profile
from .clustering import plot_feature_clusters
from .cv import plot_cv_splits, plot_fold_feature_drift
from .metalabel import plot_meta_label_entropy, plot_meta_label_precision_recall
from .regimes import plot_sadf_regimes
from .pbo import plot_cpcv_paths
from . import _accessors  # noqa: F401  # side-effect: injects methods onto AssetOrganiser

__all__ = [
    "DiagnosticResult",
    "DiagnosticWarning",
    "FigureFactory",
    "plot_cusum_events",
    "plot_multi_asset_events",
    "plot_sample_concurrency",
    "plot_barrier_trajectories",
    "plot_downsampling_shift",
    "plot_stationarity_profile",
    "plot_feature_clusters",
    "plot_cv_splits",
    "plot_fold_feature_drift",
    "plot_meta_label_entropy",
    "plot_meta_label_precision_recall",
    "plot_sadf_regimes",
    "plot_cpcv_paths",
]
