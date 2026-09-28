"""
AssetOrganiser Accessors Module

Provides CachedAccessor for binding diagnostic methods to classes via the `.diagnostics` namespace.
"""

import pandas as pd

from pyquantflow.data.assetorganiser import AssetOrganiser
from pyquantflow.data.sk_transformers import GSADFTransformer
from pyquantflow.model.classifier import PrimarySecondaryClassifier
from pyquantflow.model.cross_validation import (
    CombinatorialPurgedKFold,
    PurgedKFoldCV,
)
from pyquantflow.model.feature_evaluation import (
    FeatureEvaluator,
    StationaryTransformer,
)

from ._renderer import DiagnosticResult
from .events import plot_multi_asset_events
from .uniqueness import plot_sample_concurrency


class CachedAccessor:
    """
    Custom property-like object to lazy-load and cache an accessor.
    """

    def __init__(self, name: str, accessor_cls: type) -> None:
        self._name = name
        self._accessor_cls = accessor_cls

    def __get__(self, obj, cls):
        if obj is None:
            return self._accessor_cls
        accessor_obj = self._accessor_cls(obj)
        object.__setattr__(obj, self._name, accessor_obj)
        return accessor_obj


def register_diagnostics_accessor(name: str):
    def decorator(accessor_cls):
        setattr(accessor_cls.get_target_class(), name, CachedAccessor(name, accessor_cls))
        return accessor_cls
    return decorator


class AODiagnostics:
    @classmethod
    def get_target_class(cls):
        return AssetOrganiser

    def __init__(self, obj: AssetOrganiser):
        self._obj = obj

    def plot_cusum_events(self) -> DiagnosticResult:
        if self._obj.cusum_events_map is None:
            raise AttributeError(
                "cusum_events_map is None. Call downsample_to_cusum_events() first."
            )
        return plot_multi_asset_events(
            multi_asset_df=self._obj.multi_asset,
            tickers=list(self._obj.cusum_events_map.keys()),
            events_map=self._obj.cusum_events_map,
        )

    def plot_sample_concurrency(
        self, concurrency_threshold_pct: float = 0.75
    ) -> DiagnosticResult:
        if "t1" not in self._obj.multi_asset.columns:
            raise KeyError("'t1' column missing. Call apply_continuous_labels() first.")

        t1_series = self._obj.multi_asset["t1"]
        if isinstance(t1_series.index, pd.MultiIndex):
            if "ticker" in t1_series.index.names:
                t1_series = t1_series.reset_index(level="ticker", drop=True)

        weight_col = self._obj.weight_col if self._obj.weight_col else "weight"
        weight_series = (
            self._obj.multi_asset[weight_col] if weight_col in self._obj.multi_asset.columns else None
        )
        if weight_series is not None and isinstance(weight_series.index, pd.MultiIndex):
            if "ticker" in weight_series.index.names:
                weight_series = weight_series.reset_index(level="ticker", drop=True)

        return plot_sample_concurrency(
            t1_series=t1_series,
            weight_series=weight_series,
            concurrency_threshold_pct=concurrency_threshold_pct,
        )

register_diagnostics_accessor("diagnostics")(AODiagnostics)


class STDiagnostics:
    @classmethod
    def get_target_class(cls):
        return StationaryTransformer

    def __init__(self, obj: StationaryTransformer):
        self._obj = obj

    def plot_stationarity_profile(self, raw_series, col, max_lags=40):
        from pyquantflow.data.features.fractional_differentiation import adf_screened_ffd
        from .features import plot_stationarity_profile

        d_star = self._obj.optimal_d_.get(col, 1.0)
        ffd_series, _ = adf_screened_ffd(raw_series, d=d_star, thres=self._obj.ffd_thres)
        return plot_stationarity_profile(
            raw_series, ffd_series, d_star, ticker=col, max_lags=max_lags
        )

register_diagnostics_accessor("diagnostics")(STDiagnostics)


class FEDiagnostics:
    @classmethod
    def get_target_class(cls):
        return FeatureEvaluator

    def __init__(self, obj: FeatureEvaluator):
        self._obj = obj

    def plot_feature_clusters(self, df, regime_id=None):
        from .clustering import plot_feature_clusters

        all_features = self._obj.features + self._obj.raw_features
        corr_matrix = df[all_features].corr()

        if self._obj.importance_df is not None:
            regime_results = self._obj.importance_df
        else:
            raise ValueError(
                "importance_df is None. You must run evaluate_importance() "
                "on the FeatureEvaluator before plotting feature clusters."
            )

        return plot_feature_clusters(
            regime_results=regime_results,
            correlation_matrix=corr_matrix,
            regime_id=regime_id,
        )

register_diagnostics_accessor("diagnostics")(FEDiagnostics)


class CVDiagnostics:
    def __init__(self, obj):
        self._obj = obj

    def plot_splits(self, X, y):
        from .cv import plot_cv_splits
        return plot_cv_splits(self._obj, X, y)

setattr(PurgedKFoldCV, "diagnostics", CachedAccessor("diagnostics", CVDiagnostics))
setattr(CombinatorialPurgedKFold, "diagnostics", CachedAccessor("diagnostics", CVDiagnostics))


class PSCDiagnostics:
    @classmethod
    def get_target_class(cls):
        return PrimarySecondaryClassifier

    def __init__(self, obj: PrimarySecondaryClassifier):
        self._obj = obj

    def plot_meta_diagnostics(self, X, y_true):
        from .metalabel import plot_meta_label_entropy

        enriched = self._obj.transform(X).copy()
        enriched["label"] = y_true

        return plot_meta_label_entropy(enriched)

register_diagnostics_accessor("diagnostics")(PSCDiagnostics)


class GSADFDiagnostics:
    @classmethod
    def get_target_class(cls):
        return GSADFTransformer

    def __init__(self, obj: GSADFTransformer):
        self._obj = obj

    def plot_sadf_regimes(self, price_series, sadf_series, critical_value=1.4, events=None):
        from .regimes import plot_sadf_regimes
        return plot_sadf_regimes(price_series, sadf_series, critical_value, events)

register_diagnostics_accessor("diagnostics")(GSADFDiagnostics)
