import warnings
from typing import List, Dict, Optional, Callable, Union

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy
import scipy.spatial.distance
import scipy.stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import silhouette_score
from tsfeatures import tsfeatures

# Shared FFD + ADF utilities (canonical source: data.features.fractional_differentiation)
from pyquantflow.data.features.fractional_differentiation import (
    frac_diff_ffd,
    adf_screened_ffd,
    _adf_test_stat,
    _adf_p_value,
)


class StationaryTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms non-stationary features using Fractional Differentiation (FFD)
    and applies z-score standardisation to normalise volatility regimes.

    Supports two z-score modes via ``z_mode``:

    * ``"rolling"`` (default) — causal rolling z-score computed per
      timestep from past data only.  Already leak-safe by construction;
      no future data enters the rolling window.  Best for financial data
      with regime shifts.
    * ``"global"`` — traditional mean / std frozen during ``fit()`` and
      projected unchanged during ``transform()``.  Suitable for use
      inside ``sklearn.pipeline.Pipeline`` objects where the
      cross-validator calls ``fit`` on training data and ``transform``
      on validation data, guaranteeing strict train/test separation.
    """

    def __init__(
        self,
        d_grid: np.ndarray = np.arange(0.0, 1.05, 0.05),
        significance_level: float = 0.05,
        rolling_z_window: int = 20,
        ffd_thres: float = 1e-4,
        z_mode: str = "rolling",
    ):
        self.d_grid = d_grid
        self.significance_level = significance_level
        self.rolling_z_window = rolling_z_window
        self.ffd_thres = ffd_thres
        self.z_mode = z_mode
        self.optimal_d_ = {}
        self.z_mean_ = {}
        self.z_std_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        """
        Determines the optimal differencing order d* for each feature column.
        Groups by 'ticker' if X is a MultiIndex DataFrame to prevent leakage.

        When ``z_mode='global'``, also computes and stores per-column
        ``z_mean_`` and ``z_std_`` from the FFD-transformed training data.
        """
        is_multi_index = isinstance(X.index, pd.MultiIndex)

        for col in X.columns:
            if is_multi_index:
                # Apply FFD per ticker group to prevent cross-asset leakage,
                # then evaluate stationarity globally across the differenced panel.
                optimal_d = 1.0
                for d_candidate in self.d_grid:
                    try:
                        unstacked = X[col].unstack(level="ticker")
                        diff_unstacked = unstacked.apply(
                            lambda s: frac_diff_ffd(
                                s, d=d_candidate, thres=self.ffd_thres
                            )
                        )
                        # Calculate ADF per ticker and aggregate
                        t_stats = diff_unstacked.apply(_adf_test_stat)
                        p_values = t_stats.apply(_adf_p_value)
                        p_value = p_values.mean()

                        diff_series = diff_unstacked.stack(level="ticker", dropna=False)

                        if diff_series.index.names != X.index.names:
                            diff_series = diff_series.reorder_levels(X.index.names)

                        diff_series = diff_series.reindex(X.index)
                    except Exception:
                        diff_series = frac_diff_ffd(
                            X[col], d=d_candidate, thres=self.ffd_thres
                        )
                        t_stat = _adf_test_stat(diff_series)
                        p_value = _adf_p_value(t_stat)

                    if p_value <= self.significance_level:
                        optimal_d = d_candidate
                        break

                self.optimal_d_[col] = optimal_d
            else:
                # Single-asset path: delegate entirely to adf_screened_ffd
                _, d_star = adf_screened_ffd(
                    X[col],
                    d=None,
                    thres=self.ffd_thres,
                    significance_level=self.significance_level,
                    d_grid=self.d_grid,
                )
                self.optimal_d_[col] = d_star

        # --- Global z-score: freeze mean/std from training data ---
        if self.z_mode == "global":
            for col in X.columns:
                d = self.optimal_d_.get(col, 1.0)
                if is_multi_index:
                    try:
                        unstacked = X[col].unstack(level="ticker")
                        diff_unstacked = unstacked.apply(
                            lambda s: adf_screened_ffd(s, d=d, thres=self.ffd_thres)[0]
                        )
                        diff_series = diff_unstacked.stack(level="ticker", dropna=False)
                        if diff_series.index.names != X.index.names:
                            diff_series = diff_series.reorder_levels(X.index.names)
                        diff_series = diff_series.reindex(X.index)
                    except Exception:
                        diff_series, _ = adf_screened_ffd(
                            X[col], d=d, thres=self.ffd_thres
                        )
                else:
                    diff_series, _ = adf_screened_ffd(X[col], d=d, thres=self.ffd_thres)

                clean = diff_series.dropna()
                self.z_mean_[col] = float(clean.mean())
                self.z_std_[col] = float(clean.std())

        self.feature_names_in_ = list(X.columns)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the FFD transformation using optimal d*, followed by z-score
        standardisation (rolling or global, depending on ``z_mode``).
        """
        X_out = pd.DataFrame(index=X.index)
        is_multi_index = isinstance(X.index, pd.MultiIndex)

        for col in X.columns:
            d = self.optimal_d_.get(col, 1.0)

            # 1. Apply FFD via adf_screened_ffd in explicit mode
            if is_multi_index:
                try:
                    unstacked = X[col].unstack(level="ticker")
                    diff_unstacked = unstacked.apply(
                        lambda s: adf_screened_ffd(s, d=d, thres=self.ffd_thres)[0]
                    )
                    diff_series = diff_unstacked.stack(level="ticker", dropna=False)

                    if diff_series.index.names != X.index.names:
                        diff_series = diff_series.reorder_levels(X.index.names)

                    diff_series = diff_series.reindex(X.index)
                except Exception:
                    diff_series, _ = adf_screened_ffd(X[col], d=d, thres=self.ffd_thres)
            else:
                diff_series, _ = adf_screened_ffd(X[col], d=d, thres=self.ffd_thres)

            # 2. Z-Score Standardisation
            if self.z_mode == "global":
                # Frozen parameters from fit()
                g_mean = self.z_mean_.get(col, 0.0)
                g_std = self.z_std_.get(col, 1.0)
                if g_std == 0 or np.isnan(g_std):
                    g_std = 1.0
                z_score = (diff_series - g_mean) / g_std
            else:
                # Causal Rolling Z-Score Normalisation
                if is_multi_index:
                    roll = diff_series.groupby(level="ticker")
                    mean = roll.transform(
                        lambda s: s.rolling(self.rolling_z_window).mean()
                    )
                    std = roll.transform(
                        lambda s: s.rolling(self.rolling_z_window).std()
                    )
                else:
                    mean = diff_series.rolling(self.rolling_z_window).mean()
                    std = diff_series.rolling(self.rolling_z_window).std()

                # To avoid division by zero for constant periods
                std = std.replace(0, np.nan)
                z_score = (diff_series - mean) / std

            X_out[col] = z_score

        return X_out

    def get_feature_names_out(self, input_features=None):
        """
        Returns feature names for the transformer output.

        Implements the sklearn ``get_feature_names_out`` protocol so that
        ``StationaryTransformer`` integrates seamlessly into
        ``sklearn.pipeline.Pipeline`` objects.

        Parameters
        ----------
        input_features : array-like of str, optional
            Input feature names.  If ``None``, uses ``feature_names_in_``
            stored during ``fit()``.

        Returns
        -------
        np.ndarray of str
            Output feature names (identical to input — no columns are
            added or removed).
        """
        if input_features is not None:
            return np.asarray(input_features)
        if hasattr(self, "feature_names_in_"):
            return np.asarray(self.feature_names_in_)
        raise ValueError(
            "No feature names available. Call fit() first or provide input_features."
        )


class FeatureEvaluator:
    """
    A pipeline manager for financial feature diagnosis, transformation,
    clustering, and out-of-sample evaluation.
    """

    def __init__(
        self,
        features: List[str],
        target_col: Optional[str] = None,
        weight_col: Optional[str] = None,
        t1_col: Optional[str] = None,
        cv: Optional[BaseEstimator] = None,
        significance_level: float = 0.05,
        freq: int = 1,
        memory_threshold: float = 0.10,
        raw_features: Optional[List[str]] = None,
    ):
        """
        Parameters
        ----------
        features : List[str]
            Columns to be fractionally differentiated and screened by Gate 1
            (ACF1 memory check) via ``StationaryTransformer``.
        raw_features : Optional[List[str]], default None
            Columns that bypass the ``StationaryTransformer`` entirely and are
            passed through as-is.  No FFD, no rolling z-score, and no Gate 1
            ACF1 pruning is applied.  Useful for natively stationary features
            such as microstructure spreads, volume, or categorical metadata.
        """
        self.features = list(features)
        self.raw_features: List[str] = list(raw_features) if raw_features else []
        self.target_col = target_col
        self.weight_col = weight_col
        self.t1_col = t1_col
        self.cv = cv
        self.significance_level = significance_level
        self.freq = freq
        self.memory_threshold = memory_threshold
        self.stationary_transformer = StationaryTransformer(
            significance_level=self.significance_level
        )
        self.importance_df = None
        self.regime_clusters_ = None

    def fit_transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Dual-path feature ingestion:

        **Transform path** (``self.features``)
            1. Apply ``StationaryTransformer`` (FFD + rolling z-score).
            2. Gate 1 Pruning — drop features whose ACF1 falls below
               ``self.memory_threshold``.  ``self.features`` is updated
               in-place to reflect survivors.

        **Raw pass-through path** (``self.raw_features``)
            Columns are extracted from ``df`` unchanged.  No FFD,
            no rolling z-score, and no ACF1 pruning is applied.
            ``self.raw_features`` is **never mutated** by this method.

        Both paths are concatenated with metadata columns and returned as a
        single unified DataFrame.

        NaNs generated by FFD are propagated to preserve panel integrity.
        """
        # --- Transform path ---
        X = df[self.features]

        self.stationary_transformer.fit(X)
        X_trans = self.stationary_transformer.transform(X)

        # Gate 1 Pruning: memory check (ACF1)
        # We use pandas natively to evaluate ACF1 globally, bypassing tsfeatures for speed on edge devices
        valid_features = []
        for feat in self.features:
            try:
                if (
                    isinstance(X_trans.index, pd.MultiIndex)
                    and "ticker" in X_trans.index.names
                ):
                    acf1_val = (
                        X_trans[feat]
                        .groupby(level="ticker")
                        .apply(lambda s: s.autocorr(lag=1))
                        .mean()
                    )
                else:
                    acf1_val = X_trans[feat].autocorr(lag=1)
            except Exception:
                acf1_val = np.nan

            if pd.notna(acf1_val):
                if acf1_val > self.memory_threshold:
                    valid_features.append(feat)
                else:
                    warnings.warn(
                        f"Feature '{feat}' failed Gate 1 memory check "
                        f"(ACF1 = {acf1_val:.4f} <= threshold {self.memory_threshold:.4f}) and was dropped."
                    )
            else:
                # Keep the feature if ACF1 profiling is unavailable or returns NaN
                valid_features.append(feat)

        self.features = valid_features
        X_trans = X_trans[self.features]

        # --- Raw pass-through path ---
        frames_to_concat = [X_trans]
        if self.raw_features:
            # Validate that all requested raw columns exist in df
            missing = [c for c in self.raw_features if c not in df.columns]
            if missing:
                raise KeyError(
                    f"raw_features columns not found in DataFrame: {missing}"
                )
            frames_to_concat.append(df[self.raw_features])

        # Merge back with targets and metadata
        cols_to_keep = [self.target_col]
        if self.weight_col:
            cols_to_keep.append(self.weight_col)
        if self.t1_col:
            cols_to_keep.append(self.t1_col)

        frames_to_concat.append(df[cols_to_keep])
        df_out = pd.concat(frames_to_concat, axis=1)

        # We do NOT drop NaNs here. They are propagated to preserve panel integrity.
        return df_out

    def compute_time_series_profiles(
        self,
        df: pd.DataFrame,
        columns: List[str],
        groupby_level: Optional[str] = "ticker",
    ) -> pd.DataFrame:
        """
        Uses Nixtla's tsfeatures to compute statistical metrics.
        Reshapes the MultiIndex panel data into ['unique_id', 'ds', 'y']
        before passing to tsfeatures.
        """
        if groupby_level == "ticker":
            # Use-Case B: Asset Clustering. unique_id = ticker (or ticker::feature)
            df_reset = df[columns].reset_index()
            if len(columns) == 1:
                col = columns[0]
                df_ts = df_reset.rename(
                    columns={"ticker": "unique_id", "datetime": "ds", col: "y"}
                )
                df_ts = df_ts[["unique_id", "ds", "y"]].dropna(subset=["y"])
            else:
                # Multiple columns: unique_id becomes ticker::feature
                df_melt = pd.melt(
                    df_reset,
                    id_vars=["datetime", "ticker"],
                    value_vars=columns,
                    var_name="feature",
                    value_name="y",
                )
                df_melt["unique_id"] = (
                    df_melt["ticker"].astype(str)
                    + "::"
                    + df_melt["feature"].astype(str)
                )
                df_ts = df_melt.rename(columns={"datetime": "ds"})[
                    ["unique_id", "ds", "y"]
                ].dropna(subset=["y"])
        else:
            # Use-Case A: Feature Diagnostics. unique_id = feature
            df_reset = df[columns].reset_index()
            if "datetime" in df_reset.columns:
                id_vars = ["datetime"]
                if "ticker" in df_reset.columns:
                    id_vars.append("ticker")
            else:
                df_reset["datetime"] = np.arange(len(df_reset))
                id_vars = ["datetime"]

            df_melt = pd.melt(
                df_reset,
                id_vars=id_vars,
                value_vars=columns,
                var_name="unique_id",
                value_name="y",
            )
            df_ts = df_melt.rename(columns={"datetime": "ds"})[
                ["unique_id", "ds", "y"]
            ].dropna(subset=["y"])

        # Run tsfeatures
        profiles = tsfeatures(df_ts, freq=self.freq, threads=1)
        return profiles.set_index("unique_id")

    @staticmethod
    def _coerce_numeric(data: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures all columns in ``data`` are numeric before distance-matrix
        computation.  Non-numeric columns (boolean, string, object, category)
        are encoded via ``OrdinalEncoder`` so that hierarchical clustering
        does not raise a ``ValueError`` on mixed-type inputs.

        A warning is emitted listing the columns that were encoded so the
        caller is aware that ordinal encoding was applied.
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [c for c in data.columns if c not in numeric_cols]

        if not non_numeric_cols:
            return data

        warnings.warn(
            f"Non-numeric columns detected in cluster_entities: {non_numeric_cols}. "
            "Applying OrdinalEncoder before distance-matrix computation. "
            "Consider pre-processing categorical features if the default ordinal "
            "encoding does not reflect their true relationship."
        )

        from sklearn.preprocessing import OrdinalEncoder

        enc = OrdinalEncoder()
        encoded = pd.DataFrame(
            enc.fit_transform(data[non_numeric_cols].astype(str)),
            columns=non_numeric_cols,
            index=data.index,
        )
        # Reconstruct in original column order
        return pd.concat([data[numeric_cols], encoded], axis=1)[data.columns]

    def cluster_entities(
        self,
        data: pd.DataFrame,
        method: str = "correlation",
        n_clusters: Optional[int] = None,
    ) -> Dict[int, List[Union[str, int]]]:
        """
        Groups entities hierarchically.
        If method == 'correlation', clusters the columns of data (features).
        If method == 'euclidean', clusters the rows of data (assets).

        Non-numeric columns (boolean, string, category) are automatically
        encoded via ``OrdinalEncoder`` before the distance matrix is
        constructed so that mixed-type feature sets do not cause a
        ``ValueError`` inside ``scipy.cluster.hierarchy``.
        """
        # Type-safe coercion: encode any non-numeric columns before clustering
        data = self._coerce_numeric(data)

        if method == "correlation":
            labels_list = data.columns.tolist()
            if len(labels_list) == 0:
                return {}
            if len(labels_list) == 1:
                return {1: [labels_list[0]]}
            corr = data.corr(method="pearson")
            dist_matrix = np.sqrt(0.5 * (1 - corr.clip(-1, 1)))
            condensed_dist = scipy.spatial.distance.squareform(
                dist_matrix.values, checks=False
            )
        elif method == "euclidean":
            labels_list = data.index.tolist()
            if len(labels_list) == 0:
                return {}
            if len(labels_list) == 1:
                return {1: [labels_list[0]]}
            from sklearn.preprocessing import StandardScaler

            scaled_data = StandardScaler().fit_transform(data)
            condensed_dist = scipy.spatial.distance.pdist(
                scaled_data, metric="euclidean"
            )
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        Z = scipy.cluster.hierarchy.linkage(condensed_dist, method="ward")

        if n_clusters is None:
            best_score = -1.0
            best_n = 2
            max_clusters = max(2, len(labels_list) - 1)
            if len(labels_list) <= 2:
                n_clusters = len(labels_list)
            else:
                if method == "correlation":
                    dist_sq = dist_matrix.values
                else:
                    dist_sq = scipy.spatial.distance.squareform(condensed_dist)

                for k in range(2, max_clusters + 1):
                    labels = scipy.cluster.hierarchy.fcluster(
                        Z, k, criterion="maxclust"
                    )
                    if len(np.unique(labels)) > 1:
                        score = silhouette_score(dist_sq, labels, metric="precomputed")
                        if score > best_score:
                            best_score = score
                            best_n = k
                n_clusters = best_n

        labels = scipy.cluster.hierarchy.fcluster(Z, n_clusters, criterion="maxclust")
        clusters = {}
        for i, lbl in enumerate(labels_list):
            if labels[i] not in clusters:
                clusters[labels[i]] = []
            clusters[labels[i]].append(lbl)

        return clusters

    def evaluate_importance(
        self,
        df: pd.DataFrame,
        estimator: BaseEstimator,
        metric: Callable,
        metric_kwargs: Optional[dict] = None,
        balance_classes: bool = True,
        greater_is_better: bool = True,
        needs_proba: bool = True,
    ) -> Dict[int, Dict[str, pd.DataFrame]]:
        """
        Runs the Macro-Regime Loop.
        1. Clusters assets into regimes based on their statistical profiles.
        2. Iteratively performs Clustered MDA and SFI on each regime's data slice.

        Parameters
        ----------
        df : pd.DataFrame
            The prepared panel DataFrame (output of ``fit_transform_features``).
        estimator : BaseEstimator
            A scikit-learn-compatible estimator.
        metric : Callable
            A scoring/loss callable with signature ``metric(y_true, y_pred, **metric_kwargs)``.
        metric_kwargs : dict, optional
            Extra keyword arguments forwarded to ``metric``.
        balance_classes : bool, default True
            Whether to multiply sample weights by balanced class weights during fitting.
        greater_is_better : bool, default True
            Set to ``True`` for accuracy/score metrics (e.g. ``f1_score``, ``accuracy_score``)
            where a higher value is better.  Set to ``False`` for loss metrics
            (e.g. ``brier_score_loss``, ``log_loss``) where a lower value is better.
            This flag determines the sign convention of the MDA calculation:
            - ``True``  → importance = baseline_score − perturbed_score
            - ``False`` → importance = perturbed_score − baseline_score
        needs_proba : bool, default True
            Set to ``True`` if the metric requires probability outputs
            (e.g. ``log_loss``, ``roc_auc_score``, ``brier_score_loss``).
            Set to ``False`` if the metric operates on hard class labels
            (e.g. ``f1_score``, ``accuracy_score``).
        """
        metric_kwargs = metric_kwargs or {}
        groupby_level = "ticker"

        # Combined ordered list: transformed features first, then raw pass-through
        all_features = self.features + self.raw_features

        # 1. Macro-Regime Profiling & Asset Clustering
        if isinstance(df.index, pd.MultiIndex) and groupby_level in df.index.names:
            profiles = self.compute_time_series_profiles(
                df, all_features, groupby_level=groupby_level
            )

            if len(all_features) == 1:
                entity_profiles = profiles.fillna(0)
            else:
                idx_df = profiles.index.to_series().str.split("::", n=1, expand=True)
                idx_df.columns = ["ticker", "feature"]
                profiles_copy = profiles.copy()
                profiles_copy.index = pd.MultiIndex.from_arrays(
                    [idx_df["ticker"], idx_df["feature"]]
                )
                entity_profiles = profiles_copy.unstack(level="feature")
                entity_profiles.columns = [
                    f"{col[0]}_{col[1]}" for col in entity_profiles.columns
                ]
                entity_profiles = entity_profiles.fillna(0)

            self.regime_clusters_ = self.cluster_entities(
                entity_profiles, method="euclidean"
            )
        else:
            self.regime_clusters_ = {0: [None]}

        regime_results = {}

        for regime_id, entities in self.regime_clusters_.items():
            if entities == [None]:
                df_regime = df.copy()
            else:
                df_regime = df[
                    df.index.get_level_values(groupby_level).isin(entities)
                ].copy()

            if len(df_regime) == 0:
                continue

            # 2. Cluster Features (Multicollinearity neutralisation for this regime)
            feature_clusters = self.cluster_entities(
                df_regime[all_features], method="correlation"
            )

            X = df_regime[all_features]
            y = df_regime[self.target_col]

            mda_scores = {c_id: [] for c_id in feature_clusters.keys()}
            sfi_scores = {c_id: [] for c_id in feature_clusters.keys()}

            for step, (train_idx, val_idx) in enumerate(self.cv.split(df_regime, y)):
                X_train = X.iloc[train_idx]
                X_val = X.iloc[val_idx]
                y_train = y.iloc[train_idx]
                y_val = y.iloc[val_idx]

                fit_params = {}
                if self.weight_col and self.weight_col in df_regime.columns:
                    sample_weight = (
                        df_regime[self.weight_col].iloc[train_idx].values.copy()
                    )
                    if balance_classes:
                        from sklearn.utils.class_weight import compute_sample_weight

                        class_weights = compute_sample_weight(
                            "balanced", np.ravel(y_train)
                        )
                        sample_weight = sample_weight * class_weights
                    if sample_weight.sum() > 0:
                        sample_weight = sample_weight / sample_weight.mean()

                    if hasattr(estimator, "steps"):
                        final_step = estimator.steps[-1][0]
                        fit_params[f"{final_step}__sample_weight"] = sample_weight
                    else:
                        fit_params["sample_weight"] = sample_weight

                # --- SFI ---
                for c_id, cols in feature_clusters.items():
                    est_sfi = clone(estimator)
                    est_sfi.fit(X_train[cols], y_train, **fit_params)

                    if needs_proba and hasattr(est_sfi, "predict_proba"):
                        preds = est_sfi.predict_proba(X_val[cols])
                        if preds.ndim > 1 and preds.shape[1] == 2:
                            preds = preds[:, 1]
                    else:
                        preds = est_sfi.predict(X_val[cols])

                    try:
                        score = metric(y_val, preds, **metric_kwargs)
                    except Exception:
                        score = np.nan
                    sfi_scores[c_id].append(score)

                # --- MDA ---
                est_mda = clone(estimator)
                est_mda.fit(X_train, y_train, **fit_params)

                if needs_proba and hasattr(est_mda, "predict_proba"):
                    base_preds = est_mda.predict_proba(X_val)
                    if base_preds.ndim > 1 and base_preds.shape[1] == 2:
                        base_preds = base_preds[:, 1]
                else:
                    base_preds = est_mda.predict(X_val)

                try:
                    baseline_score = metric(y_val, base_preds, **metric_kwargs)
                except Exception:
                    baseline_score = np.nan

                for c_id, cols in feature_clusters.items():
                    X_val_pert = X_val.copy()
                    for col in cols:
                        X_val_pert[col] = np.random.permutation(X_val_pert[col].values)

                    if needs_proba and hasattr(est_mda, "predict_proba"):
                        pert_preds = est_mda.predict_proba(X_val_pert)
                        if pert_preds.ndim > 1 and pert_preds.shape[1] == 2:
                            pert_preds = pert_preds[:, 1]
                    else:
                        pert_preds = est_mda.predict(X_val_pert)

                    try:
                        pert_score = metric(y_val, pert_preds, **metric_kwargs)
                    except Exception:
                        pert_score = np.nan

                    # greater_is_better=True  (accuracy/score): importance = drop in performance
                    # greater_is_better=False (loss metric):     importance = rise in loss
                    if greater_is_better:
                        mda = baseline_score - pert_score
                    else:
                        mda = pert_score - baseline_score
                    mda_scores[c_id].append(mda)

            # Aggregate results for this regime
            results = {}
            sfi_agg = []
            for c_id, cols in feature_clusters.items():
                sfi_agg.append(
                    {
                        "cluster_id": c_id,
                        "features": ", ".join(cols),
                        "sfi_mean": np.mean(sfi_scores[c_id]),
                        "sfi_std": np.std(sfi_scores[c_id]),
                    }
                )
            results["SFI"] = pd.DataFrame(sfi_agg).set_index("cluster_id")

            mda_agg = []
            for c_id, cols in feature_clusters.items():
                mda_agg.append(
                    {
                        "cluster_id": c_id,
                        "features": ", ".join(cols),
                        "mda_mean": np.mean(mda_scores[c_id]),
                        "mda_std": np.std(mda_scores[c_id]),
                    }
                )
            results["MDA"] = pd.DataFrame(mda_agg).set_index("cluster_id")

            regime_results[regime_id] = results

        self.importance_df = self._convert_results_to_table(regime_results)

        return regime_results

    @staticmethod
    def _convert_results_to_table(
        results: Dict[int, Dict[str, pd.DataFrame]],
    ) -> pd.DataFrame:
        """
        Converts the nested dictionary of regime results into a single consolidated DataFrame.
        """
        if not results:
            return pd.DataFrame()

        regime_dfs = []
        for regime_id, inner_dict in results.items():
            sfi_df = inner_dict["SFI"]
            mda_df = inner_dict["MDA"]

            # Join SFI and MDA on cluster_id, keeping the features list from one side
            merged = sfi_df.join(mda_df[["mda_mean", "mda_std"]], how="outer")
            regime_dfs.append(merged)

        return pd.concat(regime_dfs, keys=results.keys(), axis=0)
