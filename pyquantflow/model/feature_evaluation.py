import pandas as pd
import numpy as np
import scipy.stats
import scipy.spatial.distance
import scipy.cluster.hierarchy
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import silhouette_score
from typing import List, Dict, Optional, Callable, Union
from tsfeatures import tsfeatures

# Import FFD logic
from pyquantflow.data.features.fractional_differentiation import frac_diff_ffd


def _adf_test_stat(series: pd.Series, lags: int = 1) -> float:
    """
    Computes the Augmented Dickey-Fuller t-statistic (constant only).
    Native implementation to avoid statsmodels dependency.
    """
    y = series.dropna().values
    if len(y) < lags + 2:
        return np.nan

    dy = np.diff(y)
    y_lag = y[:-1]
    Y = dy[lags:]
    n = len(Y)

    X = np.zeros((n, lags + 2))
    X[:, 0] = y_lag[lags - 1 : -1]  # y_{t-1}
    X[:, 1] = 1.0  # constant
    for i in range(lags):
        X[:, 2 + i] = dy[lags - 1 - i : -1 - i]  # lagged differences

    try:
        # Solve OLS: X * beta = Y
        beta, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan

    # Standard error of beta[0]
    # Check if residuals is empty (happens in exact fit)
    if len(residuals) == 0:
        residuals = np.sum((Y - X @ beta) ** 2)
    else:
        residuals = residuals[0]

    dof = max(1, n - X.shape[1])
    mse = residuals / dof

    try:
        cov_matrix = mse * np.linalg.inv(X.T @ X)
        se_gamma = np.sqrt(cov_matrix[0, 0])
    except np.linalg.LinAlgError:
        return np.nan

    if se_gamma == 0:
        return np.nan

    t_stat = beta[0] / se_gamma
    return float(t_stat)


def _adf_p_value(t_stat: float) -> float:
    """
    Approximates the MacKinnon p-value for the ADF test (constant only model)
    based on the t-statistic. Uses linear interpolation between key critical values.
    """
    if np.isnan(t_stat):
        return 1.0

    cv_1_pct = -3.43
    cv_5_pct = -2.86
    cv_10_pct = -2.57

    if t_stat <= cv_1_pct:
        return 0.01
    elif t_stat <= cv_5_pct:
        return 0.01 + 0.04 * (t_stat - cv_1_pct) / (cv_5_pct - cv_1_pct)
    elif t_stat <= cv_10_pct:
        return 0.05 + 0.05 * (t_stat - cv_5_pct) / (cv_10_pct - cv_5_pct)
    else:
        return 1.0


class StationaryTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms non-stationary features using Fractional Differentiation (FFD)
    and applies a causal rolling z-score to standardise volatility regimes.
    """

    def __init__(
        self,
        d_grid: np.ndarray = np.arange(0.0, 1.05, 0.05),
        significance_level: float = 0.05,
        rolling_z_window: int = 20,
        ffd_thres: float = 1e-4,
    ):
        self.d_grid = d_grid
        self.significance_level = significance_level
        self.rolling_z_window = rolling_z_window
        self.ffd_thres = ffd_thres
        self.optimal_d_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        """
        Determines the optimal differencing order d* for each feature column.
        Groups by 'ticker' if X is a MultiIndex DataFrame to prevent leakage.
        """
        is_multi_index = isinstance(X.index, pd.MultiIndex)

        for col in X.columns:
            optimal_d = 1.0
            for d in self.d_grid:
                # Apply FFD. Groupby prevents bleeding data across assets.
                if is_multi_index:
                    try:
                        diff_series = (
                            X[col]
                            .groupby(level="ticker", group_keys=False)
                            .apply(
                                lambda s: frac_diff_ffd(s, d=d, thres=self.ffd_thres)
                            )
                        )
                    except Exception:
                        # Fallback if groupby fails for some reason
                        diff_series = frac_diff_ffd(X[col], d=d, thres=self.ffd_thres)
                else:
                    diff_series = frac_diff_ffd(X[col], d=d, thres=self.ffd_thres)

                # Evaluate stationarity globally across the differenced panel
                t_stat = _adf_test_stat(diff_series)
                p_value = _adf_p_value(t_stat)

                if p_value <= self.significance_level:
                    optimal_d = d
                    break

            self.optimal_d_[col] = optimal_d

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the FFD transformation using optimal d*, followed by rolling z-score.
        """
        X_out = pd.DataFrame(index=X.index)
        is_multi_index = isinstance(X.index, pd.MultiIndex)

        for col in X.columns:
            d = self.optimal_d_.get(col, 1.0)

            # 1. Apply FFD
            if is_multi_index:
                try:
                    diff_series = (
                        X[col]
                        .groupby(level="ticker", group_keys=False)
                        .apply(lambda s: frac_diff_ffd(s, d=d, thres=self.ffd_thres))
                    )
                except Exception:
                    diff_series = frac_diff_ffd(X[col], d=d, thres=self.ffd_thres)
            else:
                diff_series = frac_diff_ffd(X[col], d=d, thres=self.ffd_thres)

            # 2. Causal Rolling Z-Score Normalisation
            if is_multi_index:
                roll = diff_series.groupby(level="ticker")
                mean = roll.transform(lambda s: s.rolling(self.rolling_z_window).mean())
                std = roll.transform(lambda s: s.rolling(self.rolling_z_window).std())
            else:
                mean = diff_series.rolling(self.rolling_z_window).mean()
                std = diff_series.rolling(self.rolling_z_window).std()

            # To avoid division by zero for constant periods
            std = std.replace(0, np.nan)
            z_score = (diff_series - mean) / std

            X_out[col] = z_score

        return X_out


class FeatureEvaluator:
    """
    A pipeline manager for financial feature diagnosis, transformation,
    clustering, and out-of-sample evaluation.
    """

    def __init__(
        self,
        features: List[str],
        target_col: str,
        weight_col: Optional[str] = None,
        t1_col: Optional[str] = None,
        cv: Optional[BaseEstimator] = None,
        significance_level: float = 0.05,
        freq: int = 1,
        memory_threshold: float = 0.10,
    ):
        self.features = list(features)
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

    def fit_transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Diagnoses stationarity and transforms features per asset.
        NaNs generated by FFD are propagated forward.
        Then, performs Gate 1 Pruning by checking memory preservation (ACF1)
        on the transformed features and dropping those that fail the threshold.
        """
        X = df[self.features]

        self.stationary_transformer.fit(X)
        X_trans = self.stationary_transformer.transform(X)

        # Gate 1 Pruning: memory check (ACF1)
        # We call compute_time_series_profiles with groupby_level=None to evaluate globally
        profiles = self.compute_time_series_profiles(
            X_trans, columns=self.features, groupby_level=None
        )

        valid_features = []
        for feat in self.features:
            if feat in profiles.index and "acf1" in profiles.columns:
                acf1_val = profiles.loc[feat, "acf1"]
                if acf1_val > self.memory_threshold:
                    valid_features.append(feat)
                else:
                    import warnings

                    warnings.warn(
                        f"Feature '{feat}' failed Gate 1 memory check "
                        f"(ACF1 = {acf1_val:.4f} <= threshold {self.memory_threshold:.4f}) and was dropped."
                    )
            else:
                # Keep the feature if ACF1 profiling is unavailable
                valid_features.append(feat)

        self.features = valid_features
        X_trans = X_trans[self.features]

        # Merge back with targets and metadata
        cols_to_keep = [self.target_col]
        if self.weight_col:
            cols_to_keep.append(self.weight_col)
        if self.t1_col:
            cols_to_keep.append(self.t1_col)

        df_out = pd.concat([X_trans, df[cols_to_keep]], axis=1)

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
        profiles = tsfeatures(df_ts, freq=self.freq)
        return profiles.set_index("unique_id")

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
        """
        if method == "correlation":
            corr = data.corr(method="pearson")
            dist_matrix = np.sqrt(0.5 * (1 - corr.clip(-1, 1)))
            labels_list = data.columns.tolist()
            condensed_dist = scipy.spatial.distance.squareform(
                dist_matrix.values, checks=False
            )
        elif method == "euclidean":
            from sklearn.preprocessing import StandardScaler

            scaled_data = StandardScaler().fit_transform(data)
            condensed_dist = scipy.spatial.distance.pdist(
                scaled_data, metric="euclidean"
            )
            labels_list = data.index.tolist()
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
    ) -> Dict[int, Dict[str, pd.DataFrame]]:
        """
        Runs the Macro-Regime Loop.
        1. Clusters assets into regimes based on their statistical profiles.
        2. Iteratively performs Clustered MDA and SFI on each regime's data slice.
        """
        metric_kwargs = metric_kwargs or {}
        groupby_level = "ticker"

        # 1. Macro-Regime Profiling & Asset Clustering
        if isinstance(df.index, pd.MultiIndex) and groupby_level in df.index.names:
            profiles = self.compute_time_series_profiles(
                df, self.features, groupby_level=groupby_level
            )

            if len(self.features) == 1:
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

            regime_clusters = self.cluster_entities(entity_profiles, method="euclidean")
        else:
            regime_clusters = {0: [None]}

        regime_results = {}

        for regime_id, entities in regime_clusters.items():
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
                df_regime[self.features], method="correlation"
            )

            X = df_regime[self.features]
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

                    if metric.__name__ in ("log_loss", "roc_auc_score") and hasattr(
                        est_sfi, "predict_proba"
                    ):
                        preds = est_sfi.predict_proba(X_val[cols])
                        if preds.ndim > 1 and preds.shape[1] == 2:
                            preds = preds[:, 1]
                    else:
                        preds = est_sfi.predict(X_val[cols])

                    score = metric(y_val, preds, **metric_kwargs)
                    sfi_scores[c_id].append(score)

                # --- MDA ---
                est_mda = clone(estimator)
                est_mda.fit(X_train, y_train, **fit_params)

                if metric.__name__ in ("log_loss", "roc_auc_score") and hasattr(
                    est_mda, "predict_proba"
                ):
                    base_preds = est_mda.predict_proba(X_val)
                    if base_preds.ndim > 1 and base_preds.shape[1] == 2:
                        base_preds = base_preds[:, 1]
                else:
                    base_preds = est_mda.predict(X_val)
                baseline_score = metric(y_val, base_preds, **metric_kwargs)

                for c_id, cols in feature_clusters.items():
                    X_val_pert = X_val.copy()
                    for col in cols:
                        np.random.shuffle(X_val_pert[col].values)

                    if metric.__name__ in ("log_loss", "roc_auc_score") and hasattr(
                        est_mda, "predict_proba"
                    ):
                        pert_preds = est_mda.predict_proba(X_val_pert)
                        if pert_preds.ndim > 1 and pert_preds.shape[1] == 2:
                            pert_preds = pert_preds[:, 1]
                    else:
                        pert_preds = est_mda.predict(X_val_pert)

                    pert_score = metric(y_val, pert_preds, **metric_kwargs)

                    if metric.__name__ == "log_loss":
                        mda = pert_score - baseline_score
                    else:
                        mda = baseline_score - pert_score
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

        return regime_results
