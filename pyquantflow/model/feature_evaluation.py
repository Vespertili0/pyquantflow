import pandas as pd
import numpy as np
import scipy.stats
import scipy.signal
import scipy.spatial.distance
import scipy.cluster.hierarchy
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import silhouette_score
from typing import List, Dict, Optional, Callable, Union

# Import FFD logic (already implemented in pyquantflow)
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
    X[:, 1] = 1.0                   # constant
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
        ffd_thres: float = 1e-4
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
                        diff_series = X[col].groupby(level="ticker", group_keys=False).apply(
                            lambda s: frac_diff_ffd(s, d=d, thres=self.ffd_thres)
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
                    diff_series = X[col].groupby(level="ticker", group_keys=False).apply(
                        lambda s: frac_diff_ffd(s, d=d, thres=self.ffd_thres)
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
        significance_level: float = 0.05
    ):
        self.features = features
        self.target_col = target_col
        self.weight_col = weight_col
        self.t1_col = t1_col
        self.cv = cv
        self.significance_level = significance_level
        self.stationary_transformer = StationaryTransformer(
            significance_level=self.significance_level
        )
        
    def fit_transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Diagnoses stationarity and transforms features per asset.
        Drops rows with NaN values resulting from FFD lag.
        """
        X = df[self.features]
        
        self.stationary_transformer.fit(X)
        X_trans = self.stationary_transformer.transform(X)
        
        # Merge back with targets and metadata
        cols_to_keep = [self.target_col]
        if self.weight_col:
            cols_to_keep.append(self.weight_col)
        if self.t1_col:
            cols_to_keep.append(self.t1_col)
            
        df_out = pd.concat([X_trans, df[cols_to_keep]], axis=1)
        
        # Drop rows where any of the newly transformed features are NaN (due to FFD/rolling window)
        df_out = df_out.dropna(subset=self.features)
        return df_out

    def compute_time_series_profiles(
        self, 
        df: pd.DataFrame, 
        columns: List[str], 
        groupby_level: Optional[str] = "ticker"
    ) -> pd.DataFrame:
        """
        Computes statistical descriptors (ADF stat, ACF1, spectral entropy) for each time series.
        """
        results = []
        
        # Helper function to compute descriptors for a single Series
        def _profile_series(series: pd.Series) -> Dict[str, float]:
            s_clean = series.dropna()
            if len(s_clean) < 10:
                return {"adf_stat": np.nan, "adf_p_value": np.nan, "acf1": np.nan, "spectral_entropy": np.nan}
                
            # ADF
            t_stat = _adf_test_stat(s_clean)
            p_val = _adf_p_value(t_stat)
            
            # ACF1
            acf1 = s_clean.autocorr(lag=1)
            
            # Spectral Entropy
            f, pxx = scipy.signal.periodogram(s_clean.values)
            pxx_norm = pxx / pxx.sum()
            # Add a small epsilon to avoid log(0)
            pxx_norm = pxx_norm[pxx_norm > 0]
            entropy = scipy.stats.entropy(pxx_norm)
            
            return {
                "adf_stat": t_stat,
                "adf_p_value": p_val,
                "acf1": acf1,
                "spectral_entropy": entropy
            }

        if groupby_level and isinstance(df.index, pd.MultiIndex):
            for col in columns:
                # Group by entity, compute profile, then average metrics across entities
                profiles = df[col].groupby(level=groupby_level).apply(_profile_series)
                
                # Convert list of dicts to DataFrame for easy median aggregation
                profiles_df = pd.DataFrame(profiles.tolist())
                median_profile = profiles_df.median().to_dict()
                median_profile['feature'] = col
                results.append(median_profile)
        else:
            for col in columns:
                profile = _profile_series(df[col])
                profile['feature'] = col
                results.append(profile)
                
        return pd.DataFrame(results).set_index('feature')

    def cluster_entities(
        self, 
        df: pd.DataFrame, 
        columns: List[str], 
        n_clusters: Optional[int] = None
    ) -> Dict[int, List[str]]:
        """
        Groups features hierarchically to neutralise the substitution effect.
        Uses distance metric: d = sqrt(0.5 * (1 - rho)).
        If n_clusters is None, determines ONIC using silhouette scores.
        """
        X = df[columns]
        
        # 1. Compute Pearson correlation matrix
        corr = X.corr(method='pearson')
        
        # 2. Convert to distance matrix
        dist_matrix = np.sqrt(0.5 * (1 - corr.clip(-1, 1)))
        
        # Extract condensed distance matrix required by linkage
        condensed_dist = scipy.spatial.distance.squareform(dist_matrix, checks=False)
        
        # 3. Hierarchical Linkage
        Z = scipy.cluster.hierarchy.linkage(condensed_dist, method='ward')
        
        # 4. Determine ONIC if n_clusters is None
        if n_clusters is None:
            best_score = -1.0
            best_n = 2
            max_clusters = max(2, len(columns) - 1)
            for k in range(2, max_clusters + 1):
                labels = scipy.cluster.hierarchy.fcluster(Z, k, criterion='maxclust')
                score = silhouette_score(dist_matrix, labels, metric='precomputed')
                if score > best_score:
                    best_score = score
                    best_n = k
            n_clusters = best_n
            
        # 5. Extract clusters
        labels = scipy.cluster.hierarchy.fcluster(Z, n_clusters, criterion='maxclust')
        
        clusters = {}
        for i, col in enumerate(columns):
            c_id = labels[i]
            if c_id not in clusters:
                clusters[c_id] = []
            clusters[c_id].append(col)
            
        return clusters

    def evaluate_importance(
        self,
        df: pd.DataFrame,
        estimator: BaseEstimator,
        metric: Callable,
        metric_kwargs: Optional[dict] = None,
        balance_classes: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Runs OOS Clustered MDA and Clustered SFI using the shared CV splits.
        """
        metric_kwargs = metric_kwargs or {}
        
        # 1. Group Features
        clusters = self.cluster_entities(df, self.features)
        
        X = df[self.features]
        y = df[self.target_col]
        
        mda_scores = {c_id: [] for c_id in clusters.keys()}
        sfi_scores = {c_id: [] for c_id in clusters.keys()}
        
        for step, (train_idx, val_idx) in enumerate(self.cv.split(df, y)):
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]
            
            fit_params = {}
            if self.weight_col and self.weight_col in df.columns:
                sample_weight = df[self.weight_col].iloc[train_idx].values.copy()
                
                if balance_classes:
                    from sklearn.utils.class_weight import compute_sample_weight
                    class_weights = compute_sample_weight("balanced", np.ravel(y_train))
                    sample_weight = sample_weight * class_weights
                
                if sample_weight.sum() > 0:
                    sample_weight = sample_weight / sample_weight.mean()
                
                if hasattr(estimator, "steps"):
                    final_step = estimator.steps[-1][0]
                    fit_params[f"{final_step}__sample_weight"] = sample_weight
                else:
                    fit_params["sample_weight"] = sample_weight

            # --- SFI (Single Feature Importance) ---
            for c_id, cols in clusters.items():
                est_sfi = clone(estimator)
                est_sfi.fit(X_train[cols], y_train, **fit_params)
                
                if metric.__name__ in ("log_loss", "roc_auc_score") and hasattr(est_sfi, "predict_proba"):
                    preds = est_sfi.predict_proba(X_val[cols])
                    if preds.ndim > 1 and preds.shape[1] == 2:
                        preds = preds[:, 1]
                else:
                    preds = est_sfi.predict(X_val[cols])
                    
                score = metric(y_val, preds, **metric_kwargs)
                sfi_scores[c_id].append(score)
                
            # --- MDA (Mean Decrease Accuracy) ---
            # Train model on ALL features
            est_mda = clone(estimator)
            est_mda.fit(X_train, y_train, **fit_params)
            
            # Baseline score
            if metric.__name__ in ("log_loss", "roc_auc_score") and hasattr(est_mda, "predict_proba"):
                base_preds = est_mda.predict_proba(X_val)
                if base_preds.ndim > 1 and base_preds.shape[1] == 2:
                    base_preds = base_preds[:, 1]
            else:
                base_preds = est_mda.predict(X_val)
            baseline_score = metric(y_val, base_preds, **metric_kwargs)
            
            # Perturb each cluster
            for c_id, cols in clusters.items():
                X_val_pert = X_val.copy()
                for col in cols:
                    np.random.shuffle(X_val_pert[col].values)
                    
                if metric.__name__ in ("log_loss", "roc_auc_score") and hasattr(est_mda, "predict_proba"):
                    pert_preds = est_mda.predict_proba(X_val_pert)
                    if pert_preds.ndim > 1 and pert_preds.shape[1] == 2:
                        pert_preds = pert_preds[:, 1]
                else:
                    pert_preds = est_mda.predict(X_val_pert)
                    
                pert_score = metric(y_val, pert_preds, **metric_kwargs)
                
                # MDA: Baseline performance - Perturbed performance
                # Note: If metric is an error (like log_loss), lower is better. 
                # So perturbation increases error. 
                # degradation = pert_score - baseline_score
                if metric.__name__ == "log_loss":
                    mda = pert_score - baseline_score
                else:
                    mda = baseline_score - pert_score
                mda_scores[c_id].append(mda)

        # Aggregate results
        results = {}
        
        sfi_agg = []
        for c_id, cols in clusters.items():
            sfi_agg.append({
                "cluster_id": c_id,
                "features": ", ".join(cols),
                "sfi_mean": np.mean(sfi_scores[c_id]),
                "sfi_std": np.std(sfi_scores[c_id])
            })
        results["SFI"] = pd.DataFrame(sfi_agg).set_index("cluster_id")
        
        mda_agg = []
        for c_id, cols in clusters.items():
            mda_agg.append({
                "cluster_id": c_id,
                "features": ", ".join(cols),
                "mda_mean": np.mean(mda_scores[c_id]),
                "mda_std": np.std(mda_scores[c_id])
            })
        results["MDA"] = pd.DataFrame(mda_agg).set_index("cluster_id")
        
        return results
