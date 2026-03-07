from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import pandas as pd
import numpy as np
from .features.sadf import get_sadf_jax as gsadf_values
from .labels.trend_scanning import trend_scanning
from .features.fractional_differentiation import frac_diff_ffd
from .labels.triple_barrier import apply_triple_barrier as triple_barrier_labels


class FractionalDiffTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn wrapper for Fractional Differentiation (FFD).
    Transforms a price series into a stationary series while preserving memory.
    """
    def __init__(self, d=0.4, thres=1e-3):
        self.d = d
        self.thres = thres

    def fit(self, X, y=None):
        """
        Stateless transformer, fit does nothing.
        """
        return self

    def transform(self, X):
        """
        Applies fractional differentiation.
        
        Args:
            X (pd.Series or pd.DataFrame): Time series data.
            
        Returns:
            pd.Series or pd.DataFrame: Fractionally differentiated data.
        """
        # Handle DataFrame (apply to specific column or single column DF)
        if isinstance(X, pd.DataFrame):
            if X.shape[1] == 1:
                series = X.iloc[:, 0]
            else:
                # If multi-column, apply to all (common use case for feature sets)
                # or raise error. We'll apply column-wise.
                return X.apply(lambda col: frac_diff_ffd(col, self.d, self.thres))
        else:
            series = X

        return frac_diff_ffd(series, self.d, self.thres)


class TrendScanningTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn wrapper for Trend Scanning.
    Transforms a price series into t-statistics indicating trend strength.
    """
    def __init__(self, windows=[5, 10, 15, 20, 30, 40, 60, 80, 100, 120, 150, 180]):
        self.windows = windows

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Applies trend scanning.
        
        Args:
            X (pd.Series or pd.DataFrame): Price series.
            
        Returns:
            pd.Series or pd.DataFrame: t-statistics.
        """
        if isinstance(X, pd.DataFrame):
            if X.shape[1] == 1:
                series = X.iloc[:, 0]
            else:
                # Apply column-wise
                return X.apply(lambda col: trend_scanning(col, self.windows))
        else:
            series = X

        return trend_scanning(series, self.windows)


class GSADFTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn wrapper for GSADF (Generalized SADF).
    Transforms a log-price series into GSADF statistics for bubble detection.
    """
    def __init__(self, model='linear', min_length=20, add_const=False, lags=1):
        self.model = model
        self.min_length = min_length
        self.add_const = add_const
        self.lags = lags

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Calculates GSADF statistics.
        
        Args:
            X (pd.Series or pd.DataFrame): Log-price series.
            
        Returns:
            pd.Series or pd.DataFrame: GSADF statistics.
        """
        if isinstance(X, pd.DataFrame):
            if X.shape[1] == 1:
                series = X.iloc[:, 0]
            else:
                # Apply column-wise
                return X.apply(lambda col: gsadf_values(
                    col, self.model, self.lags, self.min_length, self.add_const
                ))
        else:
            series = X

        return gsadf_values(series, self.model, self.lags, self.min_length, self.add_const)


class TripleBarrierLabeler(BaseEstimator, TransformerMixin):
    """
    Sklearn wrapper for Triple Barrier Labeling.
    
    NOTE: Unlike standard transformers, this requires specific column names 
    if X is a DataFrame (price vs stop loss).
    
    This is typically used to generate 'y' (targets) rather than transform 'X',
    but wrapping it allows for integration into data prep pipelines.
    """
    def __init__(self, price_col='close', sl_col='sl',
                 horizon=10, tp_mult=2.0):
        self.price_col = price_col
        self.sl_col = sl_col
        self.horizon = horizon
        self.tp_mult = tp_mult

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Generates labels.
        
        Args:
            X (pd.DataFrame):
                Must contain `price_col` and `sl_col`.
        
        Returns:
            pd.DataFrame: DataFrame containing `label` and `t1`.
        """
        if isinstance(X, pd.DataFrame):
            if self.price_col not in X.columns or self.sl_col not in X.columns:
                raise ValueError(f"X must contain {self.price_col} and {self.sl_col}")
            prices = X[self.price_col]
            sls = X[self.sl_col]
        else:
            raise ValueError("X must be a DataFrame containing price and stop loss columns.")
            
        return triple_barrier_labels(
            prices=prices,
            sl_col=sls,
            tp_mult=self.tp_mult,
            horizon=self.horizon
        )