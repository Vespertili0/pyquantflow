from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import pandas as pd
import numpy as np
from .sadf import gsadf_values
from .trend_scanning import trend_scanning
from .fractional_differentiation import frac_diff_ffd
from .triple_barrier import triple_barrier_labels


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
    def __init__(self, min_length=None, add_trend=False, lags=1):
        self.min_length = min_length
        self.add_trend = add_trend
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
                    col, self.min_length, self.add_trend, self.lags
                ))
        else:
            series = X

        return gsadf_values(series, self.min_length, self.add_trend, self.lags)


class TripleBarrierLabeler(BaseEstimator, TransformerMixin):
    """
    Sklearn wrapper for Triple Barrier Labeling.
    
    NOTE: Unlike standard transformers, this requires specific column names 
    if X is a DataFrame (price vs volatility), or assumes fixed barriers if 
    volatility is missing.
    
    This is typically used to generate 'y' (targets) rather than transform 'X',
    but wrapping it allows for integration into data prep pipelines.
    """
    def __init__(self, price_col='close', vol_col=None, 
                 vertical_barrier_steps=10, pt=1.0, sl=1.0):
        self.price_col = price_col
        self.vol_col = vol_col
        self.vertical_barrier_steps = vertical_barrier_steps
        self.pt = pt
        self.sl = sl

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Generates labels.
        
        Args:
            X (pd.DataFrame or pd.Series): 
                - If DataFrame: Must contain `price_col`. If `vol_col` is set,
                  must also contain `vol_col` for dynamic barriers.
                - If Series: Treated as price. Volatility is assumed None (fixed barriers).
        
        Returns:
            pd.Series: Labels (-1, 0, 1).
        """
        volatility = None
        
        if isinstance(X, pd.DataFrame):
            # Extract Price
            if self.price_col not in X.columns:
                # Fallback: try first column
                prices = X.iloc[:, 0]
            else:
                prices = X[self.price_col]
            
            # Extract Volatility if specified
            if self.vol_col is not None:
                if self.vol_col in X.columns:
                    volatility = X[self.vol_col]
                else:
                    raise ValueError(f"Volatility column '{self.vol_col}' not found in X.")
        else:
            # Assume Series is Price
            prices = X
            # Volatility remains None
            
        return triple_barrier_labels(
            price_series=prices,
            volatility=volatility,
            vertical_barrier_steps=self.vertical_barrier_steps,
            pt=self.pt,
            sl=self.sl
        )