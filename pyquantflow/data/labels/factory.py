import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union

from .sample_weights import get_sample_weights
from .triple_barrier import apply_triple_barrier
from .trend_scanning import trend_scanning


class BaseLabelFactory(ABC):
    """
    Abstract Base Class defining the contract for all labelling and
    sample weighting factories.
    """

    @abstractmethod
    def generate_labels(
        self, ticker_df: pd.DataFrame, price_col: str = "Close"
    ) -> pd.DataFrame:
        """
        Generates labels and resolution timestamps (t1) from a ticker's DataFrame.

        Parameters
        ----------
        ticker_df : pd.DataFrame
            DataFrame containing the price data and any pre-computed features.
            The index must be a DatetimeIndex.
        price_col : str, default='close'
            The column name representing the price series.

        Returns
        -------
        pd.DataFrame
            DataFrame containing at least:
            - 'label': The categorical or continuous label.
            - 't1': The timestamp when the label was resolved.
        """
        pass

    @abstractmethod
    def generate_weights(self, t1: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Calculates sample uniqueness or concurrency-adjusted weights.

        Parameters
        ----------
        t1 : pd.Series
            Resolution timestamps (index must be DatetimeIndex).
        returns : pd.Series
            Returns corresponding to the entry timestamps.

        Returns
        -------
        pd.Series
            Series of sample weights aligned with the input index.
        """
        pass


class TripleBarrierLabelFactory(BaseLabelFactory):
    """
    Concrete factory implementing the Triple Barrier labelling method.
    """

    def __init__(
        self,
        pt_mult: float = 1.0,
        sl_mult: float = 1.0,
        horizon: int = 10,
        vol_span: int = 100,
    ):
        """
        Initialises the TripleBarrierLabelFactory.

        Args:
            pt_mult (float): Multiplier for the take-profit barrier.
            sl_mult (float): Multiplier for the stop-loss barrier.
            horizon (int): The maximum number of bars to hold before timeout (vertical barrier).
            vol_span (int): The EWMA span used to calculate dynamic volatility if `sl_col`
                            is not pre-computed in the DataFrame.
        """
        self.pt_mult = pt_mult
        self.sl_mult = sl_mult
        self.horizon = horizon
        self.vol_span = vol_span

    def generate_labels(
        self, ticker_df: pd.DataFrame, price_col: str = "Close"
    ) -> pd.DataFrame:
        prices = ticker_df[price_col]

        # Determine Stop-Loss dynamically based on volatility
        vol = prices.pct_change().ewm(span=self.vol_span).std()
        sl_col = prices - prices * vol * self.sl_mult

        labels_df = apply_triple_barrier(
            prices=prices, sl_col=sl_col, tp_mult=self.pt_mult, horizon=self.horizon
        )
        return labels_df

    def generate_weights(self, t1: pd.Series, returns: pd.Series) -> pd.Series:
        return get_sample_weights(t1=t1, returns=returns)


class TrendScanningLabelFactory(BaseLabelFactory):
    """
    Concrete factory implementing the Trend Scanning labelling method.
    """

    def __init__(
        self,
        windows: Union[List[int], int] = [5, 10, 20, 40, 80, 120],
        bins: Union[List[float], np.ndarray] = [-10.0, 12.0],
    ):
        """
        Initialises the TrendScanningLabelFactory.

        Args:
            windows (list | int): Look-forward window sizes to scan.
            bins (list | np.ndarray): Bin boundaries for np.digitize to categorise trends.
        """
        if isinstance(windows, int):
            self.windows = [windows]
        else:
            self.windows = windows
        self.bins = bins

    def generate_labels(
        self, ticker_df: pd.DataFrame, price_col: str = "Close"
    ) -> pd.DataFrame:
        prices = ticker_df[price_col]

        labels_df = trend_scanning(series=prices, windows=self.windows)

        # Categorise 't_value' into 'label' for standardisation while retaining 't_value'
        if "t_value" in labels_df.columns:
            t_vals = labels_df["t_value"]
            valid = t_vals.notna()
            labels_df["label"] = np.nan
            if valid.any():
                labels_df.loc[valid, "label"] = np.digitize(
                    t_vals.loc[valid].values, self.bins
                ).astype(float)

        return labels_df

    def generate_weights(self, t1: pd.Series, returns: pd.Series) -> pd.Series:
        return get_sample_weights(t1=t1, returns=returns)
