from typing import Optional, Union, Generator, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator



class PurgedKFoldCV(BaseCrossValidator):
    """
    sklearn-compatible Purged K-Fold cross-validator with embargo.

    This cross validator splits time-series data while properly addressing leakage
    by purging observations in the training set whose evaluation period overlaps 
    with the test set, and optionally embargoing periods after the test set.

    Supports single-asset and multi-asset datasets (via MultiIndex).
    Works with triple-barrier event end times (`t1`). Can accept `t1` as an 
    external `pd.Series` or dynamically extract it from the input features DataFrame 
    `X` if a column name string is provided.
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    t1 : pd.Series or str, optional
        Event end times (should have the same index as X). 
        If a string is passed, it is assumed to be the column name in `X`
        containing the event end times. If None, purging is completely disabled
        but embargoing still applies.
    embargo_pct : float, default=0.01
        Fraction of the dataset's unique times to embargo after each test fold
        to prevent subsequent train fold leakage.
    datetime_level : str or int, default="datetime"
        Name or index of the datetime level in a MultiIndex. Ignored if `X` 
        has a standard DatetimeIndex.
    """
    def __init__(self, 
                 n_splits: int = 5, 
                 t1: Optional[Union[pd.Series, str]] = None, 
                 embargo_pct: float = 0.01, 
                 datetime_level: Union[str, int] = "datetime") -> None:
        self.n_splits = n_splits
        self.t1 = t1
        self.embargo_pct = embargo_pct
        self.datetime_level = datetime_level

    def get_n_splits(self, 
                     X: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None, 
                     y: Optional[Union[pd.Series, np.ndarray]] = None, 
                     groups: Optional[np.ndarray] = None) -> int:
        """
        Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits

    def _extract_times(self, X: pd.DataFrame) -> pd.Index:
        """
        Extract the datetime index from X, supporting standard DatetimeIndex 
        and MultiIndex architectures.
        """
        idx = X.index
        if isinstance(idx, pd.MultiIndex):
            return idx.get_level_values(self.datetime_level)
        return idx

    def split(self, 
              X: pd.DataFrame, 
              y: Optional[Union[pd.Series, np.ndarray]] = None, 
              groups: Optional[np.ndarray] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into training and test sets.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training data containing the features. The index must either be 
            a chronological DatetimeIndex or a MultiIndex with a datetime level.
        y : pd.Series or np.ndarray, optional
            The target variable for supervised learning problems.
        groups : np.ndarray, optional
            Group labels for the samples used while splitting the dataset into
            train/test set. Not used by this spliter, maintained for compatibility.
            
        Yields
        ------
        train : np.ndarray
            The training set indices for that split.
        test : np.ndarray
            The testing set indices for that split.
        """
        # 1. Extract times as a Series to safely execute vectorized logic
        times = pd.Series(self._extract_times(X))
        
        # 2. Get chronological boundaries (No need to sort X itself!)
        unique_times = np.sort(times.unique())
        n_unique = len(unique_times)
        
        fold_size = n_unique // self.n_splits
        embargo = int(n_unique * self.embargo_pct)

        # 3. Safely align t1 to X's index
        if self.t1 is not None:
            if isinstance(self.t1, str):
                t1_series = X[self.t1]
            else:
                t1_series = self.t1
            # .reindex ensures perfect alignment even if X has duplicate indices or is unsorted
            t1_times = pd.Series(t1_series).reindex(X.index).values
            t1_times = pd.to_datetime(t1_times)
        else:
            t1_times = pd.Series([pd.NaT] * len(X)).values

        for k in range(self.n_splits):
            # Define time windows
            test_start = k * fold_size
            test_end = (k + 1) * fold_size if k < self.n_splits - 1 else n_unique
            
            test_times = unique_times[test_start:test_end]
            test_min = test_times.min()
            test_max = test_times.max()

            embargo_start = test_end
            embargo_end = min(test_end + embargo, n_unique)
            embargo_times = unique_times[embargo_start:embargo_end]

            # Generate Masks
            test_mask = times.isin(test_times).values
            embargo_mask = times.isin(embargo_times).values

            # CORRECTED PURGE MASK: 
            # Drop if sample started before test ends AND expires after test begins
            if self.t1 is not None:
                purge_mask = (times <= test_max) & (t1_times >= test_min)
            else:
                purge_mask = np.zeros(len(X), dtype=bool)

            # Final training mask: Must not be test, embargoed, or purged
            train_mask = ~(test_mask | embargo_mask | purge_mask)

            yield np.where(train_mask)[0], np.where(test_mask)[0]