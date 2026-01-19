import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator



class PurgedKFoldCV(BaseCrossValidator):
    """
    sklearn-compatible Purged K-Fold cross-validator with embargo.
    Supports single-asset and multi-asset datasets.
    Works with triple-barrier t1 labels.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    t1 : pd.Series
        Event end times (same index as X).
    embargo_pct : float
        Fraction of the dataset to embargo after each test fold.
    datetime_level : str or int
        Name or index of the datetime level in a MultiIndex.
    """

    def __init__(self, n_splits=5, t1=None, embargo_pct=0.01, datetime_level="datetime"):
        self.n_splits = n_splits
        self.t1 = t1
        self.embargo_pct = embargo_pct
        self.datetime_level = datetime_level

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def _extract_times(self, X):
        """Extract datetime index from X, supporting MultiIndex."""
        if hasattr(X, "index"):
            idx = X.index
        elif self.t1 is not None:
            if len(X) != len(self.t1):
                raise ValueError("X (numpy) and t1 must have same length.")
            idx = self.t1.index
        else:
            raise ValueError("X must be a pandas DataFrame/Series or t1 must be provided to infer times.")

        if isinstance(idx, pd.MultiIndex):
            # Allow both name and integer level
            if isinstance(self.datetime_level, str):
                return idx.get_level_values(self.datetime_level)
            else:
                return idx.get_level_values(self.datetime_level)
        else:
            return idx

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test sets.
        """

        # Extract datetime index
        times = self._extract_times(X)

        # Ensure sorted by time (critical for multi-asset MultiIndex)
        order = np.argsort(times.values)
        if hasattr(X, "iloc"):
            X = X.iloc[order]
        else:
            X = X[order]
        times = times[order]

        # Unique sorted timestamps
        unique_times = np.sort(times.unique())
        n_unique = len(unique_times)

        fold_size = n_unique // self.n_splits
        embargo = int(n_unique * self.embargo_pct)

        # Convert t1 to aligned DatetimeIndex
        if self.t1 is not None:
            # t1 should be indexed by original time index (bar index)
            # times contains the bar index, sorted.
            # So t1.loc[times] gives the t1 values in sorted order.
            t1_times = pd.DatetimeIndex(self.t1.loc[times])
        else:
            t1_times = pd.DatetimeIndex([pd.NaT] * len(X))

        for k in range(self.n_splits):

            # Test fold time window
            test_start = k * fold_size
            test_end = (k + 1) * fold_size if k < self.n_splits - 1 else n_unique

            test_times = unique_times[test_start:test_end]

            # Embargo window
            embargo_start = test_end
            embargo_end = min(test_end + embargo, n_unique)
            embargo_times = unique_times[embargo_start:embargo_end]

            # Masks
            test_mask = times.isin(test_times)
            embargo_mask = times.isin(embargo_times)

            # Purge: remove training samples whose t1 overlaps test window
            test_min = test_times.min()
            test_max = test_times.max()

            purge_mask = (t1_times >= test_min) & (t1_times <= test_max)

            # Final training mask
            train_mask = ~(test_mask | embargo_mask | purge_mask)

            yield np.where(train_mask)[0], np.where(test_mask)[0]
