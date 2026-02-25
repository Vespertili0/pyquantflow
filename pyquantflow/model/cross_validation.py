import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from itertools import combinations


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
        idx = X.index

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
        X = X.iloc[order]
        times = times[order]

        # Unique sorted timestamps
        unique_times = np.sort(times.unique())
        n_unique = len(unique_times)

        fold_size = n_unique // self.n_splits
        embargo = int(n_unique * self.embargo_pct)

        # Convert t1 to aligned DatetimeIndex
        if self.t1 is not None:
            t1_times = pd.DatetimeIndex(self.t1.loc[X.index])
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


class CombinatorialPurgedKFold(BaseCrossValidator):
    def __init__(self, n_splits=5, n_test_splits=2, purge_limit=0, embargo_limit=0):
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_limit = purge_limit
        self.embargo_limit = embargo_limit

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # 1. Split indices into N blocks
        block_size = n_samples // self.n_splits
        block_bounds = [(i * block_size, (i + 1) * block_size) for i in range(self.n_splits)]
        block_bounds[-1] = (block_bounds[-1][0], n_samples) # Ensure last block covers remainder
        
        # 2. Generate all combinations of k test blocks
        all_block_indices = list(range(self.n_splits))
        for test_blocks in combinations(all_block_indices, self.n_test_splits):
            test_indices = []
            train_indices = []
            
            # Sort test blocks to handle purging/embargoing logically
            test_blocks = sorted(test_blocks)
            train_blocks = [i for i in all_block_indices if i not in test_blocks]

            # Construct Test Set
            for i in test_blocks:
                start, end = block_bounds[i]
                test_indices.extend(indices[start:end])

            # Construct Train Set with Purging & Embargoing
            for i in train_blocks:
                start, end = block_bounds[i]
                
                # Check for overlap with any test block
                for j in test_blocks:
                    test_start, test_end = block_bounds[j]
                    
                    # If train block is immediately before a test block, PURGE the end
                    if end > test_start and start < test_start:
                        end = test_start - self.purge_limit
                    
                    # If train block is immediately after a test block, EMBARGO the start
                    if start < test_end and end > test_end:
                        start = test_end + self.embargo_limit
                
                if start < end:
                    train_indices.extend(indices[start:end])

            yield np.array(train_indices), np.array(test_indices)

    def get_n_splits(self, X=None, y=None, groups=None):
        from math import comb
        return comb(self.n_splits, self.n_test_splits)