import unittest
from math import comb

import numpy as np
import pandas as pd
from pyquantflow.model.cross_validation import PurgedKFoldCV, CombinatorialPurgedKFold


class TestPurgedKFoldCV(unittest.TestCase):
    def test_get_n_splits_default(self):
        """Test get_n_splits with default initialization parameters."""
        cv = PurgedKFoldCV()
        self.assertEqual(cv.get_n_splits(), 5)

    def test_get_n_splits_custom(self):
        """Test get_n_splits with custom initialization parameter."""
        cv = PurgedKFoldCV(n_splits=10)
        self.assertEqual(cv.get_n_splits(), 10)

    def test_get_n_splits_with_arguments(self):
        """Test get_n_splits when passing arguments like X, y, groups."""
        cv = PurgedKFoldCV(n_splits=3)
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([1, 0, 1])
        groups = np.array([1, 2, 3])
        self.assertEqual(cv.get_n_splits(X=X, y=y, groups=groups), 3)


# ---------------------------------------------------------------------------
# CombinatorialPurgedKFold — temporal slicing tests
# ---------------------------------------------------------------------------


def _build_panel(n_days=100, tickers=("AAA", "BBB", "CCC")):
    """Build a multi-asset panel DataFrame with (datetime, ticker) MultiIndex."""
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    rows = []
    for dt in dates:
        for tk in tickers:
            rows.append({"datetime": dt, "ticker": tk, "feature": np.random.randn()})
    df = pd.DataFrame(rows).set_index(["datetime", "ticker"]).sort_index()
    return df


class TestCombinatorialPurgedKFoldTemporal(unittest.TestCase):
    """Comprehensive tests for temporal CombinatorialPurgedKFold."""

    def setUp(self):
        self.tickers = ("AAA", "BBB", "CCC")
        self.n_days = 100
        self.panel = _build_panel(n_days=self.n_days, tickers=self.tickers)

    # --- Core: no datetime leakage ---

    def test_no_datetime_leakage(self):
        """No single datetime must exist in both train and test for any split."""
        cv = CombinatorialPurgedKFold(n_splits=5, n_test_splits=2)

        for train_idx, test_idx in cv.split(self.panel):
            train_times = set(self.panel.index[train_idx].get_level_values("datetime"))
            test_times = set(self.panel.index[test_idx].get_level_values("datetime"))
            self.assertTrue(
                train_times.isdisjoint(test_times),
                "Found datetime in both train and test sets.",
            )

    def test_all_tickers_kept_together(self):
        """Every datetime in the test set should have all tickers present."""
        cv = CombinatorialPurgedKFold(n_splits=5, n_test_splits=2)
        n_tickers = len(self.tickers)

        for train_idx, test_idx in cv.split(self.panel):
            test_df = self.panel.iloc[test_idx]
            counts = test_df.groupby(level="datetime").size()
            self.assertTrue(
                (counts == n_tickers).all(),
                f"Not all tickers present for some test dates: {counts[counts != n_tickers]}",
            )

    # --- Purge window ---

    def test_purge_window_correctness(self):
        """With purge_limit=2, the 2 unique datetimes before each test block
        must be excluded from training across all tickers."""
        purge_limit = 2
        cv = CombinatorialPurgedKFold(
            n_splits=5, n_test_splits=2, purge_limit=purge_limit
        )

        unique_times = np.sort(self.panel.index.get_level_values("datetime").unique())
        n_unique = len(unique_times)
        block_size = n_unique // 5

        for split_idx, (train_idx, test_idx) in enumerate(cv.split(self.panel)):
            train_times = set(self.panel.index[train_idx].get_level_values("datetime"))
            test_times = sorted(
                set(self.panel.index[test_idx].get_level_values("datetime"))
            )

            # Find block start indices for each contiguous test block
            # by locating positions in unique_times
            test_time_set = set(test_times)
            for i in range(5):
                start_idx = i * block_size
                block_start_dt = unique_times[start_idx]
                if block_start_dt in test_time_set:
                    # This block is a test block — check purge window
                    purge_start = max(0, start_idx - purge_limit)
                    for dt_idx in range(purge_start, start_idx):
                        purge_dt = unique_times[dt_idx]
                        if purge_dt not in test_time_set:
                            self.assertNotIn(
                                purge_dt,
                                train_times,
                                f"Split {split_idx}: purged datetime {purge_dt} "
                                f"found in training set.",
                            )

    # --- Embargo window ---

    def test_embargo_window_correctness(self):
        """With embargo_limit=3, the 3 unique datetimes after each test block
        must be excluded from training."""
        embargo_limit = 3
        cv = CombinatorialPurgedKFold(
            n_splits=5, n_test_splits=2, embargo_limit=embargo_limit
        )

        unique_times = np.sort(self.panel.index.get_level_values("datetime").unique())
        n_unique = len(unique_times)
        block_size = n_unique // 5

        for split_idx, (train_idx, test_idx) in enumerate(cv.split(self.panel)):
            train_times = set(self.panel.index[train_idx].get_level_values("datetime"))
            test_times = set(self.panel.index[test_idx].get_level_values("datetime"))

            for i in range(5):
                start_idx = i * block_size
                end_idx = (i + 1) * block_size if i < 4 else n_unique

                # Check if this block is a test block
                block_start_dt = unique_times[start_idx]
                if block_start_dt in test_times:
                    # Embargo: datetimes immediately AFTER the block
                    embargo_end = min(n_unique, end_idx + embargo_limit)
                    for dt_idx in range(end_idx, embargo_end):
                        embargo_dt = unique_times[dt_idx]
                        if embargo_dt not in test_times:
                            self.assertNotIn(
                                embargo_dt,
                                train_times,
                                f"Split {split_idx}: embargoed datetime "
                                f"{embargo_dt} found in training set.",
                            )

    # --- Combined purge + embargo row count ---

    def test_combined_purge_embargo_row_exclusion(self):
        """Purge+embargo should exclude the correct number of rows (accounting
        for panel width = number of tickers)."""
        purge_limit = 2
        embargo_limit = 1
        n_tickers = len(self.tickers)

        cv_no_excl = CombinatorialPurgedKFold(
            n_splits=5, n_test_splits=2, purge_limit=0, embargo_limit=0
        )
        cv_with_excl = CombinatorialPurgedKFold(
            n_splits=5,
            n_test_splits=2,
            purge_limit=purge_limit,
            embargo_limit=embargo_limit,
        )

        for (train_no, _), (train_yes, _) in zip(
            cv_no_excl.split(self.panel), cv_with_excl.split(self.panel)
        ):
            rows_removed = len(train_no) - len(train_yes)
            # Each purge/embargo datetime removes n_tickers rows, but only if
            # those datetimes are not already in the test set
            self.assertTrue(
                rows_removed >= 0,
                "Purge+embargo should never increase training rows.",
            )
            # At a minimum, at least some rows should be removed
            # (unless all purge/embargo fall inside test blocks already)
            if rows_removed > 0:
                self.assertEqual(
                    rows_removed % n_tickers,
                    0,
                    "Row removal should be a multiple of n_tickers.",
                )

    # --- get_n_splits ---

    def test_get_n_splits(self):
        """Verify the combinatorial count is correct."""
        cv = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2)
        self.assertEqual(cv.get_n_splits(), comb(6, 2))

        cv2 = CombinatorialPurgedKFold(n_splits=5, n_test_splits=3)
        self.assertEqual(cv2.get_n_splits(), comb(5, 3))

    def test_actual_splits_count_matches(self):
        """Number of yielded splits must equal get_n_splits."""
        cv = CombinatorialPurgedKFold(n_splits=5, n_test_splits=2)
        splits = list(cv.split(self.panel))
        self.assertEqual(len(splits), cv.get_n_splits())

    # --- Single-asset (non-MultiIndex) ---

    def test_single_asset_datetime_index(self):
        """Works correctly with a plain DatetimeIndex (single asset)."""
        dates = pd.date_range("2020-01-01", periods=50, freq="B")
        df = pd.DataFrame({"feature": np.random.randn(50)}, index=dates)
        df.index.name = "datetime"

        cv = CombinatorialPurgedKFold(
            n_splits=5, n_test_splits=2, purge_limit=1, embargo_limit=1
        )

        for train_idx, test_idx in cv.split(df):
            train_times = set(df.index[train_idx])
            test_times = set(df.index[test_idx])
            self.assertTrue(train_times.isdisjoint(test_times))

    # --- Edge case: purge/embargo overlap with adjacent test blocks ---

    def test_adjacent_test_blocks_purge_overlap(self):
        """When two test blocks are adjacent, purge/embargo datetimes that fall
        inside the other test block should not cause issues."""
        cv = CombinatorialPurgedKFold(
            n_splits=4, n_test_splits=2, purge_limit=5, embargo_limit=5
        )

        for train_idx, test_idx in cv.split(self.panel):
            train_times = set(self.panel.index[train_idx].get_level_values("datetime"))
            test_times = set(self.panel.index[test_idx].get_level_values("datetime"))
            self.assertTrue(train_times.isdisjoint(test_times))


if __name__ == "__main__":
    unittest.main()
