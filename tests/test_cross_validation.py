import unittest
import numpy as np
import pandas as pd
from pyquantflow.model.cross_validation import PurgedKFoldCV


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


if __name__ == "__main__":
    unittest.main()
