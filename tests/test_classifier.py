import unittest
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from pyquantflow.data.database import DatabaseManager
from pyquantflow.model.classifier import (
    PrimarySecondaryClassifier,
    IchimokuBaselineClassifier,
)


class TestPrimarySecondaryClassifier(unittest.TestCase):
    """Unit tests for the PrimarySecondaryClassifier meta-modelling class."""

    def setUp(self):
        source_db_path = os.path.join(os.path.dirname(__file__), "stocks.db")
        self.ohlc_data = None

        if os.path.exists(source_db_path):
            try:
                db_manager = DatabaseManager(db_path=source_db_path)
                for ticker in ["FMG.AX", "CBA.AX"]:
                    df = db_manager.get_data(ticker)
                    if not df.empty and len(df) >= 150:
                        self.ohlc_data = df
                        break
                db_manager.conn.close()
            except Exception:
                pass

        if self.ohlc_data is None:
            self.ohlc_data = self.generate_synthetic_ohlc()

        # Construct realistic features and targets for meta-labelling testing
        df = self.ohlc_data.copy()
        df["feat_1"] = df["Close"].pct_change(5)
        df["feat_2"] = df["Close"].rolling(10).std()
        df["feat_3"] = df["Close"].rolling(20).mean() / df["Close"]
        df = df.dropna()

        # Create binary labels
        # Primary target: 1 if future 5-day return > 0, else 0
        df["y_primary"] = (df["Close"].shift(-5) > df["Close"]).astype(int)
        # Secondary target: 1 if return was positive AND absolute return was larger than 1%
        df["y_secondary"] = (df["Close"].shift(-5) > df["Close"] * 1.01).astype(int)
        df = df.dropna()

        self.X = df[["feat_1", "feat_2", "feat_3"]]
        self.y = df[["y_primary", "y_secondary"]]

    def generate_synthetic_ohlc(self, n=200, seed=42):
        """Generates synthetic OHLC data as fallback."""
        np.random.seed(seed)
        dates = pd.date_range(start="2023-01-01", periods=n, freq="D")
        returns = np.random.normal(0, 0.01, n)
        price_path = 100 * np.cumprod(1 + returns)
        high = price_path * (1 + np.abs(np.random.normal(0, 0.005, n)))
        low = price_path * (1 - np.abs(np.random.normal(0, 0.005, n)))
        open_ = (high + low) / 2 + np.random.normal(0, 0.002, n)
        high = np.maximum(high, np.maximum(open_, price_path))
        low = np.minimum(low, np.minimum(open_, price_path))
        volume = np.random.randint(1000, 100000, n)
        df = pd.DataFrame(
            {
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": price_path,
                "Volume": volume,
            },
            index=dates,
        )
        return df

    def test_instantiation_prefitted_true(self):
        """Test instantiation with prefitted=True attribute propagation."""
        primary = RandomForestClassifier(n_estimators=10, random_state=42)
        secondary = LogisticRegression(random_state=42)

        clf = PrimarySecondaryClassifier(
            primary_model=primary,
            secondary_model=secondary,
            primary_features=["feat_1", "feat_2"],
            secondary_features=["feat_2", "feat_3"],
            prefitted=True,
        )

        self.assertTrue(clf.prefitted)
        self.assertIs(clf.primary_model_, primary)
        self.assertIs(clf.secondary_model_, secondary)

    def test_instantiation_prefitted_false(self):
        """Test instantiation with prefitted=False marks attributes as None."""
        primary = RandomForestClassifier(n_estimators=10, random_state=42)
        secondary = LogisticRegression(random_state=42)

        clf = PrimarySecondaryClassifier(
            primary_model=primary,
            secondary_model=secondary,
            primary_features=["feat_1", "feat_2"],
            secondary_features=["feat_2", "feat_3"],
            prefitted=False,
        )

        self.assertFalse(clf.prefitted)
        self.assertIsNone(clf.primary_model_)
        self.assertIsNone(clf.secondary_model_)

    def test_calculate_entropy(self):
        """Test internal Shannon entropy computation on candidate probabilities."""
        clf = PrimarySecondaryClassifier(
            primary_model=None,
            secondary_model=None,
            primary_features=[],
            secondary_features=[],
        )

        # Test case 1: Uniform distribution (maximum uncertainty)
        # H = - (0.5 * ln(0.5) + 0.5 * ln(0.5)) = ln(2) ~ 0.693147
        probas = np.array([[0.5, 0.5]])
        entropy_val = clf._calculate_entropy(probas)
        self.assertEqual(entropy_val.shape, (1, 1))
        self.assertAlmostEqual(entropy_val[0, 0], np.log(2.0))

        # Test case 2: Certain distribution (zero uncertainty)
        probas_certain = np.array([[1.0, 0.0], [0.0, 1.0]])
        entropy_val_certain = clf._calculate_entropy(probas_certain)
        self.assertEqual(entropy_val_certain.shape, (2, 1))
        self.assertAlmostEqual(entropy_val_certain[0, 0], 0.0)
        self.assertAlmostEqual(entropy_val_certain[1, 0], 0.0)

    def test_fit_and_predict_workflows(self):
        """Test standard fit, predict, and predict_proba workflows."""
        clf = PrimarySecondaryClassifier(
            primary_model=RandomForestClassifier(n_estimators=10, random_state=42),
            secondary_model=LogisticRegression(random_state=42),
            primary_features=["feat_1", "feat_2"],
            secondary_features=["feat_2", "feat_3"],
            cv_generator=KFold(n_splits=3, shuffle=False),
            prefitted=False,
        )

        # Fit models
        clf.fit(self.X, self.y)
        self.assertFalse(clf.prefitted)
        self.assertIsNotNone(clf.primary_model_)
        self.assertIsNotNone(clf.secondary_model_)

        # Predict
        preds = clf.predict(self.X)
        self.assertIsInstance(preds, np.ndarray)
        self.assertEqual(len(preds), len(self.X))

        # Predict proba
        probas = clf.predict_proba(self.X)
        self.assertIsInstance(probas, np.ndarray)
        self.assertEqual(probas.shape, (len(self.X), 2))
        np.testing.assert_array_almost_equal(probas.sum(axis=1), np.ones(len(self.X)))

    def test_fit_with_sample_weight(self):
        """Test fitting workflow with sample_weight propagation."""
        clf = PrimarySecondaryClassifier(
            primary_model=RandomForestClassifier(n_estimators=10, random_state=42),
            secondary_model=LogisticRegression(random_state=42),
            primary_features=["feat_1", "feat_2"],
            secondary_features=["feat_2", "feat_3"],
            cv_generator=KFold(n_splits=3, shuffle=False),
            prefitted=False,
        )

        weights = np.ones(len(self.X))
        weights[:10] = 2.0

        # Fit with weights
        clf.fit(self.X, self.y, sample_weight=weights)

        preds = clf.predict(self.X)
        self.assertEqual(len(preds), len(self.X))

    def test_fit_numpy_target_arrays(self):
        """Test that fit method accepts targets as raw numpy arrays."""
        clf = PrimarySecondaryClassifier(
            primary_model=RandomForestClassifier(n_estimators=10, random_state=42),
            secondary_model=LogisticRegression(random_state=42),
            primary_features=["feat_1", "feat_2"],
            secondary_features=["feat_2", "feat_3"],
            cv_generator=KFold(n_splits=3, shuffle=False),
            prefitted=False,
        )

        y_arr = self.y.values
        clf.fit(self.X, y_arr)

        preds = clf.predict(self.X)
        self.assertEqual(len(preds), len(self.X))

    def test_transform_enrichment(self):
        """Test transformation workflow enriches dataset with model outputs."""
        clf = PrimarySecondaryClassifier(
            primary_model=RandomForestClassifier(n_estimators=10, random_state=42),
            secondary_model=LogisticRegression(random_state=42),
            primary_features=["feat_1", "feat_2"],
            secondary_features=["feat_2", "feat_3"],
            cv_generator=KFold(n_splits=3, shuffle=False),
            prefitted=False,
        )

        clf.fit(self.X, self.y)
        res = clf.transform(self.X)

        self.assertIsInstance(res, pd.DataFrame)
        self.assertEqual(len(res), len(self.X))

        expected_cols = [
            "primary_pred",
            "primary_proba",
            "primary_entropy",
            "secondary_proba",
            "final_decision",
        ]
        for col in expected_cols:
            self.assertIn(col, res.columns)


class TestIchimokuBaselineClassifier(unittest.TestCase):
    """Unit tests for the stateless IchimokuBaselineClassifier."""

    def _make_X(self, n=50, regime_values=None):
        """
        Creates a synthetic feature DataFrame with an ``ichimoku_regime`` column.
        """
        np.random.seed(42)
        if regime_values is None:
            regime_values = np.random.randint(0, 2, n)
        return pd.DataFrame(
            {
                "feat_a": np.random.randn(n),
                "feat_b": np.random.randn(n),
                "ichimoku_regime": regime_values,
            }
        )

    def test_fit_is_stateless_and_returns_self(self):
        """fit() must return self and set classes_ without altering state."""
        clf = IchimokuBaselineClassifier()
        X = self._make_X()
        y = np.random.randint(0, 2, len(X))

        result = clf.fit(X, y)

        self.assertIs(result, clf)
        np.testing.assert_array_equal(clf.classes_, np.array([0, 1]))

    def test_predict_extracts_regime_column(self):
        """predict() must return the ichimoku_regime column as integers."""
        regime = np.array([1, 0, 1, 1, 0])
        X = self._make_X(n=5, regime_values=regime)

        clf = IchimokuBaselineClassifier()
        clf.fit(X)
        preds = clf.predict(X)

        self.assertIsInstance(preds, np.ndarray)
        self.assertEqual(preds.dtype, int)
        np.testing.assert_array_equal(preds, regime)

    def test_predict_proba_shape_and_row_sums(self):
        """predict_proba() must return (N, 2) array with rows summing to 1."""
        regime = np.array([1, 0, 1, 0, 1])
        X = self._make_X(n=5, regime_values=regime)

        clf = IchimokuBaselineClassifier()
        clf.fit(X)
        probas = clf.predict_proba(X)

        self.assertEqual(probas.shape, (5, 2))
        np.testing.assert_array_almost_equal(probas.sum(axis=1), np.ones(5))

        # Class 1 probability should match the regime signal exactly
        np.testing.assert_array_equal(probas[:, 1], regime.astype(float))

    def test_predict_raises_on_missing_column(self):
        """predict() must raise KeyError when regime column is absent."""
        X = pd.DataFrame({"feat_a": [1.0, 2.0], "feat_b": [3.0, 4.0]})
        clf = IchimokuBaselineClassifier()
        clf.fit(X)

        with self.assertRaises(KeyError):
            clf.predict(X)

    def test_custom_regime_col_name(self):
        """Classifier honours a custom regime_col parameter."""
        X = pd.DataFrame({"feat": [1.0, 2.0, 3.0], "my_regime": [0, 1, 0]})
        clf = IchimokuBaselineClassifier(regime_col="my_regime")
        clf.fit(X)
        preds = clf.predict(X)

        np.testing.assert_array_equal(preds, np.array([0, 1, 0]))


if __name__ == "__main__":
    unittest.main()
