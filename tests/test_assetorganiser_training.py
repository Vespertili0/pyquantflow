import unittest
import pandas as pd
import numpy as np
from pyquantflow.data.assetorganiser import AssetOrganiser
from pyquantflow.model.manager import ClassifierEngine
from pyquantflow.model.training import HyperparameterOptimiser
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


class MockEstimator:
    """A mock estimator to catch fit parameters."""

    def __init__(self):
        self.fit_called = False
        self.sample_weights_received = None

    def fit(self, X, y, sample_weight=None):
        self.fit_called = True
        self.sample_weights_received = sample_weight
        return self

    def predict(self, X):
        return np.zeros(len(X))

    def predict_proba(self, X):
        return np.ones((len(X), 2)) * 0.5


class TestDataHierarchyIntegration(unittest.TestCase):
    def setUp(self):
        # Create dummy multi-asset data
        dates = pd.date_range("2020-01-01", periods=10)
        df_a = pd.DataFrame(
            {
                "feature1": np.random.randn(10),
                "target": np.random.randint(0, 2, 10),
                "weight": np.random.uniform(0.5, 1.5, 10),
            },
            index=dates,
        )
        df_a.index.name = "datetime"

        df_b = pd.DataFrame(
            {
                "feature1": np.random.randn(10),
                "target": np.random.randint(0, 2, 10),
                "weight": np.random.uniform(0.5, 1.5, 10),
            },
            index=dates,
        )
        df_b.index.name = "datetime"

        self.data_map = {"AAA": df_a, "BBB": df_b}

    def test_pipeline_integration_with_weights(self):
        # 1. Asset Organiser
        organiser = AssetOrganiser(
            data_map=self.data_map,
            cutoff_date="2020-01-08",
            target_features=["target"],
            weight_col="weight",
        )
        organiser.prepare_multi_asset_frame()
        payload = organiser.get_classifierengine_payload(features=["feature1"])

        # Verify payload structure
        self.assertIn("feature1", payload["features"])
        self.assertNotIn("weight", payload["features"])
        self.assertNotIn("target", payload["features"])
        self.assertEqual(payload["weight_col"], "weight")

        # 2. Setup Mock Optimiser & Engine
        optimiser = HyperparameterOptimiser(
            study_name="test_study", direction="maximize"
        )
        engine = ClassifierEngine(optimiser=optimiser)

        # We'll use a standard sklearn estimator to test pipeline extraction
        mock_model = DecisionTreeClassifier()
        pipe = Pipeline([("mock_tree", mock_model)])

        def mock_factory(trial):
            return pipe

        cv = KFold(n_splits=2)

        # 3. Run Pipeline
        engine.run_pipeline(
            **payload,
            model_factory=mock_factory,
            cv=cv,
            n_trials=2,  # Very small optuna run
            balance_classes=True,
        )

        # 4. Verify that the final retrained model received the weights
        final_pipe = engine.best_estimator_
        self.assertIsNotNone(final_pipe)

        # We can't easily intercept the internal optuna loops without breaking encapsulation,
        # but we can verify the final fit step correctly stripped features and targets
        # and would have attempted to pass sample_weights.
        # (Since we used a real DecisionTree, it will raise an error internally if sample_weights
        # were misaligned in length, proving the extraction math works).

        self.assertTrue(hasattr(final_pipe.steps[-1][1], "classes_"))


class TestAssetOrganiserFlexibility(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2020-01-01", periods=10)
        df_a = pd.DataFrame(
            {
                "feature1": np.random.randn(10),
                "target": np.random.randint(0, 2, 10),
            },
            index=dates,
        )
        df_a.index.name = "datetime"
        self.data_map = {"AAA": df_a}

        # Create multi_asset format DataFrame manually
        self.multi_asset = df_a.copy()
        self.multi_asset["ticker"] = "AAA"
        self.multi_asset = self.multi_asset.reset_index().set_index(
            ["datetime", "ticker"]
        )

    def test_both_provided_raises_value_error(self):
        with self.assertRaises(ValueError):
            AssetOrganiser(
                data_map=self.data_map,
                multi_asset=self.multi_asset,
                cutoff_date="2020-01-08",
                target_features=["target"],
            )

    def test_neither_provided_raises_value_error(self):
        with self.assertRaises(ValueError):
            AssetOrganiser(
                cutoff_date="2020-01-08",
                target_features=["target"],
            )

    def test_missing_cutoff_date_raises_value_error(self):
        with self.assertRaises(ValueError):
            AssetOrganiser(
                data_map=self.data_map,
                target_features=["target"],
            )

    def test_missing_target_features_raises_value_error(self):
        with self.assertRaises(ValueError):
            AssetOrganiser(
                data_map=self.data_map,
                cutoff_date="2020-01-08",
            )

    def test_multi_asset_split_immediately(self):
        organiser = AssetOrganiser(
            multi_asset=self.multi_asset,
            cutoff_date="2020-01-08",
            target_features=["target"],
        )
        # Should be split immediately in __init__
        self.assertIsNotNone(organiser.multi_asset_train)
        self.assertIsNotNone(organiser.multi_asset_test)
        self.assertEqual(len(organiser.multi_asset_train), 7)
        self.assertEqual(len(organiser.multi_asset_test), 3)

    def test_get_classifierengine_payload_with_tickers_filter(self):
        # Create multiple tickers to verify filtering works
        dates = pd.date_range("2020-01-01", periods=10)
        df_a = pd.DataFrame(
            {
                "feature1": np.random.randn(10),
                "target": np.random.randint(0, 2, 10),
            },
            index=dates,
        )
        df_a.index.name = "datetime"

        df_b = pd.DataFrame(
            {
                "feature1": np.random.randn(10),
                "target": np.random.randint(0, 2, 10),
            },
            index=dates,
        )
        df_b.index.name = "datetime"

        data_map = {"AAA": df_a, "BBB": df_b}

        organiser = AssetOrganiser(
            data_map=data_map,
            cutoff_date="2020-01-08",
            target_features=["target"],
        )
        organiser.prepare_multi_asset_frame()

        # Test default behaviour: return full multi-asset dataframe (both AAA and BBB)
        payload_default = organiser.get_classifierengine_payload(features=["feature1"])
        tickers_in_train_default = (
            payload_default["X_train"].index.get_level_values("ticker").unique()
        )
        self.assertEqual(set(tickers_in_train_default), {"AAA", "BBB"})

        # Test filtering behaviour with a list of tickers: return only AAA
        payload_filtered = organiser.get_classifierengine_payload(
            features=["feature1"], tickers=["AAA"]
        )
        tickers_in_train_filtered = (
            payload_filtered["X_train"].index.get_level_values("ticker").unique()
        )
        tickers_in_test_filtered = (
            payload_filtered["X_test"].index.get_level_values("ticker").unique()
        )
        self.assertEqual(set(tickers_in_train_filtered), {"AAA"})
        self.assertEqual(set(tickers_in_test_filtered), {"AAA"})

        # Ensure y_train and y_test are also filtered (containing only AAA ticker indices)
        self.assertEqual(
            set(payload_filtered["y_train"].index.get_level_values("ticker").unique()),
            {"AAA"},
        )
        self.assertEqual(
            set(payload_filtered["y_test"].index.get_level_values("ticker").unique()),
            {"AAA"},
        )

    def test_to_tsfeatures_format(self):
        organiser = AssetOrganiser(
            multi_asset=self.multi_asset,
            cutoff_date="2020-01-08",
            target_features=["target"],
        )

        # Test exporting entire dataset
        df_all = organiser.to_tsfeatures_format(value_col="feature1", subset="all")
        self.assertEqual(list(df_all.columns), ["unique_id", "ds", "y"])
        self.assertEqual(len(df_all), 10)
        self.assertTrue((df_all["unique_id"] == "AAA").all())

        # Test exporting train dataset
        df_train = organiser.to_tsfeatures_format(value_col="feature1", subset="train")
        self.assertEqual(list(df_train.columns), ["unique_id", "ds", "y"])
        self.assertEqual(len(df_train), 7)

        # Test exporting test dataset
        df_test = organiser.to_tsfeatures_format(value_col="feature1", subset="test")
        self.assertEqual(list(df_test.columns), ["unique_id", "ds", "y"])
        self.assertEqual(len(df_test), 3)

        # Test raising ValueError on invalid subset
        with self.assertRaises(ValueError):
            organiser.to_tsfeatures_format(value_col="feature1", subset="invalid")

        # Test raising KeyError on invalid value_col
        with self.assertRaises(KeyError):
            organiser.to_tsfeatures_format(value_col="nonexistent", subset="all")


if __name__ == "__main__":
    unittest.main()
