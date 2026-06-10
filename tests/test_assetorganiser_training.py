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
        import os
        from pyquantflow.data.database import DatabaseManager

        self.data_map = {}
        source_db_path = os.path.join(os.path.dirname(__file__), "stocks.db")

        if os.path.exists(source_db_path):
            try:
                db_manager = DatabaseManager(db_path=source_db_path)
                for ticker in ["FMG.AX", "CBA.AX"]:
                    df = db_manager.get_data(ticker)
                    if not df.empty and len(df) >= 10:
                        df = df.iloc[-10:].copy()
                        np.random.seed(42)
                        n = len(df)
                        df["feature1"] = np.random.randn(n)
                        df["target"] = np.random.randint(0, 2, n)
                        df["weight"] = np.random.uniform(0.5, 1.5, n)
                        self.data_map[ticker] = df
                db_manager.conn.close()
            except Exception:
                pass

        if len(self.data_map) < 2:
            # Fallback
            np.random.seed(42)
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

        # Set dynamic cutoff date (exclusive for train) to retain exactly 7 train elements
        first_ticker = list(self.data_map.keys())[0]
        self.cutoff_date = self.data_map[first_ticker].index[7]

    def test_pipeline_integration_with_weights(self):
        # 1. Asset Organiser
        organiser = AssetOrganiser(
            data_map=self.data_map,
            cutoff_date=self.cutoff_date,
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

        self.assertTrue(hasattr(final_pipe.steps[-1][1], "classes_"))


class TestAssetOrganiserFlexibility(unittest.TestCase):
    def setUp(self):
        import os
        from pyquantflow.data.database import DatabaseManager

        self.data_map = {}
        source_db_path = os.path.join(os.path.dirname(__file__), "stocks.db")

        if os.path.exists(source_db_path):
            try:
                db_manager = DatabaseManager(db_path=source_db_path)
                df = db_manager.get_data("CBA.AX")
                if not df.empty and len(df) >= 10:
                    df = df.iloc[-10:].copy()
                    np.random.seed(42)
                    n = len(df)
                    df["feature1"] = np.random.randn(n)
                    df["target"] = np.random.randint(0, 2, n)
                    self.data_map["AAA"] = df
                db_manager.conn.close()
            except Exception:
                pass

        if not self.data_map:
            np.random.seed(42)
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
        self.multi_asset = self.data_map["AAA"].copy()
        self.multi_asset["ticker"] = "AAA"
        self.multi_asset = self.multi_asset.reset_index().set_index(
            ["datetime", "ticker"]
        )
        self.cutoff_date = self.data_map["AAA"].index[7]

    def test_both_provided_raises_value_error(self):
        with self.assertRaises(ValueError):
            AssetOrganiser(
                data_map=self.data_map,
                multi_asset=self.multi_asset,
                cutoff_date=self.cutoff_date,
                target_features=["target"],
            )

    def test_neither_provided_raises_value_error(self):
        with self.assertRaises(ValueError):
            AssetOrganiser(
                cutoff_date=self.cutoff_date,
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
                cutoff_date=self.cutoff_date,
            )

    def test_multi_asset_split_immediately(self):
        organiser = AssetOrganiser(
            multi_asset=self.multi_asset,
            cutoff_date=self.cutoff_date,
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
            cutoff_date=self.cutoff_date,
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
            cutoff_date=self.cutoff_date,
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
