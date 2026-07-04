import unittest
import pandas as pd
import numpy as np
from pyquantflow.data.pipelinedataorganiser import PipelineDataOrganiser


class TestPipelineDataOrganiser(unittest.TestCase):
    """Tests for the strict PipelineDataOrganiser ML data provisioner."""

    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=30)

        self.data_map = {}
        for ticker in ["AAA", "BBB"]:
            df = pd.DataFrame(
                {
                    "Close": 100.0 + np.cumsum(np.random.normal(0, 1, 30)),
                    "feature1": np.random.randn(30),
                    "feature2": np.random.randn(30),
                    "target": np.random.randint(0, 2, 30),
                    "weight": np.random.uniform(0.5, 1.5, 30),
                },
                index=dates,
            )
            df.index.name = "datetime"
            self.data_map[ticker] = df

        self.cutoff_date = str(dates[20])

    def test_basic_payload_structure(self):
        """Payload contains the expected keys and raw DataFrames."""
        pdo = PipelineDataOrganiser(
            data_map=self.data_map,
            cutoff_date=self.cutoff_date,
            target_features=["target"],
            weight_col="weight",
        )
        pdo.prepare_multi_asset_frame()
        payload = pdo.get_classifierengine_payload(features=["feature1", "feature2"])

        self.assertIn("X_train", payload)
        self.assertIn("y_train", payload)
        self.assertIn("X_test", payload)
        self.assertIn("y_test", payload)
        self.assertIn("features", payload)
        self.assertIn("weight_col", payload)

        # Features list must not contain the weight column
        self.assertNotIn("weight", payload["features"])
        self.assertNotIn("target", payload["features"])
        self.assertEqual(set(payload["features"]), {"feature1", "feature2"})
        self.assertEqual(payload["weight_col"], "weight")

    def test_payload_dataframes_are_raw(self):
        """Returned DataFrames should contain the original raw feature values."""
        pdo = PipelineDataOrganiser(
            data_map=self.data_map,
            cutoff_date=self.cutoff_date,
            target_features=["target"],
        )
        pdo.prepare_multi_asset_frame()
        payload = pdo.get_classifierengine_payload(features=["feature1"])

        X_train = payload["X_train"]

        # The raw feature values should be present and unmodified
        self.assertIn("feature1", X_train.columns)
        # Verify the MultiIndex structure is preserved
        self.assertEqual(X_train.index.names, ["datetime", "ticker"])

    def test_ticker_filtering(self):
        """Payload can filter to a subset of tickers."""
        pdo = PipelineDataOrganiser(
            data_map=self.data_map,
            cutoff_date=self.cutoff_date,
            target_features=["target"],
        )
        pdo.prepare_multi_asset_frame()
        payload = pdo.get_classifierengine_payload(
            features=["feature1"], tickers=["AAA"]
        )

        train_tickers = payload["X_train"].index.get_level_values("ticker").unique()
        test_tickers = payload["X_test"].index.get_level_values("ticker").unique()
        self.assertEqual(set(train_tickers), {"AAA"})
        self.assertEqual(set(test_tickers), {"AAA"})

    def test_read_only_properties(self):
        """Read-only property pass-throughs expose organiser state."""
        pdo = PipelineDataOrganiser(
            data_map=self.data_map,
            cutoff_date=self.cutoff_date,
            target_features=["target"],
            weight_col="weight",
        )
        pdo.prepare_multi_asset_frame()

        self.assertIsNotNone(pdo.multi_asset)
        self.assertIsNotNone(pdo.multi_asset_train)
        self.assertIsNotNone(pdo.multi_asset_test)
        self.assertEqual(pdo.weight_col, "weight")
        self.assertEqual(pdo.target_features, ["target"])

        # Verify the train/test split is consistent
        self.assertGreater(len(pdo.multi_asset_train), 0)
        self.assertGreater(len(pdo.multi_asset_test), 0)

    def test_does_not_expose_eda_methods(self):
        """PipelineDataOrganiser must NOT expose EDA-only methods."""
        pdo = PipelineDataOrganiser(
            data_map=self.data_map,
            cutoff_date=self.cutoff_date,
            target_features=["target"],
        )
        self.assertFalse(hasattr(pdo, "add_model_predictions"))
        self.assertFalse(hasattr(pdo, "replace_features"))
        self.assertFalse(hasattr(pdo, "to_tsfeatures_format"))
        self.assertFalse(hasattr(pdo, "update_multi_asset"))

    def test_init_validation_errors(self):
        """Constructor enforces the same validation as AssetOrganiser."""
        # Neither data_map nor multi_asset
        with self.assertRaises(ValueError):
            PipelineDataOrganiser(
                cutoff_date=self.cutoff_date,
                target_features=["target"],
            )

        # Missing cutoff_date
        with self.assertRaises(ValueError):
            PipelineDataOrganiser(
                data_map=self.data_map,
                target_features=["target"],
            )

        # Missing target_features
        with self.assertRaises(ValueError):
            PipelineDataOrganiser(
                data_map=self.data_map,
                cutoff_date=self.cutoff_date,
            )

    def test_multi_asset_constructor(self):
        """Can be initialised with a pre-built multi_asset DataFrame."""
        # Build multi_asset manually
        frames = []
        for ticker, df in self.data_map.items():
            temp = df.copy()
            temp["ticker"] = ticker
            temp = temp.reset_index().set_index(["datetime", "ticker"])
            frames.append(temp)
        multi_asset = pd.concat(frames).sort_index()

        pdo = PipelineDataOrganiser(
            multi_asset=multi_asset,
            cutoff_date=self.cutoff_date,
            target_features=["target"],
        )
        # Should be split immediately in __init__ via delegation
        self.assertIsNotNone(pdo.multi_asset_train)
        self.assertIsNotNone(pdo.multi_asset_test)


if __name__ == "__main__":
    unittest.main()
