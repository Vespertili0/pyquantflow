import unittest
import pandas as pd
import numpy as np
import os
from pyquantflow.data.assetorganiser import AssetOrganiser
from pyquantflow.data.database import DatabaseManager
from sklearn.tree import DecisionTreeClassifier


class MockLabelFactory:
    def generate_labels(self, df, price_col="Close"):
        res = pd.DataFrame({"label": np.zeros(len(df)), "t1": df.index}, index=df.index)
        return res

    def generate_weights(self, t1, returns):
        return pd.Series(np.ones(len(returns)), index=returns.index)


class MockClassifier:
    def fit(self, X, y, sample_weight=None):
        self.fitted = True
        return self

    def transform(self, X):
        X_out = X.copy()
        X_out["mock_pred"] = 1
        return X_out


class TestAssetOrganiserCoverage(unittest.TestCase):
    def setUp(self):
        source_db_path = os.path.join(os.path.dirname(__file__), "stocks.db")
        self.data_map = {}

        if os.path.exists(source_db_path):
            try:
                db_manager = DatabaseManager(db_path=source_db_path)
                for ticker in ["FMG.AX", "CBA.AX"]:
                    df = db_manager.get_data(ticker)
                    if not df.empty and len(df) >= 100:
                        df = df.iloc[-100:].copy()
                        df["feature1"] = np.random.randn(len(df))
                        df["target"] = np.random.randint(0, 2, len(df))
                        df["returns"] = df["Close"].pct_change()
                        self.data_map[ticker] = df
                db_manager.conn.close()
            except Exception:
                pass

        if len(self.data_map) < 2:
            dates = pd.date_range("2020-01-01", periods=100)
            df_a = pd.DataFrame(
                {
                    "Close": 100.0 + np.cumsum(np.random.normal(0, 1.0, 100)),
                    "feature1": np.random.randn(100),
                    "target": np.random.randint(0, 2, 100),
                },
                index=dates,
            )
            df_a["returns"] = df_a["Close"].pct_change()
            df_a.index.name = "datetime"

            df_b = pd.DataFrame(
                {
                    "Close": 100.0 + np.cumsum(np.random.normal(0, 1.0, 100)),
                    "feature1": np.random.randn(100),
                    "target": np.random.randint(0, 2, 100),
                },
                index=dates,
            )
            df_b["returns"] = df_b["Close"].pct_change()
            df_b.index.name = "datetime"
            self.data_map = {"AAA": df_a, "BBB": df_b}

        self.ticker_1 = list(self.data_map.keys())[0]
        self.ticker_2 = list(self.data_map.keys())[1]
        self.cutoff_date = str(self.data_map[self.ticker_1].index[60])

    def test_downsample_to_events_type_error(self):
        organiser = AssetOrganiser(
            data_map=self.data_map,
            cutoff_date=self.cutoff_date,
            target_features=["target"],
        )
        with self.assertRaises(TypeError):
            organiser.downsample_to_events("invalid format")

    def test_downsample_cusum_training_data_missing(self):
        organiser = AssetOrganiser(
            data_map=self.data_map,
            cutoff_date=self.cutoff_date,
            target_features=["target"],
        )
        # downsample_to_cusum_events calls prepare_multi_asset_frame automatically if multi_asset is None
        # so to test ValueError, we explicitly set multi_asset_train to None after multi_asset is created
        organiser.prepare_multi_asset_frame()
        organiser.multi_asset_train = None
        with self.assertRaises(ValueError):
            organiser.downsample_to_cusum_events(
                target_events_train=5, filter_col="returns"
            )

    def test_downsample_cusum_ticker_missing_train(self):
        # Create a ticker that is only present in test data
        dates = pd.date_range(
            pd.to_datetime(self.cutoff_date) + pd.Timedelta(days=1), periods=10
        )
        # Ensure it has all the columns from the original dataframe so that dropna() does not drop it
        first_ticker = list(self.data_map.keys())[0]
        cols = self.data_map[first_ticker].columns
        df_c = pd.DataFrame(index=dates, columns=cols)
        df_c = df_c.fillna(0.0)
        df_c.index.name = "datetime"
        data_map = self.data_map.copy()
        data_map["CCC"] = df_c

        organiser = AssetOrganiser(
            data_map=data_map, cutoff_date=self.cutoff_date, target_features=["target"]
        )
        organiser.prepare_multi_asset_frame()

        # Ticker CCC is not in train data, should fallback to alpha_min
        alphas = organiser.downsample_to_cusum_events(
            target_events_train=5, filter_col="returns", alpha_min=0.5
        )
        self.assertEqual(alphas["CCC"], 0.5)

    def test_downsample_cusum_target_events_dict_key_error(self):
        organiser = AssetOrganiser(
            data_map=self.data_map,
            cutoff_date=self.cutoff_date,
            target_features=["target"],
        )
        organiser.prepare_multi_asset_frame()
        # Pass a dict for target events but miss a ticker
        with self.assertRaises(KeyError):
            organiser.downsample_to_cusum_events(
                target_events_train={self.ticker_1: 5}, filter_col="returns"
            )

    def test_downsample_cusum_missing_vol_col(self):
        organiser = AssetOrganiser(
            data_map=self.data_map,
            cutoff_date=self.cutoff_date,
            target_features=["target"],
        )
        organiser.prepare_multi_asset_frame()
        # Pass vol_col="missing_vol" which does not exist
        # It should ignore the KeyError and compute dynamic volatility instead
        alphas = organiser.downsample_to_cusum_events(
            target_events_train=5, filter_col="returns", vol_col="missing_vol"
        )
        self.assertIn(self.ticker_1, alphas)

    def test_apply_continuous_labels_missing_factory(self):
        organiser = AssetOrganiser(
            data_map=self.data_map,
            cutoff_date=self.cutoff_date,
            target_features=["target"],
        )
        with self.assertRaises(ValueError):
            organiser.apply_continuous_labels()

    def test_apply_sample_weights_errors(self):
        organiser = AssetOrganiser(
            data_map=self.data_map,
            cutoff_date=self.cutoff_date,
            target_features=["target"],
        )

        # 1. Multi-asset None (implicitly covered if not prepared but wait, prepare is needed)
        # However, let's test missing label factory
        organiser.prepare_multi_asset_frame()
        with self.assertRaises(ValueError):
            organiser.apply_sample_weights()

        organiser.label_factory = MockLabelFactory()
        # 2. Missing 't1' column
        with self.assertRaises(KeyError):
            organiser.apply_sample_weights()

    def test_fit_quant_classifier_and_transform(self):
        organiser = AssetOrganiser(
            data_map=self.data_map,
            cutoff_date=self.cutoff_date,
            target_features=["target"],
        )

        # 1. Data not prepared
        with self.assertRaises(ValueError):
            organiser.fit_quant_classifier()

        organiser.prepare_multi_asset_frame()

        # 2. No classifier
        with self.assertRaises(ValueError):
            organiser.fit_quant_classifier()

        # 3. Valid classification
        organiser.classifier = MockClassifier()
        organiser.fit_quant_classifier()

        self.assertTrue(organiser.classifier.fitted)

        transformed_test = organiser.get_transformed_multiasset_testdata()
        self.assertIn("mock_pred", transformed_test.columns)

        ticker_test = organiser.get_transformed_test_ticker(self.ticker_1)
        self.assertIn("mock_pred", ticker_test.columns)

    def test_add_model_predictions(self):
        organiser = AssetOrganiser(
            data_map=self.data_map,
            cutoff_date=self.cutoff_date,
            target_features=["target"],
        )
        organiser.prepare_multi_asset_frame()

        model = DecisionTreeClassifier(random_state=42)
        X_train = organiser.multi_asset_train[["feature1"]]
        y_train = organiser.multi_asset_train["target"]
        model.fit(X_train, y_train)

        organiser.add_model_predictions(model, features=["feature1"], prefix="primary")

        self.assertIn("primary_pred", organiser.multi_asset.columns)
        self.assertIn("primary_entropy", organiser.multi_asset.columns)
        self.assertIn("primary_proba0", organiser.multi_asset.columns)
        self.assertIn("primary_proba1", organiser.multi_asset.columns)

        # Test filtering
        len_before = len(organiser.multi_asset)
        organiser.add_model_predictions(
            model, features=["feature1"], prefix="secondary", filter_prediction=1
        )
        # Only instances where prediction == 1 remain
        self.assertTrue(len(organiser.multi_asset) <= len_before)
        self.assertTrue((organiser.multi_asset["secondary_pred"] == 1).all())

    def test_update_multi_asset_index_error(self):
        organiser = AssetOrganiser(
            data_map=self.data_map,
            cutoff_date=self.cutoff_date,
            target_features=["target"],
        )
        df = pd.DataFrame({"A": [1, 2]})
        with self.assertRaises(ValueError):
            organiser.update_multi_asset(df)

    def test_replace_features(self):
        organiser = AssetOrganiser(
            data_map=self.data_map,
            cutoff_date=self.cutoff_date,
            target_features=["target"],
        )
        organiser.prepare_multi_asset_frame()

        # Invalid index
        df = pd.DataFrame({"A": [1, 2]})
        with self.assertRaises(ValueError):
            organiser.replace_features(df, ["feature1"])

        # Uninitialised multi_asset
        org_uninit = AssetOrganiser(
            data_map=self.data_map,
            cutoff_date=self.cutoff_date,
            target_features=["target"],
        )
        # Not calling prepare_multi_asset_frame()
        org_uninit.multi_asset = None

        valid_df = organiser.multi_asset.copy()
        with self.assertRaises(ValueError):
            org_uninit.replace_features(valid_df, ["feature1"])

        # Normal operation
        valid_df["feature1"] = valid_df["feature1"] * 2
        valid_df["new_feat"] = 1
        organiser.replace_features(valid_df, ["feature1"])

        # It should replace feature1, but not include new_feat since it's not in original features list
        # Actually it only updates surviving features.
        self.assertIn("feature1", organiser.multi_asset.columns)


if __name__ == "__main__":
    unittest.main()
