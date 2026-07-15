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

    def test_apply_ichimoku_regime(self):
        """
        apply_ichimoku_regime() must:
        - inject the 'ichimoku_regime' column (values 0 or 1 only),
        - drop all raw Ichimoku component columns,
        - preserve the original row count (no rows dropped),
        - work correctly across multiple tickers.
        """
        # Build a dedicated 200-bar OHLC panel with High and Low columns.
        # 200 bars ensures Ichimoku's warm-up (kijun=26 + displacement=26 + senkou_b=52)
        # is satisfied without eating all available data.
        n = 200
        np.random.seed(99)
        dates = pd.date_range("2018-01-01", periods=n, freq="D")

        def _make_ohlc(seed_offset=0):
            np.random.seed(99 + seed_offset)
            close = 100.0 + np.cumsum(np.random.normal(0, 1.0, n))
            high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
            low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
            open_ = (high + low) / 2
            high = np.maximum(high, np.maximum(open_, close))
            low = np.minimum(low, np.minimum(open_, close))
            return pd.DataFrame(
                {"Open": open_, "High": high, "Low": low, "Close": close},
                index=dates,
            )

        ohlc_data_map = {
            "TICKER_A": _make_ohlc(seed_offset=0),
            "TICKER_B": _make_ohlc(seed_offset=1),
        }
        for df in ohlc_data_map.values():
            df.index.name = "datetime"

        organiser = AssetOrganiser(
            data_map=ohlc_data_map,
            cutoff_date="2019-01-01",
            target_features=["Close"],
        )
        organiser.prepare_multi_asset_frame()

        row_count_before = len(organiser.multi_asset)

        organiser.apply_ichimoku_regime()

        # 1. The regime column must be present
        self.assertIn("ichimoku_regime", organiser.multi_asset.columns)

        # 2. No raw Ichimoku components should remain
        _RAW_COLS = [
            "tenkan_sen",
            "kijun_sen",
            "span_a",
            "span_b",
            "span_a_shifted",
            "span_b_shifted",
        ]
        for col in _RAW_COLS:
            self.assertNotIn(
                col,
                organiser.multi_asset.columns,
                msg=f"Raw Ichimoku column '{col}' was not dropped.",
            )

        # 3. No rows must have been dropped
        self.assertEqual(
            len(organiser.multi_asset),
            row_count_before,
            msg="apply_ichimoku_regime() must not drop any rows.",
        )

        # 4. Regime values must be strictly 0 or 1
        unique_vals = set(organiser.multi_asset["ichimoku_regime"].unique())
        self.assertTrue(
            unique_vals.issubset({0, 1}),
            msg=f"Unexpected regime values: {unique_vals}",
        )

        # 5. Train/test splits must be refreshed
        self.assertIsNotNone(organiser.multi_asset_train)
        self.assertIsNotNone(organiser.multi_asset_test)


class TestAssetOrganiserNewBranches(unittest.TestCase):
    """
    Covers branches in AssetOrganiser that were not exercised by the
    previous test classes.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_organiser(self, n: int = 200, seed: int = 0) -> AssetOrganiser:
        """
        Returns a freshly constructed AssetOrganiser backed by two synthetic
        tickers.  *prepare_multi_asset_frame* is NOT called so tests can
        exercise the lazy-init paths themselves where needed.
        """
        np.random.seed(seed)
        dates = pd.date_range("2020-01-01", periods=n, freq="D")

        def _make_df(s: int = 0) -> pd.DataFrame:
            np.random.seed(seed + s)
            close = 100.0 + np.cumsum(np.random.normal(0, 1.0, n))
            high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
            low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
            df = pd.DataFrame(
                {
                    "Open": (high + low) / 2,
                    "High": high,
                    "Low": low,
                    "Close": close,
                    "Volume": np.random.randint(1000, 100_000, n).astype(float),
                    "feature1": np.random.randn(n),
                    "target": np.random.randint(0, 2, n),
                },
                index=dates,
            )
            df.index.name = "datetime"
            return df

        data_map = {"AAA": _make_df(0), "BBB": _make_df(1)}
        cutoff = str(dates[150])

        return AssetOrganiser(
            data_map=data_map,
            cutoff_date=cutoff,
            target_features=["target"],
        )

    def _make_organiser_with_labels(
        self, n: int = 200, seed: int = 0
    ) -> AssetOrganiser:
        """
        Returns an organiser that has already had continuous labels applied
        via MockLabelFactory, ready for weight / pipeline tests.
        """
        org = self._make_organiser(n=n, seed=seed)
        org.label_factory = MockLabelFactory()
        org.prepare_multi_asset_frame()
        org.apply_continuous_labels()
        return org

    # ------------------------------------------------------------------
    # downsample_to_events — list / set / dict inputs
    # ------------------------------------------------------------------

    def test_downsample_to_events_list_input(self):
        """A plain Python list of timestamps must filter correctly."""
        org = self._make_organiser()
        org.prepare_multi_asset_frame()

        # Pick 10 datetime values from the prepared panel
        all_dts = org.multi_asset.index.get_level_values("datetime").unique()
        event_list = list(all_dts[:10])

        org.downsample_to_events(event_list)

        # After down-sampling each row's datetime must be in the event set
        remaining_dts = org.multi_asset.index.get_level_values("datetime")
        event_set = {pd.Timestamp(t) for t in event_list}
        for dt in remaining_dts:
            self.assertIn(pd.Timestamp(dt), event_set)

    def test_downsample_to_events_set_input(self):
        """A Python set of timestamps must filter correctly."""
        org = self._make_organiser()
        org.prepare_multi_asset_frame()

        all_dts = org.multi_asset.index.get_level_values("datetime").unique()
        event_set_input = {pd.Timestamp(t) for t in all_dts[:15]}

        org.downsample_to_events(event_set_input)

        remaining_dts = org.multi_asset.index.get_level_values("datetime")
        for dt in remaining_dts:
            self.assertIn(pd.Timestamp(dt), event_set_input)

    def test_downsample_to_events_dict_input(self):
        """
        A per-ticker dict must retain only the matching dates for each ticker
        and exclude tickers not in the dict.
        """
        org = self._make_organiser()
        org.prepare_multi_asset_frame()

        aaa_dts = (
            org.multi_asset.xs("AAA", level="ticker")
            .index.get_level_values("datetime")
            .unique()
        )
        # Only include AAA events; BBB has no entry → all BBB rows dropped
        events_dict = {"AAA": pd.DatetimeIndex(aaa_dts[:20])}

        org.downsample_to_events(events_dict)

        tickers_remaining = set(
            org.multi_asset.index.get_level_values("ticker").unique()
        )
        # BBB must have been filtered out entirely
        self.assertNotIn("BBB", tickers_remaining)
        self.assertIn("AAA", tickers_remaining)

        # AAA rows must all be in the event set
        aaa_event_set = {pd.Timestamp(t) for t in aaa_dts[:20]}
        aaa_dts_remaining = org.multi_asset.xs(
            "AAA", level="ticker"
        ).index.get_level_values("datetime")
        for dt in aaa_dts_remaining:
            self.assertIn(pd.Timestamp(dt), aaa_event_set)

    # ------------------------------------------------------------------
    # downsample_to_cusum_events — uniqueness objective error paths
    # ------------------------------------------------------------------

    def test_cusum_uniqueness_requires_t1_col(self):
        """
        objective='uniqueness' without t1_col must raise ValueError before
        any CUSUM computation starts.
        """
        org = self._make_organiser()
        org.prepare_multi_asset_frame()

        with self.assertRaises(ValueError):
            org.downsample_to_cusum_events(
                target_events_train=10,
                filter_col="feature1",
                objective="uniqueness",
                t1_col=None,
            )

    def test_cusum_uniqueness_missing_t1_column_in_data(self):
        """
        When objective='uniqueness', t1_col is given but the column is absent
        from the training data, a KeyError must be raised.
        """
        org = self._make_organiser()
        org.prepare_multi_asset_frame()

        # t1_col is specified but the column doesn't exist in the DataFrame
        with self.assertRaises(KeyError):
            org.downsample_to_cusum_events(
                target_events_train=10,
                filter_col="feature1",
                objective="uniqueness",
                t1_col="nonexistent_t1",
            )

    # ------------------------------------------------------------------
    # apply_continuous_labels — happy path & duplicate-column deduplication
    # ------------------------------------------------------------------

    def test_apply_continuous_labels_happy_path(self):
        """
        After apply_continuous_labels, 'label' and 't1' columns must exist
        in multi_asset and train/test splits must be refreshed.
        """
        org = self._make_organiser()
        org.label_factory = MockLabelFactory()
        org.prepare_multi_asset_frame()

        org.apply_continuous_labels()

        self.assertIn("label", org.multi_asset.columns)
        self.assertIn("t1", org.multi_asset.columns)
        self.assertIsNotNone(org.multi_asset_train)
        self.assertIsNotNone(org.multi_asset_test)

    def test_apply_continuous_labels_drops_duplicate_cols(self):
        """
        Calling apply_continuous_labels twice must not create duplicate columns.
        """
        org = self._make_organiser()
        org.label_factory = MockLabelFactory()
        org.prepare_multi_asset_frame()

        org.apply_continuous_labels()
        col_count_after_first = len(org.multi_asset.columns)

        org.apply_continuous_labels()
        col_count_after_second = len(org.multi_asset.columns)

        self.assertEqual(
            col_count_after_first,
            col_count_after_second,
            msg="Duplicate label/t1 columns should be dropped on second call.",
        )

    # ------------------------------------------------------------------
    # apply_sample_weights — happy path
    # ------------------------------------------------------------------

    def test_apply_sample_weights_happy_path(self):
        """
        Full weight pipeline: continuous labels → sample weights injected.
        The weight column must be present and all values must be positive.
        """
        org = self._make_organiser_with_labels()
        org.weight_col = "weight"

        org.apply_sample_weights()

        self.assertIn("weight", org.multi_asset.columns)
        self.assertTrue(
            (org.multi_asset["weight"] > 0).all(),
            msg="All sample weights must be positive after clipping.",
        )
        # Splits refreshed
        self.assertIsNotNone(org.multi_asset_train)
        self.assertIsNotNone(org.multi_asset_test)

    def test_apply_sample_weights_uses_default_weight_col_name(self):
        """
        When weight_col was not set at construction time,
        apply_sample_weights should default the column name to 'weight'
        and update self.weight_col accordingly.
        """
        org = self._make_organiser_with_labels()
        # weight_col is None by default from _make_organiser
        self.assertIsNone(org.weight_col)

        org.apply_sample_weights()

        self.assertEqual(org.weight_col, "weight")
        self.assertIn("weight", org.multi_asset.columns)

    # ------------------------------------------------------------------
    # build_learning_pipeline — end-to-end orchestration
    # ------------------------------------------------------------------

    def test_build_learning_pipeline_returns_alphas(self):
        """
        build_learning_pipeline must return a dict of calibrated alphas
        (one per ticker) after running the full label → CUSUM → weight chain.
        """
        org = self._make_organiser(n=200, seed=7)
        org.label_factory = MockLabelFactory()
        org.weight_col = "weight"

        alphas = org.build_learning_pipeline(
            target_events_train=20,
            filter_col="Close",
            price_col="Close",
        )

        self.assertIsInstance(alphas, dict)
        self.assertIn("AAA", alphas)
        self.assertIn("BBB", alphas)
        # Alphas must be within the default search range
        for ticker, alpha in alphas.items():
            self.assertGreaterEqual(alpha, 0.5)
            self.assertLessEqual(alpha, 3.0)

    # ------------------------------------------------------------------
    # update_multi_asset — happy path
    # ------------------------------------------------------------------

    def test_update_multi_asset_happy_path(self):
        """
        update_multi_asset with a correctly indexed DataFrame must update
        self.multi_asset and refresh the train/test splits.
        """
        org = self._make_organiser()
        org.prepare_multi_asset_frame()

        original_df = org.multi_asset.copy()
        original_df["new_col"] = 99.0

        org.update_multi_asset(original_df)

        self.assertIn("new_col", org.multi_asset.columns)
        self.assertTrue((org.multi_asset["new_col"] == 99.0).all())
        self.assertIsNotNone(org.multi_asset_train)
        self.assertIsNotNone(org.multi_asset_test)

    # ------------------------------------------------------------------
    # replace_features — surviving and dropped features
    # ------------------------------------------------------------------

    def test_replace_features_surviving_and_dropped(self):
        """
        replace_features must:
        - update surviving features with transformed values,
        - drop features absent from the transformed DataFrame,
        - align the multi_asset index to the transformed DataFrame's index.
        """
        org = self._make_organiser()
        org.prepare_multi_asset_frame()

        # Create a transformed DataFrame: keep feature1 (scaled) but NOT target
        transformed = org.multi_asset[["feature1"]].copy()
        transformed["feature1"] = transformed["feature1"] * 10.0

        # Drop a few rows to test index alignment
        transformed = transformed.iloc[5:]

        original_features = ["feature1", "target"]
        org.replace_features(transformed, original_features)

        # Row count must match the trimmed transformed DataFrame
        self.assertEqual(len(org.multi_asset), len(transformed))

        # feature1 must be updated to the scaled values
        np.testing.assert_array_almost_equal(
            org.multi_asset["feature1"].values,
            transformed["feature1"].values,
        )

        # 'target' was in original_features but not in transformed → dropped
        self.assertNotIn(
            "target",
            org.multi_asset.columns,
            msg="Failed feature 'target' should have been dropped.",
        )

    # ------------------------------------------------------------------
    # get_classifierengine_payload — weight_col stripped from features
    # ------------------------------------------------------------------

    def test_get_classifierengine_payload_strips_weight_col_from_features(self):
        """
        When weight_col is present in the features list passed to
        get_classifierengine_payload it must be silently removed from
        the returned 'features' key to prevent metadata leakage.
        """
        org = self._make_organiser()
        org.weight_col = "feature1"  # Pretend feature1 is also the weight col
        org.prepare_multi_asset_frame()

        payload = org.get_classifierengine_payload(
            features=["feature1", "target"],
        )

        self.assertNotIn(
            "feature1",
            payload["features"],
            msg="weight_col 'feature1' must be stripped from payload features.",
        )
        # The non-weight feature must remain
        self.assertIn("target", payload["features"])

    # ------------------------------------------------------------------
    # to_tsfeatures_format — lazy prepare paths for train / test subsets
    # ------------------------------------------------------------------

    def test_to_tsfeatures_format_lazy_train(self):
        """
        Calling to_tsfeatures_format(subset='train') on a cold organiser
        (prepare_multi_asset_frame not yet called) must trigger preparation
        and return a valid DataFrame.
        """
        org = self._make_organiser()
        # Do NOT call prepare_multi_asset_frame() — test lazy init path
        self.assertIsNone(org.multi_asset_train)

        df_train = org.to_tsfeatures_format(value_col="Close", subset="train")

        self.assertIn("unique_id", df_train.columns)
        self.assertIn("ds", df_train.columns)
        self.assertIn("y", df_train.columns)
        self.assertGreater(len(df_train), 0)

    def test_to_tsfeatures_format_lazy_test(self):
        """
        Same lazy-init check for subset='test'.
        """
        org = self._make_organiser()
        self.assertIsNone(org.multi_asset_test)

        df_test = org.to_tsfeatures_format(value_col="Close", subset="test")

        self.assertIn("unique_id", df_test.columns)
        self.assertGreater(len(df_test), 0)


if __name__ == "__main__":
    unittest.main()
