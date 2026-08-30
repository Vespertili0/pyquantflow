import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pyquantflow.backtesting.batchbacktest import BatchBacktester
from pyquantflow.strategies.example_strategy import SmaCross
from pyquantflow.data.assetorganiser import AssetOrganiser


@pytest.fixture
def sample_data():
    """Provides a valid sample DataFrame for backtesting."""
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "Open": np.random.rand(10) * 10,
            "High": np.random.rand(10) * 12,
            "Low": np.random.rand(10) * 8,
            "Close": np.random.rand(10) * 11,
            "Volume": np.random.randint(100, 1000, 10),
        },
        index=dates,
    )
    return df


@pytest.fixture
def backtester():
    """Initialises a BatchBacktester with an in-memory database."""
    return BatchBacktester(":memory:")


def test_validate_data_valid(backtester, sample_data):
    """Test that a valid DataFrame passes validation unaltered."""
    validated = backtester._validate_data(sample_data)
    pd.testing.assert_frame_equal(validated, sample_data)


def test_validate_data_missing_columns(backtester, sample_data):
    """Test that missing mandatory columns raise a ValueError."""
    df_missing = sample_data.drop(columns=["Close"])
    with pytest.raises(ValueError, match="Data must contain columns:"):
        backtester._validate_data(df_missing)


@patch("pyquantflow.backtesting.batchbacktest.Backtest")
def test_run_single_backtest_success(mock_backtest, backtester, sample_data):
    """Test the single backtest execution behaviour and timezone normalisation."""
    mock_bt_instance = mock_backtest.return_value
    mock_stats = MagicMock()
    mock_stats.to_dict.return_value = {"Return [%]": 10.0, "Sharpe Ratio": 1.5}
    mock_bt_instance.run.return_value = mock_stats

    # Call with non-DatetimeIndex to test conversion
    df_no_tz = sample_data.copy()
    df_no_tz.index = df_no_tz.index.tz_localize(None)

    stats = backtester.run_single_backtest(
        df=df_no_tz,
        strategy_class=SmaCross,
        cash=10000,
        commission=0.0,
        trade_on_close=False,
        finalize_trades=True,
        margin=0.5,
        custom_param=123,
    )

    # Check conversion to UTC DatetimeIndex
    assert isinstance(df_no_tz.index, pd.DatetimeIndex)
    assert df_no_tz.index.tz is not None

    mock_backtest.assert_called_once()
    kwargs = mock_backtest.call_args[1]
    assert kwargs["margin"] == 0.5
    assert kwargs["cash"] == 10000

    mock_bt_instance.run.assert_called_once_with(custom_param=123)
    assert stats == {"Return [%]": 10.0, "Sharpe Ratio": 1.5}


def test_run_batch_backtest_conflict(backtester, sample_data):
    """Test that providing both data and asset_organiser raises an error."""
    organiser = MagicMock(spec=AssetOrganiser)
    with pytest.raises(ValueError, match="Cannot provide both 'data' and 'asset_organiser'"):
        backtester.run_batch_backtest(SmaCross, data=sample_data, asset_organiser=organiser)


def test_run_batch_backtest_no_data(backtester):
    """Test behaviour when neither data nor asset_organiser is provided."""
    results = backtester.run_batch_backtest(SmaCross)
    assert backtester.results == {"individual_results": {}, "average_metrics": {}}
    assert results == {}


def test_run_batch_backtest_invalid_data(backtester):
    """Test behaviour when data is not a DataFrame or dictionary."""
    with pytest.raises(ValueError, match="Data must be a pandas DataFrame or a dictionary"):
        backtester.run_batch_backtest(SmaCross, data=[1, 2, 3])


@patch.object(BatchBacktester, "run_single_backtest")
def test_run_batch_backtest_single_df(mock_run_single, backtester, sample_data):
    """Test execution with a single DataFrame."""
    mock_run_single.return_value = {"Return [%]": 5.0}

    # symbols="all" should default to "asset"
    avg = backtester.run_batch_backtest(SmaCross, data=sample_data)
    assert "asset" in backtester.results["individual_results"]
    assert avg["Return [%]"] == 5.0

    # symbols=["CUSTOM"]
    avg2 = backtester.run_batch_backtest(SmaCross, data=sample_data, symbols=["CUSTOM"])
    assert "CUSTOM" in backtester.results["individual_results"]


@patch.object(BatchBacktester, "run_single_backtest")
def test_run_batch_backtest_dict(mock_run_single, backtester, sample_data):
    """Test execution with a dictionary of DataFrames."""
    mock_run_single.side_effect = [{"Return [%]": 2.0}, {"Return [%]": 4.0}]
    data_map = {"SYM1": sample_data, "SYM2": sample_data}

    avg = backtester.run_batch_backtest(SmaCross, data=data_map)
    ind_res = backtester.results["individual_results"]
    assert "SYM1" in ind_res
    assert "SYM2" in ind_res
    assert avg["Return [%]"] == 3.0


@patch.object(BatchBacktester, "run_single_backtest")
def test_run_batch_backtest_organiser(mock_run_single, backtester, sample_data):
    """Test execution integrated with AssetOrganiser."""
    mock_run_single.return_value = {"Return [%]": 10.0}
    
    organiser = MagicMock(spec=AssetOrganiser)
    multi_index = pd.MultiIndex.from_tuples(
        [("2023-01-01", "SYM1"), ("2023-01-02", "SYM2")], 
        names=["datetime", "ticker"]
    )
    organiser.get_transformed_multiasset_testdata.return_value = pd.DataFrame(index=multi_index)
    organiser.get_transformed_test_ticker.return_value = sample_data

    # Test "all"
    backtester.run_batch_backtest(SmaCross, asset_organiser=organiser, symbols="all")
    assert "SYM1" in backtester.results["individual_results"]
    assert "SYM2" in backtester.results["individual_results"]
    
    # Test unknown symbol warning (should just log and ignore without crashing)
    backtester.run_batch_backtest(SmaCross, asset_organiser=organiser, symbols=["NONEXISTENT"])
    assert "NONEXISTENT" not in backtester.results["individual_results"]


@patch.object(BatchBacktester, "run_single_backtest")
def test_run_batch_backtest_exception_handling(mock_run_single, backtester, sample_data):
    """Test that a single asset failure is caught and recorded without terminating the batch."""
    # Simulate an error for SYM1 and success for SYM2
    def side_effect(df, **kwargs):
        if len(mock_run_single.call_args_list) == 1:
            raise ValueError("Test error")
        return {"Return [%]": 10.0}
        
    mock_run_single.side_effect = side_effect
    data_map = {"SYM1": sample_data, "SYM2": sample_data}
    
    avg = backtester.run_batch_backtest(SmaCross, data=data_map)
    ind_res = backtester.results["individual_results"]
    
    assert ind_res["SYM1"] == {"Error": "ValueError"}
    assert ind_res["SYM2"] == {"Return [%]": 10.0}
    assert avg["Return [%]"] == 10.0


def test_calculate_averages_empty(backtester):
    """Test calculating averages on empty results."""
    avg = backtester._calculate_averages({})
    assert pd.isna(avg["Return [%]"])


def test_calculate_averages_with_errors_and_nans(backtester):
    """Test averaging logic with errors, NaNs, and Timedelta values."""
    results = {
        "SYM1": {"Error": "Exception"},
        "SYM2": {"Return [%]": 5.0, "Sharpe Ratio": np.nan, "Duration": pd.Timedelta(days=1)}
    }
    avg = backtester._calculate_averages(results)
    assert avg["Return [%]"] == 5.0
    assert pd.isna(avg["Sharpe Ratio"])
    assert avg["Duration"] == pd.Timedelta(days=1)


@patch("pyquantflow.backtesting.batchbacktest.BacktestDatabaseManager")
def test_save_batch_results_no_results(mock_db, backtester):
    """Test saving results when none exist."""
    backtester.results = None
    assert backtester.save_batch_results() is None


@patch("pyquantflow.backtesting.batchbacktest.datetime")
def test_save_batch_results_success(mock_datetime, backtester):
    """Test successful serialisation and storage of batch results."""
    mock_datetime.now.return_value = pd.to_datetime("2023-01-01")
    
    backtester.results = {
        "individual_results": {"SYM1": {"Return [%]": 10.0}}
    }
    backtester.strategy_class = SmaCross
    
    # Mock the database manager
    mock_db = MagicMock()
    backtester.results_db = mock_db
    
    batch_name = backtester.save_batch_results()
    
    assert batch_name == "2023-01-01_SmaCross"
    mock_db.save_result.assert_called_once_with("SYM1", {"Return [%]": 10.0}, batch_name)
