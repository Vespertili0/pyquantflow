import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any, Tuple
from backtesting import Backtest
from .backtest_database import BacktestDatabaseManager
from ..data.assetorganiser import AssetOrganiser

class BatchBacktester:
    def __init__(self, results_db_path: str = "backtest_results.db"):
        self.results_db = BacktestDatabaseManager(results_db_path)
        self.results: Optional[Dict[str, Any]] = None
        self.strategy_class: Optional[type] = None

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        return df

    def run_single_backtest(self, df: pd.DataFrame, strategy_class: type, 
                             cash: float, commission: float | Tuple[float, float],
                             trade_on_close: bool, finalize_trades: bool, 
                             **strategy_params) -> Dict[str, Any]:
        """Validates Data and executes a single backtest."""
        df = self._validate_data(df)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        
        bt = Backtest(df, strategy_class, cash=cash, commission=commission,
                      trade_on_close=trade_on_close, finalize_trades=finalize_trades)
        stats = bt.run(**strategy_params)
        return stats.to_dict()

    def run_batch_backtest(self, strategy_class: type, data: Optional[pd.DataFrame | Dict[str, pd.DataFrame]] = None,
                           asset_organiser: Optional[AssetOrganiser] = None, symbols: Optional[str | List[str]] = "all", 
                           cash: float = 10000, commission: float | Tuple[float, float] = (3.0, 0.0), 
                           trade_on_close: bool = False, finalize_trades: bool = True, **strategy_params) -> Dict[str, Any]:
        """
        Runs backtests for a list of tickers and aggregates results.
        
        Args:
            strategy_class: The strategy class to use.
            data: pd.DataFrame or dict. 
                  If DataFrame, maps to an "asset" symbol unless symbols list is provided.
                  If dict, keys are symbols, values are DataFrames.
            asset_organiser: AssetOrganiser instance containing transformed test data.
            symbols: 'all' to run all available, or a List of specific tickers to run. 
            cash: Initial cash.
            commission: Commission rate.
            trade_on_close: Whether to trade on close.
            finalize_trades: Whether to finalize trades.
            **strategy_params: Additional arguments for the strategy.
        
        Returns:
            dict: A dictionary containing 'individual_results' and 'average_metrics'.
        """
        self.strategy_class = strategy_class
        individual_results = {}
        data_map: Dict[str, pd.DataFrame] = {}

        if asset_organiser is not None:
            if data is not None:
                raise ValueError("Cannot provide both 'data' and 'asset_organiser'. Choose one.")
            
            # Use AssetOrganiser
            multiasset_test_data = asset_organiser.get_transformed_multiasset_testdata()
            available_symbols = getattr(multiasset_test_data.index.get_level_values('ticker'), 'unique', lambda: [])()
            if callable(available_symbols):
                available_symbols = available_symbols()
            else:
                 available_symbols = list(available_symbols)

            if symbols == "all":
                target_symbols = available_symbols
            elif isinstance(symbols, list):
                target_symbols = symbols
            else:
                target_symbols = [symbols] # Handle single string case
            
            for sym in target_symbols:
                if sym in available_symbols:
                    data_map[sym] = asset_organiser.get_transformed_test_ticker(sym)
                else:
                    print(f"Warning: Symbol '{sym}' not found in AssetOrganiser test data.")
                    
        elif data is not None:
            # Direct data handling
            if isinstance(data, pd.DataFrame):
                symbol = symbols[0] if (isinstance(symbols, list) and len(symbols) > 0) else ("asset" if symbols == "all" else symbols)
                data_map[symbol] = data
            elif isinstance(data, dict):
                data_map = data
            else:
                raise ValueError("Data must be a pandas DataFrame or a dictionary of DataFrames.")
        else:
            self.results = {'individual_results': {}, 'average_metrics': {}}
            return self.results

        for sym, df in data_map.items():
            print(f"Running backtest for {sym}...")
            try:
                stats_dict = self.run_single_backtest(
                    df=df, strategy_class=strategy_class, cash=cash, commission=commission,
                    trade_on_close=trade_on_close, finalize_trades=finalize_trades, **strategy_params
                )
                individual_results[sym] = stats_dict
                print(f"Finished {sym}: Return {stats_dict['Return [%]']:.2f}%")
            except Exception as e:
                print(f"Error running backtest for {sym}: {e}")
                individual_results[sym] = {'Error': str(e)}

        # Calculate averages
        average_metrics = self._calculate_averages(individual_results)
        
        self.results = {
            'individual_results': individual_results,
            'average_metrics': average_metrics
        }

        return self.results

    def save_batch_results(self):
        """
        Saves the batch backtest results to the database with a unique batch run name.
        Uses the results and strategy_class stored in self from the last run.

        Returns:
            str: The generated batch run name, or None if no results available.
        """
        from datetime import datetime

        if not self.results or not self.strategy_class:
            print("No results to save.")
            return None

        individual_results = self.results.get('individual_results', {})
        batch_run_name = f"{datetime.now().strftime('%Y-%m-%d')}_{self.strategy_class.__name__}"

        for ticker, result in individual_results.items():
            self.results_db.save_result(ticker, result, batch_run_name)

        return batch_run_name

    def _calculate_averages(self, results):
        """
        Calculates average metrics from a dictionary of results.
        """
        metrics_to_average = [
            'Duration', 'Exposure Time [%]', 'Equity Final [$]', 'Equity Peak [$]',
            'Return [%]', 'Buy & Hold Return [%]', 'Return (Ann.) [%]', 'Volatility (Ann.) [%]',
            'CAGR [%]', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Alpha [%]', 'Beta',
            'Max. Drawdown [%]', 'Avg. Drawdown [%]', 'Max. Drawdown Duration', 'Avg. Drawdown Duration',
            '# Trades', 'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]', 'Avg. Trade [%]',
            'Max. Trade Duration', 'Avg. Trade Duration', 'Profit Factor', 'Expectancy [%]',
            'SQN', 'Kelly Criterion'
        ]
        
        aggregated_values = {metric: [] for metric in metrics_to_average}
        
        for sym, stats in results.items():
            if 'Error' in stats:
                continue
            
            for metric in metrics_to_average:
                if metric in stats:
                    val = stats[metric]
                    # Handle Timedelta and other non-numeric types if necessary
                    # backtesting.py returns Timedeltas for durations.
                    # We can convert them to days or seconds for averaging.
                    if isinstance(val, pd.Timedelta):
                        val = val.total_seconds()
                    
                    # Handle NaN or infinite
                    if pd.isna(val) or np.isinf(val):
                        continue
                        
                    aggregated_values[metric].append(val)

        averages = {}
        for metric, values in aggregated_values.items():
            if values:
                avg = np.mean(values)
                # Convert back to Timedelta if it was a duration
                if 'Duration' in metric:
                     averages[metric] = pd.Timedelta(seconds=avg)
                else:
                    averages[metric] = avg
            else:
                averages[metric] = np.nan
                
        return averages
