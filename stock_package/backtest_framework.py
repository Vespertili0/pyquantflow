import pandas as pd
import numpy as np
from .data.database import DatabaseManager
from .backtesting.engine import BacktestRunner

class BatchBacktester:
    def __init__(self, db_path="stocks.db"):
        self.db_manager = DatabaseManager(db_path)
        self.runner = BacktestRunner()

    def run_batch_backtest(self, tickers, strategy_class, **kwargs):
        """
        Runs backtests for a list of tickers and aggregates results.
        
        Args:
            tickers (list): List of ticker symbols.
            strategy_class: The strategy class to use.
            **kwargs: Additional arguments for the backtest (cash, commission, strategy params).
        
        Returns:
            dict: A dictionary containing 'individual_results' and 'average_metrics'.
        """
        data_map = {}
        for ticker in tickers:
            # Ensure we have data. 
            # Check if data exists, if not try to add it (simple check)
            df = self.db_manager.get_data(ticker)
            if df.empty:
                print(f"Data for {ticker} not found in DB. Attempting to fetch...")
                self.db_manager.add_ticker(ticker)
                df = self.db_manager.get_data(ticker)
            
            if not df.empty:
                data_map[ticker] = df
            else:
                print(f"Could not retrieve data for {ticker}. Skipping.")

        if not data_map:
            return {'individual_results': {}, 'average_metrics': {}}

        # Run backtests
        individual_results = self.runner.run(strategy_class, data_map, **kwargs)
        
        # Calculate averages
        average_metrics = self._calculate_averages(individual_results)
        
        return {
            'individual_results': individual_results,
            'average_metrics': average_metrics
        }

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
