import pandas as pd
import numpy as np
from .engine import BacktestRunner
from .backtest_database import BacktestDatabaseManager

class BatchBacktester:
    def __init__(self, results_db_path="backtest_results.db"):
        self.results_db = BacktestDatabaseManager(results_db_path)
        self.runner = BacktestRunner()
        self.results = None
        self.strategy_class = None

    def run_batch_backtest(self, data_map, strategy_class, **kwargs):
        """
        Runs backtests for a list of tickers and aggregates results.
        
        Args:
            data_map (dict): Dictionary of {ticker: DataFrame}.
            strategy_class: The strategy class to use.
            **kwargs: Additional arguments for the backtest (cash, commission, strategy params).
        
        Returns:
            dict: A dictionary containing 'individual_results' and 'average_metrics'.
        """
        self.strategy_class = strategy_class

        if not data_map:
            self.results = {'individual_results': {}, 'average_metrics': {}}
            return self.results

        # Run backtests
        individual_results = self.runner.run(strategy_class, data_map, **kwargs)
        
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
