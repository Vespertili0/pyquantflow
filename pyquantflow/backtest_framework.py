import pandas as pd
import numpy as np
from .backtesting.engine import BacktestRunner
from .backtesting.backtest_database import BacktestDatabaseManager

class BatchBacktester:
    def __init__(self, results_db_path="backtest_results.db"):
        self.results_db = BacktestDatabaseManager(results_db_path)
        self.runner = BacktestRunner()

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

    def save_batch_results(self, results, strategy_class):
        """
        Saves the batch backtest results to the database with a unique batch run name.

        Args:
            results (dict): The dictionary containing 'individual_results' (as returned by run_batch_backtest).
            strategy_class: The strategy class used for the backtest.

        Returns:
            str: The generated batch run name.
        """
        from datetime import datetime

        individual_results = results.get('individual_results', {})
        batch_run_name = f"{datetime.now().strftime('%Y-%m-%d')}_{strategy_class.__name__}"

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
