
import pandas as pd
from stock_package.backtest_framework import BatchBacktester
from stock_package.strategies.example_strategy import SmaCross

def test_batch_backtester():
    tickers = ['FMG.AX', 'CBA.AX']
    backtester = BatchBacktester(db_path="stocks.db")
    
    # Run backtest
    results = backtester.run_batch_backtest(tickers, SmaCross, cash=10000, commission=.002)
    
    # Verify structure
    assert 'individual_results' in results
    assert 'average_metrics' in results
    
    # Verify individual results
    for ticker in tickers:
        assert ticker in results['individual_results']
        stats = results['individual_results'][ticker]
        
        # Check for some key metrics
        assert 'Return [%]' in stats
        assert 'Sharpe Ratio' in stats
        assert '# Trades' in stats
        assert 'Max. Drawdown [%]' in stats
        
        print(f"\nResults for {ticker}:")
        print(f"Return: {stats['Return [%]']:.2f}%")
        print(f"Sharpe: {stats['Sharpe Ratio']:.2f}")
        print(f"Trades: {stats['# Trades']}")

    # Verify averages
    print("\nAverage Metrics:")
    for metric, value in results['average_metrics'].items():
        print(f"{metric}: {value}")
        
    assert 'Return [%]' in results['average_metrics']
    assert 'Sharpe Ratio' in results['average_metrics']

if __name__ == "__main__":
    test_batch_backtester()
