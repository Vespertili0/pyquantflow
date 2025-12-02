# pyquantflow

**Analyze. Backtest. Trade.**

A powerful, local-first stock analysis and backtesting framework designed for data persistence and strategy validation. Built on top of `yfinance` and `backtesting.py`, it gives you control over your data and your strategies.

## Installation

```bash
pip install -r requirements.txt
pip install .
```

## Usage

### 1. Setup Stock Database

Initialize your local SQLite database and fetch historical data for your favorite tickers.

```python
from pyquantflow.data.database import DatabaseManager

# Initialize database (defaults to stocks.db, using a separate one for example)
db = DatabaseManager('example_stocks.db')

# Add a ticker and fetch data starting from 2023
db.add_ticker('AAPL', start_year=2023)

# Retrieve data as a Pandas DataFrame
data = db.get_data('AAPL')
print(data.tail())
```

### 2. Run a Single Backtest

Test your trading strategies using the built-in engine.

```python
from pyquantflow.backtesting.engine import BacktestRunner
from pyquantflow.strategies.example_strategy import SmaCross
from pyquantflow.data.database import DatabaseManager

# Get data
db = DatabaseManager('example_stocks.db')
data = db.get_data('AAPL')

# Run backtest
runner = BacktestRunner()
# Note: Ensure data is not empty before running backtest
if not data.empty:
    results = runner.run(SmaCross, data, symbol='AAPL', cash=10000, commission=.002)
    print(f"Return: {results['AAPL']['Return [%]']:.2f}%")
else:
    print("No data available for backtest.")
```

### 3. Run Batch Backtest with Result Persistence

Run backtests for multiple tickers and save results to a SQLite database.

```python
from pyquantflow.backtesting.batchbacktest import BatchBacktester
from pyquantflow.strategies.example_strategy import SmaCross
from pyquantflow.data.database import DatabaseManager

# Get data for multiple tickers
db = DatabaseManager('example_stocks.db')
tickers = ['AAPL', 'MSFT']
data_map = {}
for ticker in tickers:
    data = db.get_data(ticker)
    if not data.empty:
        data_map[ticker] = data

# Run batch backtest
# results will be saved to 'backtest_results.db' by default
backtester = BatchBacktester(results_db_path='backtest_results.db')
results = backtester.run_batch_backtest(data_map, SmaCross, cash=10000, commission=.002)

print("Average Metrics:", results['average_metrics'])
```
