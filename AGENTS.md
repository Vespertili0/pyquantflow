# AGENTS.md - pyquantflow Context & Reminders

## Project Goal
A local-first stock analysis and backtesting framework designed for data persistence and strategy validation. It bridges the gap between simple technical analysis and Financial Machine Learning.

## Key Constraints & Environment
- **OS**: Linux
- **Core Libraries**: 
  - `yfinance`: For fetching market data.
  - `backtesting`: For event-driven backtesting execution.
  - `pandas` & `numpy`: For data manipulation and numerical operations.
  - `scikit-learn` & `skfolio`: For machine learning modelling and advanced portfolio cross-validation (Walk-Forward, CPCV).
  - `mlflow` & `optuna`: For MLOps tracking and hyperparameter optimisation.
  - `jax`: Advanced numerical acceleration.
- **Primary Mechanism**: Local-first data architecture relying heavily on SQLite-backed persistence (`stocks.db` for market data, `backtest_results.db` for strategy results).

## Architecture
- **`pyquantflow/`**: Main source code containing the core modules.
  - **`data/`**: Data ingestion (`database.py`), advanced features (Fractional Diff, Trend Scanning, Triple Barrier), and `pandas-pipe` indicators.
  - **`model/`**: MLOps workflow, cross-validation (`PurgedKFoldCV`), hyperparameter optimisation (`optuna`), and `mlflow` tracking.
  - **`backtesting/`**: Event-driven backtesting engine (wraps `backtesting.py`) and batch execution with SQLite persistence.
  - **`portfolio/`**: `StrategyLab` for analysing/validating portfolio strategies via walk-forward and combinatorial purged cross-validation using `skfolio`.
  - **`collection/`**: Definitions of major ticker groupings (e.g., ASX 20/50/100/200).
  - **`strategies/`**: Trading strategies.
- **`dashboard/`**: Streamlit application (`dashboard.py`) for visual exploration of price data and backtest results.
- **`tests/`**: Unit tests utilising `unittest`.

### Database Schema
The SQLite database (`stocks.db`) has two main tables:
1. `tickers`: Stores ticker symbols and metadata (`id`, `ticker`, `first_added`, `last_updated`).
2. `price_data`: Stores historical price data (`id`, `ticker_id`, `datetime`, `open`, `high`, `low`, `close`, `volume`).

The Backtest Results database (`backtest_results.db`) has one main table:
1. `backtest_results`: Stores results of backtest runs (`id`, `ticker`, `batch_run_name`, `metrics`).

### CI/CD
A GitHub Actions workflow is set up in `.github/workflows/tests.yml`. It runs `unittest` tests via `python -m unittest discover tests` on Push and Pull Request to `main` and `sandbox` branches across Python versions 3.10 to 3.12. A separate `release.yml` handles releases.

## Current Status
- Core SQLite database managers for market data and batch backtesting results are fully implemented.
- Financial ML concepts (e.g., Triple-Barrier labelling) and MLOps workflows are integrated.
- `StrategyLab` portfolio journey simulations and cross-validation pipelines (WalkForward, CPCV) are operational.
- A Streamlit dashboard is available to visualise ingested data and backtest outcomes.

## Reminders
- [ ] Run tests via `python -m unittest discover tests` and ensure `requirements.txt` dependencies are installed before testing.
- [ ] Monitor SQLite database sizes (`stocks.db` and `backtest_results.db`) and check for query efficiency when retrieving data for large batch backtests.
- [ ] Apply British spelling rules throughout documentation and code (e.g., Optimiser instead of Optimizer, labelling instead of labeling, visualise instead of visualize).
- [ ] Ensure CI/CD GitHub Actions workflow passes on `main` and `sandbox` branch pushes and PRs.

## Next Steps
1. **Develop Statistical-Backtesting integration**: Follow up on the "in development" section from the README to implement statistical backtesting methodologies.
2. **Expand Pipeline Orchestration**: Introduce a workflow orchestrator (such as Prefect or Airflow) or advanced ML pipeline abstractions to string together data ingestion, labelling, feature engineering, model training, and backtesting into automated workflows.
3. **Enhance the Dashboard**: Add more analytical pages to the Streamlit dashboard, such as a dedicated tab for visualising `StrategyLab` cross-validation metrics.
