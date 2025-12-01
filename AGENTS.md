# AGENTS.md

## Project Overview
This project contains a stock analysis and backtesting package (`stock_package`). It uses a SQLite database to store stock data and `backtesting` library for strategy testing.

## Directory Structure
- `stock_package/`: Main source code.
  - `data/`: Data ingestion and database management.
  - `backtesting/`: Backtesting engine.
  - `db/`: Database models and ORM/interface definitions.
  - `models/`: Machine learning models (if any).
  - `strategies/`: Trading strategies.
- `tests/`: Unit tests.

## Running Tests
The project uses `unittest`. To run tests:
```bash
python -m unittest discover tests
```

Dependencies required for testing:
- `pandas`
- `yfinance`
- `backtesting`

## Database Schema
The SQLite database (`stocks.db`) has two main tables (managed by `stock_package.data.database`):
1. `tickers`: Stores ticker symbols and metadata.
   - `id`: INTEGER PRIMARY KEY
   - `ticker`: TEXT UNIQUE
   - `first_added`: TIMESTAMP
   - `last_updated`: TIMESTAMP
2. `price_data`: Stores historical price data.
   - `id`: INTEGER PRIMARY KEY
   - `ticker_id`: INTEGER (Foreign Key)
   - `datetime`: TIMESTAMP
   - `open`: REAL
   - `high`: REAL
   - `low`: REAL
   - `close`: REAL
   - `volume`: REAL

## CI/CD
A GitHub Actions workflow is set up in `.github/workflows/main.yml`.
It runs tests on:
- Push to `main`
- Pull Request to `main`

The workflow installs dependencies and runs `python -m unittest discover tests`.
