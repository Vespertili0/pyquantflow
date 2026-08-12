import streamlit as st
import pandas as pd
import sqlite3
import altair as alt
import os
import json
from pathlib import Path

# Set page configuration
st.set_page_config(page_title="Stock & Backtest Dashboard", layout="wide")

st.title("Stock & Backtest Dashboard")

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")

# Database Paths
BASE_DIR = Path(__file__).resolve().parent.parent
default_stock_db = str(BASE_DIR / "stocks.db")
default_backtest_db = str(BASE_DIR / "backtest_results.db")

stock_db_path = st.sidebar.text_input("Stock Database Path", value=default_stock_db)
backtest_db_path = st.sidebar.text_input(
    "Backtest Database Path", value=default_backtest_db
)

# --- Helper Functions ---


def is_safe_path(path_str):
    """Checks if a path is safe to access (relative to BASE_DIR)."""
    try:
        requested_path = Path(path_str).resolve()
        return requested_path.is_relative_to(BASE_DIR)
    except Exception:
        return False


def get_db_connection(db_path):
    """Creates a database connection."""
    if not is_safe_path(db_path):
        st.error("Error connecting to database.")
        return None
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except Exception:
        st.error("Error connecting to database.")
        return None


def get_tickers(conn):
    """Fetches list of tickers from the database."""
    try:
        query = "SELECT ticker FROM tickers ORDER BY ticker"
        df = pd.read_sql_query(query, conn)
        return df["ticker"].tolist()
    except Exception:
        st.error("Error fetching tickers.")
        return []


def get_stock_data(conn, ticker):
    """Fetches stock data for a given ticker."""
    try:
        # First get ticker_id
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM tickers WHERE ticker = ?", (ticker,))
        row = cursor.fetchone()
        if not row:
            return pd.DataFrame()

        ticker_id = row[0]
        query = """
            SELECT datetime, open, high, low, close, volume
            FROM price_data
            WHERE ticker_id = ?
            ORDER BY datetime
        """
        df = pd.read_sql_query(
            query, conn, params=(ticker_id,), parse_dates=["datetime"]
        )
        return df
    except Exception:
        st.error("Error fetching stock data.")
        return pd.DataFrame()


def get_backtest_results(conn):
    """Fetches backtest results."""
    try:
        query = "SELECT * FROM backtest_results"
        df = pd.read_sql_query(query, conn)
        if not df.empty and "metrics" in df.columns:
            # Parse JSON metrics and expand into columns
            df["metrics"] = df["metrics"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
            metrics_df = pd.json_normalize(df["metrics"])
            df = pd.concat([df.drop("metrics", axis=1), metrics_df], axis=1)
        return df
    except Exception:
        st.error("Error fetching backtest results.")
        return pd.DataFrame()


# --- Main Content ---

tab1, tab2 = st.tabs(["Stock Data", "Backtest Results"])

# --- Tab 1: Stock Data ---
with tab1:
    st.header("Stock Data Visualisation")

    if is_safe_path(stock_db_path) and os.path.exists(stock_db_path):
        conn_stock = get_db_connection(stock_db_path)
        if conn_stock:
            tickers = get_tickers(conn_stock)

            if tickers:
                selected_ticker = st.selectbox("Select Ticker", tickers)

                if selected_ticker:
                    df_stock = get_stock_data(conn_stock, selected_ticker)

                    if not df_stock.empty:
                        st.write(f"Displaying data for **{selected_ticker}**")

                        # Altair Candle Chart
                        base = alt.Chart(df_stock).encode(
                            x="datetime:T",
                            tooltip=[
                                "datetime",
                                "open",
                                "high",
                                "low",
                                "close",
                                "volume",
                            ],
                        )

                        rule = base.mark_rule().encode(
                            alt.Y("low:Q").title("Price").scale(zero=False),
                            alt.Y2("high:Q"),
                        )

                        bar = base.mark_bar().encode(
                            y="open:Q",
                            y2="close:Q",
                            color=alt.condition(
                                "datum.open <= datum.close",
                                alt.value("#06982d"),  # Green
                                alt.value("#ae1325"),  # Red
                            ),
                        )

                        chart = (
                            (rule + bar)
                            .properties(
                                width="container",
                                height=600,
                                title=f"{selected_ticker} Price History",
                            )
                            .interactive()
                        )

                        st.altair_chart(chart, use_container_width=True)

                        with st.expander("View Raw Data"):
                            st.dataframe(df_stock)
                    else:
                        st.warning(f"No data found for {selected_ticker}")
            else:
                st.warning("No tickers found in the database.")

            conn_stock.close()
    else:
        st.error("Stock database not found.")

# --- Tab 2: Backtest Results ---
with tab2:
    st.header("Backtest Results")

    if is_safe_path(backtest_db_path) and os.path.exists(backtest_db_path):
        conn_backtest = get_db_connection(backtest_db_path)
        if conn_backtest:
            df_results = get_backtest_results(conn_backtest)

            if not df_results.empty:
                # Filter by batch_run_name if available
                if "batch_run_name" in df_results.columns:
                    batch_runs = df_results["batch_run_name"].unique().tolist()
                    selected_batch = st.multiselect(
                        "Filter by Batch Run", batch_runs, default=batch_runs
                    )

                    if selected_batch:
                        df_results = df_results[
                            df_results["batch_run_name"].isin(selected_batch)
                        ]

                st.dataframe(df_results)
            else:
                st.info("No backtest results found.")

            conn_backtest.close()
    else:
        st.error("Backtest database not found.")
