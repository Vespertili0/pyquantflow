import streamlit as st
import pandas as pd
import sqlite3
import altair as alt
import os

# Set page configuration
st.set_page_config(page_title="Stock & Backtest Dashboard", layout="wide")

st.title("Stock & Backtest Dashboard")

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")

# Database Paths
default_stock_db = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stocks.db")
default_backtest_db = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backtest_results.db")

stock_db_path = st.sidebar.text_input("Stock Database Path", value=default_stock_db)
backtest_db_path = st.sidebar.text_input("Backtest Database Path", value=default_backtest_db)

# --- Helper Functions ---

def get_db_connection(db_path):
    """Creates a database connection."""
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except Exception as e:
        st.error(f"Error connecting to database at {db_path}: {e}")
        return None

def get_tickers(conn):
    """Fetches list of tickers from the database."""
    try:
        query = "SELECT ticker FROM tickers ORDER BY ticker"
        df = pd.read_sql_query(query, conn)
        return df['ticker'].tolist()
    except Exception as e:
        st.error(f"Error fetching tickers: {e}")
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
        df = pd.read_sql_query(query, conn, params=(ticker_id,), parse_dates=['datetime'])
        return df
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()

def get_backtest_results(conn):
    """Fetches backtest results."""
    try:
        query = "SELECT * FROM backtest_results"
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        st.error(f"Error fetching backtest results: {e}")
        return pd.DataFrame()

# --- Main Content ---

tab1, tab2 = st.tabs(["Stock Data", "Backtest Results"])

# --- Tab 1: Stock Data ---
with tab1:
    st.header("Stock Data Visualization")
    
    if os.path.exists(stock_db_path):
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
                            x='datetime:T',
                            tooltip=['datetime', 'open', 'high', 'low', 'close', 'volume']
                        )
                        
                        rule = base.mark_rule().encode(
                            alt.Y('low:Q')
                                .title('Price')
                                .scale(zero=False),
                            alt.Y2('high:Q')
                        )
                        
                        bar = base.mark_bar().encode(
                            y='open:Q',
                            y2='close:Q',
                            color=alt.condition(
                                "datum.open <= datum.close",
                                alt.value("#06982d"),   # Green
                                alt.value("#ae1325")    # Red
                                )  
                        )
                        
                        chart = (rule + bar).properties(
                            width='container',
                            height=600,
                            title=f"{selected_ticker} Price History"
                        ).interactive()
                        
                        st.altair_chart(chart, use_container_width=True)
                        
                        with st.expander("View Raw Data"):
                            st.dataframe(df_stock)
                    else:
                        st.warning(f"No data found for {selected_ticker}")
            else:
                st.warning("No tickers found in the database.")
            
            conn_stock.close()
    else:
        st.error(f"Stock database not found at {stock_db_path}")

# --- Tab 2: Backtest Results ---
with tab2:
    st.header("Backtest Results")
    
    if os.path.exists(backtest_db_path):
        conn_backtest = get_db_connection(backtest_db_path)
        if conn_backtest:
            df_results = get_backtest_results(conn_backtest)
            
            if not df_results.empty:
                # Filter by batch_run_name if available
                if 'batch_run_name' in df_results.columns:
                    batch_runs = df_results['batch_run_name'].unique().tolist()
                    selected_batch = st.multiselect("Filter by Batch Run", batch_runs, default=batch_runs)
                    
                    if selected_batch:
                        df_results = df_results[df_results['batch_run_name'].isin(selected_batch)]
                
                st.dataframe(df_results)
            else:
                st.info("No backtest results found.")
            
            conn_backtest.close()
    else:
        st.error(f"Backtest database not found at {backtest_db_path}")
