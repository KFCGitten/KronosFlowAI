import os
import sys
import sqlite3
import logging
import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import List, Optional

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import DB_PATH, PROJECT_ROOT

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DataIngestion")

# Default OMXS30 tickers list (using Yahoo Finance tickers for Sweden)
DEFAULT_TICKERS = [
    "VOLV-B.ST", "ERIC-B.ST", "SEB-A.ST", "SHB-A.ST", "SWED-A.ST",
    "HM-B.ST", "AZN.ST", "ABB.ST", "TELIA.ST", "TELE2-B.ST",
    "SAND.ST", "ALFA.ST", "HEXA-B.ST", "SKF-B.ST", "INVE-B.ST",
    "SCA-B.ST", "BOL.ST", "NIBE-B.ST", "ATCO-A.ST", "ESSITY-B.ST"
]

def init_db(db_path: str = DB_PATH):
    """Initializes the SQLite database and creates the price table if it doesn't exist."""
    logger.info(f"Initializing database at: {os.path.relpath(db_path, PROJECT_ROOT)}")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS historical_prices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        record_date TEXT NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume INTEGER,
        UNIQUE(ticker, record_date)
    );
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully.")

def fetch_and_store_ticker(ticker: str, start_date: str, end_date: str, db_path: str = DB_PATH) -> int:
    """Fetches historical daily data for a ticker and stores it in SQLite."""
    logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    
    try:
        # Fetch raw data from yfinance
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval="1d")
        
        if df.empty:
            logger.warning(f"No data returned for {ticker}.")
            return 0
        
        # Clean and format the DataFrame
        df = df.reset_index()
        df['record_date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        df['ticker'] = ticker.upper()
        
        # Select and rename columns to match our DB schema
        db_cols = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'ticker': 'ticker',
            'record_date': 'record_date'
        }
        
        # Keep only required columns
        df = df[[col for col in db_cols.keys() if col in df.columns]]
        df = df.rename(columns=db_cols)
        
        # Open connection
        conn = sqlite3.connect(db_path)
        
        # Use INSERT OR REPLACE strategy to handle duplicates gracefully
        # Convert df to list of tuples for sqlite executemany
        records = df[['ticker', 'record_date', 'open', 'high', 'low', 'close', 'volume']].values.tolist()
        
        cursor = conn.cursor()
        cursor.executemany("""
        INSERT OR REPLACE INTO historical_prices (ticker, record_date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, records)
        
        inserted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        logger.info(f"Stored {len(records)} price records for {ticker}.")
        return len(records)
        
    except Exception as e:
        logger.error(f"Failed to fetch or store data for {ticker}: {e}")
        return 0

def run_ingestion(tickers: List[str] = DEFAULT_TICKERS, start_date: str = "2016-01-01", end_date: Optional[str] = None):
    """Initializes DB and fetches price history for all tickers in the list."""
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
        
    init_db()
    
    total_records = 0
    success_count = 0
    
    for ticker in tickers:
        records = fetch_and_store_ticker(ticker, start_date, end_date)
        if records > 0:
            success_count += 1
            total_records += records
            
    logger.info(f"Ingestion complete. Successfully processed {success_count}/{len(tickers)} tickers. Total records: {total_records}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest historical stock data into SQLite.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS, help="List of tickers to fetch.")
    parser.add_argument("--start", default="2016-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD), default is today.")
    
    args = parser.parse_args()
    run_ingestion(tickers=args.tickers, start_date=args.start, end_date=args.end)
