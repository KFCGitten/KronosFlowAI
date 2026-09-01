# ====== PREPROCESSING MODULE (SVAMP-STYLE) ======
import os
import sys
import pandas as pd
import numpy as np
from typing import Tuple, List

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import DB_PATH, SEED
from src.features import FeatureEngine

def prepare_clean_dataset(ticker: str) -> pd.DataFrame:
    """
    Loads, cleans, and runs the feature engineering pipeline for a single ticker.
    This encapsulates technical indicators, handling of missing values, and market regimes.
    """
    print(f"\n========================================\nPREPROCESSING: {ticker}\n========================================")
    fe = FeatureEngine()
    df = fe.prepare_pipeline(ticker)
    
    if df.empty:
        print(f"[-] {ticker} has insufficient data.")
        return pd.DataFrame()
        
    print(f"[+] Loaded and preprocessed {len(df)} records for {ticker}.")
    return df

if __name__ == "__main__":
    from src.ingestion import DEFAULT_TICKERS
    for ticker in DEFAULT_TICKERS[:2]:
        prepare_clean_dataset(ticker)
