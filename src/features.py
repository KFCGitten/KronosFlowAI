import os
import sys
import sqlite3
import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import DB_PATH

def get_latest_data_date(db_path: str = DB_PATH) -> str:
    """Returns the most recent record_date in the database.

    Used as a fixed reference point for the train/out-of-sample split, instead of
    ``pd.Timestamp.now()``. Anchoring the split to the data itself (rather than to
    wall-clock time) makes training and evaluation runs reproducible as long as the
    database isn't re-ingested in between.
    """
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT MAX(record_date) FROM historical_prices").fetchone()
    conn.close()
    return row[0]

class FeatureEngine:
    def __init__(self, sequence_length: int = 30):
        self.sequence_length = sequence_length
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=3)
        self.kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
        
        # Define base features that will be used for modeling
        self.base_features = [
            'rsi_14', 'macd_diff', 'sma_ratio_10_50', 
            'dist_sma_200', 'atr_norm', 'stoch_k', 'bb_width',
            'turnover_surge', 'pca_1', 'pca_2', 'pca_3', 'market_regime'
        ]

    def load_raw_data(self, ticker: str, db_path: str = DB_PATH) -> pd.DataFrame:
        """Loads price records for a given ticker from the SQLite database."""
        conn = sqlite3.connect(db_path)
        query = """
        SELECT ticker, record_date, open, high, low, close, volume 
        FROM historical_prices 
        WHERE ticker = ? 
        ORDER BY record_date ASC
        """
        df = pd.read_sql_query(query, conn, params=(ticker.upper(),))
        conn.close()
        return df

    def compute_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates standard technical indicators for trading strategy."""
        df = df.copy()
        
        # 1. Trend
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_200'] = df['close'].rolling(200).mean()
        df['sma_ratio_10_50'] = df['sma_10'] / (df['sma_50'] + 1e-9)
        df['dist_sma_200'] = (df['close'] / (df['sma_200'] + 1e-9)) - 1
        
        # 2. Momentum (RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # 3. Trend Convergence (MACD)
        macd = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        signal = macd.ewm(span=9).mean()
        df['macd_diff'] = macd - signal
        
        # 4. Volatility (ATR & Bollinger Bands)
        df['atr_norm'] = df['close'].rolling(14).std() / (df['close'] + 1e-9)
        bb_middle = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_width'] = (bb_middle + 2*bb_std - (bb_middle - 2*bb_std)) / (bb_middle + 1e-9)
        
        # 5. Oscillators (Stochastic K)
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14 + 1e-9))
        
        # 6. Volume Surge
        df['turnover'] = df['volume'] * df['close']
        df['turnover_surge'] = df['turnover'] / (df['turnover'].rolling(20).mean() + 1e-9)
        
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handles missing values by forward/backward filling and replacing infinities."""
        df = df.replace([np.inf, -np.inf], np.nan)
        # Forward fill then backward fill to make sure no NaNs remain
        df = df.ffill().bfill().fillna(0)
        return df

    def apply_unsupervised_learning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies PCA and K-Means clustering as unsupervised features representing market regimes."""
        df = df.copy()
        
        # Select indicators for unsupervised learning
        unsupervised_cols = [
            'rsi_14', 'macd_diff', 'sma_ratio_10_50', 
            'dist_sma_200', 'atr_norm', 'stoch_k', 'bb_width', 'turnover_surge'
        ]
        
        features_to_fit = df[unsupervised_cols].values
        
        # Scale temporarily for unsupervised fit
        scaled_temp = RobustScaler().fit_transform(features_to_fit)
        
        # 1. Dimensionality Reduction (PCA)
        pca_features = self.pca.fit_transform(scaled_temp)
        df['pca_1'] = pca_features[:, 0]
        df['pca_2'] = pca_features[:, 1]
        df['pca_3'] = pca_features[:, 2]
        
        # 2. Clustering (K-Means)
        df['market_regime'] = self.kmeans.fit_predict(scaled_temp)
        
        return df

    def generate_labels(
        self,
        df: pd.DataFrame,
        sell_threshold: float = -0.02,
        buy_threshold: float = 0.02,
        horizon_days: int = 5,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Generates look-ahead target labels (0=SELL, 1=HOLD, 2=BUY) based on a 5-day horizon.

        Thresholds are symmetric (+/-2%) by default. The original version used an asymmetric
        -3% / +1.5% split, which made BUY trigger far more easily than SELL and starved the
        HOLD class further — see the "Iteration: v1 -> v2" section in the project notebook.
        """
        df = df.copy()
        labels = []

        for i in range(len(df) - horizon_days):
            current_price = df['close'].iloc[i]
            future_highs = df['high'].iloc[i + 1 : i + 1 + horizon_days]
            future_lows = df['low'].iloc[i + 1 : i + 1 + horizon_days]

            label = 1 # Default is HOLD

            # Check if any future low drops to/below sell_threshold (Trigger SELL)
            for low in future_lows:
                if (low / current_price) - 1 <= sell_threshold:
                    label = 0
                    break

            # If not triggered SELL, check if any future high reaches/exceeds buy_threshold (Trigger BUY)
            if label == 1:
                for high in future_highs:
                    if (high / current_price) - 1 >= buy_threshold:
                        label = 2
                        break

            labels.append(label)
            
        # For the last 5 records where future is unavailable, label as HOLD (1)
        labels.extend([1] * horizon_days)
        df['target'] = labels
        return df

    def prepare_pipeline(self, ticker: str) -> pd.DataFrame:
        """Runs the entire feature engineering pipeline on a ticker and returns the final DataFrame."""
        df = self.load_raw_data(ticker)
        if df.empty or len(df) < 200:
            return pd.DataFrame()
            
        df = self.compute_technical_indicators(df)
        df = self.handle_missing_values(df)
        df = self.apply_unsupervised_learning(df)
        df = self.generate_labels(df)
        
        # Final cleanup
        df = self.handle_missing_values(df)
        return df

    def create_sequences(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Splits scaled dataset into 3D sequences [samples, seq_len, features] and targets."""
        if df.empty or len(df) <= self.sequence_length:
            return np.array([]), np.array([]), []
            
        # Fit scaler on features
        feature_data = df[self.base_features].values
        if is_training:
            scaled_features = self.scaler.fit_transform(feature_data)
        else:
            scaled_features = self.scaler.transform(feature_data)
            
        X, y, dates = [], [], []
        target_values = df['target'].values
        record_dates = df['record_date'].values
        
        for i in range(len(df) - self.sequence_length - 5):
            X.append(scaled_features[i : i + self.sequence_length])
            y.append(target_values[i + self.sequence_length - 1])
            dates.append(record_dates[i + self.sequence_length - 1])
            
        return np.array(X), np.array(y), dates

if __name__ == "__main__":
    # Test script if executed directly
    print("Testing FeatureEngine...")
    fe = FeatureEngine()
    # Let's see if we have database file yet
    if os.path.exists(DB_PATH):
        # Pick a ticker if exists in database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT ticker FROM historical_prices LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        
        if row:
            ticker = row[0]
            print(f"Loading data for {ticker}...")
            df = fe.prepare_pipeline(ticker)
            print("Dataframe shape:", df.shape)
            print("First 3 rows:")
            print(df[['record_date', 'close', 'pca_1', 'market_regime', 'target']].head(3))
            
            X, y, dates = fe.create_sequences(df)
            print("Sequences shape:", X.shape)
            print("Targets shape:", y.shape)
        else:
            print("Database is empty. Please run ingestion first.")
    else:
        print("Database not found. Run ingestion to create it.")
