# ====== AUTOMATED EDA MODULE (SVAMP-STYLE) ======
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List

# Import from central project config (located at root)
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from config import EDA_DIR, DB_PATH
from src.features import FeatureEngine

def generate_eda(tickers: List[str] = None):
    """
    Generates automated exploratory data analysis plots and statistics,
    saving the visual results to the outputs/eda directory.
    """
    print(f"\n========================================\nGENERATING AUTOMATED EDA\n========================================")
    os.makedirs(EDA_DIR, exist_ok=True)
    
    fe = FeatureEngine()
    
    # 1. Gather some quick aggregated metrics or price paths
    if not tickers:
        from src.ingestion import DEFAULT_TICKERS
        tickers = DEFAULT_TICKERS
        
    all_data = []
    for ticker in tickers[:5]: # Visualise top 5 tickers to prevent plot cluttering
        df = fe.prepare_pipeline(ticker)
        if not df.empty:
            df['ticker'] = ticker
            all_data.append(df)
            
    if not all_data:
        print("[-] No data available to perform EDA.")
        return
        
    df_all = pd.concat(all_data, ignore_index=True)
    
    # --- Plot 1: Technical Indicators Distributions ---
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_all, x='ticker', y='rsi_14', palette='viridis')
    plt.title("RSI (14) Distribution by Ticker")
    plt.xlabel("Ticker")
    plt.ylabel("RSI (14)")
    plt.grid(axis='y', alpha=0.3)
    dist_path = os.path.join(EDA_DIR, "rsi_distribution.png")
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved RSI distribution to: {os.path.relpath(dist_path, project_root)}")
    
    # --- Plot 2: Correlation Heatmap of Base Features ---
    plt.figure(figsize=(10, 8))
    corr = df_all[fe.base_features].corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', fmt=".2f")
    plt.title("Technical Features Correlation Matrix")
    plt.tight_layout()
    corr_path = os.path.join(EDA_DIR, "features_correlation.png")
    plt.savefig(corr_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved correlation heatmap to: {os.path.relpath(corr_path, project_root)}")
    
    # --- Plot 3: Target Class Distribution ---
    plt.figure(figsize=(8, 5))
    class_counts = df_all['target'].value_counts().sort_index()
    # Map 0 -> SELL, 1 -> HOLD, 2 -> BUY
    class_names = class_counts.index.map({0: 'SELL (Down)', 1: 'HOLD', 2: 'BUY (Up)'})
    plt.bar(class_names, class_counts.values, color=['#ff6b6b', '#ffd166', '#06d6a0'])
    plt.title("Target Label Distribution (OOS Horizon)")
    plt.ylabel("Number of Samples")
    for i, v in enumerate(class_counts.values):
        plt.text(i, v + 50, str(v), ha='center', fontweight='bold')
    class_dist_path = os.path.join(EDA_DIR, "class_distribution.png")
    plt.savefig(class_dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved class distribution to: {os.path.relpath(class_dist_path, project_root)}")
    
    print("\n========================================\nEDA GENERATION COMPLETE\n========================================")

if __name__ == "__main__":
    generate_eda()
