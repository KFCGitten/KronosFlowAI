import os
import sys
import pickle
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

# Add project root to sys.path to resolve src imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import MODEL_DIR as MODELS_DIR, REPORT_PATH, PROJECT_ROOT
from src.features import FeatureEngine, get_latest_data_date
from src.models import LSTMWithAttention

def load_rf_model() -> Tuple[Any, Any]:
    """Loads the trained Random Forest classifier and the associated RobustScaler."""
    rf_path = os.path.join(MODELS_DIR, "random_forest.pkl")
    scaler_path = os.path.join(MODELS_DIR, "robust_scaler.pkl")
    
    with open(rf_path, "rb") as f:
        rf_model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        
    return rf_model, scaler

def load_lstm_model(input_size: int) -> LSTMWithAttention:
    """Loads the trained PyTorch LSTM model with Attention mechanism."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model = LSTMWithAttention(input_size=input_size, hidden_size=64, num_layers=2, output_size=3).to(device)
    
    lstm_path = os.path.join(MODELS_DIR, "lstm_attention.pth")
    lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
    lstm_model.eval()
    return lstm_model

def simulate_portfolio_trading(df_test: pd.DataFrame, signals: List[int], initial_capital: float = 10000.0) -> Dict[str, Any]:
    """
    Simulates trading on test data.
    signals: list of predicted actions for each step (0=SELL, 1=HOLD, 2=BUY).
    """
    cash = initial_capital
    shares = 0.0
    portfolio_values = []
    trades_count = 0
    winning_trades = 0
    last_buy_price = 0.0
    
    # We iterate day by day matching aligned signals
    for idx, row in df_test.iterrows():
        current_close = row['close']
        signal = signals[idx]
        
        # BUY signal
        if signal == 2 and shares == 0:
            shares = cash / current_close
            cash = 0
            last_buy_price = current_close
            trades_count += 1
            
        # SELL signal
        elif signal == 0 and shares > 0:
            cash = shares * current_close
            shares = 0
            if current_close > last_buy_price:
                winning_trades += 1
                
        # Record daily portfolio value
        port_val = cash + (shares * current_close)
        portfolio_values.append(port_val)
        
    final_value = cash + (shares * df_test['close'].iloc[-1])
    total_return_pct = ((final_value / initial_capital) - 1.0) * 100.0
    
    # Calculate Buy and Hold Return
    bh_shares = initial_capital / df_test['close'].iloc[0]
    bh_final_value = bh_shares * df_test['close'].iloc[-1]
    bh_return_pct = ((bh_final_value / initial_capital) - 1.0) * 100.0
    
    win_rate = (winning_trades / trades_count) * 100.0 if trades_count > 0 else 0.0
    
    return {
        "final_value": final_value,
        "return_pct": total_return_pct,
        "buy_and_hold_return_pct": bh_return_pct,
        "total_trades": trades_count,
        "win_rate": win_rate,
        "portfolio_history": portfolio_values
    }

CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = ['DOWN', 'HOLD', 'UP']

def compute_roc_pr(y_true: List[int], y_proba: np.ndarray) -> Dict[str, Any]:
    """Computes one-vs-rest ROC and precision-recall curves per class, plus macro ROC-AUC.

    y_proba must have columns ordered [DOWN, HOLD, UP] (i.e. CLASS_LABELS order).
    """
    y_true_bin = label_binarize(y_true, classes=CLASS_LABELS)

    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
        per_class[name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "roc_auc": float(auc(fpr, tpr)),
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "pr_auc": float(average_precision_score(y_true_bin[:, i], y_proba[:, i])),
        }

    macro_roc_auc = float(roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr"))
    return {"per_class": per_class, "macro_roc_auc": macro_roc_auc}

def run_evaluation(tickers: List[str], split_days: int = 180):
    """Runs out-of-sample evaluation and backtests for both RF and LSTM across validation tickers."""
    fe = FeatureEngine()
    
    # 1. Load RF model and scaler
    try:
        rf_model, scaler = load_rf_model()
        fe.scaler = scaler # Align scale
    except Exception as e:
        print(f"Error loading models: {e}. Did you run training first?")
        return
        
    latest_date = pd.Timestamp(get_latest_data_date())
    split_date = (latest_date - pd.Timedelta(days=split_days)).strftime("%Y-%m-%d")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    y_true_all = []
    y_pred_rf_all = []
    y_pred_lstm_all = []
    y_proba_rf_all = []
    y_proba_lstm_all = []

    rf_backtest_stats = []
    lstm_backtest_stats = []
    
    # Load LSTM on first success
    lstm_model = None
    
    for ticker in tickers:
        print(f"Evaluating ticker {ticker}...")
        df = fe.prepare_pipeline(ticker)
        if df.empty:
            continue
            
        # Select out-of-sample validation data
        df_test = df[df['record_date'] > split_date].copy()
        if len(df_test) <= fe.sequence_length:
            print(f"Skipping {ticker}: Insufficient test data.")
            continue
            
        # Create sequences using the fitted scaler
        X, y, dates = fe.create_sequences(df_test, is_training=False)
        if len(X) == 0:
            continue
            
        # Align test dataframe index to match target dates exactly
        df_test_aligned = df_test[df_test['record_date'].isin(dates)].reset_index(drop=True)
        
        # A. Predict with Random Forest
        X_flat = X.reshape(X.shape[0], -1)
        rf_preds = rf_model.predict(X_flat)
        # Reindex predict_proba columns to the fixed [DOWN, HOLD, UP] order, in case
        # rf_model.classes_ ever differs from that order (e.g. a rare class missing at fit time).
        rf_proba_raw = rf_model.predict_proba(X_flat)
        rf_proba = np.zeros((rf_proba_raw.shape[0], len(CLASS_LABELS)))
        for col, cls in enumerate(rf_model.classes_):
            rf_proba[:, CLASS_LABELS.index(cls)] = rf_proba_raw[:, col]

        # B. Predict with LSTM
        if lstm_model is None:
            lstm_model = load_lstm_model(input_size=X.shape[2])

        with torch.no_grad():
            lstm_outputs = lstm_model(torch.FloatTensor(X).to(device))
            lstm_preds = torch.argmax(lstm_outputs, dim=1).cpu().numpy()
            lstm_proba = F.softmax(lstm_outputs, dim=1).cpu().numpy()

        y_true_all.extend(y.tolist())
        y_pred_rf_all.extend(rf_preds.tolist())
        y_pred_lstm_all.extend(lstm_preds.tolist())
        y_proba_rf_all.append(rf_proba)
        y_proba_lstm_all.append(lstm_proba)
        
        # Run Backtests
        rf_bt = simulate_portfolio_trading(df_test_aligned, rf_preds)
        lstm_bt = simulate_portfolio_trading(df_test_aligned, lstm_preds)
        
        rf_backtest_stats.append({
            "ticker": ticker,
            "return_pct": rf_bt["return_pct"],
            "bh_return_pct": rf_bt["buy_and_hold_return_pct"],
            "trades": rf_bt["total_trades"],
            "win_rate": rf_bt["win_rate"]
        })
        
        lstm_backtest_stats.append({
            "ticker": ticker,
            "return_pct": lstm_bt["return_pct"],
            "bh_return_pct": lstm_bt["buy_and_hold_return_pct"],
            "trades": lstm_bt["total_trades"],
            "win_rate": lstm_bt["win_rate"]
        })
        
    if not y_true_all:
        print("No evaluation data generated.")
        return

    y_proba_rf_all = np.concatenate(y_proba_rf_all, axis=0)
    y_proba_lstm_all = np.concatenate(y_proba_lstm_all, axis=0)

    # --- Classification Reports ---
    target_names = CLASS_NAMES
    
    print("\n" + "="*50)
    print("🤖 RANDOM FOREST CLASSIFICATION REPORT")
    rf_report = classification_report(y_true_all, y_pred_rf_all, target_names=target_names, output_dict=True, zero_division=0)
    print(classification_report(y_true_all, y_pred_rf_all, target_names=target_names, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true_all, y_pred_rf_all))
    
    print("\n" + "="*50)
    print("🧠 LSTM WITH ATTENTION CLASSIFICATION REPORT")
    lstm_report = classification_report(y_true_all, y_pred_lstm_all, target_names=target_names, output_dict=True, zero_division=0)
    print(classification_report(y_true_all, y_pred_lstm_all, target_names=target_names, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true_all, y_pred_lstm_all))
    print("="*50 + "\n")
    
    # --- Financial Reports ---
    rf_avg_return = np.mean([x["return_pct"] for x in rf_backtest_stats])
    bh_avg_return = np.mean([x["bh_return_pct"] for x in rf_backtest_stats])
    lstm_avg_return = np.mean([x["return_pct"] for x in lstm_backtest_stats])
    
    print(f"Random Forest Avg Return: {rf_avg_return:.2f}% (Buy & Hold Avg: {bh_avg_return:.2f}%)")
    print(f"LSTM Attention Avg Return: {lstm_avg_return:.2f}% (Buy & Hold Avg: {bh_avg_return:.2f}%)")
    
    # Compute confusion matrices
    rf_cm = confusion_matrix(y_true_all, y_pred_rf_all).tolist()
    lstm_cm = confusion_matrix(y_true_all, y_pred_lstm_all).tolist()

    # --- AUC-ROC & Precision-Recall (one-vs-rest per class) ---
    rf_roc_pr = compute_roc_pr(y_true_all, y_proba_rf_all)
    lstm_roc_pr = compute_roc_pr(y_true_all, y_proba_lstm_all)

    print(f"Random Forest macro ROC-AUC: {rf_roc_pr['macro_roc_auc']:.3f}")
    print(f"LSTM Attention macro ROC-AUC: {lstm_roc_pr['macro_roc_auc']:.3f}")

    # Save results to json for Streamlit dashboard
    summary_report = {
        "random_forest": {
            "classification": rf_report,
            "confusion_matrix": rf_cm,
            "backtests": rf_backtest_stats,
            "avg_return": rf_avg_return,
            "roc_pr": rf_roc_pr
        },
        "lstm": {
            "classification": lstm_report,
            "confusion_matrix": lstm_cm,
            "backtests": lstm_backtest_stats,
            "avg_return": lstm_avg_return,
            "roc_pr": lstm_roc_pr
        },
        "buy_and_hold_avg_return": bh_avg_return
    }
    
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(summary_report, f, indent=4)
    print(f"\nEvaluation summary report saved to: {os.path.relpath(REPORT_PATH, PROJECT_ROOT)}")

if __name__ == "__main__":
    from src.ingestion import DEFAULT_TICKERS
    run_evaluation(DEFAULT_TICKERS)
