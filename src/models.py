import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.ensemble import RandomForestClassifier

# Add project root to sys.path to resolve src imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import MODEL_DIR as MODELS_DIR, PROJECT_ROOT, SEED
from src.features import FeatureEngine, get_latest_data_date

# --- 1. Attention Mechanism ---
class Attention(nn.Module):
    def __init__(self, hidden_size: int):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # lstm_output shape: [batch, seq_len, hidden_size]
        weights = self.attention(lstm_output) # [batch, seq_len, 1]
        weights = F.softmax(weights, dim=1)  # Softmax over sequence length
        
        # Weighted sum: context vector
        context = torch.sum(weights * lstm_output, dim=1) # [batch, hidden_size]
        return context, weights

# --- 2. PyTorch LSTM with Attention Model ---
class LSTMWithAttention(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, output_size: int = 3, dropout: float = 0.2):
        super(LSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attention = Attention(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, seq_len, input_size]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0)) # [batch, seq_len, hidden_size]
        context, _ = self.attention(lstm_out) # [batch, hidden_size]
        logits = self.fc(context)             # [batch, output_size]
        return logits

# --- 3. Custom Focal Loss for Class Imbalance ---
class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        
        # Focal formula: (1 - pt)^gamma * ce_loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# --- 4. Training Pipeline ---
def train_all_models(tickers: List[str], split_days: int = 180):
    """Loads ticker data, prepares features, splits by date, and trains both RF and LSTM models."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    fe = FeatureEngine()

    # Aggregated lists to train models across all tickers
    X_train_list, y_train_list = [], []

    # Split date: train on older data, validate on the last `split_days`, anchored to the
    # latest date actually present in the DB (not wall-clock "now") so repeated runs are
    # reproducible between ingestions.
    latest_date = pd.Timestamp(get_latest_data_date())
    split_date = (latest_date - pd.Timedelta(days=split_days)).strftime("%Y-%m-%d")
    print(f"Split date for Out-Of-Sample validation (latest data: {latest_date.date()}): {split_date}")
    
    for ticker in tickers:
        print(f"Processing data for {ticker}...")
        df = fe.prepare_pipeline(ticker)
        if df.empty:
            print(f"Skipping {ticker}: Insufficient data.")
            continue
            
        # Filter for training data
        df_train = df[df['record_date'] <= split_date]
        if len(df_train) <= fe.sequence_length:
            continue
            
        X, y, _ = fe.create_sequences(df_train, is_training=True)
        if len(X) > 0:
            X_train_list.append(X)
            y_train_list.append(y)
            
    if not X_train_list:
        print("Error: No training data generated.")
        return
        
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    
    print(f"Total training samples: {X_train.shape[0]}")
    print(f"Class distribution: {np.bincount(y_train)}")
    
    # ------------------
    # A. Train Random Forest
    # ------------------
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=12, random_state=42, n_jobs=-1, class_weight="balanced"
    )
    
    # Flatten sequence dimensions for Random Forest: [samples, seq_len * features]
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    rf_model.fit(X_train_flat, y_train)
    
    rf_path = os.path.join(MODELS_DIR, "random_forest.pkl")
    with open(rf_path, "wb") as f:
        pickle.dump(rf_model, f)
    print(f"Random Forest model saved to: {os.path.relpath(rf_path, PROJECT_ROOT)}")
    
    # Save the scaler from FeatureEngine so we can apply it during validation
    scaler_path = os.path.join(MODELS_DIR, "robust_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(fe.scaler, f)
    print(f"RobustScaler saved to: {os.path.relpath(scaler_path, PROJECT_ROOT)}")
    
    # ------------------
    # B. Train PyTorch LSTM
    # ------------------
    print("Training PyTorch LSTM with Attention model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X_train.shape[2]
    
    lstm_model = LSTMWithAttention(input_size=input_size, hidden_size=64, num_layers=2, output_size=3).to(device)
    
    # Compute class weights for Focal Loss to balance classes
    class_counts = np.bincount(y_train)
    total_samples = sum(class_counts)
    # Inverse frequency weights: total / (classes * count)
    class_weights = total_samples / (len(class_counts) * class_counts + 1e-9)
    weights_tensor = torch.FloatTensor(class_weights).to(device)
    print(f"Class weights applied to loss: {class_weights}")
    
    criterion = FocalLoss(alpha=weights_tensor, gamma=2.0)
    optimizer = torch.optim.AdamW(lstm_model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # PyTorch Dataloader
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), 
        torch.LongTensor(y_train)
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    lstm_model.train()
    epochs = 30
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = lstm_model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            
        avg_loss = epoch_loss / len(dataset)
        print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
        
    lstm_path = os.path.join(MODELS_DIR, "lstm_attention.pth")
    torch.save(lstm_model.state_dict(), lstm_path)
    print(f"LSTM model saved to: {os.path.relpath(lstm_path, PROJECT_ROOT)}")

if __name__ == "__main__":
    from src.ingestion import DEFAULT_TICKERS
    train_all_models(DEFAULT_TICKERS)
