import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ── Konstanta ────────────────────────────────────────────────────────────────
SEQ_LEN = 14
BATCH_SIZE = 32

FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "return_1d", "return_7d", "volatility_7d", "volatility_14d",
    "ma_7", "ma_21", "ma_cross",
    "hl_ratio", "oc_ratio",
    "volume_ma7", "volume_ratio",
    "sentiment_score", "sentiment_lag1", "sentiment_lag2", "sentiment_rolling7",
    "rsi_14"
]
TARGET = "Close"


# ── Dataset ───────────────────────────────────────────────────────────────────
class BTCDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Preprocessing ─────────────────────────────────────────────────────────────
def make_sequences(df):
    """Scale fitur + buat sliding window sequences."""
    df = df.copy().sort_values("Date").reset_index(drop=True)

    # Target: Close hari besok
    df["target"] = df[TARGET].shift(-1)
    df = df.dropna().reset_index(drop=True)

    feature_scaler = MinMaxScaler()
    target_scaler  = MinMaxScaler()

    X_scaled = feature_scaler.fit_transform(df[FEATURES].values)
    y_scaled = target_scaler.fit_transform(df[["target"]].values)

    X_seq, y_seq = [], []
    for i in range(len(df) - SEQ_LEN):
        X_seq.append(X_scaled[i : i + SEQ_LEN])
        y_seq.append(y_scaled[i + SEQ_LEN])

    X_seq = np.array(X_seq)  # (n, SEQ_LEN, n_features)
    y_seq = np.array(y_seq)  # (n, 1)

    # Split 80% train / 20% test (tidak random karena time series!)
    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    return X_train, X_test, y_train, y_test, feature_scaler, target_scaler


def make_loaders(X_train, y_train, X_test, y_test):
    train_ds = BTCDataset(X_train, y_train)
    test_ds  = BTCDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


# ── Model ─────────────────────────────────────────────────────────────────────
class LSTMGRUHybrid(nn.Module):
    def __init__(self, n_features, lstm_hidden=128, gru_hidden=64, fc_hidden=32, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        self.gru = nn.GRU(
            input_size=lstm_hidden,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(gru_hidden, fc_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden, 1)

    def forward(self, x):
        # x: (batch, SEQ_LEN, n_features)
        out, _ = self.lstm(x)           # (batch, SEQ_LEN, lstm_hidden)
        out, _ = self.gru(out)          # (batch, SEQ_LEN, gru_hidden)
        out = self.dropout(out[:, -1])  # ambil timestep terakhir
        out = self.relu(self.fc1(out))
        return self.fc2(out)            # (batch, 1)


# ── Training Loop ─────────────────────────────────────────────────────────────
def train(model, train_loader, val_loader, epochs=100, lr=1e-3, patience=15, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validasi
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_losses.append(criterion(pred, y_batch).item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "checkpoints/best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping di epoch {epoch+1}")
                break

    # Load bobot terbaik
    model.load_state_dict(torch.load("checkpoints/best_model.pt"))
    return history


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(model, X, device="cpu"):
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy()
    return preds


# ── Test Run ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.makedirs("checkpoints", exist_ok=True)

    df = pd.read_csv("../data/processed/feature_engineering.csv", parse_dates=["Date"])

    X_train, X_test, y_train, y_test, f_scaler, t_scaler = make_sequences(df)
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

    train_loader, val_loader = make_loaders(X_train, y_train, X_test, y_test)

    model = LSTMGRUHybrid(n_features=len(FEATURES))
    print(model)

    history = train(model, train_loader, val_loader, epochs=100, lr=0.001, patience=15)
    print("Training selesai ✅")