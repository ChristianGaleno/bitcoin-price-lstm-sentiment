import os
import torch
import pandas as pd

from src.model import (
    LSTMGRUHybrid, make_sequences, make_loaders,
    train, predict, FEATURES
)
from src.evaluate import inverse_close, print_metrics, plot_results, plot_loss

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Load Data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/feature_engineering.csv", parse_dates=["Date"])
print(f"Data shape: {df.shape}")

# ── Preprocessing ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test, f_scaler, t_scaler = make_sequences(df)
print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

train_loader, val_loader = make_loaders(X_train, y_train, X_test, y_test)

# ── Load Model ───────────────────────────────────────────────────────────────
model = LSTMGRUHybrid(n_features=len(FEATURES))
model.load_state_dict(torch.load("src/checkpoints/best_model.pt"))
model.eval()
print(model)

# ── Evaluasi ──────────────────────────────────────────────────────────────────
y_pred = predict(model, X_test, device=DEVICE)

y_true_inv = inverse_close(y_test, t_scaler)
y_pred_inv = inverse_close(y_pred, t_scaler)

metrics = print_metrics(y_true_inv, y_pred_inv)
plot_results(y_true_inv, y_pred_inv)