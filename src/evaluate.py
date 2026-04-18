import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def inverse_close(y_scaled, target_scaler):
    """Kembalikan nilai ke skala harga asli (USD)."""
    return target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()


def print_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2   = r2_score(y_true, y_pred)

    print("=" * 40)
    print(f"  MAE  : ${mae:,.2f}")
    print(f"  RMSE : ${rmse:,.2f}")
    print(f"  MAPE : {mape:.2f}%")
    print(f"  R²   : {r2:.4f}")
    print("=" * 40)

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def plot_results(y_true, y_pred, dates=None):
    plt.figure(figsize=(14, 5))
    if dates is not None:
        plt.plot(dates, y_true, label="Actual",    color="steelblue")
        plt.plot(dates, y_pred, label="Predicted", color="tomato", linestyle="--")
    else:
        plt.plot(y_true, label="Actual",    color="steelblue")
        plt.plot(y_pred, label="Predicted", color="tomato", linestyle="--")

    plt.title("BTC Price — Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_loss(history):
    plt.figure(figsize=(10, 4))
    plt.plot(history["train_loss"], label="Train Loss", color="steelblue")
    plt.plot(history["val_loss"],   label="Val Loss",   color="tomato")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()