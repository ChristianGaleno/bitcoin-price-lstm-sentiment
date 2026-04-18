import pandas as pd
import os

def feat_engineer(df):
    # Return & Volatilitas
    df["return_1d"] = df["Close"].pct_change()                     # daily return
    df["return_7d"] = df["Close"].pct_change(7)                    # 7-day return
    df["volatility_7d"] = df["Close"].rolling(7).std()             # volatilitas mingguan
    df["volatility_14d"] = df["Close"].rolling(14).std()           # volatilitas 2 mingguan

    # Moving Average
    df["ma_7"] = df["Close"].rolling(7).mean()                     # MA 7 hari
    df["ma_21"] = df["Close"].rolling(21).mean()                   # MA 21 hari
    df["ma_cross"] = df["ma_7"] - df["ma_21"]                      # sinyal golden/death cross

    # Price Range
    df["hl_ratio"] = (df["High"] - df["Low"]) / df["Close"]        # range hari ini relatif
    df["oc_ratio"] = (df["Close"] - df["Open"]) / df["Open"]       # body candle relatif

    df["volume_ma7"] = df["Volume"].rolling(7).mean()              # rata-rata volume
    df["volume_ratio"] = df["Volume"] / df["volume_ma7"]           # volume hari ini vs rata-rata

    df["sentiment_lag1"] = df["sentiment_score"].shift(1)          # sentimen kemarin
    df["sentiment_lag2"] = df["sentiment_score"].shift(2)          # sentimen 2 hari lalu
    df["sentiment_rolling7"] = df["sentiment_score"].rolling(7).mean()  # tren sentimen
    df["rsi_14"] = compute_rsi(df["Close"], 14)
    df = df.dropna().reset_index(drop=True)

    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    
    return 100 - (100 / (1 + rs))


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "processed_data.csv"))
    df = feat_engineer(df)
    df.to_csv(os.path.join(BASE_DIR, "data", "processed", "feature_engineering.csv"), index=False)
    print(df.head(10))
    print(df.shape)

