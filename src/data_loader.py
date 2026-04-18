import yfinance as yf
import pandas as pd
import os

def load_price_data(ticker: str = "BTC-USD",
                    start: str = "2021-11-05",
                    end: str = "2024-09-13") -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, multi_level_index=False)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.index = pd.to_datetime(df.index)
    df.index.name = 'Date'
    return df

price_data = load_price_data()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
price_data.to_csv(os.path.join(BASE_DIR, "data", "raw", "price.csv"))
print("Saved ✅")