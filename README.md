# 🪙 Bitcoin Price Prediction Using LSTM-GRU Hybrid with Sentiment Analysis






> Skripsi — Teknik Informatika, Telkom University  
> Prediksi harga Bitcoin menggunakan model deep learning hybrid LSTM-GRU yang dikombinasikan dengan fitur sentiment analysis dari data sosial media.

***

## 📌 Overview

Model ini memprediksi harga penutupan (*Close*) Bitcoin pada hari berikutnya menggunakan:
- **LSTM-GRU Hybrid** — menangkap dependensi temporal jangka pendek dan panjang
- **21 fitur teknikal & sentimen** — termasuk RSI, Moving Average, Volume Ratio, dan Sentiment Score
- **Sliding window sequences** dengan `SEQ_LEN = 14` hari

***

## 📁 Struktur Project

```
bitcoin-prediction/
├── data/
│   ├── raw/                    # Data mentah (OHLCV + sentiment)
│   └── processed/
│       └── feature_engineering.csv
├── src/
│   ├── model.py                # Arsitektur LSTM-GRU Hybrid
│   ├── main.py                 # Entry point training & evaluasi
│   └── checkpoints/            # Bobot model terbaik (.pt)
├── notebooks/
│   └── EDA.ipynb               # Exploratory Data Analysis
├── requirements.txt
└── README.md
```

***

## ⚙️ Instalasi

```bash
# 1. Clone repository
git clone https://github.com/username/bitcoin-prediction.git
cd bitcoin-prediction

# 2. Buat virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

***

## 🚀 Cara Penggunaan

### Training Model
```bash
python src/main.py
```

### Konfigurasi utama (`src/model.py`)
| Parameter | Value | Keterangan |
|---|---|---|
| `SEQ_LEN` | 14 | Panjang window sequence |
| `BATCH_SIZE` | 32 | Ukuran batch training |
| `lstm_hidden` | 128 | Hidden units LSTM |
| `gru_hidden` | 64 | Hidden units GRU |
| `dropout` | 0.2 | Dropout regularization |
| `epochs` | 100 | Total epoch training |
| `lr` | 1e-3 | Learning rate Adam |

***

## 🧠 Arsitektur Model

```
Input (batch, 14, 21)
    ↓
LSTM (2 layers, hidden=128)
    ↓
GRU  (1 layer,  hidden=64)
    ↓
Dropout (0.2)
    ↓
FC (64 → 32) + ReLU
    ↓
FC (32 → 1)
    ↓
Output: Prediksi Close besok
```

***

## 📊 Fitur yang Digunakan

| Kategori | Fitur |
|---|---|
| **OHLCV** | Open, High, Low, Close, Volume |
| **Return** | return_1d, return_7d |
| **Volatilitas** | volatility_7d, volatility_14d |
| **Moving Average** | ma_7, ma_21, ma_cross |
| **Rasio Harga** | hl_ratio, oc_ratio |
| **Volume** | volume_ma7, volume_ratio |
| **Sentimen** | sentiment_score, sentiment_lag1, sentiment_lag2, sentiment_rolling7 |
| **Indikator** | rsi_14 |

***

## 📈 Hasil Evaluasi

| Metrik | Nilai |
|---|---|
| R² Score | ~0.3 |
| Loss Function | MSELoss |
| Optimizer | Adam |

***

## 🛠️ Tech Stack

- **Python** 3.10+
- **PyTorch** — model deep learning
- **Scikit-Learn** — preprocessing (MinMaxScaler)
- **Pandas / NumPy** — manipulasi data
- **Matplotlib** — visualisasi hasil

***

## 📝 Requirements

```
torch>=2.0.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
```

***

## 👤 Author

**Christian Galeno**  
Mahasiswa Informatika — Telkom University  

***

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
