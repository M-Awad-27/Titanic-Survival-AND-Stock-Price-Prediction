"""
stock_data_loader.py
────────────────────
Dedicated data loader and preprocessor for the Stock Price Prediction task.

Responsibilities:
  1. Download historical stock data via ``yfinance`` (caches locally as CSV).
  2. Validate and clean the downloaded data.
  3. Scale features (MinMaxScaler) and create sliding-window sequences
     suitable for time-series models (LSTM, Linear Regression, etc.).
  4. Return chronologically-split train/test arrays + metadata.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


# ─── Configuration ───────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume"]
TARGET_COL = "Close"  # column index 3 in FEATURE_COLS
TRAIN_RATIO = 0.8


# ─── Download ────────────────────────────────────────────────────────────────

def download_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    data_dir: str | None = None,
) -> str:
    """
    Downloads historical OHLCV data from Yahoo Finance via ``yfinance``.

    Parameters
    ----------
    ticker     : Stock symbol (e.g. ``'AAPL'``).
    start_date : Start date in ``YYYY-MM-DD`` format.
    end_date   : End date in ``YYYY-MM-DD`` format.
    data_dir   : Directory to cache the CSV.  Defaults to ``./data``.

    Returns
    -------
    file_path : str – Absolute path to the saved CSV.
    """
    save_dir = data_dir or DATA_DIR
    os.makedirs(save_dir, exist_ok=True)

    file_name = f"{ticker}_{start_date}_{end_date}.csv"
    file_path = os.path.join(save_dir, file_name)

    if os.path.exists(file_path):
        print(f"[stock_data_loader] Cached data found at {file_path}")
        return file_path

    print(
        f"[stock_data_loader] Downloading {ticker} data "
        f"({start_date} → {end_date}) via yfinance …"
    )
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}' "
            f"in range [{start_date}, {end_date}]."
        )

    df.to_csv(file_path)
    print(f"[stock_data_loader] Saved to {file_path}")
    return file_path


# ─── CSV Reader (handles multi-header yfinance files) ────────────────────────

def _read_stock_csv(csv_path: str) -> pd.DataFrame:
    """
    Reads a stock CSV that may have multi-row headers produced by ``yfinance``.
    Returns a clean DataFrame indexed by Date with capitalised column names.
    """
    # Peek at the first few lines to detect multi-header layout
    with open(csv_path, "r") as f:
        first_lines = [f.readline() for _ in range(3)]

    is_multi = any("Ticker" in line or "Price" in line for line in first_lines)

    if is_multi:
        df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
        df.columns = [col[0] for col in df.columns]
        df.index.name = "Date"
    else:
        df = pd.read_csv(csv_path, index_col=0)

    # Normalise column names
    df.columns = [
        col.strip().capitalize() if isinstance(col, str) else col
        for col in df.columns
    ]

    # Ensure datetime index sorted ascending
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    return df


# ─── Preprocessing & Sequence Generation ─────────────────────────────────────

def load_stock_data(
    csv_path: str | None = None,
    ticker: str = "AAPL",
    start_date: str = "2020-01-01",
    end_date: str = "2023-01-01",
    seq_length: int = 10,
):
    """
    End-to-end loader for stock time-series data.

    Workflow:
      1. Read (or download) the CSV.
      2. Validate required OHLCV columns.
      3. Scale all features with ``MinMaxScaler``.
      4. Build sliding-window sequences of length ``seq_length``.
      5. Chronologically split into train (80 %) and test (20 %).

    Parameters
    ----------
    csv_path    : Direct path to an existing CSV. If None, data is downloaded.
    ticker      : Stock ticker used when downloading.
    start_date  : Download start date.
    end_date    : Download end date.
    seq_length  : Number of past days in each input window.

    Returns
    -------
    X_train       : np.ndarray – shape (N_train, seq_length, n_features)
    X_test        : np.ndarray – shape (N_test,  seq_length, n_features)
    y_train       : np.ndarray – shape (N_train, 1) – scaled Close price
    y_test        : np.ndarray – shape (N_test,  1) – scaled Close price
    train_dates   : pd.DatetimeIndex
    test_dates    : pd.DatetimeIndex
    feature_scaler: MinMaxScaler – fitted on features
    target_scaler : MinMaxScaler – fitted on Close prices
    """
    # 1. Acquire CSV
    if csv_path is None:
        csv_path = download_stock_data(ticker, start_date, end_date)

    df = _read_stock_csv(csv_path)

    # 2. Validate columns
    for col in FEATURE_COLS:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing in {csv_path}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)
    print(
        f"[stock_data_loader] Loaded {len(df)} rows "
        f"({df.index.min().date()} → {df.index.max().date()})"
    )

    # 3. Scale
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    data = df[FEATURE_COLS].values
    close_prices = np.array(df[TARGET_COL].values, dtype=np.float64).reshape(-1, 1)

    scaled_data = feature_scaler.fit_transform(data)
    target_scaler.fit(close_prices)

    # 4. Create sliding-window sequences
    target_idx = FEATURE_COLS.index(TARGET_COL)  # index of Close in features

    X_seq, y_seq = [], []
    for i in range(len(scaled_data) - seq_length):
        X_seq.append(scaled_data[i : i + seq_length])
        y_seq.append(scaled_data[i + seq_length, target_idx])

    X_seq = np.array(X_seq, dtype=np.float64)
    y_seq = np.array(y_seq, dtype=np.float64).reshape(-1, 1)

    # 5. Chronological train/test split
    split_idx = int(len(X_seq) * TRAIN_RATIO)

    dates = df.index[seq_length:]
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx:]

    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    print(
        f"[stock_data_loader] Train: {X_train.shape}, Test: {X_test.shape}, "
        f"Seq length: {seq_length}"
    )

    return (
        X_train, X_test,
        y_train, y_test,
        train_dates, test_dates,
        feature_scaler, target_scaler,
    )


# ─── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    (
        X_tr, X_te,
        y_tr, y_te,
        tr_dates, te_dates,
        f_scaler, t_scaler,
    ) = load_stock_data()

    print(f"\nX_train shape : {X_tr.shape}")
    print(f"X_test  shape : {X_te.shape}")
    print(f"y_train shape : {y_tr.shape}")
    print(f"y_test  shape : {y_te.shape}")
    print(f"Train period  : {tr_dates[0].date()} → {tr_dates[-1].date()}")
    print(f"Test  period  : {te_dates[0].date()} → {te_dates[-1].date()}")
