"""Download adjusted closes with yfinance, turn them into log returns, cache as CSV."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data"


def load_prices(ticker: str = "SPY", start: str = "2010-01-01", end: str = "2024-12-31",
                refresh: bool = False) -> pd.Series:
    """Adjusted close prices, cached in data/<ticker>_<start>_<end>.csv."""
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{ticker}_{start}_{end}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)["close"]
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # newer yfinance returns a one-column frame
        close = close.iloc[:, 0]
    close = close.dropna().rename("close")
    close.to_frame().to_csv(path)
    return close


def log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices).diff().dropna().rename("r")
