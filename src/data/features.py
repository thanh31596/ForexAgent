"""Technical features + walk-forward split indices (pure pandas / ta)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import numpy as np
import pandas as pd
import ta

from src.config import (
    ATR_WINDOW,
    BOLLINGER_STD,
    BOLLINGER_WINDOW,
    EMA_WINDOWS,
    MACD_FAST_WINDOW,
    MACD_SIGNAL_WINDOW,
    MACD_SLOW_WINDOW,
    PREDICTION_HORIZON,
    RETURN_LAGS,
    RSI_WINDOW,
    SMA_WINDOWS,
    TARGET_TYPE,
    VOLATILITY_WINDOW,
    WALK_FORWARD_GAP,
    WALK_FORWARD_N_SPLITS,
    WALK_FORWARD_STRATEGY,
    WALK_FORWARD_TEST_SIZE,
)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with indicator columns + target ``y``."""
    out = df.copy()
    close = out["Close"]
    out["rsi"] = ta.momentum.RSIIndicator(close, window=RSI_WINDOW).rsi()
    macd = ta.trend.MACD(
        close,
        window_slow=MACD_SLOW_WINDOW,
        window_fast=MACD_FAST_WINDOW,
        window_sign=MACD_SIGNAL_WINDOW,
    )
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(close, window=BOLLINGER_WINDOW, window_dev=BOLLINGER_STD)
    out["bb_high"] = bb.bollinger_hband()
    out["bb_low"] = bb.bollinger_lband()
    for w in SMA_WINDOWS:
        out[f"sma_{w}"] = ta.trend.SMAIndicator(close, window=w).sma_indicator()
    for w in EMA_WINDOWS:
        out[f"ema_{w}"] = ta.trend.EMAIndicator(close, window=w).ema_indicator()
    out["volatility"] = close.pct_change().rolling(VOLATILITY_WINDOW).std()
    out["atr"] = ta.volatility.AverageTrueRange(
        out["High"], out["Low"], close, window=ATR_WINDOW
    ).average_true_range()
    for lag in RETURN_LAGS:
        out[f"ret_lag_{lag}"] = close.pct_change(lag)
    if TARGET_TYPE == "log_return":
        out["y"] = np.log(close).diff(PREDICTION_HORIZON).shift(-PREDICTION_HORIZON)
    elif TARGET_TYPE == "direction":
        out["y"] = (close.shift(-PREDICTION_HORIZON) > close).astype(float)
    else:
        out["y"] = close.shift(-PREDICTION_HORIZON)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Infer training feature columns (exclude OHLCV + target)."""
    exclude = {"Open", "High", "Low", "Close", "Volume", "y"}
    return [c for c in df.columns if c not in exclude]


def walk_forward_indices(
    n_samples: int,
    *,
    n_splits: int = WALK_FORWARD_N_SPLITS,
    test_size: int = WALK_FORWARD_TEST_SIZE,
    gap: int = WALK_FORWARD_GAP,
    strategy: Literal["expanding", "sliding"] = WALK_FORWARD_STRATEGY,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` integer index arrays (chronological, leak-safe)."""
    if n_samples <= test_size + gap + 10:
        # degenerate: single split
        train = np.arange(0, max(1, n_samples - test_size - gap))
        test = np.arange(n_samples - test_size, n_samples)
        yield train, test
        return

    for k in range(n_splits):
        test_end = n_samples - k * test_size
        test_start = test_end - test_size
        if test_start <= 0:
            continue
        train_end = test_start - gap
        if train_end <= 1:
            continue
        if strategy == "expanding":
            train_start = 0
        else:
            train_start = max(0, train_end - test_size * 5)
        train_idx = np.arange(train_start, train_end)
        test_idx = np.arange(test_start, test_end)
        if len(train_idx) < 10 or len(test_idx) < 5:
            continue
        yield train_idx, test_idx
