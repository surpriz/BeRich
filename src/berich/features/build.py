"""Assemble the model feature matrix from an OHLCV frame.

`build_features` returns a frame aligned to the input index with one column per
feature. All features are causal (see :mod:`berich.features.indicators`). The
``FEATURE_COLUMNS`` constant is the canonical, ordered feature list used by every
model so training and inference never disagree on column order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from berich.features import indicators as ind

FEATURE_COLUMNS: list[str] = [
    "ret_1",
    "ret_5",
    "mom_10",
    "mom_20",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "atr_pct",
    "rvol_20",
    "close_sma20_ratio",
    "close_sma50_ratio",
    "volume_z20",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the canonical feature matrix from an OHLCV frame.

    Returns a frame with exactly ``FEATURE_COLUMNS``, indexed like ``df``. Rows where
    any feature is still warming up are left as NaN for the caller to drop.
    """
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    feats = pd.DataFrame(index=df.index)

    feats["ret_1"] = close.pct_change(1)
    feats["ret_5"] = close.pct_change(5)
    feats["mom_10"] = ind.momentum(close, 10)
    feats["mom_20"] = ind.momentum(close, 20)
    feats["rsi_14"] = ind.rsi(close, 14)

    # Normalize MACD components by price so they are stationary across price levels
    # and time (raw MACD scales with the absolute price and drifts heavily otherwise).
    macd = ind.macd(close).div(close, axis=0)
    feats[["macd", "macd_signal", "macd_hist"]] = macd

    # ATR as a fraction of price — comparable across tickers and price levels.
    feats["atr_pct"] = ind.atr(high, low, close, 14) / close
    feats["rvol_20"] = ind.realized_vol(close, 20)

    sma20 = close.rolling(20, min_periods=20).mean()
    sma50 = close.rolling(50, min_periods=50).mean()
    feats["close_sma20_ratio"] = close / sma20 - 1.0
    feats["close_sma50_ratio"] = close / sma50 - 1.0

    feats["volume_z20"] = ind.zscore(volume.astype(float), 20)

    return feats[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
