"""Parquet-backed OHLCV cache.

One Parquet file per ticker under ``<data_dir>/ohlcv/<TICKER>.parquet``. The store
only does I/O and merging; it has no knowledge of yfinance. The canonical schema
is a :class:`pandas.DataFrame` indexed by a timezone-naive ``DatetimeIndex`` named
``date`` with float columns ``open, high, low, close, volume``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
INDEX_NAME = "date"


class OhlcvStore:
    """Read/write/merge OHLCV frames in a Parquet cache directory."""

    def __init__(self, ohlcv_dir: Path, *, interval: str = "1d") -> None:
        self.ohlcv_dir = ohlcv_dir
        self.interval = interval
        # Intraday bars carry a meaningful time component; daily bars are normalized
        # to midnight so one calendar day is one row. ``.normalize()`` would collapse
        # every 1h bar of a day onto the same midnight index — fatal for intraday.
        self._intraday = interval != "1d"

    def _path(self, ticker: str) -> Path:
        return self.ohlcv_dir / f"{ticker.upper()}.parquet"

    def exists(self, ticker: str) -> bool:
        return self._path(ticker).exists()

    def load(self, ticker: str) -> pd.DataFrame | None:
        """Return the cached frame for ``ticker`` or ``None`` if not cached."""
        path = self._path(ticker)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def last_date(self, ticker: str) -> pd.Timestamp | None:
        """Most recent date present in the cache, or ``None`` if empty/missing."""
        df = self.load(ticker)
        if df is None or df.empty:
            return None
        return df.index.max()

    def save(self, ticker: str, df: pd.DataFrame) -> None:
        """Validate, merge with any existing cache, and persist atomically."""
        df = self._normalize(df)
        existing = self.load(ticker)
        if existing is not None:
            df = self._merge(existing, df)
        self.ohlcv_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._path(ticker).with_suffix(".parquet.tmp")
        df.to_parquet(tmp)
        tmp.replace(self._path(ticker))

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce a frame to the canonical schema, raising on missing columns."""
        missing = set(OHLCV_COLUMNS) - set(df.columns)
        if missing:
            msg = f"OHLCV frame missing columns: {sorted(missing)}"
            raise ValueError(msg)
        out = df[OHLCV_COLUMNS].copy()
        idx = pd.DatetimeIndex(df.index).tz_localize(None)
        out.index = idx if self._intraday else idx.normalize()
        out.index.name = INDEX_NAME
        return out[~out.index.duplicated(keep="last")].sort_index()

    @staticmethod
    def _merge(existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
        """Concatenate frames, newer rows winning on overlapping dates."""
        combined = pd.concat([existing, fresh])
        combined = combined[~combined.index.duplicated(keep="last")]
        return combined.sort_index()
