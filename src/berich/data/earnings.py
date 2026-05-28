"""Earnings-date cache (yfinance source).

For each ticker we persist the historical earnings calendar (date, estimate,
actual, surprise%) into ``data/earnings/<TICKER>.parquet``. Loading and merging
mirror the OHLCV store: the file owns disk I/O and deduplication; nothing here
knows about yfinance beyond the single ``fetch_earnings`` function.

Yfinance's ``Ticker.earnings_dates`` returns up to ~25 rows mixing past
quarters (with reported EPS + surprise%) and a few future scheduled dates
(estimate only). ETFs and indices return an empty frame — we cache it as
an empty file so subsequent runs don't re-hit the network for nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd
import yfinance as yf

if TYPE_CHECKING:
    from pathlib import Path

    from berich.config import Config

logger = logging.getLogger(__name__)

EARNINGS_COLUMNS = ["eps_estimate", "reported_eps", "surprise_pct"]
INDEX_NAME = "date"

_RENAME = {
    "EPS Estimate": "eps_estimate",
    "Reported EPS": "reported_eps",
    "Surprise(%)": "surprise_pct",
}


@dataclass
class EarningsReport:
    """Per-ticker outcome of an earnings refresh, mirrors :class:`IngestReport`."""

    ticker: str
    rows_total: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.warnings


class EarningsStore:
    """Read / write / merge per-ticker earnings frames in a Parquet directory.

    Schema mirror of :class:`~berich.data.store.OhlcvStore`: tz-naive DatetimeIndex
    named ``date``, float columns ``eps_estimate``, ``reported_eps``,
    ``surprise_pct``. Future-dated rows (estimate only, no reported value) are
    kept — they're how the feature builder knows when the *next* announcement is.
    """

    def __init__(self, earnings_dir: Path) -> None:
        self.earnings_dir = earnings_dir

    def _path(self, ticker: str) -> Path:
        return self.earnings_dir / f"{ticker.upper()}.parquet"

    def exists(self, ticker: str) -> bool:
        return self._path(ticker).exists()

    def load(self, ticker: str) -> pd.DataFrame | None:
        """Return the cached earnings frame for ``ticker`` or ``None`` if not cached."""
        path = self._path(ticker)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def save(self, ticker: str, df: pd.DataFrame) -> None:
        """Validate, merge with any existing cache, and persist atomically."""
        df = self._normalize(df)
        existing = self.load(ticker)
        if existing is not None and not existing.empty:
            df = self._merge(existing, df)
        self.earnings_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._path(ticker).with_suffix(".parquet.tmp")
        df.to_parquet(tmp)
        tmp.replace(self._path(ticker))

    def has_any_data(self) -> bool:
        """True iff at least one ticker file has at least one row."""
        if not self.earnings_dir.exists():
            return False
        for path in self.earnings_dir.glob("*.parquet"):
            try:
                df = pd.read_parquet(path)
                if not df.empty:
                    return True
            except (OSError, ValueError):
                continue
        return False

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        """Coerce a frame to the canonical schema. Accepts empty frames as-is."""
        if df is None or df.empty:
            empty = pd.DataFrame(columns=pd.Index(EARNINGS_COLUMNS))
            empty.index = pd.DatetimeIndex([], name=INDEX_NAME)
            return empty
        out = df.copy()
        # Drop the time-of-day and tz so the index is comparable to bar dates
        # (which are tz-naive at midnight).
        out.index = pd.DatetimeIndex(out.index).tz_localize(None).normalize()
        out.index.name = INDEX_NAME
        out = out[[c for c in EARNINGS_COLUMNS if c in out.columns]]
        # Drop duplicates by date keeping the most recent (handles yfinance
        # returning the same quarter twice across overlapping fetches).
        return out[~out.index.duplicated(keep="last")].sort_index()

    @staticmethod
    def _merge(existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
        """Union the two frames, newer rows winning on overlapping dates."""
        combined = pd.concat([existing, fresh])
        combined = combined[~combined.index.duplicated(keep="last")]
        return combined.sort_index()


def fetch_earnings(ticker: str) -> pd.DataFrame:
    """Download earnings dates for one ticker; return the canonical schema (empty if none)."""
    raw = yf.Ticker(ticker).earnings_dates
    if raw is None or raw.empty:
        empty = pd.DataFrame(columns=pd.Index(EARNINGS_COLUMNS))
        empty.index = pd.DatetimeIndex([], name=INDEX_NAME)
        return empty
    out = raw.rename(columns=_RENAME)
    out = out[[c for c in EARNINGS_COLUMNS if c in out.columns]]
    out.index = pd.DatetimeIndex(out.index).tz_localize(None).normalize()
    out.index.name = INDEX_NAME
    return out[~out.index.duplicated(keep="last")].sort_index()


def update_earnings(
    config: Config,
    tickers: list[str] | None = None,
) -> list[EarningsReport]:
    """Refresh earnings for ``tickers`` (defaults to the watchlist).

    Idempotent: re-running is a no-op for tickers already cached today.
    Tickers that return zero rows (ETFs, indices) are still cached as an
    empty file so the next call doesn't re-fetch. Network errors are caught
    per ticker so one bad symbol doesn't kill the batch.

    Passing a wider list (e.g. ``config.tickers_for_universe("all")``)
    lets the Phase 7 PEAD dataset cover small/mid caps — yfinance's
    earnings_dates endpoint isn't rate-limited the way AV news is, so a
    few hundred sequential requests is fine here.
    """
    store = EarningsStore(config.earnings_dir)
    target = tickers if tickers is not None else config.watchlist
    reports: list[EarningsReport] = []
    for ticker in target:
        report = _update_one(ticker, store)
        reports.append(report)
        level = logging.WARNING if report.warnings else logging.INFO
        logger.log(
            level,
            "earnings %s: %d rows%s",
            ticker,
            report.rows_total,
            f" — {'; '.join(report.warnings)}" if report.warnings else "",
        )
    return reports


def _update_one(ticker: str, store: EarningsStore) -> EarningsReport:
    try:
        fresh = fetch_earnings(ticker)
    except Exception as exc:  # noqa: BLE001 — yfinance throws a zoo of unrelated types
        return EarningsReport(ticker=ticker, warnings=[f"fetch failed: {exc.__class__.__name__}"])
    store.save(ticker, fresh)
    merged = store.load(ticker)
    return EarningsReport(
        ticker=ticker,
        rows_total=0 if merged is None else len(merged),
    )
