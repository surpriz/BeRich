"""yfinance ingestion with incremental updates into the Parquet cache.

`update_watchlist` is the public entry point: for each ticker it fetches only the
bars newer than what is already cached, merges them, and reports integrity warnings
(NaNs, suspicious gaps). Training/serving never call yfinance directly — they read
the cache via :class:`~berich.data.store.OhlcvStore`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING

import pandas as pd
import yfinance as yf

from berich.data.store import OHLCV_COLUMNS, OhlcvStore

if TYPE_CHECKING:
    from datetime import date

    from berich.config import Config

logger = logging.getLogger(__name__)

# A daily series should have ~252 bars/year; more than this many consecutive
# missing business days between rows signals a data problem worth flagging.
MAX_GAP_BUSINESS_DAYS = 5


@dataclass
class IngestReport:
    """Per-ticker outcome of an ingestion run."""

    ticker: str
    rows_added: int = 0
    total_rows: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.warnings


def fetch_ticker(
    ticker: str,
    *,
    start: date | pd.Timestamp,
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download OHLCV for one ticker and return it in the canonical schema.

    Columns are lowercased to ``open, high, low, close, volume`` and the index is
    a tz-naive ``DatetimeIndex``. Returns an empty frame if yfinance has nothing.
    """
    raw = yf.download(
        ticker,
        start=pd.Timestamp(start),
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    # yfinance returns a column MultiIndex (field, ticker) for single tickers too.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=str.lower)
    return raw[[c for c in OHLCV_COLUMNS if c in raw.columns]]


def _check_integrity(df: pd.DataFrame) -> list[str]:
    """Return human-readable warnings about NaNs and large date gaps."""
    warnings: list[str] = []
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        warnings.append(f"NaNs in columns {nan_cols}")
    if len(df) >= 2:  # noqa: PLR2004
        gaps = df.index.to_series().diff().dropna()
        max_gap = gaps.max()
        if max_gap > timedelta(days=MAX_GAP_BUSINESS_DAYS + 4):
            worst = gaps.idxmax().date()
            warnings.append(f"gap of {max_gap.days}d ending {worst}")
    return warnings


def update_watchlist(config: Config) -> list[IngestReport]:
    """Incrementally refresh every watchlist + context-series ticker.

    Context tickers (VIX, rates, credit, sector ETFs from
    :attr:`Config.context_tickers`) are downloaded alongside the watchlist into
    the same Parquet cache so they're available to the feature builder. They're
    not predicted or traded — features/build.py reads them as cross-asset
    inputs only.
    """
    store = OhlcvStore(config.ohlcv_dir)
    reports: list[IngestReport] = []
    seen: set[str] = set()
    for ticker in [*config.watchlist, *config.context_tickers]:
        upper = ticker.upper()
        if upper in seen:
            continue
        seen.add(upper)
        report = _update_one(ticker, config, store)
        reports.append(report)
        level = logging.WARNING if report.warnings else logging.INFO
        logger.log(
            level,
            "%s: +%d rows (%d total)%s",
            ticker,
            report.rows_added,
            report.total_rows,
            f" — {'; '.join(report.warnings)}" if report.warnings else "",
        )
    return reports


def _update_one(ticker: str, config: Config, store: OhlcvStore) -> IngestReport:
    last = store.last_date(ticker)
    # Re-fetch the last cached day too, so adjusted prices stay consistent.
    start = (last - timedelta(days=1)) if last is not None else config.data.start_date
    before = store.load(ticker)
    before_rows = 0 if before is None else len(before)

    fresh = fetch_ticker(
        ticker,
        start=start,
        interval=config.data.interval,
        auto_adjust=config.data.auto_adjust,
    )
    if not fresh.empty:
        store.save(ticker, fresh)

    merged = store.load(ticker)
    if merged is None or merged.empty:
        return IngestReport(ticker, warnings=["no data returned"])
    return IngestReport(
        ticker=ticker,
        rows_added=len(merged) - before_rows,
        total_rows=len(merged),
        warnings=_check_integrity(merged),
    )
