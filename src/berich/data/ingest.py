"""yfinance ingestion with incremental updates into the Parquet cache.

`update_watchlist` is the public entry point: for each ticker it fetches only the
bars newer than what is already cached, merges them, and reports integrity warnings
(NaNs, suspicious gaps). Training/serving never call yfinance directly — they read
the cache via :class:`~berich.data.store.OhlcvStore`.

Phase 6 extends the same pipeline to wider universes (200–400 tickers): a
thread-pool driver, exponential backoff on transient failures, and a
post-fetch liquidity gate that drops names too thin to trade realistically.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# Wider-universe quality gates: skip tickers with too little history or too
# little dollar volume to be realistically tradable in a swing-trading book.
MIN_HISTORY_BARS = 500
MIN_MEDIAN_VOLUME = 500_000

# Polite parallelism + retry for yfinance. The free-tier endpoint occasionally
# 429s under burst — backoff is exponential with a small base, max 3 attempts.
DEFAULT_MAX_WORKERS = 10
FETCH_MAX_ATTEMPTS = 3
FETCH_BACKOFF_BASE_SECONDS = 1.5


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
    """Refresh every ticker the daily scheduler needs — watchlist + multi-asset universes.

    The legacy ``watchlist`` (mega-cap, what the model was trained on) is the
    foundation; the polish-v2 multi-asset ``universes`` block (FR stocks,
    forex, crypto, commodities) is unioned in for the dashboard's
    "experimental" views. Tickers absent from any list are simply not
    refreshed. The Phase 6 wider-universe driver
    (:func:`update_universe`) still exists for the mid/small-cap probes.
    """
    store = OhlcvStore(config.ohlcv_dir)
    reports: list[IngestReport] = []
    for ticker in config.all_runtime_tickers():
        report = _update_one(ticker, config, store)
        reports.append(report)
        _log_report(report)
    return reports


def update_universe(
    config: Config,
    tickers: list[str],
    *,
    max_workers: int = DEFAULT_MAX_WORKERS,
    enforce_liquidity: bool = True,
) -> list[IngestReport]:
    """Phase 6 — refresh a wide universe (200–400 tickers) in parallel.

    Behaviour mirrors :func:`update_watchlist` per ticker (incremental fetch,
    Parquet merge), but the batch runs through a thread pool with exponential
    backoff on transient yfinance failures and an optional liquidity gate
    (median volume + min history). The gate doesn't delete the cache for a
    failing ticker; it just appends a warning so downstream code can skip it
    cleanly via ``store.load`` returning a frame whose ``IngestReport`` is
    marked not-ok.
    """
    store = OhlcvStore(config.ohlcv_dir)
    reports: list[IngestReport] = []

    def _task(ticker: str) -> IngestReport:
        report = _update_one_with_retry(ticker, config, store)
        if enforce_liquidity and report.ok:
            warn = _liquidity_warning(store, ticker)
            if warn:
                report.warnings.append(warn)
        return report

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_task, t): t for t in tickers}
        for future in as_completed(futures):
            report = future.result()
            reports.append(report)
            _log_report(report)
    # Stable order by ticker for deterministic test/log output.
    reports.sort(key=lambda r: r.ticker)
    return reports


def _log_report(report: IngestReport) -> None:
    level = logging.WARNING if report.warnings else logging.INFO
    logger.log(
        level,
        "%s: +%d rows (%d total)%s",
        report.ticker,
        report.rows_added,
        report.total_rows,
        f" — {'; '.join(report.warnings)}" if report.warnings else "",
    )


def _update_one_with_retry(ticker: str, config: Config, store: OhlcvStore) -> IngestReport:
    """``_update_one`` with exponential backoff on yfinance hiccups."""
    last_exc: Exception | None = None
    for attempt in range(FETCH_MAX_ATTEMPTS):
        try:
            return _update_one(ticker, config, store)
        except (OSError, RuntimeError, ValueError) as exc:
            last_exc = exc
            sleep_seconds = FETCH_BACKOFF_BASE_SECONDS * (2**attempt)
            logger.warning(
                "%s: fetch attempt %d/%d failed (%s); retrying in %.1fs",
                ticker,
                attempt + 1,
                FETCH_MAX_ATTEMPTS,
                exc.__class__.__name__,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)
    return IngestReport(
        ticker=ticker,
        warnings=[f"fetch failed after {FETCH_MAX_ATTEMPTS} attempts: {last_exc!r}"],
    )


def _liquidity_warning(store: OhlcvStore, ticker: str) -> str | None:
    """Return a human-readable warning if a freshly-cached ticker fails the gate."""
    df = store.load(ticker)
    if df is None or df.empty:
        return "no cached data"
    if len(df) < MIN_HISTORY_BARS:
        return f"history {len(df)} bars < {MIN_HISTORY_BARS} minimum"
    median_volume = float(df["volume"].median())
    if median_volume < MIN_MEDIAN_VOLUME:
        return f"median volume {median_volume:,.0f} < {MIN_MEDIAN_VOLUME:,} minimum"
    return None


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
