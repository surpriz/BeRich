"""Per-ticker news cache backed by Alpha Vantage NEWS_SENTIMENT.

Each ticker gets a Parquet under ``data/news/<TICKER>.parquet`` with one row
per article. Deduplication is by URL — the API returns the same article in
multiple time windows on overlap, so we always merge and keep last.

API contract (free tier): 25 requests / day, each returns up to 1000 articles
in the time window. Pagination is via ``time_from`` (we walk forward in time
from the last cached timestamp). When the daily quota is exhausted Alpha
Vantage returns an ``Information`` key with a human-readable message and no
``feed`` — we catch that as :class:`RateLimitError`, log a warning, and skip
the rest of the batch so the scheduler doesn't crash.

This module owns I/O and the on-disk schema only; FinBERT scoring lives in
:mod:`berich.models.finbert_scorer` and writes back into the same store.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import httpx
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

    from berich.config import Config

logger = logging.getLogger(__name__)

NEWS_COLUMNS = [
    "time_published",
    "title",
    "summary",
    "source",
    "url",
    "overall_sentiment_score",
    "ticker_sentiment_score",
    "relevance_score",
    "finbert_neg",
    "finbert_neu",
    "finbert_pos",
    "finbert_score",  # pos - neg, convenience pre-computed at scoring time
]

AV_ENDPOINT = "https://www.alphavantage.co/query"
AV_KEY_ENV = "ALPHAVANTAGE_KEY"
AV_MAX_LIMIT = 1000

# Historical horizon we ever try to pull. Older content has been stripped from
# the AV index anyway, so going further back is wasted requests.
DEFAULT_START = pd.Timestamp("2022-04-01T00:00:00")


class RateLimitError(RuntimeError):
    """Raised when Alpha Vantage signals the daily quota is exhausted."""


@dataclass
class NewsReport:
    """Per-ticker outcome of one ``update_news_watchlist`` call."""

    ticker: str
    rows_added: int = 0
    rows_total: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.warnings


class NewsStore:
    """Read / write / merge per-ticker news frames in a Parquet directory."""

    def __init__(self, news_dir: Path) -> None:
        self.news_dir = news_dir

    def _path(self, ticker: str) -> Path:
        return self.news_dir / f"{ticker.upper()}.parquet"

    def exists(self, ticker: str) -> bool:
        return self._path(ticker).exists()

    def load(self, ticker: str) -> pd.DataFrame | None:
        """Return the cached news frame for ``ticker`` or ``None`` if not cached."""
        path = self._path(ticker)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def last_time(self, ticker: str) -> pd.Timestamp | None:
        """Most recent ``time_published`` cached for ``ticker``, or ``None``."""
        df = self.load(ticker)
        if df is None or df.empty or df["time_published"].isna().all():
            return None
        return pd.Timestamp(df["time_published"].max())

    def save(self, ticker: str, df: pd.DataFrame) -> None:
        """Normalize + merge with the existing cache + persist atomically."""
        df = self._normalize(df)
        existing = self.load(ticker)
        if existing is not None and not existing.empty:
            df = self._merge(existing, df)
        self.news_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._path(ticker).with_suffix(".parquet.tmp")
        df.to_parquet(tmp)
        tmp.replace(self._path(ticker))

    def update_finbert(
        self,
        ticker: str,
        scores: pd.DataFrame,
    ) -> int:
        """Patch FinBERT score columns for the rows whose URL appears in ``scores``.

        ``scores`` must carry ``url`` plus the four ``finbert_*`` columns. Idempotent:
        running it again with the same rows is a no-op. Returns the number of rows
        updated.
        """
        if scores.empty:
            return 0
        current = self.load(ticker)
        if current is None or current.empty:
            return 0
        merged = current.set_index("url")
        update = scores.set_index("url")[["finbert_neg", "finbert_neu", "finbert_pos", "finbert_score"]]
        common = merged.index.intersection(update.index)
        if common.empty:
            return 0
        merged.loc[common, ["finbert_neg", "finbert_neu", "finbert_pos", "finbert_score"]] = update.loc[
            common, ["finbert_neg", "finbert_neu", "finbert_pos", "finbert_score"]
        ]
        merged = merged.reset_index()[NEWS_COLUMNS]
        self.news_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._path(ticker).with_suffix(".parquet.tmp")
        merged.to_parquet(tmp)
        tmp.replace(self._path(ticker))
        return int(len(common))

    def has_any_data(self) -> bool:
        if not self.news_dir.exists():
            return False
        for path in self.news_dir.glob("*.parquet"):
            try:
                if not pd.read_parquet(path).empty:
                    return True
            except (OSError, ValueError):
                continue
        return False

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            empty = pd.DataFrame(columns=pd.Index(NEWS_COLUMNS))
            return empty.astype(
                {c: float for c in (
                    "overall_sentiment_score",
                    "ticker_sentiment_score",
                    "relevance_score",
                    "finbert_neg",
                    "finbert_neu",
                    "finbert_pos",
                    "finbert_score",
                )},
                errors="ignore",
            )
        out = df.copy()
        for col in NEWS_COLUMNS:
            if col not in out.columns:
                out[col] = pd.NA
        out = out[NEWS_COLUMNS]
        out["time_published"] = pd.to_datetime(out["time_published"], errors="coerce")
        # De-dup by URL across the merged set; keep the most recent (== the row
        # most likely to carry FinBERT scores from a prior pass).
        return out[~out["url"].duplicated(keep="last")].sort_values("time_published")

    @staticmethod
    def _merge(existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
        combined = pd.concat([existing, fresh])
        # If the same URL appears in both, prefer the existing row when it
        # carries a FinBERT score and the fresh one doesn't — avoids losing
        # GPU work to an idempotent re-fetch.
        combined["__has_score"] = combined["finbert_score"].notna()
        combined = combined.sort_values(["url", "__has_score"]).drop_duplicates(
            "url", keep="last"
        )
        combined = combined.drop(columns="__has_score")
        return combined.sort_values("time_published")


# ----------------------------------------------------------------- AV fetcher ----


def _api_key() -> str:
    key = os.environ.get(AV_KEY_ENV)
    if not key:
        msg = f"{AV_KEY_ENV} not set in environment"
        raise RuntimeError(msg)
    return key


def _format_time_from(ts: pd.Timestamp) -> str:
    """Alpha Vantage wants ``YYYYMMDDTHHMM`` (no seconds, no timezone)."""
    return pd.Timestamp(ts).strftime("%Y%m%dT%H%M")


def _parse_av_time(raw: str) -> pd.Timestamp | None:
    """AV uses ``YYYYMMDDTHHMMSS`` UTC for ``time_published``."""
    try:
        return pd.Timestamp(datetime.strptime(raw, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)).tz_convert(
            None
        )
    except (ValueError, TypeError):
        return None


def _parse_feed(feed: list[dict[str, Any]], ticker: str) -> pd.DataFrame:
    """Flatten the AV feed for one ticker into the cache schema."""
    rows: list[dict[str, Any]] = []
    upper_ticker = ticker.upper()
    for item in feed:
        ticker_block = next(
            (
                ts
                for ts in item.get("ticker_sentiment", [])
                if ts.get("ticker", "").upper() == upper_ticker
            ),
            None,
        )
        rows.append(
            {
                "time_published": _parse_av_time(item.get("time_published", "")),
                "title": item.get("title"),
                "summary": item.get("summary"),
                "source": item.get("source"),
                "url": item.get("url"),
                "overall_sentiment_score": _coerce_float(item.get("overall_sentiment_score")),
                "ticker_sentiment_score": _coerce_float(
                    ticker_block.get("ticker_sentiment_score") if ticker_block else None
                ),
                "relevance_score": _coerce_float(
                    ticker_block.get("relevance_score") if ticker_block else None
                ),
                "finbert_neg": pd.NA,
                "finbert_neu": pd.NA,
                "finbert_pos": pd.NA,
                "finbert_score": pd.NA,
            }
        )
    df = pd.DataFrame(rows, columns=pd.Index(NEWS_COLUMNS))
    return df.dropna(subset=["url"])


def _coerce_float(value: Any) -> float | None:  # noqa: ANN401 — third-party JSON values
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_news(
    ticker: str,
    *,
    since: pd.Timestamp | None = None,
    api_key: str | None = None,
    limit: int = AV_MAX_LIMIT,
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    """Fetch one page of news for ``ticker`` since ``since`` (or DEFAULT_START)."""
    key = api_key or _api_key()
    time_from = _format_time_from(since or DEFAULT_START)
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker.upper(),
        "time_from": time_from,
        "limit": str(min(limit, AV_MAX_LIMIT)),
        "sort": "EARLIEST",  # walk forward for stable pagination
        "apikey": key,
    }
    request = client or httpx.Client(timeout=30.0)
    try:
        response = request.get(AV_ENDPOINT, params=params)
    finally:
        if client is None:
            request.close()
    response.raise_for_status()
    payload = response.json()
    # AV uses "Information" (free tier quota), "Note" (5-call/min throttle), and
    # "Error Message" (bad request) — flatten all three into our typed exception.
    for trigger in ("Information", "Note", "Error Message"):
        if trigger in payload and "feed" not in payload:
            msg = f"Alpha Vantage {trigger}: {payload[trigger]}"
            raise RateLimitError(msg)
    feed = payload.get("feed", [])
    return _parse_feed(feed, ticker)


def update_news_watchlist(
    config: Config,
    *,
    max_pages_per_ticker: int = 4,
    api_key: str | None = None,
) -> list[NewsReport]:
    """Refresh news for every watchlist ticker, walking forward from the cache tip.

    ``max_pages_per_ticker`` caps how many sequential requests we'll burn on
    one symbol — useful both as a guardrail against runaway pagination and to
    keep each invocation predictable against the 25-req/day free quota.
    Rate-limit signals abort the *batch* (further tickers are skipped with a
    warning) rather than the whole process.
    """
    store = NewsStore(config.news_dir)
    reports: list[NewsReport] = []
    key = api_key or _api_key()
    rate_limited = False
    with httpx.Client(timeout=30.0) as client:
        for ticker in config.watchlist:
            if rate_limited:
                reports.append(NewsReport(ticker=ticker, warnings=["skipped: AV quota exhausted"]))
                continue
            try:
                report = _update_one(ticker, config, store, client=client, api_key=key,
                                     max_pages=max_pages_per_ticker)
            except RateLimitError as exc:
                rate_limited = True
                reports.append(NewsReport(ticker=ticker, warnings=[str(exc)]))
                logger.warning("news %s: %s", ticker, exc)
                continue
            reports.append(report)
            level = logging.WARNING if report.warnings else logging.INFO
            logger.log(
                level,
                "news %s: +%d rows (%d total)%s",
                ticker,
                report.rows_added,
                report.rows_total,
                f" — {'; '.join(report.warnings)}" if report.warnings else "",
            )
    return reports


def _update_one(
    ticker: str,
    config: Config,  # noqa: ARG001 — kept symmetric with update_watchlist signature
    store: NewsStore,
    *,
    client: httpx.Client,
    api_key: str,
    max_pages: int,
) -> NewsReport:
    """Pull up to ``max_pages`` pages from AV starting at the cache tip."""
    last = store.last_time(ticker)
    cursor = (last + pd.Timedelta(seconds=1)) if last is not None else DEFAULT_START
    before_rows = 0 if (cur := store.load(ticker)) is None else len(cur)
    rows_added = 0
    for _page in range(max_pages):
        fresh = fetch_news(ticker, since=cursor, api_key=api_key, client=client)
        if fresh.empty:
            break
        store.save(ticker, fresh)
        rows_added += len(fresh)
        new_cursor = fresh["time_published"].max()
        if pd.isna(new_cursor):
            break
        next_cursor = pd.Timestamp(new_cursor) + pd.Timedelta(seconds=1)
        if next_cursor <= cursor:
            # Defensive: API returned older-or-equal timestamps; stop to avoid loops.
            break
        cursor = next_cursor
        if len(fresh) < AV_MAX_LIMIT // 2:
            # Heuristic: short page == end of stream. Saves a final empty round-trip.
            break
    merged = store.load(ticker)
    total = 0 if merged is None else len(merged)
    return NewsReport(
        ticker=ticker,
        rows_added=int(total - before_rows),
        rows_total=int(total),
    )
