"""Event-level features for the PEAD task.

Computed strictly at ``event_date`` (the announcement date) — i.e. we use
information available **at or before** the close of the event day. Forward
information (the close of ``entry_date`` and beyond) is reserved for the
label only. This matches the after-close announcement convention used by
the daily-bar earnings features in Phase 5a.

The feature set deliberately overlaps with the Phase 5a / 5b modules so a
model can recombine "what's the earnings surprise" + "what's the macro
regime" + "what's the news sentiment" into one event-level decision. When a
data source is missing for a given (ticker, event) we substitute a neutral
default (0 or the relevant clamp value) rather than dropping the row, so an
event without news still produces a usable feature vector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from berich.features.build import SECTOR_MAP
from berich.features.indicators import rsi

PEAD_FEATURE_COLUMNS: list[str] = [
    "surprise_pct",
    "surprise_4q_mean",
    "days_since_last_earnings",
    "pre_announce_run_5d",
    "pre_announce_vol_20d",
    "market_regime_spy_rvol",
    "news_count_pre_5d",
    "sentiment_mean_pre_5d",
    "sector_relative_5d",
    "log_volume_median",
    "pre_rsi_14",
]

WINDOW_PRE_RUN = 5
WINDOW_VOL = 20
WINDOW_NEWS = 5
DAYS_CLAMP = 90
NEUTRAL_LOG_VOLUME = float(np.log(1e6))  # ~1M shares: a sensible neutral for missing data


def _safe_pct_change(series: pd.Series, periods: int) -> float:
    """Percent change over ``periods`` bars, NaN-tolerant (returns 0 if not enough history)."""
    if len(series) <= periods:
        return 0.0
    end = float(series.iloc[-1])
    start = float(series.iloc[-1 - periods])
    if start <= 0 or np.isnan(start) or np.isnan(end):
        return 0.0
    return end / start - 1.0


def _rolling_log_return_std(close: pd.Series, window: int) -> float:
    """Standard deviation of daily log returns over the trailing ``window`` bars."""
    if len(close) < window + 1:
        return 0.0
    log_ret = np.log(
        close.iloc[-(window + 1) :].to_numpy() / close.iloc[-(window + 2) : -1].to_numpy()
    )
    if len(log_ret) == 0:
        return 0.0
    return float(np.nanstd(log_ret))


def _build_one(  # noqa: PLR0912 — straight-line per-source feature extraction
    event_date: pd.Timestamp,
    *,
    ohlcv: pd.DataFrame,
    earnings: pd.DataFrame,
    market: pd.DataFrame | None,
    sector_etf: pd.DataFrame | None,
    news: pd.DataFrame | None,
) -> dict[str, float]:
    """Compute the feature row for one (ticker, event_date) tuple."""
    # Use only bars dated <= event_date so the feature row is strictly causal.
    history = ohlcv.loc[ohlcv.index <= event_date]
    if history.empty:
        return _neutral_row()

    # Current-event surprise: the row in earnings with index == event_date (if present)
    # and that has a reported surprise. Future-only rows (estimate but no reported EPS)
    # have NaN surprise; treat as 0.
    current_surprise = 0.0
    if event_date in earnings.index:
        raw_surprise = earnings.loc[event_date, "surprise_pct"]
        if isinstance(raw_surprise, pd.Series):  # duplicate-date safety
            raw_surprise = raw_surprise.iloc[0]
        if not pd.isna(raw_surprise):
            current_surprise = float(raw_surprise)

    # Historical surprises = all earnings with date < event_date (strict past).
    past = earnings.loc[earnings.index < event_date]
    past_surprises = past["surprise_pct"].dropna().tail(4)
    surprise_4q_mean = float(past_surprises.mean()) if not past_surprises.empty else 0.0

    if past.empty:
        days_since = float(DAYS_CLAMP)
    else:
        delta = (event_date - past.index.max()).days
        days_since = min(float(delta), float(DAYS_CLAMP))

    close = history["close"]
    pre_run_5d = _safe_pct_change(close, WINDOW_PRE_RUN)
    pre_vol_20d = _rolling_log_return_std(close, WINDOW_VOL)
    pre_rsi = (
        float(rsi(close, 14).iloc[-1])
        if len(close) > 14  # noqa: PLR2004
        else 50.0
    )

    if market is not None and not market.empty:
        market_history = market.loc[market.index <= event_date]
        market_regime = _rolling_log_return_std(market_history["close"], WINDOW_VOL)
    else:
        market_regime = 0.0

    if sector_etf is not None and not sector_etf.empty:
        sector_history = sector_etf.loc[sector_etf.index <= event_date]
        own_5d = pre_run_5d
        sector_5d = _safe_pct_change(sector_history["close"], WINDOW_PRE_RUN)
        sector_relative_5d = own_5d - sector_5d
    else:
        sector_relative_5d = 0.0

    if news is not None and not news.empty:
        window_start = event_date - pd.Timedelta(days=WINDOW_NEWS)
        recent_news = news[
            (news["time_published"] >= window_start) & (news["time_published"] <= event_date)
        ]
        news_count = float(len(recent_news))
        # Drop NaN finbert scores so they don't drag the mean toward 0.
        sentiment = recent_news["finbert_score"].dropna()
        sentiment_mean = float(sentiment.mean()) if not sentiment.empty else 0.0
    else:
        news_count = 0.0
        sentiment_mean = 0.0

    median_volume = float(history["volume"].median())
    if median_volume <= 0 or np.isnan(median_volume):
        log_volume_median = NEUTRAL_LOG_VOLUME
    else:
        log_volume_median = float(np.log(median_volume))

    return {
        "surprise_pct": current_surprise,
        "surprise_4q_mean": surprise_4q_mean,
        "days_since_last_earnings": days_since,
        "pre_announce_run_5d": pre_run_5d,
        "pre_announce_vol_20d": pre_vol_20d,
        "market_regime_spy_rvol": market_regime,
        "news_count_pre_5d": news_count,
        "sentiment_mean_pre_5d": sentiment_mean,
        "sector_relative_5d": sector_relative_5d,
        "log_volume_median": log_volume_median,
        "pre_rsi_14": pre_rsi,
    }


def _neutral_row() -> dict[str, float]:
    """Row of safe defaults for an event whose feature panel is empty."""
    return {
        "surprise_pct": 0.0,
        "surprise_4q_mean": 0.0,
        "days_since_last_earnings": float(DAYS_CLAMP),
        "pre_announce_run_5d": 0.0,
        "pre_announce_vol_20d": 0.0,
        "market_regime_spy_rvol": 0.0,
        "news_count_pre_5d": 0.0,
        "sentiment_mean_pre_5d": 0.0,
        "sector_relative_5d": 0.0,
        "log_volume_median": NEUTRAL_LOG_VOLUME,
        "pre_rsi_14": 50.0,
    }


def sector_etf_for(ticker: str) -> str | None:
    """Return the sector-ETF ticker for ``ticker`` via the :data:`SECTOR_MAP`."""
    return SECTOR_MAP.get(ticker.upper())


def build_pead_features(
    events: pd.DataFrame,
    *,
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    earnings_by_ticker: dict[str, pd.DataFrame],
    market: pd.DataFrame | None = None,
    news_by_ticker: dict[str, pd.DataFrame] | None = None,
    sector_by_etf: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Compute the per-event feature matrix aligned to ``events``.

    Returns a frame with one row per event in ``events`` and columns ==
    :data:`PEAD_FEATURE_COLUMNS`. Caller is responsible for keeping
    ``events`` and ``ohlcv_by_ticker`` in sync (a ticker missing from the
    OHLCV dict gets a neutral row rather than being dropped — the user can
    apply their own dropna afterward if they prefer the strict policy).
    """
    sector_by_etf = sector_by_etf or {}
    news_by_ticker = news_by_ticker or {}
    rows: list[dict[str, float]] = []
    for _, event in events.iterrows():
        ticker = str(event["ticker"])
        raw_event_date = pd.Timestamp(event["event_date"])
        if pd.isna(raw_event_date):
            rows.append(_neutral_row())
            continue
        event_date: pd.Timestamp = raw_event_date  # ty: ignore[invalid-assignment]
        ohlcv = ohlcv_by_ticker.get(ticker)
        earnings = earnings_by_ticker.get(ticker)
        if ohlcv is None or earnings is None:
            rows.append(_neutral_row())
            continue
        etf_ticker = sector_etf_for(ticker)
        sector_etf = sector_by_etf.get(etf_ticker) if etf_ticker is not None else None
        news = news_by_ticker.get(ticker)
        rows.append(
            _build_one(
                event_date,
                ohlcv=ohlcv,
                earnings=earnings,
                market=market,
                sector_etf=sector_etf,
                news=news,
            )
        )
    return pd.DataFrame(rows, columns=pd.Index(PEAD_FEATURE_COLUMNS))
