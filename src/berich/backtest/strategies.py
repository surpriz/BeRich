"""Daily-return series builders for the Phase 9 core-satellite portfolio.

Each builder returns a single ``pd.Series`` indexed by calendar day. The
portfolio engine in :mod:`berich.backtest.portfolio` aligns these on a
shared index and applies the chosen weights.

Three components are implemented:

- :func:`build_bnh_returns` — pure long buy & hold of a single ticker
  (defaults to SPY).
- :func:`build_pead_returns` — the average daily return of all PEAD
  positions currently open. Zero on days with no live position
  (cash). This is what makes PEAD a satellite rather than a core.
- :func:`build_calendar_returns` — turn-of-month rule on a single
  ticker: long during the last ``window`` business days of each month
  + the first ``window`` of the next; flat otherwise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from berich.data.store import OhlcvStore
    from berich.datasets.pead import PeadDataset

DEFAULT_PEAD_HOLD_DAYS = 5
DEFAULT_CAL_WINDOW = 3


def build_bnh_returns(store: OhlcvStore, ticker: str = "SPY") -> pd.Series:
    """Daily simple returns of a single ticker, NaN→0 on the first bar."""
    df = store.load(ticker)
    if df is None or df.empty:
        return pd.Series(dtype=float, name=f"bnh_{ticker.lower()}")
    return df["close"].pct_change().fillna(0.0).rename(f"bnh_{ticker.lower()}")


def build_pead_returns(
    dataset: PeadDataset,
    store: OhlcvStore,
    *,
    hold_days: int = DEFAULT_PEAD_HOLD_DAYS,
) -> pd.Series:
    """Daily PEAD strategy return = mean of returns of currently-held positions.

    For each PEAD event we open a position at ``entry_date`` and exit
    ``hold_days`` bars later. On any given day d, the strategy's daily
    return is the average per-event daily return ``close[t]/close[t-1] - 1``
    over the events that are live on d (entry_date <= d < entry_date + hold_days
    in bar-index terms). Days with no live event return 0 (cash).

    The series is built on the union of all (ticker, day) pairs touched by
    any event, then reindexed to a continuous business-day calendar so
    portfolio math has a clean grid to align with.
    """
    if dataset.events.empty:
        return pd.Series(dtype=float, name="pead")

    # Per-event per-day return contributions, keyed by date.
    contributions: dict[pd.Timestamp, list[float]] = {}
    ohlcv_cache: dict[str, pd.DataFrame] = {}

    for _, event in dataset.events.iterrows():
        ticker = str(event["ticker"])
        raw_entry = pd.Timestamp(event["entry_date"])
        if pd.isna(raw_entry):
            continue
        entry_date: pd.Timestamp = raw_entry  # ty: ignore[invalid-assignment]
        if ticker not in ohlcv_cache:
            loaded = store.load(ticker)
            if loaded is None or loaded.empty:
                continue
            ohlcv_cache[ticker] = loaded
        ohlcv = ohlcv_cache[ticker]
        if entry_date not in ohlcv.index:
            continue
        entry_idx = int(ohlcv.index.get_loc(entry_date))
        last_idx = min(entry_idx + hold_days, len(ohlcv) - 1)
        # Day-on-day returns inside the [entry_idx, last_idx] window.
        for k in range(1, last_idx - entry_idx + 1):
            day = ohlcv.index[entry_idx + k]
            ret = float(
                ohlcv.iloc[entry_idx + k]["close"] / ohlcv.iloc[entry_idx + k - 1]["close"] - 1.0
            )
            contributions.setdefault(day, []).append(ret)

    if not contributions:
        return pd.Series(dtype=float, name="pead")

    days = sorted(contributions.keys())
    averaged = [float(np.mean(contributions[d])) for d in days]
    series = pd.Series(averaged, index=pd.DatetimeIndex(days), name="pead")
    # Reindex onto a continuous business-day grid so portfolio math is clean.
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="B")
    return series.reindex(full_index).fillna(0.0).rename("pead")


def build_calendar_returns(
    store: OhlcvStore,
    *,
    ticker: str = "SPY",
    window: int = DEFAULT_CAL_WINDOW,
) -> pd.Series:
    """Long the ticker during the [last ``window`` of month, first ``window`` next] window.

    Returns are the ticker's daily simple returns when in the window, 0
    when flat. Transition days (0→1 and 1→0) pay the standard fee+slippage
    paid by every other engine in this project (1 bp commission + 5 bps
    slippage per side).
    """
    df = store.load(ticker)
    if df is None or df.empty:
        return pd.Series(dtype=float, name=f"calendar_{ticker.lower()}")

    rets = df["close"].pct_change().fillna(0.0)
    index = pd.DatetimeIndex(df.index)

    idx_series = pd.Series(index)
    info = pd.DataFrame({"date": index})
    info["year"] = idx_series.dt.year.to_numpy()
    info["month"] = idx_series.dt.month.to_numpy()
    info["rank_fwd"] = info.groupby(["year", "month"]).cumcount() + 1
    info["rank_bwd"] = info.groupby(["year", "month"]).cumcount(ascending=False) + 1
    in_window = ((info["rank_fwd"] <= window) | (info["rank_bwd"] <= window)).to_numpy()
    position = pd.Series(in_window.astype(int), index=index)

    edge = position != position.shift(1).fillna(0)
    cost_per_edge = (1.0 + 5.0) / 1e4  # 1 bp fee + 5 bps slippage, same as Phase 2
    cost = edge.astype(float) * cost_per_edge

    strat = rets * position - cost
    return strat.rename(f"calendar_{ticker.lower()}")
