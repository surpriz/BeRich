"""Event-level labels for the Post-Earnings Announcement Drift (PEAD) task.

Each row of the PEAD dataset is **one earnings announcement** (not one
trading day) — a fundamentally different target shape from the daily
triple-barrier label used by the LightGBM baseline. For each
``(ticker, event_date)`` we build:

- ``t1`` = first trading day strictly after the announcement (we treat the
  call as after-market, so the first practical entry is the next session's
  open).
- ``fwd_return_5d``  = ``close[t1+5d] / close[t1] - 1``
- ``fwd_return_20d`` = ``close[t1+20d] / close[t1] - 1``
- ``label_drift_5d``  = 1 iff ``fwd_return_5d  > 0.02``
- ``label_drift_20d`` = 1 iff ``fwd_return_20d > 0.05``

Rows whose forward window runs past the cache tip are dropped — exactly
as the daily triple-barrier labeller does for incomplete forward windows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DRIFT_5D_THRESHOLD = 0.02  # 2% positive drift over 5 trading days
DRIFT_20D_THRESHOLD = 0.05  # 5% positive drift over 20 trading days
FWD_5D = 5
FWD_20D = 20


@dataclass
class PeadEvent:
    """One earnings announcement with its forward returns + binary labels."""

    ticker: str
    event_date: pd.Timestamp  # the announcement date
    entry_date: pd.Timestamp  # the trading day used as entry (t1 = first bar > event)
    entry_price: float
    fwd_return_5d: float
    fwd_return_20d: float
    label_drift_5d: int
    label_drift_20d: int


def build_pead_events(
    ohlcv: pd.DataFrame,
    earnings: pd.DataFrame,
    *,
    ticker: str,
) -> pd.DataFrame:
    """Turn one ticker's OHLCV + earnings calendar into a PEAD event frame.

    Returns a frame with one row per announcement that has a complete 20-bar
    forward window in the cache. Empty if the cache or the earnings calendar
    is empty.
    """
    if ohlcv is None or ohlcv.empty or earnings is None or earnings.empty:
        return _empty_frame()

    bar_dates = pd.DatetimeIndex(ohlcv.index)
    raw_event_dates = pd.Series(pd.DatetimeIndex(earnings.index))
    event_dates = pd.DatetimeIndex(raw_event_dates.dt.normalize().unique()).sort_values()
    closes = ohlcv["close"].to_numpy(dtype=float)

    events: list[PeadEvent] = []
    for event in event_dates:
        # entry bar = first trading day strictly after the announcement.
        pos_after = bar_dates.searchsorted(event, side="right")
        if pos_after >= len(bar_dates) - FWD_20D:
            # Not enough forward bars to evaluate the 20-day label; skip.
            continue
        entry_idx = int(pos_after)
        entry_price = float(closes[entry_idx])
        if entry_price <= 0 or np.isnan(entry_price):
            continue
        close_5 = float(closes[entry_idx + FWD_5D])
        close_20 = float(closes[entry_idx + FWD_20D])
        fwd_5 = close_5 / entry_price - 1.0
        fwd_20 = close_20 / entry_price - 1.0
        events.append(
            PeadEvent(
                ticker=ticker,
                event_date=event,
                entry_date=bar_dates[entry_idx],
                entry_price=entry_price,
                fwd_return_5d=fwd_5,
                fwd_return_20d=fwd_20,
                label_drift_5d=int(fwd_5 > DRIFT_5D_THRESHOLD),
                label_drift_20d=int(fwd_20 > DRIFT_20D_THRESHOLD),
            )
        )
    if not events:
        return _empty_frame()

    out = pd.DataFrame(
        [
            {
                "ticker": e.ticker,
                "event_date": e.event_date,
                "entry_date": e.entry_date,
                "entry_price": e.entry_price,
                "fwd_return_5d": e.fwd_return_5d,
                "fwd_return_20d": e.fwd_return_20d,
                "label_drift_5d": e.label_drift_5d,
                "label_drift_20d": e.label_drift_20d,
            }
            for e in events
        ]
    )
    out["event_date"] = pd.to_datetime(out["event_date"])
    out["entry_date"] = pd.to_datetime(out["entry_date"])
    return out


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=pd.Index(
            [
                "ticker",
                "event_date",
                "entry_date",
                "entry_price",
                "fwd_return_5d",
                "fwd_return_20d",
                "label_drift_5d",
                "label_drift_20d",
            ]
        )
    )
