"""Earnings-derived features (Phase 5a).

For each bar date ``t`` we compute six features from the cached earnings
calendar of the ticker:

- ``days_to_next_earnings``    distance to the next scheduled earnings (clamped)
- ``days_since_last_earnings`` distance since the most recent past earnings
- ``last_surprise_pct``         signed surprise % of the most recent past earnings
- ``surprise_4q_mean``          mean signed surprise over the last 4 past earnings
- ``pre_earnings_window``       1 iff next earnings is within ``WINDOW`` days
- ``post_earnings_window``      1 iff last earnings was within ``WINDOW`` days

Causality convention. Earnings schedules are public well in advance, so the
*date* of a future announcement is usable at every prior bar. The *outcome*
(reported EPS, surprise %) is treated as unknown until the day strictly after
the announcement — i.e. for features at date ``t`` we only look at past
earnings with date ``< t`` when reading reported values. This matches the
"pas de leakage post-annonce du jour J avant la cloche fermeture" rule from
the Phase 5a brief: after-close announcements at date J are conservatively
treated as if first known on J+1.

Tickers without an earnings cache (or with an empty one, e.g. SPY/^VIX) get
the neutral-default row: clamp-day windows and zeros. This is what lets the
ETF benchmark survive the join+dropna step rather than being silently
dropped from the panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EARNINGS_FEATURE_COLUMNS: list[str] = [
    "days_to_next_earnings",
    "days_since_last_earnings",
    "last_surprise_pct",
    "surprise_4q_mean",
    "pre_earnings_window",
    "post_earnings_window",
]

DAYS_CLAMP = 90  # cap distances so far-from-earnings rows compare on a bounded scale
WINDOW_DAYS = 5  # pre/post window (inclusive) for the binary flags
SURPRISES_LOOKBACK = 4  # rolling-mean window for ``surprise_4q_mean``


def _neutral_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Defaults used when no earnings data is available for the ticker.

    Distances pinned to the clamp value (== "far from any announcement"),
    surprises and window flags zeroed. This is intentionally bland: a ticker
    without an earnings track (SPY, ^VIX) should look like "no earnings
    pressure right now" rather than NaN, otherwise its rows are dropped.
    """
    return pd.DataFrame(
        {
            "days_to_next_earnings": float(DAYS_CLAMP),
            "days_since_last_earnings": float(DAYS_CLAMP),
            "last_surprise_pct": 0.0,
            "surprise_4q_mean": 0.0,
            "pre_earnings_window": 0.0,
            "post_earnings_window": 0.0,
        },
        index=index,
        columns=pd.Index(EARNINGS_FEATURE_COLUMNS),
    )


def build_earnings_features(
    target_index: pd.DatetimeIndex,
    earnings: pd.DataFrame | None,
) -> pd.DataFrame:
    """Compute the 6 earnings features aligned to ``target_index``.

    ``earnings`` is the per-ticker frame from :class:`EarningsStore` (date
    index, columns ``eps_estimate / reported_eps / surprise_pct``). When it's
    ``None`` or empty the neutral-default row is returned instead.
    """
    if earnings is None or earnings.empty:
        return _neutral_frame(target_index)

    earnings = earnings.sort_index()
    earnings_dates = pd.DatetimeIndex(earnings.index).to_numpy("datetime64[ns]")
    surprises = earnings["surprise_pct"].to_numpy(dtype=float, copy=True)
    bar_dates = target_index.to_numpy("datetime64[ns]")

    n_bars = len(bar_dates)
    out = _neutral_frame(target_index).copy()

    # `np.searchsorted` gives us, per bar, the insertion point that splits the
    # earnings calendar into strict past (< t) and present-or-future (>= t).
    past_count = np.searchsorted(earnings_dates, bar_dates, side="left")
    next_idx = past_count  # first earnings date with date >= t

    # days_to_next_earnings — uses scheduled dates, valid even before announcement.
    days_to_next = np.full(n_bars, float(DAYS_CLAMP))
    has_future = next_idx < len(earnings_dates)
    if has_future.any():
        next_dates = earnings_dates[np.where(has_future, next_idx, 0)]
        # Treat same-day (date == t) as "0 days away" — still in the future for
        # the purposes of the next-announcement clock.
        delta_next = (next_dates - bar_dates).astype("timedelta64[D]").astype(float)
        days_to_next = np.where(
            has_future, np.minimum(delta_next, float(DAYS_CLAMP)), float(DAYS_CLAMP)
        )
    out["days_to_next_earnings"] = days_to_next

    # days_since_last_earnings — uses past earnings only (strict <).
    days_since_last = np.full(n_bars, float(DAYS_CLAMP))
    has_past = past_count > 0
    if has_past.any():
        last_idx = np.where(has_past, past_count - 1, 0)
        last_dates = earnings_dates[last_idx]
        delta_since = (bar_dates - last_dates).astype("timedelta64[D]").astype(float)
        days_since_last = np.where(
            has_past, np.minimum(delta_since, float(DAYS_CLAMP)), float(DAYS_CLAMP)
        )
    out["days_since_last_earnings"] = days_since_last

    # last_surprise_pct — only readable after the announcement (strict <). NaN
    # surprises (future estimate-only rows that somehow leaked) are treated as 0.
    last_surprise = np.zeros(n_bars)
    if has_past.any():
        last_idx = np.where(has_past, past_count - 1, 0)
        last_surprise = np.where(has_past, np.nan_to_num(surprises[last_idx], nan=0.0), 0.0)
    out["last_surprise_pct"] = last_surprise

    # surprise_4q_mean — rolling mean of the last 4 past surprises (also strict <).
    # Walk the bars in order; maintain a small ring of the most recent surprises.
    surprise_4q = np.zeros(n_bars)
    ring: list[float] = []
    pointer = 0  # number of earnings already consumed (date < current_bar)
    for i in range(n_bars):
        # Pull in any newly-past earnings (date < bar_dates[i]).
        while pointer < past_count[i]:
            surprise_val = surprises[pointer]
            if not np.isnan(surprise_val):
                ring.append(float(surprise_val))
                if len(ring) > SURPRISES_LOOKBACK:
                    ring.pop(0)
            pointer += 1
        surprise_4q[i] = float(np.mean(ring)) if ring else 0.0
    out["surprise_4q_mean"] = surprise_4q

    out["pre_earnings_window"] = (days_to_next <= WINDOW_DAYS).astype(float)
    # Pinning the post window to a strict 1..WINDOW_DAYS range keeps the same-day
    # observation out of "post" — it isn't yet at day t, only from t+1.
    out["post_earnings_window"] = (
        (days_since_last >= 1) & (days_since_last <= WINDOW_DAYS)
    ).astype(float)

    return out
