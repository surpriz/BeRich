"""Point-in-time fundamental features (Phase 11b).

From the cached quarterly statements we derive slow-moving quality / growth / leverage
ratios, aligned **point-in-time**: each quarter (period-end ``P``) only becomes visible at
``P + PUBLICATION_LAG_DAYS`` (a conservative ~75-day filing lag), then is forward-filled to
the daily bars. A bar at date ``t`` therefore only ever sees quarters whose filing date is
``<= t`` — no lookahead.

Tickers without a fundamentals cache (ETFs, indices, FX, crypto) or before their first
filed quarter get the neutral-default row (zeros), so they survive the join+dropna rather
than being dropped from the panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FUNDAMENTAL_FEATURE_COLUMNS: list[str] = [
    "f_net_margin",  # net income / revenue (quarterly)
    "f_roe",  # trailing-4Q net income / total equity
    "f_debt_to_equity",  # total debt / total equity
    "f_revenue_growth_yoy",  # revenue vs the same quarter a year ago
]

PUBLICATION_LAG_DAYS = 75  # conservative quarterly filing lag (period-end -> first known)
_CLIP = 10.0  # clip extreme ratios so a tiny denominator can't dominate


def _neutral_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        dict.fromkeys(FUNDAMENTAL_FEATURE_COLUMNS, 0.0),
        index=index,
        columns=pd.Index(FUNDAMENTAL_FEATURE_COLUMNS),
    )


def build_fundamental_features(
    target_index: pd.DatetimeIndex,
    fundamentals: pd.DataFrame | None,
    *,
    lag_days: int = PUBLICATION_LAG_DAYS,
) -> pd.DataFrame:
    """Compute point-in-time fundamental ratios aligned to ``target_index``."""
    if fundamentals is None or fundamentals.empty:
        return _neutral_frame(target_index)

    q = fundamentals.sort_index()
    revenue = q["revenue"]
    net_income = q["net_income"]
    equity = q["total_equity"].replace(0.0, np.nan)

    feats = pd.DataFrame(index=q.index)
    feats["f_net_margin"] = (net_income / revenue.replace(0.0, np.nan)).clip(-_CLIP, _CLIP)
    feats["f_roe"] = (net_income.rolling(4, min_periods=4).sum() / equity).clip(-_CLIP, _CLIP)
    feats["f_debt_to_equity"] = (q["total_debt"] / equity).clip(-_CLIP, _CLIP)
    feats["f_revenue_growth_yoy"] = (revenue / revenue.shift(4) - 1.0).clip(-_CLIP, _CLIP)

    # Each quarter becomes known only after the filing lag; index by that availability date.
    feats.index = q.index + pd.Timedelta(days=lag_days)
    feats = feats[~feats.index.duplicated(keep="last")].sort_index()

    # Forward-fill the last known quarter onto each bar; bars before the first filing -> 0.
    aligned = feats.reindex(target_index, method="ffill")
    return aligned[FUNDAMENTAL_FEATURE_COLUMNS].fillna(0.0)


__all__ = ["FUNDAMENTAL_FEATURE_COLUMNS", "PUBLICATION_LAG_DAYS", "build_fundamental_features"]
